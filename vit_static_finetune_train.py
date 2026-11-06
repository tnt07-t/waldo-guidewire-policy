"""Bounded last-block adaptation; real data and outputs stay on Perlmutter.

No feature cache, test partition, resume, automatic retry or allocation creation.
Checkpoints contain the adapted final block AND classifier, bound to a base hash.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from preprocessing import CHECKPOINT, DINOv2Preprocessor
from vit_static_finetune import ViTStaticFinetune
from vit_gru_debug import supervise, training_inputs
from vit_gru_train import check_partitions, metrics, save_bytes, save_json, validation_inputs

CONFIG = dict(seed=0, batch_size=32, microbatch_size=1, block_lr=1e-5,
              head_lr=1e-3, optimizer='Adam', weight_decay=0., max_epochs=100,
              max_updates=900, patience=20, training_seconds=1200., overall_seconds=1200.)


def resolve_config(**overrides):
    config = dict(CONFIG, **overrides)
    for key in ('seed', 'batch_size', 'microbatch_size', 'max_epochs', 'max_updates', 'patience'):
        if type(config[key]) is not int:
            raise ValueError('Expected integer ' + key)
    if not 0 <= config['seed'] < 2**32:
        raise ValueError('Invalid seed')
    for key in ('batch_size', 'max_epochs', 'max_updates', 'patience'):
        if not 1 <= config[key] <= CONFIG[key]:
            raise ValueError('Invalid or excessive ' + key)
    if not 1 <= config['microbatch_size'] <= config['batch_size']:
        raise ValueError('Invalid microbatch size')
    for key in ('training_seconds', 'overall_seconds'):
        if not 0 < config[key] <= CONFIG[key]:
            raise ValueError('Invalid or excessive ' + key)
    # Recipe changes require an explicit comparison, not silent CLI overrides.
    for key in ('block_lr', 'head_lr', 'optimizer', 'weight_decay'):
        if config[key] != CONFIG[key]:
            raise ValueError('Unsupported recipe change: ' + key)
    return config


def state_hash(model, *, frozen_only=False):
    """Content identity independent of torch archive/storage identifiers."""
    trainable = model.trainable_parameter_counts() if frozen_only else {}
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name in trainable:
            continue
        value = value.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_optimizer(model, config):
    block = list(model.encoder.backbone.blocks[-1].parameters())
    head = list(model.head.parameters())
    selected = block + head
    expected = [p for p in model.parameters() if p.requires_grad]
    if len({id(p) for p in selected}) != len(selected) or {id(p) for p in selected} != {id(p) for p in expected}:
        raise ValueError('Optimizer scope differs from final block plus classifier')
    return torch.optim.Adam([dict(params=block, lr=config['block_lr'], role='final_block'),
                             dict(params=head, lr=config['head_lr'], role='classifier')],
                            weight_decay=config['weight_decay'])


def image_batch(dataset, indices, prepare, device):
    goals, latest = [], []
    for index in indices:
        case = dataset[int(index)]
        goals.append(prepare.prepare_image(case['goal'])['tensor'])
        latest.append(prepare.prepare_image(case['history'][-1])['tensor'])
    return torch.stack(goals).to(device), torch.stack(latest).to(device)


def evaluate(model, dataset, metadata, labels, prepare, microbatch, check):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    scores = []
    try:
        with torch.no_grad():
            for indices in torch.arange(len(dataset)).split(microbatch):
                check()
                goal, latest = image_batch(dataset, indices, prepare, device)
                scores.append(model(goal, latest, metadata[indices].to(device)).cpu())
                check()
    finally:
        model.train(was_training)
    logits = torch.cat(scores)
    return metrics(logits, labels.cpu()), logits


def optimizer_step(model, optimizer, indices, dataset, metadata, labels, prepare, microbatch, check):
    """Case-weighted accumulation, including nondivisible final effective batch."""
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for part in indices.split(microbatch):
        check()
        goal, latest = image_batch(dataset, part, prepare, device)
        loss = F.cross_entropy(model(goal, latest, metadata[part].to(device)), labels[part].to(device), reduction='sum') / len(indices)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        check()
    for parameter in model.parameters():
        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
            raise RuntimeError('Missing/nonfinite trainable gradient')
        if not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError('Unexpected frozen gradient')
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
        raise RuntimeError('Nonfinite updated weights')


def save_checkpoint(directory, model, context, epoch, updates, validation):
    buffer = io.BytesIO()
    torch.save(dict(format_version='static-last-block-v1', context=context,
                    block={k: v.detach().cpu() for k, v in model.encoder.backbone.blocks[-1].state_dict().items()},
                    head={k: v.detach().cpu() for k, v in model.head.state_dict().items()},
                    epoch=epoch, updates=updates, validation=validation), buffer)
    # Only best.pt is retained; two full adapted checkpoints exceed the old cap.
    save_bytes(directory, 'best.pt', buffer.getvalue())


def load_checkpoint(path, *, device='cpu', factory=ViTStaticFinetune):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 'static-last-block-v1':
        raise ValueError('Unsupported adaptation checkpoint')
    context = checkpoint['context']
    if context['checkpoint'] != CHECKPOINT:
        raise ValueError('Wrong pretrained checkpoint')
    resolve_config(**context['config'])
    scales = torch.tensor(context['metadata_scale_depth_time'])
    if scales.shape != (2,) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Invalid metadata scales')
    model = factory(device='cpu')
    if state_hash(model.encoder) != context['base_encoder_sha256']:
        raise ValueError('Pretrained encoder content mismatch')
    if model.trainable_parameter_counts() != context['trainable_parameters']:
        raise ValueError('Trainable scope mismatch')
    model.encoder.backbone.blocks[-1].load_state_dict(checkpoint['block'], strict=True)
    model.head.load_state_dict(checkpoint['head'], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Nonfinite checkpoint state')
    if state_hash(model, frozen_only=True) != context['frozen_sha256']:
        raise ValueError('Frozen state mismatch')
    return model.to(device).eval(), checkpoint


def train(model, train_data, validation_data, prepare, directory, context, *, clock=time.monotonic, factory=ViTStaticFinetune):
    config = resolve_config(**context['config'])
    optimizer = make_optimizer(model, config)
    generator = torch.Generator().manual_seed(config['seed'])
    start = clock()

    def check():
        if clock() - start >= config['training_seconds']:
            raise TimeoutError('training_time_limit')

    history, best_logits = [], None
    best, stale, updates, stop = -1., 0, 0, 'max_epochs'
    try:
        for epoch in range(1, config['max_epochs'] + 1):
            for indices in torch.randperm(len(train_data[0]), generator=generator).split(config['batch_size']):
                if updates >= config['max_updates']:
                    stop = 'max_updates'
                    break
                optimizer_step(model, optimizer, indices, *train_data, prepare, config['microbatch_size'], check)
                updates += 1
                check()
            else:
                train_scores, _ = evaluate(model, *train_data, prepare, config['microbatch_size'], check)
                val_scores, logits = evaluate(model, *validation_data, prepare, config['microbatch_size'], check)
                if state_hash(model, frozen_only=True) != context['frozen_sha256']:
                    raise RuntimeError('Frozen parameters/buffers changed')
                check()
                if val_scores['balanced_accuracy'] > best:
                    save_checkpoint(directory, model, context, epoch, updates, val_scores)
                    best, stale, best_logits = val_scores['balanced_accuracy'], 0, logits
                else:
                    stale += 1
                history.append(dict(epoch=epoch, updates=updates, train=train_scores, validation=val_scores))
                save_json(directory, 'history.json', history)
                check()
                if stale >= config['patience']:
                    stop = 'early_stopping'
                    break
                if updates >= config['max_updates']:
                    stop = 'max_updates'
                    break
                continue
            break
    except TimeoutError:
        optimizer.zero_grad(set_to_none=True)
        stop = 'training_time_limit'
    if state_hash(model, frozen_only=True) != context['frozen_sha256']:
        raise RuntimeError('Frozen parameters/buffers changed')
    result = dict(status='budget_limited' if stop == 'training_time_limit' else 'completed' if history else 'incomplete',
                  stop_reason=stop, completed_epochs=len(history), updates=updates,
                  training_seconds=clock()-start, reload_verified=False, frozen_unchanged=True)
    if best_logits is not None:
        device = next(model.parameters()).device
        # Free live model/optimizer GPU storage before reconstructing the selected model.
        model.cpu()
        del optimizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        restored, checkpoint = load_checkpoint(Path(directory)/'best.pt', device=device, factory=factory)
        # The independent parent deadline still bounds this reload verification.
        scores, logits = evaluate(restored, *validation_data, prepare, config['microbatch_size'], lambda: None)
        torch.testing.assert_close(logits, best_logits, rtol=1e-6, atol=1e-7)
        result.update(best_epoch=checkpoint['epoch'], best_validation=scores, reload_verified=True)
        save_bytes(directory, 'selected_validation.pt', _score_bytes(logits, validation_data[2]))
    return result


def _score_bytes(logits, labels):
    buffer = io.BytesIO()
    torch.save(dict(logits=logits, labels=labels.cpu()), buffer)
    return buffer.getvalue()


def run_worker(args):
    import importlib.metadata
    from dataset import GuidewireDataset
    config = resolve_config(microbatch_size=args.microbatch_size)
    torch.set_num_threads(2)
    torch.manual_seed(config['seed'])
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model = ViTStaticFinetune(device=device)
    # Match the historical classifier seed independently of backbone construction.
    from vit_static import ViTStaticHead
    torch.manual_seed(config['seed'])
    model.head = ViTStaticHead().to(device)
    context = dict(config=config, checkpoint=CHECKPOINT,
                   base_encoder_sha256=state_hash(model.encoder), frozen_sha256=state_hash(model, frozen_only=True),
                   trainable_parameters=model.trainable_parameter_counts(),
                   git_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip(),
                   git_dirty=bool(subprocess.run(['git', 'status', '--porcelain'], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip()),
                   code_sha256={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in
                                ('vit_static_finetune_train.py', 'vit_static_finetune.py', 'vit_static.py', 'vit_gru_train.py', 'vit_gru_debug.py', 'dataset.py', 'preprocessing.py')},
                   versions={name: importlib.metadata.version(name) for name in ('torch', 'timm', 'Pillow')},
                   preparation='Existing normalized full-view RGB 518; no augmentation',
                   metadata_scaling='Training population std; no centering',
                   feature_roles=['latest_summary384', 'goal_summary384', 'six_depth_time_pairs12'])
    if args.readiness_only:
        context['metadata_scale_depth_time'] = [1., 1.]
        save_json(args.output_dir, 'run_context.json', context)
        goal = torch.randn(1, 3, 518, 518, device=device)
        latest = torch.randn_like(goal)
        metadata = torch.zeros(1, 6, 2, device=device)
        optimizer = make_optimizer(model, config)
        before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        loss = F.cross_entropy(model(goal, latest, metadata), torch.tensor([0], device=device))
        loss.backward()
        for parameter in model.parameters():
            if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
                raise RuntimeError('Missing/nonfinite readiness gradient')
        optimizer.step()
        if state_hash(model, frozen_only=True) != context['frozen_sha256']:
            raise RuntimeError('Frozen readiness state changed')
        if not any(not torch.equal(p, before[name]) for name, p in model.named_parameters() if name.startswith('encoder.') and p.requires_grad):
            raise RuntimeError('Final block did not update')
        if not any(not torch.equal(p, before[name]) for name, p in model.named_parameters() if name.startswith('head.') and p.requires_grad):
            raise RuntimeError('Classifier did not update')
        with torch.no_grad():
            expected = model(goal, latest, metadata).cpu()
        save_checkpoint(args.output_dir, model, context, 0, 1, {})
        model.cpu()
        del optimizer, before, loss
        restored, _ = load_checkpoint(args.output_dir/'best.pt', device=device)
        with torch.no_grad():
            actual = restored(goal, latest, metadata).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        save_json(args.output_dir, 'summary.json', dict(status='readiness_passed', real_data_used=False,
                  reload_verified=True, frozen_unchanged=True,
                  trainable_parameters=sum(context['trainable_parameters'].values()),
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None))
        return 0
    train_set = GuidewireDataset(args.data_dir, split_position=0, partition='train')
    val_set = GuidewireDataset(args.data_dir, split_position=0, partition='validation')
    check_partitions(train_set, val_set)
    if (len(train_set), len(val_set)) != (267, 80):
        raise ValueError('Expected real split-0 counts 267/80')
    metadata, labels, scales = training_inputs(train_set)
    val_metadata, val_labels = validation_inputs(val_set, scales)
    context.update(metadata_scale_depth_time=scales.tolist(), train_rows=list(train_set.indices), validation_rows=list(val_set.indices),
                   validation_episodes=[val_set.records[i]['episode'] for i in val_set.indices])
    # Validate both label classes before performing any optimizer update.
    for target in (labels, val_labels):
        if set(target.tolist()) != {0, 1}:
            raise ValueError('Both action classes required')
    save_json(args.output_dir, 'run_context.json', context)
    result = train(model, (train_set, metadata, labels), (val_set, val_metadata, val_labels),
                   DINOv2Preprocessor(), args.output_dir, context)
    result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    save_json(args.output_dir, 'summary.json', result)
    return 0 if result['reload_verified'] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=os.environ.get('DATA_DIR'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--readiness-only', action='store_true', help='Actual pretrained synthetic forward/backward/reload; no real cases')
    parser.add_argument('--microbatch-size', type=int, default=1)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        config = resolve_config(microbatch_size=args.microbatch_size)
    except ValueError as error:
        parser.error(str(error))
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('PSCRATCH') or (args.data_dir is None and not args.readiness_only):
        parser.error('Use an allocated Perlmutter compute node with PSCRATCH and DATA_DIR')
    scratch = Path(os.environ['PSCRATCH']).resolve()
    args.data_dir = args.data_dir.resolve() if args.data_dir else None
    args.output_dir = args.output_dir.resolve()
    paths = [args.output_dir] + ([args.data_dir] if args.data_dir else [])
    if any(scratch not in p.parents for p in paths):
        parser.error('Data/output must be beneath PSCRATCH')
    if args.data_dir and (args.output_dir == args.data_dir or args.data_dir in args.output_dir.parents):
        parser.error('Output must be outside dataset')
    if not args.readiness_only and not args.data_dir.is_dir():
        parser.error('Missing dataset')
    if args.worker:
        if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
            parser.error('Worker requires an empty output directory')
        return run_worker(args)
    if args.output_dir.exists():
        parser.error('Choose a new output directory')
    args.output_dir.mkdir(parents=True, mode=0o700)
    start = time.monotonic()
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
               '--output-dir', str(args.output_dir), '--device', args.device,
               '--microbatch-size', str(args.microbatch_size)]
    if args.data_dir:
        command += ['--data-dir', str(args.data_dir)]
    if args.readiness_only:
        command.append('--readiness-only')
    code, timeout = supervise(command, seconds=120 if args.readiness_only else config['overall_seconds'])
    path = args.output_dir/'summary.json'
    result = json.loads(path.read_text()) if path.exists() else {}
    if timeout or code:
        result['status'] = 'overall_time_limit' if timeout else 'worker_failed'
    result.update(process_seconds=time.monotonic()-start, process_exit_code=code, overall_limit_reached=timeout)
    save_json(args.output_dir, 'summary.json', result)
    print('Status: {}. Private artifacts remain in the output directory.'.format(result['status']))
    return code


if __name__ == '__main__':
    sys.exit(main())
