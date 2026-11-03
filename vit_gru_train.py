"""Bounded split-0 development training. Real artifacts stay on Perlmutter.

Fresh initialization only. Checkpoints support evaluation, not training resume.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from vit_gru_debug import freeze_and_snapshot, verify_frozen, supervise, training_inputs

# Settings for the first agreed experiment, not a hyperparameter sweep.
CONFIG = dict(seed=0, hidden_size=64, batch_size=32, lr=0.001,
              optimizer="Adam", weight_decay=0.0, max_epochs=100,
              max_updates=900, patience=20, head_seconds=120,
              extraction_seconds=300, overall_seconds=600,
              split_position=0, image_chunk_size=1)
OUTPUT_BYTES = 20 * 1024 * 1024


def save_bytes(directory, name, data, limit=OUTPUT_BYTES):
    """Count old files plus temporary replacement against the storage cap."""
    directory = Path(directory)
    used = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    if used + len(data) > limit:
        raise RuntimeError('Output storage limit reached')
    temporary = directory / (name + '.tmp')
    with temporary.open('xb') as handle:
        handle.write(data)
    temporary.replace(directory / name)


def save_json(directory, name, value):
    save_bytes(directory, name, (json.dumps(value, indent=2, allow_nan=False)+'\n').encode())


def validation_inputs(dataset, scales):
    import torch
    rows, labels = [], []
    for index in dataset.indices:
        record = dataset.records[index]
        depth, offsets = record['depth_relative_mm'], record['history_offsets_s']
        if len(depth) != 6 or len(offsets) != 6:
            raise ValueError('Expected six metadata steps')
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in depth + offsets):
            raise ValueError('Nonfinite metadata')
        if depth[-1] != 0 or offsets[-1] != 0 or any(a > b for a, b in zip(offsets, offsets[1:])):
            raise ValueError('Invalid relative metadata order/endpoints')
        if type(record['label']) is not int or record['label'] not in (0, 1):
            raise ValueError('Invalid label')
        rows.append(list(zip(depth, offsets)))
        labels.append(record['label'])
    # Reuse the training scales: validation must never choose its own scales.
    metadata = (torch.tensor(rows, dtype=torch.float64) / scales).float()
    if not torch.isfinite(metadata).all():
        raise ValueError('Nonfinite scaled metadata')
    return metadata, torch.tensor(labels, dtype=torch.long)


def check_partitions(train, validation):
    if train.partition != 'train' or validation.partition != 'validation':
        raise ValueError('Only training and validation allowed')
    if train.split_position != 0 or validation.split_position != 0:
        raise ValueError('Only split position 0 allowed')
    if not 0 < len(train) <= 267 or not 0 < len(validation) <= 80:
        raise ValueError('Partition case budget exceeded or empty')
    for dataset in (train, validation):
        if len(set(dataset.indices)) != len(dataset):
            raise ValueError('Duplicate partition rows')
    if set(train.indices) & set(validation.indices):
        raise ValueError('Overlapping partition rows')
    episodes = [{d.records[i]['episode'] for i in d.indices} for d in (train, validation)]
    if episodes[0] & episodes[1]:
        raise ValueError('Overlapping partition episodes')


def extract(datasets, prepare, encoder, *, seconds=300, clock=time.monotonic):
    import torch
    start = clock()
    result = []
    if sum(map(len, datasets)) > 347:
        raise ValueError('Extraction case budget exceeded')
    for dataset in datasets:
        histories, goals = [], []
        for index in range(len(dataset)):
            if clock() - start >= seconds:
                raise TimeoutError('extraction_time_limit')
            prepared = prepare.prepare_case(dataset[index])
            feature = encoder.encode(prepared['goal'], prepared['history'], image_chunk_size=1)
            # Encoder inference tensors cannot be saved for backward by the head.
            # Clone outside inference mode; retain only small feature vectors on CPU.
            histories.append(feature['history_features'].detach().cpu().clone())
            goals.append(feature['goal_features'].detach().cpu().clone())
            if clock() - start >= seconds:
                raise TimeoutError('extraction_time_limit')
        result.append((torch.stack(histories), torch.stack(goals)))
    return result


def metrics(logits, labels):
    import torch
    from torch.nn import functional as F
    if logits.shape != (len(labels), 2) or not torch.isfinite(logits).all():
        raise ValueError('Invalid/nonfinite scores')
    if labels.dtype != torch.long or set(labels.tolist()) != {0, 1}:
        raise ValueError('Both integer label classes required')
    # Match static evaluation: exact score ties predict ADVANCE.
    predicted = (logits[:, 1] >= logits[:, 0]).long()
    counts = torch.bincount(labels * 2 + predicted, minlength=4).reshape(2, 2)
    recalls = counts.diag().double() / counts.sum(1)
    return dict(loss=F.cross_entropy(logits, labels).item(),
                accuracy=(predicted == labels).float().mean().item(),
                balanced_accuracy=recalls.mean().item(),
                retract_recall=recalls[0].item(), advance_recall=recalls[1].item(),
                confusion_matrix=counts.tolist(), n=len(labels))


def evaluate(head, data):
    import torch
    # Evaluation measures performance without computing gradients or changing weights.
    was_training = head.training
    head.eval()
    try:
        with torch.no_grad():
            logits = head(*data[:3])
            return metrics(logits, data[3]), logits.detach().clone()
    finally:
        head.train(was_training)


def save_checkpoint(directory, name, head, context, epoch, updates, scores):
    import torch
    buffer = io.BytesIO()
    torch.save(dict(format_version=1, state_dict={k: v.detach().cpu() for k, v in head.state_dict().items()},
                    context=context, epoch=epoch, updates=updates, validation=scores), buffer)
    save_bytes(directory, name, buffer.getvalue())


def load_checkpoint(path, device='cpu'):
    import torch
    from vit_gru import ViTGRUHead
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['format_version'] != 1 or checkpoint['context']['config']['hidden_size'] != 64:
        raise ValueError('Unsupported checkpoint configuration')
    scales = torch.tensor(checkpoint['context']['metadata_scale_depth_time'])
    if scales.shape != (2,) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Invalid checkpoint scales')
    head = ViTGRUHead().to(device)
    head.load_state_dict(checkpoint['state_dict'], strict=True)
    if any(not torch.isfinite(p).all() for p in head.parameters()):
        raise ValueError('Nonfinite checkpoint weights')
    head.eval()
    return head, checkpoint


def train_fresh(train, validation, directory, context, *, max_epochs=100,
                max_updates=900, seconds=120, patience=20, clock=time.monotonic):
    import torch
    from vit_gru import ViTGRUHead
    from torch.nn import functional as F
    if not 1 <= max_epochs <= 100 or not 1 <= max_updates <= 900 or not 1 <= patience <= 20:
        raise ValueError('Training work limits exceeded')
    if not math.isfinite(seconds) or not 0 < seconds <= 120:
        raise ValueError('Invalid training deadline')
    # Construct NEW learned weights and optimizer state on every run.
    # The seed makes local repeated runs reproducible; no tiny-fit weights are loaded.
    torch.manual_seed(0)
    head = ViTGRUHead().to(train[0].device)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.001, weight_decay=0)
    generator = torch.Generator().manual_seed(0)
    start = clock()
    initial, _ = evaluate(head, train)
    evaluate(head, validation)  # Validate labels and tensor contracts before updates.
    best, stale, updates = -1.0, 0, 0
    history, best_logits = [], None
    stop = 'max_epochs'
    # One epoch visits every training case once, in a newly shuffled order.
    # Each data tuple is (history features, goal features, scaled metadata, labels).
    for epoch in range(1, max_epochs + 1):
        complete = True
        head.train()
        # Keep the smaller last batch: 267 cases give eight batches of 32 and one of 11.
        for batch in torch.randperm(len(train[3]), generator=generator).split(32):
            if clock() - start >= seconds or updates >= max_updates:
                stop = 'head_time_limit' if clock() - start >= seconds else 'max_updates'
                complete = False
                break
            batch = batch.to(train[0].device)
            # Clear the previous batch's gradients, then score this batch.
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(*(x[batch] for x in train[:3])), train[3][batch])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            # Backward computes how each learned weight contributed to the error.
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise RuntimeError('Invalid training gradients')
            # Adam uses those gradients to update only the GRU/classifier weights.
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in head.parameters()):
                raise RuntimeError('Nonfinite head weights')
            updates += 1
        if not complete or clock() - start >= seconds:
            if complete:
                stop = 'head_time_limit'
            break
        # Measure both partitions at the SAME weights after a complete epoch.
        # Validation is used for selection only; it never enters optimizer.step().
        train_scores, _ = evaluate(head, train)
        val_scores, logits = evaluate(head, validation)
        entry = dict(epoch=epoch, updates=updates, train=train_scores, validation=val_scores)
        history.append(entry)
        save_checkpoint(directory, 'latest.pt', head, context, epoch, updates, val_scores)
        # Strict improvement keeps the earlier checkpoint on equal scores.
        # stale counts complete epochs since the last improvement.
        if val_scores['balanced_accuracy'] > best:
            best, stale, best_logits = val_scores['balanced_accuracy'], 0, logits.cpu()
            save_checkpoint(directory, 'best.pt', head, context, epoch, updates, val_scores)
        else:
            stale += 1
        save_json(directory, 'history.json', history)
        if stale >= patience:
            stop = 'early_stopping'
            break
        if updates >= max_updates:
            stop = 'max_updates'
            break
    result = dict(status='completed' if history else 'incomplete', stop_reason=stop,
                  updates=updates, completed_epochs=len(history), initial_train=initial,
                  head_seconds=clock()-start, reload_verified=False)
    if history:
        # Reconstruct a separate model from disk and check its predictions.
        # This tests evaluation reload, not interrupted-training resumption.
        restored, checkpoint = load_checkpoint(Path(directory)/'best.pt', train[0].device)
        scores, logits = evaluate(restored, validation)
        if not torch.allclose(logits.cpu(), best_logits, rtol=1e-6, atol=1e-7):
            raise RuntimeError('Checkpoint predictions changed after reload')
        result.update(best_epoch=checkpoint['epoch'], best_validation=scores, reload_verified=True)
    return result


def run_worker(args):
    import torch
    import importlib.metadata
    from dataset import GuidewireDataset
    from preprocessing import CHECKPOINT, DINOv2Preprocessor
    from vit import DINOv2Encoder
    torch.set_num_threads(2)
    # Only train/validation are instantiated; test images remain reserved.
    train = GuidewireDataset(args.data_dir, split_position=0, partition='train')
    validation = GuidewireDataset(args.data_dir, split_position=0, partition='validation')
    check_partitions(train, validation)
    if (len(train), len(validation)) != (267, 80):
        raise ValueError('Real split-0 counts must be 267/80')
    # Learn unit-conversion scales from training metadata only.
    metadata, labels, scales = training_inputs(train)
    val_metadata, val_labels = validation_inputs(validation, scales)
    if set(labels.tolist()) != {0, 1} or set(val_labels.tolist()) != {0, 1}:
        raise ValueError('Both classes required in each development partition')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    # Frozen pretrained ViT does the expensive image work once.
    # Later epochs reuse its outputs rather than encoding the images again.
    encoder = DINOv2Encoder(device=device)
    snapshot = freeze_and_snapshot(encoder.backbone)
    start = time.monotonic()
    features = extract((train, validation), DINOv2Preprocessor(), encoder)
    extraction_seconds = time.monotonic()-start
    verify_frozen(encoder.backbone, snapshot)
    context = dict(config=CONFIG, checkpoint=CHECKPOINT,
                   preparation='Existing full-view 518-square DINOv2Preprocessor; no augmentation',
                   metadata_scale_depth_time=scales.tolist(), metadata_scaling='training population std; no centering',
                   train_rows=list(train.indices), validation_rows=list(validation.indices),
                   code_sha256={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                for name in ('vit_gru_train.py', 'vit_gru.py', 'vit_gru_debug.py', 'vit.py', 'preprocessing.py', 'dataset.py')},
                   git_commit=subprocess.run(['git','rev-parse','HEAD'], cwd=Path(__file__).parent,
                                             capture_output=True, text=True).stdout.strip(),
                   git_dirty=bool(subprocess.run(['git','status','--porcelain'], cwd=Path(__file__).parent,
                                                 capture_output=True, text=True).stdout.strip()),
                   versions={name: importlib.metadata.version(name) for name in ('torch','timm','Pillow')})
    save_json(args.output_dir, 'run_context.json', context)
    data = [tuple(x.to(device) for x in (*feature, meta, target))
            for feature, meta, target in zip(features, (metadata,val_metadata), (labels,val_labels))]
    result = train_fresh(*data, args.output_dir, context)
    verify_frozen(encoder.backbone, snapshot)
    result.update(encoder_unchanged=True, extraction_seconds=extraction_seconds,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
    save_json(args.output_dir, 'summary.json', result)
    return 0 if result['status']=='completed' else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=os.environ.get('DATA_DIR'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('PSCRATCH') or args.data_dir is None:
        parser.error('Use an allocated Perlmutter compute node with PSCRATCH and DATA_DIR set')
    scratch = Path(os.environ['PSCRATCH']).resolve()
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    if any(scratch not in p.parents for p in (args.data_dir,args.output_dir)):
        parser.error('Data/output must be beneath PSCRATCH')
    if args.output_dir == args.data_dir or args.data_dir in args.output_dir.parents:
        parser.error('Output must be outside dataset')
    if not args.data_dir.is_dir():
        parser.error('Missing dataset directory')
    if args.worker:
        return run_worker(args)
    if args.output_dir.exists():
        parser.error('Choose a new output directory')
    args.output_dir.mkdir(parents=True, mode=0o700)
    start = time.monotonic()
    # Parent process can stop the worker even if a native call/download hangs.
    # This terminates the worker, NOT the Slurm allocation; release that separately.
    code, timeout = supervise([sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                              '--data-dir', str(args.data_dir), '--output-dir', str(args.output_dir),
                              '--device', args.device], seconds=600)
    path = args.output_dir/'summary.json'
    result = json.loads(path.read_text()) if path.exists() else {}
    if timeout or code:
        result['status'] = 'overall_time_limit' if timeout else 'worker_failed'
    result.update(process_seconds=time.monotonic()-start, process_exit_code=code,
                  overall_limit_reached=timeout)
    save_json(args.output_dir, 'summary.json', result)
    print('Status: {}. Private results remain in output directory.'.format(result['status']))
    return code


if __name__ == '__main__':
    sys.exit(main())
