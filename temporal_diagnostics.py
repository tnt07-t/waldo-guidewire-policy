"""Checkpoint diagnostics on Perlmutter. Aggregate outputs, no feature cache.

Attention routing, intervention sensitivity and metadata recoverability are
distinct diagnostics. None establishes semantic localization or causal reasoning.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from temporal_evaluation import tensors, assess, probabilities, metadata_scale, write_json
from temporal_models import FeatureHead


def stages(head, data):
    """Observe normal forward outputs; separately recompute per-head routing."""
    if head.kind != "attention":
        raise ValueError("Attention checkpoint required")
    head.eval()
    captured = {}
    def capture_summary(module, inputs, output):
        captured['summary'] = output[:, -1].detach()
    hook = head.temporal.register_forward_hook(capture_summary)
    try:
        with torch.no_grad():
            logits = head(*data[:3])
            raw = torch.cat((data[0], data[2]), -1)
            projected = head.project(raw)
            sequence = projected if head.position is None else projected + head.position
            layer = head.temporal.layers[0]
            query = layer.norm1(sequence) if layer.norm_first else sequence
            _, weights = layer.self_attn(query, query, query, need_weights=True,
                                          average_attn_weights=False)
    finally:
        hook.remove()
    return {"raw_steps": raw.detach(), "projected_steps": projected.detach(),
            "summary": captured['summary'], "attention": weights.detach(),
            "logits": logits.detach()}


def representation_stats(value):
    """Numerical spread/rank, not semantic information content."""
    flat = value.detach().double().cpu().reshape(len(value), -1)
    centered = flat - flat.mean(0)
    eigen = torch.linalg.eigvalsh(centered @ centered.T).clamp_min(0)
    if eigen.sum() > 0:
        p = eigen / eigen.sum()
        rank = float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum()))
    else:
        rank = 0.
    result = {"shape": list(value.shape), "rms": float(flat.square().mean().sqrt()),
              "between_case_std_mean": float(flat.std(0, unbiased=False).mean()),
              "effective_rank": rank}
    if value.ndim == 3:
        x = value.double()
        result['adjacent_step_rms_difference'] = ((x[:, 1:] - x[:, :-1]).square().mean((0, 2)).sqrt().cpu().tolist())
    return result


def routing_summary(weights):
    """Aggregate [case, head, query step, source step] attention probabilities."""
    w = weights.double().cpu()
    entropy = -(w * w.clamp_min(1e-30).log()).sum(-1) / np.log(w.shape[-1])
    return {"mean_matrix_per_head": w.mean(0).tolist(),
            "std_matrix_per_head": w.std(0, unbiased=False).tolist(),
            "latest_query_mean_per_head": w[:, :, -1].mean(0).tolist(),
            "latest_query_normalized_entropy_per_head": entropy[:, :, -1].mean(0).tolist(),
            "note": "Rows=query steps; columns=source steps, f1 oldest to f6 latest. Entropy 1=uniform, 0=concentrated. These are routing weights, not causal importance or pixel maps."}


def interventions(head, data, training):
    """Fixed-checkpoint perturbations using training means; no retraining."""
    base_p = probabilities(head, data)
    base = assess(head, data)
    h, g, m, y = data
    controls = []
    mean_h, mean_g, mean_m = [x.mean(0, keepdim=True) for x in training[:3]]
    for step in range(6):
        edited = h.clone(); edited[:, step] = mean_h[:, step]
        controls.append(("replace_visual_f" + str(step+1), edited, g, m))
    controls.extend([
        ("repeat_latest_visual", h[:, -1:].expand_as(h), g, m),
        ("reverse_visual_only", h.flip(1), g, m),
        ("replace_goal", h, mean_g.expand_as(g), m),
        ("replace_all_metadata", h, g, mean_m.expand_as(m))])
    for channel, name in enumerate(('depth', 'time')):
        edited = m.clone(); edited[:, :, channel] = mean_m[:, :, channel]
        controls.append(('replace_' + name, h, g, edited))
    result = {}
    for name, ch, cg, cm in controls:
        changed = (ch, cg, cm, y)
        p = probabilities(head, changed)
        score = assess(head, changed)
        result[name] = {"metrics": score,
                        "balanced_accuracy_delta": score['balanced_accuracy'] - base['balanced_accuracy'],
                        "mean_absolute_probability_change": float(np.abs(p-base_p).mean()),
                        "prediction_flip_fraction": float(((p >= .5) != (base_p >= .5)).mean())}
    return {"baseline": base, "controls": result,
            "note": "Training-mean replacements may be out of distribution. Reversal changes image/metadata alignment. No-refit sensitivity does not prove usefulness, necessity or lost information."}


def metadata_recoverability(representations, metadata, episodes):
    """Fixed ridge probes on an episode-disjoint split WITHIN training only."""
    episodes = np.asarray(episodes)
    unique = np.unique(episodes)
    if len(unique) < 4:
        return {"status": "unavailable", "reason": "At least four training episodes required"}
    rng = np.random.default_rng(0)
    held = rng.permutation(unique)[:max(1, len(unique)//4)]
    test = np.isin(episodes, held); fit_rows = ~test
    target = metadata.detach().cpu().numpy().reshape(len(metadata), 12)
    results = {}
    for name, values in representations.items():
        x = values.detach().cpu().numpy().reshape(len(values), -1)
        probe = make_pipeline(StandardScaler(), Ridge(alpha=1., solver='lsqr'))
        probe.fit(x[fit_rows], target[fit_rows])
        pred = probe.predict(x[test])
        mse = np.mean((pred-target[test])**2, axis=0)
        variance = np.var(target[test], axis=0)
        by_channel = {}
        for channel, label in enumerate(('depth','time')):
            cols = np.arange(channel, 12, 2)
            variable = cols[variance[cols] > 1e-12]
            by_channel[label] = {"mse": float(mse[cols].mean()),
                                 "mean_r2_nonconstant_steps": float((1-mse[variable]/variance[variable]).mean()) if len(variable) else None,
                                 "variable_steps": len(variable)}
        results[name] = by_channel
    return {"status": "completed", "fit_cases": int(fit_rows.sum()), "heldout_training_cases": int(test.sum()),
            "results": results,
            "note": "Fixed alpha=1; scaler/probe fit only on inner training episodes. Targets are scaled depth/time. Raw inputs explicitly contain these targets. Changes reflect linear recoverability under this probe, not semantic information loss, task relevance, or a classifier comparison."}


def diagnose(head, data, training, training_episodes):
    observed = stages(head, data)
    train_stages = stages(head, training)
    with torch.no_grad():
        weight = head.project.weight
        visual = torch.nn.functional.linear(data[0], weight[:, :-2])
        meta = torch.nn.functional.linear(data[2], weight[:, -2:])
    return {"routing": routing_summary(observed['attention']),
            "representations": {name: representation_stats(observed[name]) for name in ('raw_steps','projected_steps','summary')},
            "projected_input_contribution_rms": {"visual": float(visual.square().mean().sqrt()),
                                                  "metadata": float(meta.square().mean().sqrt()),
                                                  "note": "Projection bias and positions excluded; magnitude is not importance."},
            "input_sensitivity": interventions(head, data, training),
            "metadata_recoverability": metadata_recoverability(
                {k: train_stages[k] for k in ('raw_steps','projected_steps','summary')}, training[2], training_episodes)}


def spatial_sensitivity(head, encoder, prepared, data, grid=3):
    """One predeclared case: content-region occlusion -> action probability.

    This is a perturbation map through CNN + head, never an attention pixel map.
    One image is re-encoded per edit; all other embeddings remain unchanged.
    """
    head.eval(); encoder.eval()
    baseline = float(probabilities(head,data)[0])
    device = data[0].device
    result = {}
    for role in range(7):
        image = prepared['goal'] if role == 0 else prepared['history'][role-1]
        geometry = prepared['preparation']['goal'] if role == 0 else prepared['preparation']['history'][role-1]
        left,top,_,_ = geometry['padding_ltrb']
        width,height = geometry['resized_size_wh']
        cells = []
        for row in range(grid):
            for col in range(grid):
                x0,x1 = left+col*width//grid,left+(col+1)*width//grid
                y0,y1 = top+row*height//grid,top+(row+1)*height//grid
                edited = image.clone(); edited[:,y0:y1,x0:x1] = 0.
                with torch.no_grad():
                    feature = encoder(edited.unsqueeze(0).to(device))
                    h,g = data[0].clone(),data[1].clone()
                    if role == 0:
                        g[0] = feature[0]
                    else:
                        h[0,role-1] = feature[0]
                    p = float(probabilities(head,(h,g,data[2],data[3]))[0])
                cells.append({'row':row,'column':col,'advance_probability_delta':p-baseline})
        result['goal' if role == 0 else 'f'+str(role)] = cells
    return {'baseline_advance_probability':baseline,'grid':grid,'roles':result,
            'note':'Rows top-to-bottom, columns left-to-right over resized image content, excluding padding. Each region replaced with normalized zero. Signed action-probability sensitivity, not wire detection, pixel attention or proof of correct reasoning.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=os.environ.get('DATA_DIR'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--partition', choices=('train','validation'), default='train')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--spatial-case-index', type=int, help='Optional single TRAINING-local case index; fixed 3x3 occlusion per image')
    args = parser.parse_args(argv)
    if not os.environ.get('PSCRATCH') or args.data_dir is None:
        parser.error('Run on Perlmutter with PSCRATCH and DATA_DIR')
    scratch = Path(os.environ['PSCRATCH']).resolve()
    args.data_dir, args.checkpoint, args.output_dir = [p.resolve() for p in (args.data_dir,args.checkpoint,args.output_dir)]
    if any(scratch not in p.parents for p in (args.data_dir,args.checkpoint,args.output_dir)):
        parser.error('All data/checkpoint/output paths must be beneath personal PSCRATCH')
    if args.output_dir.exists() or args.output_dir == args.data_dir or args.data_dir in args.output_dir.parents:
        parser.error('Use a new output directory outside the dataset')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved['kind'] != 'attention':
        parser.error('Attention checkpoint required')
    cfg = saved['config']
    if cfg['encoder'] != 'resnet18':
        parser.error('This diagnostic CLI currently supports ResNet checkpoints only')
    head = FeatureHead('attention',cfg['feature_dim'],cfg['width'],cfg['dropout'],
                       position_embeddings=cfg.get('position_embeddings',True),
                       classifier_width=cfg.get('classifier_width',cfg['width'])).to(args.device)
    head.load_state_dict(saved['head_state']); head.eval()
    from dataset import GuidewireDataset
    from cnn import FrameEncoder
    from preprocessing import DINOv2Preprocessor
    from evaluation import collect, check_partitions
    datasets = {p:GuidewireDataset(args.data_dir,split_position=cfg['split_position'],partition=p)
                for p in ('train','validation','test')}
    check_partitions(datasets)
    encoder = FrameEncoder().to(args.device).eval()
    if args.spatial_case_index is not None and not 0 <= args.spatial_case_index < len(datasets['train']):
        parser.error('Spatial index must select one training case')
    from cnn_attention_debug import freeze_and_snapshot, verify_frozen
    snapshot = freeze_and_snapshot(encoder.backbone)
    _, features = collect(datasets['train'],DINOv2Preprocessor(),{'resnet18':encoder})
    training = tensors(features['resnet18'],args.device)
    scale = torch.tensor(cfg['metadata_scale'],device=args.device)
    if not torch.allclose(metadata_scale(training[2]),scale,rtol=1e-5,atol=1e-6):
        raise ValueError('Checkpoint scales do not match current training metadata')
    training = (training[0],training[1],training[2]/scale,training[3])
    data = training
    if args.partition == 'validation':
        _, vf = collect(datasets['validation'],DINOv2Preprocessor(),{'resnet18':encoder})
        d = tensors(vf['resnet18'],args.device)
        data = (d[0],d[1],d[2]/scale,d[3])
    report = diagnose(head,data,training,features['resnet18']['episode'])
    if args.spatial_case_index is not None:
        index = args.spatial_case_index
        prepared = DINOv2Preprocessor().prepare_case(datasets['train'][index])
        report['spatial_sensitivity'] = spatial_sensitivity(
            head,encoder,prepared,tuple(x[index:index+1] for x in training))
    verify_frozen(encoder.backbone,snapshot)
    report.update(partition=args.partition, cases=len(data[3]), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  checkpoint_epoch=saved.get('checkpoint_epoch',saved.get('best_epoch')),
                  checkpoint_role=saved.get('checkpoint_role','selected'), checkpoint_config=cfg,
                  source_sha256={p:hashlib.sha256((Path(__file__).parent/p).read_bytes()).hexdigest()
                                 for p in ('temporal_diagnostics.py','temporal_models.py','temporal_evaluation.py','evaluation.py','cnn.py','preprocessing.py','dataset.py')},
                  torch_version=str(torch.__version__), device=args.device)
    report['encoder_unchanged'] = True
    args.output_dir.mkdir(parents=True,mode=0o700)
    write_json(args.output_dir/'diagnostics.json',report)
    lines = ['# Temporal diagnostics','', 'Aggregate checkpoint inspection; no semantic localization or final performance claim.', '',
             'Partition: {}. Cases: {}. Checkpoint: {}, epoch {}.'.format(
                 args.partition,len(data[3]),report['checkpoint_role'],report['checkpoint_epoch']), '',
             '## Latest-query temporal attention', '', '| Head | f1 | f2 | f3 | f4 | f5 | f6 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for i,row in enumerate(report['routing']['latest_query_mean_per_head']):
        lines.append('| {} | {} |'.format(i+1,' | '.join('{:.3f}'.format(v) for v in row)))
    lines += ['', '## Fixed-checkpoint input sensitivity', '', '| Replacement | BA delta | Mean probability change | Prediction flips |',
              '|---|---:|---:|---:|']
    for name,r in report['input_sensitivity']['controls'].items():
        lines.append('| {} | {:.3f} | {:.4f} | {:.3f} |'.format(name,r['balanced_accuracy_delta'],r['mean_absolute_probability_change'],r['prediction_flip_fraction']))
    lines += ['', '## Representation spread', '',
              '| Stage | RMS | Between-case std | Effective rank |','|---|---:|---:|---:|']
    for name,r in report['representations'].items():
        lines.append('| {} | {:.4f} | {:.4f} | {:.2f} |'.format(name,r['rms'],r['between_case_std_mean'],r['effective_rank']))
    lines += ['', 'Scale/rank differences alone do not establish semantic loss.', '',
              '## Metadata linear recoverability', '']
    recovery = report['metadata_recoverability']
    if recovery['status'] == 'completed':
        lines += ['Episode-disjoint holdout inside training only; fixed ridge decoder.', '',
                  '| Stage | Depth mean R2 | Time mean R2 |','|---|---:|---:|']
        for name,channels in recovery['results'].items():
            values=[channels[k]['mean_r2_nonconstant_steps'] for k in ('depth','time')]
            lines.append('| {} | {} |'.format(name,' | '.join('unavailable' if v is None else '{:.3f}'.format(v) for v in values)))
    else:
        lines.append(recovery['reason'])
    lines += ['', 'Detailed representation spread and training-only metadata recoverability are in diagnostics.json.',
              'Attention is across frames, not pixels. Replacement results may involve distribution shift; no retraining was performed.']
    if 'spatial_sensitivity' in report:
        private_lines = ['# Private single-training-case sensitivity', '']
        private_lines += ['', '## Training-case spatial perturbations', '',
                  'Signed ADVANCE-probability changes by content cell; this is not pixel attention.', '']
        for role,cells in report['spatial_sensitivity']['roles'].items():
            private_lines += [role, '', '| Row | Left | Center | Right |','|---|---:|---:|---:|']
            for row in range(3):
                values = [c['advance_probability_delta'] for c in cells if c['row']==row]
                private_lines.append('| {} | {} |'.format(row+1,' | '.join('{:+.4f}'.format(v) for v in values)))
            private_lines.append('')
        (args.output_dir/'spatial_sensitivity.private.md').write_text('\n'.join(private_lines)+'\n',encoding='utf-8')
    (args.output_dir/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Diagnostics completed; inspect aggregate summary.md. Full report/config stays on Perlmutter.')


if __name__ == '__main__':
    main()
