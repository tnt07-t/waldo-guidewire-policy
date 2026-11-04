"""Read-only review of saved development trials. Standard library only.

Reads fixed JSON report names; never opens datasets, predictions or checkpoints.
IPython is needed only by the notebook that displays the returned HTML.
"""
from datetime import datetime, timezone
from html import escape
import json
import math
import os
from pathlib import Path


CONFIG_KEYS = ('model', 'architecture', 'seed', 'split_position', 'hidden_size',
               'batch_size', 'lr', 'optimizer', 'weight_decay', 'max_epochs',
               'max_updates', 'patience', 'head_seconds', 'extraction_seconds',
               'overall_seconds', 'image_chunk_size')
METRIC_KEYS = ('loss', 'accuracy', 'balanced_accuracy', 'retract_recall',
               'advance_recall', 'confusion_matrix', 'n')
MAX_JSON_BYTES = 8 * 1024 * 1024


def default_roots():
    """Scan only known output trees, never all of scratch or the dataset."""
    scratch = os.environ.get('PSCRATCH')
    if not scratch:
        return []
    return [Path(scratch) / name for name in
            ('waldo-gru-training', 'waldo-static-training', 'waldo-evaluation')]


def _read(path, root):
    if not path.exists():
        return None
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Report link leaves its output root')
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('Report exceeds 8 MiB review limit')
    def reject_constant(value):
        raise ValueError('Nonfinite JSON constant')
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)


def _scores(value):
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in METRIC_KEYS if key in value}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _pct(value):
    return f'{100 * value:.2f}%' if _number(value) and 0 <= value <= 1 else 'unknown'


def _text(value):
    return escape('unknown' if value is None else str(value))


def _epoch(value):
    return type(value) is int and value > 0


def _history(value):
    if not isinstance(value, list):
        raise ValueError('History must be a list')
    result, previous = [], 0
    for row in value:
        if not isinstance(row, dict) or not _epoch(row.get('epoch')) or row['epoch'] <= previous:
            raise ValueError('History epochs must be positive and strictly increasing')
        previous = row['epoch']
        result.append(dict(epoch=row['epoch'], train=_scores(row.get('train')),
                           validation=_scores(row.get('validation'))))
    return result


def _identity(config, context):
    # Legacy defaults follow the loader, but only when a training config exists.
    if not config:
        return 'Unknown model (context missing)'
    model = config.get('model', 'gru')
    if model == 'static_latest_goal_metadata':
        return 'Static latest + goal + metadata'
    if model == 'static_grid_goal_metadata':
        return 'Static spatial grid + goal + metadata'
    if model == 'gru':
        return 'GRU / ' + str(config.get('architecture', 'late_goal'))
    return str(model) + ' / ' + str(config.get('architecture', 'unspecified'))


def discover_runs(roots, *, max_runs=500):
    """Return sanitized aggregate run records and readable discovery warnings.

    Supports the full PyTorch trainer and evaluation.py probe metrics. A partial
    or malformed run remains visible instead of silently looking successful.
    Dates use report mtime, which can change if artifacts are copied.
    """
    runs, warnings, seen = [], [], set()
    if not roots:
        return [], ['No output roots configured. In the notebook, set ROOTS to your scratch output directories.']
    for entry in roots:
        root = Path(entry).expanduser().resolve()
        if not root.is_dir():
            warnings.append(f'Output root not found: {root}')
            continue
        candidates = set()
        for name in ('summary.json', 'run_context.json', 'history.json', 'metrics.json'):
            candidates.update(path.parent for path in root.rglob(name))
        for directory in sorted(candidates):
            if directory.resolve() in seen:
                continue
            seen.add(directory.resolve())
            if len(runs) >= max_runs:
                warnings.append(f'Review limited to {max_runs} entries; narrow ROOTS to see other runs.')
                return runs, warnings
            label = root.name + '/' + str(directory.relative_to(root))
            problems = []
            def read(name, expected):
                try:
                    value = _read(directory / name, root)
                    if value is not None and not isinstance(value, expected):
                        raise ValueError('Unexpected report shape')
                    return value
                except (OSError, ValueError) as error:
                    problems.append(f'{name}: {type(error).__name__}')
                    return None
            summary = read('summary.json', dict)
            context = read('run_context.json', dict)
            raw_history = read('history.json', list)
            probes = read('metrics.json', dict)
            if summary is None and context is None and raw_history is None and probes is not None:
                results = probes.get('results', {})
                if not isinstance(results, dict):
                    warnings.append(label + ': malformed probe results')
                    continue
                timestamp = (directory / 'metrics.json').stat().st_mtime
                for name, scores in results.items():
                    if len(runs) >= max_runs:
                        warnings.append(f'Review limited to {max_runs} entries.')
                        return runs, warnings
                    if not isinstance(scores, dict):
                        warnings.append(label + ': malformed probe result')
                        continue
                    runs.append(dict(id=label + ' :: ' + str(name), path=str(directory),
                                     approach='Probe / ' + str(name), kind='probe', status='completed',
                                     config=dict(split_position=probes.get('split_position')),
                                     selected=_scores(scores.get('validation')),
                                     selected_train=_scores(scores.get('train')), final={}, history=[],
                                     best_epoch=None, completed_epochs=None, updates=None,
                                     timestamp=timestamp, revision=probes.get('git_commit'),
                                     dirty=probes.get('git_dirty'), encoder_unchanged=None,
                                     reload_verified=None, parameters=None,
                                     feature_roles=None, metadata_scaling='per-column centered scaling (evaluation.py)',
                                     timing={}, stop_reason='probe selection', problems=list(problems),
                                     selected_C=scores.get('selected_C')))
                continue
            summary, context = summary or {}, context or {}
            config = context.get('config', {})
            if not isinstance(config, dict):
                problems.append('run_context.json: invalid config')
                config = {}
            config = {key: config[key] for key in CONFIG_KEYS if key in config}
            try:
                history = _history(raw_history) if raw_history is not None else []
            except ValueError:
                history = []
                problems.append('history.json: invalid epoch order')
            best_epoch = summary.get('best_epoch')
            selected_row = next((r for r in history if r['epoch'] == best_epoch), {})
            final = history[-1] if history else {}
            head = context.get('head', {})
            if not isinstance(head, dict):
                head = {}
            reports = [directory / n for n in ('summary.json', 'run_context.json', 'history.json')
                       if (directory / n).is_file()]
            timestamp = max((p.stat().st_mtime for p in reports), default=0)
            status = summary.get('status', 'incomplete')
            if problems:
                status = 'report_error'
            runs.append(dict(id=label, path=str(directory), kind='training',
                             approach=_identity(config, context), config=config,
                             selected=_scores(summary.get('best_validation')),
                             selected_train=_scores(selected_row.get('train')), final=final,
                             history=history, best_epoch=best_epoch,
                             completed_epochs=summary.get('completed_epochs'), updates=summary.get('updates'),
                             timestamp=timestamp, revision=context.get('git_commit'), dirty=context.get('git_dirty'),
                             encoder_unchanged=summary.get('encoder_unchanged'),
                             reload_verified=summary.get('reload_verified'),
                             parameters=head.get('trainable_parameters'), feature_roles=head.get('feature_roles'),
                             metadata_scaling=context.get('metadata_scaling'),
                             timing={k: summary.get(k) for k in ('extraction_seconds', 'head_seconds',
                                     'process_seconds', 'peak_cuda_allocated_bytes')},
                             stop_reason=summary.get('stop_reason'), problems=problems, status=status))
    return runs, warnings


def _chart(series, title, metric, *, selected_epoch=None, x_label='Epoch', connect=True):
    """Self-contained SVG with separate paths across missing metric points."""
    valid = []
    for label, rows, color, dashed in series:
        points = [(epoch, value) for epoch, value in rows
                  if _epoch(epoch) and _number(value) and value >= 0
                  and (metric == 'loss' or value <= 1)]
        if points:
            valid.append((label, rows, points, color, dashed))
    if not valid:
        return '<p class="muted">No saved ' + _text(metric) + ' curve for this entry.</p>'
    xmin = min(e for _, _, points, _, _ in valid for e, _ in points)
    xmax = max(e for _, _, points, _, _ in valid for e, _ in points)
    ymax = max(v for _, _, points, _, _ in valid for _, v in points) * 1.08 if metric == 'loss' else 1.
    ymax = max(ymax, .01)
    x = lambda e: 60 + 660 * (e - xmin) / max(xmax - xmin, 1)
    y = lambda v: 235 - 185 * v / ymax
    legend_height = len(valid) * 19
    output = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 {290 + legend_height}" role="img" aria-label="{_text(title)}">',
              '<rect width="100%" height="100%" fill="white"/>',
              f'<text x="60" y="24" font-size="16" fill="#182635">{_text(title)}</text>']
    for i in range(5):
        value = ymax * i / 4
        output.append(f'<path d="M60 {y(value):.2f} H720" stroke="#dce1e5" fill="none"/>')
        label = f'{value:.2f}' if metric == 'loss' else f'{value*100:.0f}%'
        output.append(f'<text x="8" y="{y(value)+4:.2f}" font-size="12" fill="#465464">{label}</text>')
    for label, rows, points, color, dashed in valid:
        segments, segment = [], []
        for epoch, value in rows:
            if (_epoch(epoch) and _number(value) and value >= 0
                    and (metric == 'loss' or value <= 1)):
                segment.append(f'{x(epoch):.2f},{y(value):.2f}')
            elif segment:
                segments.append(segment); segment = []
        if segment:
            segments.append(segment)
        dash = ' stroke-dasharray="5 4"' if dashed else ''
        for points_string in (segments if connect else []):
            output.append(f'<polyline points="{" ".join(points_string)}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        for epoch, value in points:
            output.append(f'<circle cx="{x(epoch):.2f}" cy="{y(value):.2f}" r="2" fill="{color}"/>')
    if _epoch(selected_epoch) and xmin <= selected_epoch <= xmax:
        output.append(f'<path d="M{x(selected_epoch):.2f} 45 V235" stroke="#777" stroke-dasharray="3 4"/>')
        output.append(f'<text x="{min(x(selected_epoch)+4, 590):.2f}" y="44" font-size="11" fill="#555">Selected epoch {selected_epoch}</text>')
    output.extend([f'<text x="60" y="255" font-size="12">{xmin}</text>',
                   f'<text x="700" y="255" font-size="12">{xmax}</text>',
                   f'<text x="270" y="275" font-size="12">{_text(x_label)}</text>'])
    for i, (label, _, _, color, dashed) in enumerate(valid):
        output.append(f'<text x="60" y="{295+i*19}" fill="{color}" font-size="12">{_text(label)}</text>')
    output.append('</svg>')
    return ''.join(output)


def _score_table(rows):
    headers = ('Measurement', 'Loss', 'Accuracy', 'Balanced accuracy', 'RETRACT recall', 'ADVANCE recall', 'Cases')
    result = ['<table><thead><tr>' + ''.join('<th>'+h+'</th>' for h in headers) + '</tr></thead><tbody>']
    for label, scores in rows:
        if not scores:
            continue
        loss = f"{scores['loss']:.4f}" if _number(scores.get('loss')) else 'unknown'
        values = [label, loss] + [_pct(scores.get(key)) for key in
                  ('accuracy', 'balanced_accuracy', 'retract_recall', 'advance_recall')] + [scores.get('n')]
        result.append('<tr>'+''.join('<td>'+_text(v)+'</td>' for v in values)+'</tr>')
    return ''.join(result)+'</tbody></table>'


def render_review(runs, warnings=(), *, sort_by='recorded', selected_ids=None, notes=None,
                  comparison_ids=None, compact_index=False, include_progress=True):
    """Render aggregate HTML/SVG; no JavaScript, network, writes or dependencies.

    SELECTED_IDS filters comparison/detail panels only; all discovered entries
    remain in the chronological overview. Missing metadata is never inferred
    from directory names. Notes stay separate from immutable saved evidence.
    """
    if sort_by not in ('recorded', 'validation_ba'):
        raise ValueError('sort_by must be recorded or validation_ba')
    notes = notes or {}
    chron = sorted(runs, key=lambda r: (r['timestamp'], r['id']))
    order = {r['id']: i for i, r in enumerate(chron, 1)}
    if sort_by == 'validation_ba':
        runs = sorted(chron, key=lambda r: (-r['selected'].get('balanced_accuracy')
                      if _number(r['selected'].get('balanced_accuracy')) else float('inf'), r['id']))
    else:
        runs = chron
    chosen = [r for r in runs if selected_ids is None or r['id'] in selected_ids]
    comparisons = [r for r in chosen if comparison_ids is None or r['id'] in comparison_ids]
    output = ['''<div class="waldo-review"><style>
    .waldo-review {font:14px/1.5 system-ui,sans-serif;color:#182635;background:white;padding:18px;max-width:1250px}
    .waldo-review h2 {font-size:22px;margin:24px 0 8px}.waldo-review h3 {font-size:17px;margin:10px 0}
    .waldo-review table {border-collapse:collapse;width:100%;font-size:12px;margin:12px 0}
    .waldo-review th {text-align:left;background:#f0f3f5}.waldo-review td,.waldo-review th {padding:8px;border-bottom:1px solid #dce1e5;vertical-align:top}
    .waldo-review .scroll {overflow-x:auto}.waldo-review .muted {color:#596776}
    .waldo-review .notice {border-left:3px solid #8a6318;padding:8px 12px;background:#fff9ec}
    .waldo-review details {border-top:1px solid #cbd4dc;padding:12px 0}.waldo-review summary {cursor:pointer;font-weight:600}
    .waldo-review .charts {display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
    .waldo-review svg {width:100%;height:auto}.waldo-review code {overflow-wrap:anywhere}
    </style><h2>Saved experiment review</h2>
    <p>Development validation only. Reads saved reports; does not train, use a GPU, load images or open checkpoints.</p>
    <p class="notice">Private derived results: keep this executed notebook and its outputs on Perlmutter.
    Scores are selected validation results, not final-test performance. Different seeds, splits, optimizers,
    scalers and capacities are not automatically matched comparisons. No averaging across incompatible trials.</p>''']
    if warnings:
        output.append('<details open><summary>Discovery notices</summary><ul>'+
                      ''.join('<li>'+_text(w)+'</li>' for w in warnings)+'</ul></details>')
    if not runs:
        if compact_index:
            output.append('<p>No saved reports found in the standard output trees. The prefilled journal above still shows documented trials. No training is rerun and no epoch curves are fabricated.</p>')
        else:
            output.append('<p>No saved runs found. Set ROOTS in the settings cell to the directories containing your existing outputs, then Run All again. No training is needed.</p>')
        return ''.join(output)+'</div>'
    if compact_index:
        output.append(f'<details><summary>All discovered saved reports ({len(runs)})</summary>')
    output.append('<h2>Trial index</h2><p class="muted">Trial numbers follow report-file modification time (UTC), not verified experiment start dates. Copied reports can move in this ordering. Change SORT_BY in the settings cell to rank by selected validation BA.</p>')
    headings = ('Trial / recorded UTC', 'Run / approach', 'State', 'Split / seed', 'Optimizer / lr / decay',
                'Selected / final epoch', 'Train BA at selection', 'Validation BA', 'RETRACT / ADVANCE recall', 'Revision')
    output.append('<div class="scroll"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headings)+'</tr></thead><tbody>')
    for run in runs:
        config, score = run['config'], run['selected']
        recorded = datetime.fromtimestamp(run['timestamp'], timezone.utc).strftime('%Y-%m-%d %H:%M')
        note = notes.get(run['id'], {})
        label = note.get('label', run['id'])
        values = [f"{order[run['id']]} / {recorded}", str(label)+' / '+run['approach'], run['status'],
                  f"{config.get('split_position', 'unknown')} / {config.get('seed', 'unknown')}",
                  f"{config.get('optimizer', 'unknown')} / {config.get('lr', 'unknown')} / {config.get('weight_decay', 'unknown')}",
                  f"{run['best_epoch']} / {run['completed_epochs']}" if run['kind']=='training' else 'probe (no epochs)',
                  _pct(run['selected_train'].get('balanced_accuracy')), _pct(score.get('balanced_accuracy')),
                  _pct(score.get('retract_recall'))+' / '+_pct(score.get('advance_recall')),
                  str(run['revision'])[:10] if run['revision'] else 'unknown']
        output.append('<tr>'+''.join('<td>'+_text(v)+'</td>' for v in values)+'</tr>')
    output.append('</tbody></table></div>')
    if compact_index:
        output.append('</details>')
    progress=[]
    approaches=sorted({r['approach'] for r in chosen if r['status']=='completed'})
    progress_colors=('#1769aa', '#a8461e', '#41744a', '#76569a', '#6c681d', '#545b63')
    for index, approach in enumerate(approaches):
        progress.append((approach, [(order[r['id']], r['selected'].get('balanced_accuracy'))
                                   for r in chosen if r['approach']==approach and r['status']=='completed'],
                         progress_colors[index % len(progress_colors)], False))
    if include_progress:
        output.append('<h2>Selected validation scores over saved trials</h2><p class="muted">One point per completed entry. Trial order follows report-file timestamps; these are different selected models, not a final-test learning trajectory.</p>')
        output.append(_chart(progress, 'Selected validation balanced accuracy', 'balanced_accuracy',
                             x_label='Trial number (report-file order)', connect=False))
    output.append('<h2>Comparison overlays</h2><p class="muted">Selected entries only. Dashed curves are validation; solid curves are training. Legend numbers refer to the trial index. Selection uses each run’s saved rule; overlays do not establish causal improvement.</p>')
    colors = ('#1769aa', '#a8461e', '#41744a', '#76569a', '#6c681d', '#545b63')
    if len(comparisons) > 12:
        output.append('<p class="notice">Select up to 12 run IDs in SELECTED_IDS for readable overlays; all entries remain listed above and below.</p>')
    else:
        output.append('<div class="charts">')
        for metric, title in [('loss', 'Cross-entropy loss'), ('balanced_accuracy', 'Balanced accuracy')]:
            series=[]
            for index, run in enumerate(comparisons):
                if run['kind'] != 'training':
                    continue
                for part in ('train', 'validation'):
                    label=f"{notes.get(run['id'], {}).get('label', 'Trial '+str(order[run['id']]))} — {part}"
                    series.append((label, [(r['epoch'], r[part].get(metric)) for r in run['history']], colors[index % len(colors)], part=='validation'))
            output.append(_chart(series, title+' by epoch', metric))
        output.append('</div>')
    output.append('<h2>Trial details and decisions</h2><p>Click a trial below to see curves, selected-versus-final metrics, confusion counts and saved settings. Known trials already include their question, observation and decision. Additional saved reports are listed with their recorded settings.</p>')
    for run in chosen:
        number=order[run['id']]
        note=notes.get(run['id'], {})
        output.append(f'<details><summary>Trial {number}: {_text(note.get("label", run["id"]))} — {_pct(run["selected"].get("balanced_accuracy"))} validation BA</summary>')
        for key in ('question', 'observation', 'decision'):
            if note.get(key):
                output.append('<p><strong>'+key.title()+':</strong> '+_text(note[key])+'</p>')
        output.append('<p class="muted">Source: <code>'+_text(run['path'])+'</code></p>')
        if run['problems']:
            output.append('<p class="notice">'+_text('; '.join(run['problems']))+'</p>')
        output.append(_score_table([('Selected train', run['selected_train']), ('Selected validation', run['selected']),
                                    ('Final evaluated train', run['final'].get('train', {})),
                                    ('Final evaluated validation', run['final'].get('validation', {}))]))
        cm=run['selected'].get('confusion_matrix')
        if (isinstance(cm,list) and len(cm)==2 and all(isinstance(r,list) and len(r)==2 for r in cm)
                and all(type(v) is int and v>=0 for row in cm for v in row)):
            output.append('<p>Selected validation confusion counts: rows = true labels, columns = predictions.</p><table><tr><th>True / predicted</th><th>RETRACT</th><th>ADVANCE</th></tr>')
            for label,row in zip(('RETRACT','ADVANCE'),cm):
                output.append('<tr><th>'+label+'</th>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>')
            output.append('</table>')
        if run['kind']=='training':
            output.append('<div class="charts">')
            for metric,title in [('loss','Cross-entropy loss'),('balanced_accuracy','Balanced accuracy'),
                                  ('retract_recall','RETRACT recall'),('advance_recall','ADVANCE recall')]:
                series=[(part,[(r['epoch'],r[part].get(metric)) for r in run['history']],color,part=='validation')
                        for part,color in [('train','#1769aa'),('validation','#a8461e')]]
                output.append(_chart(series,title,metric,selected_epoch=run['best_epoch']))
            output.append('</div>')
        else:
            output.append('<p class="muted">Probe report: no epoch history was saved. Centered per-column scaling and validation-selected C differ from the PyTorch recipe.</p>')
        settings=dict(run['config'], metadata_scaling=run['metadata_scaling'],
                      trainable_parameters=run['parameters'], feature_roles=run['feature_roles'],
                      git_commit=run['revision'], git_dirty=run['dirty'], stop_reason=run['stop_reason'],
                      updates=run['updates'], encoder_unchanged=run['encoder_unchanged'],
                      reload_verified=run['reload_verified'], **run['timing'])
        if run['kind']=='probe':
            settings['selected_C']=run['selected_C']
        output.append('<details><summary>Saved aggregate settings and checks</summary><pre>'+_text(json.dumps(settings,indent=2))+'</pre></details>')
        output.append('<p class="muted">Training/extraction durations are not end-to-end decision latency. Unknown means the artifact did not record that field; folder names are not treated as provenance.</p></details>')
    output.append('</div>')
    return ''.join(output)


def load_journal():
    """Load only the curated, permitted aggregate log shipped with the code."""
    path = Path(__file__).resolve().parent / 'docs' / 'grp2' / 'EXPERIMENT_HISTORY.json'
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('format_version') != 1 or not isinstance(value.get('trials'), list):
        raise ValueError('Unsupported experiment journal')
    return value


def link_journal(journal, runs):
    """Conservatively associate known trials with private reports, without merges.

    A documented path or full epoch/score fingerprint is required. Conflicting
    metrics/configs and multiple candidates stay explicit; names alone never
    replace saved provenance. Recorded snapshots are never overwritten.
    """
    linked=[]
    for trial in journal['trials']:
        candidates, conflicts=[],[]
        for run in runs:
            if run['kind'] != 'training':
                continue
            hinted=bool(trial.get('artifact_hint')) and run['id']==trial['artifact_hint']
            score=run['selected'].get('balanced_accuracy')
            expected=trial.get('validation',{}).get('balanced_accuracy')
            fingerprint=(run['best_epoch']==trial.get('best_epoch') and
                         run['completed_epochs']==trial.get('completed_epochs') and
                         _number(score) and _number(expected) and abs(score-expected)<.0001)
            if not hinted and not fingerprint:
                continue
            mismatch=[]
            for key in ('best_epoch','completed_epochs'):
                if run.get(key) is not None and trial.get(key) is not None and run[key]!=trial[key]:
                    mismatch.append(key)
            if _number(score) and _number(expected) and abs(score-expected)>=.0001:
                mismatch.append('validation balanced accuracy')
            for key,value in trial.get('protocol',{}).items():
                actual=run['config'].get(key)
                # Old training configs follow the same loader defaults.
                if actual is None and run['config'] and key=='model': actual='gru'
                if actual is None and run['config'] and key=='architecture': actual='late_goal'
                if actual is not None and actual!=value:
                    mismatch.append(key)
            if mismatch:
                if hinted:
                    conflicts.append(run['id']+': '+', '.join(mismatch))
            else:
                candidates.append(run['id'])
        if len(candidates)==1:
            evidence='Saved report linked automatically'
            artifact_id=candidates[0]
        elif len(candidates)>1:
            evidence='Multiple matching reports; no automatic choice'
            artifact_id=None
        elif conflicts:
            evidence='Report conflicts with recorded trial; inspect details'
            artifact_id=None
        else:
            evidence='Recorded result retained; saved report not located'
            artifact_id=None
        linked.append(dict(trial=trial,artifact_id=artifact_id,evidence=evidence,
                           candidates=candidates,conflicts=conflicts))
    return linked


def render_journal(journal, linked, runs):
    """Concise prefilled history; missing curves are never fabricated."""
    out=['<section class="waldo-journal" style="font:14px/1.5 system-ui,sans-serif;color:#182635;background:white;padding:18px;max-width:1250px">',
         '<h2>What we tried — experiment journal</h2>',
         '<p>'+_text(journal.get('takeaway',''))+'</p>',
         '<p>Dates and narrative come from the recorded project log. Scores below are the owner-reported historical snapshots; linked private reports remain separate evidence. No code edits or GPU needed.</p>',
         '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:13px"><thead><tr>']
    out.extend('<th style="text-align:left;padding:8px;border-bottom:1px solid #cbd4dc">'+h+'</th>' for h in
               ('Date / trial','What changed','Train BA at selection','Validation BA','Saved evidence'))
    out.append('</tr></thead><tbody>')
    run_by_id={r['id']:r for r in runs}
    for item in linked:
        trial=item['trial']
        linked_run=run_by_id.get(item['artifact_id'])
        train_ba=(linked_run['selected_train'].get('balanced_accuracy') if linked_run else None)
        if _number(train_ba):
            train_label=_pct(train_ba)+' (saved)'
        else:
            train_label=_pct(trial.get('selected_train',{}).get('balanced_accuracy'))+' (recorded)'
        values=[trial.get('date','')+' / '+trial['title'],trial.get('change',''),
                train_label,
                _pct(trial.get('validation',{}).get('balanced_accuracy')),item['evidence']]
        out.append('<tr>'+''.join('<td style="padding:8px;border-bottom:1px solid #dce1e5;vertical-align:top">'+_text(v)+'</td>' for v in values)+'</tr>')
    out.append('</tbody></table></div>')
    series=[]
    colors=('#1769aa','#a8461e','#41744a','#76569a','#6c681d','#545b63')
    used={x['artifact_id'] for x in linked if x['artifact_id']}
    for index,item in enumerate(linked,1):
        trial=item['trial']
        series.append((f"{index}. {trial['title']} (recorded)",[(index,trial.get('validation',{}).get('balanced_accuracy'))],colors[(index-1)%len(colors)],False))
    extra=sorted((r for r in runs if r['kind']=='training' and r['status']=='completed' and r['id'] not in used),key=lambda r:(r['timestamp'],r['id']))
    for index,run in enumerate(extra,len(linked)+1):
        series.append((f"{index}. {run['id']} (saved report)",[(index,run['selected'].get('balanced_accuracy'))],colors[(index-1)%len(colors)],False))
    out.append(_chart(series,'Selected validation balanced accuracy across trials','balanced_accuracy',x_label='Documented order; additional reports follow',connect=False))
    out.append('<p>Recorded main trials use the documented sequence; additional saved reports follow in file-time order. Points do not imply statistical significance or a final-test improvement trend.</p>')
    out.append('<h3>Why each trial mattered</h3>')
    for item in linked:
        trial=item['trial']
        out.append('<details style="border-top:1px solid #cbd4dc;padding:10px 0"><summary style="cursor:pointer;font-weight:600">'+_text(trial['title'])+' — '+_pct(trial.get('validation',{}).get('balanced_accuracy'))+' validation BA</summary>')
        for key in ('question','observation','decision','provenance'):
            if trial.get(key):
                out.append('<p><strong>'+key.title()+':</strong> '+_text(trial[key])+'</p>')
        out.append('<p>Selected epoch '+_text(trial.get('best_epoch'))+'; stopped after '+_text(trial.get('completed_epochs'))+' epochs / '+_text(trial.get('updates'))+' updates.</p>')
        out.append('<p><strong>Evidence:</strong> '+_text(item['evidence'])+'. Source: '+_text(trial.get('source'))+'.</p>')
        if item['conflicts']:
            out.append('<p>'+_text('; '.join(item['conflicts']))+'</p>')
        if len(item['candidates'])>1:
            out.append('<p>Candidate reports: '+_text('; '.join(item['candidates']))+'</p>')
        run=run_by_id.get(item['artifact_id'])
        if not run or not run['history']:
            out.append('<p>No saved epoch history located for this recorded trial. Its aggregate result remains logged; no curve is synthesized or training rerun.</p>')
        else:
            out.append('<p>Saved curves and final-epoch details appear below under <strong>'+_text(trial['title'])+'</strong>.</p>')
        out.append('</details>')
    out.append('<details style="border-top:1px solid #cbd4dc;padding:10px 0"><summary style="cursor:pointer;font-weight:600">Earlier setup, diagnostics and eight-case learning check</summary>')
    for work in journal.get('other_work',[]):
        out.append('<p><strong>'+_text(work['date']+' · '+work['title'])+'</strong> — '+_text(work['status'])+'</p><p>'+_text(work.get('question',''))+' '+_text(work.get('outcome',''))+'</p>')
    out.append('</details>')
    probe=journal.get('probe_baseline',{})
    if probe:
        out.append('<details style="border-top:1px solid #cbd4dc;padding:10px 0"><summary style="cursor:pointer;font-weight:600">'+_text(probe.get('date','')+' · '+probe.get('title','Probe comparison'))+'</summary><p>'+_text(probe.get('note',''))+'</p>')
        out.append(_score_table([(r['approach'],dict(balanced_accuracy=r['validation_ba'],retract_recall=r['retract_recall'],advance_recall=r['advance_recall'])) for r in probe.get('scores',[])]))
        out.append('<p>Train BA (recorded): '+_text('; '.join(r['approach']+' '+_pct(r['train_ba']) for r in probe.get('scores',[])))+'</p></details>')
    out.append('<details style="border-top:1px solid #cbd4dc;padding:10px 0"><summary style="cursor:pointer;font-weight:600">Discussed next steps — not completed experiments</summary>')
    for option in journal.get('next_options',[]):
        state=option['status']
        model=option.get('model')
        if model and any(r['config'].get('model')==model and r['status']=='completed' for r in runs):
            state='Completed saved report discovered; see trial details below'
        elif option.get('implementation_file') and (Path(__file__).resolve().parent/option['implementation_file']).is_file():
            state='Local implementation file present; no completed real run found in these reports (verification not inferred)'
        out.append('<p><strong>'+_text(option['title'])+'</strong> — '+_text(state)+'. '+_text(option['reason'])+'</p>')
    out.append('</details></section>')
    return ''.join(out)


def review_saved_experiments(roots=None):
    """No-edit notebook entrypoint: historical journal plus live saved reports."""
    runs,notices=discover_runs(default_roots() if roots is None else roots)
    journal=load_journal()
    linked=link_journal(journal,runs)
    notes={item['artifact_id']:dict(label=item['trial']['title'],
                                  question=item['trial'].get('question'),
                                  observation=item['trial'].get('observation'),
                                  decision=item['trial'].get('decision'))
           for item in linked if item['artifact_id']}
    known=[item['artifact_id'] for item in linked if item['artifact_id']]
    other=sorted((r for r in runs if r['kind']=='training' and r['status']=='completed' and r['history'] and r['id'] not in known),key=lambda r:(r['timestamp'],r['id']))
    # Include recent discoveries automatically; all reports retain detail panels.
    comparisons=(known+[r['id'] for r in other])[-4:]
    return render_journal(journal,linked,runs)+render_review(
        runs,notices,notes=notes,comparison_ids=comparisons,compact_index=True,include_progress=False)


def prepare_notebook():
    """Create a private notebook copy on scratch, preserving an existing copy."""
    scratch = os.environ.get('PSCRATCH')
    if not scratch or not Path(scratch).is_dir():
        raise ValueError('Set PSCRATCH to your existing scratch directory first')
    destination = Path(scratch).resolve() / 'waldo-review' / 'experiment_review.ipynb'
    if destination.exists():
        return destination
    # The copy embeds only local code/output-root paths, never report contents.
    source = Path(__file__).resolve().with_suffix('.ipynb')
    notebook = json.loads(source.read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            if 'from experiment_review import' in code:
                code = 'import sys\nsys.path.insert(0, ' + repr(str(source.parent)) + ')\n' + code
            if 'ROOTS = default_roots()' in code:
                code = code.replace('ROOTS = default_roots()', 'ROOTS = [Path(p) for p in ' +
                                    repr([str(p) for p in default_roots()]) + ']')
            cell['source'] = code.splitlines(keepends=True)
            cell['execution_count'], cell['outputs'] = None, []
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(notebook, handle, indent=1)
        handle.write('\n')
    return destination


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare a private saved-experiment review notebook; no training or GPU.')
    parser.add_argument('--prepare-notebook', action='store_true', required=True)
    parser.parse_args()
    try:
        path = prepare_notebook()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print('Private review notebook (existing copy is preserved):')
    print(path)
    print('JupyterLab: File > Open from Path; paste the path above. Then Run > Run All Cells.')


if __name__ == '__main__':
    main()
