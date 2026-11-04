"""Create a private SVG of training and validation curves without extra packages."""
import argparse
import html
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history', type=Path, help='Private history.json from a training run')
    parser.add_argument('--output', type=Path, required=True, help='New SVG path on Perlmutter')
    parser.add_argument('--title', default='GRU training and validation by epoch')
    args = parser.parse_args()
    history = json.loads(args.history.read_text())
    if args.output.exists():
        parser.error('Output already exists; choose a new filename')
    if not history:
        parser.error('History is empty')
    epochs = [row['epoch'] for row in history]
    if any(type(e) is not int or e < 1 for e in epochs) or epochs != sorted(set(epochs)):
        parser.error('Epochs must be positive and strictly increasing')
    panels = [('Cross-entropy loss (lower is better)', 'loss'),
              ('Balanced accuracy (higher is better)', 'balanced_accuracy')]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="700" viewBox="0 0 960 700">',
           '<rect width="960" height="700" fill="white"/>',
           '<style>text{font:16px sans-serif;fill:#222}.grid{stroke:#ddd}.axis{stroke:#333}</style>',
           f'<text x="50" y="30" font-weight="bold">{html.escape(args.title)}</text>',
           '<text x="620" y="30" fill="#1769aa">Training</text><line x1="590" y1="24" x2="612" y2="24" stroke="#1769aa" stroke-width="3"/>',
           '<text x="790" y="30">Validation</text><line x1="758" y1="24" x2="780" y2="24" stroke="#c0392b" stroke-width="3"/>']
    for index, (title, metric) in enumerate(panels):
        values = [[float(row[part][metric]) for row in history] for part in ('train', 'validation')]
        if any(not math.isfinite(v) or v < 0 or (metric == 'balanced_accuracy' and v > 1)
               for series in values for v in series):
            parser.error('Metrics must be finite and within their expected range')
        x0, x1 = 85, 910
        top, bottom = 85 + index * 310, 310 + index * 310
        ymax = 1.0 if metric == 'balanced_accuracy' else max(max(map(max, values)), 0.01) * 1.05
        svg.append(f'<text x="{x0}" y="{top-20}">{html.escape(title)}</text>')
        for tick in range(5):
            value = ymax * tick / 4
            y = bottom - (bottom-top) * tick / 4
            svg.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/><text x="15" y="{y+5:.1f}">{value:.2f}</text>')
        svg.append(f'<line class="axis" x1="{x0}" y1="{bottom}" x2="{x1}" y2="{bottom}"/>')
        for series, color in zip(values, ('#1769aa', '#c0392b')):
            span = max(epochs[-1] - epochs[0], 1)
            points = ' '.join(f'{x0+(x1-x0)*(e-epochs[0])/span:.1f},{bottom-(bottom-top)*v/ymax:.1f}' for e,v in zip(epochs,series))
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{x0}" y="{bottom+25}">Epoch {epochs[0]}</text><text x="{x1-85}" y="{bottom+25}">Epoch {epochs[-1]}</text>')
    svg.append('</svg>')
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(svg) + '\n')
    print('Saved', args.output)


if __name__ == '__main__':
    main()
