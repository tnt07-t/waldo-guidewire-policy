# Group 1 — current training setup

The working architecture is frozen ResNet-18, shared LayerNorm and 512-to-64
projection, a one-layer GRU with hidden size 64, and a 128-to-64-to-2 MLP. The
head has 67,586 trainable parameters. Depth/time normalization uses training
statistics. The goal is a separate reference, not a history step.

The classifier returns probabilities in class order `[RETRACT, ADVANCE]`.
Training and evaluation pass `return_logits=True` for class-weighted
cross-entropy. Softmax is not an overfitting remedy.

## Regularization comparison and CSV recording

The trainer now provides a true unregularized `baseline` (weight decay, dropout
and feature noise all zero), plus `light`, `medium` and `heavy` weight-decay
presets of 0.01, 0.1 and 1.0. All four use 200 epochs by default, dropout/noise
off, and identical architecture and training controls. These strengths are
candidates, not validated improvements. See [the hypotheses, settings and
run commands](NEXT_EXPERIMENTS.md).

The old baseline remains available as `legacy-baseline` (weight decay 1.0,
dropout 0.3, noise 0, 200 epochs). `moderate`, `strict` and `extra-strict` preserve
their previous 50-epoch, noise-1/2/3 configurations. Historical results from
these settings do not describe the new unregularized control.

Run on Perlmutter inside an allocation with `DATA_DIR` configured:

```sh
python train_cnngru.py --data-dir "$DATA_DIR" --split-position 0 \
  --sweep --seeds 0 1 2 \
  --output-dir "${PSCRATCH:?PSCRATCH must be set}/waldo-regularization/split0-trial1"
```

The sweep saves per-epoch CSVs for all four presets and all three seeds, plus
`summary.csv` with final-window metrics and paired baseline differences. A single
run can save with `--config light --metrics-csv PATH`. CSVs record training and
validation accuracy/loss, additional classification metrics, and effective run
settings. Completed rows are flushed immediately and existing outputs cannot be
overwritten. Without a CSV option, single runs remain terminal/in-memory only.

Learning rate defaults to 0.001, batch size to 16, and summary window to 20.
Training completes the requested epoch count; there is no best-epoch selection,
early stopping or checkpoint saving. `run(args, features=...)` still returns the
final model and metrics; `echo=False` suppresses terminal output.

Optional controls include `--shuffle-train-labels` for a seeded training-label
null control, `--summary-window N`, and `--feature-cache` for frozen features in
an external Perlmutter directory. The trainer loads only training and validation
partitions. Real data, derived features and experiment outputs stay on Perlmutter.

## Local verification

Use synthetic inputs without accessing the private dataset or downloading weights:

```sh
python -m unittest discover -s tests -p 'test_train_cnngru.py' -v
```

All 35 synthetic training tests passed locally. Tests cover effective presets, CSV accuracy and interruption handling, matched
sweeps and baseline differences, preservation of inputs, and existing model and
training behavior. Real-data regularization outcomes remain unmeasured.
