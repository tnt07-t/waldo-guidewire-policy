# Group 1 — regularization hypothesis experiment

The experiment compares three separately trained CNN–GRU models with light,
medium and heavy regularization against an unregularized control. These are
four instances of the same architecture, not different architectures. Strengths
are experimental settings; their effectiveness has not been established.

## Matched run configurations

| CLI preset | AdamW weight decay | MLP dropout | Feature noise | Epochs |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 0 | 0 | 0 | 200 |
| `light` | 0.01 | 0 | 0 | 200 |
| `medium` | 0.1 | 0 | 0 | 200 |
| `heavy` | 1.0 | 0 | 0 | 200 |

Only weight decay changes. All use the frozen ResNet-18, shared LayerNorm and
512-to-64 projection, 64-unit GRU, and 128-to-64-to-2 classifier (67,586 trainable
parameters). Learning rate is 0.001, batch size 16 and gradient clipping 1.0.
Class weights and metadata normalization come from training only. The architecture,
frozen encoder and optimization controls remain common to all four runs;
“unregularized” means zero explicit weight decay, dropout and feature noise.

The previous default was already regularized (weight decay 1.0, dropout 0.3).
It is now named `legacy-baseline`. The existing `moderate`, `strict` and
`extra-strict` noise presets keep their previous settings and 50-epoch budgets.
They are excluded from this experiment. Historical baseline results must not be
relabeled as the new unregularized control.

## Hypotheses and comparison rules

For each severity, the null hypothesis is that regularization does not improve
validation balanced accuracy relative to the matched baseline. The alternative
is a positive improvement. The working prediction is that light or medium decay
reduces overfitting, while heavy decay may underfit. This prediction is untested.

Predeclare the same split, seeds, epoch budget and final averaging window for all
four configurations. The default comparison uses split 0, seeds 0/1/2, 200 epochs
and the final 20 epochs. Each run starts from fresh, seeded weights; the same seed
pairs initialization, batch order and (if enabled) shuffled training labels.
Frozen features are extracted once and reused across the sweep.

- Primary outcome: last-window validation balanced accuracy, candidate minus
  baseline for the same seed. Positive differences favor the candidate.
- Supporting outcomes: validation accuracy and AUC (higher is better), and
  validation cross-entropy loss (lower is better). Both class recalls are logged.
- Overfitting diagnostic: training minus validation accuracy and balanced accuracy.
  A smaller gap alone is insufficient: both scores can deteriorate together.
- Inspect all seed pairs and report their spread, including negative results.
  Compare the fixed final window rather than choosing each model's peak epoch.

The summary supplies descriptive paired effects for hypothesis evaluation, not
p-values or a statistical-significance verdict. Epochs are correlated, repeated
seeds reuse the same validation examples, and supplied splits may overlap. The
per-window standard deviation describes epoch variation, not a confidence interval.
Do not treat epoch rows or split/seed combinations as independent dataset samples.
A population-level significance test needs a separately agreed episode-aware
protocol; aggregate epoch CSVs do not establish it. Keep test data reserved until
model choices are fixed.

## Run and save results on Perlmutter

From the repository root inside an allocation, with `DATA_DIR` set:

```sh
python train_cnngru.py --data-dir "$DATA_DIR" --split-position 0 \
  --sweep --seeds 0 1 2 --epochs 200 --summary-window 20 \
  --feature-cache "${PSCRATCH:?PSCRATCH must be set}/waldo-feature-cache" \
  --output-dir "${PSCRATCH:?PSCRATCH must be set}/waldo-regularization/split0-trial1"
```

This runs 12 models: baseline plus three severities, each with three seeds. The
output directory must be new; choose a new trial name for a repeat. It contains:

- `baseline-seed0.csv`, `light-seed0.csv`, `medium-seed0.csv`, `heavy-seed0.csv`,
  and corresponding files for the other seeds: one row per completed epoch.
- `summary.csv`: one row per completed run with final-window means/stds, accuracy
  gaps, and `*_delta_vs_baseline` columns matched within the seed. Baseline deltas
  are zero. Loss and gap deltas favor a reduction when negative.

Each epoch CSV includes `epoch`, `seconds`, `train_acc`, `val_acc`, `train_loss`,
`val_loss`, balanced accuracies, AUCs and class recalls. Effective configuration,
seed, split, epoch budget, learning rate, batch size, weight decay, dropout,
feature noise, gradient clipping, image size, summary window and null-label flag
are repeated on every row; device and PyTorch version are also recorded. Record
the tested Git revision alongside your experiment notes.

Accuracy is a fraction in [0, 1]. Training and validation metrics are evaluated
at the end of each epoch in evaluation mode, with dropout/noise disabled. Both
losses are class-weighted cross-entropy using training-derived class weights;
weight-decay penalties are not added to the logged loss. Training loss is the
post-epoch full-training-set loss, not the mean of noisy minibatch update losses.

Files are flushed after every epoch/run so completed observations survive a later
training exception. Existing CSVs and sweep directories are never overwritten.
Partial runs remain partial; there is no automatic resume or checkpoint saving.
Keep all real-data experiment outputs and caches on Perlmutter.

For an individual run:

```sh
python train_cnngru.py --data-dir "$DATA_DIR" --split-position 0 --seed 0 \
  --config light --metrics-csv "${PSCRATCH:?PSCRATCH must be set}/waldo-regularization/light-trial2.csv"
```

Substitute `medium`, `heavy` or `baseline` as needed. Single runs allow explicit
`--weight-decay`, `--dropout`, `--feature-noise` and `--epochs` overrides; the CSV
records their actual values. Sweep mode fixes regularization presets and accepts
shared training controls such as `--epochs`, `--lr`, and `--batch-size`. It rejects
regularization overrides so the intended comparison cannot silently change.

For the existing memorization diagnostic, repeat the sweep with
`--shuffle-train-labels` and a different output directory. Validation labels stay
real. No command loads the test partition. Without `--metrics-csv` or `--sweep`,
a single run retains the terminal/in-memory behavior.
