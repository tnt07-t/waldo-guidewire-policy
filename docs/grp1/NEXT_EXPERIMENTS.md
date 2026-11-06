# Group 1 — two stronger regularization configurations

The earlier train/validation gap, near-perfect fitting of shuffled training labels, and rising validation loss support an overfitting diagnosis. They do not establish that regularization alone can recover useful signal. These are candidate configurations to test, not validated fixes.

## Presets

Both configurations use the same frozen ResNet, shared normalization/projection, 64-unit GRU and MLP. They have identical parameter counts and evaluation behavior for identical weights. Only the training noise differs between the two presets.

| Setting | Moderate | Strict |
| --- | ---: | ---: |
| CLI preset | `--config moderate` | `--config strict` |
| AdamW weight decay | 1.0 | 1.0 |
| Gaussian feature-noise std | 1.0 | 2.0 |
| MLP dropout | 0.3 | 0.3 |
| Epoch budget | 50 | 50 |
| Learning rate | 0.001 | 0.001 |
| Batch size | 16 | 16 |
| Gradient clipping | 1.0 | 1.0 |
| Terminal summary window | Last 20 epochs | Last 20 epochs |

Noise is added independently to normalized history and goal features before the shared projection, during training only. A standard deviation of 1.0 is approximately the scale of a LayerNorm-normalized feature vector; 2.0 is deliberately aggressive and may erase useful information. It does not perturb depth/time metadata.

Moderate uses the previously tested noise strength that reduced null fitting most. Strict tests whether stronger noise meaningfully limits memorization further. Both use a predeclared 50-epoch budget to limit optimization time. Fifty epochs is a hypothesis, not a demonstrated optimum, and does not guarantee an absence of memorization.

Default `--config baseline` preserves noise 0 and 200 epochs. Explicit `--epochs`, `--weight-decay`, and `--feature-noise` override the selected preset, including explicit zero values. The terminal prints the effective settings. Keep the preset defaults for the first comparison.

## First matched comparison

Run on Perlmutter from the repository root, inside an allocation with `DATA_DIR` set:

```sh
python train_cnngru.py --data-dir "$DATA_DIR" --split-position 0 --seed 0 --config baseline --epochs 50
python train_cnngru.py --data-dir "$DATA_DIR" --split-position 0 --seed 0 --config moderate
python train_cnngru.py --data-dir "$DATA_DIR" --split-position 0 --seed 0 --config strict
```

The baseline must also use 50 epochs so that it isolates the noise effect. Compare the same epochs 31–50 for all three, rather than comparing these scores directly with the former 181–200 window. The moderate-versus-strict comparison changes only feature-noise strength. To isolate epoch budget separately, hold noise fixed and compare 50 versus 200 epochs in a later bounded experiment.

Repeat each command with `--shuffle-train-labels` for its matched null control. If the first comparison is promising, repeat the same three configurations and controls on splits 0–2 and seeds 0–2. This is a fixed comparison, not a new search over strengths. An optional `--feature-cache "$PSCRATCH/waldo-feature-cache"` reuses frozen features; it must stay outside the repository on Perlmutter.

Training prints metrics only. No result folders, CSVs, plots, logs, JSON summaries or model checkpoints are created. No test partition is loaded.

## What would count as progress?

Read the final-window validation AUC, balanced accuracy, both class recalls, train/validation balanced-accuracy gap and null training balanced accuracy. The gap is printed explicitly. Null runs also label their training fit explicitly.

- Lower null training balanced accuracy demonstrates reduced fitting of arbitrary labels. A training-side screening target is null BA around 0.55–0.70 while real training BA remains around 0.65–0.90. These are exploratory targets, not universal thresholds or proof of success.
- A smaller train/validation gap is useful only if validation performance holds or improves. Both scores collapsing to 0.5 is underfitting, not a successful solution.
- A candidate should improve validation AUC versus both its matching real-label baseline and its own null on most split/seed pairs. Check both class recalls so an ADVANCE-heavy predictor does not look successful through raw accuracy alone.
- If null fitting remains high under both presets, they have not sufficiently limited memorization. If null fit falls but real validation remains near chance, more regularization alone has not recovered a usable signal. If strict damages both training fit and validation, prefer moderate or return to baseline.

Seed repeats share validation examples, and supplied splits can overlap. They are not independent datasets. Do not select an isolated peak epoch or treat a small gain on one split as confirmation. Keep the test partition reserved until choices are fixed.

If neither configuration improves transfer, prioritize spatial feature retention, goal/current-frame relationships and private review of labels/alignment before increasing model capacity or launching a wider regularization search.
