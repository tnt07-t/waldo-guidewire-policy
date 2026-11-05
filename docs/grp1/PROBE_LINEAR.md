# Frozen ResNet linear probes

Branch: `suhani-experiments`. Script: `probe_cnn_features.py`. Baseline model: `GoalOnlyLinearModel` / `infer_goal_linear.py`.

## Objective

Determine whether frozen ResNet-18 features contain information useful for ADVANCE vs RETRACT, and whether removing the stem maxpool improves linear-probe performance.

## Setup

- Split position **0** for every run; seeds **0** and **1**.
- Frozen ResNet encoder; train a linear classifier (class-weighted CE, AdamW, select best epoch by validation balanced accuracy).
- Compare standard maxpool vs `--remove-maxpool`.
- GPU on Perlmutter. Train 267 (97 RETRACT / 170 ADVANCE); validation 80 (34 / 46).
- **Test split not evaluated.**
- Always-ADVANCE validation baseline: accuracy 0.575, balanced accuracy 0.500, AUC 0.500.
- Balanced accuracy = mean of RETRACT recall and ADVANCE recall.

### Feature designs

| Design | Definition | Dim |
| --- | --- | ---: |
| `latest_goal` | concat(latest history frame, goal) | 1024 |
| `mean_goal` | concat(mean history, goal) | 1024 |
| `goal_only` | goal only | 512 |

## Results (all runs)

| maxpool | seed | design | best_epoch | train_bal | val_bal | val_acc | val_auc | val_loss |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| kept | 0 | latest_goal | 95 | 0.726 | 0.561 | 0.575 | 0.488 | 0.7472 |
| kept | 0 | mean_goal | 100 | 0.701 | 0.536 | 0.537 | 0.452 | 0.7776 |
| kept | 0 | goal_only | 77 | 0.648 | 0.522 | 0.525 | 0.423 | 0.7486 |
| kept | 1 | latest_goal | 74 | 0.704 | 0.561 | 0.575 | 0.493 | 0.7500 |
| kept | 1 | mean_goal | 75 | 0.666 | 0.513 | 0.550 | 0.411 | 0.8006 |
| kept | 1 | goal_only | 90 | 0.675 | 0.558 | 0.562 | 0.426 | 0.7482 |
| removed | 0 | latest_goal | 43 | 0.596 | 0.596 | 0.650 | 0.558 | 0.6954 |
| removed | 0 | mean_goal | 55 | 0.509 | 0.522 | 0.450 | 0.396 | 0.7084 |
| removed | 0 | goal_only | 63 | 0.609 | 0.583 | 0.600 | 0.512 | 0.7042 |
| removed | 1 | latest_goal | 75 | 0.656 | 0.542 | 0.575 | 0.453 | 0.7213 |
| removed | 1 | mean_goal | 57 | 0.559 | 0.515 | 0.588 | 0.420 | 0.7345 |
| removed | 1 | goal_only | 23 | 0.512 | 0.664 | 0.675 | 0.702 | 0.6777 |

### Validation confusion (rows true, cols pred; [RETRACT, ADVANCE])

**Maxpool kept, seed 0:** latest `[[16,18],[16,30]]`; mean `[[18,16],[21,25]]`; goal `[[17,17],[21,25]]`

**Maxpool kept, seed 1:** latest `[[16,18],[16,30]]`; mean `[[9,25],[11,35]]`; goal `[[18,16],[19,27]]`

**Maxpool removed, seed 0:** latest `[[8,26],[2,44]]`; mean `[[34,0],[44,2]]`; goal `[[16,18],[14,32]]`

**Maxpool removed, seed 1:** latest `[[11,23],[11,35]]`; mean `[[1,33],[0,46]]`; goal `[[20,14],[12,34]]`

### Two-seed averages (descriptive; one validation split)

| maxpool | design | mean val_bal | mean val_acc | mean val_auc |
| --- | --- | ---: | ---: | ---: |
| kept | latest_goal | 0.5610 | 0.5750 | 0.4905 |
| kept | mean_goal | 0.5245 | 0.5435 | 0.4315 |
| kept | goal_only | 0.5400 | 0.5435 | 0.4245 |
| removed | latest_goal | 0.5690 | 0.6125 | 0.5055 |
| removed | mean_goal | 0.5185 | 0.5190 | 0.4080 |
| **removed** | **goal_only** | **0.6235** | **0.6375** | **0.6070** |

## Interpretation

1. **Best individual run:** `goal_only`, maxpool removed, seed 1 — val bal 0.664, acc 0.675, AUC 0.702 (RETRACT 20/34, ADVANCE 34/46). Do **not** report the best seed alone as the overall result.
2. **Best mean config:** `goal_only` + remove maxpool (mean val bal 0.6235, AUC 0.6070). Seed variance is large (seed 0: 0.583 / 0.512; seed 1: 0.664 / 0.702).
3. **Maxpool removal did not consistently help `latest_goal`** (seed 0 up, seed 1 down; mean +0.008 bal / +0.015 AUC).
4. **`mean_goal` was poor** (mean AUC &lt; 0.5); without maxpool it collapsed to nearly all-RETRACT or all-ADVANCE by seed — inspect before treating as an architecture conclusion.
5. **History concat did not beat `goal_only`** on average without maxpool. That does not prove temporal info is useless; probes are not a GRU.
6. Scope limits: frozen + linear only; two seeds / one split; keep the test set reserved.

## Implemented baseline (best config)

Code on `suhani-experiments`:

- `model_goal_linear.py` — `GoalOnlyLinearModel` (frozen ResNet, default `remove_maxpool=True`, linear on goal).
- `probe_cnn_features.py --preset best` — trains only `goal_only` with `--remove-maxpool`, writes `best_model.pt`.
- `infer_goal_linear.py` — loads `best_model.pt` and scores a batch.

## Running instructions (Perlmutter)

```sh
export TORCH_HOME="${PSCRATCH}/model-cache/torch"
export DATA_DIR="/global/cfs/cdirs/m4999/studentdata/shorthorizon2"
cd ~/CS5470_FinalProject   # or your clone of suhani-experiments

# Full 3-design comparison (maxpool kept)
srun -n 1 --gpus=1 python probe_cnn_features.py \
  --data-dir "$DATA_DIR" --split-position 0 --seed 0 \
  --run-name "probe-cnn-split0-seed0-$(date +%Y%m%d-%H%M%S)"

# Full comparison with stem maxpool removed
srun -n 1 --gpus=1 python probe_cnn_features.py \
  --data-dir "$DATA_DIR" --split-position 0 --seed 0 --remove-maxpool \
  --run-name "probe-cnn-nomaxpool-split0-seed0-$(date +%Y%m%d-%H%M%S)"

# Best recipe only (goal_only + remove-maxpool) → best_model.pt
srun -n 1 --gpus=1 python probe_cnn_features.py \
  --data-dir "$DATA_DIR" --split-position 0 --seed 0 --preset best \
  --run-name "probe-best-goalonly-nomaxpool-split0-seed0-$(date +%Y%m%d-%H%M%S)"

# Inference smoke (validation batch; not for tuning)
srun -n 1 --gpus=1 python infer_goal_linear.py \
  --data-dir "$DATA_DIR" --split-position 0 --partition validation \
  --checkpoint docs/grp1/results/<run-name>/best_model.pt
```

Repeat with `--seed 1` for a second seed. Do not pass `--partition test` for model selection.

## Result folders

Under `docs/grp1/results/`:

| Config | Folder |
| --- | --- |
| maxpool kept, seed 0 | `probe-cnn-split0-20261105-125328` |
| maxpool kept, seed 1 | `probe-cnn-seed1-split0-20261105-131811` |
| maxpool removed, seed 0 | `probe-cnn-nomaxpool-split0-20261105-131333` |
| maxpool removed, seed 1 | `probe-cnn-nomaxpool-seed1-split0-20261105-132018` |

Each run has `summary.json` and per-design `metrics.csv` / `best_linear.pt`. Encoding times are not inference latency benchmarks.

## Implementation checks before more experiments

- Encoder frozen + stays in eval; feature extraction deterministic across seeds for fixed inputs/config.
- No random augment during extraction; any fitted norm uses train stats only.
- Label / AUC class order; optimizer, LR, class weights, best-epoch criterion; train/val metrics from the same checkpoint.
- Investigate no-maxpool `goal_only` seed 1 (train bal 0.512 vs val bal 0.664) — not proof of a bug, but check.
- Cached features (if added later) must key on maxpool + prep size.
- Procedure-level train/val separation; test reserved.

Next priorities: [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md). Aggregate metrics may live here; real images/features stay on Perlmutter.
