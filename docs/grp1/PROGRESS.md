# Group 1 — Andrew and Suhani

Read the [shared approach notes](../APPROACHES.md), [dataset notes](../DATASET.md), and [project README](../../README.md) before continuing. This file holds Group 1 plans, progress, and permitted aggregate results. Shared environment, preprocessing, interfaces, and evaluation decisions belong in the approach notes. Encoder design notes are in [cnn.md](cnn.md); 512-d GRU-readiness probes are in [CNN_DIAGNOSTICS.md](CNN_DIAGNOSTICS.md).

## Proposed scope — supplied by project owner

- CNN–GRU: a shared CNN encodes frame-map pairs, combines features with depth, and feeds a GRU action predictor.
- CNN–LSTM: replace GRU with LSTM while keeping visual inputs and depth encoding consistent.
- Current-input baseline: test whether history helps; exact inputs and implementation remain to be agreed.

These are team proposals, not confirmed course requirements. Individual ownership is open. The shared optional JEPA extension remains in the approach notes.

## Current status and next steps

Frozen ResNet-18 + GRU + goal MLP policy trains with `train_cnngru.py` (CNN frozen; LayerNorm on 512-d features; metadata standardization; class-weighted CE). Runs write under `docs/grp1/results/<run-name>/` via `--run-name` (not `--output-dir`). `--patience 0` disables early stopping. Local synthetic tests passed. Real images/features stay on Perlmutter. Inference and a two-epoch training smoke check passed on Perlmutter. The test partition has not been used.

**Completed owner-run reference (LayerNorm, split 0, seed 0, lr 1e-3):** `docs/grp1/results/layernorm-reference-split0-20261104-231846`. 267 train / 80 val. Stopped epoch 56; best val bal-acc at epoch 26: train bal **0.809**, val bal **0.608**, val acc **0.637**, val AUC **0.499**, val loss **0.9688**. Confusion (rows true, cols pred; RETRACT/ADVANCE): `[[14, 20], [9, 37]]`. Train nearly memorizes later (e.g. ~99% train acc around epoch 53) while validation stays weak and val loss rises — clear overfitting. More capacity / long memorization runs are lower priority. LayerNorm benefit still needs matched comparisons already logged under H0/H1/H1b.

**Completed H2 learning-rate comparison (owner-run):** same split/seed and settings as the reference except **lr 3e-4**, batch 32, epochs 200, patience 30. Folder `docs/grp1/results/h2-lr3e4-split0-20261104-232604`. Early stop after 30 epochs without bal-acc gain. Best epoch **28**: train bal **0.788**, val bal **0.612**, val acc **0.637**, val AUC **0.541**, val loss **0.8588**. Confusion: RETRACT `[15, 19]`, ADVANCE `[10, 36]`. Versus reference: essentially the same peak val bal-acc (~0.61); slightly better best-epoch AUC/loss; overfitting pattern unchanged.

Basic dataset validation has already passed; do not repeat without a concrete reason. Current-input baseline and multi-split reporting remain open.

**Immediate next steps:** treat H2 as negative/null for LR-only gains on this split; prioritize regularization / less overfitting or a current-input baseline before more capacity. Do not use the test partition. Log any further matched ablations here.

## Progress log

### 2026-11-04 — LayerNorm reference + H2 LR 3e-4 (owner-run Perlmutter)

**Plumbing:** all local synthetic tests passed; Perlmutter inference and two-epoch training smoke passed. Code uses LayerNorm on features; `train_cnngru.py` uses `--run-name` and writes `docs/grp1/results/<run-name>/`; `--patience 0` disables early stopping. Test partition unused.

**Reference** (`layernorm-reference-split0-20261104-231846`): split 0, seed 0, lr `1e-3`. 267 train / 80 val. Stopped epoch 56; best val bal-acc epoch 26 — train bal 0.809, val bal 0.608, val acc 0.637, val AUC 0.499, val loss 0.9688; confusion `[[14, 20], [9, 37]]`. Train can reach ~99% acc while validation stays near chance AUC and val loss rises → overfitting, not underfitting.

**H2** (`h2-lr3e4-split0-20261104-232604`): lr `3e-4` only change vs reference protocol (batch 32, patience 30, 200 epochs max). Command (Perlmutter):

```sh
python train_cnngru.py \
  --data-dir "$DATA_DIR" \
  --split-position 0 \
  --seed 0 \
  --epochs 200 \
  --patience 30 \
  --batch-size 32 \
  --lr 0.0003 \
  --run-name "h2-lr3e4-split0-$(date +%Y%m%d-%H%M%S)"
```

Early stop at epoch 58. Best epoch 28 — train bal 0.788, val bal 0.612, val acc 0.637, val AUC 0.541, val loss 0.8588; confusion RETRACT `[15, 19]`, ADVANCE `[10, 36]`. Trainable params 330,626; class weights RETRACT/ADVANCE `[1.376, 0.785]`.

**Reading:** lowering LR alone does not fix generalization on split 0 (val bal-acc ~0.61 either way). Prefer next work on overfitting / signal (dropout, weight decay, baseline comparison, other splits) over capacity or another long memorization run. LayerNorm still lacks a new matched ablate in this pair (both runs already use LayerNorm).

### 2026-11-05 - LayerNorm adopted as the only feature normalization (local)

Following the split-0 results below, `CNNGRUModel` now always applies one shared `nn.LayerNorm(512)` to every history frame and the goal in `forward_features`, before the GRU and the MLP.
The `feature_norm` option, the `none` and `zscore` paths (`FeatureStandardizer`, `set_feature_stats`, `feature_norm_from_state_dict`), `--feature-norm`, and the raw-feature-scale log were removed.
Default run names are back to `cnngru-split<S>-seed<N>-<timestamp>`.
Checkpoints from the earlier `layer` runs keep the same `feature_norm.weight` / `feature_norm.bias` keys and still load; `none` and `zscore` checkpoints no longer do.
Reason: LayerNorm fixes the H0 underfitting and was the more stable of the two normalizations; its validation gain over H0 (0.586 vs 0.548 mean) is still within noise, so further work should target overfitting.

Not verified: the simplified `tests/test_train_cnngru.py` has not been run; no training run with this code exists yet.
No staging/commit/push in this note.

### 2026-11-05 - H1 / H1b feature normalization option and results

Hypothesis H1: the raw 512-d frozen-ResNet features are post-ReLU (non-negative, uncentered, uneven per-dimension scale), which saturates the GRU gates and lets the 512 goal values dominate the 128 GRU values in the MLP input.
Normalizing them should drop train loss clearly below 0.6 and make validation balanced accuracy less jumpy.

`CNNGRUModel(feature_norm=...)` adds one normalization module, shared by every history frame and the goal, applied in `forward_features` before the GRU and the MLP:
- `none` (H0): identity; state_dict and initialization are unchanged, so earlier checkpoints still load.
- `layer` (H1): `nn.LayerNorm(512)` per sample across the 512 dimensions (1,024 extra trainable parameters).
- `zscore` (H1b): per-dimension training-set mean/std pooled over history frames and goals, fitted with `set_feature_stats` and stored as buffers, like the metadata statistics.

Depth/offset metadata is not part of it; it is already standardized and is concatenated after the 512 features are normalized.
`train_cnngru.py --feature-norm {none,layer,zscore}` (default `none`); the default run name is now `cnngru-<norm>-split<S>-seed<N>-<timestamp>`.
Every run also logs the raw training-feature scale (mean, mean L2, zero fraction, per-dimension std min/median/max and their ratio) and stores it as `raw_feature_scale` in `summary.json`.
`infer_cnngru.py` detects the normalization from the checkpoint keys.
Optimizer, lr, weight decay and dropout are unchanged so only the normalization varies between H0, H1 and H1b.

Comparison protocol: same split, seeds 0-2 for each of H0/H1/H1b; compare final and best train loss, best validation balanced accuracy (mean +- std over seeds), and the epoch-to-epoch std of validation balanced accuracy over the last 20 epochs.

Owner-run results on Perlmutter (split 0, seeds 0-2, patience 20; 267 train / 80 validation cases):

| Mode | Best val bal acc (seeds 0/1/2) | Mean | Best-epoch val AUC | Lowest train loss |
| --- | --- | --- | --- | --- |
| H0 `none` | 0.569 / 0.558 / 0.518 | 0.548 | 0.426 / 0.538 / 0.411 | about 0.65 |
| H1 `layer` | 0.608 / 0.579 / 0.571 | 0.586 | 0.499 / 0.492 / 0.460 | 0.21 / 0.19 / 0.02 |
| H1b `zscore` | 0.574 / 0.600 / 0.565 | 0.580 | 0.591 / 0.506 / 0.520 | about 0.0000 |

Raw train features: mean 0.664, mean L2 18.93, no zeros, per-dimension std 0.042 to 0.694 (ratio 16.5), no constant dimensions.

Reading: the optimization half of H1 is confirmed.
Without normalization the head underfits (train loss stays near 0.65-0.69) and often collapses to predicting one class for the whole validation set.
Both normalizations let it fit the training set (LayerNorm gradually, z-score memorizes it within about 15 epochs).
The generalization half is not confirmed.
The +0.03-0.04 mean validation balanced-accuracy gain is about 3 of 80 cases, within seed spread and the roughly 0.055 binomial standard error, and is a max over many noisy epochs.
Best-epoch validation AUC stays near 0.5, and validation loss rises from the first few epochs, so the bottleneck is now overfitting / weak signal, not optimization.
Validation balanced accuracy still swings by 0.1 or more between consecutive epochs.

Follow-up: early stopping is now off by default (`--patience 0` runs all `--epochs`; `--patience N` restores it), still keeping the best-validation epoch.

Not verified: no unit-test output has been reported for `tests/test_train_cnngru.py`, including the new `test_patience_zero_runs_all_epochs`; no full-200-epoch runs exist yet.
No staging/commit/push in this note.

### 2026-11-05 - CNN-GRU training script `train_cnngru.py` (local)

Added `train_cnngru.py`, which trains the GRU + MLP head on one split with the frozen ResNet-18.
Every train/validation image is encoded once at startup and the 512-d features are kept in memory only; nothing is written to disk except optional metrics and the checkpoint.
The test partition is never loaded.

Input fixes:
- `CNNGRUEncoder` now standardizes `[depth_mm, offset_s]` per channel with training-set mean/std (`set_metadata_stats`), stored as buffers so checkpoints and `infer_cnngru.py` reuse them.
- `GoalClassifier(dropout=0.3)` applies dropout to the fused 640-d input and the hidden layer.
- `CNNGRUModel.forward_features` runs the head on cached features; `forward` gives the same output from images.

Training: class-weighted cross-entropy (`n / (2 * count_c)`), AdamW (lr `1e-3`, weight decay `1e-2`), batch 16, gradient clipping at 1.0, up to 200 epochs with early stopping (patience 20) on validation balanced accuracy, ties broken by lower validation loss.
Each epoch prints train and validation loss, accuracy and balanced accuracy (both in eval mode), plus validation ROC AUC and per-class recall; the best epoch is marked `*best`.
Every run writes to its own folder `docs/grp1/results/<run-name>/` (default name `cnngru-split<S>-seed<N>-<timestamp>`, or `--run-name`): `train.log`, `metrics.csv`, `summary.json`, `curves.png` and `best_model.pt` (a `CNNGRUModel` state_dict).
Checkpoints are git-ignored; the other files hold aggregate metrics only, and `summary.json` omits the personal `data_dir` path.

Not verified: invented-input `tests/test_train_cnngru.py` has not been run; no Perlmutter training run or results exist yet.
No staging/commit/push in this note.

### 2026-11-03 - Full CNN-GRU model in `model_cnngru.py` (local)

Added `model_cnngru.py` with `CNNGRUModel(pretrained=True)`, which composes the existing pieces into one yes / no model.
`forward(history, depth, offsets, goal)` runs `CNNGRUEncoder` for the `[B, 128]` history summary and `encode_goal(self.encoder.cnn, goal)` for the `[B, 512]` goal features, so history and goal share one frozen ResNet-18.
`GoalClassifier` then maps the concatenated 640-d vector to `[B, 2]` logits.
The module docstring records the expected batch layout (output of `prepare_case` plus collate) and the intended training step: `cross_entropy(logits, label)`, prediction by `argmax`.

Supporting changes: added `GoalClassifier` (640 -> 128 -> ReLU -> 2) to `mlp_cnn.py` and removed its MLP TODO; `gru.py` only had a trailing-newline change.

Not verified: no test or forward pass has been run on this model yet, and no training or accuracy results exist.
No staging/commit/push in this note.

### 2026-11-03 — Goal→ResNet helper in `mlp_cnn.py` (local)

`mlp_cnn.encode_goal` runs the prepared goal through the shared `FrameEncoder` (ResNet-18). Dropped redundant `prepare_goal` (prep already comes from `prepare_case`). Invented-input `tests/test_mlp_cnn.py` passed locally. MLP fusion still pending. No staging/commit/push in this note.

### 2026-11-03 — ResNet→GRU in `gru.py` (local)

`gru.CNNGRUEncoder` runs the images through the shared `FrameEncoder` (ResNet-18).

### 2026-11-02 — Diagnostics correctness fixes (local)

Updated `cnn_diagnostics.py` / tests / docs without changing `cnn.py` encoder behavior. Fixes: int16 red-mask arithmetic on pre-normalization RGB; empty-mask `unavailable`; multi translated same-shape controls with shared pre-prep replacement; rename to occlusion sensitivity with padding exclusion and original-frame heatmaps; temporal report of signed and absolute depth steps with abs-only correlation and explicit non-claims. Tool-verified: ten invented-input tests OK. Real-case GPU re-run pending. No staging/commit/push.

### 2026-11-02 — One-case GPU diagnostics passed (owner-run; pre-fix code)

Owner-run on Perlmutter with `pytorch/2.13.0`. Invented-input `test_cnn_diagnostics.py`: five tests OK on the login node and again under `srun` on a shared-interactive GPU allocation. An initial diagnostics attempt failed because the scratch dataset copy lacked `cases.jsonl`; after setting `DATA_DIR` to the shared studentdata `shorthorizon2` tree (files present), the real-case run completed on GPU. Public ResNet-18 weights downloaded to the scratch Torch Hub cache. No features or images were saved.

**One training case** (`--split-position 0 --partition train --case-index 0`), device `cuda`, shapes `(6, 512)` / `(512,)`:

- Wire occlusion: score_range `0.246`, score_std `0.078`; per-frame max/median about `1.62`–`1.76`; max L2 about `3.02`–`3.27`. Argmax center often `(0.357, 0.643)` (frame 2 at `(0.500, 0.214)`). Structured sensitivity, but argmax≠wire until preview overlay confirms it.
- Goal marker: `12465` marker pixels; marker L2 `3.965` vs control `1.344`; marker/control `2.95`. Goal-vs-history L2 mean `2.897`, min `2.437`.
- Motion (feature math only): mean step L2 `2.474`; first–last L2 `2.931`; first–last cosine distance `0.012`; wire_score first–last delta `-0.016`; feature–depth step corr `-0.226`.

**Reading (pre-fix code only):** goal-marker and non-flat history features looked usable for a GRU trial, with caveats on argmax≠verified wire and weak depth association. Not action-accuracy evidence. Numbers predate the uint8/control/temporal fixes above—re-run before citing them.

### 2026-11-02 — 512-d feature diagnostics added (local)

Implemented `cnn_diagnostics.py` and `tests/test_cnn_diagnostics.py` outside `cnn.py`. Probes score only pooled 512-d outputs: grid occlusion on history frames, red-marker vs equal-area control occlusion on the goal, and direction-style aggregates computed from the history feature sequence / wire scores (optional depth-step correlation). Usage notes are in [CNN_DIAGNOSTICS.md](CNN_DIAGNOSTICS.md).

**Verification:** five invented-input unit tests passed locally (tool-verified; Python 3.9, torch 2.2.2) and later owner-run on Perlmutter. No training or feature saving.

### 2026-11-02 — One-case Perlmutter encoder check passed

Owner-reported: running `cnn.py` on Perlmutter for one real case passed. This confirms preparation, pretrained load, freezing, and goal/history bookkeeping for the inspected case. Device (`cpu`/`cuda`), exact shapes printed, optional `--size` / `--remove-maxpool` usage, and remote HEAD were not reported. No training or feature saving. Visual usefulness, GPU performance measurement, and action accuracy remain unverified. Do not repeat this completed smoke check without a relevant change.

### 2026-10-31 — Frozen ResNet-18 encoder implemented

Replaced the CNN outline with a working `FrameEncoder` in `cnn.py` (`023d125`). Defaults: pretrained ResNet-18, `fc -> Identity`, keep average pooling (512-d features), freeze all backbone weights, force backbone `eval()` even when the parent module trains, encode history `(B, T, 3, H, W)` and goal `(B, 3, H, W)` with shared weights, and optionally replace the stem max-pool with identity. A CLI smoke path loads one split case through the existing preprocessor and asserts finite `(1, 6, 512)` / `(1, 512)` outputs without saving features.

ResNet-18 was chosen over ConvNeXt Tiny and MobileNetV2 as the first backbone because those alternatives were judged too lightweight for the initial capacity target; no comparative guidewire bake-off has been run. Documentation of that choice, architectural decisions, and Perlmutter instructions lives in [cnn.md](cnn.md).
