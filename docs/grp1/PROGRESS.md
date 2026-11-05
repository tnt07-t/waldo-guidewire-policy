# Group 1 — Andrew and Suhani

Read the [shared approach notes](../APPROACHES.md), [dataset notes](../DATASET.md), and [project README](../../README.md) before continuing. This file holds Group 1 plans, progress, and permitted aggregate results. Shared environment, preprocessing, interfaces, and evaluation decisions belong in the approach notes. Encoder design notes are in [cnn.md](cnn.md); 512-d GRU-readiness probes are in [CNN_DIAGNOSTICS.md](CNN_DIAGNOSTICS.md).

## Proposed scope — supplied by project owner

- CNN–GRU: a shared CNN encodes frame-map pairs, combines features with depth, and feeds a GRU action predictor.
- CNN–LSTM: replace GRU with LSTM while keeping visual inputs and depth encoding consistent.
- Current-input baseline: test whether history helps; exact inputs and implementation remain to be agreed.

These are team proposals, not confirmed course requirements. Individual ownership is open. The shared optional JEPA extension remains in the approach notes.

## Current status and next steps

Owner-reported schedule (2026-10-30): Friday — Suhani and Andrew work on CNNs and understanding the data. By Sunday (2026-11-01) — aim to have the CNN work done or finalizing. This was a target, not a completion report for the full CNN–GRU/LSTM policies.

The frozen ResNet-18 frame encoder in `cnn.py` is implemented locally (commit `023d125`): ImageNet-pretrained backbone, 512-d pooled features, shared weights for separate goal and six-frame history encoding, fully frozen BatchNorm-safe eval behavior, and optional max-pool removal. It reuses `DINOv2Preprocessor` for RGB/full-view resize/pad and ImageNet normalization; that module's DINOv2 checkpoint label does not name this CNN. Backbone choice rationale and Perlmutter test commands are in [cnn.md](cnn.md).

Owner-run Perlmutter one-case encoder verification passed (2026-11-02): the `cnn.py` smoke check completed successfully. Exact printed device, remote HEAD, and optional flags were not supplied; do not repeat the completed default check without a code/data change or concrete concern.

`mlp_cnn.py` encodes the prepared goal through the shared frozen ResNet (`encode_goal` → `[512]` / `[B, 512]`) for MLP fusion with the GRU `[128]` summary. Prep stays in `DINOv2Preprocessor.prepare_case`; history/GRU stay in `gru.py`. Teammate can add the MLP in `mlp_cnn.py`.

`mlp_cnn.GoalClassifier` is the MLP head over `concat(gru_summary [B,128], goal_features [B,512])` (640 -> 128 -> 2 logits, no / yes). `model_cnngru.CNNGRUModel` wires the full path: history through `CNNGRUEncoder`, goal through `encode_goal` with the same shared CNN, then `GoalClassifier`.

`cnn_diagnostics.py` remains available for 512-d probes. The current-input baseline, training loop, and evaluation are not implemented. Basic dataset validation has already passed as recorded in the dataset notes; do not repeat it without a concrete reason.

**Immediate next steps:** write the training and evaluation script around `CNNGRUModel` (cross-entropy on `label`, argmax for prediction), then run a one-case forward/backward check on Perlmutter.

## Progress log

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
