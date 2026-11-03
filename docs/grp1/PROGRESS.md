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

`cnn_diagnostics.py` adds separate 512-d probes—history **occlusion sensitivity** (not wire detection), goal **marker-region perturbation** vs translated controls (edits before prep), and **temporal feature variation** vs aligned depth (magnitude association only)—so a later GRU’s inputs can be judged without changing `cnn.py`. Diagnostics were corrected after the first owner GPU run (uint8 red overflow, control design, wording, temporal correlation). Ten invented-input regression tests passed locally (tool-verified). A re-run on Perlmutter with overlays is needed before treating marker/occlusion aggregates as current. GRU/LSTM, depth/time fusion, the current-input baseline, and temporal-model training/evaluation are not implemented in the Group 1 path. Basic dataset validation has already passed as recorded in the dataset notes; do not repeat it without a concrete reason.

**Immediate next steps:** re-run fixed `cnn_diagnostics.py` on Perlmutter with `--output-dir`; inspect mask/heatmap overlays on the machine; then agree on temporal model (GRU vs LSTM), depth/time use, and goal fusion before implementing the next stage.

Upstream commit `023d125` implements the frozen ResNet-18 encoder in `cnn.py`: 512 features/image, separate goal/history calls, retained default maxpool, frozen backbone and evaluation-mode BatchNorm. The first shared GPU evaluation has completed; aggregate scores and timing are recorded once in EVALUATION.md. Basic dataset validation has already passed; do not repeat it without a concrete reason.

**Evaluation clarification:** the first shared frozen-feature evaluation has since completed; aggregate results are recorded in [EVALUATION.md](../EVALUATION.md). It trains a separate scikit-learn logistic classifier on detached ResNet features. `cnn.py` is unchanged, no trainable classifier was added inside ResNet, and the CNN weights/BatchNorm statistics stayed fixed. Sabrina confirmed that the CNN should remain frozen for the coming experiments. High probe training accuracy does not mean the CNN was fine-tuned.

Sabrina authorized the shared before-GRU evaluation milestone on 2026-11-01. [EVALUATION.md](../EVALUATION.md) describes the controlled ResNet/DINOv2 comparisons and diagnostic ablations. The new probes do not implement Group 1 GRU/LSTM models or change their ownership; remaining team contracts still need agreement.

## Progress log

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
