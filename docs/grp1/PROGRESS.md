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

`cnn_diagnostics.py` adds separate 512-d probes—history **occlusion sensitivity** (not wire detection), goal **marker-region perturbation** vs translated controls (edits before prep), and **temporal feature variation** vs aligned depth (magnitude association only)—so a later GRU’s inputs can be judged without changing `cnn.py`. Diagnostics were corrected after the first owner GPU run (uint8 red overflow, control design, wording, temporal correlation). Ten invented-input regression tests passed locally (tool-verified). A re-run on Perlmutter with overlays is needed before treating marker/occlusion aggregates as current. GRU/LSTM, depth/time fusion, the current-input baseline, training, and evaluation are not implemented. Basic dataset validation has already passed as recorded in the dataset notes; do not repeat it without a concrete reason.

**2026-11-03 update:** the goal-centered wire preprocessor and its frozen-feature probe are built and partly evaluated; neither the 256 nor the 512 crop beat the 0.547 full-frame baseline on split 0 validation.
See [the 2026-11-03 log entry](#2026-11-03---wire-preserving-preprocessing-and-frozen-feature-probe-owner-run-ladder-stopped-early) for results, limits and what was not run.

**Immediate next steps:** re-run fixed `cnn_diagnostics.py` on Perlmutter with `--output-dir`; inspect mask/heatmap overlays on the machine; then agree on temporal model (GRU vs LSTM), depth/time use, and goal fusion before implementing the next stage.

## Progress log

### 2026-11-03 - Wire-preserving preprocessing and frozen-feature probe (owner-run, ladder stopped early)

**What was built** (tickets in [tickets/](tickets/README.md), branch `andrew/grp1-preprocessing-tickets`, `d063196` to `804c98f`):
`WirePreprocessor` in `preprocessing.py` crops a square around `goal_xy` at native resolution from the unmarked goal and the six history frames.
It supports `raw` channels or `enhanced` channels (raw, black top-hat, frame difference), an optional full-view context, a seeded random-center control and training-only channel standardization.
`cnn.py`, `cnn_diagnostics.py` and Sabrina's `evaluation.py` (imported unchanged, then extended) share one set of preprocessor flags.
`evaluation.py` gained opt-in controls, and `select_preprocessing.py` chooses settings with episode-grouped inner CV on the training partition only.
Design and commands are in [CNN.md](CNN.md#wire-preprocessing) and [EVALUATION.md](../EVALUATION.md#preprocessor-options).

**Unit tests (owner-run):** 46 tests passed at `31cbfa9` (T04-T08).
The suite including the T09-T16 tests was not reported; the real runs below exercised that code without errors.

**Reader, one real case (owner-run):** the new `goal_raw` decoded at 1065 x 1050 RGB and `goal_xy` had two values.
The notebook check that `goal_raw` has no marker was not reported.

**Static probe, split 0 validation** (80 cases; owner-run on an A100; probe `resnet18:latest_goal`; frozen ImageNet ResNet-18):

| Input preparation | Validation BA | Train BA |
| --- | ---: | ---: |
| DINOv2 full frame, 518 (baseline) | 0.547 | 1.000 |
| Wire raw crop, 256 | 0.515 | 1.000 |
| Wire raw crop, 512 (selected below) | 0.506 | 0.923 |
| Metadata only (no images) | 0.547 | 0.580 |
| Majority class | 0.500 | 0.500 |

The baseline reproduces Sabrina's 0.547 exactly, so the default path is unchanged after the refactors.
Every image row of every run (latest, goal only, latest + goal, with metadata, mean history) stayed between 0.45 and 0.57.
One split's 95% interval is about +/-0.11, so none of these differ from chance or from metadata only.

**Selection inside training** (`select_preprocessing.py`, 5-fold episode-grouped inner CV, `latest_goal`, validation not used):

| Configuration | Inner BA mean | Inner BA std |
| --- | ---: | ---: |
| raw, crop 192 | 0.459 | 0.058 |
| raw, crop 256 | 0.485 | 0.088 |
| raw, crop 384 | 0.503 | 0.058 |
| **raw, crop 512 (selected)** | **0.597** | **0.052** |
| raw, crop 640 | 0.547 | 0.039 |
| raw, crop 768 | 0.550 | 0.078 |
| enhanced, crop 512, top-hat 7, median difference, standardized | 0.558 | 0.025 |

Enhanced lost at its one tried setting, so difference mode and top-hat size were never searched.
The selected 0.597 did not hold on validation (0.506), which fits a winner's-curse effect from choosing the best of seven noisy scores.

**Latency** (wire raw 256, `--benchmark`, A100, goal + newest frame, batch one, repeated in-memory case): preparation 9.9 ms median (10.2 p95), encoding 5.9 ms (6.0), total 15.8 ms (16.1).
Preparation covers all seven crops, since the wire path prepares the whole case.
Timing for the 512 crop was not reported.

**Reading:** frozen ImageNet ResNet-18 features with a linear probe show no detectable ADVANCE/RETRACT signal on split 0, whether the input is the downscaled full frame or a native-resolution crop of 256 or 512.
The wire being too small after downscaling is therefore not the whole explanation.
All probes overfit (train BA 0.92-1.00).
This is a development result on one split, not a final or test-set result.

**Not run (stopped by the owner on 2026-11-03):**

- the T13 notebook check (top-hat size and crop candidates were not chosen visually);
- the T15 controls (shuffled labels, random center, channel ablation) and the five-split runs;
- the wire-mode diagnostics overlays (`cnn_diagnostics.py` ran for DINOv2; its output was not reported);
- enhanced channels on validation (by design: they lost the selection).

**Code version:** the branch head pulled on Perlmutter for these runs; the exact commit was not reported (the latest pushed was `804c98f`).

**Proposed next steps (not decided):** the ticket ladder's fallback is to train part of the network (conv1 or layer4) or a small CNN on the crops.
Both raw RETRACT records shared on 2026-11-03 show the wire advancing before the decision, so a temporal model over all six frames plus depths may matter more than a better single frame.
The preprocessor stays opt-in; `dinov2` remains the default.

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
