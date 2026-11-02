# Group 1 — Andrew and Suhani

Read the [shared approach notes](../APPROACHES.md), [dataset notes](../DATASET.md), and [project README](../../README.md) before continuing. This file holds Group 1 plans, progress, and permitted aggregate results. Shared environment, preprocessing, interfaces, and evaluation decisions belong in the approach notes. Encoder design notes and Perlmutter smoke-check commands are in [cnn.md](cnn.md).

## Proposed scope — supplied by project owner

- CNN–GRU: a shared CNN encodes frame-map pairs, combines features with depth, and feeds a GRU action predictor.
- CNN–LSTM: replace GRU with LSTM while keeping visual inputs and depth encoding consistent.
- Current-input baseline: test whether history helps; exact inputs and implementation remain to be agreed.

These are team proposals, not confirmed course requirements. Individual ownership is open. The shared optional JEPA extension remains in the approach notes.

## Current status and next steps

Owner-reported schedule (2026-10-30): Friday — Suhani and Andrew work on CNNs and understanding the data. By Sunday (2026-11-01) — aim to have the CNN work done or finalizing. This was a target, not a completion report for the full CNN–GRU/LSTM policies.

The frozen ResNet-18 frame encoder in `cnn.py` is implemented locally (commit `023d125`): ImageNet-pretrained backbone, 512-d pooled features, shared weights for separate goal and six-frame history encoding, fully frozen BatchNorm-safe eval behavior, and optional max-pool removal. It reuses `DINOv2Preprocessor` for RGB/full-view resize/pad and ImageNet normalization; that module's DINOv2 checkpoint label does not name this CNN. Backbone choice rationale and Perlmutter test commands are in [cnn.md](cnn.md).

Owner-run Perlmutter one-case encoder verification passed (2026-11-02): the `cnn.py` smoke check completed successfully. Exact printed device, remote HEAD, and optional flags were not supplied; do not repeat the completed default check without a code/data change or concrete concern. GRU/LSTM, depth/time fusion, the current-input baseline, training, and evaluation are not implemented. Basic dataset validation has already passed as recorded in the dataset notes; do not repeat it without a concrete reason.

**Immediate next steps:** inspect prepared previews for wire/target visibility; then agree on temporal model (GRU vs LSTM), depth/time use, and goal fusion before implementing the next stage.

## Progress log

### 2026-11-02 — One-case Perlmutter encoder check passed

Owner-reported: running `cnn.py` on Perlmutter for one real case passed. This confirms preparation, pretrained load, freezing, and goal/history bookkeeping for the inspected case. Device (`cpu`/`cuda`), exact shapes printed, optional `--size` / `--remove-maxpool` usage, and remote HEAD were not reported. No training or feature saving. Visual usefulness, GPU performance measurement, and action accuracy remain unverified. Do not repeat this completed smoke check without a relevant change.

### 2026-10-31 — Frozen ResNet-18 encoder implemented

Replaced the CNN outline with a working `FrameEncoder` in `cnn.py` (`023d125`). Defaults: pretrained ResNet-18, `fc -> Identity`, keep average pooling (512-d features), freeze all backbone weights, force backbone `eval()` even when the parent module trains, encode history `(B, T, 3, H, W)` and goal `(B, 3, H, W)` with shared weights, and optionally replace the stem max-pool with identity. A CLI smoke path loads one split case through the existing preprocessor and asserts finite `(1, 6, 512)` / `(1, 512)` outputs without saving features.

ResNet-18 was chosen over ConvNeXt Tiny and MobileNetV2 as the first backbone because those alternatives were judged too lightweight for the initial capacity target; no comparative guidewire bake-off has been run. Documentation of that choice, architectural decisions, and Perlmutter instructions lives in [cnn.md](cnn.md).
