# CNN 512-d feature diagnostics

`cnn_diagnostics.py` probes whether the **pooled ResNet-18 512-d vectors** a later GRU would receive change under controlled edits. It is separate from `cnn.py`: no localization heads, no direction/action head, no training, and no held-out tuning.

These checks do **not** measure ADVANCE/RETRACT accuracy, wire detection, or target understanding. Optional overlays under `--output-dir` are for on-Perlmutter visual verification only; keep them off shared docs and off your laptop.

## What each probe scores

All feature distances use **L2** on the same 512-d interface the GRU would see (`history` → `(6, 512)`, `goal` → `(512)`). Encoding runs in eval mode with gradients disabled; the frozen backbone stays frozen.

1. **Occlusion sensitivity (history; not wire detection):** grid-occlude squares on each prepared history tensor; record L2 change. Patches that fall in preparation padding are excluded. Defaults: `patch=74`, `stride=74`, replacement `0.0` in normalized space (~ImageNet mean after `DINOv2Preprocessor`). Heatmaps can be saved on the **original** frames with the max-change patch outlined. The highest-change patch is **not** labeled as the wire without ground truth.
2. **Goal marker-region perturbation:** detect red-ish pixels on the **original goal RGB before ImageNet normalization** (uint8 range `[0,255]`; arithmetic in int16). Thresholds: `R >= 150` and `R >= G + 40` and `R >= B + 40`. Replace the marker with mean-color RGB, then run the normal preprocessor. Compare against several **same-shape** masks translated to valid non-overlapping locations, using the same replacement method. Reports marker L2, the control L2 distribution, and marker/control ratios; ratios are `unavailable` if the denominator is ~0. Empty masks or no valid controls are reported as `unavailable` rather than scored. This is sensitivity to a marker-region edit—not proof the encoder understands the target.
3. **Temporal feature variation:** consecutive history feature L2 distances aligned with length-6 `depth_relative_mm` (dataset order: oldest→newest). Reports signed and absolute depth steps separately. Optional Pearson correlation uses **feature L2 vs absolute depth change only** (magnitude association). Constant/missing/short sequences yield `unavailable` correlations. Feature variation does **not** establish wire-motion understanding, advance/retract recognition, or prediction quality.

## Synthetic check — laptop or Perlmutter

Needs PyTorch and Pillow only (no torchvision import at module import time):

```sh
python3 -m unittest discover -s tests -p 'test_cnn_diagnostics.py' -v
```

Tool-verified locally after the diagnostics fix (2026-11-02): ten invented-input tests passed (Python 3.9 / torch 2.2.2). They cover gray-vs-red overflow, empty masks, control placement, zero-distance ratios, padding exclusion, temporal alignment edge cases, and overlay bookkeeping. They do not load ResNet weights or real cases.

## One-case run — Perlmutter terminal

Uses `pytorch/2.13.0` (torch + torchvision). Grid occlusion is heavier than the encoder smoke check; prefer a short GPU allocation. Confirm `cases.jsonl` exists under `DATA_DIR`.

```sh
# Run on Perlmutter (GPU session recommended for the real-case probe).
module load pytorch/2.13.0
export TORCH_HOME="${PSCRATCH:?}/model-cache/torch"
export DATA_DIR="${PSCRATCH}/CS5470_FinalProject/shorthorizon2"
if [ ! -f "$DATA_DIR/cases.jsonl" ]; then
  export DATA_DIR=/global/cfs/cdirs/m4999/studentdata/shorthorizon2
fi
test -f "$DATA_DIR/cases.jsonl"

cd "${REPO_DIR:?Set REPO_DIR to your waldo-guidewire-policy clone}"
OUT="${PSCRATCH:?}/waldo-cnn-diagnostics/case0"
mkdir -p "$OUT"

python3 -m unittest discover -s tests -p 'test_cnn_diagnostics.py' -v
python3 cnn_diagnostics.py \
  --data-dir "$DATA_DIR" \
  --split-position 0 \
  --partition train \
  --case-index 0 \
  --output-dir "$OUT"
```

Optional: `--patch` / `--stride`, `--size`, `--remove-maxpool`.

Share only permitted printed aggregates. Keep overlay PNGs on Perlmutter. Do not transfer images, masks, feature vectors, or sensitive errors off the machine.

## Prior owner-run note (2026-11-02, pre-fix)

An earlier GPU run completed on one training case with the previous diagnostics code (uint8 red arithmetic, single corner control, “wire” wording, and a magnitude-style depth correlation). Those aggregates are **not** revalidated under the fixed probes. Re-run with the command above before treating marker/occlusion numbers as current evidence. See [PROGRESS.md](PROGRESS.md).

## How to read outcomes

| Signal | Encouraging pattern | Weak / unavailable |
| --- | --- | --- |
| Occlusion sensitivity | max L2 ≫ median among content patches | flat grid; no content patches |
| Goal marker perturbation | marker L2 above control distribution | empty mask; no controls; ratio unavailable |
| Temporal variation | non-trivial step L2; abs-depth corr only if defined | near-zero trajectory; corr unavailable |

Encoder architecture remains in [cnn.md](cnn.md). Implementation outcomes belong in [PROGRESS.md](PROGRESS.md).
