# CNN encoder trial — ResNet-18

`cnn.py` implements a frozen ImageNet-pretrained ResNet-18 frame encoder for Group 1. It turns prepared goal and history images into 512-dimensional feature vectors. GRU/LSTM fusion, depth/time use, training, and action prediction are not part of this file.

Owner-run one-case smoke check on Perlmutter passed (2026-11-02). Exact printed device, shapes, optional flags, and remote HEAD were not supplied; do not repeat the completed default check without a relevant change. Real images, tensors, and feature values must stay on Perlmutter. Passing this check does not measure action accuracy.

To test whether those 512-d vectors look informative for a later GRU (wire sensitivity, goal marker, motion math on the feature sequence), use the separate [CNN_DIAGNOSTICS.md](CNN_DIAGNOSTICS.md) / `cnn_diagnostics.py` path. Do not fold those probes into `cnn.py`.

## Why ResNet-18 instead of ConvNeXt Tiny or MobileNetV2

Group 1 chose ResNet-18 as the first CNN backbone. ConvNeXt Tiny and MobileNetV2 remain plausible alternatives, but were set aside for this trial because they are too lightweight for the starting capacity we want.

Relevant project context, not a completed bake-off:

- Earlier very small locally hosted models reportedly performed poorly; insufficient capacity is a suggested explanation, not a proven cause. See the shared approach notes.
- MobileNetV2 is optimized for efficient mobile inference and uses a comparatively small feature budget. That efficiency is useful later for latency, but is a weak first baseline if under-capacity is already a concern.
- ConvNeXt Tiny is stronger than MobileNet-class models, yet still sits at the light end of modern ConvNet families. It is a reasonable later substitute, not the first capacity choice here.
- ResNet-18 is still compact, ships in torchvision with the existing Perlmutter `pytorch/2.13.0` module, and produces a familiar 512-d pooled descriptor without adding timm as a CNN dependency.

No guidewire accuracy or latency comparison among these three has been run. Keep the encoder swappable; replacing ResNet-18 later should reuse the raw reader and most of the preparation path, with normalization and input size rechecked.

## Architectural decisions in `cnn.py`

| Decision | Choice | Why |
| --- | --- | --- |
| Backbone | ImageNet-pretrained ResNet-18 (`ResNet18_Weights.DEFAULT`) | Agreed Group 1 starting encoder; torchvision-native on the shared module. |
| Output | Replace `fc` with `Identity`; keep final average pooling | Per-image features of width 512, not 1000-way ImageNet logits. |
| History shape | `(B, T, 3, H, W) -> (B, T, 512)` with `T=6` | Matches the dataset's six chronological frames; frame order is preserved after a temporary batch×time reshape. |
| Goal handling | Same encoder weights; goal encoded as `(B, 3, H, W) -> (B, 512)` | Goal is a separate reference, never a seventh history step. |
| Freezing | `requires_grad_(False)` plus `train()` override that keeps the backbone in `eval()` | Fully frozen baseline so BatchNorm running stats stay fixed while a later GRU/classifier trains. |
| Max-pool stem | Kept by default; `--remove-maxpool` optional | Standard ResNet-18 first; optional identity max-pool keeps larger internal maps if the thin wire needs it. |
| Depth / time | Not consumed here | Stay aligned with history in the reader; fusion belongs in the temporal policy. |
| Preparation | Reuse `DINOv2Preprocessor` | Same RGB, proportional bicubic resize, centered mean-color pad, and ImageNet mean/std. The preprocessor's DINOv2 checkpoint label describes preparation provenance only, not this backbone. |
| Default trial size | 518 (multiple of 14) | Reuses the existing full-view trial tensors. ResNet adaptive pooling accepts non-224 inputs; 224 remains available when divisible by 14. Neither size is a final training decision. |
| Smoke entrypoint | `python cnn.py ...` | One real case, shape and finite checks only; no training, scoring, or feature writes. |

Remaining encoder follow-ups recorded in code: visual inspection of prepared wire/target visibility, temporal models and goal fusion, frozen-baseline training, then optional layer4 fine-tuning with a smaller backbone learning rate.

## Dependencies

CNN smoke checks need PyTorch and torchvision from the existing Perlmutter module. They do not need timm. Preparation still uses Pillow through the shared preprocessor.

```sh
# Run on Perlmutter, in your own terminal.
module load pytorch/2.13.0
python3 -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__)'
```

Owner-confirmed module packages include torch 2.13.0+cu130 and torchvision 0.28.0+cu130. Do not reinstall or replace that environment automatically.

## One-case encoder check — Perlmutter terminal

Owner-reported default one-case run passed on 2026-11-02. Keep the commands below for reruns after encoder/preparation changes or when checking optional flags. Laptop edits do not read the dataset. NERSC permits very small login-node tests; larger or prolonged work belongs on compute nodes.

```sh
# Run on Perlmutter.
module load pytorch/2.13.0
export DATA_DIR="${PSCRATCH:?PSCRATCH must be set}/CS5470_FinalProject/shorthorizon2"
test -d "$DATA_DIR"

cd "${REPO_DIR:?Set REPO_DIR to your waldo-guidewire-policy clone}"
python3 cnn.py --split-position 0 --partition train
```

Expected textual outcome only: device name, history feature shape `(1, 6, 512)`, goal feature shape `(1, 512)`, and `Frozen encoder check passed.` Share only those permitted aggregates. Do not copy images, tensors, feature values, or sensitive errors off Perlmutter.

Optional variants:

```sh
# Same check without the early ResNet max-pool.
python3 cnn.py --split-position 0 --partition train --remove-maxpool

# Alternate square size accepted by the shared preprocessor (multiple of 14).
python3 cnn.py --split-position 0 --partition train --size 224
```

Weights download or cache access must work on Perlmutter. Keeping Torch Hub caches under scratch is recommended before constructing the encoder if home-quota limits appear:

```sh
export TORCH_HOME="${PSCRATCH:?}/model-cache/torch"
```

## Meaning and limits

Passing shapes and finite checks shows that preparation, pretrained load, freezing, and goal/history bookkeeping work for the inspected case. It does not show that the features preserve wire location, that history helps, or that ADVANCE/RETRACT prediction will beat the baseline.

Next Group 1 steps: confirm diagnostic occlusion peaks against prepared wire/target previews, agree on GRU versus LSTM and depth/time fusion, then implement only the approved temporal stage. Diagnostics aggregates are in [CNN_DIAGNOSTICS.md](CNN_DIAGNOSTICS.md) and [progress](PROGRESS.md).
