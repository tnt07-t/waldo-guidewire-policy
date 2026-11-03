# CNN encoder trial — ResNet-18

`cnn.py` implements a frozen ImageNet-pretrained ResNet-18 frame encoder for Group 1. It turns prepared goal and history images into 512-dimensional feature vectors. GRU/LSTM fusion, depth/time use, training, and action prediction are not part of this file.

Owner-run one-case smoke check on Perlmutter passed (2026-11-02). Exact printed device, shapes, optional flags, and remote HEAD were not supplied; do not repeat the completed default check without a relevant change. Real images, tensors, and feature values must stay on Perlmutter. Passing this check does not measure action accuracy.

To test whether those 512-d vectors change under occlusion / marker-region / temporal probes useful for a later GRU, use the separate [CNN_DIAGNOSTICS.md](CNN_DIAGNOSTICS.md) / `cnn_diagnostics.py` path. Those probes do not claim wire detection or action recognition. Do not fold them into `cnn.py`.

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

### Preprocessor options

`cnn.py`, `cnn_diagnostics.py` and `evaluation.py` share one set of input flags (`add_preprocessor_arguments` and `build_preprocessor` in `preprocessing.py`).
Without them the default `dinov2` full-view preparation runs exactly as before.

| Flag | Applies to | Default |
| --- | --- | --- |
| `--preprocessor {dinov2,wire}` | both | `dinov2` |
| `--size` | `dinov2` only | 518 |
| `--crop-size` | `wire` only | 256 |
| `--channels {raw,enhanced}` | `wire` only | `raw` |
| `--tophat-size` | `wire --channels enhanced` | 11 |
| `--difference {median,previous}` | `wire --channels enhanced` | `median` |
| `--context-size` | `wire` only | off |

A flag given for the other preprocessor is rejected with an error, not ignored; for example `--size` with `wire`.
`wire` crops a square around `goal_xy` at native resolution from the unmarked goal and the six history frames (see [the ticket index](tickets/README.md)).

```sh
# Run on Perlmutter, in a GPU allocation: compare old and new paths on one case.
export TORCH_HOME="${PSCRATCH:?}/model-cache/torch"
python3 cnn.py --split-position 0 --partition train
python3 cnn.py --split-position 0 --partition train --preprocessor wire --crop-size 256
python3 cnn.py --split-position 0 --partition train --preprocessor wire --crop-size 256 --channels enhanced
```

Each run prints the device, `(1, 6, 512)`, `(1, 512)` and `Frozen encoder check passed.`

## Wire preprocessing

**Why:** `DINOv2Preprocessor` shrinks the whole 1065 x 1050 frame to 518, so the dark 3-4 px guidewire becomes about 1.5-2 px before the ResNet stem and is averaged away by global pooling.

**Design** (`WirePreprocessor` in `preprocessing.py`):

- one square box around `goal_xy` (rounded half up) is cropped at native resolution from the unmarked goal (`goal_raw`) and all six history frames, since the view is fixed within a case;
- outside the image the box repeats edge pixels; the crop size must be a multiple of 32;
- `raw` repeats the gray crop to three channels with ImageNet normalization, as the frozen ResNet expects;
- `enhanced` uses raw, a black top-hat (thin dark structures) and a frame difference (median or previous frame); the goal's difference channel is zero, and `--standardize-channels` fits the extra channels' scale on training cases;
- optional `context_size` adds a downscaled full view; `random_center_seed` and `zero_channels` exist for controls.

**Chosen parameters:** none are adopted yet; `dinov2` stays the default.
Training-only selection preferred a raw 512 crop (inner BA 0.597), but it scored 0.506 on split 0 validation against 0.547 for the full-frame baseline.
The visual check of top-hat size and crop size (T13 notebook cells) was not run.
Results, latency and limits are recorded once in [the 2026-11-03 progress entry](PROGRESS.md#2026-11-03---wire-preserving-preprocessing-and-frozen-feature-probe-owner-run-ladder-stopped-early).

**Commands:** the flags are in [Preprocessor options](#preprocessor-options); the probe, controls and selection commands are in [EVALUATION.md](../EVALUATION.md#preprocessor-options).

Weights download or cache access must work on Perlmutter. Keeping Torch Hub caches under scratch is recommended before constructing the encoder if home-quota limits appear:

```sh
export TORCH_HOME="${PSCRATCH:?}/model-cache/torch"
```

## Meaning and limits

Passing shapes and finite checks shows that preparation, pretrained load, freezing, and goal/history bookkeeping work for the inspected case. It does not show that the features preserve wire location, that history helps, or that ADVANCE/RETRACT prediction will beat the baseline.

Next Group 1 steps: re-run fixed diagnostics with overlays on Perlmutter, agree on GRU versus LSTM and depth/time fusion, then implement only the approved temporal stage. See [CNN_DIAGNOSTICS.md](CNN_DIAGNOSTICS.md) and [progress](PROGRESS.md).
