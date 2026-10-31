# DINOv2 preparation and feature trial

`preprocessing.py` prepares images for the initial `vit_small_patch14_dinov2.lvd142m` trial. `vit.py` now implements inference-only extraction; actual pretrained-model execution on Perlmutter remains unverified. Training resolution, freezing policy, and batch size remain undecided.

The trial uses 518×518 inputs: convert to RGB, resize the entire image proportionally with Pillow bicubic interpolation, and center it on a padded square. For 1065×1050 images, the resized image is 518×511 with 3 pixels above and 4 below. Nothing is cropped; shrinking can still reduce wire visibility.

Pixels are converted to float32, divided by 255, then normalized channel-by-channel as `(pixel - mean) / std`. The [checkpoint configuration](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/blob/main/config.json) specifies mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. Padding is RGB `(124, 116, 104)`, the nearest 8-bit representation of that mean, so normalized padding is approximately zero. No random augmentation is applied.

## Dependencies and synthetic check

This step requires Pillow and PyTorch only; it does not require torchvision, timm, a GPU, or downloaded weights. On Perlmutter, use the existing module environment and first confirm that PyTorch imports. Do not reinstall or replace the shared environment automatically.

From the repository root, this command runs only the new invented-image tests:

```sh
python3 -m unittest discover -s tests -p 'test_preprocessing.py' -v
```

It checks full-view geometry, edge retention, RGB/scaling, unchanged originals, goal/history separation, sequence alignment, repeatability, and invalid sizes. These checks cannot establish visibility on real data. Results and environment limits are recorded in [progress](PROGRESS.md).

## One-case preview — owner-run on Perlmutter only

Before any authorized code handoff, save the remote notebook and preserve a separate copy of its edits and sensitive outputs on Perlmutter. No Git handoff is implied here. The laptop implementation must be available in the remote clone before these imports work.

Run the following as a **new notebook cell on Perlmutter**, after the existing raw-loading cell has set `case` to one training case. This reuses the successful raw loading rather than repeating setup. Keep the shared notebook output-free; all previews and tensors remain on Perlmutter.

```python
import torch
from IPython.display import display
from preprocessing import DINOv2Preprocessor

prepared = DINOv2Preprocessor().prepare_case(case)
assert prepared["goal"].shape == (3, 518, 518)
assert prepared["history"].shape == (6, 3, 518, 518)
assert torch.isfinite(prepared["goal"]).all()
assert torch.isfinite(prepared["history"]).all()
assert prepared["depth_relative_mm"] == case["depth_relative_mm"]
assert prepared["history_offsets_s"] == case["history_offsets_s"]

print("Goal: original, then prepared")
display(case["goal"], prepared["previews"]["goal"])
for i, (original, preview) in enumerate(
    zip(case["history"], prepared["previews"]["history"]), start=1
):
    print(f"History {i}: original, then prepared")
    display(original, preview)
```

Inspect wire-tip and red-target visibility, all edges, and sensible history order. View prepared images at their actual pixel size as well as zoomed if useful. Share only a permitted textual outcome; do not transfer images, tensors, records, or sensitive errors off Perlmutter. Stop here before feature extraction if shrinking has obscured important details.

## Returned interface

- `goal`: float32 tensor `[3, 518, 518]`.
- `history`: float32 tensor `[6, 3, 518, 518]`, preserving the supplied order.
- Original metadata keys: copied unchanged; labels and identifiers are not encoder inputs.
- `previews`: separate goal and six history RGB images, exactly matching the pixels used for tensor conversion.
- `preparation`: checkpoint name, size, interpolation, padding color, normalization, and separate goal/history geometry. Dimensions are `(width, height)`; padding is `(left, top, right, bottom)`.

These are one-case tensors, without a case-batch dimension. The raw `case` remains unchanged. The next encoder should receive only the goal/history tensors, with one image at a time for the first inference check. Smaller synthetic sizes are supported when divisible by 14; changing a real trial size requires a corresponding encoder configuration and another visibility check.

## Encoder dependencies — Perlmutter terminal

Preparation tests and initial preview inspection have passed as recorded in progress. Do not repeat them without a relevant change. In the same terminal with the existing PyTorch module active, check the remaining library imports:

```sh
python3 -c 'import timm, torchvision; print("timm", timm.__version__, "torchvision", torchvision.__version__)'
```

This reads no images, downloads no weights, and uses no GPU. If an import fails, record the non-sensitive error and resolve environment compatibility before the encoder run; do not blindly upgrade the module's PyTorch. No particular timm version has been verified remotely yet.

The encoder tests use an invented model, so they need only the already-used torch/Pillow environment:

```sh
python3 -m unittest discover -s tests -p 'test_vit.py' -v
```

## One-case pretrained extraction — Perlmutter notebook

After an authorized code handoff, add a new cell below the successful preview cell. Preserve the remote notebook and outputs on Perlmutter before pulling new code. The laptop does not have those new cells.

The actual extraction loads public pretrained weights through timm, and processes image tensors locally. It does not call an external inference service. Weight download/cache access must work on Perlmutter. Use an allocated compute session or GPU-backed notebook for this encoder trial; no scheduler allocation is created by this code. CPU execution is supported but can be slower. The notebook kernel must provide the same compatible packages as the checked terminal.

```python
import os
from pathlib import Path
import torch

# Keep downloaded weights in scratch. Set before constructing the encoder.
os.environ.setdefault("HF_HOME", str(Path(os.environ["PSCRATCH"]) / "model-cache/huggingface"))
os.environ.setdefault("TORCH_HOME", str(Path(os.environ["PSCRATCH"]) / "model-cache/torch"))
from vit import DINOv2Encoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Encoder device:", device)
encoder = DINOv2Encoder(device=device)
features = encoder.encode(prepared["goal"], prepared["history"])
assert features["goal_features"].shape == (384,)
assert features["history_features"].shape == (6, 384)
assert torch.isfinite(features["goal_features"]).all()
assert torch.isfinite(features["history_features"]).all()
print("One-case feature shapes and finite-value checks passed.")
```

These are whole-image summary vectors, using the checkpoint's own class-token pooling as in its [feature extraction example](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m). Only image tensors are passed into the encoder. No labels, identifiers, depth, or time enter it. No parameter updates or feature files are produced. A valid feature vector is not evidence of action accuracy or wire localization. This wrapper is intentionally inference-only and must not be used unchanged as a trainable component later.

## Small-batch follow-up — only after the one-case run passes

The encoder also accepts goals `[B,3,518,518]` and histories `[B,6,3,518,518]`, returning `[B,384]` and `[B,6,384]`. Start with two training cases. Compare batched results against each case encoded individually with `torch.testing.assert_close`; inspect discrepancies rather than requiring bitwise equality. The default image chunk is one even when B is two, so case grouping does not silently raise GPU workload. An explicit `image_chunk_size=2` can then test processing images together; this is not a training batch-size decision.

Do not run a dataset-wide feature sweep. Report only permitted shape/check results, device, software versions, and memory/runtime aggregates. Keep actual feature values and all sensitive outputs on Perlmutter. For later timing, warm up the model, synchronize CUDA around measurements, and separate preprocessing, weight loading, and encoder runtime. No GPU latency or memory measurement has been made yet.
