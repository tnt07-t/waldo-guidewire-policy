# T08 - `WirePreprocessor`, raw mode

**Status:** Todo
**Code written:** laptop (no execution)
**Tested:** Perlmutter, by the owner
**Depends on:** T04, T05, T07
**Files:** `preprocessing.py`, `tests/test_preprocessing.py`

## Why

This is the first usable step of the new pipeline: native-resolution goal-centered crops, fed to the frozen ResNet exactly as it expects.
It isolates the effect of cropping before any new channels are added.

## Scope

- Add `WirePreprocessor(crop_size=256, channels="raw")` with the same `prepare_image` / `prepare_case` interface as `DINOv2Preprocessor`.
- `prepare_case` reads `case["goal_raw"]` for the goal and `case["goal_xy"]` for the crop center; history frames use the same box.
- `raw` mode: grayscale crop repeated to three channels, scaled by `/ 255`, then normalized with the ImageNet mean/std per channel.
  This keeps the frozen ImageNet ResNet's expected input statistics.
- Output layout matches today's: `goal [3, S, S]`, `history [6, 3, S, S]`, plus `previews` and `geometry`.
- The `preparation` record (using T05) lists the class name, crop size, channel mode, normalization constants and that no resize was applied.
- Unknown `channels` values raise `ValueError` (T09 adds `enhanced`).

## Out of scope

- Top-hat and difference channels (T09), context view (T10), CLI flags (T11).

## Acceptance criteria

- Invented 1065x1050 case: shapes are `(3, 256, 256)` and `(6, 3, 256, 256)`, all finite.
- The three channels of a raw-mode tensor are identical before normalization.
- A pixel drawn at `goal_xy` on every invented frame appears at the tensor center in goal and history.
- `FrameEncoder(pretrained=False)` accepts the output and returns `(1, 6, 512)` and `(1, 512)`.
- The raw case dictionary is not modified.

## Run on Perlmutter (owner)

```bash
module load pytorch/2.13.0
cd "${REPO_DIR:?}" && git pull
python3 -m unittest discover -s tests -v
```
