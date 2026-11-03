# T07 - Goal-centered native-resolution crop

**Status:** Todo
**Code written:** laptop (no execution)
**Tested:** Perlmutter, by the owner
**Depends on:** T01
**Files:** `preprocessing.py`, `tests/test_preprocessing.py`

## Why

Cropping around `goal_xy` at native resolution keeps the 3-4 px wire intact and makes the crop center the target.
Because the view is fixed within a case, one crop box applies to the goal and all six history frames.

## Scope

- Implement `goal_crop(image, goal_xy, size)` returning the cropped grayscale image and its geometry.
- Convert to grayscale (`"L"`) before cropping; history frames are black and white, and the unmarked goal is used for the goal.
- Center: round `goal_xy` to the nearest pixel; box is `(cx - size // 2, cy - size // 2)` to `(+size, +size)`.
- Parts of the box outside the image are filled by edge replication, not a constant color.
- `size` must be a positive multiple of 32 (ResNet's total stride), so 192, 256, 384 and 512 are valid.
- Geometry uses the existing keys so diagnostics keep working, plus the crop box:
  - `original_size_wh`;
  - `crop_box_ltrb` in original-image pixels (may extend outside the image);
  - `resized_size_wh` equal to the content size inside the crop;
  - `padding_ltrb` giving how many pixels on each side came from replication.

## Out of scope

- Any rescaling; this ticket crops at 1x only.
- Choosing the default size (T13 informs it; start with 256).

## Acceptance criteria

Tests on invented images:

- A marked pixel at `goal_xy` lands at crop center `(size // 2, size // 2)`.
- Goals near each of the four borders produce the right `padding_ltrb` and replicated edge values.
- Odd or non-multiple-of-32 sizes raise `ValueError`.
- The input image is unchanged.

## Run on Perlmutter (owner)

```bash
module load pytorch/2.13.0
cd "${REPO_DIR:?}" && git pull
python3 -m unittest tests.test_preprocessing -v
```
