# T10 - Optional full-view context input

**Status:** Todo, optional
**Code written:** laptop (no execution)
**Tested:** Perlmutter, by the owner
**Depends on:** T08
**Files:** `preprocessing.py`, `tests/test_preprocessing.py`

## Why

If the wire tip is often outside the crop, the crop alone cannot see it.
A small downscaled full view gives coarse context about where the wire is.
Build this only if T13 shows the tip falls outside a reasonable crop (384 or less) in more than about 10% of cases.

## Scope

- Add `context_size=None` to `WirePreprocessor`; when set (for example 224, a multiple of 32), also return:
  - `goal_context [3, C, C]` and `history_context [6, 3, C, C]`;
  - grayscale repeated to three channels, proportional bicubic resize, centered padding with the mean gray, ImageNet normalization.
- The crop outputs (`goal`, `history`) are unchanged whether or not context is enabled.
- Geometry for the context images uses the existing `padding_ltrb` convention.

## Out of scope

- How a policy fuses crop and context features; that belongs to the GRU work.

## Acceptance criteria

- With `context_size=None` the output has no context keys and equals T08/T09 output.
- With `context_size=224` shapes are `(3, 224, 224)` and `(6, 3, 224, 224)`.
- All four image edges survive the context resize (same style as the existing edge test).

## Run on Perlmutter (owner)

```bash
module load pytorch/2.13.0
cd "${REPO_DIR:?}" && git pull
python3 -m unittest tests.test_preprocessing -v
```
