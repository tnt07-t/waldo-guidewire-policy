# T09 - Enhanced channels: top-hat and difference

**Status:** Todo
**Code written:** laptop (no execution)
**Tested:** Perlmutter, by the owner
**Depends on:** T06, T08
**Files:** `preprocessing.py`, `tests/test_preprocessing.py`

## Why

Raw grayscale wastes two of the three input channels on copies.
Two channels can instead carry wire-specific signal:

- **top-hat:** thin dark structures (the wire), with smooth anatomy removed;
- **difference:** what moved, with static anatomy cancelled (the view is fixed within a case).

## Scope

- Add `channels="enhanced"`: channel 0 raw, channel 1 black top-hat (T06), channel 2 difference.
- New parameters with these defaults:
  - `tophat_size=11` (odd, wider than the wire);
  - `difference="median"`: frame minus the per-pixel median of the six history crops; `"previous"`: frame minus the previous frame, with zeros for the first frame.
- The goal has no timeline, so its difference channel is zeros.
  Record this choice in the `preparation` record so it can be revisited.
- Normalization:
  - channel 0 uses ImageNet normalization as in raw mode;
  - channels 1 and 2 are scaled by `/ 255`, then standardized with constants passed to the constructor (`tophat_stats=(mean, std)`, `difference_stats=(mean, std)`, defaulting to `(0.0, 1.0)`).
- Add a helper `channel_stats(preprocessor, cases)` that returns those constants from a list of cases.
  Its docstring must say it may only be called on training cases.
- Add `zero_channels=()` so T15 can zero chosen channels after normalization for ablations.

## Out of scope

- Picking `tophat_size` or `difference` (T13 and T16 do that).
- Training conv1; the frozen-encoder caveat is documented in T17.

## Acceptance criteria

Tests on invented cases:

- Six identical history frames give an all-zero difference channel for both modes.
- A dark 4 px line that moves between frames appears in the difference channel only where it moved.
- The same line is bright in the top-hat channel and the background is near zero.
- `zero_channels=(1,)` zeroes exactly channel 1.
- `channel_stats` on invented cases returns finite values and matches a direct computation.
- Raw mode output from T08 is unchanged.

## Run on Perlmutter (owner)

```bash
module load pytorch/2.13.0
cd "${REPO_DIR:?}" && git pull
python3 -m unittest tests.test_preprocessing -v
```
