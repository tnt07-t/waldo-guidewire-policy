# T13 - Notebook cells: channel check and crop-size boxes

**Status:** Todo
**Code written:** laptop (cells are written, never executed there)
**Tested:** Perlmutter, by the owner, who runs and views the cells
**Depends on:** T09
**Files:** `data_inspect.ipynb`

## Why

Two parameters should be set from looking at training images, without labels:

- `tophat_size`: the smallest window where the whole wire lights up but blob-shaped anatomy does not;
- `crop_size`: the smallest crop that contains the wire tip in at least 90% of cases.

## Scope

Add a "Group 1 - wire preprocessing inspection (Perlmutter only)" section with output-free cells:

1. **Channel check.**
   For a fixed, seeded sample of 20 training cases from split 0, show the raw crop, the top-hat at sizes 7, 11 and 15, and the difference channel for the newest frame, side by side.
2. **Crop-size boxes.**
   For a seeded sample of 40 training cases, draw 192, 256, 384 and 512 boxes around `goal_xy` on the newest history frame.
   Provide a small tally cell where the viewer records, per case, the smallest box containing the tip (or "outside").
   The tally stays in the notebook session or a file under `$PSCRATCH`, never in the repository.
3. A markdown cell explaining how to read both views and what to report (counts only).

## Out of scope

- Automatic tip detection.
- Using validation or test cases.

## Acceptance criteria

- The committed notebook has no outputs (clear before committing).
- Cells use the shared `DATA_DIR` setup already in the notebook.
- The permitted report is aggregate: chosen `tophat_size`, and how many of 40 tips fall inside each box size.
