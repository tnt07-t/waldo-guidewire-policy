# T03 - Record owner-reported data facts

**Status:** Todo
**Code written:** laptop (docs only)
**Tested:** nothing to run
**Depends on:** nothing
**Files:** `docs/DATASET.md`

## Why

The preprocessor design depends on facts about the images that are not yet in the shared dataset notes.
Teammates and agents should find them in one place, labeled as owner-reported.

## Scope

Add a short dated subsection (2026-11-03, owner-reported visual inspection) to `DATASET.md` stating:

- `goal_xy` uses the same pixel coordinates as the history frames; the goal image shows the same view.
- The guidewire appears dark, about 3-4 px wide and long relative to its width.
- History frames are black and white (grayscale content).
- The view does not shift between frames within a case.
- Typical tip distance from `goal_xy` is not yet known.

Note how many cases the observation covered if the owner can say; otherwise say coverage was unspecified.

## Out of scope

- Claims about every case in the dataset.
- Changes to the dataset itself.

## Acceptance criteria

- The facts appear once, in `DATASET.md`, clearly marked as owner-reported with coverage stated or marked unspecified.
- Other docs link to that section instead of repeating it.
