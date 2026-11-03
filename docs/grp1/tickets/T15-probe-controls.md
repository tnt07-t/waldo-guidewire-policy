# T15 - Probe controls

**Status:** Todo
**Code written:** laptop (no execution)
**Tested:** Perlmutter, by the owner (unit tests here; real runs in T17)
**Depends on:** T14
**Files:** `evaluation.py` (or a small `evaluation_controls.py` that imports it), tests, `docs/EVALUATION.md`

## Why

With 80 validation cases, one split's balanced accuracy has a 95% interval of about +/-0.11, and cases within an episode are correlated.
These controls separate a real preprocessing gain from noise or from a shortcut.

## Scope

Add four opt-in modes, each reported next to the normal result:

1. **Shuffled labels.**
   Refit the probe on training labels permuted within the training set, `--permutations 20` by default, seeded.
   Report the distribution of validation balanced accuracy; the real result must sit clearly above it.
2. **Random center.**
   Crop around a seeded random point instead of `goal_xy`, at least `crop_size / 2` away from the goal and inside the image.
   If accuracy does not drop, goal centering is not what helps.
   Implement as a `WirePreprocessor` option that records the chosen centers in its `preparation` record.
3. **Channel ablation.**
   Use T09's `zero_channels` to zero the top-hat or difference channel for both training and validation, and refit.
   If nothing changes, that channel adds nothing.
4. **All splits.**
   Run the same config on split positions 0-4 and report per-split validation balanced accuracy plus their mean.
   Do not pool cases across splits as if independent.

## Out of scope

- Any use of test partitions.

## Acceptance criteria

- Invented-input tests: permutation keeps class counts; random centers are seeded, valid and far enough from the goal; ablated channels are zero.
- Each mode writes its results into the run's private `metrics.json` and a short line in `summary.md`.

## Run on Perlmutter (owner)

```bash
module load pytorch/2.13.0
cd "${REPO_DIR:?}" && git pull
python3 -m unittest discover -s tests -v
```
