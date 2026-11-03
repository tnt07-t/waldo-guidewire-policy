# T17 - Run the evaluation ladder

**Status:** Todo
**Code written:** none; the agent prepares the exact commands for each step
**Tested:** Perlmutter, by the owner, in a GPU allocation
**Depends on:** T11-T16
**Files:** none changed; results go into T18

## Why

This is where we find out whether the preprocessor helps or not.
Each step changes one thing, so any gain has a clear cause.

## Steps

1. **Diagnostics re-run (already pending).**
   Run fixed `cnn_diagnostics.py` on the same training case three ways: `dinov2` at 518, `wire` raw at the chosen crop, `wire` enhanced.
   Look at the overlays on Perlmutter; the strongest occlusion response should move toward the wire.
2. **Parameter selection.**
   Run T16's inner CV on split 0's training partition.
3. **Confirm on validation, split 0.**
   Run the probe for: `dinov2` (reproduce .547), `wire` raw, `wire` with the chosen enhanced settings.
4. **Controls (T15).**
   Shuffled labels, random center and channel ablation for the chosen config.
5. **All splits.**
   Repeat step 3 on split positions 0-4.
6. **Latency (T12)** with one real case on the same GPU node.

## What counts as working

- Inner CV and validation both beat the `dinov2` baseline and the metadata-only baseline (.547).
- The gain is consistent across most of the five splits, not one lucky split.
- The real result is clearly above the shuffled-label distribution.
- Random-center crops score lower than goal-centered crops.
- Ablating an enhanced channel lowers the score (otherwise drop that channel).
- The train/validation gap shrinks compared with the current near-1.0 train scores.

## If it does not work

- Raw crop helps but enhanced does not: the frozen ImageNet conv1 may not use the new channels.
  Next try is training conv1 (or layer4) or a small CNN from scratch on the crops; that is a separate ticket.
- Nothing helps: check T13's tip tally; the crop may be missing the tip, which points to T10.

## Reporting

Share only permitted aggregates: configuration, balanced accuracy, recalls, intervals, timing.
Keep images, features and per-case outputs on Perlmutter.
