# T16 - Episode-grouped inner CV for parameter selection

**Status:** Todo
**Code written:** laptop (no execution)
**Tested:** Perlmutter, by the owner (unit tests here; real runs in T17)
**Depends on:** T14
**Files:** `evaluation.py` (or a small `select_preprocessing.py`), tests, `docs/EVALUATION.md`

## Why

About ten preprocessing configurations will be compared.
Choosing among them on the same 80 validation cases each time inflates the winner's score.
Selection should happen inside the training partition, with validation only confirming the final choice.

## Scope

- Split the training partition into 5 folds with `sklearn.model_selection.GroupKFold`, grouping by `episode` so no episode is in two folds.
- For each preprocessing configuration, extract frozen features once, fit the existing probe on 4 folds and score balanced accuracy on the fifth, for all 5 rotations.
  C is chosen inside each training rotation only.
- Report the mean and standard deviation of inner-fold balanced accuracy per configuration.
- Search one parameter at a time, in this order, keeping the others at their current best:
  1. crop size (from T13's candidates);
  2. channel mode (raw vs enhanced);
  3. difference mode (median vs previous);
  4. top-hat size (only if T13 left more than one plausible value);
  5. context view on/off (only if T10 was built).
- When two configurations are within one standard deviation, prefer the cheaper one (smaller crop, fewer channels).
- Only the final chosen configuration is then scored once on validation.

## Out of scope

- A full grid search.
- Test partitions.

## Acceptance criteria

- An invented-input test checks that no episode appears in both the fitting and scoring folds of any rotation.
- Output is a table of configuration, mean and std inner balanced accuracy, and the selected configuration.

## Run on Perlmutter (owner)

```bash
module load pytorch/2.13.0
cd "${REPO_DIR:?}" && git pull
python3 -m unittest discover -s tests -v
```
