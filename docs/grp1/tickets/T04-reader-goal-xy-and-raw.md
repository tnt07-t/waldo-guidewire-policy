# T04 - Expose `goal_xy` and the unmarked goal from the reader

**Status:** Todo
**Code written:** laptop (no execution)
**Tested:** Perlmutter, by the owner: synthetic tests, then one real-case decode
**Depends on:** T01
**Files:** `dataset.py`, `tests/test_dataset.py`, `docs/LOADER.md`

## Why

The wire preprocessor crops around `goal_xy`, and it should crop the unmarked goal (`goal_raw.png`) so the red marker does not dominate the crop.
`GuidewireDataset.__getitem__` currently returns neither.
`goal_xy` is in each record; the unmarked goal path is the first entry of `clean_paths`.

## Scope

- Add two keys to the returned case dictionary:
  - `goal_xy`: the record's `goal_xy` as a list of two numbers, unchanged;
  - `goal_raw`: the image at `clean_paths[0]`, read with the same safe `_read_image` path checks.
- Validate that `clean_paths` has seven entries and `goal_xy` has two, raising `ValueError` otherwise.
- Keep every existing key and its behavior unchanged.

## Design notes

- This is an additive change to Tran's reader.
  Mention it to Tran before merging, since shared interfaces need agreement.
- Reading one extra image per case adds a little decode time; that is acceptable for now.
  If it matters later, add an opt-in flag rather than changing defaults.
- The owner confirms on Perlmutter (one case) that `clean_paths[0]` is the unmarked goal; the schema says so but it has not been checked in this repository.
  The agent provides the command in the hand-off.

## Acceptance criteria

- New synthetic tests build an invented case directory with `clean_paths` and `goal_xy` and check both new keys.
- A test checks that an unsafe `clean_paths[0]` is rejected like other paths.
- All existing reader tests pass unchanged.
- `LOADER.md` lists the two new keys.

## Run on Perlmutter (owner)

```bash
module load pytorch/2.13.0
cd "${REPO_DIR:?}" && git pull
python3 -m unittest tests.test_dataset -v
```

Then one real case, reporting only keys, sizes and modes (no pixel values or paths):

```bash
python3 -c "import os; from dataset import GuidewireDataset as D; c = D(os.environ['DATA_DIR'], split_position=0, partition='train')[0]; print(sorted(c), c['goal_raw'].size, c['goal_raw'].mode, len(c['goal_xy']))"
```

The owner also views `goal` and `goal_raw` side by side in the notebook to confirm the raw one has no marker.
