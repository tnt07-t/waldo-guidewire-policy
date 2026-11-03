# T11 - Preprocessor flags in `cnn.py` and `cnn_diagnostics.py`

**Status:** Todo
**Code written:** laptop (no execution)
**Tested:** Perlmutter, by the owner: tests, then one real case in a GPU allocation
**Depends on:** T08 (include T09 and T10 options once they exist)
**Files:** `preprocessing.py`, `cnn.py`, `cnn_diagnostics.py`, `tests/test_cnn_diagnostics.py`, `docs/grp1/CNN.md`, `docs/grp1/CNN_DIAGNOSTICS.md`

## Why

The smoke check and diagnostics must be able to run on either preprocessor so the two can be compared on the same case.

## Scope

- Add one factory, `build_preprocessor(args)` or similar, used by both scripts so the flags stay identical.
- Flags:
  - `--preprocessor {dinov2,wire}` (default `dinov2`, keeping today's behavior);
  - `--crop-size` (default 256), `--channels {raw,enhanced}` (default `raw`);
  - `--tophat-size` (default 11), `--difference {median,previous}` (default `median`);
  - `--context-size` if T10 exists.
- The existing `--size` flag keeps applying to `dinov2` only; reject `--size` with `wire` rather than ignoring it.
- In `cnn_diagnostics.py`, the goal-marker probe needs the marked goal.
  With `wire`, either run the marker probe on the marked goal crop or report it as `unavailable`; do not silently use the unmarked goal.
  Coordinate the choice with Suhani, who owns the diagnostics.
- Occlusion patch and stride defaults (74) were sized for 518 inputs.
  With a 256 crop, scale them (for example 32/32) and print the values used.

## Out of scope

- Changing what the probes measure.

## Acceptance criteria

- Running either script without new flags gives the same output as before.
- Invented-input tests cover the factory, including the rejected `--size` with `wire`.
- Docs show the new commands with a "Run on Perlmutter" label.

## Run on Perlmutter (owner)

```bash
module load pytorch/2.13.0
cd "${REPO_DIR:?}" && git pull
python3 -m unittest discover -s tests -v
```

Then one training case, in a GPU allocation, comparing old and new paths:

```bash
export TORCH_HOME="${PSCRATCH:?}/model-cache/torch"
python3 cnn.py --split-position 0 --partition train
python3 cnn.py --split-position 0 --partition train --preprocessor wire --crop-size 256
python3 cnn.py --split-position 0 --partition train --preprocessor wire --crop-size 256 --channels enhanced
```

Report the printed device, shapes and pass line for each.
