# T12 - Preprocessing latency benchmark

**Status:** Todo
**Code written:** laptop (no execution)
**Tested:** Perlmutter, by the owner, in a GPU allocation
**Depends on:** T09
**Files:** new `bench_preprocessing.py`, `docs/grp1/CNN.md`

## Why

The project goal includes lower decision latency.
Preprocessing already dominates today's ResNet timing (about 29 of 34 ms per case on Perlmutter, from [EVALUATION.md](../../EVALUATION.md)).
A rough one-off prototype on invented images suggested crop 256 + top-hat + median difference costs about a quarter of the current full-view prep; the real numbers must come from Perlmutter.

## Scope

- A script that times, per configuration:
  - full case preparation (goal + six frames);
  - incremental preparation: one new frame, with the goal and previous five frames already prepared (the live-use cost per decision);
  - ResNet encoding of the prepared tensors, reported separately.
- Configurations: `dinov2` at 518; `wire` raw and enhanced at crop 256 and 384; top-hat on CPU and on GPU.
- Warm up first, take the median and p95 over at least 20 repeats, and call `torch.cuda.synchronize()` around GPU timings.
- Default input is invented 1065x1050 images; `--data-dir` with `--split-position` uses one real training case.
- Print device, torch version and CPU/GPU names.
  Save nothing except optional JSON under a user-given `--output-dir`.
- Keep the default run short (well under a minute on a GPU node).

## Out of scope

- End-to-end policy latency (no policy exists yet).

## Acceptance criteria

- Output is a small table that can be pasted into `CNN.md` as permitted aggregates.
- The owner reports the table for both invented images and one real case.

## Run on Perlmutter (owner)

```bash
module load pytorch/2.13.0
cd "${REPO_DIR:?}" && git pull
export TORCH_HOME="${PSCRATCH:?}/model-cache/torch"
python3 bench_preprocessing.py
python3 bench_preprocessing.py --data-dir "$DATA_DIR" --split-position 0
```
