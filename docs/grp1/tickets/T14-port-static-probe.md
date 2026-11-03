# T14 - Bring the static probe to Group 1 with preprocessor options

**Status:** Todo, needs owner approval before starting
**Code written:** laptop (no execution)
**Tested:** Perlmutter, by the owner
**Depends on:** T08, approval from Sabrina/Tran to reuse `evaluation.py`
**Files:** `evaluation.py`, `tests/test_evaluation.py`, `requirements-eval.txt` (from `origin/static-evaluation`, commit `070328f`), `docs/EVALUATION.md`

## Why

Sabrina's frozen-feature logistic probe is the main test of whether a preprocessor helps the task.
It lives on the `static-evaluation` branch, not on `main`, and it always uses `DINOv2Preprocessor`.
The ResNet latest + goal baseline there scored .547 validation balanced accuracy on split 0.

## Scope

- With approval, bring `evaluation.py`, its tests and `requirements-eval.txt` onto a Group 1 branch unchanged first (one commit), so later diffs are reviewable.
- Then add preprocessor selection using the T11 factory and flags.
  `collect` currently takes one `prepare`; allow running one preprocessor config per invocation and record its `preparation` record in `metrics.json`.
- Keep `--encoders dinov2` working only with the `dinov2` preprocessor; reject the combination otherwise.
- Keep every existing default, metric, C selection and bootstrap unchanged so results stay comparable with Sabrina's run.

## Out of scope

- Controls and inner CV (T15, T16).
- Changing the classifier.

## Acceptance criteria

- Existing `tests/test_evaluation.py` passes unchanged.
- A new invented-input test runs `collect` with a `WirePreprocessor` and a non-pretrained `FrameEncoder` and checks feature shapes.
- `docs/EVALUATION.md` gains a short "Preprocessor options" note.

## Run on Perlmutter (owner)

Needs scikit-learn on Perlmutter (T01 checks this).

```bash
module load pytorch/2.13.0
cd "${REPO_DIR:?}" && git pull
python3 -m unittest tests.test_evaluation -v
```
