# T01 - Perlmutter test setup

**Status:** Todo
**Code written:** nothing to write; the agent only prepares the commands
**Tested:** Perlmutter, by the owner (login node is enough)
**Depends on:** nothing
**Files:** none in the repository

## Why

Every later ticket is verified by the owner running the test suite on Perlmutter.
Before that, confirm the standard command works and that later tickets' dependencies are available there.
Nothing runs on the laptop: it has no dataset and not enough compute.

## Scope

The owner runs, on Perlmutter:

```bash
module load pytorch/2.13.0
cd "${REPO_DIR:?Set REPO_DIR to your waldo-guidewire-policy clone}"
git pull
python3 -m unittest discover -s tests -v
python3 -c "import torch, torchvision, PIL, numpy; print(torch.__version__, torchvision.__version__, PIL.__version__, numpy.__version__)"
python3 -c "import sklearn; print(sklearn.__version__)"
```

- If `sklearn` is missing from the module, note it.
  T14-T16 need it; Sabrina's static evaluation used a scratch venv on top of the module (see [EVALUATION.md](../../EVALUATION.md)), which is the likely fix.
- Do not reinstall or replace the module environment.

## Acceptance criteria

- The owner reports the test summary (counts of passed/failed) and the printed versions.
- Any existing test failure is recorded as a separate issue, not ignored.
- The working test command is the one in the [tickets README](README.md); update it there if it had to change.
