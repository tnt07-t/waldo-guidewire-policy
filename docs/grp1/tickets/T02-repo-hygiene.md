# T02 - Untrack `__pycache__` and fix `.gitignore`

**Status:** Todo
**Code written:** laptop (git and `.gitignore` only; no Python is run)
**Tested:** Perlmutter, by the owner
**Depends on:** nothing
**Files:** `.gitignore`, tracked `__pycache__/*.pyc` and `tests/__pycache__/*.pyc`

## Why

Compiled Python files were committed in `a42a92c`.
They change on every run, create noisy diffs, and can conflict between teammates' Python versions.

## Scope

- Add `__pycache__/` and `*.pyc` to `.gitignore`, keeping the existing personal-docs entries.
- Remove the tracked `.pyc` files from the index with `git rm --cached` (files on disk may stay).

## Out of scope

- Any other ignore rules or repository restructuring.

## Acceptance criteria

- `git ls-files | grep pycache` prints nothing.
- After the owner runs the tests on Perlmutter, `git status` there shows no new files.
- The change is its own commit, separate from feature work.

## Run on Perlmutter (owner)

```bash
module load pytorch/2.13.0
cd "${REPO_DIR:?}" && git pull
python3 -m unittest discover -s tests -v
git status --short
```
