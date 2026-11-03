# T05 - Make `prepare_case` metadata preprocessor-specific

**Status:** Todo
**Code written:** laptop (no execution)
**Tested:** Perlmutter, by the owner
**Depends on:** T01
**Files:** `preprocessing.py`, `tests/test_preprocessing.py`

## Why

`DINOv2Preprocessor.prepare_case` hardcodes the DINOv2 checkpoint name and fixed preparation fields in its `preparation` record.
A second preprocessor must record its own settings, and the CNN must not be labeled with a DINOv2 checkpoint.

## Scope

- Split the shared work of `prepare_case` (six-step validation, metadata deepcopy, stacking goal/history tensors, previews, geometry) from the preprocessor-specific `preparation` record.
  A small base class or a module-level helper are both fine; pick whichever keeps the code shortest.
- `DINOv2Preprocessor` keeps producing exactly the same dictionary as before, including every `preparation` field.

## Out of scope

- Any change to DINOv2 resize, padding or normalization.
- Adding the wire preprocessor itself (T08).

## Acceptance criteria

- A new test prepares an invented case and compares the full output (tensors with `torch.equal`, metadata and `preparation` dict) against the pre-refactor behavior.
  Since nothing runs on the laptop, keep a frozen copy of the old `prepare_case` inside the test file as the reference and compare against it, rather than hand-typed expected values.
- All existing preprocessing and ViT tests pass.

## Run on Perlmutter (owner)

```bash
module load pytorch/2.13.0
cd "${REPO_DIR:?}" && git pull
python3 -m unittest tests.test_preprocessing tests.test_vit -v
```
