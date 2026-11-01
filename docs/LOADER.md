# Original-resolution case loader

`dataset.py` provides the first data-reading layer. It opens original images without resizing, cropping, color conversion, normalization, augmentation, or tensor conversion. Only Pillow and Python's standard library are needed. A shared ML environment has not yet been adopted; dependency versions are not pinned.

Real data access runs only on Perlmutter. The laptop tests use entirely invented images and records.

For how raw reading differs from model preparation and the incoming TODO outlines, see [how the files fit together](APPROACHES.md#how-the-current-files-fit-together).

## Select cases

```python
import os
from dataset import GuidewireDataset

cases = GuidewireDataset(
    os.environ["DATA_DIR"],
    split_position=0,
    partition="train",
)
case = cases[0]
```

This example selects the first supplied split object and its training partition for inspection. It is not an agreed experimental split choice. `split_position` means the zero-based position in `splits.json`, not its `fold` field. Both the position and partition must be explicitly supplied. The loader selects records using their zero-based positions in `cases.jsonl` and preserves the partition list's order.

## Returned case

- `goal`: one detached Pillow image from `paths[0]`, including the supplied target marking.
- `history`: six detached Pillow images from `paths[1:]`, in chronological order.
- `depth_relative_mm`: six original numbers, aligned with history.
- `history_offsets_s`: six original numbers, aligned with history.
- `label`: original action label (`0` RETRACT, `1` ADVANCE), kept separate from image inputs.
- `case_id`, `episode`, `row_index`: tracing information, not predictive inputs.

Image size, mode, and pixels are preserved. Images are decoded on request, not all loaded at construction. Files are closed before images are returned. Missing or unreadable images cause an error instead of silently removing cases. Path checks prevent reading outside the dataset root. Metadata JSON is read at construction, but the supplied files are never modified.

## Scope of the first version

The new [before-GRU evaluation runner](EVALUATION.md) reuses this reader unchanged, extracts train/validation features in memory, and checks partition episode separation before decoding. It is not a replacement DataLoader and does not alter supplied splits. Local evaluation tests use invented arrays only.

Request cases individually with `cases[0]`, `cases[1]`, and so on. One-case loading and notebook display have been demonstrated on Perlmutter. This is a simple reader, not yet a PyTorch `DataLoader`: batching, worker processes, model preprocessing, and tensor stacking remain deferred. No cases are shuffled.

## Local checks — laptop, invented data only

`tests/test_dataset.py` is a code-test file, not another dataset or an evaluation split. It temporarily creates tiny invented PNGs and JSON records, then checks that the loader returns the correct images and matching metadata. It uses no real cases and does not create new train/validation/test partitions. The supplied `test` partition is different: it is reserved for evaluating a model.

With Pillow available in the chosen local interpreter, run from the repository root:

```sh
python3 -m unittest discover -s tests -v
```

The checks cover record selection, goal/history ordering, metadata alignment, original pixels/mode/size (including color), and failure on unsafe or unreadable paths. All six passed in owner-run Perlmutter execution at `9221135`; one real case also decoded successfully. See the [Group 2 milestone](grp2/PROGRESS.md) for results and limits. This does not establish visual correctness or full-dataset decoding.

## Real-data handoff — Perlmutter only

The laptop files must first be transferred through the owner's authorized Git handoff. They are not automatically present in the SSH VS Code clone. No commit or push is implied by these instructions.

Once the code is present, use an environment where `python3` can import Pillow. The owner already confirmed Python 3.12.13 and Pillow 12.3.0 after loading `pytorch/2.13.0`; full ML compatibility remains unverified. From the Perlmutter repository root, with the existing `DATA_DIR` set:

```sh
python3 dataset.py --split-position 0 --partition train --count 1
```

This decodes one selected case through the new loader and reports image sizes/modes and metadata lengths. It does not repeat full dataset validation, choose an evaluation protocol, or display sensitive records. Successful output should report seven images, six depths, and six time offsets. Original dimensions are `(1065, 1050)`; the inspected case returned RGB images. Other cases' modes have not been checked through this reader. The split position is an inspection example, not an evaluation decision.

The owner also displayed the goal and six history images in the Perlmutter notebook. Successful decoding/display does not prove visual or semantic correctness; those assessments remain open. Do not copy images or sensitive errors off the system. Model-ready preprocessing and evaluation remain undecided.

## Display the images — notebook cells on Perlmutter

Open `data_inspect.ipynb` in SSH-connected VS Code and select the Python interpreter from the existing `pytorch/2.13.0` module. Run **Load data**, then **Sample plots**. The first cell displays the separate goal; the second displays the six ordered history images with depth/time annotations. Paste/run Python inside notebook cells, not the Bash terminal.

The notebook uses the same raw reader and portable scratch layout. It applies no preprocessing. The shared file contains no saved outputs; clear outputs before committing and keep sensitive visual results on Perlmutter. Remaining notebook sections are inspection prompts, not implemented analyses.
