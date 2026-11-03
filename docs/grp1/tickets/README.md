# Group 1 tickets - wire-preserving CNN preprocessing

These tickets break the goal-centered wire preprocessor and its evaluation into small pieces.
Each ticket is meant to be picked up by one person or one AI agent and finished on its own.
Read this page first, then the ticket, then only the files the ticket names.

## Why this work exists

The current CNN input reuses `DINOv2Preprocessor`: the full 1065x1050 frame is resized to 518 and padded.
A 3-4 px guidewire becomes about 1.5-2 px after that resize, about 0.5 px after the ResNet stem, and about 5% of one cell in the final 17x17 grid.
The 512-d feature is an average over those 289 cells, so the background dominates it.
Static evaluation on split 0 agrees: ResNet latest + goal reached validation balanced accuracy .547, the same as metadata only (see [EVALUATION.md](../../EVALUATION.md)).

The proposed preprocessor keeps the wire intact instead:

- crop a square around `goal_xy` from the original frames at native resolution (no downscaling);
- optionally add wire-specific channels: a black top-hat (thin dark structures) and a frame difference (moving structures);
- keep the output layout identical to today's so `cnn.py` needs no change.

## Data facts these tickets rely on

Owner-reported on 2026-11-03 (recorded formally by T03):

- `goal_xy` uses the same pixel coordinates as the history frames, and the goal image shows the same view.
- The guidewire is dark and about 3-4 px wide; it is long and thin.
- History frames are black and white.
- The view does not shift between frames within a case, so frames need no alignment.
- How far the wire tip usually is from `goal_xy` is unknown (T13 measures it).

## Where code is written and where it runs

All execution happens on Perlmutter.
The laptop has neither the dataset nor enough compute, so nothing is run there, including unit tests.

| Role | Does | Does not |
| --- | --- | --- |
| Agent (laptop) | Reads code, writes code, tests and docs, self-reviews by reading | Run Python, tests, scripts, notebooks or benchmarks |
| Owner (Perlmutter) | Pulls the change, runs the ticket's commands, reports permitted output | Copy data, images or features off Perlmutter |

Git commands that only inspect or record changes are fine on the laptop.
Committing and pushing still need the owner's go-ahead each time.

## How a ticket gets done

1. **Implement.**
   The agent writes the code and its invented-input tests, then re-reads the diff against the acceptance criteria.
   Because nothing runs locally, the agent checks imports, shapes and edge cases by careful reading, not by trial runs.
2. **Hand off.**
   The agent finishes with: the files changed, the exact Perlmutter commands from the ticket's "Run on Perlmutter" section, and the expected output.
3. **Owner runs on Perlmutter.**
   The owner gets the change onto their Perlmutter clone (usually push a branch, then pull there) and runs the commands.
4. **Fix from the report.**
   The owner pastes the permitted output (test results, shapes, aggregates, error tracebacks without sensitive values).
   The agent fixes from that report and hands off again.
5. **Done** only when the owner reports the Perlmutter run passed.
   Record the code version that was tested.

Standard Perlmutter test setup, used by every ticket (T01 confirms it works):

```bash
module load pytorch/2.13.0
cd "${REPO_DIR:?Set REPO_DIR to your waldo-guidewire-policy clone}"
git pull
python3 -m unittest discover -s tests -v
```

Unit tests use invented inputs and are light enough for a login node.
Anything that loads ResNet weights on real cases, runs the probe, or benchmarks belongs in a GPU allocation.

## Ground rules

- Real images, tensors, features and notebook outputs stay on Perlmutter.
  Tests use invented images only.
- Do not change `DINOv2Preprocessor` output or Group 2's code paths.
  Existing tests must keep passing.
- Defaults of existing scripts stay as they are, so earlier results remain reproducible.
- Never use test partitions to choose anything.
  Training data chooses parameters, validation confirms, test is used once at the end.
- Match the surrounding code style: short docstrings, comments only for non-obvious logic, `unittest` tests with invented inputs.
- Record meaningful outcomes in [PROGRESS.md](../PROGRESS.md) when a ticket finishes (T18 does the final write-up).

## Tickets and order

| ID | Title | Perlmutter run | Depends on |
| --- | --- | --- | --- |
| [T01](T01-perlmutter-test-setup.md) | Perlmutter test setup | Existing tests, dependency check | - |
| [T02](T02-repo-hygiene.md) | Untrack `__pycache__` and fix `.gitignore` | Existing tests | - |
| [T03](T03-record-data-facts.md) | Record owner-reported data facts | None (docs) | - |
| [T04](T04-reader-goal-xy-and-raw.md) | Expose `goal_xy` and unmarked goal from the reader | Tests + one real case | T01 |
| [T05](T05-preparation-metadata-refactor.md) | Make `prepare_case` metadata preprocessor-specific | Tests | T01 |
| [T06](T06-black-tophat.md) | Fast black top-hat filter | Tests | T01 |
| [T07](T07-goal-centered-crop.md) | Goal-centered native-resolution crop | Tests | T01 |
| [T08](T08-wire-preprocessor-raw.md) | `WirePreprocessor`, raw mode | Tests | T04, T05, T07 |
| [T09](T09-enhanced-channels.md) | Enhanced channels: top-hat and difference | Tests | T06, T08 |
| [T10](T10-context-view.md) | Optional full-view context input | Tests | T08 |
| [T11](T11-cli-integration.md) | Preprocessor flags in `cnn.py` and `cnn_diagnostics.py` | Tests + one real case | T08 (T09, T10 when present) |
| [T12](T12-latency-benchmark.md) | Preprocessing latency benchmark | GPU node | T09 |
| [T13](T13-inspection-notebook-cells.md) | Notebook cells: channel check and crop-size boxes | Notebook | T09 |
| [T14](T14-port-static-probe.md) | Bring the static probe to Group 1 with preprocessor options | Tests | T08, owner approval |
| [T15](T15-probe-controls.md) | Probe controls: shuffled labels, random center, channel ablation, all splits | Tests | T14 |
| [T16](T16-grouped-inner-cv.md) | Episode-grouped inner CV for parameter selection | Tests | T14 |
| [T17](T17-experiment-ladder.md) | Run the evaluation ladder | GPU node, full ladder | T11-T16 |
| [T18](T18-docs-and-progress.md) | Documentation and progress write-up | None (docs) | T17 |

T01-T03 and T06-T07 can be implemented in parallel.
Batching several tickets into one Perlmutter run is fine, as long as each ticket's commands and results are reported separately.
T10 is optional: build it only if T13 shows the tip is often outside reasonable crop sizes.

## Evaluation ladder (what "it works" means)

All steps run on Perlmutter.

| Step | Question | Ticket |
| --- | --- | --- |
| Unit tests | Does the code do what we think? | T04-T11, T14-T16 |
| Visual check | Is the wire bright and the background dark in the top-hat channel on real frames? | T13 |
| Feature diagnostics | Do the 512-d features respond near the wire? | T17 (uses `cnn_diagnostics.py`) |
| Task probe | Do the features predict ADVANCE/RETRACT better than .547 by more than noise? | T14-T17 |
| Full pipeline | Does the gain survive inside the CNN-GRU? | Later, not in these tickets |

On 80 validation cases (34 RETRACT, 46 ADVANCE) the 95% interval on balanced accuracy is about +/-0.11 for one split, so a single-split gain of a few points is noise.
T15 adds the controls that separate a real gain from a lucky one.
