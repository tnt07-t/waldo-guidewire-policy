# ViT–GRU debugging setup and teammate handoff

## Current state — 2026-11-03

Tran requested implementation of the planned eight-case setup. The code now exists on local `tran/vit-gru-debug`, based on `0f77490`; publication and the first real eight-case GPU debugging check are complete. The run passed at `b9d1bdf`; see [results and the current session handoff](PROGRESS.md#session-handoff--2026-11-03). The commands below describe the completed check, not a request to repeat it. Shared documentation stays in the original checkout on `main`. The separate sibling checkout `waldo-guidewire-policy-vit-gru-debug` holds code. Main's documentation does not automatically appear on the separate debug branch; read the current notes in the main checkout until the branches are explicitly synchronized.

| File | Role |
|---|---|
| `vit_gru.py` | One-layer, 64-unit PyTorch GRU and goal-fusion classifier; 115,650 trainable parameters |
| `vit_gru_debug.py` | Fresh eight-case selection, training-only scaling, frozen extraction, bounded training and private report |
| `tests/test_vit_gru.py` | Synthetic wiring, gradients, freezing, selection/scaling and timeout checks |
| `guidewire/evaluation/probes.py`, `tests/test_evaluation.py` | Unchanged copies of the static evaluation at `070328f`; later comparison utilities |

Existing `guidewire/data/dataset.py`, `guidewire/data/preprocessing.py`, `guidewire/encoders/vit.py` and the notebook are unchanged. No new project configuration/instruction files were added. Real data, features and outputs remain on Perlmutter. Current test evidence and Git state are recorded in [PROGRESS.md](PROGRESS.md#session-handoff--2026-11-03).

## Implemented experiment

- Split position 0, training partition only; select four cases per label with seed 0. Validation/test are not used for fitting, scaling or selection.
- Fit one population standard deviation for depth and one for time across all six steps of all training cases. Divide without subtracting a mean; use scale 1 for a constant channel. Preserve signs, zeros and step alignment.
- Reuse existing full-view 518-square DINOv2 preparation and 384-value features. Extract only the eight selected cases, once, one image per encoder call; retain features in memory.
- Feed `[B,6,384]` image features plus `[B,6,2]` scaled depth/time to the GRU. Reset hidden state per case. Concatenate the final 64-value state with the 384-value goal, then `Linear(448,64) → ReLU → Linear(64,2)`. Scores use RETRACT=0, ADVANCE=1.
- Freeze encoder parameters, keep evaluation mode and check both parameters and buffers are unchanged before/after head training. Convert inference outputs to ordinary detached tensors outside inference mode so backward works.
- Fresh head and Adam optimizer each invocation; full batch of eight, learning rate 0.001, no extra regularization. Stop after 300 updates or 120 seconds of head work. The head clock is checked between updates; one in-flight update and final measurement can finish after that deadline.
- A parent process stops the worker at 600 seconds overall, including imports, pretrained loading and extraction, with at most five seconds for termination before forced kill. Limits can be reduced, not increased, through the public CLI.
- Save configuration, scales, private selected row indices, loss/accuracy, update count, gradient/update checks, elapsed time, peak allocated GPU memory, hardware/software versions and source-file hashes in `debug_summary.json`. No image/feature cache or model checkpoint is saved.

## Static evaluation reuse and coordination

1. Read this handoff and the [GRU trial rationale](GRU_TRIAL.md). Check for any unpublished overlapping GRU/trainer work before making parallel edits. Tran is preparing this ViT–GRU path; your existing static evaluation is retained.
2. Keep `guidewire/evaluation/probes.py` as the static comparison tool. Its `latest_goal_metadata` mode already supplies latest image + goal + all historical depth/time values. Do not rebuild that baseline or rerun the completed static experiment merely to resume.
3. Your feature-role conventions and metric helpers remain useful for later evaluation. Your logistic-regression optimizer does not train the GRU. Its per-column centered scaler and validation-selected regularization also differ from this debugging run. Agree on the matched full-training protocol before comparing results.
4. The tiny runner deliberately does not call the full static-evaluation runner: it must not extract whole train/validation image partitions or select settings using validation. It uses only torch/Pillow/timm for real extraction/training; NumPy and scikit-learn are also needed to run the imported evaluation tests.
5. After the code has been explicitly reviewed and shared, run the synthetic checks below in your Perlmutter environment, then one bounded real-data check. Keep private reports on your account; report permitted aggregates to the team. One account's success does not verify another account's environment.

## Before transferring code

The tiny setup and full-training runner/tests are published on `tran/vit-gru-debug`; the full-training implementation is recorded at `69fd063`. Shared November 3 results and handoff updates are published on main and reconciled into `static-evaluation`. Check the active checkout's Git log/status before updating Perlmutter. The full-training implementation has not yet been confirmed available in the remote compute checkout. Preserve notebook edits before synchronization; publication does not itself start training.

Before any future pull that could affect the remote notebook, save it in the notebook UI and preserve an independent copy **on Perlmutter**. From the Perlmutter repository directory:

```sh
backup_dir=$(mktemp -d "${PSCRATCH:?}/waldo-notebook-backup.XXXXXX")
cp -p data_inspect.ipynb "$backup_dir/"
```

`mktemp` creates a new uniquely named backup directory; `cp -p` preserves the saved notebook and its file timestamps. This does not capture unsaved browser edits. Never copy its outputs to the laptop or GitHub. The GRU setup itself does not require editing the notebook.

## Synthetic checks — laptop or Perlmutter, no real data

Run from the debug checkout with its Python environment active:

```sh
python -B -m unittest discover -s tests -p 'test_vit_gru.py' -v
python -B -m unittest discover -s tests -p 'test_evaluation.py' -v
python -B vit_gru_debug.py --help
```

`-B` avoids writing Python bytecode files. `unittest discover` finds the named test file and runs invented-input checks; it does not load pretrained weights or real cases. The timeout test intentionally checks that an existing output directory is rejected, so an argparse error message can appear within an otherwise passing test run.

The GRU tests check expected shapes/parameter count, gradient flow through every history step and goal/metadata inputs, case-independent hidden state, metadata order, deterministic balanced selection, training-only scaling, inference-to-training conversion, finite optimizer updates, unchanged encoder state and bounded execution. Synthetic tiny fitting verifies implementation behavior, not real task performance.

## Eight-case check — Perlmutter only

Run in an allocated compute-node shell using the working torch/timm environment. Do not perform extraction/training on a login node. Reuse the allocation/environment guidance in [EVALUATION.md](../EVALUATION.md#compute-allocation-and-cache-troubleshooting); use your own available account. No new package installation is prescribed here.

Before execution, confirm the terminal is actually on Perlmutter and inside your allocation:

```sh
hostname
printf 'SLURM_JOB_ID=%s\nPSCRATCH=%s\n' "$SLURM_JOB_ID" "$PSCRATCH"
git branch --show-current
git rev-parse HEAD
git status --short
```

The branch should contain the reviewed GRU files. Record the actual revision/dirty state rather than assuming a laptop update reached Perlmutter. Set paths in that shell; replace the output suffix with a new name for each intentionally authorized run:

```sh
export DATA_DIR="${PSCRATCH:?}/CS5470_FinalProject/shorthorizon2"
export HF_HOME="$PSCRATCH/waldo-cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$PSCRATCH/waldo-cache/torch"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME"
python -B vit_gru_debug.py \
  --device cuda \
  --output-dir "$PSCRATCH/waldo-gru-debug/tiny-01"
```

The script obtains `DATA_DIR` from the environment. Data/output paths must be under personal scratch; output must be a new directory outside the dataset. Public weights may download into the scratch cache; images are never uploaded. `--device cpu` is an explicit alternative on an allocated node, but its runtime may exhaust the same limits. Do not increase limits automatically or start another experiment after a failure.

## Inspect and report

Inspect `debug_summary.json` locally on Perlmutter. It contains selected row indices and derived scales, so keep the whole file private. Share only permitted aggregates:

- Exit/status and stop reason; actual update count and whether the overall limit was reached.
- Initial/final training loss and accuracy on the same eight cases.
- Finite gradients, changed head weights, unchanged encoder parameters/buffers.
- Extraction/head/overall duration, peak allocated CUDA bytes and GPU model.
- Code revision/dirty status and relevant package versions; source hashes identify uncommitted code more precisely.

Expected structural results are eight balanced cases, 115,650 trainable parameters, finite gradients, changed head weights and unchanged encoder state. Falling loss and fitting all eight examples are useful debugging evidence, not a guaranteed outcome within the cap. `memorized_eight` is only a convenience flag for perfect training accuracy plus lower loss; it is not an acceptance criterion for scientific usefulness.

Exit 0 means the bounded loop completed with at least one checked update. It does not promise perfect fitting. `no_updates`, `worker_failed` or `overall_time_limit` indicate an incomplete check; do not present their partial results as successful training. A timeout can produce only a status/elapsed-time report if training never reached measurement. Preserve that report and inspect the private terminal error before proposing a fix.

## Next steps after the check

1. Record the permitted aggregates in Group 2 progress, including exact code/environment and any failures. Do not repeat manual masking or completed basic loader/CPU checks without a new reason.
2. If the bounded check fails, investigate input alignment, finite values, extraction/backward compatibility and optimization. Propose a specific fix; do not silently widen budgets, add validation tuning or reject the encoder based on tiny fitting alone.
3. If it works, agree on the full-training schedule, validation metric/checkpoint rule, regularization and comparable tuning budgets. Initialize a fresh GRU/classifier; never continue the tiny-fit weights into the development experiment.
4. Compare 64 versus 128 hidden units during full training/validation. Prioritize the static-evaluation latest-image + goal + matching historical-metadata baseline, with an explicit matched protocol. Its old development score is context, not an automatically matched comparator.
5. Expand to the planned four-model comparison later. Keep test partitions reserved and agree on the five supplied splits' reporting protocol before final evaluation. End-to-end latency must include preparation, encoding and the head; the tiny runner's training/extraction timing is not decision latency.

Tiny fitting does not establish generalization, goal use, visual-history benefit or optimal actions. Full training, expanded comparisons, persistent feature caches and fine-tuning remain outside this setup milestone.

## Full-training setup — 2026-11-03

The separately implemented `vit_gru_train.py` uses the [agreed settings](GRU_TRIAL.md#agreed-first-full-training-setup--2026-11-03). Code/tests are published at `69fd063` on `tran/vit-gru-debug`; shared documentation updates are published on main. Setup passed local checks; no real full run or GPU allocation has occurred. The old eight-case commands above are historical and must not be rerun.

Before allocation, run in the debug worktree's existing environment (laptop synthetic inputs only, or Perlmutter):

```sh
python -B -m unittest discover -s tests -v
python -B vit_gru_train.py --help
```

The new tests check actual loss reduction and fresh initialization, partial batches, validation isolation, known confusion counts, training-only scaling, episode separation, tied-checkpoint retention/early stopping, prediction-preserving reload, nonfinite inputs, work/time/storage limits, and invented PNGs through the real loader/preprocessor with an injected encoder. No pretrained download or real data is used.

For a separately initiated real run, first preserve remote notebook work, update the Perlmutter checkout to the published `69fd063` through an authorized update, and verify imports/version and synthetic tests before requesting GPU time, then confirm the shell is on an allocated Perlmutter compute node. Reuse the existing environment/cache instructions; no new installation is required. The runner requires `SLURM_JOB_ID`, `PSCRATCH`, a valid `DATA_DIR` under scratch, and a new output directory outside the dataset. Fixed public CLI limits cannot be increased.

```sh
python -B vit_gru_train.py \
  --device cuda \
  --output-dir "$PSCRATCH/waldo-gru-training/split0-seed0-01"
```

Use at most the agreed 15-minute allocation and release it immediately after success, failure or timeout; the worker supervisor does not cancel the Slurm allocation. Do not automatically retry. All output stays on Perlmutter: `run_context.json`, `history.json`, `summary.json`, `best.pt`, `latest.pt`. Public weight cache storage is separate from the 20 MiB run-output cap; reuse the existing cache. No persistent image-feature cache is created.

Reload on Perlmutter in the same code environment:

```python
from vit_gru_train import load_checkpoint
head, saved = load_checkpoint(output_dir / "best.pt", device="cuda")
scales = saved["context"]["metadata_scale_depth_time"]
```

`output_dir` is a `pathlib.Path` to the private run. The loaded head expects the same prepared frozen features and metadata divided by the saved scales. This API loads for evaluation, not continued training. The runner already checks reloaded best-checkpoint predictions against the saved in-memory reference.

Report only permitted aggregates: actual revision/dirty state, stop reason, completed epochs/updates, best epoch and validation metrics, extraction/process timing, peak CUDA allocated bytes, encoder unchanged and reload verification. Keep complete reports, row indices, scales and weights private. A successful run is development evidence, not proof of temporal usefulness or decision latency; test remains reserved.
