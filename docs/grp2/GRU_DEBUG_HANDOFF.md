# ViT–GRU debugging setup and teammate handoff

## Current state — 2026-11-02

Tran requested implementation of the planned eight-case setup. The code now exists on local `tran/vit-gru-debug`, based on `0f77490`; remote publication and real-data execution remain pending. Shared documentation stays in the original checkout on `main`. The separate sibling checkout `waldo-guidewire-policy-vit-gru-debug` holds code. Main's documentation does not automatically appear on the separate debug branch; read the current notes in the main checkout until the branches are explicitly synchronized.

| File | Role |
|---|---|
| `vit_gru.py` | One-layer, 64-unit PyTorch GRU and goal-fusion classifier; 115,650 trainable parameters |
| `vit_gru_debug.py` | Fresh eight-case selection, training-only scaling, frozen extraction, bounded training and private report |
| `tests/test_vit_gru.py` | Synthetic wiring, gradients, freezing, selection/scaling and timeout checks |
| `evaluation.py`, `tests/test_evaluation.py` | Unchanged copies of the static evaluation at `070328f`; later comparison utilities |

Existing `dataset.py`, `preprocessing.py`, `vit.py` and the notebook are unchanged. No new project configuration/instruction files were added. Real data, features and outputs remain on Perlmutter. Current test evidence and Git state are recorded in [PROGRESS.md](PROGRESS.md#session-handoff--2026-11-02).

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
2. Keep `evaluation.py` as the static comparison tool. Its `latest_goal_metadata` mode already supplies latest image + goal + all historical depth/time values. Do not rebuild that baseline or rerun the completed static experiment merely to resume.
3. Your feature-role conventions and metric helpers remain useful for later evaluation. Your logistic-regression optimizer does not train the GRU. Its per-column centered scaler and validation-selected regularization also differ from this debugging run. Agree on the matched full-training protocol before comparing results.
4. The tiny runner deliberately does not call the full static-evaluation runner: it must not extract whole train/validation image partitions or select settings using validation. It uses only torch/Pillow/timm for real extraction/training; NumPy and scikit-learn are also needed to run the imported evaluation tests.
5. After the code has been explicitly reviewed and shared, run the synthetic checks below in your Perlmutter environment, then one bounded real-data check. Keep private reports on your account; report permitted aggregates to the team. One account's success does not verify another account's environment.

## Before transferring code

The new work is local; a remote `git pull` cannot retrieve it until an authorized push. Check `git log -3 --oneline` and `git status --short` in both checkouts for the exact local revisions and any remaining changes. The intended commit scopes are shared documentation on main, imported static evaluation on the debug branch, and the GRU runner/tests on the debug branch. Ask Tran immediately before committing; publishing or synchronizing branches requires explicit authorization. This guide does not itself authorize those actions.

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
