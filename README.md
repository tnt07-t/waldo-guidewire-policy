# waldo-guidewire-policy

CS 5470 final project: predict the next operator-action proxy (ADVANCE or RETRACT) from a target-marked goal reference, six history images, and relative depth information. Labels describe inferred operator behavior, not verified optimal actions.

The approach is open-ended: explore specialized smaller models that exceed the baseline's accuracy while reducing decision latency relative to the tested frontier model. The proposed architectures are starting directions, not an exhaustive list. Prior results and unresolved benchmark details are recorded in the approach notes.

## Project notes

- [Documentation guide](docs/README.md): reading order for teammates and a separate entry point for AI agents.
- [Dataset](docs/DATASET.md): schema, restrictions, and preprocessing/split verification.
- [Approaches](docs/APPROACHES.md): team proposal, dependencies, decisions, shared working/logging process, and where to resume.
- [ViT–GRU debugging handoff](docs/grp2/GRU_DEBUG_HANDOFF.md): implemented eight-case setup, testing commands, Sabrina's steps and remaining Perlmutter verification. Code is local to the separate debug worktree; shared notes stay on main.
- [Static evaluation](docs/EVALUATION.md): Sabrina's frozen-feature classifiers and reported development results; evaluation code/tests are imported into the separate `tran/vit-gru-debug` worktree for reuse.
- [Raw case loader](docs/LOADER.md): usage; six synthetic tests, one real-case decode, and goal/history display passed on Perlmutter. Visual/semantic assessment remains open; the raw reader itself applies no preprocessing.
- [Initial ViT preparation](docs/grp2/VIT_TRIAL.md): full-view DINOv2 tensors, previews, and inference-only encoder; owner-run pretrained CPU checks are recorded in Group 2 progress.
- [ViT feature diagnostics](docs/grp2/VIT_DIAGNOSTICS.md): bounded inspection of whether extracted ViT features look usable before temporal-model work.
- [CNN encoder trial](docs/grp1/CNN.md): frozen ResNet-18 frame features in `cnn.py`; owner-run one-case Perlmutter smoke check passed.
- [CNN 512-d diagnostics](docs/grp1/CNN_DIAGNOSTICS.md): separate occlusion/marker/temporal probes on the vectors a GRU would see; owner-run one-case GPU diagnostics passed, then correctness fixes were applied (Perlmutter re-run pending). `data_inspect.ipynb` includes original-image inspection cells, with no saved outputs.
- Group progress: [Group 1 — Andrew and Suhani](docs/grp1/PROGRESS.md) and [Group 2 — Tran and Sabrina](docs/grp2/PROGRESS.md). Shared contracts stay in the approach notes; each group records its own plans and outcomes.

The official rubric has not been supplied. Team proposals and unapproved suggestions are not course requirements.

## Where work runs

Repository: <https://github.com/tnt07-t/waldo-guidewire-policy>.

Code and documentation may be edited on a laptop. Dataset access and inspection must stay on Perlmutter. A manually opened SSH terminal does not make an assistant's tools remote; verify the execution location before running commands.

Development handoff: edit and check changes in the laptop clone, then share reviewed changes through GitHub. Teammates pull the changes into their Perlmutter clones and run data-dependent checks there. Record which code version was tested when available; local checks alone do not establish that the data workflow works remotely.

Each teammate uses their own account and scratch directory. Keep personal usernames, absolute personal paths, environment notes, and setup progress in a private note outside the repository. Shared commands use `$PSCRATCH`, `$DATA_DIR`, or placeholders.

Use the same setup procedure and relative dataset layout for everyone; `$PSCRATCH` resolves to each person's own storage. A successful check on one account does not establish another teammate's setup status. The shared Python environment remains to be selected.

## Portable data setup

**Run on Perlmutter**, in your own terminal. `$PSCRATCH` is your personal scratch directory. Confirm it is set before proceeding:

```sh
printf 'PSCRATCH=%s\n' "$PSCRATCH"
```

For a new copy, choose an unused destination under your scratch directory. The commands below stop if that destination already exists, avoiding an accidental nested copy. `cp -a` copies the package while preserving its internal hard links.

```sh
(
  set -eu
  : "${PSCRATCH:?PSCRATCH must be set}"
  export DATA_DIR="$PSCRATCH/CS5470_FinalProject/shorthorizon2"
  if [ -e "$DATA_DIR" ]; then
    printf 'Destination already exists; inspect it before copying: %s\n' "$DATA_DIR"
    exit 1
  fi
  mkdir -p "$PSCRATCH/CS5470_FinalProject"
  cp -a /global/cfs/cdirs/m4999/studentdata/shorthorizon2 "$DATA_DIR"
)
```

For an existing copy, **run on Perlmutter** in each new terminal session:

```sh
export DATA_DIR="${PSCRATCH:?PSCRATCH must be set}/CS5470_FinalProject/shorthorizon2"
test -d "$DATA_DIR" && printf 'Dataset directory exists\n'
```

`export` makes the location available to programs in that terminal session; it does not persist to future sessions. If paths fail, check actual directory names rather than treating transcript escapes such as `\_` as literal characters.

## Environment status

The owner confirmed Python 3.12.13 and Pillow 12.3.0 after loading the candidate `pytorch/2.13.0` module on Perlmutter. Subsequent Group 2 checks verified pretrained DINOv2 CPU extraction using a scratch venv with timm 1.0.30; Group 1 reports a one-case CNN GPU diagnostic run. Tran's GPU path and team-wide environment adoption remain open; Sabrina separately reports a completed GPU static evaluation. See the approach notes for evidence and next steps. Do not repeat the completed environment inventory or basic dataset validation.

## Project Environment Setup
Setup the conda environment for the project
```
module load conda
```

For first time, create environment with `environment.yml` file. If conda environment already created, skip this step. 
```
conda env create -f environment.yml
```

Activate and update environment
```
conda activate final_project_env
conda env update -f environment.yml --prune
```

## Lightweight collaboration

Teammates and AI assistants should start with the [documentation guide](docs/README.md), which identifies the relevant project references for each audience.

- Keep confirmed facts, team proposals, unapproved suggestions, and open questions labeled separately.
- Update the relevant document after material progress; record dated decisions once in `APPROACHES.md` rather than duplicating logs.
- Before parallel implementation, agree on input/output interfaces, preprocessing, and evaluation. Individual ownership is still open.
- Keep sensitive data and derived sensitive artifacts on Perlmutter. Never put credentials in documentation.

The repository includes shared notes, group progress directories, a raw-case reader (`dataset.py`), and synthetic loader tests. Raw loading, initial preparation previews, and pretrained DINOv2 CPU extraction for one and two cases have passed owner-run checks on Perlmutter. Group 2 ran a one-frame feature-sensitivity check; its inconclusive task-level result is recorded in the group log. The [frozen ViT–GRU debugging setup](docs/grp2/GRU_DEBUG_HANDOFF.md) is now implemented in the separate debug worktree and passed all 44 synthetic tests. Next: an authorized code handoff, then the bounded eight-training-case Perlmutter check. Notebook CUDA availability is confirmed; explicit diagnostic device and inference memory/runtime measurements remain unreported. Sabrina's separate static development results are recorded in the evaluation reference; they do not establish GRU performance or full-policy comparative latency.
