# waldo-guidewire-policy

CS 5470 final project: predict the next operator-action proxy (ADVANCE or RETRACT) from a target-marked goal reference, six history images, and relative depth information. Labels describe inferred operator behavior, not verified optimal actions.

The approach is open-ended: explore specialized smaller models that exceed the baseline's accuracy while reducing decision latency relative to the tested frontier model. The proposed architectures are starting directions, not an exhaustive list. Prior results and unresolved benchmark details are recorded in the approach notes.

## Project notes

- [Documentation guide](docs/README.md): reading order for teammates and a separate entry point for AI agents.
- [Dataset](docs/DATASET.md): schema, restrictions, and preprocessing/split verification.
- [Approaches](docs/APPROACHES.md): team proposal, dependencies, decisions, shared working/logging process, and where to resume.
- [Raw case loader](docs/LOADER.md): usage; six synthetic tests, one real-case decode, and goal/history display passed on Perlmutter. Visual/semantic assessment remains open; the raw reader itself applies no preprocessing.
- [Initial ViT preparation](docs/grp2/VIT_TRIAL.md): full-view DINOv2 tensors and inference-only encoder; initial CPU checks and the first GPU development evaluation passed on Perlmutter.
- `cnn.py` implements a frozen pretrained ResNet-18 feature encoder (512 values/image). Its first GPU evaluation is recorded in the evaluation guide; only a separate logistic-regression probe was trained. `data_inspect.ipynb` includes original-image inspection cells with no saved outputs.
- [Before-GRU evaluation](docs/EVALUATION.md): first split-0 GPU run completed; aggregate results show large training/validation gaps. The guide records all comparisons, component timing, caveats and improvement hypotheses. No final-test evaluation or fine-tuning has been done.
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

The owner confirmed Python 3.12.13 and Pillow 12.3.0 after loading the candidate `pytorch/2.13.0` module on Perlmutter. This establishes the raw reader's image dependency; full ML compatibility and team-wide adoption remain open. See the approach notes for evidence and next steps. Do not repeat the completed environment inventory or basic dataset validation.

## Lightweight collaboration

Teammates and AI assistants should start with the [documentation guide](docs/README.md), which identifies the relevant project references for each audience.

- Keep confirmed facts, team proposals, unapproved suggestions, and open questions labeled separately.
- Update the relevant document after material progress; record dated decisions once in `APPROACHES.md` rather than duplicating logs.
- Before parallel implementation, agree on input/output interfaces, preprocessing, and evaluation. Individual ownership is still open.
- Keep sensitive data and derived sensitive artifacts on Perlmutter. Never put credentials in documentation.

The repository includes raw loading, full-view preparation, frozen image encoders, and the before-GRU evaluation runner. The first Perlmutter GPU classifier comparison completed; reviewing uncertainty and matched errors is next before a bounded regularization/representation experiment. No GRU or fine-tuning is implemented. See the evaluation guide for results and the group logs for verification evidence.
