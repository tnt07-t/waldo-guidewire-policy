# CNN temporal attention and shared development evaluation

**Scope — 2026-11-03:** CNN–temporal attention implementation and
shared neural-baseline evaluation, developed alongside the independent
ViT–GRU check. Existing reader, preprocessing, encoder and notebook interfaces
are unchanged. Development settings and real-data verification remain separate
from the final team evaluation protocol.

**Current requested experiment:** see [ATTENTION_EXPERIMENTS.md](ATTENTION_EXPERIMENTS.md)
for the eight-configuration static/mean/attention comparison, position on/off,
widths 32/64/128, fixed final MLP, rationale and results status. The two-model
commands below remain examples rather than the requested full sweep.

**Next stage (2026-11-04):** inspect the completed fixed-64 run using
[learning/representation diagnostics](ATTENTION_EXPERIMENTS.md#learning-and-representation-diagnostics--2026-11-04).
Per-epoch training BA and class behavior precede further capacity changes.
Width/position sweeps remain deferred; no additional accuracy result is implied.

## Where the cases come from

The supplied `splits.json` assigns existing cases to partitions. Position 0
contains 267 training, 80 validation and 69 test cases. A case contains a
separate goal and six historical images, not a single image. The experiment does not create additional cases or combine the five alternative split assignments. The new
runner checks row/episode separation before extracting images. It reads test
membership for separation only, never test images or predictions.

## Implemented architecture

`temporal_models.py` contains heads and an image-to-logit `FrozenPolicy`:

1. Existing ImageNet-pretrained frozen ResNet-18, default maxpool retained.
   Shared weights encode the separate goal and each of six chronological frames.
   Existing full-view RGB 518-square resize/padding/normalization is unchanged.
2. Each history step: 512 CNN values + scaled relative depth + scaled time = 514.
   Scale depth/time by the respective population standard deviations over all
   training steps; do not center; constant channels use scale 1. Reuse these
   scales for validation/inference. No separate depth MLP is used in this trial.
3. Linear projection 514→64; add learned six-position embeddings.
4. One Transformer encoder layer: width 64, four heads, feedforward width 128,
   GELU, dropout 0.1. All six observations are current/past; within-window
   attention is unmasked and does not introduce future observations.
5. Take the final-position output (64 values), concatenate separate goal (512),
   then Linear(576,64) → ReLU → Dropout(0.1) → Linear(64,2).
   Scores are RETRACT=0 and ADVANCE=1. Goal enters after temporal processing.

Attention runs across six frame vectors, not across image patches. Position
embeddings encode frame order; supplied time offsets encode unequal spacing.
The goal-reference/“artery map” correspondence remains a dataset interpretation,
not a new input source.

## Matched baselines and scope

`temporal_evaluation.py` defaults to `latest_mlp attention` with the SAME encoder,
partitions, training-only metadata scaling and optimization/selection rules:

| Head | Visual inputs | Metadata | Summary |
|---|---|---|---|
| latest_mlp | latest + goal | all six depths/times | 1036→64→2 MLP |
| mean_mlp | averaged history + goal | all six depths/times | 1036→64→2 MLP |
| attention | six history images + goal | aligned per step | projected temporal attention |
| gru / lstm | six history images + goal | aligned per step | projected one-layer recurrent model |

All heads use ReLU/dropout in the final classifier. GRU/LSTM options are local
comparison controls, not replacements for existing model code or recorded
results. The projection changes the architecture relative to the direct
386-input ViT GRU. Coordinate input/fusion/projection settings with Group 1
before claiming a pure GRU-versus-attention comparison. Optional `--encoder
dinov2` reuses the same heads with 384-d image features; this is not the ViT–GRU runner.

The MLP tests whether nonlinear static prediction improves over logistic
regression. Attention versus MLP changes visual history AND architecture and
capacity; it does not alone establish the benefit of temporal order.
At default settings the attention head has 103,874 trainable parameters; the
latest-image MLP has 66,498. These exclude the frozen CNN.

Default development budget: AdamW lr .001, weight decay .001, minibatch 32,
100 epochs maximum, early stopping after 15 epochs without strictly higher
validation balanced accuracy, seed 0. Training-only inverse-frequency class
weights and gradient clipping at 1.0 are used. The earliest checkpoint wins ties.
These defaults are adjustable and have not been selected by real-data evidence.
Agree on comparable budgets before tuning. A validation-selected result is not
an unbiased test estimate. Prefer repeating the leading comparison across seeds
to selecting whichever seed looks best. CUDA results need not be bitwise identical.

## Synthetic checks — laptop or Perlmutter

From repository root in the project environment:

```sh
python3 -B -m unittest discover -s tests -p 'test_temporal.py' -v
python3 -B temporal_evaluation.py --help
```

Windows local environment: `.venv-eval/Scripts/python.exe`. Real data never
belongs on the laptop. Tests use invented arrays/images and injected encoders,
including the complete runner/report/checkpoint flow, no public weights.
All 42 repository tests passed locally, including nine new temporal tests;
pretrained ResNet/real-data/GPU integration remains to be checked on Perlmutter.

## Perlmutter setup and run

Code must first reach the Perlmutter checkout through the normal Git handoff;
uncommitted laptop files are not available through a pull. Preserve the saved
remote notebook and an independent Perlmutter copy before updating that checkout.
No notebook changes are needed. Work separately from the active ViT–GRU debug checkout.

### Code and environment check — before allocation

From the intended Perlmutter repository directory, confirm the files exist:

```sh
pwd
git branch --show-current
git status --short
ls temporal_models.py temporal_evaluation.py compare_runs.py tests/test_temporal.py
```

If files are missing, stop here and complete the code transfer. A pull cannot
retrieve uncommitted or unpublished laptop changes. Do not overwrite an active
teammate checkout. Use the existing PyTorch module plus the scratch venv used
for the static evaluation; its exact personal activation path belongs outside
shared documentation. Verify dependencies before requesting GPU time:

```sh
module load pytorch/2.13.0
# Activate the existing evaluation scratch venv here, if needed.
python3 -B -c "import torch, torchvision, PIL, numpy, sklearn; print('Dependencies available')"
python3 -B -m unittest discover -s tests -p 'test_temporal.py' -v
```

These synthetic checks use no GPU, real cases or pretrained weights.

### GPU allocation

Use the previously working GPU account, replacing `YOUR_GPU_ACCOUNT`.
`shared_interactive` supports a one-GPU interactive job. The CPU request below
matches the prior successful allocation. Wait for allocation to be granted
before running the job-step command:

```sh
salloc --account=YOUR_GPU_ACCOUNT --constraint=gpu --qos=shared_interactive --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --time=01:00:00
srun --ntasks=1 --cpus-per-task=32 --gpus=1 --pty bash -i
```

The second command enters the compute-node shell. A second `salloc` is not
needed if already inside a suitable allocation. Record the job ID printed by
Slurm. [NERSC interactive-job guidance](https://docs.nersc.gov/jobs/interactive/)
requires explicitly requesting GPUs for both allocation and job step.

Restore the working module/venv in that compute shell, then verify CUDA:

```sh
module load pytorch/2.13.0
# Activate the same existing scratch venv, if needed.
hostname
echo "$SLURM_JOB_ID"
nvidia-smi
python3 -B -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU visible')"
```

Run extraction/training only after CUDA is available. Record the GPU model for
latency comparisons; this does not prove pretrained model integration works.

### Scratch paths and experiment commands

**First real-data verification:** run the bounded eight-case attention check in
[ATTENTION_EXPERIMENTS.md](ATTENTION_EXPERIMENTS.md#eight-case-integration-check)
before the full sweep. A two-epoch full-partition run is not a tiny-case check.

Use the existing torch/torchvision/Pillow/NumPy/scikit-learn environment; see
`requirements-eval.txt`. The general `environment.yml` omits NumPy/scikit-learn
as explicit dependencies and should not replace the working NERSC environment.
Load the existing module/venv, then use an allocated GPU compute-node shell.
Full extraction/training must not run on a login node.

```sh
hostname
printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
git branch --show-current
git rev-parse HEAD
git status --short
export DATA_DIR="${PSCRATCH:?}/CS5470_FinalProject/shorthorizon2"
export TORCH_HOME="$PSCRATCH/waldo-cache/torch"
export HF_HOME="$PSCRATCH/waldo-cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$TORCH_HOME" "$HF_HUB_CACHE"

python3 -B -m unittest discover -s tests -p 'test_temporal.py' -v

# Smoke the complete real-data extraction/reporting path with a short head budget.
# This STILL extracts all training/validation images; it is not an eight-case check.
python3 -B temporal_evaluation.py --split-position 0 --device cuda \
  --models latest_mlp attention --epochs 2 --patience 1 \
  --output-dir "$PSCRATCH/waldo-temporal/smoke-01"

# Requested full experiment: eight configurations, fresh heads, seed 0.
python3 -B temporal_evaluation.py --split-position 0 --device cuda \
  --attention-sweep --seeds 0 --benchmark \
  --output-dir "$PSCRATCH/waldo-temporal/attention-sweep-01"
```

Check dataset presence before launching:
`test -f "$DATA_DIR/cases.jsonl" && test -f "$DATA_DIR/splits.json"`.
Use a new output suffix if a directory already exists; preserve prior evidence.
The short smoke command is optional integration debugging and still extracts
the full train/validation set. The full sweep is the requested comparison.

### Inspect results and release resources

```sh
cat "$PSCRATCH/waldo-temporal/attention-sweep-01/status.json"
cat "$PSCRATCH/waldo-temporal/attention-sweep-01/summary.md"
```

Completed status and all eight model rows plus majority establish runner
completion, not scientific success. Review class recalls, uncertainty, training
gaps and timing before choosing a model. Keep private files on Perlmutter.

When finished, `exit` leaves the `srun` shell; a second `exit` leaves `salloc`.
From the login shell check `squeue -u "$USER"`. If the recorded allocation is
still present, use `scancel JOB_ID` for that allocation only. Do not cancel a
different session or a teammate's job. Merely closing the terminal is not the
resource-release check.

Each invocation extracts frozen train/validation features once, in memory,
and reuses them across selected heads/seeds. No feature cache is persisted.
A new output directory prevents overwriting previous results. The runner checks
the encoder parameters/buffers stayed unchanged. The full run stores checkpoints
on Perlmutter; this is separate from the ViT–GRU no-checkpoint tiny debugging scope.
Neither command has been executed on real data by assistant tools.

## Outputs and tracking

- `summary.md`: aggregate comparison; suitable for team sharing after review.
- `metrics.json`: config, source hashes, Git/environment provenance, learning
  curves, selected epoch, metrics, episode-bootstrap BA intervals, parameter
  counts and optional latency. Config includes private paths/scales; keep whole
  report on Perlmutter and share only permitted aggregates.
- `validation_predictions.private.json`: aligned case-level predictions; private.
- `<model>-seed<N>.pt`: selected head weights and reconstruction config/scales;
  private. These do not include encoder weights or optimizer state for resuming.
- `status.json`: extracting/completed marker. A failed run may leave partial
  files; absent completed status is not a successful full comparison.

Track explicitly selected old and new reports together:

```sh
python3 -B compare_runs.py \
  "$PSCRATCH/waldo-evaluation/dev-static-01/metrics.json" \
  "$PSCRATCH/waldo-temporal/attention-sweep-01/metrics.json" \
  --output "$PSCRATCH/waldo-temporal/comparison-01.md"
```

The ledger preserves runs/model/seeds/split and protocol context rather than
declaring a winner across different tuning budgets. It accepts old static and
new neural report formats, not the ViT–GRU eight-case report (which is not validation
evaluation). Compare repeated seeds and uncertainty alongside validation scores.
Intervals resample validation episodes conditional on the selected fit; the intervals
are not corrected for checkpoint selection or a paired significance test.

`--benchmark` uses five warm-ups and 20 synchronized repetitions of one decoded
training case, batch one. It includes preparation, transfer, image encoder and
head, excludes disk IO, and assumes no feature reuse. Latest uses two images;
attention/other temporal heads use seven. Preparation and model time, total
median/p95 and peak allocated CUDA memory are separate. This is a same-case
microbenchmark, not workload-wide p95. Old static encoder-only timings exclude
the classifier and should not be ranked directly against these totals.

## Load a CNN checkpoint for inference

On Perlmutter, using a prepared case from the existing preprocessor:

```python
import torch
from cnn import FrameEncoder
from temporal_models import FeatureHead, FrozenPolicy

saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
cfg = saved["config"]
head = FeatureHead(saved["kind"], cfg["feature_dim"], cfg["width"], cfg["dropout"],
                   position_embeddings=cfg.get("position_embeddings", True),
                   classifier_width=cfg.get("classifier_width", cfg["width"]))
head.load_state_dict(saved["head_state"])
policy = FrozenPolicy(FrameEncoder(), head).to(device).eval()
meta = torch.tensor(list(zip(case["depth_relative_mm"], case["history_offsets_s"])),
                    dtype=torch.float32, device=device).unsqueeze(0)
meta = meta / torch.tensor(cfg["metadata_scale"], device=device)
with torch.no_grad():
    logits = policy(prepared["goal"].unsqueeze(0).to(device),
                    prepared["history"].unsqueeze(0).to(device), meta)
    action = logits.argmax(-1)  # 0 retract, 1 advance
```

This snippet is for ResNet checkpoints. DINOv2 uses its inference wrapper and
the head directly, not `FrozenPolicy`. Check config/provenance before loading.

PyTorch layer reference:
https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
