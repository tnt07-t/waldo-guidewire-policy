# CNN temporal attention: architecture and setup

Implementation and commands for the shared neural development runner.
[Experiment results](ATTENTION_EXPERIMENTS.md) record completed comparisons,
important findings and current decisions. Width/position sweeps remain deferred.

## Data and architecture

Split position 0 contains 267 training, 80 validation and 69 reserved test cases.
Each case contains a separate marked goal reference, six chronological images,
and aligned relative depth/time. Real data and private artifacts stay on Perlmutter.
The runner checks row/episode separation and never evaluates the test partition.

The attention path in `temporal_models.py` is:

1. Frozen ImageNet ResNet-18 encodes each history image and the goal separately
   into 512 values. Full-view RGB preparation uses 518-square resize/padding.
2. Append depth and time to each history vector: 514 values per step. Divide
   metadata by training-step population standard deviations; constant scales use 1.
3. Project each step to 64 values and add learned position embeddings.
4. Apply one four-head Transformer layer, feedforward width 128, GELU and dropout .1.
   Attention is across six frame vectors, not pixels. All observations precede
   the decision; attention within the window is unmasked.
5. Concatenate the final-position output with the separate goal: 576 values.
   Classify through Linear(576,64), activation, dropout and Linear(64,2).
   RETRACT=0; ADVANCE=1.

The final MLP defaults to ReLU. `--classifier-activation leaky_relu` uses slope .01.
Training-only embedding standardization (`--feature-normalization standard`)
currently supports **latest_goal_mlp only**. Saved buffers preserve fitted statistics.
Pixel normalization and metadata scaling are separate operations.

| Model | Predictive inputs | Head parameters at width 64 |
|---|---|---:|
| `latest_mlp` | Latest + goal + all depth/time pairs | 66,498 |
| `mean_mlp` | Mean history embedding + goal + all depth/time pairs | 66,498 |
| `attention` | Aligned history/depth/time; late goal fusion | 103,874 |
| `latest_goal_mlp` | Latest + goal; no metadata | 65,730 |
| `goal_metadata_mlp` | Goal + all depth/time pairs; no history images | 33,730 |

Local GRU/LSTM controls and optional DINOv2 encoding are available; neither
constitutes the teammates' recurrent implementation. Comparing attention to a
static head changes inputs, architecture and capacity, so temporal benefit is
not isolated by that comparison alone.

## Perlmutter setup and run

Run commands in the Perlmutter checkout, including through a VS Code remote
terminal. Laptop tests use synthetic inputs; real training runs on a compute node.
Pull the published branch into a clean checkout before allocating GPU time.

```sh
module load pytorch/2.13.0
# Activate the existing scratch evaluation environment if required.
python3 -B -c "import torch, torchvision, PIL, numpy, sklearn; print('Dependencies available')"
python3 -B -m unittest discover -s tests -v
python3 -B temporal_evaluation.py --help
```

Windows synthetic checks use `.venv-eval/Scripts/python.exe` instead of `python3`.
The module must be restored after reconnecting or entering a new compute shell.

Replace the account placeholder with an available GPU account. Skip allocation
if already inside a suitable job; wait for allocation before launching `srun`.

```sh
salloc --account=YOUR_GPU_ACCOUNT --constraint=gpu --qos=shared_interactive --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --time=01:00:00
srun --ntasks=1 --cpus-per-task=32 --gpus=1 --pty bash -i
module load pytorch/2.13.0
# Restore the same scratch environment if required.
nvidia-smi
python3 -B -c "import torch; print('CUDA:', torch.cuda.is_available())"
export DATA_DIR="${PSCRATCH:?}/CS5470_FinalProject/shorthorizon2"
export TORCH_HOME="$PSCRATCH/waldo-cache/torch"
mkdir -p "$TORCH_HOME"
test -f "$DATA_DIR/cases.jsonl" && test -f "$DATA_DIR/splits.json"
git rev-parse HEAD
```

Use a new output directory for each run. The completed eight-case check used
`cnn_attention_debug.py`; repeat only after changes requiring integration checks.
The functioning standardized reference is reproduced with:

```sh
python3 -B temporal_evaluation.py --split-position 0 --device cuda \
  --models latest_goal_mlp --width 64 --classifier-width 64 \
  --classifier-activation leaky_relu --feature-normalization standard \
  --seeds 0 --batch-size 32 --epochs 100 --patience 15 \
  --lr .001 --weight-decay .001 --dropout .1 \
  --output-dir "$PSCRATCH/waldo-temporal/latest-goal-standardized-NEW"
```

This is a reference command, not a newly selected experiment. Completed
regularization comparisons changed only dropout to .3 or weight decay to .01.
The first raw-feature comparison used `--models latest_mlp mean_mlp attention`
without standardization or activation overrides. Each invocation extracts frozen
training/validation features once in memory, then trains the requested heads.

Inspect `status.json`, `summary.md` and `learning_summary.md`. Completion verifies
the pipeline finished; class recalls and learning curves determine interpretation.
When finished, exit the compute shell and allocation shell, then check
`squeue -u "$USER"`. Cancel only the recorded allocation if still active.

## DINOv2–temporal attention comparison — prepared 2026-11-05

The existing `--encoder dinov2` path substitutes frozen DINOv2 Small/14 for
ResNet-18. Each image yields 384 features; depth/time produce 386-value steps,
projected to the same 64-wide temporal encoder. Late fusion joins the 64-value
summary with the 384-value goal, followed by the 448→64→2 classifier. Backbone
pretraining and feature dimension differ; this is a comparison of pretrained
encoder pipelines, not architecture alone.

The first owner-reported DINOv2 comparison completed: latest/mean/attention
validation BA .597/.627/.565. See the experiment log for recalls, learning behavior
and limits. Earlier logistic probes use a different classifier protocol. The
following reproduces the completed settings in an allocated Perlmutter GPU shell
after the environment and scratch paths above are configured. The successful
run used timm 1.0.30 in a scratch venv with module packages inherited. Export the
HF cache before starting Python to avoid the previously observed home-cache
locking failure; no Hub token was needed.

```sh
export HF_HOME="${PSCRATCH:?}/waldo-cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"
python3 -B -c "import torch, timm; print('CUDA:', torch.cuda.is_available()); print('timm:', timm.__version__)"
python3 -B temporal_evaluation.py --split-position 0 --device cuda \
  --encoder dinov2 --models latest_mlp mean_mlp attention \
  --width 64 --classifier-width 64 --classifier-activation relu \
  --feature-normalization none --seeds 0 --batch-size 32 \
  --epochs 100 --patience 15 --lr .001 --weight-decay .001 --dropout .1 \
  --benchmark --output-dir "$PSCRATCH/waldo-temporal/dinov2-attention-dev64-01"
```

These settings match the original CNN comparison. Its selected validation BA
was .597 latest, .601 mean and .500 attention. Fresh-head ReLU failure remains
a possibility; inspect train BA and class counts before interpreting a low score
as an encoder limitation. The successful standardized static recipe is a different
protocol and standardization is not yet supported for attention.

For a fresh matched CNN reference, repeat the same command with
`--encoder resnet18` and output `resnet18-attention-dev64-01`. Do not launch either
run twice or overwrite existing evidence. Review completion and selected/final
behavior before further experiments. Both runs extract full training/validation
features; runtime and completion by a particular deadline are not established.

Record selected train/validation BA, both recalls, AUC, selected epoch, final
behavior, trainable parameter count and preparation/model latency. Compare
attention between encoders and against each encoder's static/mean controls.
Keep validation comparisons exploratory and the test partition reserved.

### Latest/mean replication — seeds 1 and 2

Keep the completed seed-0 evidence. From the same working Perlmutter environment,
the completed replication used four fresh fits without repeating attention or
changing the training recipe. Commands are a record; reuse requires new output paths:

```sh
python3 -B temporal_evaluation.py --split-position 0 --device cuda \
  --encoder dinov2 --models latest_mlp mean_mlp \
  --width 64 --classifier-width 64 --classifier-activation relu \
  --feature-normalization none --seeds 1 2 --batch-size 32 \
  --epochs 100 --patience 15 --lr .001 --weight-decay .001 --dropout .1 \
  --benchmark --output-dir "$PSCRATCH/waldo-temporal/dinov2-latest-mean-seeds12-01"
```

Then review `summary.md` and `learning_summary.md`. Combine reports with:

```sh
python3 -B compare_runs.py \
  "$PSCRATCH/waldo-temporal/dinov2-attention-dev64-01/metrics.json" \
  "$PSCRATCH/waldo-temporal/dinov2-latest-mean-seeds12-01/metrics.json" \
  --output "$PSCRATCH/waldo-temporal/dinov2-latest-mean-ledger-01.md"
```

This ledger keeps seed rows separate. Assess all seeds and paired BA differences;
no best-seed selection or automatic follow-up run is planned.

## Diagnostics and outputs

`temporal_diagnostics.py --help` lists checkpoint-inspection options. Inspect
selected and final checkpoints separately: the final epoch is a learning diagnostic,
not a replacement for validation selection. Attention weights show frame routing;
mean-input replacements measure fixed-checkpoint sensitivity, not retrained ablations.

| Artifact | Purpose |
|---|---|
| `summary.md` | Aggregate development comparison |
| `learning_summary.md`, `learning_curves.csv` | Selected/final behavior and per-epoch evaluation |
| `metrics.json` | Settings, provenance, metrics, uncertainty and optional timing; private |
| `<model>-seed<N>.pt` | Selected head, reconstruction config and normalization buffers; private |
| `<model>-seed<N>-final.pt` | Final-epoch head for diagnostics; private |
| `validation_predictions.private.json` | Case-level predictions; private |
| `status.json` | Run completion marker |

Checkpoints omit encoder weights and optimizer state. Reviewed aggregate summaries
may be shared; full reports and case-level outputs remain on Perlmutter.
`compare_runs.py` combines explicitly selected static/neural reports into a ledger;
different inputs, tuning budgets and selection protocols must remain visible.

`--benchmark` times one decoded training case with five warm-ups and 20 synchronized
batch-one repetitions. Preparation and model time are reported separately; disk IO
is excluded and features are not reused. Latest uses two images, temporal models
seven. These timings are not workload-wide p95 or isolated temporal-head latency.

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
                   classifier_width=cfg.get("classifier_width", cfg["width"]),
                   classifier_activation=cfg.get("classifier_activation", "relu"),
                   feature_normalization=cfg.get("feature_normalization", "none"))
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
