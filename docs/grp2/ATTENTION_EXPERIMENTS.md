# CNN–attention development experiments

## Scope and rationale — 2026-11-03

**Updated execution order — 2026-11-03:** following the revised experiment plan,
first run the current 64-wide attention model with learned positions, plus the
latest-image and mean-history MLP controls. Fix the budget and checkpoint rule
before execution: seed 0, batch 32, AdamW .001, weight decay .001, dropout .1,
100 epochs maximum, patience 15, earliest strictly best validation BA checkpoint.
The 32/64/128 position-on/off sweep remains implemented but deferred until the
first full development result is reviewed. This supersedes the immediate-sweep
execution order below; it does not remove or change the available ablations.

```sh
python3 -B temporal_evaluation.py --split-position 0 --device cuda \
  --models latest_mlp mean_mlp attention --width 64 --classifier-width 64 \
  --seeds 0 --batch-size 32 --epochs 100 --patience 15 \
  --lr 0.001 --weight-decay 0.001 --dropout 0.1 --benchmark \
  --output-dir "$PSCRATCH/waldo-temporal/dev64-01"
```

Review train/validation learning curves, class recalls, uncertainty and latency.
Strong training with weak validation motivates regularization, earlier stopping
or simpler models; weak training and validation motivates input/optimization/
capacity checks. These are diagnostic hypotheses, not established causes.
Compare against matched-input baselines before attributing value to attention.
Do not repeatedly expand the search to chase small changes on the same 80
validation cases. Test data stays reserved for the fixed final protocol.

The experiment plan specifies retaining the final nonlinear MLP, comparing attention with
and without learned position embeddings, trying projection widths 32/64/128,
and evaluating static, mean-history and attention models. Implemented as
`temporal_evaluation.py --attention-sweep`. This supersedes the two-model
default for this specific experiment; the old CLI remains available.

## Data and evaluation

Real development evaluation loads the supplied Perlmutter dataset, not generated
examples. Split position 0 assigns 267 existing cases to training and 80 to
validation; each case has seven images, metadata and an action label. Frozen
CNN features feed newly initialized trainable heads. Each head learns from
training labels, then predictions are compared against validation labels.
Training fits newly initialized head weights; the pretrained encoder remains
fixed. Test cases are never predicted or used to choose settings.

Local unit/smoke tests instead generate invented images/labels and inject a tiny
encoder. Their results only establish working code, gradients, reports and
checkpoints. They are not guidewire performance. No synthetic accuracy may be
substituted for the real comparison.

## Experiments implemented

| Variant | Historical images | Attention width | Learned position | Final MLP hidden width |
|---|---:|---:|---|---:|
| latest_mlp | latest only | — | — | 64 |
| mean_mlp | six averaged | — | — | 64 |
| attention-w32-pos1 / pos0 | six | 32 | on / off | 64 |
| attention-w64-pos1 / pos0 | six | 64 | on / off | 64 |
| attention-w128-pos1 / pos0 | six | 128 | on / off | 64 |

Baselines are run once per seed, not redundantly for each attention width.
All receive the goal and the same six depths/times. Attention feedforward width
is twice its projection width and four heads are retained. Thus this is an
attention-width/capacity comparison, not an isolated input-projection ablation.
The final MLP always retains a 64-unit hidden layer, although its input size
necessarily changes with the temporal summary width.

## Architecture decisions and tradeoffs

- Frozen ResNet-18 and existing full-view preparation: reuse established inputs
  and constrain trainable capacity. Frozen features cannot adapt to the task.
- Final Linear → ReLU → Dropout → Linear MLP: retained in the experiment plan;
  nonlinear goal/history fusion. No result yet establishes it beats a linear
  final head. ReLU is simple and inexpensive; it is fixed, not being tuned here.
- GELU inside PyTorch's attention feedforward block: retained from the initial
  setup. Different role from classifier ReLU; no activation comparison is claimed.
- Dropout .1: regularizes training, disabled at inference. Held fixed; potential
  underfitting remains possible. Weight decay .001 is also held fixed.
- Position on/off: tests explicit learned step identities. Both variants retain
  relative time, depth and a final-step query. “Off” does NOT remove all ordering
  information or make the model time-blind.
- Width 32/64/128: smaller may reduce cost/capacity, larger may retain more
  information but overfit. No best width is assumed.
- Goal late fusion: simple, consistent across controls. Attention does not see
  the goal during history aggregation; goal-conditioned attention is deferred.
- Latest versus mean versus attention: distinguishes a static nonlinear baseline,
  cheap unordered visual aggregation and temporal attention. Capacity and input
  differences limit causal attribution; a gain is not proof of temporal reasoning.

## Development protocol

Same split, frozen features (extracted once into RAM), training-only channel std
scaling, AdamW .001, batch 32, maximum 100 epochs, gradient clip 1, earliest best
validation BA checkpoint, patience 15, class-balanced training loss. All models
use the same budget; not necessarily equal compute or equal parameter counts.
Seed 0 is the initial comparison. Repeat the leading comparisons with seeds
0/1/2 later under a fixed agreed budget; never report only the best seed.
Eight settings reuse validation, so selection optimism is a concern. Do not
expand the grid automatically after seeing scores. Final test evaluation remains
reserved until architecture and model-selection rules are fixed.

## Run on Perlmutter

### Eight-case integration check

Added `cnn_attention_debug.py`, adapted from the team's bounded ViT–GRU runner
at `b9d1bdf` without changing the ViT–GRU files. Uses split-0 training only, four
cases per label, seed 0, frozen ResNet-18, projection 64, position embeddings
enabled and a 64-unit final MLP. Metadata scales use all training metadata,
but only eight cases' images are decoded/extracted. Features stay in RAM.
Training is full-batch Adam .001, maximum 300 updates or 120 seconds head work,
with a 600-second parent-worker deadline and up to five seconds termination
cleanup. No validation selection, model checkpoint or persisted features.
Dropout .1 is active for updates and disabled for initial/final scoring.

Transfer the new script and its tests before allocating. In the allocated
compute shell with dependencies and DATA_DIR/TORCH_HOME already configured:

```sh
python3 -B -m unittest discover -s tests -p 'test_cnn_attention_debug.py' -v
python3 -B cnn_attention_debug.py --device cuda \
  --output-dir "$PSCRATCH/waldo-temporal/tiny-attention-01"
```

The supervisor uses Linux process groups and runs on Perlmutter; Windows checks
exercise the worker directly with invented images and an injected CNN. Three
synthetic tests cover selection/scaling, real backward/optimizer updates and
time/update caps, frozen-state detection, raw-loader-to-report integration,
eight selected rows and no saved checkpoints. Real GPU compatibility remains
unverified until this run completes.

Inspect `debug_summary.json` on Perlmutter. Keep the entire file private because
it includes selected row indices and derived scales. Share only status, updates,
initial/final loss/accuracy, finite gradients, head weight changes, encoder
unchanged flag, timing and GPU model. Expected head count is 103,874. Exit zero
requires a completed loop with at least one update; it does not promise fitting
all eight cases. A no-update/worker-failed/timeout report is an incomplete check.
Inspect the error and propose a specific fix rather than widening caps or
launching the sweep after failure. Falling loss and perfect training accuracy
are debugging evidence only, not validation performance or correct goal use.
After success, start fresh heads for the development sweep below.

### Full development comparison

Use the setup and allocation instructions in
[CNN_ATTENTION.md](CNN_ATTENTION.md#perlmutter-setup-and-run). Check that the
new files are present before allocating a GPU. Preserve remote notebooks before
updating code; uncommitted laptop files cannot be retrieved with a pull.

```sh
python3 -B -m unittest discover -s tests -p 'test_temporal.py' -v

python3 -B temporal_evaluation.py --split-position 0 --device cuda \
  --attention-sweep --seeds 0 --benchmark \
  --output-dir "$PSCRATCH/waldo-temporal/attention-sweep-01"

python3 -B compare_runs.py \
  "$PSCRATCH/waldo-evaluation/dev-static-01/metrics.json" \
  "$PSCRATCH/waldo-temporal/attention-sweep-01/metrics.json" \
  --output "$PSCRATCH/waldo-temporal/attention-comparison-01.md"
```

The sweep flag selects the eight variants regardless of `--models`, `--width`,
`--classifier-width` or `--no-position-embeddings`. Each checkpoint carries its
actual width, classifier width and position flag; reconstruct using those values.
No head resumes weights from another configuration. Head/feature timings are
not substitutes for total inference latency. Benchmark reports preparation,
model and total median/p95 on one in-memory case; disk IO is excluded.

## Verification and remaining work

### 2026-11-03 — Windows verification

Supplied terminal output confirming all ten temporal tests passed in
4.392 seconds and all 43 repository tests passed in 4.461 seconds using the
local `.venv-eval` interpreter. The printed BA values came from invented-image
smoke tests with an injected encoder and two-epoch training caps; the scores do not
rank models on guidewire data. Real pretrained CNN/GPU integration and the
Perlmutter development sweep remain pending. No additional local test rerun is
needed unless code changes or a new concern arises.

**Real-data verification:** owner-run eight-case and fixed-64 development
results are recorded below. Direct assistant SSH authentication was unavailable;
reported aggregates were reviewed without accessing private data. The width/
position sweep has not run. No final model decision is established.

## Results and model decisions

### End-of-day assessment and next experiment

Completed evidence supports documenting and sharing the first run without
another architecture sweep. Width 128 is a plausible capacity test, not a
demonstrated remedy. The eight-case fit establishes functioning updates on a
small subset; it does not establish sufficient full-partition capacity. Conversely,
weak selected-checkpoint training BA does not prove undercapacity: that checkpoint
was chosen by validation, and later-epoch training BA was not recorded.

**Next diagnostic change (planned, not implemented):** record train BA, per-class
recalls, prediction counts and feature/metadata magnitude summaries. Existing
training loss includes dropout and class weighting, so loss alone cannot identify
memorization, underfitting, poor optimization or a validation shift. Inspecting
input magnitudes does not justify fitting normalization on validation.

If stable optimization still produces weak training fit, a predeclared 64/128
comparison becomes a reasonable capacity test. If training fit improves while
validation stays weak, prioritize generalization controls rather than additional
width. Keep input protocol, final MLP, positions and training/selection budget
fixed; report parameter changes and multiple seeds where feasible. This would
change the temporal projection, attention and feedforward capacity together,
not isolate a single layer. No success claim follows merely from trying 128.

Earlier recommendations to prioritize stronger regularization or greater width
were hypotheses. The linear C results did not show a universal regularization
benefit, and the first attention run does not identify a capacity bottleneck.
Latest MLP receives goal and full metadata, so the current result cannot be
described as latest-image-only superiority. Mean's small BA advantage and goal-only
DINOv2's earlier score do not establish a winning input or encoder across different
heads/protocols. Repeated validation exploration risks selection optimism.

**Wrap-up decision:** retain the completed three-model comparison, tiny-fit
verification, latency scope and known limitations; defer architecture changes.
No additional real run, 128-width result or diagnostic rerun is claimed.

### Eight-case report review

Supplied aggregate report confirms finite gradients, changed head weights and
unchanged encoder. Extraction 3.466 seconds, head work 2.087 seconds, complete
process 19.298 seconds, peak allocated CUDA memory 86,251,520 bytes on an
NVIDIA A100-SXM4-80GB. These are tiny-training/extraction measurements, not
online decision latency or an isolated CNN footprint.

### 2026-11-03 — First fixed-64 development comparison

Supplied aggregate reports cover seed 0, frozen ResNet-18, 267 training and 80
validation cases. Validation-selected results (not final test estimates):

| Model | Train BA | Validation BA | Retract recall | Advance recall | Selected epoch | Epochs run | Total median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Latest MLP | .577 | .597 | .824 | .370 | 2 | 17 | 33.07 |
| Mean-history MLP | .596 | .601 | .941 | .261 | 5 | 20 | 131.17 |
| Attention, width 64, positions enabled | .500 | .500 | 1.000 | .000 | 1 | 16 | 132.24 |

Training minibatch loss decreased .750→.678 (latest), .753→.666 (mean) and
.728→.681 (attention). Attention's selected checkpoint predicts only RETRACT;
later reported epochs remained at or below .500 validation BA. Training BA at
later epochs was not recorded, so decreasing loss alone cannot establish strong
training fit or identify overfitting as the sole cause. Tiny fitting succeeded,
but full-partition optimization/generalization remains weak.

Conditional episode-bootstrap BA intervals: latest [.470,.701], mean [.513,.686],
attention [.500,.500]. Selection across epochs is not accounted for; the mean
interval is not a definitive superiority test. The degenerate attention interval
follows mechanically from constant-class predictions, not robust performance.

Batch-one same-case microbenchmarks excluded disk IO and assumed no feature reuse.
Preparation/model median ms: latest 27.19/5.89, mean 112.71/18.42, attention
113.26/18.77. Latest encoded two images versus seven for mean/attention. Most
measured temporal-pipeline cost is preparation/encoding rather than evidence of
an expensive attention head. Mean and attention model times differ by only .34 ms,
which is not isolated attention latency. Workload-wide latency remains unmeasured.

**Decision:** retain latest MLP as the provisional speed/accuracy reference;
mean's .0045 BA gain does not justify a superiority claim or approximately
fourfold measured total latency. Do not select attention or launch the width/
position sweep yet. Review optimization, feature/metadata magnitudes and
train/validation learning behavior before a bounded follow-up. No final model
or test result is established.

### 2026-11-03 — Eight-case Perlmutter check completed

Supplied compute-node terminal output showing an NVIDIA A100-SXM4-80GB,
CUDA available, and `cnn_attention_debug.py --device cuda` completing with
300 updates. Training loss decreased from 0.692842 to 0.000071 and final
eight-case training accuracy was 1.000. Reported status: `completed`.
The private report is under `$PSCRATCH/waldo-temporal/tiny-attention-01`.

This verifies owner-reported execution and tiny-subset fitting of the CNN–attention
path. It is not validation performance, temporal-use evidence or model selection.
The normal worker verifies frozen encoder state before successful completion;
explicit report fields for gradients/freezing, durations, memory, versions and
source hashes have not yet been reviewed. Full development sweep remains pending;
start fresh heads rather than carrying over tiny-fit weights (none were saved).

Actual metrics, learning curves, counts, selected checkpoints and source hashes
will be saved by the runner. Keep full reports/checkpoints/predictions on
Perlmutter; share reviewed aggregate tables only. Record here after the run:

**Local verification:** all 43 repository tests passed, including ten temporal
tests. An end-to-end invented-image smoke ran all eight configurations with
two-epoch caps, verified selected-checkpoint reconstruction for the no-position
32-wide model, and checked validation rows only in prediction output. All six
attention variants passed backward/gradient checks. Without learned positions,
permuting earlier aligned image/metadata pairs while preserving the latest query
left outputs unchanged, as expected; this is a structural check, not real task
evidence. Synthetic scores are deliberately not recorded as scientific results.

| Evidence | Status |
|---|---|
| Real validation BA, recalls and episode intervals | Reported above; full confusion counts remain to review |
| Training fit versus validation gap | Selected-checkpoint metrics reviewed; later train BA unavailable |
| Batch-one A100 preprocessing/model/total latency | Same-case measurements reported above |
| Position on/off comparison at each width | Pending |
| Width/capacity tradeoff and seed stability | Pending |
| Chosen model and reasons | No decision yet |

Prefer a faster/smaller model when accuracy differences are inconclusive and
class recalls are acceptable. Examine the accuracy–latency frontier rather than
declaring the single largest validation score the winner. No numeric minimum
accuracy or maximum latency has been agreed; document those before final selection.
