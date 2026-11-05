# CNN–attention development experiments

## Learning and representation diagnostics — 2026-11-04

**Scope:** investigate the fixed-64 attention run before changing architecture.
The completed tiny fit established working updates; the first full comparison
did not establish whether training classification improved after the selected
epoch. No new width, optimizer, input normalization or position choice is selected.

### Per-epoch measurements

`temporal_evaluation.py` now assesses complete training and validation partitions
after each epoch in evaluation mode with no gradients. Reports include ordinary
cross-entropy loss, accuracy, balanced accuracy (BA), both class recalls, confusion
counts, predicted RETRACT/ADVANCE counts and mean ADVANCE probability. BA is the
mean of the two class recalls. The original dropout-active, class-weighted
minibatch loss remains separately named and must not be directly equated with
the new unweighted evaluation loss.

Initial, final-epoch and validation-selected metrics remain distinct. The best
checkpoint rule remains earliest strictly highest validation BA; training BA
does not select checkpoints. Standalone `<model>-seed<N>-history.json` is updated
after each completed epoch so partial progress survives later interruption.
Additional evaluation changes head-work duration; inference timing and the
architecture are unchanged. No additional CNN extraction occurs per epoch.

### Fixed-checkpoint diagnostics

`temporal_diagnostics.py` loads an existing ResNet attention checkpoint, defaults
to the training partition and fits no new action classifier. Features are
extracted once into RAM. Checkpoint metadata scales must match current training
metadata. Reports contain:

| Diagnostic | Question | Interpretation limit |
|---|---|---|
| Per-head 6×6 temporal routing, latest-query weights and entropy | Are attention weights uniform, concentrated or biased toward particular history steps? | Frame routing is not pixel localization or causal importance; direct goal fusion and residual paths also affect output |
| Fixed-checkpoint frame/goal/depth/time replacements | Which input edits change probabilities or decisions? | Training means/reversal can create unfamiliar inputs; no-refit sensitivity does not establish necessity or benefit from retraining |
| Raw/projected/summary spread, effective rank and frame differences | Does representation variation collapse across steps/cases? | Learned scales and different dimensionalities confound direct comparisons; rank is not semantic information content |
| Inner-training episode-disjoint ridge probes for depth/time | Is known numerical metadata linearly recoverable after projection/summary? | Fixed alpha=1 decoder, not a new navigation classifier; raw input explicitly contains targets; poor decoding does not prove information is absent or task-relevant |
| Optional 3×3 content occlusion on one training case | Which coarse image edits change the predicted action probability through CNN and head? | Normalized-zero masks may be unfamiliar; sensitivity is not wire detection, correct target reasoning or an attention pixel map |

The metadata probe uses a fixed seed-0 episode holdout inside training only,
fits scaling/decoder on the remaining training episodes, and excludes constant
target steps from R². True validation/test is not used to tune the probes.
No image embeddings, per-case attention weights or checkpoints are persisted by
the diagnostic runner. Full JSON includes checkpoint configuration/private paths
and scales and remains on Perlmutter; aggregate Markdown can be reviewed for
sharing. No test-partition diagnostic option is provided.

Temporal attention receives pooled 512-value image vectors, so no original
spatial map is available inside that layer. Diagnosing loss of specific wire-tip,
red-marker or anatomical information requires targeted image controls or
annotations. The current tools identify candidate bottlenecks and known metadata
recoverability; semantic loss is not inferred from a lower effective rank.

PyTorch exposes per-head weights through `need_weights=True` and
`average_attn_weights=False`; inspection recomputes routing from the normal
layer input without changing the production forward path. Interpretation follows
the caution in [Jain and Wallace](https://aclanthology.org/N19-1357/) while retaining
the qualifications discussed by [Wiegreffe and Pinter](https://aclanthology.org/D19-1002/).

### Perlmutter execution order

First inspect the already saved checkpoint; no architecture rerun is needed:

```sh
python3 -B temporal_diagnostics.py \
  --checkpoint "$PSCRATCH/waldo-temporal/dev64-01/attention-seed0.pt" \
  --partition train --device cuda --spatial-case-index 0 \
  --output-dir "$PSCRATCH/waldo-temporal/diagnostics-dev64-01"
```

This extracts the 267 training cases but no validation images. Optional spatial
inspection selects training-local case 0 before observing the map and performs
63 single-image regional edits (seven images × nine cells). Omit
`--spatial-case-index 0` for temporal/numerical diagnostics only. Validation
inspection requires explicit `--partition validation`; use sparingly, after
training diagnostics, without repeated configuration searches.

Then rerun the fixed development comparison with added epoch measurements:

```sh
python3 -B temporal_evaluation.py --split-position 0 --device cuda \
  --models latest_mlp mean_mlp attention --width 64 --classifier-width 64 \
  --seeds 0 --batch-size 32 --epochs 100 --patience 15 \
  --lr 0.001 --weight-decay 0.001 --dropout 0.1 \
  --output-dir "$PSCRATCH/waldo-temporal/dev64-diagnostics-01"
```

The second run starts fresh heads and repeats the previous recipe to add missing
measurements. The old run remains preserved. Exact GPU reproducibility is not
guaranteed. Benchmarking is omitted because architecture/input computation is
unchanged; prior scoped latency evidence remains separate. Both commands require
the working environment, DATA_DIR/TORCH_HOME and a GPU compute allocation.

### Verification and result record

All 54 repository synthetic tests passed locally, including eight new diagnostic
tests. The diagnostic CLI help and diff-format checks passed. The full invented-
image test trains a checkpoint, reloads the unchanged architecture, generates
training-only diagnostics and verifies encoder freezing and report creation.
This verification supplies no guidewire accuracy or real attention map.

The training run also writes `learning_summary.md` (selected versus final train/
validation BA) and `learning_curves.csv` (per-epoch loss, BA, recalls and prediction
counts). `diagnostics.json` contains detailed aggregate numerical evidence;
the diagnostic `summary.md` includes aggregate temporal routing, interventions,
representation spread and metadata probe results. Optional single-case spatial
tables are written separately to `spatial_sensitivity.private.md` and remain on
Perlmutter, along with the full JSON report. Only reviewed aggregate summaries
are suitable for sharing.

Synthetic tests cover full-forward equality during inspection, per-head weight
normalization, no model-state changes, a uniform-attention control, a constant
representation control, irrelevant-frame interventions on latest-only heads,
known recoverable metadata versus a constant representation, full-checkpoint
CLI/report flow, per-epoch counts/BA consistency and selected/final separation.
Spatial tests verify one-image re-encoding and seven 3×3 grids. Synthetic tests
do not establish real diagnostic outcomes or semantic validity.

**Real results:** training-partition diagnostics of the selected epoch-1
checkpoint were supplied and reviewed below. The new per-epoch development run
remains pending. No semantic loss, focus location or architecture improvement
is established. No additional model sweep is automatic.

### 2026-11-04 — Selected-checkpoint training diagnostics

Supplied report covers 267 training cases at selected epoch 1. Latest-query
attention averaged across cases is near uniform for every head (entries
.163–.171, compared with uniform 1/6 ≈ .167). Mean weights alone do not establish
uniform routing in individual cases; per-case variability and average entropy
in the full private report remain to review.

Training-mean image replacements changed mean ADVANCE probabilities by less
than .00005 at f1–f5 (displayed .0000), .0005 at f6, .0001 for repeated latest
imagery, and .0004 for reversed visual history. Goal replacement changed mean
absolute probability by .0108. Metadata/depth/time replacements displayed .0000.
Every intervention produced zero prediction flips and zero BA delta. These are
fixed-checkpoint sensitivities, not controlled retraining or proof inputs lack
useful information. Rounding masks small nonzero effects. Goal perturbation
was the largest reported effect but establishes neither correct target use nor
goal-only superiority.

| Stage | RMS | Between-case std | Effective rank | Depth probe R² | Time probe R² |
|---|---:|---:|---:|---:|---:|
| Raw steps | .8401 | .1838 | 41.96 | .395 | .792 |
| Projected steps | 1.4730 | .1304 | 25.83 | .651 | .456 |
| Final summary | 1.0021 | .0485 | 21.81 | -.221 | -.126 |

Summary representations retain variation but are less variable across cases;
different dimensions, learned scales and normalization prevent direct semantic
loss attribution. Negative summary probe R² means poor held-out linear recovery
under this fixed decoder. Raw inputs explicitly include the depth/time targets,
yet raw recovery is imperfect and projected depth recovery is higher: probe
regularization, correlated features, limited episodes and generalization affect
the result. Add target-only oracle and training-mean decoder controls before
using these probes to attribute information loss. Poor linear recovery does not
establish absence of nonlinear information or task-relevant visual detail.

**Next:** run the unchanged fixed-64 recipe with full per-epoch metrics to
distinguish weak full-training fit from later generalization failure. Review
routing entropy/variability and probe controls before any width change.
Spatial perturbation results remain private and were not supplied here.

### 2026-11-04 — Full per-epoch logging run

Supplied learning summary from the unchanged fixed-64 recipe:

| Model | Best epoch | Epochs run | Selected train BA | Selected validation BA | Final train BA | Final validation BA |
|---|---:|---:|---:|---:|---:|---:|
| Latest MLP | 2 | 17 | .577 | .597 | .699 | .478 |
| Mean-history MLP | 5 | 20 | .596 | .601 | .621 | .487 |
| Attention | 1 | 16 | .500 | .500 | .636 | .425 |

Attention does improve full-partition training classification after the selected
epoch, but validation worsens. The same end-point divergence appears in both
MLP controls. This supports investigating generalization, episode/context shift
and input reliance before a capacity increase; it does not prove an exclusive
cause or rule out optimization/capacity limitations. Selected epochs/scores match
the previously supplied run at displayed precision.

The earlier uniform-routing/sensitivity inspection describes the restored epoch-1
attention checkpoint only. It cannot characterize final-epoch representations.
The supplied run saved only the selected head; final metrics were retained, but
final weights were not separately saved. Any selected-versus-final representation
comparison needs explicit final-state persistence without changing the selection
rule. Inspect final recalls, prediction counts and losses from the existing report
before another training experiment. No 128-width or new regularization result is
claimed. Latest MLP remains a provisional speed/accuracy reference, not a final
model or evidence that visual history is intrinsically harmful.

### 2026-11-04 — Final-epoch class behavior

Supplied aggregate final metrics:

| Model / partition | Loss | BA | Retract recall | Advance recall | Predicted retract / advance |
|---|---:|---:|---:|---:|---|
| Latest / train | .6580 | .6991 | .7216 | .6765 | 125 / 142 |
| Latest / validation | .7017 | .4783 | .5000 | .4565 | 42 / 38 |
| Mean / train | .6462 | .6209 | .3711 | .8706 | 58 / 209 |
| Mean / validation | .6995 | .4866 | .1471 | .8261 | 13 / 67 |
| Attention / train | .6617 | .6364 | .5258 | .7471 | 94 / 173 |
| Attention / validation | .7141 | .4252 | .1765 | .6739 | 21 / 59 |

Attention is not a constant predictor at the final epoch. Both recalls improve
above chance on training, but validation RETRACT recall is weak and predictions
favor ADVANCE. Mean has a stronger ADVANCE preference; latest has nearly balanced
validation prediction counts and still performs poorly. Therefore one-class
collapse cannot explain all models or all stages. Validation losses exceed the
.6931 loss of assigning probability .5 to every case; confidence/calibration and
classification can differ, so this is not proof of absent predictive information.
Generalization weakness is shared across heads. No result demonstrates that
earlier images are intrinsically harmful or that width 128 is the remedy.

### Selected versus final checkpoint inspection

The runner now saves `<model>-seed<N>-final.pt` before restoring the selected
state; the original `<model>-seed<N>.pt` remains the validation-selected model.
Role/epoch are explicit in checkpoint and diagnostic reports. Final weights are
for diagnosing learning dynamics, not alternative validation-driven selection.
Training, early stopping, architecture and selection rule remain unchanged.
Older runs lack final weights and cannot reconstruct the final head from metrics.
All 55 synthetic repository tests passed after this change. Tests verify final
state saving occurs before selected-state restoration and both checkpoint roles
reload with correct epochs. No new real-data training result is implied.

For a future instrumented run, compare aggregate training diagnostics from:

```sh
python3 -B temporal_diagnostics.py \
  --checkpoint "$PSCRATCH/waldo-temporal/dev64-checkpoints-01/attention-seed0.pt" \
  --partition train --device cuda \
  --output-dir "$PSCRATCH/waldo-temporal/inspect-selected-01"
python3 -B temporal_diagnostics.py \
  --checkpoint "$PSCRATCH/waldo-temporal/dev64-checkpoints-01/attention-seed0-final.pt" \
  --partition train --device cuda \
  --output-dir "$PSCRATCH/waldo-temporal/inspect-final-01"
```

The example input directory must first be produced by the unchanged fixed-64
training command with that new output suffix. These commands are a run plan,
not completed results. No extra allocation or retraining has been initiated.

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
