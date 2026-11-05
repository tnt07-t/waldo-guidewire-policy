# CNN–attention development experiments

## Training-only image-feature standardization — 2026-11-05

**Question:** does conditioning frozen latest/goal embeddings improve the stalled
LeakyReLU latest+goal model, without changing information, initialization or
trainable architecture? The existing unstandardized LeakyReLU run is the
comparison: final train/validation AUC .687/.478 and BA .500/.500.

### Implemented transform and boundaries

`--feature-normalization standard` is scoped to `latest_goal_mlp`. For each
of the 512 latest-image channels and 512 goal-image channels separately, fit
population mean and standard deviation over the 267 training cases. Apply
`(feature - training_mean) / training_std` before concatenation and the MLP.
Channels with training std ≤ 1e-6 use scale 1 to avoid amplifying near-constant
noise. No validation/test examples or labels determine these statistics.

The transform is inside the head. Statistics are nontrainable state buffers,
fit once before initial assessment/training, retained for validation and saved
in selected/final checkpoints. Diagnostics, inference and latency measurement
reuse those buffers without refitting or double normalization. Metadata is still
excluded. Image preparation, CNN weights, goal marking and original data stay
unchanged. No feature cache is persisted.

Initialization remains PyTorch default; seed 0 creates exactly the same trainable
initial parameters as the unstandardized head. Parameter count remains 65,730.
LeakyReLU slope .01, dropout .1, 64 hidden units, AdamW .001, weight decay .001,
batch 32, 100-epoch cap, patience 15 and earliest best validation BA remain fixed.
No Kaiming/Xavier, width, threshold or learning-rate sweep accompanies this test.

Centering removes common role-specific offsets and scaling balances input units.
This may improve optimization and activation diversity; it does not promise
generalization. Scaling also changes effective weight regularization in the
original units, so improvement cannot be attributed solely to activation signs.
Previously scaled linear probes already generalize weakly. Initialization,
representation quality and episode-dependent associations remain open questions.

### Perlmutter run

After synchronizing the code and restoring the working environment, DATA_DIR
and TORCH_HOME, run in an allocated GPU compute shell:

```sh
python3 -B temporal_evaluation.py --split-position 0 --device cuda \
  --models latest_goal_mlp --width 64 --classifier-width 64 \
  --classifier-activation leaky_relu --feature-normalization standard \
  --seeds 0 --batch-size 32 --epochs 100 --patience 15 \
  --lr 0.001 --weight-decay 0.001 --dropout 0.1 --benchmark \
  --output-dir "$PSCRATCH/waldo-temporal/latest-goal-standardized-01"
cat "$PSCRATCH/waldo-temporal/latest-goal-standardized-01/learning_summary.md"
```

Inspect initial/selected/final training and validation BA/AUC/loss, prediction
counts and head health. Training improvements without validation improvements
would indicate conditioning helps fit while leaving generalization unresolved.
Validation-only gains on one seed remain provisional. For health inspection:

```sh
python3 -B temporal_diagnostics.py \
  --checkpoint "$PSCRATCH/waldo-temporal/latest-goal-standardized-01/latest_goal_mlp-seed0-final.pt" \
  --partition train --device cuda --health-only \
  --output-dir "$PSCRATCH/waldo-temporal/latest-goal-standardized-health-01"
```

Raw/legacy checkpoints keep `feature_normalization=none` and require no new
buffers. Standardized checkpoints require the matching constructor flag before
state loading; private statistics remain inside the Perlmutter checkpoint.

### Verification and results

All 65 repository synthetic tests passed. Tests verify training-only role means/
population scales, constant-channel handling, refusal to refit, unchanged buffers
under shifted validation inputs and optimization, identical trainable initial
weights/parameter counts, legacy state loading and standardized checkpoint reload.
An end-to-end invented-image training/benchmark/health run verifies transform
persistence without reading earlier history images. Code checks establish no
real performance result. **Real standardized run: pending.**

## Confirmed inactive classifier and activation control — 2026-11-05

The final latest+goal checkpoint health report confirms all 64 hidden ReLU units
are zero for all 267 training cases. Preactivations range -5.4137635 to
-.039026309. ADVANCE probability is exactly .50386977 for every case (std 0),
ROC AUC .500, weighted evaluation loss .6931771. Hidden weight/bias and output
weight loss-gradient norms are zero; only output-bias gradient is nonzero
(.0054726447). This establishes an inactive hidden layer at the inspected final
state, with input-independent output scores. The time/cause of unit death is
not established by final-state inspection alone.

This run cannot evaluate image information quality or establish metadata
necessity. Additional metadata changes both input width/initialization and
optimization geometry; a functioning metadata-enabled control does not prove
the missing numerical signal caused the inactive-ReLU failure.

### Single-factor remedy to test

Added explicit `--classifier-activation leaky_relu`, fixed negative slope .01.
The final hidden MLP activation changes; encoder, image inputs, head width,
parameter count, dropout, optimizer, learning rate, split and selection budget
remain fixed. ReLU remains the default. Attention's internal GELU is unchanged.
PyTorch's [LeakyReLU definition](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.leaky_relu.html)
retains a nonzero negative-input slope, directly addressing the zero-gradient
gate. This does not guarantee good optimization or validation generalization.

Checkpoint config records classifier activation. Diagnostic and inference loaders
reconstruct that activation; legacy checkpoints without the field use ReLU.
Both activations have no learned parameters. Start fresh with seed 0 rather than
continuing the dead checkpoint. No activation/slope sweep is introduced.

```sh
python3 -B temporal_evaluation.py --split-position 0 --device cuda \
  --models latest_goal_mlp --width 64 --classifier-width 64 \
  --classifier-activation leaky_relu --seeds 0 --batch-size 32 \
  --epochs 100 --patience 15 --lr 0.001 --weight-decay 0.001 \
  --dropout 0.1 --benchmark \
  --output-dir "$PSCRATCH/waldo-temporal/latest-goal-leaky-01"
```

First inspect per-epoch training behavior and classifier health at the final
checkpoint. Improvement in training discrimination would support resolving this
mechanical failure; validation BA may remain weak. Before making input-role
claims across models, use the same chosen activation for matched controls under
a predeclared budget. A LeakyReLU run versus metadata-enabled ReLU runs changes
activation as well as information and cannot isolate metadata value.

**Real activation-comparison result:** the owner-run learning summary is recorded
below; detailed checkpoint health remains to inspect. No GPU training was launched
by assistant tools. ReLU results remain preserved and explicitly marked as a
training failure. No width or normalization change is selected.

### 2026-11-05 — LeakyReLU classification result

Supplied latest+goal LeakyReLU learning summary reports selected epoch 1,
16 epochs completed, selected train/validation BA .500/.500 and final
train/validation BA .500/.500. The activation change did not improve reported
classification under the fixed budget. No claim of restored training is justified.

The summary alone does not establish constant probabilities, zero gradients or
inactive LeakyReLU outputs. Inspect the final checkpoint activation identity,
probability spread/AUC and gradients before selecting another remedy. A nonzero
negative-input derivative addresses ReLU gating but does not guarantee sufficient
optimization, discrimination or generalization. No additional activation/width/
learning-rate sweep or budget extension is selected from these scores.

### 2026-11-05 — LeakyReLU gradient and score inspection

Supplied final training health report confirms LeakyReLU slope .01, zero
always-zero units and nonzero hidden/output loss gradients. Hidden preactivations
remain negative (-7.8185782 to -.27501884), so all inspected hidden responses
operate on the small-slope linear branch. Mean ADVANCE probability .50686008,
std .00083268579, range [.50448966,.50840682], training ROC AUC .68684051 and
weighted eval loss .69267392. Gradient L2: hidden weight .0021314595, hidden bias
.000049902665, output weight .0043457444, output bias .0095920376.

The zero-gradient mechanism is resolved at this inspected state. Scores now
rank training cases, but all remain above .5, so thresholded training BA remains
.500. Neither useful validation ranking nor a complete classifier fix follows
from training AUC. This corrects any interpretation of unchanged BA as proof
that LeakyReLU had no effect. Negative preactivations do not mean zero LeakyReLU
output or absent gradients.

PyTorch Linear already initializes weights; local source confirms default
`kaiming_uniform_(a=sqrt(5))`, equivalent to uniform ±1/sqrt(fan_in), with bounded
random bias. This is not the activation-matched He gain for ReLU/LeakyReLU.
An explicit Kaiming hidden-weight initialization matched to slope .01 is a
plausible controlled experiment, not a guaranteed correction. Xavier is another
variance-scaling choice; no evidence yet selects either. Initialization affects
the starting state and cannot ensure the optimizer avoids negative responses
later. Input centering/scaling and optimization remain possible contributors.

**Next evidence:** inspect already saved final validation ROC AUC alongside
training AUC, score means/counts and loss before adding a new training variant.
Keep the .5 threshold unchanged for this comparison. If initialization is tested,
predeclare one hidden-layer scheme with activation/optimizer/budget fixed and
inspect initial versus final activation statistics. Do not simultaneously change
initialization, feature normalization, activation and learning rate or treat
training AUC as validation improvement.

### 2026-11-05 — Initial, selected and final ranking comparison

Supplied aggregate metrics for latest+goal LeakyReLU:

| State / partition | BA | ROC AUC | Unweighted loss |
|---|---:|---:|---:|
| Initial / train | .500 | .508 | .7074 |
| Initial / validation | .485 | .641 | .6965 |
| Selected / train | .500 | .595 | .7463 |
| Selected / validation | .500 | .543 | .7276 |
| Final / train | .500 | .687 | .6890 |
| Final / validation | .500 | .478 | .6913 |

Training ranking improves while validation ranking declines; the full failure
cannot be reduced to scores lying above the fixed .5 threshold. LeakyReLU
resolves the strict gate failure but does not restore validation discrimination.
Validation AUC near .5 gives no evidence that threshold tuning is a sufficient
remedy. Loss and AUC measure different properties: moving probabilities near .5
can reduce loss while ordering cases less usefully. Initial validation AUC .641
is one random-initialization result on the reused 80 cases, not a validated
untrained-model strategy or a reason to select random seeds using validation.

Initialization remains a possible conditioning factor, but final-state negative
preactivations do not prove poor initialization; initially useful validation
ranking deteriorates with training. Shared overfitting/episode-dependent input
associations and optimization remain plausible contributors. No cause is proven.
Avoid broad initialization/activation/threshold searches on these same cases.

One candidate next test is training-only feature centering/scaling while keeping
LeakyReLU, default initialization and optimization fixed. This would test input
conditioning, not establish an expected accuracy improvement; prior scaled linear
probes already generalize weakly. Normalizer statistics must be fitted only on
training and saved for inference. This candidate is not implemented or run, and
no new model-selection choice is established by the current report.

**Verification:** all 61 repository synthetic tests passed. A same-weight negative-
preactivation control verifies zero hidden-weight gradients under ReLU and nonzero
gradients plus an optimizer update under LeakyReLU. Parameter counts are unchanged.
End-to-end static training/checkpoint/diagnostic tests verify activation persistence
and reconstruction, including the fixed .01 negative slope. No real performance
claim follows from these synthetic checks.

## No-metadata MLP training failure inspection — 2026-11-05

Supplied per-epoch history for latest+goal without metadata shows training BA
.500 and validation BA .500 at every epoch 1–16. Training predictions are all
RETRACT for epochs 1–2, then all ADVANCE for epochs 3–16. Mean ADVANCE probability
moves .4512→.4825→.5036, then remains near .503 through .5039. Unweighted training
loss falls .7239→.6911. These values are consistent with near-constant predictions;
the reported mean alone does not establish zero case-level score variation.

Class-balanced training loss has its constant-predictor optimum at probability
.5. The recorded unweighted loss can slightly undercut log(2) when probability
is just above .5 because training has more ADVANCE labels. Thus lower loss here
does not demonstrate case discrimination. Switching class at the fixed threshold
does not constitute learning both classes. No threshold tuning is proposed.

### Candidate failure and read-only test

A fully inactive hidden ReLU is a plausible failure, not a confirmed diagnosis.
If all 64 hidden activations are zero across training, the final linear layer
receives zeros and predictions depend on output biases. Hidden/output-weight
loss gradients can then vanish while output-bias gradients remain. Other
optimization or input-conditioning issues can also produce near-constant output.

`temporal_diagnostics.py --health-only` now supports the static controls and
reports hidden units inactive across all inspected cases, zero-activation
fraction, preactivation range, probability min/max/std, weighted evaluation loss
and parameter-gradient L2 norms. ROC AUC distinguishes score ranking from a
constant thresholded decision; all-one-class predictions need not imply constant
scores. Class weights come from training only. No
optimizer step, parameter `.grad` overwrite or activation/threshold change occurs.
Full-partition gradient cancellation is possible; zero gradient alone does not
prove dead ReLU. Spatial/temporal/probe analysis is omitted in this mode.

Run from the working Perlmutter GPU allocation after pulling the update:

```sh
python3 -B temporal_diagnostics.py \
  --checkpoint "$PSCRATCH/waldo-temporal/input-controls-01/latest_goal_mlp-seed0-final.pt" \
  --partition train --device cuda --health-only \
  --output-dir "$PSCRATCH/waldo-temporal/latest-goal-health-01"
cat "$PSCRATCH/waldo-temporal/latest-goal-health-01/summary.md"
```

The static path reads only latest/goal images on training; no new training or
validation search is required. Existing attention checkpoints remain supported;
normal inspection also includes classifier health. Full config/report stays on
Perlmutter; aggregate summary can be reviewed for sharing. If final-state health
supports inactive units, inspect the saved epoch-1 state in a separate directory
before choosing one bounded remedy. No activation, normalization, learning-rate,
seed or capacity change is selected yet.

**Interpretation:** no-metadata training failure prevents treating this run as
proof metadata is necessary, or that the images lack predictive associations.
Earlier ResNet linear latest+goal results and the working full static reference
remain relevant context under different head/optimization conditions.

**Verification:** all 60 repository synthetic tests passed. Controls include an
intentionally inactive ReLU head (zero hidden/weight gradients with nonzero
output-bias gradient), an active head, unchanged weights/parameter `.grad`, and
static-checkpoint health CLI/report integration with earlier history files
absent. Real inactive-unit and gradient measurements remain pending.

## Static input-role controls — 2026-11-04

**Scope:** compare latest+goal imagery without movement metadata against goal
imagery with full movement metadata. Retain the existing latest+goal+metadata
MLP as the common reference. No new goal-only neural model is included.

| Runner model | Latest image | Marked goal image | Six depths/times | Trainable parameters |
|---|---|---|---|---:|
| latest_goal_mlp | Yes | Yes | No | 65,730 |
| goal_metadata_mlp | No | Yes | Yes | 33,730 |
| latest_mlp (existing reference) | Yes | Yes | Yes | 66,498 |

All use frozen ResNet-18 features, a 64-unit hidden MLP with ReLU and dropout .1,
and two action logits. Latest+goal maps 1024→64→2; goal+metadata maps 524→64→2;
the reference maps 1036→64→2. Metadata is appended as twelve chronological
depth/time values only in metadata-enabled heads. Excluded inputs have no
forward/gradient path into the classifier; dummy tensors preserve the shared
head interface without supplying missing-image information.

### Rationale and comparison limits

- Latest+goal versus the full static reference tests adding numerical history.
- Goal+metadata versus the full static reference tests adding the latest image.
- Goal+metadata versus latest+goal changes two input sources; that pair alone
  cannot isolate a metadata or latest-image contribution.
- No control here processes earlier history images. Compare with completed
  mean-history/attention runs cautiously: architecture/capacity and temporal
  aggregation also differ. Poor performance with history does not establish
  that useful information is absent or intrinsically noisy.
- Fixed hidden width preserves the classifier design, but input dimensionality
  changes parameter count and random initialization. One seed and selected
  validation peaks are preliminary evidence; no superiority or necessity claim
  follows from small score differences.

### Fixed development protocol and run

Split 0, 267 training/80 validation cases, seed 0, batch 32, AdamW .001, weight
decay .001, dropout .1, maximum 100 epochs, patience 15 and earliest strictly
best validation BA checkpoint. Full train/validation metrics are recorded each
epoch. Selected/final checkpoints stay separate. Scaling is training-only;
latest+goal does not consume metadata even though bookkeeping/scaling may read
the supplied fields. Test data is reserved. No optimizer or width sweep.

When only these controls are requested, extraction reads the goal and latest
image once per case, skipping the five earlier image files. Goal+metadata alone
extracts only goals. Mixed temporal comparisons retain the original extraction
path. Benchmarking processes two images for latest+goal/reference and one for
goal+metadata, includes preparation/transfer/CNN/head, excludes disk IO and
assumes no feature reuse.

Run after code synchronization and environment checks, inside an allocated
GPU compute shell with DATA_DIR and TORCH_HOME configured:

```sh
python3 -B temporal_evaluation.py --split-position 0 --device cuda \
  --models latest_goal_mlp goal_metadata_mlp latest_mlp \
  --width 64 --classifier-width 64 --seeds 0 --batch-size 32 \
  --epochs 100 --patience 15 --lr 0.001 --weight-decay 0.001 \
  --dropout 0.1 --benchmark \
  --output-dir "$PSCRATCH/waldo-temporal/input-controls-01"
```

Inspect `summary.md`, `learning_summary.md`, aggregate learning curves and scoped
latency. Private predictions, checkpoints and full configuration remain on
Perlmutter. Record real results and the next bounded decision here after review.
**Real execution/results:** the seed-0 comparison completed; reported aggregates
and limitations are recorded below. No new goal-only result is claimed.

### 2026-11-04 — Completed static input-role comparison

Supplied owner-run aggregate results, seed 0, split 0:

| Model | Selected train BA | Selected validation BA | Validation retract / advance recall | Best epoch / epochs run | Final train / validation BA |
|---|---:|---:|---|---|---|
| Latest+goal, no metadata | .500 | .500 | 1.000 / .000 | 1 / 16 | .500 / .500 |
| Goal+metadata, no history imagery | .575 | .561 | .382 / .739 | 4 / 19 | .637 / .490 |
| Latest+goal+metadata reference | .577 | .597 | .824 / .370 | 2 / 17 | .699 / .478 |

The reference reproduces prior displayed scores. Goal+metadata improves training
fit later while validation declines. Latest+goal's selected checkpoint predicts
RETRACT throughout validation; both selected and final training BA remain .500.
Final prediction counts and intermediate training behavior have not yet been
reviewed, so all-epoch collapse or unchanged probabilities cannot be assumed.

The largest selected validation BA in this comparison belongs to the reference
with latest image and metadata. This does not support a claim that historical
metadata is intrinsically noisy. The no-metadata run's weak training fit limits
its use as evidence of metadata necessity; optimization, initialization and input
conditioning remain possible contributors. Input widths change parameter counts
and initialization despite the same seed. Prior ResNet linear latest+goal BA
.547 also cautions against claiming image inputs inherently contain no signal.

Goal+metadata versus reference differs by .036 selected BA, but one seed, different
parameter counts and reused validation selection do not establish superiority.
No model here processes earlier history images; these results cannot settle
whether temporal visual history is useful or harmful. No literal goal-only neural
run occurred. Next inspect latest+goal loss/probability/count histories from the
existing report before changing capacity or launching another configuration.

**Verification:** all 58 repository synthetic tests passed. New tests verify
latest+goal has no metadata gradient path, goal+metadata has no history-image
gradient path, earlier history frames do not affect either control, and the
image-to-logit wrapper/benchmark encodes only consumed roles. An end-to-end
invented-image run succeeds with the five earlier history files absent and
reloads selected/final checkpoints for both controls. This establishes input
exclusion and runner behavior, not real-data accuracy.

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

**Completed comparison — 2026-11-04:** attention-only replay reproduced selected
epoch 1 BA .500 and final epoch 16 train/validation BA .636/.425 at displayed
precision. Both checkpoint roles were saved and inspected on 267 training cases.
No new architecture or held-out performance gain was introduced.

At epoch 16, heads 1–3 assign larger mean latest-query weights to recent frames
(f6 .217/.191/.209 versus f1 .128/.125/.126). Head 4 favors earlier frames
(f2 .203 versus f6 .150). These are case averages and do not establish task-
appropriate temporal processing. Routing changes without comparable action
sensitivity illustrate why weights alone cannot explain predictions.

| Training intervention | Selected probability change | Final probability change | Final BA delta | Final prediction flips |
|---|---:|---:|---:|---:|
| Replace f1–f4 with training means | .0000 | .0000 | .000 | .000 |
| Replace f5 | .0000 | .0000 | +.005 | .004 |
| Replace f6 | .0005 | .0007 | -.007 | .026 |
| Repeat latest visual features | .0001 | .0001 | .000 | .000 |
| Reverse visual history only | .0004 | .0005 | +.015 | .011 |
| Replace goal | .0108 | .0332 | -.136 | .352 |
| Replace all metadata / depth / time | .0000 | .0000 | .000 | .000 |

Probability change is mean absolute ADVANCE-probability change, not accuracy;
.0000 is rounded. Goal replacement reduces final training BA to approximately
.500 and flips 35.2% of predictions. A 3.32-percentage-point mean probability
change with many flips also warrants checking threshold proximity; flips alone
do not establish a strong margin. The measured goal dependence does not isolate
the red marker from reference anatomy/wire/context or establish leakage.

Raw representations are unchanged. Projected RMS/std/effective rank change
1.4730/.1304/25.83 → 2.3168/.1654/15.31; summary changes
1.0021/.0485/21.81 → .9933/.0268/16.92. Final summaries vary less across cases
despite training-classification improvement. Learned scale, dimension and layer
normalization prevent semantic-loss claims. Similar values across frames,
compression/normalization and the direct goal-classifier path are plausible
contributors, not established causes.

Final depth/time probe R²: raw .395/.792, projected .624/.455, summary
-.163/-.068. Poor summary linear recovery persists, subject to the previously
documented missing oracle/mean controls and fixed-probe limitations.

**Next hypothesis:** test a freshly trained goal+metadata static control under
the same ResNet/head/training recipe before changing attention capacity. Existing
latest+goal+metadata results provide context; align inputs, scaling and budgets
and retain validation-selection caveats. A goal-only-without-metadata experiment
would answer a different question and must be labeled separately. Fixed-head
replacement is distribution-shifting sensitivity, not a substitute for trained
input-role controls. Spatial single-case results remain private and were not
supplied in this comparison. No width, position or goal-conditioning change is
selected from the routing tables alone.

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
