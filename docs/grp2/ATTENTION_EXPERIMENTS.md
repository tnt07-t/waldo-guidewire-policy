# CNN–attention development experiments

## Current conclusion — 2026-11-05

Feature standardization resolved a major training obstacle in the latest+goal
MLP. Generalization remains weak: training reaches 100% balanced accuracy while
validation returns near chance. Higher dropout and weight decay produced no
clear validation-BA improvement. No final model is selected.

DINOv2 mean-history leads latest across seeds 0/1/2: average validation BA
.612 versus .580, a 3.20-point gain. Seed-0 attention is .565. This supports
history averaging under the development protocol, without establishing useful
temporal ordering or generalization across new episodes. Metadata necessity and
goal-marker understanding remain unresolved.

[Setup and commands](CNN_ATTENTION.md) cover implementation usage. This log records
completed approaches, evidence and decisions. Earlier logistic probes use a
separate protocol and remain in [EVALUATION.md](../EVALUATION.md).

## Shared development protocol

| Item | Setting |
|---|---|
| Data | Supplied split position 0: 267 training, 80 validation, 69 reserved test cases; episode/row separation checked |
| Inputs | Marked goal reference; up to six chronological images; aligned relative depth/time where enabled |
| Image encoder | Frozen ImageNet-pretrained ResNet-18, 512 features/image; separate goal/history roles |
| Preparation | Full-view RGB resize/pad to 518 square; pretrained pixel normalization |
| Head | 64-unit hidden classifier; activation and temporal processing specified per experiment |
| Optimization | AdamW, learning rate .001, batch 32, training-derived class weights, gradient clip 1 |
| Default regularization | Dropout .1, weight decay .001 |
| Budget | Seed 0, at most 100 epochs, patience 15 |
| Selection | Earliest strictly highest validation balanced accuracy; threshold .5 |
| Reporting | Accuracy, BA, both recalls, AUC, loss; initial/selected/final states kept distinct |

**BA** is the average of RETRACT and ADVANCE recall. Training loss during updates
includes class weighting/dropout; full-partition evaluation loss is unweighted
cross-entropy with dropout disabled. Selected checkpoints are the reported
models; final checkpoints are diagnostic. Validation is reused for development,
so scores are not unbiased final estimates. Test data remains untouched.

## Approaches implemented

| Runner model | Predictive inputs | Trainable parameters at width 64 |
|---|---|---:|
| `latest_mlp` | Latest image + goal + all depth/time pairs | 66,498 |
| `mean_mlp` | Mean of six image embeddings + goal + all depth/time pairs | 66,498 |
| `attention` | Six aligned image/depth/time steps, then late goal fusion | 103,874 |
| `latest_goal_mlp` | Latest image + goal; no metadata | 65,730 |
| `goal_metadata_mlp` | Goal + all depth/time pairs; no history imagery | 33,730 |

Static and mean pooling provide inexpensive reference heads for testing whether
history adds value. Projection limits temporal computation; positions represent
order; the final MLP provides nonlinear prediction. These are design hypotheses,
not demonstrated advantages.

Attention projects each 514-value step to 64, adds learned positions, applies
one four-head Transformer encoder layer, and joins the final-position summary
with the 512-value goal. The final MLP maps 576→64→2. Attention operates across
frames, not pixels. Internal attention feedforward uses GELU; the classifier
initially used ReLU and later supports LeakyReLU (.01 negative slope).

Optional GRU/LSTM controls and width/position sweeps exist in code but are not
completed experiments in this log. These controls do not replace teammate models.

## Completed development results

All rows use the protocol above unless the change column specifies otherwise.
Scores are validation-selected, owner-reported aggregates. Repeated diagnostic
runs reproduced earlier displayed scores and are not independent experiments.

| Date | Model / change | Selected train BA | Selected validation BA | Outcome |
|---|---|---:|---:|---|
| Nov 3 | Latest MLP, raw features, ReLU | .577 | .597 | Fast static reference; later validation deteriorates |
| Nov 3 | Mean-history MLP, raw features, ReLU | .596 | .601 | Tiny BA difference; substantially slower |
| Nov 3 | Attention, raw features, ReLU | .500 | .500 | Epoch-1 checkpoint predicts RETRACT throughout validation |
| Nov 4 | Latest+goal, raw features, ReLU | .500 | .500 | Subsequent health check confirms inactive hidden layer |
| Nov 4 | Goal+metadata, raw features, ReLU | .575 | .561 | Final train/validation BA .637/.490 |
| Nov 5 | Latest+goal, raw features, LeakyReLU | .500 | .500 | Gradients restored; weak score separation/generalization |
| Nov 5 | Latest+goal, standardized, LeakyReLU | .964 | .588 | Training fit restored; strong overfitting remains |
| Nov 5 | Same, dropout .3 | .941 | .584 | Modest probability-metric improvement; no clear BA gain |
| Nov 5 | Same reference, weight decay .01 | .964 | .588 | Negligible change in this short run |

Latest MLP retract/advance recalls were .824/.370; mean-history recalls were
.941/.261. The small BA difference conceals a stronger retract bias, not balanced
improvement. The full static reference reproduces .597 BA in the input-control run.
Comparisons across different normalization/activation settings cannot isolate
metadata value. Latest+goal versus goal+metadata changes two input sources.
Parameter counts also differ; no input or encoder superiority is established.

## Experiment record and reasoning

### Nov 3 — Integration check and static/mean/attention comparison

**Reason for the experiment:** frozen-feature logistic probes established an
initial reference, but did not train temporal models. The neural comparison
retained a final MLP for nonlinear fusion, comparing a latest-image reference,
unordered mean history and learned temporal attention. The eight-case check
preceded development evaluation to verify updates and encoder freezing.
Frozen ResNet limited training cost and capacity, with the tradeoff that visual
features could not adapt to guidewire decisions.

The real eight-case CNN–attention check completed 300 updates: training loss
.692842→.000071, accuracy 50%→100%, finite gradients, changed head weights and
unchanged CNN. This verifies tiny fitting only.

First A100 same-case batch-one median totals were 33.07 ms for latest MLP,
131.17 ms for mean-history and 132.24 ms for attention. Latest encoded two images;
temporal models encoded seven. Preparation/model medians were 27.19/5.89,
112.71/18.42 and 113.26/18.77 ms respectively. These exclude disk IO, assume no
feature reuse and are not workload-wide latency estimates. The temporal cost
was largely preparation/encoding; isolated attention latency was not measured.

### Nov 4 — Per-epoch learning and checkpoint diagnostics

**Reason for the experiment:** attention selected epoch 1 at chance BA. That
score alone could not distinguish a training failure from later overfitting.
Full-partition training metrics and selected/final checkpoint inspection were
added before increasing attention width. Frame routing, input replacements and
representation probes examined input dependence and changes through the model.

Per-epoch logging distinguishes selected from final behavior:

| Raw-feature model | Best / final epoch | Selected train / validation BA | Final train / validation BA |
|---|---|---|---|
| Latest MLP | 2 / 17 | .577 / .597 | .699 / .478 |
| Mean-history MLP | 5 / 20 | .596 / .601 | .621 / .487 |
| Attention | 1 / 16 | .500 / .500 | .636 / .425 |

Training-partition attention routing shifts from near-uniform at epoch 1 to
mild recency preference in three heads at epoch 16; the fourth favors earlier
frames. Routing changes have little accompanying history-input sensitivity.
Replacing the goal with its training-mean embedding changes final probabilities
by .0332 on average, flips 35.2% of decisions and reduces training BA by .136.
History/metadata replacements have much smaller effects. This identifies measured
goal-path dependence, not red-marker reasoning, leakage or goal-only performance.

Representation variation falls through projection/summary, but scale/rank changes
do not identify lost semantic detail. Depth/time ridge probes recover summary
metadata poorly; imperfect recovery even from raw inputs containing the targets
limits interpretation. Oracle/mean controls would be needed for stronger claims.
Attention averages and perturbation maps are diagnostic, not causal explanations.

### Nov 4 — Static input-role controls

**Reason for the experiment:** goal-replacement sensitivity and the earlier
DINOv2 goal-only probe motivated testing simpler input combinations. Fixed-input
replacement can create unfamiliar examples; separately trained controls test
whether useful prediction can be learned with fewer input sources.

Latest+goal excluded metadata; goal+metadata excluded history imagery; the full
latest+goal+metadata MLP remained the reference. Compared against that reference,
the first control tests adding numerical history and the second tests adding
the latest image. Neither control uses earlier history images. The neural
goal-only proposal was replaced by latest+goal before execution.

**Result and next decision:** selected validation BA was .500 for latest+goal,
.561 for goal+metadata and .597 for the reference. Latest+goal remained at .500
training BA through all 16 epochs, requiring a training-health inspection before
attributing its result to missing metadata. Input dimensions and parameter counts
differ, so these are preliminary controls rather than isolated causal tests.

### Nov 5 — Inactive ReLU inspection and LeakyReLU comparison

**Reason for the experiment:** constant class predictions and falling loss did
not demonstrate case discrimination. Class-balanced loss permits a constant
predictor near probability .5. Hidden activations, probability spread, AUC and
loss gradients were inspected to distinguish threshold behavior from a blocked
learning path.

The raw latest+goal ReLU model predicts a single class each epoch. Final health
inspection finds all 64 hidden units zero across 267 training cases, all hidden
preactivations negative, constant probability .50386977 and AUC .500. Only the
output-bias loss gradient is nonzero. This invalidates interpreting the run as
proof that metadata is necessary or image inputs lack information.

**Reason for the change:** confirmed inactive ReLU units motivated LeakyReLU,
which permits gradients on negative inputs while retaining the classifier
structure. Initialization changes were deferred to keep the remedy focused.

LeakyReLU restores hidden gradients, but all inspected preactivations remain on
the negative branch. Final probabilities span .50449–.50841, training AUC .687,
validation AUC .478 and BA .500 on both partitions. The gate failure is resolved;
useful validation discrimination is not. Training AUC improves .508→.687 while
validation AUC declines .641→.478; this is more than threshold misalignment.

### Nov 5 — Training-only feature standardization

**Reason for the experiment:** LeakyReLU restored gradients, but preactivations
remained negative and probabilities tightly clustered. Input conditioning was
a plausible remaining obstacle. Centering/scaling frozen features tested that
hypothesis without adding layers or width; changing initialization simultaneously
would have made the result harder to interpret.

For latest+goal, the training feature matrix is 267×1024: 512 latest-image and
512 goal-image coordinates. Per-column population mean/std are fitted on training
only, separately by role, then reused in validation/inference. Std ≤ 1e-6 uses
scale 1. This transforms CNN embeddings after pixel preparation; original images,
red markers and labels are unchanged. Metadata is excluded. Statistics are saved
as nontrainable buffers; initialization and parameter count remain unchanged.

| Standardized LeakyReLU state | Train BA / AUC / loss | Validation BA / AUC / loss |
|---|---|---|
| Initial | .487 / .470 / .7509 | .547 / .685 / .7058 |
| Selected epoch 8 | .964 / .996 / .1880 | .588 / .580 / .7652 |
| Final epoch 23 | 1.000 / 1.000 / .0133 | .510 / .543 / 1.1345 |

Conditioning was a substantial training obstacle under this recipe. Perfect
training fit does not establish transferable wire/target reasoning. Scaling also
changes effective regularization, so centering/scaling effects are not isolated.
An initially favorable random-head validation AUC is not a seed-selection strategy.

### Nov 5 — Dropout increase, then AdamW weight decay

**Reason for the experiments:** standardization restored near-perfect training
fit while validation deteriorated. Overfitting became the immediate concern;
more capacity was not justified by insufficient training fit.

**Step 1 — Dropout .3:** increase from .1 while retaining decay .001. Randomly
suppressing more hidden activations tests reduced reliance on individual units
without adding inference computation. Selected BA .584 versus .588 did not show
a clear gain, although validation AUC/loss improved modestly.

**Step 2 — Weight decay .01:** restore dropout .1 and increase decay from .001.
This tests stronger AdamW weight shrinkage independently of the dropout change.
Selected BA remained .588. The completed sequence did not justify selecting a
new regularizer or expanding the search on the same validation cases.

| Standardized LeakyReLU setting | Selected train BA | Selected validation BA / AUC / loss | Final validation BA / loss |
|---|---:|---|---|
| Dropout .1, decay .001 | .964 | .588 / .580 / .7652 | .510 / 1.1345 |
| Dropout .3, decay .001 | .941 | .584 / .597 / .7217 | .540 / 1.088 |
| Dropout .1, decay .01 | .964 | .588 / .579 / .7643 | .510 / 1.127 |

Higher dropout modestly reduces fitting and improves some probability metrics.
Increasing decay has little effect. At lr .001, decay .01 and 207 updates, isolated
decay retention is approximately .99793, ignoring gradient interactions: this was
a mild short-budget test, not evidence that weight decay generally cannot help.
No reliable winner is established from small differences on 80 related cases.

## Decisions and next steps

- Retain standardized latest+goal LeakyReLU, dropout .1 and decay .001 as the
  functioning reference. This is not the highest validated final-policy result.
- Stop the completed regularization sequence; defer width, position, initialization
  and threshold sweeps. Current capacity can memorize training labels.
- Investigate associations that transfer across episodes. Input-role comparisons
  require matching normalization/activation; standardization currently supports
  `latest_goal_mlp` only, so extensions remain unimplemented.
- Keep test reserved. Single-seed, repeatedly selected validation scores and
  incompatible protocols do not establish superiority or that history is noise.

## Nov 5 — DINOv2 temporal-attention comparison

**Reason:** the CNN attention run did not establish a history benefit. Substituting
DINOv2 tests whether a different frozen visual representation changes temporal
prediction, while static/mean controls distinguish encoder changes from gains
specific to attention. DINOv2 Small yields 384 features instead of ResNet's 512;
projection/late-fusion input sizes and head parameter counts therefore differ.
Pretraining also differs, preventing a pure CNN-versus-ViT architectural claim.

**Protocol:** the existing runner supports this path; no new model
implementation is required. The initial comparison keeps the original raw-feature
ReLU protocol, width 64, seed 0 and the shared budget. This does not transfer the
standardized static model's improvement to attention. Commands and reporting
criteria are in [the run guide](CNN_ATTENTION.md#dinov2temporal-attention-comparison--prepared-2026-11-05).
**Execution:** owner-reported real Perlmutter run completed after installing the
recorded timm 1.0.30 in a scratch environment and restoring scratch HF cache
exports. The initial failures occurred before training (missing timm, then cache
locking); neither is an accuracy result. Exact revision remains in the private report.

| DINOv2 model | Selected train / validation BA | Retract / advance recall | Best / final epoch | Final train / validation BA | Head parameters |
|---|---|---|---|---|---:|
| Latest MLP | .694 / .597 | .324 / .870 | 24 / 39 | .775 / .558 | 50,114 |
| Mean-history MLP | .728 / .627 | .471 / .783 | 40 / 55 | .777 / .536 | 50,114 |
| Attention | .746 / .565 | .412 / .717 | 22 / 37 | .951 / .478 | 87,490 |

**Interpretation and next decision:** mean pooling exceeds latest by .030 BA
with the same head parameter count, trading higher retract recall for lower
advance recall. This is the clearest current lead for historical-image value,
but does not demonstrate useful visual ordering: mean pooling discards image
order while metadata retains chronological positions. Attention underperforms
both controls and reaches .951 training BA with .478 final validation BA,
supporting overfitting rather than a complete inability to fit training data.

Compared with original ResNet results, DINOv2 latest BA is essentially unchanged,
mean rises .601→.627 and attention .500→.565. Latest class behavior nevertheless
changes sharply (ResNet recalls .824/.370 versus DINOv2 .324/.870). Equal BA does
not imply equivalent decisions. Encoder pretraining, feature distributions and
head parameter counts differ; no pure architectural superiority follows.

Retain mean pooling as the leading candidate for a bounded replication across
predeclared seeds before selecting a final method. Do not enlarge attention from
these scores alone. Repeated validation selection and one seed limit confidence;
paired episode-level errors and latency remain to review. No new run is started
or final test result claimed.

## Nov 5 — DINOv2 latest versus mean, three-seed replication

**Reason:** seed-0 mean pooling leads latest by .030 BA with equal head size,
but a single initialization cannot establish stability. This bounded replication
adds seeds 1 and 2 for both heads; existing seed 0 is retained. No GRU training
or additional attention tuning is included.

**Fixed protocol:** frozen DINOv2, split 0, raw features, ReLU, hidden width 64,
dropout .1, AdamW lr .001 / decay .001, batch 32, at most 100 epochs, patience 15,
earliest strictly best validation BA. Initial weights/optimizer are fresh per
model/seed. One invocation reuses extracted features across the four new fits.
Batch-one latency uses the existing benchmark scope.

**Review rule:** report all three seeds, each paired mean-minus-latest BA,
mean/sample standard deviation of each model's BA, both recalls and selected/final
learning behavior. Do not choose the best seed. Consistent positive differences
would justify further investigation, not prove generalization across unseen
episodes. Mixed signs would weaken the seed-0 history-benefit interpretation.
Compare latency before accepting the extra image cost. No further sweep is
queued; test remains reserved. The owner-reported seeds 1/2 run completed.

| Seed | Latest selected BA | Mean selected BA | Mean minus latest |
|---|---:|---:|---:|
| 0 | .5965 | .6266 | +.0301 |
| 1 | .5825 | .6010 | +.0185 |
| 2 | .5607 | .6081 | +.0473 |
| Mean ± sample SD | .5799 ± .0180 | .6119 ± .0132 | +.0320 ± .0145 |

| New model/seed | Selected train BA | Retract / advance recall | Best / final epoch | Final train / validation BA |
|---|---:|---|---|---|
| Latest / 1 | .656 | .882 / .283 | 4 / 19 | .700 / .489 |
| Latest / 2 | .641 | .382 / .739 | 18 / 33 | .795 / .511 |
| Mean / 1 | .642 | .441 / .761 | 22 / 37 | .764 / .533 |
| Mean / 2 | .679 | .412 / .804 | 38 / 53 | .775 / .547 |

**Finding and next decision:** mean pooling leads for all three paired seeds,
with average BA improvement 3.20 percentage points. Equal classifier dimensions,
shared starting seeds, unchanged metadata and the same frozen encoder strengthen
the evidence for averaging historical imagery under this protocol. This is not
evidence for visual order or action-relevant motion reasoning: averaging changes
image content and can also smooth noisy features.

Latest recall balance is unstable across seeds (seed 1 favors retract sharply).
Mean recalls are more consistent here but remain advance-biased: retract
.412–.471, advance .761–.804. Every fit finishes with higher training BA and lower
validation BA than at selection; overfitting remains. Standard deviations describe
initialization variation on the same development cases, not confidence intervals
or uncertainty over unseen episodes. No statistical significance is established.

Retain mean pooling as the leading DINOv2 control and stop this seed comparison.
Review the latency tradeoff below and paired episode errors, rather than pick a lucky
seed or expand attention capacity. Attention has only seed-0 evidence and cannot
be compared as a three-seed mean. No new training is queued.

### Latency and wrap-up

Both DINOv2 reports record clean revision
`f0e2d37783621699e479b51141660462ce845f95`, with the same training protocol.

| Model | Validation BA, mean ± sample SD | Per-seed total median latency, ms |
|---|---|---|
| Latest MLP, seeds 0/1/2 | .580 ± .018 | 45.37 / 45.37 / 45.26 |
| Mean-history MLP, seeds 0/1/2 | .612 ± .013 | 171.29 / 159.83 / 159.29 |
| Attention, seed 0 only | .565 | 159.94 |

Averaging the three reported per-seed medians gives 45.33 ms latest and
163.47 ms mean-history: approximately 3.6 times the latency for a 3.20-point
average BA gain. These are synchronized same-case batch-one microbenchmarks,
including preparation/transfer/encoding/head, excluding disk IO, with no feature
reuse. They are not pooled workload medians or independent latency replications;
run-to-run timing variation does not identify temporal-head overhead.

**Stopping point:** retain latest as the faster reference and mean pooling as
the strongest replicated DINOv2 control. Temporal attention has not earned its
complexity in the tested settings; this does not establish that attention cannot
help. No final method is selected. Further attention work should test a specific
training/fusion hypothesis against these controls, rather than expand capacity
without evidence. GRU implementation/tuning remains separate. Test data stays
reserved; no further training is queued.

## Artifacts and verification

Private runs live under `$PSCRATCH/waldo-temporal/`:

| Directory | Evidence |
|---|---|
| `dinov2-attention-dev64-01` | DINOv2 latest/mean/attention comparison |
| `dinov2-latest-mean-seeds12-01` | Latest/mean replication, seeds 1 and 2 |
| `tiny-attention-01` | Eight-case check |
| `dev64-01`, `dev64-diagnostics-01` | First comparison and epoch metrics |
| `attention-state-01` | Selected/final attention checkpoints |
| `input-controls-01` | Static input-role controls |
| `latest-goal-leaky-01` | Activation comparison |
| `latest-goal-standardized-01` | Conditioning reference |
| `latest-goal-standardized-drop03-01` | Dropout comparison |
| `latest-goal-standardized-wd01-01` | Weight-decay comparison |

`metrics.json` is the source for full settings, curves and provenance. Raw reports,
checkpoints, scales, predictions and single-case spatial outputs remain on
Perlmutter; reviewed aggregate summaries are shareable. Exact run revisions must
come from saved reports, not inferred from directory names.

Latest code verification: 65 synthetic tests passed, covering input exclusions,
training-only statistics, gradients, freezing, selected/final reload, diagnostic
controls and report integration. Synthetic scores are not guidewire performance.
[Run guide](CNN_ATTENTION.md) contains current commands; superseded plans remain
available in Git history.
