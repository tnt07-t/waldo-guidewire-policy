# CNN–attention development experiments

## Current conclusion — 2026-11-05

Feature standardization resolved a major training obstacle in the latest+goal
MLP. Generalization remains weak: training reaches 100% balanced accuracy while
validation returns near chance. Higher dropout and weight decay produced no
clear validation-BA improvement. No final model is selected.

Historical images have not demonstrated a benefit in the tested pipelines.
Their usefulness remains unresolved because training failures, input conditioning
and architecture differ across comparisons. Metadata necessity and goal-marker
understanding are not established.

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

## Artifacts and verification

Private runs live under `$PSCRATCH/waldo-temporal/`:

| Directory | Evidence |
|---|---|
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
