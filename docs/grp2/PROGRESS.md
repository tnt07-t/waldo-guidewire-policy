# Group 2 — Transformer models

Read the [shared approach notes](../APPROACHES.md), [dataset notes](../DATASET.md), and [project README](../../README.md) before continuing. This file holds Group 2 plans, progress, and permitted aggregate results. Shared environment, preprocessing, interfaces, and evaluation decisions belong in the approach notes.

## Proposed scope — supplied by project owner

- ViT–GRU: replace CNN with a pretrained vision transformer, retaining GRU and consistent inputs.
- CNN–temporal attention: replace GRU with temporal self-attention, retaining CNN and consistent inputs.

These are team proposals, not confirmed course requirements. Friday task ownership is recorded below; exact architecture ownership remains open. The shared optional JEPA extension remains in the approach notes.

## Current status and next steps

**2026-11-05 — Regularization sequence completed:** decay .01 with dropout .1
reproduces selected BA .588 at epoch 8; final train/validation BA 1.000/.510.
Selected validation AUC/loss .579/.7643 versus baseline .580/.7652. Dropout .3
has modest AUC/loss gains but BA .584. No clear winning regularizer; short-budget
AdamW shrinkage is small and results do not rule out other strengths. Stop the
planned sweep and retain the original standardized recipe as reference. Full
table, decay-scale calculation and follow-up limits are in the experiment log.

**2026-11-05 — Dropout .3 mixed result:** selected epoch 8 validation BA .584
versus .588 baseline; AUC improves .580→.597 and loss .7652→.7217. Selected
training BA falls to .941, but final training remains .994 and validation .540.
This is modest regularization without clear primary-metric improvement.
Next agreed comparison: AdamW decay .01 with dropout restored to .1. No new
width or automatic regularization sweep is selected.

**2026-11-05 — Regularization sequence agreed:** first test dropout .3 with
standardized latest+goal LeakyReLU, decay .001 and learning rate .001. If the
reviewed result gives no useful validation improvement, next candidate is decay
.01 with dropout restored to .1. This isolates regularizers and retains the
existing training/selection budget. No job is chained automatically and neither
new real result is claimed. Commands and criteria are in the experiment log.

**2026-11-05 — Regularization considered:** the standardized model already uses
dropout .1, AdamW decay .001 and validation checkpoint selection. Proposed next
single-factor comparison is dropout .3 with the rest fixed; no new layer/code
or real run is introduced. Reduced training fit alone is not success. The
experiment log explains the standardized 267×1024 frozen-image feature matrix
and records the proposed budget, command and interpretation limits.

**2026-11-05 — Standardization restores fit, not generalization:** latest+goal
LeakyReLU selects epoch 8: train/validation BA .964/.588, AUC .996/.580.
Final epoch 23 reaches train BA/AUC 1.000/1.000, but validation BA .510,
AUC .543 and loss 1.1345. Conditioning was a substantial training obstacle under
the fixed recipe; model width is sufficient to fit training labels. Clear
generalization failure remains. Input controls still differ in activation and
normalization, so .588 versus earlier .597/.561 cannot isolate metadata benefit.
Detailed results and follow-up limits are in the experiment log.

**2026-11-05 — Conditioning experiment prepared:** added training-only per-role
feature standardization for latest+goal LeakyReLU. Statistics are nontrainable
checkpoint buffers; validation/inference/diagnostics reuse the training transform.
Initialization, parameter count, metadata exclusion and optimization stay fixed.
All 65 synthetic tests pass, including leakage/refit guards and end-to-end reload.
Real run pending; see the
[standardization protocol](ATTENTION_EXPERIMENTS.md#training-only-image-feature-standardization--2026-11-05).

**2026-11-05 — Ranking generalization fails:** latest+goal LeakyReLU train AUC
improves .508→.687 while validation AUC declines .641→.478; selected validation
AUC .543. Restored gradients do not imply restored generalization, and the
failure is not only threshold alignment. Initial random-head validation ranking
is not a model-selection strategy. No initialization/threshold sweep is chosen;
feature conditioning remains a candidate for a single controlled follow-up.

**2026-11-05 — LeakyReLU restores gradient flow:** final latest+goal health shows
no zero-output hidden units and nonzero hidden/output gradients. Training AUC
.687, but all scores remain .5045–.5084 and BA stays .500. The gate failure is
resolved; validation usefulness remains unestablished. Default Linear initialization
was verified; activation-matched Kaiming is a candidate, not a chosen fix. Next
inspect saved validation AUC before another training change. Detailed numerical
results and initialization limits are in the experiment log.

**2026-11-05 — LeakyReLU result:** latest+goal selected/final training and
validation BA remain .500 after 16 epochs (best epoch 1). No classification
improvement demonstrated under the fixed recipe. Final checkpoint activation,
probability spread/AUC and gradients remain to inspect; identical BA does not
establish that the original zero-gradient mechanism persists. No new training
configuration is selected yet.

**2026-11-05 — Inactive ReLU confirmed:** final latest+goal head has 64/64 hidden
units inactive on training, negative preactivations throughout, constant .50386977
ADVANCE probability, AUC .500 and zero hidden/output-weight loss gradients.
Only output-bias gradient remains. Added an explicit fixed-slope LeakyReLU
classifier option as one bounded remedy; ReLU default and training recipe remain
fixed. Real remedy results are pending. See the
[activation comparison](ATTENTION_EXPERIMENTS.md#confirmed-inactive-classifier-and-activation-control--2026-11-05).

**2026-11-05 — Latest+goal failure narrowed:** every reported epoch has train/
validation BA .500; predictions switch from all RETRACT to all ADVANCE while
probabilities approach .5. The loss decline is compatible with near-constant
prediction. Added read-only static classifier-health checks for inactive ReLU
units, score spread and loss gradients; no new training or model change.
Inactive ReLU remains a hypothesis pending real checkpoint inspection. See
[failure inspection](ATTENTION_EXPERIMENTS.md#no-metadata-mlp-training-failure-inspection--2026-11-05).

**2026-11-04 — Static input-role results:** latest+goal without metadata selected/
final BA .500; goal+metadata selected validation .561, final train/validation
.637/.490; full static reference selected validation .597, final .699/.478.
The full static reference remains the highest observed point in this comparison.
Weak no-metadata training fit limits claims about metadata necessity or noise.
No earlier visual history was processed by these controls. Inspect the existing
no-metadata learning history before a follow-up; results and limitations are in
[input-role results](ATTENTION_EXPERIMENTS.md#2026-11-04--completed-static-input-role-comparison).

**2026-11-04 — Static input-role controls prepared:** added latest+goal without
metadata and goal+metadata without history imagery, alongside the full static
reference. Hidden MLP/training/checkpoint rules remain fixed; parameter counts
and image IO differ explicitly. Gradient tests verify excluded input roles.
The static extraction path skips earlier images and benchmarks consumed roles
only. Real training/results remain pending; see the
[control protocol](ATTENTION_EXPERIMENTS.md#static-input-role-controls--2026-11-04).

**2026-11-04 — Selected/final attention inspection completed:** attention-only
replay reproduces .500 selected validation BA and .636/.425 final train/validation
BA. Training-partition routing becomes mildly recency-biased in three heads,
but history/metadata replacements remain weak. Goal replacement changes mean
probability .0332, flips .352 of training predictions and reduces BA by .136.
This establishes input sensitivity, not marker reasoning or leakage. Next
candidate is a trained goal+metadata control with matched recipe; capacity
changes remain deferred. Detailed results and limits are in the
[checkpoint comparison](ATTENTION_EXPERIMENTS.md#selected-versus-final-checkpoint-inspection).

**2026-11-04 — Final class behavior reviewed:** final attention predicts both
classes, with train recalls .526/.747 versus validation .176/.674 (retract/advance).
Mean favors ADVANCE more strongly; latest validation counts are nearly balanced
and still weak. Constant prediction does not explain final behavior. Added
separate final-state checkpoint persistence and explicit role/epoch inspection
for subsequent runs; the selected model and training recipe remain unchanged.
Existing final weights cannot be recovered from metrics. No new run is claimed.

**2026-11-04 — Per-epoch learning evidence:** unchanged fixed-64 run reproduces
selected BA at displayed precision. Final train/validation BA: latest .699/.478,
mean .621/.487, attention .636/.425. Attention learns training labels beyond
epoch 1 while validation worsens; increasing width is not the first supported
response. Review class behavior/loss and generalization controls. Earlier routing
inspection applies only to the selected epoch-1 head. See
[learning results](ATTENTION_EXPERIMENTS.md#2026-11-04--full-per-epoch-logging-run).

**2026-11-04 — Epoch-1 attention inspection:** supplied training-partition report
shows near-uniform case-averaged routing and minimal prediction sensitivity to
visual history/metadata. Goal replacement changed probabilities more (.0108
mean absolute change), with no flips for any control. Summary depth/time linear
recovery is weak, but imperfect raw-target recovery limits loss attribution.
No causal focus or semantic loss is established. Full per-epoch training metrics
and decoder controls remain next; no width sweep is justified by this report.
See [diagnostic results](ATTENTION_EXPERIMENTS.md#2026-11-04--selected-checkpoint-training-diagnostics).

**2026-11-04 — Attention learning/representation diagnostics:** added complete
per-epoch train/validation BA, recalls, confusion/prediction counts, probability
means and eval loss, preserving the checkpoint rule. Initial/final/selected
metrics are distinct and epoch histories are saved incrementally. Separate
`temporal_diagnostics.py` inspects existing checkpoints: temporal routing,
fixed-input interventions, representation spread, training-only metadata
recoverability and optional one-case spatial perturbations. No architecture
change or new real result is claimed. Execution/rationale/limitations are in the
[diagnostic plan](ATTENTION_EXPERIMENTS.md#learning-and-representation-diagnostics--2026-11-04).

**2026-11-03 — End-of-day decision:** share completed tiny/full-run evidence
and defer the width/position sweep. Larger width remains a hypothesis; current
logs do not distinguish weak training fit from validation-selected checkpoint
effects or optimization issues. Next planned change is per-epoch training
metrics/prediction counts and input-magnitude diagnostics before a bounded
64/128 comparison. No further experiment has run. The tiny report explicitly
confirms finite gradients, changed head weights and unchanged encoder; timing
details and critical assessment are in [experiment results](ATTENTION_EXPERIMENTS.md).

**2026-11-03 — Fixed-64 development results:** supplied seed-0 reports show
latest/mean/attention validation BA .597/.601/.500. Attention selected epoch 1
with RETRACT-only predictions; loss decreased modestly across later epochs but
validation never improved. Latest/mean selected epochs 2/5. Same-case total
median latency 33.07/131.17/132.24 ms, excluding disk IO and feature reuse.
Latest MLP remains the provisional efficient reference; width/position sweep
is deferred pending optimization/input diagnostics. Full evidence, uncertainty
and comparison limits are recorded in [experiment results](ATTENTION_EXPERIMENTS.md#2026-11-03--first-fixed-64-development-comparison).

**2026-11-03 — Full-run order revised:** The project owner supplied guidance to establish
a fixed-architecture full training/validation result before architectural
ablations. Next run: 64-wide attention with positions, plus latest/mean MLP
controls, fixed budget and validation BA checkpoint rule. Width/position sweep
is implemented but deferred pending review. See the updated
[experiment plan](ATTENTION_EXPERIMENTS.md). No full-run result is recorded yet.

**2026-11-03 — CNN–attention tiny fit completed (owner-reported):** the
A100-SXM4-80GB run completed 300 updates; training loss 0.692842→0.000071,
eight-case accuracy 1.000, status `completed`. This checks real-data learning
plumbing only. Full report details and validation/latency comparison remain to
review/run. See [experiment results](ATTENTION_EXPERIMENTS.md#2026-11-03--eight-case-perlmutter-check-completed).

**2026-11-03 — Bounded CNN–attention check:** added separate
`cnn_attention_debug.py` for eight balanced split-0 training cases before the
full sweep. Adapts the team's ViT–GRU runtime/selection/freezing checks to frozen
ResNet and the 64-wide attention head. Maximum 300 updates/120 seconds of head
work, 600-second worker deadline; no validation tuning/checkpoints/cache.
Three synthetic worker tests passed. Perlmutter run remains pending; see the
[experiment log](ATTENTION_EXPERIMENTS.md#eight-case-integration-check).

**2026-11-03 — Testing handoff documented:** aligned attention documents with
the existing scope/protocol/verification/run-guide layout and added dependency
checks, one-GPU allocation/job-step commands, CUDA checks, scratch setup,
eight-configuration execution, result inspection and allocation cleanup.
Code must be transferred before requesting GPU time. No real run or result is
implied by this documentation update; the previously passing synthetic suite
is unchanged. NERSC's current interactive-job guidance was checked for GPU
requests in both `salloc` and `srun`.

**2026-11-03 — Requested attention ablations:** The project owner retained the final MLP
and requested position embeddings on/off and widths 32/64/128, together with
static and mean-history comparisons. `--attention-sweep` implements eight
configurations, fixes final MLP hidden width at 64, and shares extraction once
in RAM. [Experiment log](ATTENTION_EXPERIMENTS.md) records decisions, ReLU/GELU,
scaling/optimization, capacity and selection tradeoffs, run commands and pending
results. Local invented-data tests are code verification only. Real-data runs
could not be launched: Perlmutter SSH authentication was rejected. No new
accuracy/latency measurements or winning-model decision exists yet.

**2026-11-03 — CNN–attention setup:** The project owner requested the
CNN–temporal attention implementation and a shared baseline/evaluation pipeline
while Tran runs the eight-case ViT–GRU check (user-reported in progress; no result
supplied yet). Added `temporal_models.py`, `temporal_evaluation.py`,
`compare_runs.py` and invented-input tests. The frozen ResNet feeds a 64-wide,
four-head, one-layer temporal attention encoder with learned positions and late
goal fusion. A latest+goal+full-metadata MLP is the default comparison; optional
mean-history/recurrent controls share the runner. See [CNN_ATTENTION.md](CNN_ATTENTION.md)
for architecture, development defaults, Perlmutter commands, private checkpoint
handling and result tracking. No existing reader/preprocessor/encoder/static
evaluation/notebook or Tran code is changed. Real-data extraction/training,
A100 latency and validation performance of this setup remain unverified.
Tran's remote debug branch is now published at `b9d1bdf`; earlier unpublished
statements below are historical. The new local setup is not automatically
available in Perlmutter and has not been committed or pushed.

**Verification:** all 42 invented-input tests passed in the local `.venv-eval`
environment (torch 2.8.0 CPU), including nine new temporal tests. Checks cover
all-head backward passes, case independence, attention gradients through every
history step/goal/metadata, order sensitivity, static input roles, train-only
scaling, actual optimization/checkpoint reload, frozen encoder state, latency
image counts, ledger formats and a complete invented-image CLI/report smoke.
Both command help screens and diff formatting passed. Synthetic results do not
establish real-data accuracy, pretrained ResNet integration or A100 latency.

**The real eight-training-case ViT–GRU debugging check passed on Perlmutter (Tran-reported, 2026-11-03).** At `b9d1bdf`, 300 updates reduced training loss from 0.775900 to 0.000129 and reached 100% accuracy on those same eight cases. Gradients were finite, head weights changed and encoder state stayed unchanged. The GPU allocation was explicitly released. This verifies bounded real-data fitting and the frozen training path, not held-out performance or temporal usefulness.

**Current setup:** the separately bounded full-training/validation runner and synthetic tests are published at `69fd063` on `tran/vit-gru-debug` with the [agreed settings](GRU_TRIAL.md#agreed-first-full-training-setup--2026-11-03). Real full training is pending; start with fresh GRU/classifier weights. The eight-case debugging run is complete and does not need repeating. The [debugging handoff](GRU_DEBUG_HANDOFF.md) provides commands, expected evidence, failure handling and subsequent steps. [GRU_TRIAL.md](GRU_TRIAL.md) retains the design rationale and later comparisons. No full training, validation tuning, caching expansion or final evaluation is included in this milestone.

**Existing evidence:** raw loading, full-view preparation and pretrained DINOv2 CPU extraction/grouping passed prior checks. Tran's notebook imports timm 1.0.30 and reports CUDA available. The one-frame masking result is recorded and inconclusive for temporal usefulness; do not repeat masking. The static evaluation separately completed static frozen-feature probes and a reported GPU run; the unchanged evaluation module/tests are imported for later reuse. Tran's own GPU extraction/backward path is now verified within the eight-case check; static results remain a separate experiment.

**Latest aggregate findings:** the first split-0 results have now been supplied. DINOv2 goal-only has the highest observed validation BA (.609); DINOv2 latest+goal is .570 and ResNet latest+goal .547, with large train/validation gaps. Static encoder component medians are 46.83 ms total for DINOv2 and 33.87 ms for ResNet; classifier/disk IO are excluded. See EVALUATION.md for the canonical results, caveats and prioritized follow-up hypotheses. No statistically supported encoder ranking or final-test result is established; existing episode intervals and matched errors are next to inspect.

**Handoff:** implementation is on `static-evaluation`; the aggregate review and [teammate resume checklist](../EVALUATION.md#teammate-handoff--resume-here) document what is complete and what to inspect next. Keep CNN/ViT frozen. Private uncertainty/error evidence is next; no baseline rerun is needed just to resume.

**Update — 2026-11-01:** The project owner authorized the before-GRU evaluation milestone and ongoing documentation. `evaluation.py` compares fixed ResNet/DINOv2 features using separate regularized linear probes. Implementation, synthetic verification and the first real development run are complete. Earlier pending-execution or limited-authorization statements in dated entries describe historical stages. GRU and encoder fine-tuning remain outside this implementation.



**Working layout:** documentation stays in the original checkout on `main`; code stays in sibling `waldo-guidewire-policy-vit-gru-debug` on `tran/vit-gru-debug`. Both are based on `0f77490`. Main's current notes do not automatically appear in the debug checkout; read them here until branch synchronization is explicitly authorized. Both branches were published in the main conversation; documentation has not been synchronized into the debug branch.

**Experimental limits:** tiny fitting checks learning plumbing only. Later full training must start with fresh head weights, compare 64 versus 128 using validation and prioritize the static-evaluation latest-image + goal + matching historical-metadata baseline before the broader four-model comparison. A functioning GRU or a perfect eight-case fit does not establish generalization, goal use or useful temporal processing.

## Iteration plan and run record

This section tracks experiment progression; [GRU_TRIAL.md](GRU_TRIAL.md#agreed-first-full-training-setup--2026-11-03) owns the settings and [the handoff](GRU_DEBUG_HANDOFF.md#full-training-setup--2026-11-03) owns commands. An iteration means one deliberately chosen experiment followed by review, not an automatic loop of GPU runs.

| Stage | Status | Completion evidence / next action |
|---|---|---|
| Learning plumbing | Complete | Eight-case real check passed at `b9d1bdf`; allocation released; do not repeat |
| Full-run preparation | Published at `69fd063` | Runner/checkpoints and 51 synthetic tests passed; explanatory model/training comments added |
| Remote readiness | Pending | Code is published; preserve notebook before an authorized Perlmutter update, then verify revision, existing environment, synthetic tests, cached weights and output location before allocation |
| First full development run | Not started | One fresh 64-unit model, split 0, seed 0 under agreed limits |
| Results review | Pending | Release GPU first, then inspect training/validation curves and best checkpoint on Perlmutter |
| Follow-up comparison | Not selected | Choose one justified question and agree its budget; no automatic sweep |
| Final test evaluation | Reserved | Only after model choices and team reporting protocol are fixed |

**Allocation sequence:** prepare code/environment before requesting GPU time. Use the existing Perlmutter GPU workflow with a 15-minute wall-time ceiling; verify account/resource arguments at execution time. The runner is bounded to 600 seconds overall, 300 seconds of extraction and 120 seconds of training/validation (phase limits checked between operations). Execute one run, then release the allocation immediately on success, failure or timeout and confirm release. Do not keep a GPU allocated while discussing plots, editing code, or choosing the next experiment. The supervisor stops its worker but does not cancel the allocation.

**Review before another allocation:** if the run fails, preserve its private error/report and fix the specific issue locally where possible. If training loss drops but validation worsens, investigate overfitting and propose one regularization comparison. If training itself does not improve, inspect optimization/alignment before changing architecture. If validation is still improving at a cap, discuss a bounded extension; the present runner does not resume training. Once the first run is usable, prioritize the matched latest-image + goal + historical-metadata comparison and the planned 64/128 comparison under separately agreed budgets. Small validation differences require caution; seed repeats are a later proposal, not an approved automatic run list. See [when to try 128](GRU_TRIAL.md#when-to-try-128--decision-rule-recorded-2026-11-03) for plateau, overfitting and budget-limit interpretations and the matched comparison rules. This decision guidance was added at Tran’s request; no width change or new run was made.

**Record each real iteration at milestone level:** date/run purpose; code revision/dirty state; one intended change and its rationale; agreed versus actual work/time; stop reason; best epoch; training and validation loss/accuracy/balanced accuracy/class recalls/confusion counts; freeze/reload checks; checkpoint availability; allocation release confirmation; interpretation limits and concrete next decision. Keep raw reports, scales, rows, checkpoints and curves on Perlmutter; shared entries contain only permitted aggregates. Failures and capped runs must not be logged as completed 100-epoch training.

**First full-run record:** not started; no real full-training metrics, allocation ID or checkpoints exist yet. Current local implementation evidence is in the setup milestone below.

## Progress log

### 2026-11-03 — Supplied project objectives clarified

Tran supplied the objective and example-deliverable text. Recorded it once in [APPROACHES.md](../APPROACHES.md#supplied-project-objectives-and-deliverables--2026-11-03): 62% always-advance floor, aspiration toward Astra's 82%, overall and per-class accuracy, single-GPU per-decision inference latency/trade-offs for every variant, several model families and architectural/error analysis. The single-GPU condition applies to inference measurement; the excerpt states no training GPU count/type, time, batch-size or epoch limit. Owner-requested GPU budgeting remains an operational constraint. The official evaluation-set/split protocol and complete rubric remain open. This documentation clarification changes no approved experiment settings, code, compute allocation or data handling; no staging, commit or push occurred.

### 2026-11-03 — Full-training settings agreed; runner prepared locally

**Decision:** Tran approved logging and setting up the first full-training experiment after comparing candidate batch sizes, learning rates, epoch limits and regularization. Keep the frozen ViT and 64-unit GRU/classifier; use seed 0, Adam 0.001, batch 32, no weight decay, at most 100 epochs/900 updates and patience 20. Validation balanced accuracy selects the checkpoint, retaining the earlier checkpoint on ties. These are initial choices, not measured optima. [The trial plan](GRU_TRIAL.md#agreed-first-full-training-setup--2026-11-03) is the single reference for extraction, storage, timing and selection rules.

**Implementation:** added a separate full-training runner and tests in the sibling debug worktree. Fresh initialization, training-only metadata scaling, train/validation episode separation, frozen extraction reused in memory, metrics per complete epoch, bounded atomic checkpoint/report writes and verified checkpoint reload are implemented. The head architecture and training behavior are unchanged; plain-language comments now explain its dimensions, GRU memory and goal fusion, plus the new runner's extraction/update/validation/checkpoint flow. The tiny runner, loader, preprocessing, encoder and static evaluation remain unchanged. Latest/best checkpoints represent fully evaluated epochs; exact training resumption is outside scope.

**Verification:** seven new synthetic tests passed locally, including invented PNG integration through the real reader/preprocessor and injected encoder; actual head learning, fresh initialization, unchanged validation weights/gradients, checkpoint reload and early stopping, partial batches and failure limits are checked. All 51 synthetic tests passed locally in 1.285 seconds; CLI help and diff formatting passed. Existing CNN tests emitted only Pillow deprecation warnings. The full suite includes existing synthetic tiny-fit/masking tests, not repeats of the real Perlmutter experiments. No real data, pretrained downloads, full training, validation on real cases, masking, repeated tiny check or GPU allocation occurred.

**Handoff:** with Tran’s explicit authorization and confirmation, the full-training runner/tests and explanatory model comments were committed and pushed at `69fd063` on `tran/vit-gru-debug`; the saved commit message was verified against the approved subject and three scoped bullets. The debug worktree is clean. Main retains the prior uncommitted November 3 notes plus these setup updates; no project configuration/instruction files were added. Next: preserve remote notebook work before an authorized Perlmutter update, verify revision `69fd063` and the environment/tests before allocation, then separately initiate one bounded real development run. Release promptly and do not retry or expand automatically. Test stays reserved; real artifacts stay on Perlmutter.

### 2026-11-03 — Real eight-case GPU debugging check passed (Tran-run)

**Outcome and provenance:** Tran ran `vit_gru_debug.py` from `tran/vit-gru-debug` at `b9d1bdf523d01f045f64cd268dbb715d9bf63d37` on Perlmutter, using the existing module plus scratch venv. Required imports and CUDA availability passed; all 11 GRU synthetic tests passed there in 2.215 seconds before GPU allocation. The private report gives `git_dirty=true`; subsequent Tran-supplied Git status lists only untracked Python bytecode and an accidental terminal-viewer log, with no tracked source edits reported. Results below are supplied by Tran, not independently executed or read by assistant tools.

**Learning and freeze checks:** status `completed`; 300 updates, stop reason `max_updates`; initial/final cross-entropy training loss **0.775900 / 0.000129**; initial/final training accuracy **0.500 / 1.000** on the same eight selected training cases. Finite gradients, changed head weights and unchanged encoder state all reported true. Overall time limit was not reached. The supplied terminal output included a nonfatal unauthenticated public-weight download warning; download and execution succeeded without a token.

**Runtime and compute:** NVIDIA A100-SXM4-80GB; extraction **3.0601 s**, head work **1.0974 s**, whole process **17.2924 s**; peak CUDA allocated memory **132,605,440 bytes**. These are single-run training/extraction measurements, not online prediction latency or total GPU utilization. A 15-minute allocation was requested and explicitly cancelled after the run; revocation was confirmed. Exact allocation duration/charges were not measured. User requested preparing code/environment before allocating GPU time, short allocation limits, prompt release after checks and no automatic reruns or budget increases.

**Artifacts and interpretation:** the complete `debug_summary.json`, selected rows/scales and all real artifacts remain on Perlmutter; only permitted aggregates are recorded here. The runner saves no trained-head checkpoint or persistent feature cache. The result shows the implementation can fit eight real training examples with a frozen encoder. It does not measure validation/test performance, prove goal/visual-history use or establish optimal actions.

**Next:** prepare the full-training/validation runner and tests before requesting more GPU time. Agree on training and extraction budgets, batch size, validation/checkpoint metric, regularization and stopping rule; initialize a fresh head, fit scales only on training and keep test data reserved. Prioritize the later 64-versus-128 and matched-historical-metadata static-baseline comparisons under an agreed protocol. No full-training implementation or new GPU allocation is authorized by recording this result.

### 2026-11-02 — Bounded GRU setup implemented; real run pending

Following Tran's explicit setup request, implemented the 115,650-parameter head, split-0 four-per-label selection, training-only uncentered depth/time scales, frozen feature extraction and a fresh full-batch Adam loop in the separate debug worktree. Runtime/update caps match the documented plan; a parent process bounds the overall worker lifetime. Private reports retain configuration, selected rows/scales, source hashes and aggregate learning/runtime evidence on Perlmutter. No features or model checkpoints are persisted.

**Verification:** all 44 synthetic tests passed locally in 0.952 seconds, including 11 new GRU checks. Tests verify sequence/goal/metadata wiring, reset hidden state, deterministic balanced selection, training-only scaling and constant-channel handling, inference-tensor conversion, actual backward/optimizer updates, unchanged encoder parameters/buffers, non-finite rejection and both update/time limits. The invented-feature learning check reached perfect training accuracy within 40 updates and lower loss; this is not real task evidence. An additional synthetic worker smoke exercised the actual raw loader and preprocessor with invented PNGs/records, an injected encoder, 40 head updates and successful report serialization/source hashes; it saved only the summary and used no real data or pretrained weights. CLI help, syntax, local documentation links and diff formatting passed. Environment: Python 3.14.6, torch 2.15.0, NumPy 2.5.4, scikit-learn 1.9.1 and Pillow 13.0.0. Existing CNN diagnostic tests emitted Pillow deprecation warnings without failures.

**Handoff and limits:** added [GRU_DEBUG_HANDOFF.md](GRU_DEBUG_HANDOFF.md) with the static-evaluation reuse/coordination steps, local and Perlmutter commands, notebook preservation, private reporting and next experiment boundaries. No real data, GPU execution, pretrained extraction rerun, remote notebook changes or new baseline run occurred. Loader/preprocessor/encoder and the imported static-evaluation code are unchanged. At completion of implementation, main held the shared notes and debug held the code, all unstaged/uncommitted/unpushed. Tran subsequently requested separate commits by scope; confirmation immediately before committing is still required. Next: authorized code sharing, then one bounded Perlmutter run. Full-training and final-evaluation protocols remain separate work.

### 2026-11-02 — Separate debug worktree and static evaluation reuse

At Tran's request, created `tran/vit-gru-debug` and then moved it to a separate sibling worktree, keeping shared documentation on `main`. Rechecked live remote tips: `main` at `0f77490`, `static-evaluation` at `070328f`. Fast-forwarded local main from `0c4cbee` to `0f77490`, preserving every local documentation edit through a backed-up three-way reconciliation. The untracked GRU plan is preserved; `.DS_Store` files remain excluded. No staging, new commit or push occurred.

Inspected the static-evaluation evaluation implementation (`5ddb566`) and handoff (`070328f`). Imported `evaluation.py` and `tests/test_evaluation.py` unchanged into the debug worktree; brought the evaluation reference into main's documentation. Existing latest-image/goal/metadata probes, metrics and partition checks can support later comparisons. The scikit-learn optimizer, centered feature scaling, validation-selected regularization and full-partition runner do not implement the proposed tiny GRU check. No baseline rerun, GRU implementation, full training, data extraction or remote notebook update occurred. Current verification is recorded in the session handoff below.

### 2026-11-02 — Feature check completed; temporal-model design next

Tran ran the diagnostics shared in `0c4cbee` on Perlmutter. Goal-image comparison worked; numerical marker sensitivity was not reported. In the one-frame wire check, covering the tip changed the summary less than all three comparison edits. [Diagnostic notes](VIT_DIAGNOSTICS.md) retain the measurements and limitations. This static check cannot establish whether frame-to-frame information supports action prediction. Tran chose to stop additional masking and move toward a direct temporal-model trial; the proposed design remains unimplemented.

**Environment:** Tran confirmed torch 2.13.0+cu130, timm 1.0.30 and CUDA availability in the module-based notebook. The scratch venv also works but was not the final notebook interpreter. Exact diagnostic device and remote code version were not reported; no runtime/memory measurements were made. Real images, features and notebook outputs remain on Perlmutter.

**Verification:** local notebook syntax, cell IDs, empty outputs and diff formatting passed before sharing. Real-data execution is reported by Tran, not independently run by assistant tools. No model/training code was added.

**Design clarification:** at Tran's request, expanded [the GRU plan](GRU_TRIAL.md) with expected evidence from tiny fitting, a fresh start for later full training, separate validation/test roles and alternative split positions. Architecture ownership and training scope remain unconfirmed. Refreshed origin and confirmed `0f77490` remains one commit ahead of local `0c4cbee`; inspected the upstream CNN diagnostic changes and overlapping shared-document edits without reconciliation. Existing local work is preserved; no remote notebook or data was accessed, and no training, staging, commit or push occurred.

**Design review recorded:** inspected the proposed head and existing inference-only encoder. The frozen GRU trial is reasonable, with unresolved risks around whole-image summaries, late goal fusion, metadata-dominated prediction, limited independent data and unmeasured end-to-end latency. Recorded alternative width parameter counts and recommended earlier controlled comparisons in the plan rather than duplicating technical guidance here. Split 0 means the first supplied assignment object, not a verified standard cross-validation fold. Recommendations remain unapproved; no architecture or training settings were changed. Verification is limited to document/code inspection and local arithmetic/link/format checks; no new real-data evidence or model execution. Next: agree on the bounded trial and comparison protocol before implementation.

### 2026-11-02 — Documentation reconciled with completed CPU milestone

Reviewed overall and Group 2 notes against the recorded October 31 results. Updated README, shared approach status, and ViT trial instructions to remove superseded claims that preparation, timm compatibility, or pretrained CPU extraction were still pending. No new model execution or data verification occurred. The immediate next step remains the design discussion below; GPU verification/performance measurement and complete-policy implementation remain outstanding. Group 1's separate CNN GPU diagnostics do not establish Group 2 GPU readiness. Code, notebook edits, and dataset restrictions are unchanged.

### 2026-11-01 — Frozen backbone versus evaluation classifier clarified

The project owner confirmed the CNN should remain frozen for the coming experiments. Explicitly documented that evaluation fits a separate logistic-regression probe and train-only scaler on detached features; it does not modify `cnn.py`, update ResNet weights or BatchNorm statistics, or backpropagate into either image encoder. Fine-tuning remains a possible future research direction, not the next planned implementation. The current accuracy gaps describe probe generalization; no encoder was trained in this milestone.

### 2026-11-01 — First aggregate accuracy and component timing review

The project owner supplied the generated summary and encoder benchmark. Recorded the full 12-experiment BA/recall comparison and component timings once in EVALUATION.md. Main evidence: large train/validation gaps; goal-only DINOv2 above its latest+goal baseline; metadata and mean pooling did not improve rounded DINOv2 validation BA. DINOv2 encoding is approximately 3.8 times slower than ResNet at the shared 518-square preparation, but preparation reduces the measured total-time ratio to approximately 1.38. Single-case repeated timings exclude classifier/disk IO and shared-residency memory cannot rank isolated model sizes.

Proposed next order: inspect existing intervals/confusions/per-C scores and private matched errors; then a bounded stronger-regularization experiment with frozen encoders; investigate reference dependence, fusion and small-detail representation before limited fine-tuning. No new experiment was launched or model changed in this review. All real case-level artifacts remain on Perlmutter.

### 2026-11-01 — First before-GRU evaluation completed on GPU

After setting scratch HF/Torch cache locations, the project owner reported successful DINOv2 and ResNet weight downloads and the runner's completion message for split position 0, both encoders, CUDA and `--benchmark`. The preceding allocation check identified an A100-SXM4-80GB. This establishes owner-reported completion of the development run; scores, confusion matrices, uncertainty intervals, code/version provenance and component timings have not yet been reviewed. No superiority, fine-tuning gain, full-policy latency, or final-test result is claimed.

Next: inspect the aggregate summary and encoder component benchmark on Perlmutter, retain private case-level predictions there, and record aggregate findings and a bounded follow-up hypothesis. Scratch-cache recovery succeeded for this run. These new documentation entries remain local until a later Git handoff.

### 2026-11-01 — GPU visible; first evaluation blocked by cache locking

The project owner confirmed CUDA availability and NVIDIA A100-SXM4-80GB visibility in an allocated compute shell. The shared interactive queue required 32 CPU cores for one GPU; the initial four-core request was rejected. The evaluation command then failed while constructing DINOv2, at HF model-download cache `fcntl.flock`, with `OSError: [Errno 524]`. No feature extraction, classifiers, accuracy results or benchmark measurements completed. An unauthenticated HF warning preceded the exception but does not identify its cause.

Proposed recovery: explicitly place HF and Torch weight caches in personal scratch and rerun in a fresh Python process within the existing allocation. NERSC documents home-mount flock limitations and recommends scratch HF caches; the failing resolved cache path was not printed, so the diagnosis remains provisional. Recovery success is not yet reported. Portable commands and sources are in EVALUATION.md; no code or dependency change is needed for this retry.

### 2026-11-01 — Evaluation code transferred; Perlmutter tests passed

The project owner confirmed a clean Perlmutter checkout of `static-evaluation` tracking `origin/static-evaluation`, with the evaluation script and requirements present. The prior local dataset-ignore commit was preserved on a backup branch before switching. The module environment reports Python 3.12.13; dependency discovery found torch, torchvision, scikit-learn, Pillow and NumPy, with timm initially absent. A dedicated personal scratch venv reuses system packages; its dry run proposed only timm 1.0.30. Following installation/test instructions, the project owner reported all 23 synthetic tests passed in 1.780 seconds. The full installed-version inventory and exact remote commit were not supplied.

This verifies the synthetic suite on the static-evaluation environment on Perlmutter, not real-data classifier execution or pretrained GPU inference. Next: confirm the job account, allocate one GPU, verify CUDA in that allocation, and run the documented evaluation with artifacts retained on Perlmutter. This entry is a local documentation update; it has not automatically propagated to the remote clone.

### 2026-11-01 — Before-GRU evaluation implementation

Pulled upstream main to `023d125` (including frozen ResNet-18). Added a local development evaluation runner with train-only feature scaling, class-balanced logistic regression, validation C selection, fixed threshold, per-class/confusion/probability metrics, episode-bootstrap intervals, and private aligned predictions for error review. Goal-only/latest-only/goal-plus-latest/metadata/mean-history comparisons distinguish potential context, temporal and representation limitations. Optional synchronized static encoder component timing is explicitly separated from full-policy latency. Existing loader, preprocessing and encoder code is unchanged. Dependencies are documented separately; no real features or datasets were downloaded.

**Verification:** all 23 synthetic tests passed locally (six loader, five preprocessing, five encoder and seven evaluation tests), using Python 3.9.13, torch 2.8.0, NumPy 2.0.2, scikit-learn 1.6.1 and Pillow 11.3.0. Evaluation checks cover majority-class metric behavior, row/episode leakage rejection, train-only scaling and regularization tie handling, goal/latest/history feature roles, episode resampling, full invented-array comparisons, collection bookkeeping and static benchmark image selection. CLI help and `git diff --check` passed. Dependencies were installed in an isolated project venv reusing existing system packages; the first test launch occurred before installation completed and failed dependency imports, then all checks passed after installation. No real weights or data were used. Real Perlmutter classification, GPU timing and improvements from fine-tuning remain unmeasured. Changes are local and uncommitted; no GitHub push or remote code execution has occurred.

### 2026-10-31 — Pretrained feature extraction verified on CPU

Implemented and shared the inference-only DINOv2 encoder (`2834bb8`), followed by concise explanatory comments (`2b82710`). The comment pass preserved executable Python syntax trees; the shared notebook had no remaining changes. Both commits were pushed with Tran’s authorization. Laptop main and recorded origin/main match at `2b82710`. Tran confirmed remote notebook save/backup and a fast-forward-only pull; exact remote HEAD was not supplied.

**Verification:** five invented-backbone encoder tests passed locally and in Tran-run Perlmutter execution. They cover goal/history separation, ordering, grouping/chunking, inference mode, invalid inputs/non-finite outputs, and a mocked checkpoint/configuration guard. Actual pretrained CPU extraction then passed for one real training case: goal `[384]`, history `[6,384]`, all finite. A two-case run returned `[2,384]` and `[2,6,384]`; grouped outputs matched separately encoded cases at image chunk sizes 1 and 2 (`rtol=1e-4`, `atol=1e-5`). These real-data runs were Tran-run with two CPU threads and a three-minute process limit. No runtime or memory measurement was made.

**Environment:** existing `pytorch/2.13.0` supplies torch 2.13.0+cu130 and torchvision 0.28.0+cu130. A separate scratch venv reuses module packages and adds timm 1.0.30; its installation dry run proposed only timm. Public weights downloaded successfully despite a nonfatal unauthenticated HF warning; no token was needed. Notebook selection of the scratch venv remains unconfirmed; successful terminal execution does not verify that kernel.

**Meaning and limits:** pretrained loading, preparation compatibility, finite features, and case/frame bookkeeping work for the inspected cases. GPU execution, prediction accuracy, spatial usefulness, dataset-wide correctness, and inference performance remain unverified. No training, feature saving, or dataset-wide sweep occurred; real inputs, features, and notebook outputs stayed on Perlmutter. The immediate next step is the design discussion below; completed checks need not be repeated.

### 2026-10-31 — Full-view DINOv2 preparation verified

Tran approved a bounded DINOv2 Small/14 trial, after discussing DINO, DINOv2, DINOv3, and DeiT III. Implemented `preprocessing.py` and shared it in `a4d47a9`: RGB, proportional bicubic resize to fit 518×518, centered mean-color padding, checkpoint normalization, no cropping or random augmentation. Goal remains separate from six ordered history frames; metadata and originals are preserved. This is a trial configuration, not a final training decision. Usage and geometry are in [VIT_TRIAL.md](VIT_TRIAL.md).

**Verification:** all 11 local invented-data checks passed (six loader and five preparation tests), using an isolated temporary environment with Python 3.14.6, torch 2.15.0, and Pillow 13.0.0. An optional NumPy warning did not affect checks. Tran-run Perlmutter preparation tests passed all five checks in 1.398 seconds after selecting the module environment; an earlier system-Python/PIL discovery failure was resolved. Tests cover geometry/edge retention, RGB/scaling, unchanged originals, separation/order, aligned metadata, repeatability, and invalid sizes.

**Visual observation:** Tran reported prepared previews looked suitable, similar to originals but smaller. Original-image observations included usually visible wire within image edges, a red-circle goal target, and sensible history order. Coverage was unspecified; these observations do not prove dataset-wide detail retention or semantic metadata alignment.

### 2026-10-30 — Raw loading and notebook display established

Integrated the original-resolution reader with teammate outlines after incorporating upstream `77947cd`, preserving local work. Reader implementation is `922f200`; Tran-run Perlmutter verification at `9221135` passed six invented-data tests and decoded one training case from split position 0: seven RGB images at `(1065,1050)`, six depths, six time offsets, and 267 cases in the selected partition. Goal/history ordering, original pixels/modes, metadata alignment, and path/decode failures were checked synthetically. Full-dataset decoding and real semantic alignment remain unverified.

Goal and six chronological history images displayed successfully in the remote notebook after selecting the module interpreter. Shared reusable, output-free display cells were later pushed in `5a22202`; that packaged notebook was not separately reported rerun. Remaining notebook analysis sections are placeholders. Sensitive notebook edits and outputs must be preserved on Perlmutter before future handoffs. Dataset-wide structural checks and limits are recorded once in [DATASET.md](../DATASET.md).

## Remaining coordination and later experiments

The tiny setup now uses split position 0, seed 0, four cases per label, uncentered training-only channel scales, late goal concatenation and the documented bounded budget. These are implemented settings for the debugging check, not final model-selection choices. Coordinate any unpublished teammate work before parallel extensions; the published static evaluation is already reused.

After the real check, agree on the full-training schedule, validation/checkpoint metric, repeat seeds, fair baseline/width budgets, end-to-end latency boundary and final five-split reporting. Broader architecture ownership, caching and fine-tuning remain open. See the [handoff](GRU_DEBUG_HANDOFF.md#next-steps-after-the-check).

## Session handoff — 2026-11-03

**Completed:** the real eight-case check passed at `b9d1bdf`; 11 synthetic GRU tests also passed on Perlmutter. The November 3 milestone above is the canonical aggregate result. GPU allocation was explicitly released; no repeat tiny run or manual masking is needed. Full training, validation and test evaluation have not run.

**Plain-language next task:** keep the current frozen ViT + 64-unit GRU + classifier architecture initially. The term “head” in these notes means the trainable GRU/classifier downstream of the frozen image encoder; the classifier alone maps the combined history/goal representation to two action scores. There is no new architectural defect established by tiny fitting. Extend the surrounding training/evaluation program, rather than redesigning the head based on memorization.

**Current implementation:** full-training setup is published at `69fd063` on `tran/vit-gru-debug` in `vit_gru_train.py` and `tests/test_vit_gru_train.py`. Read the [agreed settings](GRU_TRIAL.md#agreed-first-full-training-setup--2026-11-03) and [new run guide](GRU_DEBUG_HANDOFF.md#full-training-setup--2026-11-03). Real full training/validation remains unrun. Fresh head, training-only scales, reserved test, bounded extraction and checkpoint reload are required.

**Next action:** preserve remote notebook work and update Perlmutter through an authorized update to the published `69fd063`, then verify remote imports/code and synthetic checks before allocation, then separately initiate the bounded real experiment. Do not rerun tiny fitting or masking. Later comparisons and final reporting remain separate decisions.

**Compute constraint:** finish code/environment setup and synthetic checks before requesting GPU time. Use an explicit short allocation and process budget, release immediately after execution and do not automatically rerun or increase limits. The agreed first-run envelope is a 15-minute allocation with a 600-second process cap; setup did not allocate or execute it.

**Git/worktrees:** shared notes remain in the original checkout on main (local HEAD and cached origin/main are `c234ea4` at handoff inspection). Code worktree is sibling `waldo-guidewire-policy-vit-gru-debug`, branch `tran/vit-gru-debug`, local and cached remote tip `69fd063` (push succeeded). Main's progress/plan/handoff updates from this side conversation are uncommitted. Main also has two excluded `.DS_Store` files. The debug worktree is clean after the authorized full-training code commit/push. Shared documentation has not been synchronized into that branch; Perlmutter receipt of the new code is unverified. Recheck status/upstream before changes. No project instruction/configuration files were created; ask immediately before any further commit.

**Data and verification limits:** real data, features, selected rows/scales, checkpoints and complete reports stay on Perlmutter. Remote report path is `$PSCRATCH/waldo-gru-debug/tiny-01/debug_summary.json`; only its supplied aggregates were recorded locally. Reported remote dirty state was untracked bytecode/terminal log only, with no tracked edits reported. No remote notebook was modified; preserve it before future updates. Tiny fitting establishes real-data learning plumbing, not generalization, goal use, temporal usefulness or online latency.

**New session:** read this file's current status, November 3 milestone and handoff, then GRU_TRIAL and GRU_DEBUG_HANDOFF from main, plus README/APPROACHES/DATASET. Preserve the uncommitted documentation update. Review the agreed full-training setup and verification; confirm an authorized Perlmutter update to `69fd063` and remote readiness checks before initiating real execution.
