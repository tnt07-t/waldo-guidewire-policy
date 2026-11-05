# Group 2 — Transformer models

Read the [shared approach notes](../APPROACHES.md), [dataset notes](../DATASET.md), and [project README](../../README.md) before continuing. This file holds Group 2 plans, progress, and permitted aggregate results. Shared environment, preprocessing, interfaces, and evaluation decisions belong in the approach notes.

## Proposed scope — supplied by project owner

- ViT–GRU: replace CNN with a pretrained vision transformer, retaining GRU and consistent inputs.
- CNN–temporal attention: replace GRU with temporal self-attention, retaining CNN and consistent inputs.

These are team proposals, not confirmed course requirements. Friday task ownership is recorded below; exact architecture ownership remains open. The shared optional JEPA extension remains in the approach notes.

## CNN–attention status — 2026-11-05

**DINOv2 comparison completed:** mean-history MLP leads latest across seeds
0/1/2, average validation BA .612 versus .580, at approximately 163 versus 45 ms
same-case latency. DINOv2 attention reaches .565 in seed 0, with substantial
later overfitting. Mean pooling is a baseline within the attention investigation,
not a GRU result. Full scores, reasoning and timing limits are in the
[experiment log](ATTENTION_EXPERIMENTS.md). No further training is queued.

The latest+goal MLP can now fit training data after embedding standardization,
but generalization remains weak. No final model is selected; reserved test data
remains untouched. [Experiment results](ATTENTION_EXPERIMENTS.md) hold the evidence;
[setup](CNN_ATTENTION.md) holds architecture and commands.

| Milestone | Finding / decision |
|---|---|
| Nov 3: integration and comparison | Eight-case attention fit passed. Selected validation BA: latest .597, mean .601, attention .500. Mean/attention cost about four times the same-case latency of latest. |
| Nov 4: learning and input diagnostics | Training improves while validation deteriorates. Final attention is more sensitive to the goal than historical imagery; this does not identify marker understanding or prove history is noise. |
| Nov 5: classifier failure isolated | Raw latest+goal ReLU had all 64 hidden units inactive. LeakyReLU restores gradients; training-only feature standardization restores fitting. |
| Nov 5: regularization comparison | Standardized reference: train/validation BA .964/.588 at epoch 8, 1.000/.510 finally. Dropout .3 gives .584 selected validation BA; decay .01 gives .588. Neither clearly improves the primary metric. |

**Next:** retain the standardized LeakyReLU reference (dropout .1, decay .001).
Investigate transferable episode associations and matched input controls before
capacity sweeps. Standardization currently supports latest+goal only; extending
other heads would require implementation. Historical-input benefit and metadata
necessity remain unresolved. Width 32/64/128 and position on/off are available
but untested. No additional experiment is queued.

## ViT–GRU notes — recorded 2026-11-03

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

### 2026-11-01 — Frozen-feature evaluation completed

The first real GPU development comparison used frozen DINOv2/ResNet-18 features
with logistic classifiers, not trained temporal models. DINOv2 goal-only reached
.609 validation BA; latest+goal reached .570. High training scores and weak
validation motivated input controls and later neural diagnostics. Protocol,
aggregate results and timing limits remain in [EVALUATION.md](../EVALUATION.md).

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
