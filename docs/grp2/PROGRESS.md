# Group 2 — Tran and Sabrina

Read the [shared approach notes](../APPROACHES.md), [dataset notes](../DATASET.md), and [project README](../../README.md) before continuing. This file holds Group 2 plans, progress, and permitted aggregate results. Shared environment, preprocessing, interfaces, and evaluation decisions belong in the approach notes.

## Proposed scope — supplied by project owner

- ViT–GRU: replace CNN with a pretrained vision transformer, retaining GRU and consistent inputs.
- CNN–temporal attention: replace GRU with temporal self-attention, retaining CNN and consistent inputs.

These are team proposals, not confirmed course requirements. Friday task ownership is recorded below; exact architecture ownership remains open. The shared optional JEPA extension remains in the approach notes.

## Current status and next steps

**The bounded ViT–GRU debugging code is implemented and locally tested.** Tran requested setup of the frozen ViT + standard one-layer, 64-unit GRU + goal + aligned depth/time model. `vit_gru.py`, `vit_gru_debug.py` and `tests/test_vit_gru.py` are in the separate `tran/vit-gru-debug` worktree. All 44 synthetic tests passed, including 11 new GRU checks. The real eight-training-case check has not run; GPU backward, task learning and runtime remain unverified for Tran's setup.

**Next:** review and share the local code through an explicitly authorized handoff, then run the bounded check on an allocated Perlmutter node. The [debugging and Sabrina handoff](GRU_DEBUG_HANDOFF.md) provides commands, expected evidence, failure handling and subsequent steps. [GRU_TRIAL.md](GRU_TRIAL.md) retains the design rationale and later comparisons. No full training, validation tuning, caching expansion or final evaluation is included in this milestone.

**Existing evidence:** raw loading, full-view preparation and pretrained DINOv2 CPU extraction/grouping passed prior checks. Tran's notebook imports timm 1.0.30 and reports CUDA available. The one-frame masking result is recorded and inconclusive for temporal usefulness; do not repeat masking. Sabrina separately completed static frozen-feature probes and a reported GPU run; her unchanged evaluation module/tests are imported for later reuse. Her account's GPU result does not verify Tran's GPU path.

**Working layout:** documentation stays in the original checkout on `main`; code stays in sibling `waldo-guidewire-policy-vit-gru-debug` on `tran/vit-gru-debug`. Both are based on `0f77490`. Main's current notes do not automatically appear in the debug checkout; read them here until branch synchronization is explicitly authorized. Remote publication remains pending.

**Experimental limits:** tiny fitting checks learning plumbing only. Later full training must start with fresh head weights, compare 64 versus 128 using validation and prioritize Sabrina's latest-image + goal + matching historical-metadata baseline before the broader four-model comparison. A functioning GRU or a perfect eight-case fit does not establish generalization, goal use or useful temporal processing.

## Progress log

### 2026-11-02 — Bounded GRU setup implemented; real run pending

Following Tran's explicit setup request, implemented the 115,650-parameter head, split-0 four-per-label selection, training-only uncentered depth/time scales, frozen feature extraction and a fresh full-batch Adam loop in the separate debug worktree. Runtime/update caps match the documented plan; a parent process bounds the overall worker lifetime. Private reports retain configuration, selected rows/scales, source hashes and aggregate learning/runtime evidence on Perlmutter. No features or model checkpoints are persisted.

**Verification:** all 44 synthetic tests passed locally in 0.952 seconds, including 11 new GRU checks. Tests verify sequence/goal/metadata wiring, reset hidden state, deterministic balanced selection, training-only scaling and constant-channel handling, inference-tensor conversion, actual backward/optimizer updates, unchanged encoder parameters/buffers, non-finite rejection and both update/time limits. The invented-feature learning check reached perfect training accuracy within 40 updates and lower loss; this is not real task evidence. An additional synthetic worker smoke exercised the actual raw loader and preprocessor with invented PNGs/records, an injected encoder, 40 head updates and successful report serialization/source hashes; it saved only the summary and used no real data or pretrained weights. CLI help, syntax, local documentation links and diff formatting passed. Environment: Python 3.14.6, torch 2.15.0, NumPy 2.5.4, scikit-learn 1.9.1 and Pillow 13.0.0. Existing CNN diagnostic tests emitted Pillow deprecation warnings without failures.

**Handoff and limits:** added [GRU_DEBUG_HANDOFF.md](GRU_DEBUG_HANDOFF.md) with Sabrina's reuse/coordination steps, local and Perlmutter commands, notebook preservation, private reporting and next experiment boundaries. No real data, GPU execution, pretrained extraction rerun, remote notebook changes or new baseline run occurred. Loader/preprocessor/encoder and Sabrina's imported code are unchanged. At completion of implementation, main held the shared notes and debug held the code, all unstaged/uncommitted/unpushed. Tran subsequently requested separate commits by scope; confirmation immediately before committing is still required. Next: authorized code sharing, then one bounded Perlmutter run. Full-training and final-evaluation protocols remain separate work.

### 2026-11-02 — Separate debug worktree and Sabrina evaluation reuse

At Tran's request, created `tran/vit-gru-debug` and then moved it to a separate sibling worktree, keeping shared documentation on `main`. Rechecked live remote tips: `main` at `0f77490`, `static-evaluation` at `070328f`. Fast-forwarded local main from `0c4cbee` to `0f77490`, preserving every local documentation edit through a backed-up three-way reconciliation. The untracked GRU plan is preserved; `.DS_Store` files remain excluded. No staging, new commit or push occurred.

Inspected Sabrina's evaluation implementation (`5ddb566`) and handoff (`070328f`). Imported `evaluation.py` and `tests/test_evaluation.py` unchanged into the debug worktree; brought the evaluation reference into main's documentation. Existing latest-image/goal/metadata probes, metrics and partition checks can support later comparisons. The scikit-learn optimizer, centered feature scaling, validation-selected regularization and full-partition runner do not implement the proposed tiny GRU check. No baseline rerun, GRU implementation, full training, data extraction or remote notebook update occurred. Current verification is recorded in the session handoff below.

### 2026-11-02 — Feature check completed; temporal-model design next

Tran ran the diagnostics shared in `0c4cbee` on Perlmutter. Goal-image comparison worked; numerical marker sensitivity was not reported. In the one-frame wire check, covering the tip changed the summary less than all three comparison edits. [Diagnostic notes](VIT_DIAGNOSTICS.md) retain the measurements and limitations. This static check cannot establish whether frame-to-frame information supports action prediction. Tran chose to stop additional masking and move toward a direct temporal-model trial; the proposed design remains unimplemented.

**Environment:** Tran confirmed torch 2.13.0+cu130, timm 1.0.30 and CUDA availability in the module-based notebook. The scratch venv also works but was not the final notebook interpreter. Exact diagnostic device and remote code version were not reported; no runtime/memory measurements were made. Real images, features and notebook outputs remain on Perlmutter.

**Verification:** local notebook syntax, cell IDs, empty outputs and diff formatting passed before sharing. Real-data execution is reported by Tran, not independently run by assistant tools. No model/training code was added.

**Design clarification:** at Tran's request, expanded [the GRU plan](GRU_TRIAL.md) with expected evidence from tiny fitting, a fresh start for later full training, separate validation/test roles and alternative split positions. Architecture ownership and training scope remain unconfirmed. Refreshed origin and confirmed `0f77490` remains one commit ahead of local `0c4cbee`; inspected the upstream CNN diagnostic changes and overlapping shared-document edits without reconciliation. Existing local work is preserved; no remote notebook or data was accessed, and no training, staging, commit or push occurred.

**Design review recorded:** inspected the proposed head and existing inference-only encoder. The frozen GRU trial is reasonable, with unresolved risks around whole-image summaries, late goal fusion, metadata-dominated prediction, limited independent data and unmeasured end-to-end latency. Recorded alternative width parameter counts and recommended earlier controlled comparisons in the plan rather than duplicating technical guidance here. Split 0 means the first supplied assignment object, not a verified standard cross-validation fold. Recommendations remain unapproved; no architecture or training settings were changed. Verification is limited to document/code inspection and local arithmetic/link/format checks; no new real-data evidence or model execution. Next: agree on the bounded trial and comparison protocol before implementation.

### 2026-11-02 — Documentation reconciled with completed CPU milestone

Reviewed overall and Group 2 notes against the recorded October 31 results. Updated README, shared approach status, and ViT trial instructions to remove superseded claims that preparation, timm compatibility, or pretrained CPU extraction were still pending. No new model execution or data verification occurred. The immediate next step remains the design discussion below; GPU verification/performance measurement and complete-policy implementation remain outstanding. Group 1's separate CNN GPU diagnostics do not establish Group 2 GPU readiness. Code, notebook edits, and dataset restrictions are unchanged.

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

The tiny setup now uses split position 0, seed 0, four cases per label, uncentered training-only channel scales, late goal concatenation and the documented bounded budget. These are implemented settings for the debugging check, not final model-selection choices. Coordinate any unpublished teammate work before parallel extensions; Sabrina's published static evaluation is already reused.

After the real check, agree on the full-training schedule, validation/checkpoint metric, repeat seeds, fair baseline/width budgets, end-to-end latency boundary and final five-split reporting. Broader architecture ownership, caching and fine-tuning remain open. See the [handoff](GRU_DEBUG_HANDOFF.md#next-steps-after-the-check).

## Session handoff — 2026-11-02

**State:** frozen ViT–GRU debugging implementation is ready for review and a first Perlmutter run. All 44 local synthetic tests passed. Actual eight-case fitting, GPU extraction/backward and memory/runtime on Tran's account remain unverified. Prior CPU extraction/grouping and one-frame masking are complete; do not repeat them without a relevant change. Sabrina's static evaluation and its reported results remain separate evidence.

**Accepted scope:** implement the eight-training-case, combined-input GRU check with hidden size 64, split position 0, seed 0, four cases per label, training-only uncentered depth/time scales, late goal fusion and Adam at 0.001. Maximum 300 updates or 120 seconds of head work, with a 600-second overall worker deadline. The head limit is checked between updates; termination can take up to five additional seconds. No validation/test tuning, persistent features or model checkpoint is part of this runner. The setup request does not authorize full training or broader comparisons.

**Git/worktrees and commit scopes:** both checkouts started from `0f77490` (last checked origin/main). Main owns the eight documentation files: README, APPROACHES, EVALUATION, Group 2 PROGRESS, VIT_DIAGNOSTICS, VIT_TRIAL, GRU_TRIAL and GRU_DEBUG_HANDOFF. The sibling `waldo-guidewire-policy-vit-gru-debug` checkout is on `tran/vit-gru-debug`: one scope contains unchanged Sabrina imports `evaluation.py` and `tests/test_evaluation.py` from `070328f`; another contains `vit_gru.py`, `vit_gru_debug.py` and `tests/test_vit_gru.py`. Tran requested commits by scope; use `git log -3 --oneline` and `git status --short` in each checkout for exact resulting revisions and pending changes. Main's two `.DS_Store` files remain excluded. Remote publication and synchronization of main's documentation into the debug branch remain pending; no push or merge is authorized by the commit request. Ask immediately before each commit.

**Verification:** 44 synthetic tests plus the synthetic worker/report smoke passed; CLI help works, imported evaluation files match Sabrina's branch byte-for-byte, existing loading/preparation/encoder/notebook files are unchanged and syntax/link/diff-format checks passed. No real dataset access or remote execution occurred. The synthetic checks do not establish real-data fit, generalization, temporal benefit, GPU compatibility or full-policy latency.

**Next action:** use [GRU_DEBUG_HANDOFF.md](GRU_DEBUG_HANDOFF.md) to review/share code through explicit authorization and preserve the remote notebook before updating Perlmutter. Run the synthetic checks in the intended environment, then the single bounded eight-case check; retain the private report there and record permitted aggregates here. A pull cannot retrieve local work that has not been pushed. Do not silently widen budgets after failure. Start fresh for later full training and coordinate a matched baseline comparison.

**Restrictions:** real data, features, selected rows/scales, caches, checkpoints and notebook outputs stay on Perlmutter. No project instruction/configuration files. Documentation remains on main; code stays in the separate debug worktree.
