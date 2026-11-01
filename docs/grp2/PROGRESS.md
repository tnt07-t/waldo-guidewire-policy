# Group 2 — Tran and Sabrina

Read the [shared approach notes](../APPROACHES.md), [dataset notes](../DATASET.md), and [project README](../../README.md) before continuing. This file holds Group 2 plans, progress, and permitted aggregate results. Shared environment, preprocessing, interfaces, and evaluation decisions belong in the approach notes.

## Proposed scope — supplied by project owner

- ViT–GRU: replace CNN with a pretrained vision transformer, retaining GRU and consistent inputs.
- CNN–temporal attention: replace GRU with temporal self-attention, retaining CNN and consistent inputs.

These are team proposals, not confirmed course requirements. Friday task ownership is recorded below; exact architecture ownership remains open. The shared optional JEPA extension remains in the approach notes.

## Current status and next steps

**Latest aggregate findings:** the first split-0 results have now been supplied. DINOv2 goal-only has the highest observed validation BA (.609); DINOv2 latest+goal is .570 and ResNet latest+goal .547, with large train/validation gaps. Static encoder component medians are 46.83 ms total for DINOv2 and 33.87 ms for ResNet; classifier/disk IO are excluded. See EVALUATION.md for the canonical results, caveats and prioritized follow-up hypotheses. No statistically supported encoder ranking or final-test result is established; existing episode intervals and matched errors are next to inspect.

**Handoff:** implementation is on `static-evaluation`; the aggregate review and [teammate resume checklist](../EVALUATION.md#teammate-handoff--resume-here) document what is complete and what to inspect next. Keep CNN/ViT frozen. Private uncertainty/error evidence is next; no baseline rerun is needed just to resume.

**Update — 2026-11-01:** Sabrina authorized the before-GRU evaluation milestone and ongoing documentation. `evaluation.py` compares fixed ResNet/DINOv2 features using separate regularized linear probes. Implementation, synthetic verification and the first real development run are complete. Earlier pending-execution or limited-authorization statements in dated entries describe historical stages. GRU and encoder fine-tuning remain outside this implementation.

Owner-reported schedule (2026-10-30): Friday — Tran sets up data loading; Sabrina works on models (specific model and deliverable unconfirmed). Tran is now starting ViT preparation/feature extraction for ViT–GRU. Remaining ownership and the Sunday deliverable remain open. The raw-loading milestone is complete for the checks below; complete policy implementation and training have not begun.

The first raw-case reader passed six synthetic tests and decoded one real training case on Perlmutter at `9221135` (owner-reported). The owner confirmed that both the goal and six history frames display in the Perlmutter notebook, with history depth/time annotations. This confirms display execution, not visual/semantic correctness. Preparation and inference-only encoder code are implemented locally. Synthetic preparation tests passed on Perlmutter and initial previews were reported suitable; actual pretrained CPU extraction and two-case grouping checks also passed. Basic dataset validation has already passed as recorded in the dataset notes; do not repeat it without a concrete reason.

The initial preparation trial is in [APPROACHES.md](../APPROACHES.md#initial-vit-preparation-trial), with preparation usage in [VIT_TRIAL.md](VIT_TRIAL.md). One/two-case CPU extraction and the later GPU development evaluation completed. Only the separate evaluation classifiers were trained; no GRU or encoder training is implemented. Remaining model interfaces and final evaluation policy need agreement.

Tran's initial visual observations: wire usually visible within image edges, red-circle goal target, and sensible history order. Inspection coverage is unspecified; these observations do not establish full-dataset correctness or metadata alignment. DINOv2 Small/14 remains the trial candidate; final model selection remains open despite the completed first GPU evaluation.

## Progress log

### 2026-11-01 — Frozen backbone versus evaluation classifier clarified

Sabrina confirmed the CNN should remain frozen for the coming experiments. Explicitly documented that evaluation fits a separate logistic-regression probe and train-only scaler on detached features; it does not modify `cnn.py`, update ResNet weights or BatchNorm statistics, or backpropagate into either image encoder. Fine-tuning remains a possible future research direction, not the next planned implementation. The current accuracy gaps describe probe generalization; no encoder was trained in this milestone.

### 2026-11-01 — First aggregate accuracy and component timing review

Sabrina supplied the generated summary and encoder benchmark. Recorded the full 12-experiment BA/recall comparison and component timings once in EVALUATION.md. Main evidence: large train/validation gaps; goal-only DINOv2 above its latest+goal baseline; metadata and mean pooling did not improve rounded DINOv2 validation BA. DINOv2 encoding is approximately 3.8 times slower than ResNet at the shared 518-square preparation, but preparation reduces the measured total-time ratio to approximately 1.38. Single-case repeated timings exclude classifier/disk IO and shared-residency memory cannot rank isolated model sizes.

Proposed next order: inspect existing intervals/confusions/per-C scores and private matched errors; then a bounded stronger-regularization experiment with frozen encoders; investigate reference dependence, fusion and small-detail representation before limited fine-tuning. No new experiment was launched or model changed in this review. All real case-level artifacts remain on Perlmutter.

### 2026-11-01 — First before-GRU evaluation completed on GPU

After setting scratch HF/Torch cache locations, Sabrina reported successful DINOv2 and ResNet weight downloads and the runner's completion message for split position 0, both encoders, CUDA and `--benchmark`. The preceding allocation check identified an A100-SXM4-80GB. This establishes owner-reported completion of the development run; scores, confusion matrices, uncertainty intervals, code/version provenance and component timings have not yet been reviewed. No superiority, fine-tuning gain, full-policy latency, or final-test result is claimed.

Next: inspect the aggregate summary and encoder component benchmark on Perlmutter, retain private case-level predictions there, and record aggregate findings and a bounded follow-up hypothesis. Scratch-cache recovery succeeded for this run. These new documentation entries remain local until a later Git handoff.

### 2026-11-01 — GPU visible; first evaluation blocked by cache locking

Sabrina confirmed CUDA availability and NVIDIA A100-SXM4-80GB visibility in an allocated compute shell. The shared interactive queue required 32 CPU cores for one GPU; the initial four-core request was rejected. The evaluation command then failed while constructing DINOv2, at HF model-download cache `fcntl.flock`, with `OSError: [Errno 524]`. No feature extraction, classifiers, accuracy results or benchmark measurements completed. An unauthenticated HF warning preceded the exception but does not identify its cause.

Proposed recovery: explicitly place HF and Torch weight caches in personal scratch and rerun in a fresh Python process within the existing allocation. NERSC documents home-mount flock limitations and recommends scratch HF caches; the failing resolved cache path was not printed, so the diagnosis remains provisional. Recovery success is not yet reported. Portable commands and sources are in EVALUATION.md; no code or dependency change is needed for this retry.

### 2026-11-01 — Evaluation code transferred; Perlmutter tests passed

Sabrina confirmed a clean Perlmutter checkout of `static-evaluation` tracking `origin/static-evaluation`, with the evaluation script and requirements present. The prior local dataset-ignore commit was preserved on a backup branch before switching. The module environment reports Python 3.12.13; dependency discovery found torch, torchvision, scikit-learn, Pillow and NumPy, with timm initially absent. A dedicated personal scratch venv reuses system packages; its dry run proposed only timm 1.0.30. Following installation/test instructions, Sabrina reported all 23 synthetic tests passed in 1.780 seconds. The full installed-version inventory and exact remote commit were not supplied.

This verifies the synthetic suite on Sabrina's Perlmutter setup, not real-data classifier execution or pretrained GPU inference. Next: confirm the job account, allocate one GPU, verify CUDA in that allocation, and run the documented evaluation with artifacts retained on Perlmutter. This entry is a local documentation update; it has not automatically propagated to the remote clone.

### 2026-11-01 — Before-GRU evaluation implementation

Pulled upstream main to `023d125` (including frozen ResNet-18). Added a local development evaluation runner with train-only feature scaling, class-balanced logistic regression, validation C selection, fixed threshold, per-class/confusion/probability metrics, episode-bootstrap intervals, and private aligned predictions for error review. Goal-only/latest-only/goal-plus-latest/metadata/mean-history comparisons distinguish potential context, temporal and representation limitations. Optional synchronized static encoder component timing is explicitly separated from full-policy latency. Existing loader, preprocessing and encoder code is unchanged. Dependencies are documented separately; no real features or datasets were downloaded.

**Verification:** all 23 synthetic tests passed locally (six loader, five preprocessing, five encoder and seven evaluation tests), using Python 3.9.13, torch 2.8.0, NumPy 2.0.2, scikit-learn 1.6.1 and Pillow 11.3.0. Evaluation checks cover majority-class metric behavior, row/episode leakage rejection, train-only scaling and regularization tie handling, goal/latest/history feature roles, episode resampling, full invented-array comparisons, collection bookkeeping and static benchmark image selection. CLI help and `git diff --check` passed. Dependencies were installed in an isolated project venv reusing existing system packages; the first test launch occurred before installation completed and failed dependency imports, then all checks passed after installation. No real weights or data were used. Real Perlmutter classification, GPU timing and improvements from fine-tuning remain unmeasured. Changes are local and uncommitted; no GitHub push or remote code execution has occurred.

### 2026-10-31 — Pretrained feature extraction verified on CPU

Implemented and shared the inference-only DINOv2 encoder (`2834bb8`), followed by concise explanatory comments (`2b82710`). The comment pass preserved executable Python syntax trees; the shared notebook had no remaining changes. Both commits were pushed with owner authorization. Laptop main and recorded origin/main match at `2b82710`. Owner confirmed remote notebook save/backup and a fast-forward-only pull; exact remote HEAD was not supplied.

**Verification:** five invented-backbone encoder tests passed locally and in owner-run Perlmutter execution. They cover goal/history separation, ordering, grouping/chunking, inference mode, invalid inputs/non-finite outputs, and a mocked checkpoint/configuration guard. Actual pretrained CPU extraction then passed for one real training case: goal `[384]`, history `[6,384]`, all finite. A two-case run returned `[2,384]` and `[2,6,384]`; grouped outputs matched separately encoded cases at image chunk sizes 1 and 2 (`rtol=1e-4`, `atol=1e-5`). These real-data runs were owner-run with two CPU threads and a three-minute process limit. No runtime or memory measurement was made.

**Environment:** existing `pytorch/2.13.0` supplies torch 2.13.0+cu130 and torchvision 0.28.0+cu130. A separate scratch venv reuses module packages and adds timm 1.0.30; its installation dry run proposed only timm. Public weights downloaded successfully despite a nonfatal unauthenticated HF warning; no token was needed. Notebook selection of the scratch venv remains unconfirmed; successful terminal execution does not verify that kernel.

**Meaning and limits:** pretrained loading, preparation compatibility, finite features, and case/frame bookkeeping work for the inspected cases. GPU execution, prediction accuracy, spatial usefulness, dataset-wide correctness, and inference performance remain unverified. No training, feature saving, or dataset-wide sweep occurred; real inputs, features, and notebook outputs stayed on Perlmutter. The immediate next step is the design discussion below; completed checks need not be repeated.

### 2026-10-31 — Full-view DINOv2 preparation verified

Tran approved a bounded DINOv2 Small/14 trial, after discussing DINO, DINOv2, DINOv3, and DeiT III. Implemented `preprocessing.py` and shared it in `a4d47a9`: RGB, proportional bicubic resize to fit 518×518, centered mean-color padding, checkpoint normalization, no cropping or random augmentation. Goal remains separate from six ordered history frames; metadata and originals are preserved. This is a trial configuration, not a final training decision. Usage and geometry are in [VIT_TRIAL.md](VIT_TRIAL.md).

**Verification:** all 11 local invented-data checks passed (six loader and five preparation tests), using an isolated temporary environment with Python 3.14.6, torch 2.15.0, and Pillow 13.0.0. An optional NumPy warning did not affect checks. Owner-run Perlmutter preparation tests passed all five checks in 1.398 seconds after selecting the module environment; an earlier system-Python/PIL discovery failure was resolved. Tests cover geometry/edge retention, RGB/scaling, unchanged originals, separation/order, aligned metadata, repeatability, and invalid sizes.

**Visual observation:** Tran reported prepared previews looked suitable, similar to originals but smaller. Original-image observations included usually visible wire within image edges, a red-circle goal target, and sensible history order. Coverage was unspecified; these observations do not prove dataset-wide detail retention or semantic metadata alignment.

### 2026-10-30 — Raw loading and notebook display established

Integrated the original-resolution reader with teammate outlines after incorporating upstream `77947cd`, preserving local work. Reader implementation is `922f200`; owner-run Perlmutter verification at `9221135` passed six invented-data tests and decoded one training case from split position 0: seven RGB images at `(1065,1050)`, six depths, six time offsets, and 267 cases in the selected partition. Goal/history ordering, original pixels/modes, metadata alignment, and path/decode failures were checked synthetically. Full-dataset decoding and real semantic alignment remain unverified.

Goal and six chronological history images displayed successfully in the remote notebook after selecting the module interpreter. Shared reusable, output-free display cells were later pushed in `5a22202`; that packaged notebook was not separately reported rerun. Remaining notebook analysis sections are placeholders. Sensitive notebook edits and outputs must be preserved on Perlmutter before future handoffs. Dataset-wide structural checks and limits are recorded once in [DATASET.md](../DATASET.md).

## Next decisions — discussion before implementation

The CPU extraction milestone is complete. The new frozen-probe development defaults and rationale are in EVALUATION.md. The following remain integration questions for a later temporal/fine-tuning model, rather than blockers to the newly authorized probes:

1. Keep pretrained ViT weights fixed initially, or fine-tune them?
2. How should the separate goal features combine with the six history features?
3. How should aligned depth/time values enter the model and be scaled?
4. Which development split, validation metrics, and model-selection procedure should the team use?

The new linear action heads are authorized for this evaluation milestone; GRU, full encoder training and persistent feature caching are not implemented. Fine-tuning and architecture changes remain proposed follow-ups requiring a concrete design based on evaluation evidence. GPU verification can follow when compute access is available. Remaining architecture ownership is still open.
