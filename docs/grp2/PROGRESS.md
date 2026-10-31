# Group 2 — Tran and Sabrina

Read the [shared approach notes](../APPROACHES.md), [dataset notes](../DATASET.md), and [project README](../../README.md) before continuing. This file holds Group 2 plans, progress, and permitted aggregate results. Shared environment, preprocessing, interfaces, and evaluation decisions belong in the approach notes.

## Proposed scope — supplied by project owner

- ViT–GRU: replace CNN with a pretrained vision transformer, retaining GRU and consistent inputs.
- CNN–temporal attention: replace GRU with temporal self-attention, retaining CNN and consistent inputs.

These are team proposals, not confirmed course requirements. Friday task ownership is recorded below; exact architecture ownership remains open. The shared optional JEPA extension remains in the approach notes.

## Current status and next steps

Owner-reported schedule (2026-10-30): Friday — Tran sets up data loading; Sabrina works on models (specific model and deliverable unconfirmed). Tran is now starting ViT preparation/feature extraction for ViT–GRU. Remaining ownership and the Sunday deliverable remain open. The raw-loading milestone is complete for the checks below; complete policy implementation and training have not begun.

The first raw-case reader passed six synthetic tests and decoded one real training case on Perlmutter at `9221135` (owner-reported). The owner confirmed that both the goal and six history frames display in the Perlmutter notebook, with history depth/time annotations. This confirms display execution, not visual/semantic correctness. Preparation and inference-only encoder code are implemented locally. Synthetic preparation tests passed on Perlmutter and initial previews were reported suitable; actual pretrained CPU extraction and two-case grouping checks also passed. Basic dataset validation has already passed as recorded in the dataset notes; do not repeat it without a concrete reason.

The initial preparation trial and staged implementation plan are in [APPROACHES.md](../APPROACHES.md#initial-vit-preparation-trial). Owner-run pretrained CPU extraction now passes for one case and two-case individual/grouped comparisons, including image chunk sizes 1 and 2. Next: obtain compute-access details for GPU verification and inference memory/runtime measurement; agree on remaining interfaces before full model integration. Preparation usage is in [VIT_TRIAL.md](VIT_TRIAL.md). No GRU or training is implemented.

Tran has now reported initial visual observations: the wire is usually visible and within the image edges, the goal target is marked with a red circle, and the history order appears sensible. Inspection coverage is unspecified; these observations do not establish full-dataset correctness or metadata alignment. DINOv2 Small/14 is the first trial candidate; final model selection remains open. Initial preparation previews were reported suitable; one-case and two-case CPU encoder checks passed. GPU verification remains pending.

## Progress log

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

The current CPU extraction milestone is complete. No command or immediate GPU allocation is needed to discuss the next model stage. Explain and agree on these choices before coding:

1. Keep pretrained ViT weights fixed initially, or fine-tune them?
2. How should the separate goal features combine with the six history features?
3. How should aligned depth/time values enter the model and be scaled?
4. Which development split, validation metrics, and model-selection procedure should the team use?

Suggested starting designs remain proposals, not approvals. GPU verification and memory/runtime measurement can follow when access details are available. GRU, action head, complete training, full feature caching, and dataset-wide extraction are not implemented or newly authorized. Friday's owner-reported assignment was Tran on loading and Sabrina on models; remaining model ownership and the Sunday deliverable still need clarification.
