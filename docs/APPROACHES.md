# Approaches and team coordination

For teammates. Start here to understand the pipeline, responsibilities, and open decisions. The [documentation guide](README.md) identifies what people and AI assistants should read.

## How the current files fit together

We are building one pipeline in stages:

1. **Read original data — `dataset.py`.** Tran's reader selects cases from a supplied partition and returns the goal, six ordered history images, depths, times, label, and tracing identifiers. It preserves original images. Six synthetic tests and one real-case decode passed on Perlmutter; see the Group 2 milestone for evidence and limits.
2. **Prepare model inputs — `preprocessing.py`.** Full-view RGB resize/padding/normalization preserves separate goal/history roles. Initial preparation tests and inspected previews passed on Perlmutter; final training preparation remains open.
3. **Extract visual features — `vit.py`, `cnn.py`.** DINOv2 yields 384 values/image; pretrained one/two-case CPU checks passed. The latest upstream CNN implementation is frozen ResNet-18 with 512 values/image. Both keep goal separate from history. GPU verification remains pending.
4. **Evaluate fixed representations — `evaluation.py`.** Sabrina authorized the before-GRU milestone on 2026-11-01. Linear probes and diagnostic input ablations measure action prediction before sequence modeling. See [evaluation protocol and rationale](EVALUATION.md); real-data classifier results remain pending.

These are compatible stages, not competing loaders. Origin's `dataset.py` was TODO-only: it proposed `(frames, depth_history, label)` without implementing loading or clarifying the separate goal and time offsets. Our reader makes those roles explicit and returns a dictionary of raw inputs. Build later model preparation on that reader; agree on interface changes before implementation.

`data_inspect.ipynb` is an optional interactive notebook for understanding the data between reading and model preparation. It combines explanations, executable Python cells, and visual results. The owner successfully displayed a goal and six history frames using cells entered on Perlmutter. The shared notebook now includes reusable, output-free loading and history-display cells based on those snippets; other analysis sections remain placeholders. Sensitive notebook outputs must stay on Perlmutter.

Raw reading remains separate from model preparation. The prepared tensors feed either encoder; the evaluation runner constructs in-memory features and fits small action classifiers. It does not change loader/encoder contracts or add a GRU.

## Team proposal — supplied by project owner

- [Group 1 — Andrew and Suhani](grp1/PROGRESS.md): CNN–GRU, CNN–LSTM, and current-input baseline. Group-specific plans and progress live there.
- [Group 2 — Tran and Sabrina](grp2/PROGRESS.md): ViT–GRU and CNN–temporal attention. Group-specific plans and progress live there.

Joint optional extension: JEPA, after the primary experiments work.

Friday task assignments and the Sunday CNN target are recorded in the group logs. Tran is starting with ViT feature extraction for the ViT–GRU approach; remaining architecture ownership is not finalized. “Artery map” correspondence to the goal reference remains open. The official final-project rubric has not been supplied; none of these choices is documented as a course requirement.

## What is decided and what is still open

The first loader preserves original images and metadata. Tran approved the initial ViT preparation below; pretrained CPU extraction has since passed for inspected cases. The configuration remains a bounded trial. On 2026-11-01 Sabrina authorized local frozen-encoder evaluation and documentation: concatenate goal/current features, compare diagnostic input sets, use train-only scaling and small validation-selected L2 probes. These development defaults are documented in EVALUATION.md; team-wide final split reporting remains open.

Before model integration, agree on input preparation, goal/history fusion, depth/time use, and batching. Then agree on the supplied split reporting protocol, metrics, and model selection. We have not selected a five-fold cross-validation protocol. Reuse shared components where appropriate rather than implementing separate data readers for each architecture.

## Initial ViT preparation trial

**Scope approved by Tran (2026-10-31):** start with the pretrained ViT image encoder for ViT–GRU, preparing and testing its inputs before implementing the complete policy. The ViT extracts features from individual images; the later GRU combines the six history steps. CNN–temporal attention is a separate Group 2 approach, not an additional component of ViT–GRU.

**Starting preparation:** preserve the full field of view (no cropping), retain RGB including the red goal marker, convert images to floating-point tensors, use the selected checkpoint's documented pixel scaling/normalization, and start without random augmentation. Keep original files unchanged and preparation separate from the raw reader. Preserve the separate goal and chronological history, along with aligned depth/time metadata; labels and identifiers are not encoder inputs. This is an initial experiment, not a team-wide preprocessing contract.

**First concrete trial (2026-10-31):** following the candidate discussion, Tran requested the first implementation step for DINOv2 Small/14 (`vit_small_patch14_dinov2.lvd142m`), using its 518×518 input size, proportional bicubic resize, centered mean-color padding, and documented scaling/normalization. This preserves the full field of view. `preprocessing.py` handles preparation; `vit.py` loads the pretrained encoder through timm, with remote compatibility still to check. See [preparation usage](grp2/VIT_TRIAL.md). The first encoder check will use inference only and one image at a time. No need to compare all candidate models or sweep resolutions before establishing this path.

**Still open:** final training resolution/preparation, trainable versus frozen encoder layers, training batch size, depth/time scaling and use, goal/history fusion, and evaluation protocol. The dataset's original 1065×1050 resolution remains the inspection reference. Tran reported that the inspected prepared previews look fine and similar but smaller; dataset-wide visibility remains unverified. Neither 224×224 nor original-resolution training has been selected.

**Original staged implementation sequence (historical; CPU stages now passed, see Group 2 log):**

1. Check remaining PyTorch/timm compatibility on Perlmutter; do not repeat completed Python/Pillow setup. The candidate discussion is complete for choosing the first trial; no multi-model sweep is required now.
2. Add explicit model-input preparation on top of `GuidewireDataset` and a ViT feature encoder, without implementing the GRU or action classifier yet. `preprocessing.py` is implemented with its one-case interface documented in the preparation guide; `vit.py` now supplies separate 384-value goal/history summaries, with synthetic bookkeeping checks only so far. Keep sensitive outputs on Perlmutter and use invented inputs for laptop checks.
3. On Perlmutter, inspect candidate prepared training images against originals for wire-tip and target visibility. Run one real case through the encoder in inference mode, checking shapes, finite features, ordering, and separate goal handling. No action-accuracy claim follows from feature extraction working.
4. Add/test a small batch, then measure GPU inference memory and runtime. Choose training batch size later using the complete model and freezing policy; inference memory alone is insufficient. Batch size 1 is a smoke check, not a final training choice. Decide encoder freezing and complete model interfaces before training integration.

Use training cases for inspection and learning; use validation for configuration/checkpoint selection. Reserve test partitions for final evaluation after choices are fixed. Reusing training data over epochs and experiments is normal; using test results to guide changes compromises final evaluation. Agree on the five supplied splits' reporting protocol before comparative experiments. Augmentation can be added later in the preparation layer without changing originals, but geometric transforms must preserve labels and sequence/goal consistency.

## Research objective and prior observations

The project owner describes an open-ended exploration of approaches not yet tried by the prior experimenters. The objective is a specialized smaller model with higher accuracy than the baseline and lower decision latency than the tested frontier model. The architecture proposal below is a starting plan, not a restriction on alternatives.

Reported context, not independently verified:

- Previously tested very small locally hosted models performed poorly. Insufficient capacity is the owner's suggested explanation; the cause has not been established.
- A frontier model referred to as “Astra 6” reportedly achieved better accuracy than the baseline. Its exact model identifier and setup are unconfirmed.
- Approximately 82% accuracy and 3–4 seconds per decision were reported. Attribution of these figures to the frontier model versus a separate baseline needs clarification before adopting numerical targets.

Open benchmark questions: model identities, baseline definition, evaluation population and split, input information, and latency measurement conditions. Keep the prior baseline distinct from the proposed current-input baseline until their relationship is confirmed.

Suggested comparison practice, pending team agreement: use matching held-out partitions and input information, and record end-to-end batch-one latency with hardware and timing boundaries. No speed or accuracy improvement has been demonstrated by this project yet.

## Shared progress

- 2026-10-30 — **Data readiness:** basic dataset checks passed in owner-run Perlmutter commands. Evidence and remaining limits are in [DATASET.md](DATASET.md). No dataset changes; do not repeat completed validation without a concrete reason.
- 2026-10-30 — **Image-reading environment:** the owner loaded `pytorch/2.13.0` and confirmed Python 3.12.13 and Pillow 12.3.0. The module changed CUDA 13.2 to 13.0 and GCC 14 to 13.2. No installation was needed. Full ML compatibility, GPU execution, and team adoption remain open. NERSC documents its provided modules as the easiest [PyTorch starting route](https://docs.nersc.gov/machinelearning/pytorch/).
- 2026-10-30 — **Collaboration and reconciliation:** separated group progress and clarified how the incoming outlines fit the raw reader. Fetched origin and fast-forwarded the laptop branch to `77947cd`; backed-up local reader, CNN outline, and notebook were restored unchanged for review. Origin's commit is part of local history. Current implementation, verification, and handoff status are recorded in [Group 2 progress](grp2/PROGRESS.md).

## Next milestone

Run the before-GRU development comparisons on Perlmutter compute, record class recalls/train-validation gaps and review model disagreements privately. Use results to choose a bounded next experiment (fusion, spatial representation, resolution, or limited fine-tuning) before implementing GRU. [EVALUATION.md](EVALUATION.md) owns the protocol and improvement hypotheses; [Group 2 progress](grp2/PROGRESS.md) owns verification evidence. Real-data accuracy, GPU performance, and full-dataset semantic correctness remain unverified.
