# Approaches and team coordination

For teammates. Start here to understand the pipeline, responsibilities, and open decisions. The [documentation guide](README.md) identifies what people and AI assistants should read.

## How the current files fit together

We are building one pipeline in stages:

1. **Read original data — `dataset.py`.** Tran's reader selects cases from a supplied partition and returns the goal, six ordered history images, depths, times, label, and tracing identifiers. It preserves original images. Six synthetic tests and one real-case decode passed on Perlmutter; see the Group 2 milestone for evidence and limits.
2. **Prepare model inputs — `preprocessing.py`.** The first DINOv2 preparation layer converts full-view RGB images into normalized tensors with separate goal/history previews; real-data verification is pending. Origin's dataset outline included this stage by proposing stacked frames in a PyTorch Dataset. Our reader deliberately starts earlier so we can inspect the data before selecting model-specific preprocessing.
3. **Extract visual features — `cnn.py` outline only.** A future encoder turns prepared images into useful numerical features. ResNet-18 is a proposal, not a selected model. Goal/history fusion and temporal prediction remain undecided.

These are compatible stages, not competing loaders. Origin's `dataset.py` was TODO-only: it proposed `(frames, depth_history, label)` without implementing loading or clarifying the separate goal and time offsets. Our reader makes those roles explicit and returns a dictionary of raw inputs. Build later model preparation on that reader; agree on interface changes before implementation.

`data_inspect.ipynb` is an optional interactive notebook for understanding the data between reading and model preparation. It combines explanations, executable Python cells, and visual results. The owner successfully displayed a goal and six history frames using cells entered on Perlmutter. The shared notebook now includes reusable, output-free loading and history-display cells based on those snippets; other analysis sections remain placeholders. Sensitive notebook outputs must stay on Perlmutter.

The tradeoff is intentional: raw reading is easier to inspect and reuse now, but it is not ready to feed directly into a CNN or ViT. The separate preparation layer now supplies one-case tensors for the DINOv2 trial. Encoder integration and case batching remain pending.

## Team proposal — supplied by project owner

- [Group 1 — Andrew and Suhani](grp1/PROGRESS.md): CNN–GRU, CNN–LSTM, and current-input baseline. Group-specific plans and progress live there.
- [Group 2 — Tran and Sabrina](grp2/PROGRESS.md): ViT–GRU and CNN–temporal attention. Group-specific plans and progress live there.

Joint optional extension: JEPA, after the primary experiments work.

Friday task assignments and the Sunday CNN target are recorded in the group logs. Tran is starting with ViT feature extraction for the ViT–GRU approach; remaining architecture ownership is not finalized. “Artery map” correspondence to the goal reference remains open. The official final-project rubric has not been supplied; none of these choices is documented as a course requirement.

## What is decided and what is still open

The first loader preserves original images and metadata. Tran has approved trying the initial ViT preparation described below. The first preparation layer is implemented; encoder code remains pending. The configuration below is a bounded trial, not a team-wide training decision.

Before model integration, agree on input preparation, goal/history fusion, depth/time use, and batching. Then agree on the supplied split reporting protocol, metrics, and model selection. We have not selected a five-fold cross-validation protocol. Reuse shared components where appropriate rather than implementing separate data readers for each architecture.

## Initial ViT preparation trial

**Scope approved by Tran (2026-10-31):** start with the pretrained ViT image encoder for ViT–GRU, preparing and testing its inputs before implementing the complete policy. The ViT extracts features from individual images; the later GRU combines the six history steps. CNN–temporal attention is a separate Group 2 approach, not an additional component of ViT–GRU.

**Starting preparation:** preserve the full field of view (no cropping), retain RGB including the red goal marker, convert images to floating-point tensors, use the selected checkpoint's documented pixel scaling/normalization, and start without random augmentation. Keep original files unchanged and preparation separate from the raw reader. Preserve the separate goal and chronological history, along with aligned depth/time metadata; labels and identifiers are not encoder inputs. This is an initial experiment, not a team-wide preprocessing contract.

**First concrete trial (2026-10-31):** following the candidate discussion, Tran requested the first implementation step for DINOv2 Small/14 (`vit_small_patch14_dinov2.lvd142m`), using its 518×518 input size, proportional bicubic resize, centered mean-color padding, and documented scaling/normalization. This preserves the full field of view. `preprocessing.py` implements preparation only; the planned encoder uses timm, with remote compatibility still to check. See [preparation usage](grp2/VIT_TRIAL.md). The first encoder check will use inference only and one image at a time. No need to compare all candidate models or sweep resolutions before establishing this path.

**Still open:** final training resolution/preparation, trainable versus frozen encoder layers, training batch size, depth/time scaling and use, goal/history fusion, and evaluation protocol. The dataset's original 1065×1050 resolution remains the inspection reference. Real-image visibility after resizing is unverified. Neither 224×224 nor original-resolution training has been selected.

**Implementation and verification sequence:**

1. Check remaining PyTorch/timm compatibility on Perlmutter; do not repeat completed Python/Pillow setup. The candidate discussion is complete for choosing the first trial; no multi-model sweep is required now.
2. Add explicit model-input preparation on top of `GuidewireDataset` and a ViT feature encoder, without implementing the GRU or action classifier yet. `preprocessing.py` is implemented with its one-case interface documented in the preparation guide; `vit.py` is the next planned component. Keep sensitive outputs on Perlmutter and use invented inputs for laptop checks.
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

The goal and six history images display successfully, and Tran has reported initial visibility/order observations in the Group 2 log. Next: inspect the implemented DINOv2 preparation on one training case, then implement one-case feature extraction under the trial above. Full metadata semantics and dataset-wide visual correctness remain unverified. See [LOADER.md](LOADER.md) for raw reader usage and [Group 2 progress](grp2/PROGRESS.md) for evidence and limits.
