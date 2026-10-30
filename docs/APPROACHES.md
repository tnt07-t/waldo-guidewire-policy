# Approaches and team coordination

For teammates. Start here to understand the pipeline, responsibilities, and open decisions. The [documentation guide](README.md) identifies what people and AI assistants should read.

## How the current files fit together

We are building one pipeline in stages:

1. **Read original data — `dataset.py`.** Tran's reader selects cases from a supplied partition and returns the goal, six ordered history images, depths, times, label, and tracing identifiers. It preserves original images. Six synthetic tests and one real-case decode passed on Perlmutter; see the Group 2 milestone for evidence and limits.
2. **Prepare model inputs — not implemented.** Choose image preparation and convert images into numerical tensors. Origin's dataset outline included this stage by proposing stacked frames in a PyTorch Dataset. Our reader deliberately starts earlier so we can inspect the data before selecting model-specific preprocessing.
3. **Extract visual features — `cnn.py` outline only.** A future encoder turns prepared images into useful numerical features. ResNet-18 is a proposal, not a selected model. Goal/history fusion and temporal prediction remain undecided.

These are compatible stages, not competing loaders. Origin's `dataset.py` was TODO-only: it proposed `(frames, depth_history, label)` without implementing loading or clarifying the separate goal and time offsets. Our reader makes those roles explicit and returns a dictionary of raw inputs. Build later model preparation on that reader; agree on interface changes before implementation.

`data_inspect.ipynb` is an optional interactive notebook for understanding the data between reading and model preparation. It combines explanations, executable Python cells, and visual results. The owner successfully displayed a goal and six history frames using cells entered on Perlmutter. The shared notebook now includes reusable, output-free loading and history-display cells based on those snippets; other analysis sections remain placeholders. Sensitive notebook outputs must stay on Perlmutter.

The tradeoff is intentional: raw reading is easier to inspect and reuse now, but it is not ready to feed directly into a CNN or ViT. Tensor conversion, model-specific preprocessing, and batching still need agreement and implementation.

## Team proposal — supplied by project owner

- [Group 1 — Andrew and Suhani](grp1/PROGRESS.md): CNN–GRU, CNN–LSTM, and current-input baseline. Group-specific plans and progress live there.
- [Group 2 — Tran and Sabrina](grp2/PROGRESS.md): ViT–GRU and CNN–temporal attention. Group-specific plans and progress live there.

Joint optional extension: JEPA, after the primary experiments work.

Friday task assignments and the Sunday CNN target are recorded in the group logs. Exact architecture ownership is not finalized. “Artery map” correspondence to the goal reference remains open. The official final-project rubric has not been supplied; none of these choices is documented as a course requirement.

## What is decided and what is still open

The first loader preserves original images and metadata. Batching, image resizing/cropping, color conversion, normalization, augmentation, and model implementation are deferred. Earlier assistant suggestions for particular encoders, image sizes, and freezing policies are not team decisions.

Before model integration, agree on input preparation, goal/history fusion, depth/time use, and batching. Then agree on the supplied split reporting protocol, metrics, and model selection. We have not selected a five-fold cross-validation protocol. Reuse shared components where appropriate rather than implementing separate data readers for each architecture.

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

The goal and six history images now display successfully in the Perlmutter notebook. Next, assess wire/target visibility and image/metadata meaning before agreeing on model preprocessing. Display success alone does not establish semantic correctness. See [LOADER.md](LOADER.md) for usage and [Group 2 progress](grp2/PROGRESS.md) for completed milestones and verification limits.
