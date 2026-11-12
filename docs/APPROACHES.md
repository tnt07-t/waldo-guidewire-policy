# Approaches and team coordination

For teammates. Start here to understand the pipeline, responsibilities, and open decisions. The [documentation guide](README.md) identifies what people and AI assistants should read.

## Supplied project objectives and deliverables — 2026-11-03

**Source:** project objective/deliverables text pasted by Tran on November 3, including the code/tools reference [Project Waldo](https://projectwaldo.ai/). The requirements below come from that supplied text; the website was not independently inspected for additional requirements. A complete grading rubric has not been supplied.

**Objectives stated in the source:**

- Predict ADVANCE or RETRACT from a decision point containing an artery map with target, recent frames and insertion-depth history.
- Beat the baselines on the evaluation set: the stated floor is the **62% always-advance score**, with the ideal of approaching **Astra's 82%**. These are supplied reference targets, not results achieved by our models.
- Report accuracy overall and separately for advance and retract cases because the classes are imbalanced. Accuracy restricted to each true class corresponds to that class's recall; report both class results alongside the overall score.
- Report **per-decision inference latency on a single GPU** and the **accuracy-versus-latency trade-off for every variant tried**.
- Try several model families, and hypothesize/analyze why some work better than others on this data.

**Example deliverables listed in the source:** a trained model with an inference script; analysis justifying architectural decisions; overall/advance/retract accuracy and latency on the evaluation set; a comparison plot against baselines; and an error analysis of incorrect predictions. The source calls these examples; it does not supply grading weights or a complete submission checklist.

**Training versus inference constraints:** “Report per-decision latency on a single GPU” specifies the hardware scope of the inference measurement. This excerpt does **not** specify training GPU count/type, a training-time budget, batch size or epoch count, nor a numerical inference-latency threshold. Our short GPU allocations, pre-allocation preparation, runtime caps and prompt release are owner-directed resource-management constraints and experiment choices, not course limits established by this excerpt. Do not infer unlimited training permission from that omission.

**Measurement plan, not an additional quoted requirement:** benchmark the complete decision pipeline (preparation, image encoding, temporal processing and classification) with documented hardware, timing boundaries, warm-up and CUDA synchronization. Training duration, tiny-run feature extraction time and head-only timing do not substitute for per-decision inference latency. Any feature-reuse scenario must state its assumptions separately.

**Still open:** which supplied partition(s)/split positions constitute the official “evaluation set”; the intended final reporting protocol; exact baseline/Astra identifiers and reproducible measurement conditions. The excerpt does not select validation versus test or authorize using test results to tune models. Continue reserving test data while these details are resolved. The older reported 3–4-second figure below is not a latency requirement in this supplied text.

## How the current files fit together

We are building one pipeline in stages:

1. **Read original data — `dataset.py`.** Tran's reader selects cases from a supplied partition and returns the goal, six ordered history images, depths, times, label, and tracing identifiers. It preserves original images. Six synthetic tests and one real-case decode passed on Perlmutter; see the Group 2 milestone for evidence and limits.
2. **Prepare model inputs — `preprocessing.py`.** The first DINOv2 preparation layer converts full-view RGB images into normalized tensors with separate goal/history previews; owner-run initial real-data previews were reported suitable, with inspection coverage unspecified. Origin's dataset outline included this stage by proposing stacked frames in a PyTorch Dataset. Our reader deliberately starts earlier so we can inspect the data before selecting model-specific preprocessing.
   Group 1 added an opt-in goal-centered native-resolution `WirePreprocessor` for the CNN; it did not beat the full-frame baseline on split 0 and is not adopted ([Group 1 progress](grp1/PROGRESS.md#2026-11-03---wire-preserving-preprocessing-and-frozen-feature-probe-owner-run-ladder-stopped-early)).
3. **Extract visual features — `vit.py` / `cnn.py`.** Inference-only DINOv2 extraction is implemented locally, with separate goal/history summary vectors; owner-run pretrained CPU checks are recorded in Group 2 progress. Group 1's frozen ResNet-18 frame encoder is implemented in `cnn.py` (512-d goal/history features; owner-run one-case Perlmutter smoke check passed). Separate `cnn_diagnostics.py` probes 512-d occlusion/marker/temporal sensitivity for a later GRU (not wire or action claims); an owner-run one-case GPU diagnostic completed, then local correctness fixes were applied and a Perlmutter re-run is pending (see Group 1 progress). Group 2 now has a locally tested frozen ViT–GRU debugging head with late goal fusion in its separate worktree; the first bounded full-training setup is agreed and implemented locally (see Group 2 protocol).
4. **Evaluate fixed representations — `evaluation.py`.** The project owner completed the first frozen-feature GPU development comparison. See [evaluation protocol and results](EVALUATION.md); this does not verify the separate GRU path.

These are compatible stages, not competing loaders. Origin's `dataset.py` was TODO-only: it proposed `(frames, depth_history, label)` without implementing loading or clarifying the separate goal and time offsets. Our reader makes those roles explicit and returns a dictionary of raw inputs. Build later model preparation on that reader; agree on interface changes before implementation.

`data_inspect.ipynb` is an optional interactive notebook for understanding the data between reading and model preparation. It combines explanations, executable Python cells, and visual results. The owner successfully displayed a goal and six history frames using cells entered on Perlmutter. The shared notebook now includes reusable, output-free loading and history-display cells based on those snippets; other analysis sections remain placeholders. Sensitive notebook outputs must stay on Perlmutter.

Raw reading remains separate from model preparation. The prepared tensors feed either encoder; the evaluation runner constructs in-memory features and fits small action classifiers. It does not change loader/encoder contracts or add a GRU.

The tradeoff is intentional: raw reading is easier to inspect and reuse now, but it is not ready to feed directly into a CNN or ViT. The separate preparation layer now supplies one-case tensors for the DINOv2 trial. Encoder inference and case-batch bookkeeping are implemented; owner-run pretrained CPU checks passed for one and two cases; the bounded real GPU extraction/backward check passed for Group 2 on November 3.

## Team proposal — supplied by project owner

- [Group 1 — Andrew and Suhani](grp1/PROGRESS.md): CNN–GRU, CNN–LSTM, and current-input baseline. Group-specific plans and progress live there.
- [Group 2 — Transformer models](grp2/PROGRESS.md): ViT–GRU and CNN–temporal attention. Group-specific plans and progress live there.

Joint optional extension: JEPA, after the primary experiments work.

Friday task assignments and the Sunday CNN target are recorded in the group logs. Tran has completed the initial CPU feature-extraction checks for the ViT–GRU approach; remaining architecture ownership is not finalized. “Artery map” correspondence to the goal reference remains open. Project objectives were supplied November 3 as recorded above; the team architecture assignments are proposals, and a complete grading rubric has not been supplied.

## What is decided and what is still open

The first loader preserves original images and metadata. Tran approved the initial ViT preparation below; pretrained CPU extraction has since passed for inspected cases. The configuration remains a bounded trial. On 2026-11-01 the project owner authorized local frozen-encoder evaluation and documentation: concatenate goal/current features, compare diagnostic input sets, use train-only scaling and small validation-selected L2 probes. These development defaults are documented in EVALUATION.md; team-wide final split reporting remains open.

The bounded real ViT–GRU GPU extraction/backward check passed on November 3; full-training settings are recorded in the Group 2 protocol.

Before model integration, agree on input preparation, goal/history fusion, depth/time use, and batching. Then agree on the supplied split reporting protocol, metrics, and model selection. We have not selected a five-fold cross-validation protocol. Reuse shared components where appropriate rather than implementing separate data readers for each architecture.

## Initial ViT preparation trial

**Scope approved by Tran (2026-10-31):** start with the pretrained ViT image encoder for ViT–GRU, preparing and testing its inputs before implementing the complete policy. The ViT extracts features from individual images; the later GRU combines the six history steps. CNN–temporal attention is a separate Group 2 approach, not an additional component of ViT–GRU.

**Starting preparation:** preserve the full field of view (no cropping), retain RGB including the red goal marker, convert images to floating-point tensors, use the selected checkpoint's documented pixel scaling/normalization, and start without random augmentation. Keep original files unchanged and preparation separate from the raw reader. Preserve the separate goal and chronological history, along with aligned depth/time metadata; labels and identifiers are not encoder inputs. This is an initial experiment, not a team-wide preprocessing contract.

**First concrete trial (2026-10-31):** following the candidate discussion, Tran requested the first implementation step for DINOv2 Small/14 (`vit_small_patch14_dinov2.lvd142m`), using its 518×518 input size, proportional bicubic resize, centered mean-color padding, and documented scaling/normalization. This preserves the full field of view. `preprocessing.py` handles preparation; `vit.py` loads the pretrained encoder through timm, with owner-run Perlmutter CPU compatibility and pretrained extraction verified. See [preparation usage](grp2/VIT_TRIAL.md). The completed first encoder check used inference only and one image at a time; two-case grouping and image chunk sizes 1 and 2 also passed. No need to compare all candidate models or sweep resolutions before establishing this path.

**Still open for later full training:** final resolution/preparation, schedule, batch size and evaluation protocol. The bounded debugging settings are now fixed as described below; any later fine-tuning or alternative scaling/fusion requires a separately scoped comparison. The dataset's original 1065×1050 resolution remains the inspection reference. Tran reported that the inspected prepared previews look fine and similar but smaller; dataset-wide visibility remains unverified. Neither 224×224 nor original-resolution training has been selected.

**Implementation and verification status:**

1. **Completed:** preparation, initial preview inspection, and Perlmutter torch/timm compatibility for CPU extraction. The scratch venv adds timm 1.0.30 to the module environment; notebook kernel selection remains unconfirmed.
2. **Completed:** inference-only DINOv2 extraction with separate 384-value goal/history vectors. Owner-run one-case and two-case CPU checks passed, including agreement between grouped and individual extraction at image chunk sizes 1 and 2. Detailed evidence and limits live in [Group 2 progress](grp2/PROGRESS.md); do not repeat these checks without a relevant change or concern.
3. **Implemented Group 2 debugging setup (2026-11-02):** frozen ViT, one-layer GRU with 64 hidden units, aligned depth/time scaled by training-only channel standard deviations without centering, and late goal concatenation into a 448→64→2 classifier. The separate runner selects four cases per label from split-0 training with seed 0 and uses full-batch Adam at 0.001, capped at 300 updates/120 seconds of head work and a 600-second worker deadline. All 44 synthetic tests passed. See [the handoff](grp2/GRU_DEBUG_HANDOFF.md) for precise limits and commands; the real eight-case check passed November 3 at `b9d1bdf`.
4. **Completed tiny verification; full setup prepared (2026-11-03):** the real check passed and its allocation was released. Tran approved preparing fresh split-0 full training with batch 32, Adam 0.001, no weight decay, 100-epoch/900-update ceilings and validation balanced-accuracy selection with patience 20. See [the agreed protocol](grp2/GRU_TRIAL.md#agreed-first-full-training-setup--2026-11-03) for all extraction/storage/time limits. Real full training is pending; no persistent feature cache, test evaluation or automatic sweep is authorized.

Use training cases for inspection and learning; use validation for configuration/checkpoint selection. Reserve test partitions for final evaluation after choices are fixed. Reusing training data over epochs and experiments is normal; using test results to guide changes compromises final evaluation. Agree on the five supplied splits' reporting protocol before comparative experiments. Augmentation can be added later in the preparation layer without changing originals, but geometric transforms must preserve labels and sequence/goal consistency.

## Research objective and prior observations

The project owner describes an open-ended exploration of approaches not yet tried by the prior experimenters. The objective is a specialized smaller model with higher accuracy than the baseline and lower decision latency than the tested frontier model. The architecture proposal below is a starting plan, not a restriction on alternatives.

Reported context, not independently verified:

- Previously tested very small locally hosted models performed poorly. Insufficient capacity is the owner's suggested explanation; the cause has not been established.
- A frontier model referred to as “Astra 6” reportedly achieved better accuracy than the baseline. Its exact model identifier and setup are unconfirmed.
- Approximately 82% accuracy and 3–4 seconds per decision were reported. Attribution of these figures to the frontier model versus a separate baseline needs clarification before adopting numerical targets.

The November 3 supplied objectives identify the 62% floor as always-advance and the 82% aspiration as Astra. Exact model identities, reproducible baseline setup, evaluation population/split, input information and latency measurement conditions remain open. Keep those supplied references distinct from the implemented static comparison classifiers; the older 3–4-second report remains unverified.

Group 2's [GRU trial plan](grp2/GRU_TRIAL.md) uses a frozen latest-image-plus-goal classifier as a later comparison, including a historical-metadata variant. The project owner already implemented both static probes on `static-evaluation` (`070328f`); see [evaluation evidence and limits](EVALUATION.md). Tran authorized copying that evaluation code/tests into the separate `tran/vit-gru-debug` worktree while keeping shared documentation on main. The 64-unit GRU head and bounded tiny trainer are now implemented and synthetically tested in that worktree. The probes are distinct from the reported prior benchmark; align scaling, training/selection rules and Group 1 baseline ownership before a matched comparison. Tran agreed to debug the combined-input GRU on eight training cases first, with multi-model comparisons deferred. The implemented tiny-run settings are documented above; full-training and matched-comparison budgets remain to be agreed.

Suggested comparison practice, pending team agreement: use matching held-out partitions and input information, and record end-to-end batch-one latency with hardware and timing boundaries. No speed or accuracy improvement has been demonstrated by this project yet.

## Shared progress

- 2026-11-06 — **Group 1 regularization experiment:** at the owner's request, added an explicit unregularized control and light/medium/heavy weight-decay variants of the same CNN–GRU, with matched epoch budgets and seeds. Added per-epoch CSV recording and a sweep summary of paired validation differences. The former regularized baseline remains available as `legacy-baseline`; historical results are not reclassified. Hypotheses and Perlmutter commands are in [Group 1's experiment protocol](grp1/NEXT_EXPERIMENTS.md). No real-data outcome or statistical significance is established by these changes.
- 2026-10-30 — **Data readiness:** basic dataset checks passed in owner-run Perlmutter commands. Evidence and remaining limits are in [DATASET.md](DATASET.md). No dataset changes; do not repeat completed validation without a concrete reason.
- 2026-10-30 — **Image-reading environment:** the owner loaded `pytorch/2.13.0` and confirmed Python 3.12.13 and Pillow 12.3.0. The module changed CUDA 13.2 to 13.0 and GCC 14 to 13.2. No installation was needed. This was the initial environment check. Subsequent Group 2 pretrained CPU verification and Group 1 one-case GPU diagnostics are recorded in their group logs; Group 2 GPU execution and team-wide adoption remain open. NERSC documents its provided modules as the easiest [PyTorch starting route](https://docs.nersc.gov/machinelearning/pytorch/).
- 2026-10-30 — **Collaboration and reconciliation:** separated group progress and clarified how the incoming outlines fit the raw reader. Fetched origin and fast-forwarded the laptop branch to `77947cd`; backed-up local reader, CNN outline, and notebook were restored unchanged for review. Origin's commit is part of local history. Current implementation, verification, and handoff status are recorded in [Group 2 progress](grp2/PROGRESS.md).

## Next milestone

The Group 2 real eight-case check passed at `b9d1bdf` and the allocation was released. The new full-training runner/tests passed 51 local synthetic tests and were committed and pushed on `tran/vit-gru-debug` at `69fd063`; shared documentation remains on main. Real full training is pending. Use [the current setup handoff](grp2/GRU_DEBUG_HANDOFF.md#full-training-setup--2026-11-03) for code sharing, checks, notebook preservation and bounded execution. Do not repeat tiny fitting or masking.

The CNN–attention fixed-64 development comparison completed: latest/mean/attention validation BA .597/.601/.500. Further attention ablations are deferred pending diagnostics; see [attention experiment results](grp2/ATTENTION_EXPERIMENTS.md). Static probe results remain in [EVALUATION.md](EVALUATION.md).

The Group 2 frozen ViT–GRU debugging code is ready for review and its first real eight-case check on Perlmutter. All 44 local synthetic tests passed; pretrained GPU extraction/backward and real-case fitting remain unverified for Tran's path. Shared documentation is on main and code is in the separate `tran/vit-gru-debug` worktree; remote publication remains pending; check the two worktrees' Git logs/status for exact local revisions. Use [the debugging/The project owner handoff](grp2/GRU_DEBUG_HANDOFF.md) for code sharing, tests, notebook preservation, execution and permitted reporting.

No further manual masking is needed. the static-evaluation existing static classifiers are retained for the later matched comparison. Full training must start fresh, with validation-only selection and agreed width/baseline budgets; final tests and the five-split protocol remain separate. Tiny fitting cannot establish generalization, visual-history benefit or full-policy latency. See [Group 2 progress](grp2/PROGRESS.md) for evidence and restrictions.
