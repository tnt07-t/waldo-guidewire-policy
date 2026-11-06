# Midpoint poster preparation — team working document

Updated November 5, 2026. **For Tran, Sabrina, Andrew and Suhani to complete together before designing the poster.** This document collects the story, evidence, figure specifications and missing inputs. It is not the finished poster. Detailed experiment records remain in the linked source notes.

**How to use:** each owner checks their prefilled section, fills the explicit blanks, selects their strongest evidence and supplies a caption explaining what it means. Treat “not recorded” as a documentation gap, not proof that a teammate has not done the work. Keep proposed experiments distinct from completed runs.

## 1. What the instructor asks us to show

Source: poster-requirements screenshot supplied by Tran.

- The problem and why it matters.
- The system/model so far, design choices, and our own ideas for addressing the problem.
- Progress and preliminary results, including graphs, measurements and negative results. Explain what we learned ourselves from the experiments.
- A brief plan through the final poster session on **Monday, December 7**. Future work should not occupy most of the poster.

**Still to fill:** midpoint date [ ]; poster dimensions/template [ ]; presentation length [ ]; final title/authors [ ].

## 2. Shared story and proposed poster layout

Working title: **Learning Guidewire Action Prediction from Visual and Movement History**.

Shared question: **Can smaller vision models predict ADVANCE/RETRACT from a target reference and recent history, and what do we gain from different encoders, temporal models and input representations?**

Provisional takeaway: our implemented pipelines can fit training examples, but validation generalization remains difficult. Static references are competitive with the temporal models tested so far. Input conditioning, class bias and overfitting matter, so architecture comparisons require more than one headline accuracy.

This takeaway draws on the documented ViT and CNN–attention experiments. Andrew and Suhani should add their latest evidence before the team settles the final claim.

### Space plan — proposal, adjustable to poster dimensions

- **Top strip, about 10%:** title, task, motivation and one-sentence takeaway.
- **Left column, about 25%:** task/data and shared pipeline; CNN encoder/diagnostics and Andrew/Suhani's model work.
- **Middle column, about 30%:** Sabrina's probe/attention progression and Tran's ViT progression, with comparable space for each. Use short hypothesis → change → result summaries.
- **Right column, about 25%:** shared results plot, class-level performance, representative learning curves and interpretation.
- **Bottom strip, about 10%:** limitations, concrete next steps, references/contribution labels.

Reserve actual space for each workstream now. Tran's five runs should fit into one compact panel, not dictate the whole poster. If Andrew/Suhani supply completed policy results, integrate them into the shared results figure and adjust the method panels accordingly.

## 3. Shared task, data and evaluation context

### Content ready to condense onto the poster

- Predict the next operator-action proxy: ADVANCE or RETRACT. Labels describe inferred behavior, not verified optimal or clinically safe decisions.
- Inputs: a separate target-marked goal reference, six chronological images, and six aligned relative depth/time pairs. The goal is not a seventh history step; image spacing may vary.
- Dataset: **416 decision points from 120 episodes; 259 ADVANCE and 157 RETRACT**.
- Current documented development split position 0: **267 training / 80 validation / 69 reserved test cases**, with episode-separated partitions. Five supplied split assignments exist; the final reporting protocol is still open.
- Initial shared image preparation: RGB, full-view proportional resize/pad to 518×518, pretrained normalization. Preserve target marker and frame/metadata alignment.
- Report overall accuracy, balanced accuracy (BA = average of the two class recalls), RETRACT recall and ADVANCE recall. A single majority prediction gets 50% BA.
- Supplied reference targets: 62% always-advance overall accuracy and approximately 82% Astra overall accuracy. These are supplied references, not reproduced results on our development protocol. Do not draw them as baselines on a BA chart.
- Motivation: study the accuracy–latency trade-off of smaller models. No team-wide speed/accuracy advantage over Astra is established yet.

**Team to fill:** agreed comparison/checkpoint rules [ ]; official final evaluation partition/protocol [ ]; reproduced baseline conditions if available [ ].

## 4. Workstream A — CNN encoder and temporal policies: Andrew & Suhani

### Recorded progress

- Implemented frozen ImageNet-pretrained ResNet-18 with shared encoding of the goal and history, producing 512-value features per image. Evaluation mode keeps BatchNorm stable.
- One real-case encoder smoke check passed on Perlmutter.
- Implemented probes for regional occlusion sensitivity, goal-marker perturbation versus controls, and temporal feature changes alongside depth changes.
- A real GPU diagnostic ran, followed by correctness fixes to marker arithmetic, controls and temporal interpretation. The recorded numerical diagnostic results predate those fixes; updated real-run evidence is needed before using them on the poster.
- Proposed CNN–GRU, CNN–LSTM and current-input baseline work appears in the group notes. Completion/results are not recorded there; **owners should update this section with their current work**.

### Scientific point and figure slot

Explain why a CNN is a useful visual backbone and what the diagnostic checks actually test. Occlusion sensitivity is not wire localization, and varying embeddings do not prove useful action information.

**Figure A:** compact CNN architecture or a permitted diagnostic summary. If policy runs exist, prioritize their validation comparison/learning curves over setup evidence. Do not use old marker-ratio numbers as corrected findings.

### Andrew & Suhani fill-in

- Individual contributions / presenter(s): [ ]
- Latest implemented model(s), inputs and fusion choices: [ ]
- Hypothesis and why this design: [ ]
- Completed run(s), revision, split/seed and checkpoint selection: [ ]
- Accuracy / BA / RETRACT recall / ADVANCE recall: [ ]
- Training-versus-validation finding, including unsuccessful attempts: [ ]
- Corrected diagnostic result and interpretation, if available: [ ]
- Figure A source + caption + one key takeaway: [ ]
- Next experiment and what decision its result would change: [ ]

Source: [Group 1 progress](grp1/PROGRESS.md), [CNN](grp1/CNN.md), [CNN diagnostics](grp1/CNN_DIAGNOSTICS.md).

## 5. Workstream B — comparative probes and CNN–attention: Sabrina

### Recorded progression and why it matters

1. **Frozen-feature logistic probes:** compare ResNet-18 and DINOv2, and isolate latest image, goal, movement metadata and mean history. Ask what the features predict before adding a temporal model.
2. **Neural static/mean/attention comparison:** keep a nonlinear classifier and test whether temporal attention helps beyond latest-image or unordered history.
3. **Diagnose weak learning before enlarging the model:** inspect selected/final checkpoints, per-epoch behavior, hidden activations and input dependence.
4. **Activation and input conditioning:** LeakyReLU restored gradients in the latest+goal control but left BA at chance. Training-only feature standardization restored strong training fit, exposing overfitting.
5. **Regularization:** higher dropout and weight decay did not clearly improve selected validation BA in the recorded runs.

### Selected evidence to use

Earlier logistic probes, separate protocol:

- Metadata-only validation BA **54.7%**; DINOv2 goal-only **60.9%**.
- DINOv2 latest+goal **57.0%**; ResNet latest+goal **54.7%**.
- Mean-history+goal **54.0%** for both encoders. Large train/validation gaps; goal-only strength does not establish leakage or target-marker reasoning.

CNN neural experiments, selected validation BA:

- Latest-image MLP + goal + all movement metadata: **59.7%**.
- Mean-history MLP + goal + all movement metadata: **60.1%**.
- Temporal attention: **50.0%**, selected epoch 1; final train/validation BA **63.6% / 42.5%**.
- Latest+goal, raw features, ReLU: **50.0%**; inactive hidden layer diagnosed.
- Latest+goal, raw features, LeakyReLU: **50.0%**.
- Latest+goal, standardized, LeakyReLU: **58.8%**, selected train BA **96.4%**.
- Standardized reference with dropout .3: **58.4%**; with weight decay .01: **58.8%**.

**Key interpretation:** improved training is not automatically improved generalization. Historical images have not demonstrated a benefit in these tested pipelines, but training failures and changed input/architecture settings limit causal conclusions.

### Figures B and C

**Figure B:** small experiment sequence: inactive classifier → gradients restored → standardization restores fit → validation gap persists. Show selected train and validation BA side by side. Label activation/standardization changes so it is clear what each experiment tests.

**Figure C, optional timing panel:** A100 same-case batch-one median total of **33.07 ms latest MLP / 131.17 ms mean-history / 132.24 ms attention**. These include preparation/model, exclude disk IO, assume no feature reuse and are not workload-wide latency. Temporal models encode seven images versus two for latest. Do not attribute the entire difference to attention itself.

### Sabrina fill-in

- Confirm ownership, current results and any newer runs: [ ]
- Choose 3–4 central comparisons; keep remaining probes as discussion backup: [ ]
- Selected class recalls/confusion counts for the chosen model(s): [ ]
- Figure B/C source, final captions and timing boundaries: [ ]
- Strongest original insight and next discriminating experiment: [ ]

Sources: [Evaluation](EVALUATION.md); newer CNN–attention records in `docs/grp2/ATTENTION_EXPERIMENTS.md` and `docs/grp2/CNN_ATTENTION.md` verified on fetched `origin/main` at `e287d2b` (content from `f0e2d37`). These files are present on remote main but not yet in this divergent working copy.

## 6. Workstream C — ViT–GRU, goal fusion and spatial features: Tran

### Contributions and experiment logic

Built the raw reader/inspection path, full-view preparation, pretrained DINOv2 extraction, frozen-encoder GRU training/checkpoint path and follow-up comparisons. DINOv2 Small/14 returns 384-value summaries; the initial temporal head uses a 64-unit GRU.

Five completed development runs, selected validation BA:

1. **Late-goal GRU, Adam — 59.21%.** Establish the full temporal reference after a successful tiny fitting check.
2. **AdamW follow-up — 57.48%.** Respond to overfitting with an optimizer/regularization comparison; no improvement in this run.
3. **Early-goal GRU — 60.87%.** Give the GRU goal context at each step, retaining late fusion. Small observed gain does not prove goal reasoning.
4. **Summary-static — 61.13%.** Latest image + goal + all historical depth/time metadata. Tests whether visual sequence processing earns its extra complexity; nearly tied with early-goal GRU.
5. **Frozen 4×4 spatial grid — 59.65%.** Retain regional patch information rather than a single image summary. No improvement over static here; not a general rejection of spatial features.

Static validation: **65.00% accuracy, 35.29% RETRACT recall, 86.96% ADVANCE recall**; correct counts 12/34 RETRACT and 40/46 ADVANCE. Grid gets 11/34 and 40/46: its BA difference from static amounts to one RETRACT case.

### Figures D and E

**Figure D:** five-run BA dot/bar plot, each labeled with its experimental change. Put the static and early-goal comparison next to each other; avoid implying a reliable ranking from small differences.

**Figure E:** real saved training/validation loss and BA curves for one representative run, with selected checkpoint marked. Show overfitting rather than listing all training logs. Increasing validation loss can reflect more confident errors even when accuracy barely changes.

### Adaptation status — pending evidence, not a sixth score

Last-block ViT adaptation asks whether task-specific visual features help. Pretrained GPU readiness passed. The first training attempt hit its 120-second training budget after nine updates, before a completed/scored epoch or selected checkpoint; no crash/OOM is established. A follow-up `...-02` command was launched in this conversation, but its outcome and remote limits remain unconfirmed. Local 20-minute caps alone do not verify the remote recipe. Add its result only after reviewing the saved summary.

### Tran fill-in

- Confirm latest adaptation status, completed epochs, selected metrics and actual recipe: [ ]
- Fill overall accuracy and both recalls for other chosen GRU comparisons: [ ]
- Confirm saved provenance where still incomplete: [ ]
- Figure D/E source + captions: [ ]
- Choose one strongest lesson and next comparison: [ ]

Backup only: eight-case fitting reached 100% training accuracy, verifying the learning path, not generalization. One-frame tip masking did not show stronger sensitivity than control regions; it cannot establish encoder failure or action usefulness.

Sources: [Group 2 progress](grp2/PROGRESS.md), [GRU trial](grp2/GRU_TRIAL.md), [ViT diagnostics](grp2/VIT_DIAGNOSTICS.md).

## 7. Shared results and figure assembly

### Main figure — comparison across workstreams

Owner: [ ]; reviewers: all. Select a few representative models from A/B/C, rather than giving every experiment equal poster space. Show validation BA with adjacent class recalls, grouping by experiment family. Include 50% majority BA. Preserve full precision in source notes; use consistent rounding in the poster.

Logistic probes, CNN neural runs and ViT neural runs use different optimization/selection settings. Label these groups and describe the figure as a development comparison, not a controlled encoder ranking. Keep missing results blank/pending, never zero. Do not invent error bars; repeated seeds and episode uncertainty are not available for every run.

### Shared pipeline diagram

Owner: [ ]. Inputs → shared preparation → branching ResNet / DINOv2 encoders → static / GRU / LSTM / temporal-attention alternatives → two action scores. Mark implemented-and-run branches separately from planned ones; do not draw every proposed architecture as completed. Goal has its own path; numerical history joins at the location used by each model. Indicate frozen versus trainable components.

### Class-performance figure

Owner: [ ]. Use paired RETRACT/ADVANCE recall bars for representative CNN and ViT models. CNN latest MLP has reported recalls **82.4% / 37.0%**; ViT summary-static has **35.29% / 86.96%**. Their similar BA can conceal opposite class preferences. Because recipes differ, this does not establish an inherent CNN/ViT bias. Verify figures against each selected report before finalizing.

### Figure delivery checklist — each owner completes

- Figure ID and intended poster location: [ ]
- Question the figure answers: [ ]
- Source run/report, code revision, split/seed and checkpoint: [ ]
- Axes, units, metric definition, included timing stages if relevant: [ ]
- Caption: observation + interpretation + material limitation: [ ]
- File/location accessible under data rules and reviewer approval: [ ]

Real data, images, features, full reports, checkpoints and sensitive derived artifacts must stay on Perlmutter. Use permitted aggregate evidence and schematic illustrations; confirm what can be exported before preparing actual-image figures. Never fabricate curves from rounded summaries.

## 8. Team conclusions and discussion prompts

Choose about three findings after all sections are updated:

- **Generalization is the main shared obstacle:** tiny fitting and strong training scores coexist with weak validation results.
- **Diagnose optimization before blaming architecture:** Sabrina's inactive-layer and standardization experiments distinguish learning failure from overfitting.
- **Visual history has not yet shown a clear advantage:** static controls remain competitive, but metadata history, capacity and training differences limit the conclusion.
- **BA alone still hides action imbalance:** show both recalls and inspect errors, not just the best score.
- **Computation depends on the entire pipeline:** extra image preparation/encoding contributes substantially to temporal-model runtime; feature reuse assumptions matter.
- **Andrew/Suhani's additional finding:** [evidence + interpretation + limitation].

Avoid claims that goal-only performance proves leakage, masking proves localization, one spatial trial disproves spatial information, or selected development scores establish final superiority. Validation is small and repeatedly used for model/checkpoint selection; test remains reserved.

## 9. Plan through December 7 — short poster panel

Fill owners and dates together; these are proposed decisions, not an automatic experiment queue.

- **Resolve current evidence gaps:** owners confirm newest runs, failed/budget-limited attempts and selected metrics. Owner/date: [ ].
- **Choose focused follow-ups per workstream:** each change should answer a stated question; coordinate common controls. Owner/date: [ ].
- **Agree on final comparison protocol:** splits/seeds, model-selection budget, both recalls, uncertainty and reserved-test handling. Owner/date: [ ].
- **Evaluate promising models and errors:** inspect disagreement/reversal cases privately; avoid broad tuning on the same validation set. Owner/date: [ ].
- **Benchmark full decisions consistently:** same single GPU, batch one, timing boundaries, warm-up/synchronization and cache assumptions. Owner/date: [ ].
- **Finalize accuracy–latency comparison and poster:** freeze model choices before final test reporting; complete figures and rehearse for December 7. Owner/date: [ ].

## 10. Before turning this into the poster

- [ ] Everyone verifies their section and adds missing recent work; record individual ownership without guessing.
- [ ] Choose final title and one shared takeaway; preserve each person's scientific contribution.
- [ ] Select figures A–E as space permits, plus the shared pipeline/results view; remove redundant plots.
- [ ] Every displayed result names metric, partition and selected/final status; missing runs are labeled pending.
- [ ] Separate component timing from full-policy timing and supplied reference scores from reproduced baselines.
- [ ] Confirm permitted figures/exports and poster specifications.
- [ ] Condense into poster copy; retain this document as preparation/discussion notes.

## Source and revision notes

Primary requirements: instructor screenshot in this conversation. Task/data: [Dataset](DATASET.md), [Approaches](APPROACHES.md). Workstream evidence is linked in each section. Latest working-copy Group 2 notes take precedence over older README status. Fetched remote main at `e287d2b` supplies the additional Sabrina evidence; its documentation matches the reviewed `f0e2d37` content.

November 5 preparation milestone: replaced the initial Tran-focused outline with a team-wide fill-in document; populated recorded CNN, static-probe, attention and ViT evidence; reserved workstream space and specified graph slots. No new experiment or outcome was inferred. Latest fine-tuning outcome remains pending. Local main and remote main have diverged, and the working copy contains unrelated edits. Proposed publication: add only this document from a temporary checkout based on fetched remote main, preserving the current checkout. The user's immediate commit-message confirmation is still required.
