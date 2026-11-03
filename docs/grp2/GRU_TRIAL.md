# Frozen ViT–GRU trial and later experiment plan

## Status and purpose

**Implemented locally — 2026-11-02:** Tran requested setup of the agreed frozen ViT + one-layer, 64-unit GRU + goal + aligned depth/time debugging path. `vit_gru.py`, `vit_gru_debug.py` and synthetic tests now exist in the separate `tran/vit-gru-debug` worktree. The implementation uses the split-0 selection, uncentered training-only scales and bounded optimizer settings below. All 44 local synthetic tests passed, including 11 new GRU checks. No real eight-case run or full training has occurred; this is not a final architecture choice.

Read [GRU_DEBUG_HANDOFF.md](GRU_DEBUG_HANDOFF.md) for exact commands, test expectations, report fields, Sabrina's steps and later experiment boundaries. Code is local to the debug branch and shared documentation stays on main; remote publication remains pending. Check each worktree's Git log/status for exact revisions. No remote notebook or dataset was changed. Further manual masking is not required.

**Existing work reused:** Sabrina's `static-evaluation` at `070328f` contains frozen-feature logistic probes, including latest image + goal + all historical depth/time values. Its module and tests are copied unchanged into the debug worktree; [EVALUATION.md](../EVALUATION.md) records her reported split-0 GPU run. The tiny GRU runner is separate: scikit-learn fitting and validation-selected regularization do not train a PyTorch GRU. Sabrina's scaler centers/scales individual columns; the GRU divides by one training-derived scale per metadata channel without centering. Agree on the full comparison protocol before treating results as matched.

**Agreed sequencing:** first run the eight-case combined-input GRU check only. Start fresh for later full training/validation, compare 64 versus 128 there and prioritize the existing latest-image + goal + matching historical-metadata baseline before expanding comparisons. Full-training schedules, tuning budgets, caching and final evaluation are not authorized by the setup request.

## Key experimental rationale — 2026-11-02

**A working GRU alone cannot tell us whether its temporal complexity is useful.** A simple classifier on frozen latest-image and goal features plus the same historical metadata is the priority comparison: it asks whether the GRU's visual-history processing adds value beyond a simpler model with access to the same depth/time information. This is a stronger experimental plan than evaluating only the GRU, not evidence that either architecture will perform better. The comparison changes visual inputs and architecture together, so it does not perfectly isolate a single cause.

Keep the sequence explicit: tiny eight-case debugging check for the combined-input GRU; fresh initialization for full training and separate validation; comparison against the latest-image classifier with matching historical metadata; then expand to the four-model comparison below. A fixed validation protocol and controlled comparisons make results easier to interpret, but do not fully explain what the model learned. Tiny memorization does not establish temporal or goal usefulness.

Prefer a manageable one-layer GRU before adding architectural complexity. Tran accepted 64 hidden units for the initial debugging run and a later 64-versus-128 validation comparison; neither is a demonstrated winner. Patch features, goal-conditioned processing and last-block fine-tuning are later, targeted experiments requiring evidence and separate scope agreement. The bounded 64-unit setup is now implemented following Tran's setup request. Full-training settings, broader architecture ownership and any additional unpublished teammate work remain to coordinate.

## Comparison baselines and later alternatives

A baseline is a reference model used to assess whether a more complex model improves results. Sabrina's static evaluation already implements latest-image + goal classifiers with and without historical metadata. Reuse these for the later comparison after agreeing on a matched protocol; a new nonlinear classifier would be a separate proposed change. These probes are not the previously reported benchmark model or its reported approximately 82% result, whose attribution/setup remain unresolved in [APPROACHES.md](../APPROACHES.md). Coordinate with Group 1's current-input baseline before duplicating work. The following four-model design remains a later experiment; only the combined-input, 64-unit GRU debugging implementation is ready.

Proposed later comparisons, all using the same frozen image encoder and goal reference:

1. Latest history image + goal, without metadata; simple classifier.
2. Latest history image + goal + all six depth/time pairs, flattened into 12 metadata values; simple classifier.
3. Six-image GRU + goal, without metadata.
4. Six-image GRU + goal + aligned depth/time; the initial temporal proposal.

The latest depth and time values are both zero by construction, so using only those values provides no varying metadata. Model 2 intentionally retains historical metadata; describe it as latest-image input with historical metadata, not as having no history. Comparing 2 versus 4 holds available metadata constant while changing visual history and architecture. Comparing 1 versus 2 and 3 versus 4 tests metadata's contribution. These comparisons do not perfectly isolate every architectural cause and do not prove goal use.

**Stage these experiments:** first debug model 4 on eight training cases. Later, after agreeing on full training, compare models 2 and 4 on full training/validation partitions before adding models 1 and 3. Keep split, preparation, validation metrics, checkpoint-selection rule and tuning allowances consistent. Record parameter counts and training budgets; repeat seeds for promising comparisons before treating small differences as reliable. Full-training budgets, checkpoint rules and seeds remain to be agreed.

The eight-case run measures training loss/accuracy on those same eight examples; it does not use the validation partition or report held-out performance. It also checks finite gradients, optimizer updates and unchanged encoder weights. Later full training uses split 0's 267 training cases and evaluates the separate 80 validation cases without gradient/weight updates. Validation guides settings/checkpoint choice; the 69 test cases remain reserved. Do not add a four-model comparison or validation-based tuning to the initial debugging run.

**Critique and alternatives:** eight is a practical debugging size, not an established optimum. One optimizer step checks basic updates; one/two cases are cheaper but cover less variation; 16 add variation at more extraction cost; immediate full training can work but makes faults slower to diagnose. Tiny fitting is common debugging practice ([CS231n](https://cs231n.github.io/neural-networks-3/), [Karpathy](https://karpathy.github.io/2019/04/25/recipe/)); even perfect fitting cannot establish generalization, correct semantic alignment, or use of goal/history. It may exploit metadata or other shortcuts.

A one-layer GRU is a manageable starting point; six steps alone do not establish the best hidden size. Use 64 hidden units for the initial debugging check, then compare 64 and 128 during later full training under matching validation rules. Do not compare widths using eight-case memorization or launch a broad size sweep. No evidence yet establishes a need for LSTM or temporal attention. Patch features or goal-conditioned processing are targeted later experiments, but poor validation results alone cannot identify spatial information or goal use as the cause. Last-block fine-tuning adds trainable capacity and requires changing the inference-only encoder path; defer until a working frozen baseline and validation evidence justify a controlled comparison.

## Roadmap: what each run would tell us

1. **Build the trainable parts — complete locally.** The GRU/classifier, bounded entry point and synthetic checks for gradients, optimizer updates and frozen-encoder behavior are implemented. Existing loading and image preparation are unchanged.
2. **Debug learning on eight training cases.** Repeatedly show the same eight examples and their labels to the model. Record starting/final loss (how strongly predictions disagree with labels), training accuracy, finite gradients, optimizer updates and unchanged encoder weights. Falling loss and fitting these examples would show that this implementation can learn those examples. This is deliberately a memorization check, not a usable policy or an accuracy benchmark. Failure within the budget means investigate the implementation, inputs and optimization; it does not by itself reject DINOv2 or the architecture.
3. **Run the actual development experiment.** After agreeing on that later scope, create a fresh GRU/classifier and train on the full selected training partition. Evaluate on its separate validation partition to see whether learning transfers to unseen episodes. Save checkpoints on Perlmutter. Report validation accuracy, balanced accuracy, per-class recall and confusion counts; choose settings/checkpoints with validation only.
4. **Test whether history helps.** A later, matched comparison against latest-image-plus-goal inputs is needed to establish temporal usefulness. The eight-case check and a single trained model cannot establish this.
5. **Final evaluation.** Use reserved test data only after architecture and model-selection choices are fixed. Agree on the team's reporting protocol before this stage.

The working-code stage passed local synthetic verification. The real eight-case check and all later training/evaluation stages remain unrun for this GRU.

## Implemented debugging split and budget

The debugging runner uses **split position 0 only**, which assigns 267 cases to training, 80 to validation and 69 to test. These are cases containing a goal, six history frames and metadata, not single images. Recorded checks found no shared episodes between these partitions. Other supplied split positions are alternative assignments of the dataset, not extra training data. Do not combine their partitions. Whether to report all five positions remains open; if adopted, initialize a fresh model for each position and fit preprocessing scales only on that position's training partition.

**What “split 0” means:** `splits.json` contains five supplied assignment objects. Python counts list positions from zero, so position 0 means the first object, containing all three lists: training (learn weights), validation (compare settings/select checkpoints), and test (final evaluation). It is not the first case, a random seed, or the first training epoch. Position 0 is not necessarily the value of that object's uninspected `fold` field. The [dataset reference](../DATASET.md#preprocessing-and-splits) records the verified counts and remaining protocol questions.

**Why start there:** it provides one fixed development setting, already used for inspection, while keeping initial work bounded. There is no evidence that it is the best or most representative split. Choosing it before observing model scores avoids selecting whichever split makes a model look strongest. The other four objects reassign the same dataset; they do not provide four new datasets. Agree on their cross-split relationships and reporting protocol before expanding experiments. A case held out in one assignment may have another role in another, so independent model initialization alone does not protect a final evaluation from development decisions informed by overlapping cases.

The tiny check selects eight cases from split 0's training partition: four per label, selected deterministically with seed 0. Extract their frozen image features once and retain them in memory on Perlmutter; do not build a dataset-wide cache. Fit metadata scales using only split 0's training metadata: population standard deviation across the six steps of all training cases, separately for depth and time; use scale 1 for a constant channel, and do not subtract a mean.

Implemented optimization: full batch of eight, Adam with learning rate 1e-3, at most 300 updates or two minutes of head training, whichever comes first, with a ten-minute overall process limit including extraction. Record whether a limit was reached, initial/final loss and training accuracy, elapsed time and peak GPU memory. These are debugging limits, not a full-training schedule or a guaranteed fit threshold. Head time is checked between updates; the parent process enforces the overall worker deadline, allowing up to five seconds for termination cleanup. Do not increase the budget automatically if the check fails. CUDA availability alone does not establish that encoder extraction/backward execution works on the GPU; this bounded run would exercise that path.

## What the model remembers and how leakage is avoided

**Learned weights** persist while the same model object exists. Creating a fresh GRU/classifier resets them; restarting the notebook kernel removes them from memory. A checkpoint saves weights so a later process can load them. Rerunning a cell may either continue the existing model or recreate it, depending on the code; the entry point should make initialization explicit. Exact training resumption also requires optimizer state and relevant random-generator state.

**GRU hidden state** is temporary memory of the six frames within one case. Reset it between cases. That reset does not erase learned weights.

For the proposed full-training run, start a fresh GRU/classifier after the tiny check. Keep the same pretrained frozen ViT. The eight cases may remain in the full training partition: reusing training data is normal and is not leakage. Validation/test must not update weights or determine metadata scales. Validation can guide development; test results must not guide model changes. Never carry trained weights from one supplied split position into another.

## What is pretrained and what we design

Reuse the existing DINOv2 Small/14 image encoder, initially with fixed weights and evaluation behavior. Each image yields a 384-value summary. It does not receive depth, time, labels, or a goal instruction. Keep goal separate from the chronological history.

Use PyTorch's standard `torch.nn.GRU`, not a pretrained GRU or a custom implementation of its internal equations. Its weights start randomly and learn from the training labels. We choose its input size, hidden size, layer count and the connections around it. Updated initial configuration (64 units accepted for the debugging start; 128 retained for later comparison):

```python
nn.GRU(input_size=386, hidden_size=64, num_layers=1,
       batch_first=True, bidirectional=False)
```

For a batch of B cases:

1. History image features: `[B, 6, 384]`.
2. Append one scaled relative depth and one scaled time offset to each matching history step: `[B, 6, 386]`.
3. GRU reads all six steps oldest to newest; use its final hidden state `[B, 64]`. Reset hidden state for each independent case; never carry it between cases or episodes.
4. Concatenate that summary with the separate goal vector `[B, 384]`, producing `[B, 448]`.
5. Train a small classifier: `Linear(448, 64) -> ReLU -> Linear(64, 2)`. The two raw scores correspond to RETRACT=0 and ADVANCE=1. Use cross-entropy with integer labels; do not apply softmax before that loss. The later 128-unit variant instead produces a 512-value concatenation and uses `Linear(512,64)` as its first classifier layer.

This gives the goal a dedicated input role. It does not guarantee the classifier learns to use it. Goal fusion after the GRU is a simple baseline, not a claim that it best represents spatial relationships. No second transformer is added.

### Why consider 64 instead of 128 hidden units?

Hidden size is the number of learned memory values retained at each history step. Both sizes read the same six steps and the same 386 inputs per step. It is not the number of images or how far back the model sees. A larger state can represent more distinctions, but also introduces more adjustable weights.

For this one-layer, unidirectional GRU with default biases, input width `I=386` and hidden width `H`, the GRU has `3*H*(I+H+2)` parameters. Including the classifier `Linear(H+384,64) -> ReLU -> Linear(64,2)` gives:

- **H=128:** 198,144 GRU parameters + 32,962 classifier parameters = **231,106 trainable parameters**.
- **H=64:** 86,784 GRU parameters + 28,866 classifier parameters = **115,650 trainable parameters**. The concatenated classifier input becomes 448 values, rather than 512.

These counts exclude the frozen encoder. Parameter shapes are documented in [PyTorch's GRU reference](https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html). The count is a capacity consideration, not a proof that 128 will overfit or 64 will generalize. With 267 training cases, including related cases within episodes, 64 is a conservative starting suggestion. It may also discard useful information; 128 remains reasonable. Six steps alone does not determine the required width. If comparing widths, use the same split and comparable training/model-selection budgets, and decide using validation rather than training fit or test scores. Do not make a broad width sweep the next priority.

## Metadata and extraction boundaries

The implemented first version divides each channel by its population standard deviation across all training steps, with scale 1 for a constant channel and without shifting zero. Save those scales with model configuration and reuse unchanged for validation/test. Preserve signs, frame ordering and supplied zero endpoints. Do not normalize each case independently or fit scales on validation/test data.

`vit.py` is inference-only. For the frozen trial, features can be extracted without encoder gradients, then passed to the trainable head. PyTorch inference tensors must be converted to ordinary detached tensors outside inference mode before a trainable layer needs to save them for backward (for example an outside-context `.clone()`). Test that an actual backward/optimizer step works, not just output shapes. Keep encoder evaluation behavior fixed. Any later fine-tuning needs a differentiable encoder path; the existing inference-only wrapper cannot be used unchanged for that.

Dataset-wide extraction/caching is not authorized by this proposal. Discuss whether to extract selected batches on demand or use a bounded Perlmutter-only cache; avoid repeating expensive frozen extraction unnecessarily during a tiny training check. Actual images/features and caches must stay on Perlmutter.

## Implementation and verification sequence

1. **Complete locally:** separate head and bounded entry point on `tran/vit-gru-debug`; existing loader/preprocessor/encoder unchanged; 44 synthetic tests pass. Sabrina's published static work was inspected and reused. Coordinate any unpublished overlap before parallel extensions.
2. **Next:** review/share the code through an explicitly authorized handoff, preserve remote notebook edits before updating a Perlmutter clone, then follow [the test/run guide](GRU_DEBUG_HANDOFF.md). Code is not yet published; a pull alone cannot retrieve it.
3. **Pending on Perlmutter:** one eight-case learning check within the implemented caps. Keep private selection/scales and all derived artifacts there; record permitted aggregate evidence in Group 2 progress.
4. **Later, separately scoped:** fresh full training with validation-only selection; compare widths and Sabrina's matched-metadata latest-image baseline before expanding experiments. Agree on full-training budgets, checkpoint selection, seeds and latency measurement. Never tune on test results.

## Freeze versus fine-tune

Start frozen as a manageable baseline given the small number of training cases; it is not established as best. If training is strong but validation weak, investigate overfitting before increasing trainable encoder capacity. If both are weak, check labels, alignment, optimization and tiny-subset fitting first. Once the baseline works, a controlled last-block fine-tuning comparison may be appropriate. Decide from validation evidence, not the single-image masking distances. Test partitions remain reserved for final evaluation.

## Design review and proposed refinements

Recorded 2026-11-02 at Tran's request. Based on repository documentation/code and the primary references linked here; no real-data review or training was performed. Existing strengths are episode separation, frozen extraction, a distinct memorization check, training-only metadata scaling, and reserved test evaluation. The following refinements remain proposals.

### Representation and goal use

`vit.py` returns one whole-image summary per image. Small wire-tip motion or target-relative position may be weakly represented; this is a risk, not a conclusion from the earlier masking check. A GRU cannot recover information absent from its inputs. DINOv2 also provides patch features describing image regions ([official model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)). Keep the summary baseline first; consider spatial/region features if evidence suggests a representation limitation, before assuming a larger temporal model is necessary. Patch processing adds memory and implementation complexity.

The proposed GRU summarizes history before seeing the goal. This is simple but may compress away goal-relevant detail. A later alternative is to provide goal information at each history step so memory formation can depend on the target. A separately trained no-goal comparison can test predictive benefit; a gain does not prove correct target-relative reasoning. Goal conditioning, patch features, and last-block fine-tuning should be separate experiments, not simultaneous changes. There is no current evidence requiring an LSTM or temporal transformer instead of the GRU.

### Comparisons needed for an interpretable result

Move controlled comparisons into the first full development experiment, rather than leaving all of them until after a single-model result:

1. **Training-majority baseline:** always predict the class most frequent in the training partition; evaluate on validation.
2. **Metadata only:** a simple classifier using the six depths and times, without images.
3. **Latest image + goal + full metadata history:** a simple classifier on frozen image features and the same metadata available to the GRU.
4. **Six images + goal + metadata:** the proposed temporal model.

Compare on identical partitions with comparable training/selection budgets. Models 3 and 4 keep metadata fixed to assess the benefit of adding historical images with the temporal architecture; this does not isolate GRU ordering from every other architectural difference. A separate matched temporal control would be needed for that narrower claim. The dataset-wide ADVANCE proportion is 259/416, approximately 62.3%; it is not the measured split-0 validation baseline.

Depth history could predict continuation of operator movement without substantial image use. This may be useful for proxy-label prediction, but would not establish visual reasoning or optimal action selection. Verify on Perlmutter that inputs are available at decision time and do not include the interval used to construct the future-action label. Existing structural checks do not establish this semantic alignment; no leakage has been demonstrated.

### Full training and uncertainty

Keep the eight-case check simple, initially without added regularization. Its runtime/update cap is an operational boundary, not an architecture rejection criterion. Before full training, specify batch size, learning rate, epoch/update limit, modest weight decay, early stopping, and one checkpoint-selection metric. Weight decay discourages large weights; early stopping limits further fitting after validation stops improving. Exact settings remain open.

Propose balanced accuracy (mean of the two class recalls) for checkpoint selection, alongside ordinary accuracy, per-class recall and confusion counts. For 80 validation cases, one changed prediction moves ordinary accuracy by 1.25 percentage points. Related cases within episodes further limit how confidently small differences can be interpreted; preserve episode separation ([grouped-evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)). Three random initializations for the leading comparison is a proposed bounded budget. Seeds measure initialization variability, not uncertainty across unseen episodes; they do not replace an agreed split/reporting protocol. Avoid repeatedly expanding settings based on small validation gains.

### Runtime and recommended next action

Freezing saves training work but does not remove encoder computation during prediction. Measure batch-one end-to-end latency through preparation, seven-image encoding and the head, with warm-up and GPU synchronization where applicable. Separately report a realistic feature-reuse scenario if goal/history features can be retained between decisions. Head-only timing on precomputed features cannot establish the project's end-to-end latency objective. Actual reuse assumptions and caching scope need agreement; artifacts remain on Perlmutter.

The review originally proposed an early four-model development comparison. The subsequently accepted sequence above stages this work: implement/debug the 64-unit combined-input GRU first, then fresh full training and the matched-metadata latest-image comparison, then expand. The debugging implementation is now complete; later experiment budgets, caching expansion and final architecture selection remain open.
