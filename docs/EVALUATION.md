# Before-GRU evaluation milestone

## Scope and rationale — 2026-11-01

Sabrina requested implementation of the immediate frozen-encoder evaluation milestone, with priority on evaluation and diagnosing model limitations. This authorizes the local evaluation implementation and documentation; it does not establish team-wide protocol adoption or authorize publishing data. Existing raw loading, image preparation, and encoder interfaces are unchanged. No GRU or encoder fine-tuning is added.

`evaluation.py` trains small, L2-regularized logistic classifiers on fixed image features. This separates visual representation quality from sequence-model complexity. DINOv2 uses its current 384-value class-token summary; ResNet-18 uses its 512-value pooled summary. Both use the existing RGB, full-view, padded 518-square preparation, retain their pretrained weights, and encode the goal separately. This is a controlled shared-input comparison, not a claim that this resolution is optimal for either architecture.

## Experiments implemented

| Experiment | Information supplied | Diagnostic question |
|---|---|---|
| majority | Training majority class only | Does learning beat a trivial classifier? |
| metadata | Six depths and six offsets | Can prior movement/timing explain labels without images? |
| latest | Last history image only | What can the current image representation predict? |
| goal_only | Goal reference only | Is the reference strongly predictive without current wire state? |
| latest_goal | Last history image + separate goal | Does goal context add predictive value? |
| latest_goal_metadata | Latest + goal + full depth/time vectors | Do numerical history signals add value? |
| mean_goal | Mean of six history embeddings + goal | Does unordered visual history help before a GRU? |

The last five run for each selected encoder. Concatenation is the first fusion baseline. The classifier is linear, so it cannot explicitly model multiplicative spatial relations between goal and wire; poor results may reflect the head/fusion, not just the encoder. `goal_only` is a diagnostic: its reference may itself contain relevant anatomy/wire information, so strong results are not proof of leakage. The last relative depth and time are zero. Earlier metadata carries temporal information, and `latest_goal_metadata` is not a purely current-input baseline. Identifiers, episode names, and row indices never enter classifiers.

## Development protocol

- Explicitly choose a supplied `--split-position`; there is no implicit fold choice. Position 0 below is a development example, not an adopted five-fold reporting policy.
- Check row duplicates, cross-partition row/episode overlap, and two-class coverage in training/validation before image extraction. Test membership is inspected for separation only; no test images, features, predictions, or label-based metrics are used.
- Fit `StandardScaler` and classifier on training only. Use balanced class weights and a fixed probability threshold of 0.5. For each feature design, select C from 0.01, 0.1, 1 by validation balanced accuracy; exact ties prefer smaller C (stronger regularization). Convergence failures stop the run.
- Report train/validation accuracy, balanced accuracy, per-class recall, and confusion matrices (rows=true, columns=prediction; order RETRACT=0, ADVANCE=1). Supplement with ROC AUC, advance average precision, log loss and Brier score. Class-weighted probabilities should not be assumed calibrated; no calibration claims follow from these scores.
- Bootstrap whole validation episodes with a fixed seed for 95% balanced-accuracy intervals. Skip resamples lacking either class and report valid counts. These intervals are conditional on the chosen fit, not corrected for regularization/design selection. Overlapping intervals are not a formal paired significance test.
- The validation set is reused across C and representation comparisons: these are development results, not unbiased final performance. Avoid broad searches based on small gains. Agree on the five supplied splits' intended reporting protocol before final comparisons; do not pool overlapping folds as independent cases.

## Run on Perlmutter

Use an allocated compute node for extraction across train/validation, even with `--device cpu`. A GPU notebook kernel or terminal must use the environment containing torch, torchvision, timm, Pillow, NumPy and scikit-learn. `requirements-eval.txt` lists additional requirements; inspect an installation dry run in the existing scratch venv before installing there. Do not replace the provided PyTorch module or run the full extraction on login nodes.

First make this implementation available in the Perlmutter clone through the normal Git handoff. These files are currently local changes, not automatically transferred or pushed. Preserve remote notebook edits/outputs before pulling code.

```sh
# Existing Perlmutter environment; DATA_DIR must already point to your scratch copy.
python -m pip show scikit-learn numpy Pillow timm torch torchvision
python -m unittest discover -s tests -v

# Example explicit development split; use a fresh run directory each time.
python evaluation.py \
  --split-position 0 \
  --encoders dinov2 resnet18 \
  --device cuda \
  --benchmark \
  --output-dir "$PSCRATCH/waldo-evaluation/dev-static-01"
```

The runner requires data/output beneath personal `$PSCRATCH`, rejects existing output directories, and has no test-evaluation option. It holds train/validation features in memory for reuse across heads; it does not save a persistent feature cache or model checkpoint. Public pretrained weights may download; no image upload is performed.

Outputs, all kept on Perlmutter:

- `summary.md`: comparison table with train/validation balanced accuracy and recalls, followed by automatic paired input-design score differences and investigation prompts. These prompts describe hypotheses, not established causes or statistical significance.
- `metrics.json`: full metrics, C candidates, episode intervals, feature counts/constant-feature counts, split/fold, software versions, code commit/dirty status, device and preprocessing. Extraction wall time is not online inference latency.
- `validation_predictions.private.json`: labels, case/episode/row references and advance probabilities aligned across experiments. Use these to review disagreements and high-confidence errors locally. Never commit or paste this file externally. Share only permitted aggregate findings after reviewing them.

`--benchmark` adds five warm-ups and 20 synchronized repetitions of one in-memory training case for each encoder. It times two images (goal + latest frame), one image per encoder call; reports median/p95 preparation, encoding, total and CUDA peak allocation. Encoding includes CPU-to-device transfer. It excludes disk loading, classifier, metadata, and history averaging. Both backbones can be resident, so total peak allocation includes both; it is not isolated model memory. This component timing is useful for initial comparison, not the final full-policy batch-one latency benchmark. Repeated same-case timing is a controlled microbenchmark, not workload-wide p95.

## Turn results into the next experiment

| Observation to investigate | Possible explanation, not established cause | Next bounded experiment |
|---|---|---|
| High train BA, low validation BA | Few independent episodes; excessive feature/head capacity | Stronger regularization, smaller head, learning curves by training episodes; keep encoder frozen |
| Both train and validation weak | Domain mismatch, missing goal interaction, weak head, label ambiguity | Inspect train cases/resizing; compare small MLP or explicit goal/frame difference/product fusion before blaming backbone |
| Goal-only comparable to latest+goal | Reference/context shortcut or useful anatomy cues | Inspect matched errors; later remove marker or use wrong-goal negative controls on training/validation only, without changing labels |
| Metadata dominates images | Operator-action persistence or timing/depth shortcut | Compare with/without metadata and review reversals; retain metadata-only baseline when adding GRU |
| Mean history improves over latest | Multiple views help; order still untested | Compare ordered GRU against mean pooling with identical inputs |
| ViT weak on tiny wire/target cases | Resize/patchification or global summary loses local detail | Inspect prepared pixels on Perlmutter; compare class token vs patch mean/spatial pooling at fixed encoder; then a bounded supported-resolution trial |
| Frozen ViT features remain weak after fusion checks | Natural-image pretraining may not transfer | Add a trainable wrapper; unfreeze last 1–2 transformer blocks with smaller backbone LR; compare same split/head/budget to frozen baseline |
| Accuracy similar but ViT slower | Token count/resolution or architecture overhead | Compare a smaller candidate or supported lower resolution, checking wire visibility and class recalls again |

The current DINOv2 wrapper uses inference mode and cannot fine-tune as-is. A later trainable module must remove that constraint explicitly; changing `eval()` alone is insufficient. Fine-tuning requires fresh features each training step: frozen cached embeddings cannot update image weights. Resolution changes require updating the current fixed-518 encoder validation and respecting checkpoint positional-embedding handling. Do not silently change shared preprocessing.

Suggested error-review worksheet, private on Perlmutter: model disagreements, true/predicted class and confidence, visible wire tip/target, blur/occlusion, goal relationship, depth trend, history spacing, and episode grouping. Review training examples first; use validation errors sparingly to formulate a small number of hypotheses. Attention maps or embedding plots may aid inspection but are not evidence of causal use of the wire. Proxy action labels measure agreement with recorded behavior, not clinical optimality.

## Verification and remaining work

All 23 invented-data tests pass locally, including seven new evaluation checks; environment details and limits are recorded in [Group 2 progress](grp2/PROGRESS.md). CLI help and whitespace checks pass. No real-data classifier results, GPU timings, fine-tuning gains, or ViT limitations have been established by this new milestone yet. Next: run the command in the existing Perlmutter environment, review private artifacts, and record aggregate results and the next hypothesis here. Do not invent results or repeat completed basic data validation.

## Method references

- [DINOv2 frozen linear evaluation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/linear.py): precedent for evaluating fixed representations with a linear classifier; our binary case/goal protocol differs.
- [scikit-learn evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html): balanced accuracy, class recall, ranking and probability metrics.
- [PyTorch CUDA timing notes](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution): GPU execution is asynchronous; synchronize timing boundaries.
