# Before-GRU evaluation milestone

## Scope and rationale — 2026-11-01

Sabrina requested implementation of the immediate frozen-encoder evaluation milestone, with priority on evaluation and diagnosing model limitations. This authorizes the local evaluation implementation and documentation; it does not establish team-wide protocol adoption or authorize publishing data. Existing raw loading, image preparation, and encoder interfaces are unchanged. No GRU or encoder fine-tuning is added.

`evaluation.py` trains small, L2-regularized logistic classifiers on fixed image features. This separates visual representation quality from sequence-model complexity. DINOv2 uses its current 384-value class-token summary; ResNet-18 uses its 512-value pooled summary. Both use the existing RGB, full-view, padded 518-square preparation, retain their pretrained weights, and encode the goal separately. This is a controlled shared-input comparison, not a claim that this resolution is optimal for either architecture.

**What is frozen and what learns:** ResNet-18 in `cnn.py` is unchanged by this evaluation: its pretrained weights remain frozen and BatchNorm stays in evaluation mode. Images are encoded under inference mode; detached NumPy features go to a separate scikit-learn logistic-regression classifier in `evaluation.py`. Only that classifier's coefficients/intercept and training-set scaling statistics are fitted. No trainable classification head is added inside the CNN, and no gradient flows back to it. DINOv2 weights also remain unchanged. High training balanced accuracy describes the probe's fit, not encoder fine-tuning. The probe is an evaluation baseline, not the eventual GRU policy.

**Current preference confirmed by Sabrina:** keep the CNN frozen for the coming experiments. Fine-tuning entries below are possible future research directions, not the next planned action or implemented changes. Continue frozen-feature evaluation first.

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
export HF_HOME="${PSCRATCH:?}/waldo-cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$PSCRATCH/waldo-cache/torch"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME"
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

### Compute allocation and cache troubleshooting

The current shared interactive queue required 32 CPU cores per GPU in Sabrina's run; a four-core request was rejected. Request matching resources in allocation and job step, using your available GPU account. Paste each command on one line; backslashes only work when immediately followed by a newline.

```sh
salloc --account=YOUR_GPU_ACCOUNT --constraint=gpu --qos=shared_interactive --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --time=01:00:00
# After allocation is granted:
srun --ntasks=1 --cpus-per-task=32 --gpus=1 --pty bash -i
```

Restore the module, scratch venv, DATA_DIR and cache exports in the resulting compute shell. CUDA availability and A100-SXM4-80GB visibility were confirmed in Sabrina's allocation. This alone does not verify pretrained inference.

The first evaluation attempt failed during DINOv2 weight download with `OSError: [Errno 524]` at `fcntl.flock`, before feature extraction. The default HF cache is usually in home; NERSC documents unsupported home-directory flock on compute nodes and recommends HF caches in scratch. Set the explicit scratch cache exports above before starting a fresh Python process. This remedy is proposed pending a successful retry; the traceback did not show the resolved cache location. Do not disable locks or delete existing caches. The unauthenticated HF warning is separate from the observed locking error. [NERSC cache guidance](https://docs.nersc.gov/machinelearning/launchers/), [home locking limitation](https://docs.nersc.gov/connect/vscode/), [HF cache variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

Outputs, all kept on Perlmutter:

- `summary.md`: comparison table with train/validation balanced accuracy and recalls, followed by automatic paired input-design score differences and investigation prompts. These prompts describe hypotheses, not established causes or statistical significance.
- `metrics.json`: full metrics, C candidates, episode intervals, feature counts/constant-feature counts, split/fold, software versions, code commit/dirty status, device and preprocessing. Extraction wall time is not online inference latency.
- `validation_predictions.private.json`: labels, case/episode/row references and advance probabilities aligned across experiments. Use these to review disagreements and high-confidence errors locally. Never commit or paste this file externally. Share only permitted aggregate findings after reviewing them.

`--benchmark` adds five warm-ups and 20 synchronized repetitions of one in-memory training case for each encoder. It times two images (goal + latest frame), one image per encoder call; reports median/p95 preparation, encoding, total and CUDA peak allocation. Encoding includes CPU-to-device transfer. It excludes disk loading, classifier, metadata, and history averaging. Both backbones can be resident, so total peak allocation includes both; it is not isolated model memory. This component timing is useful for initial comparison, not the final full-policy batch-one latency benchmark. Repeated same-case timing is a controlled microbenchmark, not workload-wide p95.

## Turn results into the next experiment

### First development results — owner-reported, 2026-11-01

Sabrina supplied the split-position-0 summary from the completed A100-SXM4-80GB run. These are validation-selected development results, not final held-out-test estimates. Exact code/version provenance and episode intervals have not yet been reviewed from the full report.

| Experiment | Train BA | Validation BA | Retract recall | Advance recall |
|---|---:|---:|---:|---:|
| majority | .500 | .500 | .000 | 1.000 |
| metadata | .580 | .547 | .529 | .565 |
| DINOv2 latest | .997 | .458 | .265 | .652 |
| DINOv2 goal only | .805 | .609 | .588 | .630 |
| DINOv2 latest + goal | .963 | .570 | .618 | .522 |
| DINOv2 latest + goal + metadata | 1.000 | .570 | .618 | .522 |
| DINOv2 mean history + goal | .940 | .540 | .559 | .522 |
| ResNet latest | .807 | .536 | .441 | .630 |
| ResNet goal only | .787 | .513 | .353 | .674 |
| ResNet latest + goal | 1.000 | .547 | .441 | .652 |
| ResNet latest + goal + metadata | .909 | .514 | .441 | .587 |
| ResNet mean history + goal | 1.000 | .540 | .471 | .609 |

**Interpretation:** large train/validation gaps show that these fitted classifiers generalize poorly on this partition. They do not establish that the pretrained encoders inherently cannot represent guidewires. DINOv2 goal-only has the largest observed validation BA (.609); adding latest-image features reduces it to .570. This motivates testing nuisance/reference shortcuts, weak linear goal/frame interaction, and overly flexible fits. Goal-only is not evidence of leakage or proof that the red marker alone explains the label: the whole reference includes anatomy and potentially wire context. Metadata adds no rounded validation BA to DINOv2; identical rounded recalls do not prove identical case predictions. Mean pooling does not improve either latest+goal baseline in this run; this does not test whether temporal order is useful.

**Component benchmark:** five warm-ups and 20 repetitions of the same training case, two images per case and one image per encoder call. DINOv2 median preparation/encoding/total: 29.17/17.66/46.83 ms; total p95 46.96 ms. ResNet: 29.23/4.65/33.87 ms; total p95 34.24 ms. DINOv2 encoding was about 3.8 times slower, while total measured time was about 1.38 times slower because preparation dominates ResNet. Both models were resident: baseline allocated bytes 141,832,192; measured peaks 170,201,088 (DINOv2) and 183,196,672 (ResNet). These are shared-residency allocator peaks, not individual model footprints. Neither the timing nor its p95 represents full-policy or workload-wide latency.

**Prioritized follow-up proposals:**

1. Inspect the existing episode-bootstrap intervals, confusion matrices and per-C scores before interpreting small differences. Review matched errors privately; retain the current run as baseline evidence.
2. Keep encoders frozen. Test a bounded stronger-regularization range (for example C=0.0001, 0.001, 0.01) and optionally train-only dimensionality reduction. Prefer grouped inner training splits for choosing expanded hyperparameters; repeating validation selection increases optimism. Both encoders have hundreds of features against 267 training cases; concatenation doubles image feature width. Full-data cached features are not persisted by the current runner, so a repeat currently re-extracts them.
3. Investigate goal/reference dependence and current-frame detail on Perlmutter: inspect representative training images after resize, then goal/frame difference or product fusion with regularization and a pooled-patch alternative. Change one factor at a time; optional wrong-goal/marker-removal controls require a separately specified implementation and must preserve traceability.
4. If frozen features remain weak after those checks, compare limited last-block fine-tuning at a small backbone learning rate against the regularized frozen baseline, using the same split/input/head and comparable search budget. Fine-tuning is not the first response to near-perfect training and weak validation.
5. Retain the static and mean-history baselines when adding a GRU; judge sequence modeling with matched inputs rather than attributing any gain automatically to temporal order.

No follow-up sweep, feature cache, alternate pooling/fusion, fine-tuning or GRU has been implemented by this results review. First read the existing uncertainty/error evidence; propose a bounded change next.

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

**Latest owner-reported run:** the split-position-0 CUDA evaluation with both encoders and `--benchmark` completed after the scratch-cache exports. The aggregate scores and component timings were supplied and reviewed above. Detailed episode intervals, per-C scores and matched private predictions remain to review; no statistically supported encoder ranking or final-test claim is established.

All 23 invented-data tests pass locally, including seven new evaluation checks. Sabrina also reported all 23 passed on Perlmutter in 1.780 seconds. The first real-data development classification and encoder component timing run is complete; fine-tuning and GRU remain unimplemented. Environment evidence and limits are recorded in [Group 2 progress](grp2/PROGRESS.md). CLI help and whitespace checks pass locally. Do not repeat the completed baseline merely to resume work.

## Teammate handoff — resume here

- Branch: `static-evaluation`. Evaluation implementation starts at `5ddb566`; this documentation handoff records the subsequent owner-run evidence. Exact Perlmutter run HEAD was not captured; inspect its `metrics.json` before attributing results to an exact revision.
- Completed: frozen-encoder probes, 23 passing synthetic tests locally and on Perlmutter, one split-0 A100 development run, aggregate score/timing review, scratch-cache recovery guidance. CNN/ViT weights were not updated; `cnn.py` and `vit.py` are unchanged by this milestone.
- Private evidence remains under each owner's scratch run directory, including `metrics.json` and `validation_predictions.private.json`. The reported run used relative scratch path `waldo-evaluation/dev-static-01`. These files are not in GitHub, and other accounts may not have access. Aggregate findings are recorded above.
- First next task: review existing episode uncertainty, confusion matrices and regularization candidates, then inspect matched errors on Perlmutter. Keep the encoders frozen. Agree on a bounded stronger-regularization/fusion experiment before adding complexity; use episode-grouped inner training validation for an expanded search where feasible.
- Still open: five-split final reporting policy, detailed spatial/reference diagnostics, full-policy latency, and GRU integration. No final-test predictions, new weight updates or follow-up experiment are included in this handoff.
- End sessions by leaving the `srun` shell and then the `salloc` shell to release the GPU. Closing a terminal is not a substitute for checking the job ended. Scratch results survive release of the allocation but remain subject to normal scratch retention policy.

## Method references

- [DINOv2 frozen linear evaluation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/linear.py): precedent for evaluating fixed representations with a linear classifier; our binary case/goal protocol differs.
- [scikit-learn evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html): balanced accuracy, class recall, ranking and probability metrics.
- [PyTorch CUDA timing notes](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution): GPU execution is asynchronous; synchronize timing boundaries.
