# Group 1 — next experiments (CNN / features first)

Context from owner-run split-0 training (see [PROGRESS.md](PROGRESS.md)): with frozen ResNet-18 + LayerNorm + GRU + MLP, the head can nearly memorize training (~99% train acc) while validation stays weak (best val bal-acc ~0.61, val AUC ~0.5, val loss rises). H2 (lr `3e-4` vs `1e-3`) did not fix generalization. More capacity and long memorization runs are lower priority. The test partition must not be used for tuning.

**Working diagnosis:** the bottleneck is likely **what the frozen pooled CNN emits**, not optimizer LR or a bigger temporal head.

## Highest-value next tries (CNN / features)

### 1. Confirm the features, not the head

Run a **linear probe on frozen features only** (no GRU) on the same split — e.g. last history + goal, or mean history + goal.

| Outcome | Meaning |
| --- | --- |
| Probe also ~0.55–0.60 bal-acc / ~0.5 AUC | Representation problem → keep focusing on CNN/features |
| Probe clearly better than GRU policy | Sequence fusion may be hurting → simplify temporal side later; still improve CNN |

### 2. Don’t throw away space (main architectural bet)

Global average pooling → one 512-d vector is weak for “where is the wire vs the target.” Try, in order of cost:

1. **`--remove-maxpool`** (already in `FrameEncoder`) and compare with the same linear probe / short train.
2. **Keep spatial maps** (e.g. layer4 `B×512×H'×W'`) and pool smarter: GeM, attention pool, or a small tip/goal ROI — even a crude center-crop vs full-image ablation.
3. Longer-term: light **spatial head** (goal–history similarity) instead of only concatenating two global vectors.

### 3. Unfreeze a little of the CNN (task adaptation)

Still feature-focused, not “bigger GRU”:

- Unfreeze **layer4 only** with a small LR (e.g. `1e-5`–`5e-5`); freeze the rest; keep strong weight decay / dropout.
- Compare to fully frozen on the **same** split and seed.

If validation AUC does not move, prep or label noise may dominate capacity.

### 4. Prep ablations that change what ResNet sees

Matched runs, **one change at a time**:

- **224 vs 518** input (ResNet-native scale vs current ViT-coupled 518).
- Optional: mild emphasis on the red goal marker only if diagnostics show it is washed out — do not hack labels.

### 5. Goal vs history encoding balance

Shared backbone is fine; test whether **goal 512-d swamps** the GRU 128-d summary in the MLP:

- Project goal **512 → 128** before concat (matched ablation).
- Or tiny separate **adapters** (Linear) on goal vs history features without unfreezing ResNet.

### 6. Use diagnostics as a gate

Before another long train: occlusion / overlay checks on a few training cases — do max-sensitivity patches sit near wire/target? If not, more epochs will not invent localization. See [CNN_DIAGNOSTICS.md](CNN_DIAGNOSTICS.md).

## Lower priority (given current results)

- Another LR-only sweep  
- Deeper GRU / wider MLP  
- Long memorization runs  
- Touching the test set  

## Suggested run order

1. Frozen **linear probe** (latest+goal, mean+goal) on split 0.  
2. Same probe with **`remove_maxpool`** and/or **224 input**.  
3. If still flat → **layer4 fine-tune** with tiny LR.  
4. If probe improves but full GRU policy does not → simplify temporal fusion; if neither improves → revisit spatial features / prep / labels.

## H3 - capacity, reframed as a capacity ladder

### Original proposal

- GRU: a Linear 512 -> 128 projection before the GRU, hidden size 256, 2 layers with dropout 0.2 between them.
- MLP: 256 -> LayerNorm -> ReLU -> 64 -> 2 hidden stack, with dropout lowered to about 0.1-0.2.

### Why it is not run as stated

More capacity fixes underfitting, and the evidence points to overfitting:

| Evidence | Reading |
| --- | --- |
| LayerNorm reference reaches ~99% train accuracy while validation loss rises from the first epochs | Already overfitting |
| Best-epoch validation AUC 0.46-0.59 across H1/H1b/H2 runs | Near chance ranking |
| ResNet latest + goal linear probe 0.547 validation bal-acc ([EVALUATION.md](../EVALUATION.md)) | A linear head does about as well |
| Metadata-only probe 0.547 | Images add little so far |
| GRU + LayerNorm mean 0.586 over seeds 0-2 | Within the ~0.055 binomial standard error at n = 80 |

The current head has 330,626 trainable parameters for 267 training cases.
The proposal has about 970k (projection ~66k, 2-layer GRU ~693k, MLP ~214k), roughly 3x more, and it also lowers dropout.
Both changes push toward more overfitting.
A 2-layer GRU over only 6 time steps adds little temporal modeling for its cost.
Dimension mismatch: with GRU hidden 256 and a 512-d goal the MLP input is 768, not 640; reaching a small input requires projecting the goal too.

The part worth keeping is the 512 -> 128 projection.
It is a bottleneck that removes parameters (GRU input 514 -> 130) and acts as a regularizer.
Apply one shared projection to every history frame and the goal, after the shared LayerNorm, so both stay in one 128-d space and the goal no longer outweighs the GRU summary in the MLP (see item 5 above).

### Terminology: activation vs loss

ReLU and LeakyReLU are activation functions, not loss functions.
The loss is class-weighted cross-entropy, which fits a binary task with mild imbalance (train weights RETRACT/ADVANCE 1.376 / 0.785); focal loss is not motivated at this imbalance.
The loss change worth testing is label smoothing (0.1), because rising validation loss with saturated training accuracy signals overconfidence.
LeakyReLU only helps when ReLU units die; a 1-2 layer MLP after LayerNorm rarely has that problem, and the GRU's sigmoid/tanh gates are fixed either way.
Measure the fraction of MLP hidden units that never activate on the training set before attributing any change to the activation.

### Phase 0 - make the measurement trustworthy (run first)

- **Shuffled-label null:** current model, training labels randomly permuted (fixed permutation seed), same protocol and epoch count. This gives the "best validation bal-acc" that pure noise reaches when taking the max over ~200 noisy epochs. If it lands near 0.58, results so far are mostly selection noise.
- **Metrics:** primary validation AUC (threshold-free); also report mean validation bal-acc and AUC over the last 20 epochs, and best-epoch values only as secondary.
- **Replication:** splits 0-2 (ideally all 5) x seeds 0-2, `--patience 0`, 200 epochs; compare configurations paired by (split, seed).

Implementation:
- `train_cnngru.py --shuffle-train-labels` permutes the training labels across cases with a generator seeded by `--seed` (class counts and class weights unchanged; validation labels stay real) and logs a `NULL RUN` line.
- Every run now logs and stores in `summary.json` the final epoch (`final_epoch`) and the mean/std over the last `--summary-window` epochs (`last_window`, default 20) for train bal-acc, validation bal-acc, AUC and loss.
- `summarize_runs.py NAME=PATTERN ...` tabulates run folders (best, final and last-N metrics recomputed from `metrics.csv`, so older runs work too), prints mean +- std per group, and compares every later group with the first, paired by (split, seed), against the decision rule below.

Run on Perlmutter (from the repo root, inside a GPU allocation, with `DATA_DIR` set):

```sh
python -m unittest discover -s tests -p 'test_train_cnngru.py' -v
python -m unittest discover -s tests -p 'test_summarize_runs.py' -v
```

```sh
for split in 0 1 2; do
  for seed in 0 1 2; do
    python train_cnngru.py --data-dir "$DATA_DIR" --split-position "$split" --seed "$seed" \
      --epochs 200 --patience 0 --run-name "phase0-real-split${split}-seed${seed}"
    python train_cnngru.py --data-dir "$DATA_DIR" --split-position "$split" --seed "$seed" \
      --epochs 200 --patience 0 --shuffle-train-labels \
      --run-name "phase0-null-split${split}-seed${seed}"
  done
done
```

```sh
python summarize_runs.py --window 20 \
  null='docs/grp1/results/phase0-null-*' \
  real='docs/grp1/results/phase0-real-*'
```

Expected: null runs still reach high training bal-acc (memorizing noise) with last-N validation AUC near 0.5.
Reading the paired `real - null` block:
- `mean > 2*std: yes` on last-N validation AUC means the real model carries signal above noise; Phase 1 proceeds against that baseline.
- `no` means current results are indistinguishable from noise; prioritize the feature work above (items 1-4) over any capacity change.
- Also compare null best-epoch bal-acc with the ~0.61 reported earlier: that gap is the selection optimism of best-epoch reporting.

### Phase 1 - capacity ladder (tests the H3 claim directly)

Each rung scales width together; the projection is shared by history frames and goal.
Parameter counts are estimates.

| Rung | Projection | GRU | MLP | Approx. params |
| --- | --- | --- | --- | --- |
| XS | 512 -> 32 | 32 x 1 layer | 64 -> 32 -> 2 | 25k |
| S | 512 -> 64 | 64 x 1 | 128 -> 64 -> 2 | 67k |
| M (current) | none | 128 x 1 | 640 -> 128 -> 2 | 331k |
| L (H3) | 512 -> 128 | 256 x 2, dropout 0.2 | 384 -> 256 -> LayerNorm -> ReLU -> 64 -> 2 | ~900k |

Keep dropout 0.3, lr, weight decay and loss fixed across rungs so only capacity varies.

Predictions:
- H3 is supported if L beats M on paired validation AUC.
- The overfitting diagnosis is supported if validation AUC is flat or falls as capacity grows while every rung above XS still saturates training accuracy.
- If XS or S matches M, prefer the smaller rung as the new baseline.

### Phase 2 - isolate H3 components (one factor at a time)

Start from the best Phase 1 rung and change one factor per run:

- projection on / off, and shared with goal vs history only
- GRU 1 vs 2 layers at fixed hidden size
- GRU hidden 128 vs 256
- MLP 1 vs 2 hidden layers
- dropout 0.1 / 0.3 / 0.5 (prediction: lower dropout is not better)

### Phase 3 - activation and loss

- MLP activation ReLU vs LeakyReLU(0.01) vs GELU, logging the dead-unit fraction (prediction: no measurable difference).
- Weighted cross-entropy vs weighted cross-entropy + label smoothing 0.1.

### Decision rule

A change counts only if its mean paired gain in validation AUC exceeds about twice the spread across (split, seed) pairs and it holds on most pairs.
With 80 validation cases, differences below about 0.05 bal-acc are not detectable.
Training runs on cached features are cheap; the risk is false winners from many configurations scored on one validation set, which is why Phase 0 comes first.
The test partition stays untouched.

Expected outcome: the shared bottleneck projection and label smoothing may help; extra GRU layers and lower dropout should not.

## Bottom line

Treat ~0.61 val bal-acc as **weak visual features + an easy-to-overfit head**. Next experiments should change **what the CNN emits** (spatial pooling, resolution, light unfreeze) and judge them with a **simple probe** before another full GRU train.

Results and protocol details belong in [PROGRESS.md](PROGRESS.md); keep aggregate metrics only in shared docs; real images and features stay on Perlmutter.
