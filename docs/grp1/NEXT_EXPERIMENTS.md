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

## Bottom line

Treat ~0.61 val bal-acc as **weak visual features + an easy-to-overfit head**. Next experiments should change **what the CNN emits** (spatial pooling, resolution, light unfreeze) and judge them with a **simple probe** before another full GRU train.

Results and protocol details belong in [PROGRESS.md](PROGRESS.md); keep aggregate metrics only in shared docs; real images and features stay on Perlmutter.
