# Group 1 — next experiments (CNN / features first)

Context: frozen ResNet-18 + LayerNorm + GRU + MLP overfits train while validation stays weak (~0.61 bal-acc, ~0.5 AUC on split 0). LR-only H2 did not fix it. Linear probes ([PROBE_LINEAR.md](PROBE_LINEAR.md)) show the bottleneck is largely **what the CNN emits**, with a usable but noisy **goal-only + remove-maxpool** linear baseline. The test partition must not be used for tuning.

**Working diagnosis:** weak / poorly pooled visual features + an easy-to-overfit temporal head. Goal features with a larger stem map (`remove_maxpool`) carry more signal than history concatenations in a linear probe; seed variance is high.

---

## Already tested (do not re-run without a reason)

| Experiment | Result (split 0) | Where |
| --- | --- | --- |
| CNN-GRU + LayerNorm reference (seed 0, lr 1e-3) | Best val bal ~0.608, AUC ~0.499; train memorizes | [PROGRESS.md](PROGRESS.md) |
| H2: same with lr 3e-4 | Val bal ~0.612; no clear gain | PROGRESS |
| H0 / H1 / H1b feature norm (seeds 0–2) | LayerNorm adopted; gain over H0 within noise | PROGRESS |
| Linear probe, maxpool **kept**, seeds 0–1 | Best mean: `latest_goal` val bal 0.561 / AUC 0.491 | [PROBE_LINEAR.md](PROBE_LINEAR.md) |
| Linear probe, maxpool **removed**, seeds 0–1 | Best mean: **`goal_only`** val bal **0.624** / AUC **0.607**; `mean_goal` collapsed | PROBE_LINEAR |
| Always-ADVANCE val baseline | acc 0.575, bal 0.500, AUC 0.500 | PROBE_LINEAR |

**Best implemented baseline:** `GoalOnlyLinearModel` / `probe_cnn_features.py --preset best` (goal_only + `remove_maxpool`). See running instructions in PROBE_LINEAR.md.

---

## High-priority experiments to run

Ordered by expected information gain given the probe findings. One change at a time; match split 0; prefer ≥2 seeds; report mean ± range, not the lucky seed alone.

### 1. Confirm the best probe (trust / hygiene)

- Replicate `--preset best` on **seed 2** (and optionally another split) before treating mean 0.62 as stable.
- Work the **implementation checks** in PROBE_LINEAR.md (freeze/eval, determinism, train≪val anomaly on nomaxpool goal_only seed 1, checkpoint selection).
- **Why first:** seed 1’s 0.70 AUC may be fragile; hygiene bugs would poison every later CNN change.

### 2. Wire `remove_maxpool` into the CNN-GRU policy (matched)

- Train `train_cnngru.py` with the same stem change (`FrameEncoder(remove_maxpool=True)`), split 0, seeds 0–1, vs the existing LayerNorm reference.
- **Why:** probe gains were concentrated in **goal** features without maxpool; the full policy still uses the default stem. Ask whether GRU+MLP benefits or just overfits harder.

### 3. Goal-only linear vs CNN-GRU (fair baseline)

- Freeze comparison: `--preset best` mean metrics vs LayerNorm CNN-GRU on the **same** seeds/split (already partly available). Optionally add a short train of `GoalOnlyLinearModel` end-to-end (images → logits) to confirm feature-cache training matches.
- **Why:** if goal-only linear ≈ GRU policy, temporal fusion is not earning its complexity yet.

### 4. Input size 224 (matched probe, then policy)

- `--preset best` and/or full 3-design probe with `--size 224` (still multiple of 14) vs 518.
- **Why:** ResNet-native scale; current 518 is ViT-coupled. Deferred until after nomaxpool; now the next prep ablation.

### 5. Spatial features beyond global pool

- Keep layer4 maps (`B×512×H'×W'`) or smarter pool (GeM / attention / tip–goal ROI); re-probe before a long GRU train.
- **Why:** `mean_goal` failed and global 512-d remains weak for “where is the wire vs target.” Nomaxpool only delays downsampling; it does not add localization structure.

### 6. Light layer4 unfreeze (task adaptation)

- Unfreeze **layer4 only**, small LR (`1e-5`–`5e-5`), rest frozen; start from **remove_maxpool** backbone; compare to frozen goal-only / GRU on the same split/seeds.
- **Why:** if frozen goal features are the best linear story, modest adaptation may move AUC without full finetune.

### 7. Diagnostics gate

- Occlusion / overlay on a few **training** cases with `remove_maxpool` on vs off — do sensitive patches sit near wire/target?
- **Why:** if sensitivity is nowhere near the wire, more head capacity will not invent localization. [CNN_DIAGNOSTICS.md](CNN_DIAGNOSTICS.md).

---

## Lower priority

- Another LR-only sweep on the old stem  
- Deeper GRU / wider MLP  
- Long memorization runs  
- Touching the test set  
- Declaring victory from a single seed’s `goal_only` run  

---

## Suggested near-term order

1. Hygiene + seed-2 `--preset best`  
2. CNN-GRU + `remove_maxpool` matched train  
3. Size 224 best-probe  
4. Spatial pool / layer4 maps  
5. Layer4 fine-tune if still flat  

## Bottom line

Treat ~0.61 CNN-GRU val bal-acc as **weak pooled features + overfit head**. The linear study elevates **goal-only + remove-maxpool** as the current feature baseline (~0.62 mean val bal, high seed variance). Next work should (a) stabilize that finding, (b) port the stem change into the policy, and (c) improve **spatial** representation—not grow the GRU.

Details and commands: [PROBE_LINEAR.md](PROBE_LINEAR.md). Aggregate metrics only in shared docs; real images/features stay on Perlmutter.
