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
Parameter counts are exact trainable counts (checked in `tests/test_train_cnngru.py`).

| Rung | Projection | GRU | MLP | Trainable params |
| --- | --- | --- | --- | --- |
| XS | 512 -> 32 | 32 x 1 layer | 64 -> 32 -> 2 | 26,114 |
| S | 512 -> 64 | 64 x 1 | 128 -> 64 -> 2 | 67,586 |
| M (current) | none | 128 x 1 | 640 -> 128 -> 2 | 330,626 |
| L (H3) | 512 -> 128 | 256 x 2, dropout 0.2 | 384 -> 256 -> LayerNorm -> ReLU -> 64 -> 2 | 875,074 |

Keep dropout 0.3, lr, weight decay and loss fixed across rungs so only capacity varies.

Predictions:
- H3 is supported if L beats M on paired validation AUC.
- The overfitting diagnosis is supported if validation AUC is flat or falls as capacity grows while every rung above XS still saturates training accuracy.
- If XS or S matches M, prefer the smaller rung as the new baseline.
- Phase 0 on split 0 found no signal above the shuffled-label null, so the likeliest outcome is that all rungs are indistinguishable; the ladder is still worth running because XS / S test whether a bottleneck alone recovers any signal.

Outcome (2026-11-05, partial grid, see [PROGRESS.md](PROGRESS.md#2026-11-05---h3-phase-1-results-s-adopted-as-the-only-head)):
no rung beat the shuffled-label null and all four were within noise of each other, so S was adopted as the fixed head and the `--capacity` option and other rungs were removed from the code.
The table above and the `phase1-*` result folders remain as the record; the commands that ran the grid used the removed `--capacity` flag and no longer work.

### Phase 2 - isolate H3 components (one factor at a time)

Start from the adopted head (rung S) and change one factor per run:

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

## H4 - regularization, and how much slack to allow

### Starting point

The adopted head (rung S, 67,586 parameters) reaches about 1.000 train bal-acc while last-20 validation AUC sits at 0.44-0.50, below the shuffled-label null on split 1 (Phase 1 in [PROGRESS.md](PROGRESS.md)).
Current regularizers: dropout 0.3 on the MLP input and hidden layer, AdamW weight decay 1e-2, gradient clipping 1.0, the 64-d projection bottleneck, and no early stopping (`--patience 0`).
The head memorizes shuffled labels as fast as real ones, so today it has unlimited slack: it can fit any labeling of 267 cases.

### Umbrella hypothesis

H4: the frozen features carry a weak ADVANCE/RETRACT signal that memorization currently drowns out; limiting how much the head can memorize will raise last-20 validation AUC above the shuffled-label null.

The alternative, which Phase 0 and Phase 1 favor, is that there is no signal in these features to recover.
Regularization cannot create signal, so the plan is built to reject H4 quickly and cheaply if that is the case.

### Defining slack

"Slack" here is how much the head is allowed to fit beyond what generalizes.
It is measured three ways, all from the last 20 epochs of existing `metrics.csv` / `summary.json` fields:

| Quantity | Definition | What it tells us |
| --- | --- | --- |
| Memorization slack | train bal-acc of the shuffled-label null run at the same setting | How much the head can fit pure noise; 1.0 today, 0.5 means none |
| Learnable signal | real-label train bal-acc minus null train bal-acc | Structure in the training set the head can fit that noise cannot explain |
| Generalization gap | real-label train bal-acc minus validation bal-acc | How much of the training fit fails to transfer; about 0.5 today |

The null run is the key instrument: a regularizer that leaves null train bal-acc near 1.0 has not removed any slack, whatever it does to validation.
This is the label-randomization test of Zhang et al. (2017), applied at every regularization strength.

### Slack budget (how much to allow)

Pre-registered target region, decided before any run:

- **Too much slack:** null train bal-acc above 0.75. The head still memorizes noise; the setting is too weak.
- **Target region:** null train bal-acc 0.55-0.70 and real train bal-acc 0.65-0.90. The head cannot memorize noise but can still fit some real structure.
- **Too little slack:** real train bal-acc below 0.65, or real minus null train bal-acc below 0.05. The head underfits, and validation cannot be expected to improve.

The upper bound of 0.90 for real train bal-acc is deliberate: with 267 cases and validation bal-acc around 0.55-0.60 at best, a 0.3+ gap means the extra fit is memorization.
Strength is chosen by where it lands in this region on the training side, not by peak validation scores, so the choice does not consume validation noise.
Within the region, pick the weakest setting: it removes noise memorization at the least cost to real fit.

### Regularizers and per-family hypotheses

Each family is swept on its own, starting from rung S with everything else at defaults, and every setting is run with real and shuffled labels.

| ID | Regularizer | Sweep (current value first) | Prediction if H4 holds | New code |
| --- | --- | --- | --- | --- |
| R1 | Weight decay (AdamW) | 1e-2, 1e-1, 3e-1, 1.0 | Null train bal-acc falls first; real val AUC rises once null is in the target region | none |
| R2 | Dropout (MLP input and hidden) | 0.3, 0.5, 0.7 | Same as R1; weaker effect because the GRU and projection are not dropped | none |
| R3 | Feature noise: Gaussian noise on the LayerNormed 512-d features, training only | 0, 0.25, 0.5, 1.0 (std, in LayerNorm units) | Acts like image augmentation in feature space; best candidate because it targets the 267-case shortage directly | `--feature-noise` |
| R4 | Feature dropout: element-wise dropout on the LayerNormed 512-d features, training only | 0, 0.2, 0.4 | Stops the head from keying on a few case-specific dimensions | `--feature-dropout` |
| R5 | Frame dropout: replace whole history steps with zeros, training only | 0, 0.17 (one frame), 0.33 | Tests whether the head memorizes a single frame; small or no effect if history carries no signal | `--frame-dropout` |
| R6 | Label smoothing | 0, 0.1, 0.2 | Lowers validation loss (less overconfidence) but leaves AUC unchanged, since it does not limit memorization | `--label-smoothing` |
| R7 | Epoch budget (early stopping without using validation) | read curves at epochs 10, 25, 50, 100, 200 | Cheapest check: if validation AUC is ever above null it is early, before memorization | none (existing `metrics.csv`) |

R6 is a control: it changes calibration, not capacity, so a validation-loss gain without an AUC gain is expected and should not count as success.
R7 needs no new runs; it is read from the per-epoch AUC curves of the R0 runs, averaged over (split, seed) at each fixed epoch, and the epoch count is fixed before reading validation.
Mixup in feature space and SWA/EMA weight averaging are left out of the first pass; add them only if R1-R4 show a regularizer moving validation AUC at all.

### Protocol

1. **R0 baseline:** rung S, real and null, splits 0-2 x seeds 0-2, 200 epochs, `--patience 0` (18 runs).
   The existing `phase1-s-*` folders are not reused: feature extraction no longer draws global random numbers, so seeded runs now differ slightly from Phase 1 (see Implementation).
2. **R1-R6 sweeps:** each setting x splits 0-2 x seeds 0-2 x {real, null}.
   The 14 non-default settings give 252 runs, each about 1-1.5 minutes, roughly 4-6 GPU hours; most of it is re-encoding images, so caching features per split first cuts it to about an hour.
3. **Pick per family:** the weakest setting whose null and real train bal-acc land in the target region (training-side choice, as above).
4. **Score per family:** paired by (split, seed), compare that setting with (a) R0 real and (b) its own null.
5. **Combine:** only families that pass step 4 are combined, at most two at a time, at their chosen settings.
6. **Confirm:** run the final configuration and R0 on splits 3-4 x seeds 0-2, real and null, which no earlier step looked at.

Splits 3-4 have not been used for any choice so far, so they guard against the winner's curse seen in `select_preprocessing.py` (0.597 inner CV fell to 0.506 on validation).
Their validation episodes may overlap those of splits 0-2 (cross-fold relationships are unverified in [DATASET.md](../DATASET.md)), so this is a fresh-partition check, not untouched data.
The test partition stays unused.

### Decision rule

Primary metric: last-20 validation AUC, paired by (split, seed).

A regularizer passes only if all three hold:

1. It beats its own shuffled-label null: mean paired real - null difference > 0.05 and > 2 x std of the differences.
2. It beats R0: mean paired difference > 0.03 and > 2 x std.
3. It stays inside the slack target region.

Margins: at 80 validation cases (34 RETRACT / 46 ADVANCE) the standard error of one AUC near 0.5 is about 0.066, so single-split differences under about 0.13 are noise.
Pairing over 9 (split, seed) pairs lowers that, but seeds within one split share a validation set, so the effective sample is closer to 3 splits than 9 runs.
Differences under 0.02 AUC count as ties; on a tie, keep the simpler or stronger regularizer.

### Stop rule

If no family passes the null comparison (rule 1) at any setting, H4 is rejected: the features carry no signal for the head to recover, and further regularization work is not justified.
Work then returns to the feature items at the top of this file (spatial pooling, `--remove-maxpool`, 224 input, layer4 fine-tune), with the best-passing slack setting (or R0) as the fixed head.

### Implementation

- `CNNGRUModel(feature_noise=, feature_dropout=, frame_dropout=)` applies the R3-R5 regularizers to the LayerNormed 512-d features, in training mode only, before the shared projection.
  Frame dropout zeros a whole history step (depth/offset stay); feature dropout is element-wise; feature noise is Gaussian in LayerNorm units.
  At 0 each is skipped and draws no random numbers; none adds parameters, so checkpoints keep their keys.
- `train_cnngru.py` gains `--feature-noise`, `--feature-dropout`, `--frame-dropout` and `--label-smoothing` (training loss only; logged losses stay plain class-weighted cross-entropy).
  The `Regularization:` line in `train.log` and `args` in `summary.json` record all six settings; non-default ones are tagged in the default run name (`cnngru-wd0.1-fn0.5-null-split0-seed0-<timestamp>`).
- `--feature-cache DIR` saves the frozen ResNet features per (data dir, split, partition, input size) on first use and reloads them afterwards; a file made with other settings is an error, and a directory inside the repo is refused because the features come from real images.
  The path is kept out of `summary.json`.
- Feature extraction now gives its DataLoader a private generator, so it draws no global random numbers and a run is identical with or without the cache.
  This changes the global RNG state at the start of training relative to Phase 0/1, so earlier seeded runs are not reproduced bit for bit.
- `summarize_runs.py`: `--min-effect` (rule 2 margin), `--at-epochs` (R7), and an H4 slack block for every `NAME` / `NAME:null` group pair with null and real last-N train bal-acc, learnable signal, gap, slack region and the rule-1 test (real - null last-N val AUC > max(2 x std, 0.05)).
- Tests: `RegularizerTests`, `FeatureCacheTests` and `RunNameTests` in `tests/test_train_cnngru.py`; slack, min-effect and fixed-epoch tests in `tests/test_summarize_runs.py`.

### Commands (Perlmutter)

`sweep_h4.py` runs the whole R0-R6 sweep (270 runs: 15 settings x splits 0-2 x seeds 0-2 x real/null) inside one short GPU allocation.
It fills the feature cache once per split, then starts `--workers` processes that share the GPU; each loads the cached features once and trains its share of the runs back to back.
Every run re-seeds from its own `--seed`, so results match standalone `train_cnngru.py` runs (`tests/test_sweep_h4.py` checks this), and run folders have the same layout.
It resumes: finished runs (with `summary.json`, written last) are skipped and half-written folders are rerun, so after a time-out rerun the same command.
At the end it writes `h4-summary.txt` (the `summarize_runs.py` output below) to the repo root; per-worker output is in `h4-sweep-logs/`.

From the repo root, inside a GPU allocation, with `DATA_DIR` and `TORCH_HOME` set:

```sh
python -m unittest discover -s tests -p 'test_*.py'
python sweep_h4.py --dry-run | head
python sweep_h4.py --feature-cache "$PSCRATCH/waldo-feature-cache" --workers 8 2>&1 | tee h4-sweep.log
```

`python sweep_h4.py --summarize-only` rewrites `h4-summary.txt` from existing folders; it runs:

```sh
RUN_GROUPS=()
for tag in r0 wd0.1 wd0.3 wd1 do0.5 do0.7 fn0.25 fn0.5 fn1 fd0.2 fd0.4 frd0.17 frd0.33 ls0.1 ls0.2; do
  RUN_GROUPS+=("${tag}=docs/grp1/results/h4-${tag}-split*" "${tag}:null=docs/grp1/results/h4-${tag}-null-*")
done
python summarize_runs.py --window 20 --min-effect 0.03 --at-epochs 10,25,50,100,200 "${RUN_GROUPS[@]}"
```

Read, in order: the H4 slack block (region and rule 1 per setting), then the paired block of each setting against `r0` (rule 2), then the R7 rows (`val auc @ epoch`) of `r0`.
