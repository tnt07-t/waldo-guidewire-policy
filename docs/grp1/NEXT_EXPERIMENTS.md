# Group 1 — incremental experiments

Start from the current small CNN–GRU head with AdamW weight decay **1.0**, MLP dropout **0.3**, and feature noise **0.0**. Weight decay is a chosen working baseline, not an established improvement over shuffled-label controls.

Change one factor per comparison. Keep the split, seed, learning rate, batch size and epoch count the same. Training reports metrics to the terminal and does not save results or checkpoints. Current commands and options are in [PROGRESS.md](PROGRESS.md).

## Immediate comparison

Run the baseline once, then optionally enable Gaussian feature noise while keeping weight decay fixed:

```sh
python train_cnngru.py --data-dir "$DATA_DIR" --split-position 0 --seed 0 --epochs 200 --weight-decay 1.0 --feature-noise 0
python train_cnngru.py --data-dir "$DATA_DIR" --split-position 0 --seed 0 --epochs 200 --weight-decay 1.0 --feature-noise 0.25
```

The second command is a candidate experiment, not a recommended noise strength. Noise perturbs normalized history and goal features only during training. It is disabled during evaluation. No feature dropout, frame dropout, label smoothing, or automated regularizer sweep remains.

Compare the final-window validation AUC, balanced accuracy, both class recalls and training fit. Use `--shuffle-train-labels` for a matched null control when assessing whether a change learns useful associations. AUC and balanced accuracy near 0.5 indicate chance-level ranking or class-balanced classification. High training accuracy alone is insufficient.

Before accepting an improvement, repeat it across seeds and episode-based splits. Seeds share validation cases, so they are not independent datasets. Avoid choosing whichever epoch happens to have the largest validation score. Keep the test partition reserved until model choices are fixed.

## Subsequent steps

If a bounded feature-noise comparison does not help, return noise to zero. The next candidate changes should investigate retained spatial detail and explicit current-frame/goal relationships, first with a small regularized probe. Compare metadata-only, goal-only and current-image baselines before attributing gains to temporal history.

Inspect image visibility, goal alignment and ambiguous action labels privately on Perlmutter before enlarging the head. More capacity and more epochs are not substitutes for evidence that the inputs carry transferable signal.
