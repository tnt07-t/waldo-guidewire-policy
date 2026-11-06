# Group 1 — current training setup

The working configuration is frozen ResNet-18, shared LayerNorm and 512-to-64 projection, a one-layer GRU with hidden size 64, and a 128-to-64-to-2 MLP. The head has 67,586 trainable parameters. Depth/time normalization uses training statistics. The goal is a separate reference, not a history step.

Weight decay **1.0** is the chosen starting point. Gaussian feature noise remains available and defaults to **0.0**. MLP dropout is fixed at **0.3** in the training entry point, matching the chosen configuration. Feature dropout, frame dropout, label smoothing, and dropout sweeps have been removed.

This is a working baseline, not a claim of validated generalization. Change one factor at a time and compare the same split and seed. See [next experiments](NEXT_EXPERIMENTS.md).

## Run on Perlmutter

From the repository root in an allocated compute environment with the project dependencies and `DATA_DIR` configured:

```sh
python train_cnngru.py --data-dir "$DATA_DIR" --split-position 0 --seed 0 --epochs 200
```

Two stronger presets are available: `--config moderate` uses feature-noise std 1.0 and `--config strict` uses 2.0. Both keep weight decay 1.0 and use 50 epochs; all other training defaults match. The default `--config baseline` retains 200 epochs and noise off. Explicit `--epochs`, `--weight-decay`, and `--feature-noise` override preset values. Compare with `--config baseline --epochs 50` to hold training duration fixed. See [the comparison protocol](NEXT_EXPERIMENTS.md).

Learning rate defaults to 0.001 and batch size to 16. These presets are candidates for reducing overfitting; real-data effectiveness has not been established.

Training prints loss, accuracy, balanced accuracy, AUC and class recalls each epoch, then final and last-20 metrics plus the train/validation balanced-accuracy gap. It runs the requested epoch count and retains final weights in memory. It creates no run directory, log, CSV, JSON summary, plot, or checkpoint. Best-epoch selection and early stopping are removed. `--run-name`, `--patience`, `--dropout`, `--feature-dropout`, `--frame-dropout`, and `--label-smoothing` are no longer accepted.

Optional controls:

- `--shuffle-train-labels`: seeded training-label null control; validation labels stay real.
- `--summary-window N`: change the terminal summary window.
- `--feature-cache "$PSCRATCH/waldo-feature-cache"`: explicitly cache frozen features outside the repository. This is the only training disk output and must stay on Perlmutter.

In Python, `run(args, features=...)` returns the final model, epoch rows, final metrics, window metrics, and the generalization gap in memory. `echo=False` suppresses terminal output. The CLI does not persist trained weights for later inference.

Old Group 1 run artifacts, aggregate exports, sweep/summary tools, and detailed run-history notes have been removed from the working tree at the owner's request. Tracked historical versions remain in Git history. Other groups' experiments and shared dataset/evaluation documentation are separate.

## Local verification

Use synthetic inputs without accessing the private dataset or downloading model weights:

```sh
python -m unittest discover -s tests -p 'test_train_cnngru.py' -v
```

Real data, derived features, and dataset-dependent verification stay on Perlmutter. The trainer loads only training and validation partitions.
