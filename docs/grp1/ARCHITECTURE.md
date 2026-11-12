# CNN and GRU architecture: critique and improvement plan

Scope: the Group 1 CNN and CNN-GRU code path only.
Files covered: `cnn.py`, `gru.py`, `mlp_cnn.py`, `model_cnngru.py`, `train_cnngru.py`, `infer_cnngru.py`, the CNN part of `cnn_diagnostics.py`, and the CNN branch of `evaluation.py`.
Attention, ViT and the temporal-head experiments are out of scope.

Vocabulary follows the codebase-design glossary: a module is deep when a small interface hides a lot of behaviour, and a seam is the place where a module's interface lives.

## Status

The file relocation is done: everything now lives in the `guidewire` package (see the README layout).
The sections below still use the original flat file names, because they describe the code before the move.
Changes that edit code, such as merging the head, typed configs, `Features`, `metrics.py` and checkpoints, are not done yet.

## What the code does today

```
dataset.GuidewireDataset
  -> preprocessing (DINOv2Preprocessor or WirePreprocessor)
  -> cnn.FrameEncoder (frozen ResNet-18, 512-d)          [features cached by train_cnngru]
  -> model_cnngru.CNNGRUModel
       feature_norm (LayerNorm 512) -> projection (512 -> 64)
       gru.CNNGRUEncoder (GRU 64 + 2 metadata channels) -> summary [B, 64]
       mlp_cnn.GoalClassifier (concat summary + goal [128] -> 64 -> 2)
  -> probabilities, or logits for cross-entropy
```

The design itself is sound for a small-data study.
The encoder is frozen, features are encoded once, and only a 67,586-parameter head trains.
The friction is in how responsibilities are cut across files, not in the model.

## Strengths worth keeping

- **Frozen encoder with a safe `train()` override.**
  `FrameEncoder.train` forces the backbone back to eval mode, so BatchNorm statistics cannot drift.
  This invariant is easy to break and it is held in one place.
- **Feature caching with a metadata check.**
  `load_cached_features` refuses a cache whose metadata differs, and writes are atomic.
  This prevents silently training on stale features.
- **Two forward paths that are documented as equivalent.**
  `forward` (images) and `forward_features` (cached vectors) share the same head.
- **Normalization statistics travel with the checkpoint.**
  `meta_mean` and `meta_std` are buffers, so inference reuses training statistics.
- **Experiment hygiene.**
  Exclusive-create CSVs, seeded runs, a shuffled-label null control, and a test partition that is never loaded.

## Critique

### 1. The module names do not match what the modules contain

- `gru.py` defines `CNNGRUEncoder`, which owns a `FrameEncoder` (`self.cnn`) and a GRU.
  During training the CNN is bypassed, because `train_cnngru` extracts features first and calls `forward_features`.
  So the "encoder" that trains is only a GRU plus metadata normalization, and the real CNN is a passenger.
- `CNNGRUModel.forward_features` reaches into `self.encoder.summarize` while `feature_norm` and `projection` live on the outer model.
  The history path is therefore split across two owners: the projection is outside `encoder`, the GRU is inside it.
- `mlp_cnn.py` holds `GoalClassifier` and `encode_goal`.
  `GoalClassifier` is the entire classification head, not a goal-specific piece, and `encode_goal` is a validated wrapper around one call to the CNN.
- The module docstring in `mlp_cnn.py` describes the whole CNN-GRU data flow, which is documentation that belongs with `model_cnngru.py`.

Deletion test on `gru.py`: deleting it moves about 20 lines into `CNNGRUModel`.
Complexity does not spread across callers, so it is closer to a pass-through than a deep module.
Deletion test on `encode_goal`: the validation (shape, finiteness) reappears in `extract_features` and `cnn_diagnostics.encode_image`, which already do their own copies of it.

### 2. Shape and size constants are repeated, not owned

- `512` appears as `FrameEncoder.feature_dim`, `GOAL_FEATURE_DIM` in `mlp_cnn.py`, a literal in the `FrameEncoder` docstring, and a literal check in `cnn_diagnostics.encode_image`.
- The sequence length `6` is baked into `FrameEncoder` docstrings, `infer_cnngru.collate` comments and the smoke-test asserts in `cnn.py:main`.
- The parameter count `67,586` is written into the `model_cnngru.py` docstring and will go stale on any head change.
- Head widths (`PROJECTION_DIM`, `GRU_HIDDEN`, `MLP_HIDDEN`) are module constants, so a width sweep needs source edits rather than a configuration.

### 3. Temporal coupling: the model is not valid until `set_metadata_stats` is called

- A freshly built `CNNGRUModel` normalizes metadata with identity statistics.
  It runs, produces probabilities, and is silently wrong until `model.encoder.set_metadata_stats(...)` is called with training data.
- Only `train_cnngru.run` calls it.
  `infer_cnngru.py` builds `CNNGRUModel()` and depends on a checkpoint to overwrite the buffers.
  An untrained run (no `--checkpoint`) uses identity statistics without a warning.
- Statistics fitting is a method on the model, so the model has to be built before data is loaded, and the trainer has to know to call it.

### 4. Training and inference are two programs with no shared contract

- `train_cnngru.py` states that no checkpoints are saved, while `infer_cnngru.py` loads a `--checkpoint`.
  There is currently no producer for the file the consumer expects.
- `infer_cnngru.py` hard-codes `DINOv2Preprocessor(size=...)` and has its own `--size` flag with default 518.
  `train_cnngru.py` also hard-codes `DINOv2Preprocessor`, so the wire preprocessor that `cnn.py` already supports cannot be trained or run for inference.
- `infer_cnngru.collate` re-implements what `PreparedCases.__getitem__` plus `DataLoader` collation do in training.
  The two can drift in dtype, order or key names.
- Inference constructs `CNNGRUModel()` with the legacy dropout default of 0.3.
  Dropout is inactive in eval mode, so this is harmless, but the architecture is not recorded with the checkpoint, so nothing proves the constructor arguments match training.

### 5. `train_cnngru.py` is a 558-line module with a shallow interface

It contains, in one file:

- CSV logging (`csv_logger`),
- a dataset adapter (`PreparedCases`),
- feature extraction and an on-disk cache (`extract_features`, `cache_*`, `load_*`),
- class weights and a shuffled-label control,
- hand-written metrics (`roc_auc`, `binary_metrics`, `window_summary`),
- the training loop (`train_model`),
- 80 lines of argparse (`parse_args`),
- the run and sweep orchestration (`run`, `run_sweep`).

Specific problems:

- `run(args, features=None)` takes a whole `argparse.Namespace`.
  The interface is every flag, so tests must build namespaces, and callers other than the CLI have no clean way to call it.
- `cache_metadata` hard-codes `"preprocessor": "dinov2"` and a fixed encoder string.
  The key says nothing about the wire preprocessor or `remove_maxpool`, so adding either option would reuse a cache that does not match.
  `CACHE_VERSION` is a manual "bump it when you remember" switch.
- Class weights are computed twice: once in `run` for logging and once in `train_model` for the loss.
- Per-epoch metrics, the log line and the CSV row are each assembled from the same dictionaries in different places (`train_model`, `format_epoch`, `run`).
- The hand-written metrics use key names (`bal_acc`, `recall_retract`, `auc`) that differ from the sklearn-style names in `evaluation.py` (`balanced_accuracy`, `retract_recall`, `roc_auc`), so `compare_runs.py` cannot read training output.

### 6. Frozen-CNN knowledge is duplicated across the diagnostics and evaluation paths

- `cnn_diagnostics.encode_image`, `mlp_cnn.encode_goal` and `train_cnngru.extract_features` each run the CNN, check the shape and check `isfinite`.
- `evaluation.py` constructs `FrameEncoder().eval()` itself for the ResNet probe.
- `cnn_diagnostics.py` is 822 lines mixing pure functions (masks, distances, occlusion math) with PIL file output and a CLI.
  The pure parts are testable, but they sit beside I/O, so the file is hard to navigate.

### 7. Stale and misleading documentation inside the code

- The `cnn.py` docstring still lists "Integrate GRU/LSTM, aligned depth/time inputs, and goal fusion" as remaining work.
  That has been done.
- The `FrameEncoder` docstring refers to the DINOv2 checkpoint label as something to avoid confusing, which is a symptom of the preprocessing module being named after one model while serving two encoders.
- Comments such as "retains legacy dropout 0.3" and the preset docs in `train_cnngru.py` explain history that belongs in `PROGRESS.md`.

### 8. Testability

- `test_cnn_gru_train.py` (494 lines) is the largest CNN-side test file, which reflects how much logic is behind one module and a namespace.
- There is no test for `infer_cnngru.py`.
- Nothing asserts that `forward` on images equals `forward_features` on the cached features, even though the docstring claims they match.
  That is the property that makes caching safe.

## Proposed architecture

The aim is fewer, deeper modules with explicit seams, and one place that owns each constant.
No behaviour needs to change: results must stay numerically identical, and that should be verified on Perlmutter before and after.

### Target layout

```
cnn_gru/
  features.py     extract / cache frozen features          (deep)
  head.py         CNNGRUHead: norm, projection, GRU, MLP   (deep)
  config.py       HeadConfig, TrainConfig, presets
  metrics.py      one metrics schema
  training.py     fit(head, data, TrainConfig) -> RunResult
  checkpoint.py   save / load head + config + stats
  cli/            thin argparse entry points (train, infer)
cnn.py            FrameEncoder (unchanged, frozen encoder)
```

### 1. One head module that owns the whole trainable part

Merge `gru.py`, the head parts of `model_cnngru.py` and `GoalClassifier` into one `CNNGRUHead`.
Its interface is `forward(features: Features, *, return_logits=False)`.
Inside it owns LayerNorm, projection, noise, GRU, metadata normalization and the MLP.

- It does not own a `FrameEncoder`, because the head never needs one at training time.
- A thin `CNNGRUModel` (encoder plus head) can still exist for end-to-end inference from images.
- Benefit, locality: the history path lives in one file, and the name now says what trains.
- Benefit, leverage: tests construct a head from synthetic `[B, 6, 512]` tensors with no ResNet.

### 2. Make the head valid by construction

Replace `set_metadata_stats` after construction with a statistics object passed in.
For example `HeadConfig` plus `MetadataStats.fit(train_depth, train_offsets)`, then `CNNGRUHead(config, stats)`.
Without stats the constructor raises, so an unfitted head can never run.
Checkpoints store the config and stats, so loading rebuilds the exact architecture.

### 3. A `Features` value type and a single extraction module

Define one `Features` dataclass: `history [N,6,512]`, `goal [N,512]`, `depth [N,6]`, `offsets [N,6]`, `label [N]`, plus the provenance needed for caching.
`features.extract(dataset, encoder, preprocessor)` returns it.
The on-disk cache key is derived from the encoder settings (including `remove_maxpool`) and the preprocessor's own record, instead of a hand-written dictionary and a manual version number.

- Training, diagnostics, evaluation and inference all go through this module.
- Delete `PreparedCases`, `extract_features`, `encode_goal` and the validation copies.
- The shape and finiteness checks live once, at this seam.

### 4. Typed configuration instead of `argparse.Namespace`

`TrainConfig` (epochs, lr, weight decay, dropout, noise, batch size, seed, grad clip, window) and `HeadConfig` are plain dataclasses.
`REGULARIZATION_CONFIGS` becomes a dictionary of `TrainConfig` presets.
`training.fit(head, data, config)` returns a `RunResult`; the CLI only translates flags into configs.

- Sweeps become a loop over configs, callable from a notebook or a test without a namespace.
- Constants like `512`, `6` and the head widths come from `HeadConfig` and `FrameEncoder.feature_dim`, with one definition each.

### 5. One metrics module

Move `binary_metrics`, the AUC code and `window_summary` into `metrics.py` with the same key names `evaluation.py` uses.
Keep a numpy and a torch entry point behind the same function, or convert to numpy once.
`compare_runs.py` then reads training and evaluation output alike.

### 6. Close the train-to-inference loop

- `training.fit` returns the head, and `checkpoint.save` writes weights, config, stats and preprocessor record together.
  This needs an explicit decision, because the current docs say no checkpoints are saved; it can be opt-in with a path outside the repository.
- `infer` loads that file, builds the preprocessor from the recorded settings and runs the same `Features` pipeline.
- Delete `infer_cnngru.collate` and its separate `--size` flag.

### 7. Preprocessor selection through one seam

Both `FrameEncoder` entry points and the trainer should call `build_preprocessor(args)` like `cnn.py` already does.
The cache key then records which preprocessor was used.
This makes the wire preprocessor available to CNN-GRU training without code edits, even though it is not currently adopted.

### 8. Split `cnn_diagnostics.py`

Move the pure functions (masks, distances, occlusion math, temporal statistics) into `cnn_diagnostics/core.py`, keep file output in `report.py`, and keep a short `__main__`.
The tests already target the pure functions, so this is mostly moving code.

### 9. Documentation placement

- Keep module docstrings to what the module is and its interface.
- Move history ("legacy dropout", preset provenance, remaining work) to `PROGRESS.md` and `NEXT_EXPERIMENTS.md`.
- Update the stale "Remaining work" block in `cnn.py` now.

## Tests to add before refactoring

These lock behaviour so the refactor can be checked:

1. `forward(images)` equals `forward_features(cached features)` on a small synthetic model (eval mode, same seed).
2. A fixed-seed training run on synthetic features produces identical per-epoch rows before and after the change.
3. A checkpoint round trip (save, load, predict) matches the in-memory model.
4. A cache key changes when `remove_maxpool` or the preprocessor changes.
5. An unfitted head raises instead of running with identity statistics.

## Suggested order

1. Add the tests above against the current code, and update the stale `cnn.py` docstring.
2. Introduce `Features` and `features.py`, switch `train_cnngru` and `infer_cnngru` to it.
3. Introduce `metrics.py` and point training and `compare_runs.py` at it.
4. Merge the head into `CNNGRUHead`, with stats required at construction.
5. Introduce typed configs, then shrink `train_cnngru.py` to a CLI.
6. Add checkpoint save and load, and rewrite `infer_cnngru.py` on top of it.
7. Split `cnn_diagnostics.py`.

Steps 2 and 3 pay off even if the rest is never done.
Rerun one seed-0 split-0 baseline run on Perlmutter after each step and compare against the recorded metrics before continuing.
