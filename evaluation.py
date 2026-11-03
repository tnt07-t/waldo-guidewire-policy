"""Frozen-feature development probes. Real data/artifacts stay on Perlmutter.

No test-set evaluation, fine-tuning, or GRU is performed. Core functions accept
invented arrays so evaluation logic can be tested without images or weights.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, log_loss,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODES = ("latest", "goal_only", "latest_goal", "latest_goal_metadata", "mean_goal")
# Channel-ablation controls: zeroed enhanced channel -> result name prefix.
ABLATIONS = {1: "ablate_tophat", 2: "ablate_difference"}


def check_partitions(datasets):
    """Check supplied row membership and episode separation before encoding."""
    rows, episodes = {}, {}
    for name, dataset in datasets.items():
        indices = list(dataset.indices)
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate rows in " + name)
        rows[name] = set(indices)
        episodes[name] = {dataset.records[i]["episode"] for i in indices}
    names = list(datasets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if rows[left] & rows[right] or episodes[left] & episodes[right]:
                raise ValueError("Row or episode leakage between partitions")
    for name in ("train", "validation"):
        labels = [datasets[name].records[i]["label"] for i in datasets[name].indices]
        if set(labels) != {0, 1}:
            raise ValueError(name + " must contain both action classes")


def design_matrix(features, mode):
    """Keep sequence metadata distinct from the current-frame visual baseline."""
    if mode == "metadata":
        result = features["metadata"]
    elif mode == "latest":
        result = features["history"][:, -1]
    elif mode == "goal_only":
        result = features["goal"]
    elif mode in ("latest_goal", "latest_goal_metadata", "mean_goal"):
        frame = (features["history"].mean(axis=1) if mode == "mean_goal"
                 else features["history"][:, -1])
        parts = [frame, features["goal"]]
        if mode == "latest_goal_metadata":
            parts.append(features["metadata"])
        result = np.concatenate(parts, axis=1)
    else:
        raise ValueError("Unknown feature mode")
    result = np.asarray(result, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("Expected finite feature matrix")
    return result


def metrics(y, probability):
    """Fixed 0.5 threshold; rows=true, columns=predicted, order retract/advance."""
    y, probability = np.asarray(y), np.asarray(probability)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Both classes required for comparable metrics")
    if probability.shape != y.shape or not np.isfinite(probability).all():
        raise ValueError("Invalid probability shape/values")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be in [0,1]")
    predicted = (probability >= 0.5).astype(int)
    recalls = recall_score(y, predicted, labels=[0, 1], average=None, zero_division=0)
    return {
        "n": len(y), "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(recalls.mean()),
        "retract_recall": float(recalls[0]), "advance_recall": float(recalls[1]),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "roc_auc": float(roc_auc_score(y, probability)),
        "advance_average_precision": float(average_precision_score(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probability)),
    }


def episode_interval(y, probability, episodes, *, repeats=1000, seed=0):
    """Resample whole episodes; interval is conditional on the fitted model."""
    y, probability, episodes = map(np.asarray, (y, probability, episodes))
    unique = np.unique(episodes)
    if len(unique) < 2:
        return {"interval": None, "valid_resamples": 0}
    groups = [np.flatnonzero(episodes == episode) for episode in unique]
    rng, scores = np.random.default_rng(seed), []
    for _ in range(repeats):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        target = y[indices]
        if len(np.unique(target)) != 2:
            continue
        correct = (probability[indices] >= 0.5) == target
        scores.append(float(np.mean([correct[target == label].mean() for label in (0, 1)])))
    return {"interval": np.quantile(scores, [0.025, 0.975]).tolist() if scores else None,
            "valid_resamples": len(scores), "requested_resamples": repeats,
            "method": "episode bootstrap, conditional on selected fit; not selection-adjusted"}


def fit_probe(train_x, train_y, val_x, val_y, *, strengths=(0.01, 0.1, 1.0)):
    """Fit scaling on train only; validation selects C, ties prefer stronger L2."""
    trials, best, best_score = [], None, -1.0
    for strength in sorted(strengths):
        model = fit_checked(strength, train_x, train_y)
        score = metrics(val_y, model.predict_proba(val_x)[:, 1])["balanced_accuracy"]
        trials.append({"C": strength, "validation_balanced_accuracy": score})
        if score > best_score:
            best, best_score = model, score
    return best, trials


def fit_checked(strength, x, y):
    """Fit the standard probe with one C; nonconvergence is an error."""
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=strength, class_weight="balanced", solver="lbfgs", max_iter=3000))
    # A nonconverged candidate must not silently enter the comparison.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    return model


def label_permutations(y, permutations, seed):
    """Seeded permutations of training labels; class counts are unchanged."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(np.asarray(y)) for _ in range(permutations)]


def shuffled_label_control(train_x, train_y, val_x, val_y, *, permutations=20, seed=0):
    """Validation BA of probes refit on permuted training labels.

    C is selected exactly as for the real probe, so this is the score a probe
    reaches when features carry no label information.
    """
    scores = []
    for permuted in label_permutations(train_y, permutations, seed):
        model, _ = fit_probe(train_x, permuted, val_x, val_y)
        scores.append(metrics(val_y, model.predict_proba(val_x)[:, 1])["balanced_accuracy"])
    return {"permutations": permutations, "seed": seed,
            "validation_balanced_accuracy": scores,
            "mean": float(np.mean(scores)), "max": float(np.max(scores)),
            "quantile_95": float(np.quantile(scores, 0.95))}


def grouped_fold_index(episodes, n_splits=5):
    """Assign each training case to one of ``n_splits`` episode-grouped folds."""
    from sklearn.model_selection import GroupKFold

    episodes = np.asarray(episodes)
    fold = np.empty(len(episodes), dtype=int)
    splitter = GroupKFold(n_splits=n_splits)
    for k, (_, scored) in enumerate(splitter.split(np.zeros(len(episodes)), groups=episodes)):
        fold[scored] = k
    return fold


def inner_cv(features, mode, *, n_splits=5, strengths=(0.01, 0.1, 1.0)):
    """Episode-grouped CV inside the training partition; validation is never used.

    Each rotation scores one fold with a probe fit on the others. Its C is
    chosen by an inner grouped CV over those other folds only.
    """
    x, y = design_matrix(features, mode), np.asarray(features["y"])
    fold = grouped_fold_index(features["episode"], n_splits)
    scores, chosen = [], []
    for k in range(n_splits):
        others = [j for j in range(n_splits) if j != k]
        best, best_score = None, -1.0
        for strength in sorted(strengths):
            inner = []
            for j in others:
                fit = (fold != k) & (fold != j)
                model = fit_checked(strength, x[fit], y[fit])
                inner.append(metrics(y[fold == j], model.predict_proba(x[fold == j])[:, 1])
                             ["balanced_accuracy"])
            # Ties keep the smaller C (stronger regularization), as in fit_probe.
            if np.mean(inner) > best_score:
                best, best_score = strength, float(np.mean(inner))
        model = fit_checked(best, x[fold != k], y[fold != k])
        scores.append(metrics(y[fold == k], model.predict_proba(x[fold == k])[:, 1])
                      ["balanced_accuracy"])
        chosen.append(best)
    return {"fold_balanced_accuracy": scores, "selected_C": chosen,
            "mean": float(np.mean(scores)), "std": float(np.std(scores, ddof=1)),
            "folds": n_splits, "mode": mode, "std_note": "sample std (ddof=1) over folds"}


def all_splits_summary(results_by_split):
    """Per-split validation BA and their plain mean; cases are never pooled."""
    names = list(next(iter(results_by_split.values())))
    summary = {}
    for name in names:
        scores = {str(position): results[name]["validation"]["balanced_accuracy"]
                  for position, results in results_by_split.items()}
        summary[name] = {"validation_balanced_accuracy": scores,
                         "mean": float(np.mean(list(scores.values())))}
    return summary


def control_lines(controls, results):
    """One summary.md line per control, next to the normal result."""
    lines = []
    for name, control in controls.get("shuffled_labels", {}).items():
        lines.append("Shuffled labels, {}: real BA {:.3f}; permuted mean {:.3f}, "
                     "max {:.3f} over {} permutations.".format(
                         name, results[name]["validation"]["balanced_accuracy"],
                         control["mean"], control["max"], control["permutations"]))
    for variant, control in controls.items():
        if variant == "shuffled_labels":
            continue
        for name, result in control["results"].items():
            lines.append("{}, {}: BA {:.3f} vs normal {:.3f}.".format(
                variant, name, result["validation"]["balanced_accuracy"],
                results[name]["validation"]["balanced_accuracy"]))
    return lines


def collect(dataset, prepare, encoders):
    """Extract each case once per encoder; no persistent feature cache."""
    import torch

    data = {name: {"goal": [], "history": []} for name in encoders}
    shared = {"metadata": [], "y": [], "episode": [], "case_id": [], "row_index": []}
    for index in range(len(dataset)):
        case = dataset[index]
        prepared = prepare.prepare_case(case)
        for name, encoder in encoders.items():
            with torch.inference_mode():
                if name == "dinov2":
                    result = encoder.encode(prepared["goal"], prepared["history"])
                    goal, history = result["goal_features"], result["history_features"]
                else:
                    device = next(encoder.parameters()).device
                    goal = encoder(prepared["goal"].unsqueeze(0).to(device))[0]
                    # One image per call for comparable extraction memory bounds.
                    history = torch.cat([encoder(frame.unsqueeze(0).to(device))
                                         for frame in prepared["history"]])
            data[name]["goal"].append(goal.cpu().numpy())
            data[name]["history"].append(history.cpu().numpy())
        shared["metadata"].append(case["depth_relative_mm"] + case["history_offsets_s"])
        for key, source in (("y", "label"), ("episode", "episode"),
                            ("case_id", "case_id"), ("row_index", "row_index")):
            shared[key].append(case[source])
    shared = {key: np.asarray(value) for key, value in shared.items()}
    for name in data:
        data[name] = {key: np.asarray(value) for key, value in data[name].items()}
        data[name].update(shared)
    return shared, data


def static_images(prepare, case):
    """Goal and latest-frame tensors [2,3,H,W] plus how many images were prepared."""
    import torch
    from preprocessing import WirePreprocessor

    if isinstance(prepare, WirePreprocessor):
        # Wire crops need goal_xy, and the enhanced difference channel needs
        # all six history frames, so the whole case is prepared.
        prepared = prepare.prepare_case(case)
        return torch.stack([prepared["goal"], prepared["history"][-1]]), 7
    # Only prepare the two images consumed by this static policy.
    return torch.stack([prepare.prepare_image(case["goal"])["tensor"],
                        prepare.prepare_image(case["history"][-1])["tensor"]]), 2


def preparation_record(prepare, case):
    """Preparation settings for metrics.json, without per-image geometry."""
    from preprocessing import CHECKPOINT, DINOv2Preprocessor, MEAN, STD

    if isinstance(prepare, DINOv2Preprocessor):
        return {"size": prepare.size, "mean": MEAN, "std": STD,
                "resize": "proportional bicubic + mean padding", "dinov2_checkpoint": CHECKPOINT,
                "resnet_weights": "ResNet18_Weights.DEFAULT; maxpool retained"}
    # The preprocessor's own record from one case, without its geometry.
    record = prepare.prepare_case(case)["preparation"]
    preparation = {key: value for key, value in record.items() if key not in ("goal", "history")}
    preparation["resnet_weights"] = "ResNet18_Weights.DEFAULT; maxpool retained"
    return preparation


def benchmark_encoders(case, prepare, encoders, device, *, warmup=5, repeats=20):
    """Batch-one, current-frame + goal encoding; classifier/disk IO excluded."""
    import torch

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    _, prepared_count = static_images(prepare, case)
    timings = {}
    for name, encoder in encoders.items():
        def static_run():
            images, _ = static_images(prepare, case)
            sync()
            boundary = time.perf_counter()
            with torch.inference_mode():
                if name == "dinov2":
                    encoder._encode_images(images, image_chunk_size=1)
                else:
                    for image in images:
                        encoder(image.unsqueeze(0).to(device))
            sync()
            return boundary
        for _ in range(warmup):
            static_run()
        sync()
        baseline_memory = torch.cuda.memory_allocated() if device == "cuda" else None
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        preprocessing, encoding, total = [], [], []
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            boundary = static_run()
            end = time.perf_counter()
            preprocessing.append((boundary - start) * 1000)
            encoding.append((end - boundary) * 1000)
            total.append((end - start) * 1000)
        timings[name] = {
            "warmup": warmup, "repeats": repeats, "case_batch_size": 1,
            "images_per_case": 2, "prepared_images_per_case": prepared_count,
            "image_chunk_size": 1,
            "scope": "Repeated same in-memory training case; goal+latest encoding only. Includes CPU preparation and H2D. Excludes disk IO, classifier, and metadata. Both backbones may be resident.",
            "milliseconds": {key: {"median": float(np.median(value)),
                                   "p95": float(np.percentile(value, 95))}
                             for key, value in (("preprocessing", preprocessing),
                                                ("encoding", encoding), ("total", total))},
            "resident_cuda_bytes": baseline_memory,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        }
    return timings


def evaluate(train, validation, train_features, val_features, *, repeats=1000):
    results, predictions = {}, {}

    def record(name, train_p, val_p, **extra):
        results[name] = {"train": metrics(train["y"], train_p),
                         "validation": metrics(validation["y"], val_p),
                         "validation_balanced_accuracy_95pct": episode_interval(
                             validation["y"], val_p, validation["episode"], repeats=repeats),
                         **extra}
        predictions[name] = val_p.tolist()

    # Majority is chosen from training labels only; ties choose retract.
    majority = int(np.mean(train["y"]) > 0.5)
    record("majority", np.full(len(train["y"]), majority),
           np.full(len(validation["y"]), majority), selected_class=majority)
    specifications = [("metadata", train, validation, "metadata")]
    for name in train_features:
        specifications.extend((name + ":" + mode, train_features[name], val_features[name], mode)
                              for mode in MODES)
    for name, train_data, val_data, mode in specifications:
        train_x, val_x = design_matrix(train_data, mode), design_matrix(val_data, mode)
        model, trials = fit_probe(train_x, train["y"], val_x, validation["y"])
        record(name, model.predict_proba(train_x)[:, 1], model.predict_proba(val_x)[:, 1],
               selected_C=float(model[-1].C), candidates=trials,
               feature_count=int(train_x.shape[1]),
               constant_train_feature_count=int(np.sum(np.std(train_x, axis=0) < 1e-12)))
    return results, predictions


def diagnostic_notes(results):
    """Suggest hypotheses from comparisons, never assert causes/significance."""
    notes = ["These are development hypotheses; validation differences are not proof of improvement."]
    for name in results:
        if not name.endswith(":latest_goal"):
            continue
        encoder = name.split(":")[0]
        base = results[name]
        score = base["validation"]["balanced_accuracy"]
        gap = base["train"]["balanced_accuracy"] - score
        notes.append("{}: latest+goal train-minus-validation BA gap = {:+.3f}. "
                     "A large positive gap motivates stronger regularization and episode learning curves.".format(encoder, gap))
        for suffix, meaning in (
            ("latest", "Goal-context contribution: a weaker latest-only result motivates preserving goal input."),
            ("goal_only", "Reference shortcut check: comparable goal-only performance warrants private matched-error review."),
            ("latest_goal_metadata", "Metadata contribution: gains warrant checking movement persistence and reversal errors."),
            ("mean_goal", "Unordered history contribution: gains motivate comparing a later GRU against mean pooling.")):
            delta = results[encoder + ":" + suffix]["validation"]["balanced_accuracy"] - score
            notes.append("{} {} minus latest+goal BA = {:+.3f}. {}".format(encoder, suffix, delta, meaning))
        delta = score - results["metadata"]["validation"]["balanced_accuracy"]
        notes.append("{} latest+goal minus metadata-only BA = {:+.3f}. "
                     "If image features are weak, inspect prepared wire/target detail and fusion before fine-tuning.".format(encoder, delta))
    return notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    parser.add_argument("--split-position", type=int,
                        help="Required unless --all-splits")
    parser.add_argument("--output-dir", type=Path, required=True, help="New private directory under PSCRATCH")
    parser.add_argument("--encoders", nargs="+", choices=("dinov2", "resnet18"),
                        default=["dinov2", "resnet18"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--benchmark", action="store_true", help="Also time static encoder components on one training case")
    controls = parser.add_argument_group("controls (opt-in; reported next to the normal result)")
    controls.add_argument("--all-splits", action="store_true",
                          help="Run on split positions 0-4 and report each plus their mean")
    controls.add_argument("--shuffled-labels", action="store_true",
                          help="Refit latest+goal probes on permuted training labels")
    controls.add_argument("--permutations", type=int, default=20)
    controls.add_argument("--random-center", action="store_true",
                          help="Also crop around a seeded random point far from the goal (wire)")
    controls.add_argument("--ablate-channels", type=int, nargs="+", choices=sorted(ABLATIONS),
                          default=[], help="Also zero these enhanced channels (1 top-hat, 2 difference)")
    controls.add_argument("--control-seed", type=int, default=0)
    from preprocessing import add_preprocessor_arguments, build_preprocessor
    add_preprocessor_arguments(parser)
    args = parser.parse_args()
    try:
        prepare = build_preprocessor(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.preprocessor != "dinov2" and "dinov2" in args.encoders:
        parser.error("--encoders dinov2 needs --preprocessor dinov2; use --encoders resnet18")
    if args.context_size is not None:
        parser.error("--context-size is not used by the static probe")
    if (args.split_position is None) == (not args.all_splits):
        parser.error("Give exactly one of --split-position or --all-splits")
    if args.all_splits and (args.shuffled_labels or args.random_center or args.ablate_channels):
        parser.error("Run controls on one --split-position, not with --all-splits")
    if args.random_center and args.preprocessor != "wire":
        parser.error("--random-center needs --preprocessor wire")
    if args.ablate_channels and args.channels != "enhanced":
        parser.error("--ablate-channels needs --preprocessor wire --channels enhanced")
    if args.permutations < 1:
        parser.error("--permutations must be positive")
    scratch = os.environ.get("PSCRATCH")
    if not scratch or not args.data_dir:
        parser.error("Run on Perlmutter with PSCRATCH and DATA_DIR set")
    root = Path(scratch).resolve()
    for path in (Path(args.data_dir).resolve(), args.output_dir.resolve()):
        if root not in path.parents:
            parser.error("Data and output must be inside personal PSCRATCH")
    if args.bootstrap_repeats < 100:
        parser.error("Use at least 100 bootstrap repeats")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new run directory")

    import torch
    from dataset import GuidewireDataset

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use an allocated GPU node or explicitly choose cpu")
    positions = list(range(5)) if args.all_splits else [args.split_position]
    split_datasets = {}
    for position in positions:
        split_datasets[position] = {
            partition: GuidewireDataset(args.data_dir, split_position=position, partition=partition)
            for partition in ("train", "validation", "test")}
        check_partitions(split_datasets[position])
    encoders = {}
    if "dinov2" in args.encoders:
        from vit import DINOv2Encoder
        encoders["dinov2"] = DINOv2Encoder(device=args.device)
    if "resnet18" in args.encoders:
        from cnn import FrameEncoder
        encoders["resnet18"] = FrameEncoder().to(args.device).eval()
    first_case = split_datasets[positions[0]]["train"][0]
    preparation = preparation_record(prepare, first_case)
    timings = benchmark_encoders(first_case, prepare, encoders, args.device) if args.benchmark else None

    def extract_and_evaluate(datasets, variant_prepare):
        start = time.perf_counter()
        train, train_features = collect(datasets["train"], variant_prepare, encoders)
        validation, val_features = collect(datasets["validation"], variant_prepare, encoders)
        if args.device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        results, predictions = evaluate(train, validation, train_features, val_features,
                                        repeats=args.bootstrap_repeats)
        return {"train": train, "validation": validation, "train_features": train_features,
                "val_features": val_features, "results": results,
                "predictions": predictions, "seconds": seconds}

    runs = {position: extract_and_evaluate(split_datasets[position], prepare)
            for position in positions}

    control_results = {}
    if not args.all_splits:
        run, datasets = runs[positions[0]], split_datasets[positions[0]]
        if args.shuffled_labels:
            control_results["shuffled_labels"] = {
                name + ":latest_goal": shuffled_label_control(
                    design_matrix(run["train_features"][name], "latest_goal"), run["train"]["y"],
                    design_matrix(run["val_features"][name], "latest_goal"), run["validation"]["y"],
                    permutations=args.permutations, seed=args.control_seed)
                for name in encoders}
        variants = {}
        if args.random_center:
            variants["random_center"] = prepare.with_options(random_center_seed=args.control_seed)
        for channel in args.ablate_channels:
            variants[ABLATIONS[channel]] = prepare.with_options(zero_channels=(channel,))
        for variant, variant_prepare in variants.items():
            variant_run = extract_and_evaluate(datasets, variant_prepare)
            control_results[variant] = {
                "preparation": preparation_record(variant_prepare, first_case),
                # Majority and metadata ignore images, so only image probes are kept.
                "results": {name: result for name, result in variant_run["results"].items()
                            if ":" in name}}

    def split_report(position):
        run, datasets = runs[position], split_datasets[position]
        return {"split_position": position, "fold": datasets["train"].fold,
                "partition_counts": {name: len(data) for name, data in datasets.items()},
                "feature_extraction_wall_seconds": run["seconds"],
                "diagnostic_hypotheses": diagnostic_notes(run["results"]),
                "results": run["results"]}

    def private_predictions(position):
        run = runs[position]
        private = {key: run["validation"][key].tolist()
                   for key in ("case_id", "episode", "row_index", "y")}
        private["advance_probabilities"] = run["predictions"]
        return private

    versions = {}
    for package in ("numpy", "scikit-learn", "torch", "torchvision", "timm", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "git_commit": git.stdout.strip(), "git_dirty": bool(dirty.stdout.strip()),
              "python": platform.python_version(), "versions": versions,
              "device": args.device, "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
              "threshold": 0.5, "class_weight": "balanced", "encoders": list(encoders),
              "preparation": preparation,
              "encoder_component_benchmark": timings,
              "timing_note": "Full train/validation extraction duration, NOT online inference latency",
              "selection_note": "Development validation reused for C and design comparison; not final performance"}
    if args.all_splits:
        summary = all_splits_summary({position: runs[position]["results"] for position in positions})
        report.update({"splits": {str(position): split_report(position) for position in positions},
                       "all_splits": summary,
                       "all_splits_note": "Per-split scores and their plain mean; cases are not pooled across splits"})
        private = {str(position): private_predictions(position) for position in positions}
    else:
        report.update(split_report(positions[0]))
        if control_results:
            report["controls"] = control_results
        private = private_predictions(positions[0])
    # New run directories prevent overwriting prior evidence. All artifacts are private.
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (args.output_dir / "validation_predictions.private.json").write_text(
        json.dumps(private, indent=2, allow_nan=False), encoding="utf-8")
    if args.all_splits:
        lines = ["# Development validation results, split positions 0-4", "",
                 "Validation balanced accuracy per split and the plain mean; cases are not pooled.", "",
                 "| Experiment | " + " | ".join(str(p) for p in positions) + " | Mean |",
                 "|---|" + "---:|" * (len(positions) + 1)]
        for name, entry in summary.items():
            scores = entry["validation_balanced_accuracy"]
            lines.append("| {} | {} | {:.3f} |".format(
                name, " | ".join("{:.3f}".format(scores[str(p)]) for p in positions), entry["mean"]))
    else:
        results = runs[positions[0]]["results"]
        lines = ["# Development validation results", "", "Not final test results. Full private evidence is in metrics.json.", "",
                 "| Experiment | C | Train BA | Validation BA | Retract recall | Advance recall |",
                 "|---|---:|---:|---:|---:|---:|"]
        for name, result in results.items():
            score = result["validation"]
            lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                name, result.get("selected_C", "—"), result["train"]["balanced_accuracy"],
                score["balanced_accuracy"], score["retract_recall"], score["advance_recall"]))
        lines.extend(["", "## Diagnostic comparisons", ""])
        lines.extend("- " + note for note in diagnostic_notes(results))
        if control_results:
            lines.extend(["", "## Controls", ""])
            lines.extend("- " + line for line in control_lines(control_results, results))
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Development evaluation complete; inspect private run files on Perlmutter.")


if __name__ == "__main__":
    main()
