"""Choose WirePreprocessor settings inside the training partition only.

Episode-grouped inner CV (``evaluation.inner_cv``) scores each configuration
on the training partition; validation and test images are never encoded.
Parameters are searched one at a time in a fixed order, keeping the others
at their current best. Within one standard deviation of the best, the cheaper
configuration wins. The selected configuration is then confirmed ONCE on
validation with ``guidewire.evaluation.probes``; this script prints that command.

Real data and outputs stay on Perlmutter, under personal PSCRATCH.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from guidewire.evaluation.probes import MODES, check_partitions, collect, inner_cv


def select_within_one_std(scored):
    """Pick from ``scored`` (cheapest first) the first within one std of the best.

    Each entry has ``mean`` and ``std``; the best entry's std sets the margin.
    """
    best = max(scored, key=lambda entry: entry["mean"])
    for entry in scored:
        if entry["mean"] >= best["mean"] - best["std"]:
            return entry
    return best


def stage_candidates(parameter, best, crop_sizes, tophat_sizes):
    """Candidates for one search stage, cheapest first, from the current best."""
    if parameter == "crop_size":
        return [{"crop_size": size, "channels": "raw"} for size in sorted(crop_sizes)]
    if parameter == "channels":
        raw = {"crop_size": best["crop_size"], "channels": "raw"}
        return [raw, {**raw, "channels": "enhanced", "tophat_size": sorted(tophat_sizes)[0],
                      "difference": "median"}]
    if parameter == "difference":
        return [best, {**best, "difference": "previous"}]
    if parameter == "tophat_size":
        return [{**best, "tophat_size": size} for size in sorted(tophat_sizes)]
    raise ValueError("Unknown search parameter")


def search(score, crop_sizes, tophat_sizes):
    """Run the one-parameter-at-a-time search with ``score(config) -> entry``."""
    stages, best = [], None
    for parameter in ("crop_size", "channels", "difference", "tophat_size"):
        enhanced = best is not None and best["channels"] == "enhanced"
        if parameter == "difference" and not enhanced:
            continue
        # Top-hat size is only searched if inspection left more than one value.
        if parameter == "tophat_size" and (not enhanced or len(tophat_sizes) < 2):
            continue
        scored = [score(config) for config in
                  stage_candidates(parameter, best, crop_sizes, tophat_sizes)]
        selected = select_within_one_std(scored)
        best = selected["config"]
        stages.append({"parameter": parameter,
                       "candidates": [{key: entry[key] for key in ("config", "mean", "std")}
                                      for entry in scored],
                       "selected": best})
    return best, stages


def config_label(config):
    return ", ".join("{}={}".format(key, value) for key, value in config.items())


def confirm_command(config, split_position, standardize=False):
    flags = ["--preprocessor wire"] + ["--{} {}".format(key.replace("_", "-"), value)
                                       for key, value in config.items()]
    if standardize and config["channels"] == "enhanced":
        flags.append("--standardize-channels")
    return ("python -m guidewire.evaluation.probes --split-position {} --encoders resnet18 {} "
            "--device cuda --output-dir \"$PSCRATCH/waldo-evaluation/<new-run>\"".format(
                split_position, " ".join(flags)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    parser.add_argument("--split-position", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New private directory under PSCRATCH")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--crop-sizes", type=int, nargs="+", default=[192, 256, 384, 512],
                        help="Candidates from the notebook crop-size tally")
    parser.add_argument("--tophat-sizes", type=int, nargs="+", default=[11],
                        help="Searched only if more than one plausible value remains")
    parser.add_argument("--mode", choices=MODES, default="latest_goal",
                        help="Probe feature design scored in each fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--standardize-channels", action="store_true",
                        help="Standardize enhanced top-hat/difference channels with stats "
                             "fitted on the training partition (no labels used)")
    args = parser.parse_args()
    scratch = os.environ.get("PSCRATCH")
    if not scratch or not args.data_dir:
        parser.error("Run on Perlmutter with PSCRATCH and DATA_DIR set")
    root = Path(scratch).resolve()
    for path in (Path(args.data_dir).resolve(), args.output_dir.resolve()):
        if root not in path.parents:
            parser.error("Data and output must be inside personal PSCRATCH")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new run directory")
    if args.folds < 2:
        parser.error("--folds must be at least 2")

    import torch
    from guidewire.encoders.cnn import FrameEncoder
    from guidewire.data.dataset import GuidewireDataset
    from guidewire.data.preprocessing import WirePreprocessor, standardize_wire_channels

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use an allocated GPU node or explicitly choose cpu")
    try:
        for size in args.crop_sizes:
            WirePreprocessor(crop_size=size)
        for size in args.tophat_sizes:
            WirePreprocessor(channels="enhanced", tophat_size=size)
    except ValueError as exc:
        parser.error(str(exc))
    datasets = {partition: GuidewireDataset(args.data_dir, split_position=args.split_position,
                                          partition=partition)
                for partition in ("train", "validation", "test")}
    check_partitions(datasets)
    encoder = FrameEncoder().to(args.device).eval()

    cache = {}

    def score(config):
        # Each configuration's training features are extracted once.
        key = json.dumps(config, sort_keys=True)
        if key not in cache:
            prepare = WirePreprocessor(**config)
            extra = {}
            if args.standardize_channels and config["channels"] == "enhanced":
                # Label-free stats from the whole training partition, shared by
                # all inner folds; validation is still never touched.
                train = datasets["train"]
                prepare = standardize_wire_channels(
                    prepare, (train[i] for i in range(len(train))))
                extra["channel_stats"] = {"tophat_stats": prepare.tophat_stats,
                                          "difference_stats": prepare.difference_stats}
            _, features = collect(datasets["train"], prepare, {"resnet18": encoder})
            cache[key] = {"config": config, **extra,
                          **inner_cv(features["resnet18"], args.mode, n_splits=args.folds)}
            print("{}: inner BA {:.3f} +/- {:.3f}".format(
                config_label(config), cache[key]["mean"], cache[key]["std"]))
        return cache[key]

    selected, stages = search(score, args.crop_sizes, args.tophat_sizes)
    command = confirm_command(selected, args.split_position, args.standardize_channels)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "git_commit": git.stdout.strip(), "git_dirty": bool(dirty.stdout.strip()),
              "device": args.device, "split_position": args.split_position,
              "fold": datasets["train"].fold, "encoder": "resnet18", "mode": args.mode,
              "inner_folds": args.folds, "training_cases": len(datasets["train"]),
              "standardize_channels": args.standardize_channels,
              "selection_rule": "one parameter at a time; cheapest within one std of the best",
              "validation_note": "Validation and test were not encoded; confirm the selection once with guidewire.evaluation.probes",
              "configurations": list(cache.values()), "stages": stages,
              "selected": selected, "confirm_command": command}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "selection.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Preprocessing selection (training partition only)", "",
             "Episode-grouped {}-fold inner CV, `{}` design, ResNet-18.".format(args.folds, args.mode),
             "Validation and test were not used.", "",
             "| Configuration | Inner BA mean | Inner BA std |", "|---|---:|---:|"]
    for entry in cache.values():
        lines.append("| {} | {:.3f} | {:.3f} |".format(
            config_label(entry["config"]), entry["mean"], entry["std"]))
    lines.extend(["", "Selected: " + config_label(selected), "",
                  "Confirm once on validation:", "", "```sh", command, "```"])
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
