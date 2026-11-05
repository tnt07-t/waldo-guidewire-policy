"""Aggregate train_cnngru.py run folders and compare groups paired by split/seed.

Usage (run folders hold summary.json + metrics.csv):

  python summarize_runs.py --window 20 \
      null='docs/grp1/results/phase0-null-*' \
      real='docs/grp1/results/phase0-real-*'

Each NAME=PATTERN argument is one group; PATTERN is expanded with glob, so
quote it. For every run the table shows the best epoch (optimistic: a max
over noisy epochs), the final epoch and the mean over the last --window
epochs, recomputed from metrics.csv so older runs work too. Each group gets
mean +- std over its runs.

Every later group is then compared with the first one, paired by
(split_position, seed): differences are later - first, and a gain "passes"
the decision rule in docs/grp1/NEXT_EXPERIMENTS.md only when the mean
difference exceeds twice the std of the differences.

Aggregate metrics only; no images or case identifiers are read.
"""

import argparse
import csv
import glob
import json
import math
from pathlib import Path

from train_cnngru import window_summary

# (column label, function run -> value)
COLUMNS = (
    ("best val bal", lambda r: r["best_val_bal_acc"]),
    ("best val auc", lambda r: r["best_val_auc"]),
    ("final val auc", lambda r: r["final_val_auc"]),
    ("last-N val bal", lambda r: r["window"]["val_bal_acc"]["mean"]),
    ("last-N val auc", lambda r: r["window"]["val_auc"]["mean"]),
    ("last-N train bal", lambda r: r["window"]["train_bal_acc"]["mean"]),
)
PAIRED_METRICS = ("last-N val auc", "last-N val bal", "best val bal")


def load_run(run_dir, window):
    """One run folder -> dict of identifiers and summary metrics."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    args = summary["args"]
    return {
        "name": run_dir.name,
        "key": (int(args["split_position"]), int(args["seed"])),
        "null": bool(args.get("shuffle_train_labels", False)),
        "epochs_run": len(rows),
        "best_epoch": summary["best"]["epoch"],
        "best_val_bal_acc": summary["best"]["val"]["bal_acc"],
        "best_val_auc": summary["best"]["val"]["auc"],
        "final_val_auc": float(rows[-1]["val_auc"]),
        "window": window_summary(rows, window),
    }


def load_group(pattern, window):
    paths = sorted(path for path in glob.glob(pattern)
                   if (Path(path) / "summary.json").is_file())
    if not paths:
        raise SystemExit("No run folders with summary.json match {!r}".format(pattern))
    runs = [load_run(path, window) for path in paths]
    keys = [run["key"] for run in runs]
    if len(set(keys)) != len(keys):
        raise SystemExit("Pattern {!r} has several runs for one (split, seed)".format(pattern))
    return runs


def mean_std(values):
    """Population mean and std of the finite values; NaN if none."""
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def paired_differences(baseline, candidate, metric):
    """candidate - baseline per shared (split, seed) key, for one COLUMNS metric."""
    value = dict(COLUMNS)[metric]
    base = {run["key"]: value(run) for run in baseline}
    cand = {run["key"]: value(run) for run in candidate}
    shared = sorted(set(base) & set(cand))
    diffs = [cand[key] - base[key] for key in shared]
    mean, std = mean_std(diffs)
    return {
        "pairs": len(shared),
        "keys": shared,
        "diffs": diffs,
        "mean": mean,
        "std": std,
        "wins": sum(d > 0 for d in diffs),
        "passes": bool(shared) and not math.isnan(mean) and mean > 2 * std,
    }


def format_table(name, runs):
    header = "{:<44} {:>5} {:>4} {:>5}".format("run", "split", "seed", "best@") + "".join(
        " {:>16}".format(label) for label, _ in COLUMNS)
    lines = ["== {} ({} runs{}) ==".format(
        name, len(runs), ", shuffled-label null" if all(r["null"] for r in runs) else ""),
        header]
    for run in runs:
        lines.append("{:<44} {:>5} {:>4} {:>5}".format(
            run["name"][:44], run["key"][0], run["key"][1], run["best_epoch"]) + "".join(
            " {:>16.3f}".format(value(run)) for _, value in COLUMNS))
    stats = [mean_std([value(run) for run in runs]) for _, value in COLUMNS]
    lines.append("{:<44} {:>5} {:>4} {:>5}".format("mean +- std", "", "", "") + "".join(
        " {:>16}".format("{:.3f}+-{:.3f}".format(m, s)) for m, s in stats))
    return lines


def parse_group(text):
    name, sep, pattern = text.partition("=")
    if not sep or not name or not pattern:
        raise argparse.ArgumentTypeError("Expected NAME=PATTERN, got {!r}".format(text))
    return name, pattern


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="+", type=parse_group, metavar="NAME=PATTERN")
    parser.add_argument("--window", type=int, default=20,
                        help="Last-N epochs averaged per run (default 20)")
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be positive")

    groups = [(name, load_group(pattern, args.window)) for name, pattern in args.groups]
    print("last-N = mean over the last {} epochs of each run".format(args.window))
    for name, runs in groups:
        print()
        print("\n".join(format_table(name, runs)))

    base_name, base_runs = groups[0]
    for name, runs in groups[1:]:
        print()
        print("== paired: {} - {} ==".format(name, base_name))
        for metric in PAIRED_METRICS:
            result = paired_differences(base_runs, runs, metric)
            print("{:<16} pairs {:>2} | mean {:+.3f} std {:.3f} | {} better in {}/{} | "
                  "mean > 2*std: {}".format(
                      metric, result["pairs"], result["mean"], result["std"], name,
                      result["wins"], result["pairs"], "yes" if result["passes"] else "no"))


if __name__ == "__main__":
    main()
