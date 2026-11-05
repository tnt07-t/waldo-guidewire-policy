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
difference exceeds twice the std of the differences and --min-effect.

H4 slack report: name a shuffled-label group after its real group plus
":null" (e.g. wd0.1='...wd0.1-split*' wd0.1:null='...wd0.1-null-*').
For every such pair it prints memorization slack (null last-N train
bal-acc), learnable signal (real - null train bal-acc), the generalization
gap (real train - val bal-acc), the slack region from the pre-registered
budget, and the paired real - null last-N validation AUC test.
":null" groups are left out of the comparison with the first group.

--at-epochs 10,25,50 also reports validation AUC at fixed epochs (the H4
epoch-budget check, R7), per group and paired real - null.

  python summarize_runs.py --window 20 --min-effect 0.03 --at-epochs 10,25,50,100,200 \
      r0='docs/grp1/results/h4-r0-split*' r0:null='docs/grp1/results/h4-r0-null-*' \
      wd0.1='docs/grp1/results/h4-wd0.1-split*' wd0.1:null='docs/grp1/results/h4-wd0.1-null-*'

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
NULL_SUFFIX = ":null"

# Pre-registered H4 slack budget (docs/grp1/NEXT_EXPERIMENTS.md), on last-N
# train bal-acc. Real - null validation AUC must beat NULL_MIN_EFFECT.
TOO_MUCH_SLACK_NULL_TRAIN = 0.75
TARGET_NULL_TRAIN = (0.55, 0.70)
TARGET_REAL_TRAIN = (0.65, 0.90)
MIN_REAL_TRAIN = 0.65
MIN_LEARNABLE_SIGNAL = 0.05
NULL_MIN_EFFECT = 0.05


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
        "val_auc_by_epoch": {int(row["epoch"]): float(row["val_auc"]) for row in rows},
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


def paired_differences(baseline, candidate, metric, min_effect=0.0):
    """candidate - baseline per shared (split, seed) key.

    ``metric`` is a COLUMNS label or a function run -> value (NaN skips the
    pair). Passes when mean > 2 * std and mean > ``min_effect``.
    """
    value = dict(COLUMNS)[metric] if isinstance(metric, str) else metric
    base = {run["key"]: value(run) for run in baseline}
    cand = {run["key"]: value(run) for run in candidate}
    shared = sorted(key for key in set(base) & set(cand)
                    if not math.isnan(base[key]) and not math.isnan(cand[key]))
    diffs = [cand[key] - base[key] for key in shared]
    mean, std = mean_std(diffs)
    return {
        "pairs": len(shared),
        "keys": shared,
        "diffs": diffs,
        "mean": mean,
        "std": std,
        "wins": sum(d > 0 for d in diffs),
        "passes": (bool(shared) and not math.isnan(mean) and mean > 2 * std
                   and mean > min_effect),
    }


def val_auc_at(epoch):
    """Run -> validation AUC at ``epoch`` (NaN if the run is shorter)."""
    return lambda run: run["val_auc_by_epoch"].get(epoch, math.nan)


def train_bal(run):
    return run["window"]["train_bal_acc"]["mean"]


def val_bal(run):
    return run["window"]["val_bal_acc"]["mean"]


def slack_region(null_train, real_train):
    """Classify one setting against the pre-registered H4 slack budget."""
    if null_train > TOO_MUCH_SLACK_NULL_TRAIN:
        return "too much slack"
    if real_train < MIN_REAL_TRAIN or real_train - null_train < MIN_LEARNABLE_SIGNAL:
        return "too little slack"
    if (TARGET_NULL_TRAIN[0] <= null_train <= TARGET_NULL_TRAIN[1]
            and TARGET_REAL_TRAIN[0] <= real_train <= TARGET_REAL_TRAIN[1]):
        return "target"
    return "outside target"


def slack_report(real, null):
    """H4 slack quantities for one setting from its real and null groups."""
    if any(run["null"] for run in real) or not all(run["null"] for run in null):
        raise SystemExit("Slack report needs real-label runs and shuffled-label null runs")
    null_train = mean_std([train_bal(run) for run in null])[0]
    real_train = mean_std([train_bal(run) for run in real])[0]
    return {
        "null_train": null_train,
        "real_train": real_train,
        "learnable": real_train - null_train,
        "gap": mean_std([train_bal(run) - val_bal(run) for run in real])[0],
        "region": slack_region(null_train, real_train),
        "vs_null": paired_differences(null, real, "last-N val auc", NULL_MIN_EFFECT),
    }


def format_slack(pairs, epochs):
    lines = ["== H4 slack (last-N train bal-acc; budget null {}-{}, real {}-{}) ==".format(
        *TARGET_NULL_TRAIN, *TARGET_REAL_TRAIN),
        "{:<16} {:>10} {:>10} {:>10} {:>8} {:<18} {}".format(
            "setting", "null train", "real train", "learnable", "gap", "region",
            "real - null last-N val auc")]
    for name, real, null in pairs:
        r = slack_report(real, null)
        v = r["vs_null"]
        lines.append("{:<16} {:>10.3f} {:>10.3f} {:>+10.3f} {:>8.3f} {:<18} "
                     "pairs {} mean {:+.3f} std {:.3f} > max(2*std, {}): {}".format(
                         name, r["null_train"], r["real_train"], r["learnable"], r["gap"],
                         r["region"], v["pairs"], v["mean"], v["std"], NULL_MIN_EFFECT,
                         "yes" if v["passes"] else "no"))
        for epoch in epochs:
            e = paired_differences(null, real, val_auc_at(epoch))
            lines.append("{:<16}   val auc @ epoch {:>4}: real - null pairs {} mean {:+.3f} "
                         "std {:.3f}".format("", epoch, e["pairs"], e["mean"], e["std"]))
    return lines


def format_epoch_table(groups, epochs):
    lines = ["== validation AUC at fixed epochs (mean +- std over runs) ==",
             "{:<16}".format("group") + "".join(" {:>14}".format("epoch {}".format(e))
                                                for e in epochs)]
    for name, runs in groups:
        cells = []
        for epoch in epochs:
            m, sd = mean_std([val_auc_at(epoch)(run) for run in runs])
            cells.append(" {:>14}".format("{:.3f}+-{:.3f}".format(m, sd)))
        lines.append("{:<16}".format(name) + "".join(cells))
    return lines


def null_pairs(groups):
    """(name, real_runs, null_runs) for every NAME / NAME:null pair of groups."""
    by_name = dict(groups)
    return [(name[:-len(NULL_SUFFIX)], by_name[name[:-len(NULL_SUFFIX)]], runs)
            for name, runs in groups
            if name.endswith(NULL_SUFFIX) and name[:-len(NULL_SUFFIX)] in by_name]


def parse_epochs(text):
    try:
        epochs = [int(part) for part in text.split(",") if part]
    except ValueError:
        raise argparse.ArgumentTypeError("Expected comma-separated epochs, got {!r}".format(text))
    if not epochs or any(e <= 0 for e in epochs):
        raise argparse.ArgumentTypeError("Epochs must be positive integers")
    return epochs


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
    parser.add_argument("--min-effect", type=float, default=0.0,
                        help="Minimum mean paired gain over the first group to pass "
                             "(H4 uses 0.03); default 0 keeps only the 2*std rule")
    parser.add_argument("--at-epochs", type=parse_epochs, default=[],
                        help="Comma-separated epochs at which to report validation AUC")
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be positive")
    names = [name for name, _ in args.groups]
    if len(set(names)) != len(names):
        parser.error("Group names must be unique")

    groups = [(name, load_group(pattern, args.window)) for name, pattern in args.groups]
    print("last-N = mean over the last {} epochs of each run".format(args.window))
    for name, runs in groups:
        print()
        print("\n".join(format_table(name, runs)))

    if args.at_epochs:
        print()
        print("\n".join(format_epoch_table(groups, args.at_epochs)))

    base_name, base_runs = groups[0]
    for name, runs in groups[1:]:
        if name.endswith(NULL_SUFFIX):
            continue
        print()
        print("== paired: {} - {} ==".format(name, base_name))
        for metric in PAIRED_METRICS:
            result = paired_differences(base_runs, runs, metric, args.min_effect)
            print("{:<16} pairs {:>2} | mean {:+.3f} std {:.3f} | {} better in {}/{} | "
                  "mean > max(2*std, {:g}): {}".format(
                      metric, result["pairs"], result["mean"], result["std"], name,
                      result["wins"], result["pairs"], args.min_effect,
                      "yes" if result["passes"] else "no"))

    pairs = null_pairs(groups)
    if pairs:
        print()
        print("\n".join(format_slack(pairs, args.at_epochs)))


if __name__ == "__main__":
    main()
