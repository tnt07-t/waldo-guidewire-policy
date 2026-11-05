"""Summarize explicit static/temporal metrics files, keeping protocols separate.

Use on Perlmutter. This reads aggregate results, never private predictions.
No automatic winner: validation scores may have different selection budgets.
"""

import argparse
import json
from pathlib import Path


def summarize(paths):
    lines = ["# Development run ledger", "", "No automatic ranking: compare protocols before interpreting differences.", "",
             "| Run | Model | Split | Seed | Train BA | Validation BA | Retract recall | Advance recall | Total ms |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for path in paths:
        path = Path(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        config = report.get("config", {})
        split = config.get("split_position", report.get("split_position", "unknown"))
        for model, result in report["results"].items():
            val = result["validation"]
            train = result.get("train", {}).get("balanced_accuracy")
            latency = result.get("latency", {}).get("milliseconds", {}).get("total", {}).get("median")
            lines.append("| {} | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
                path.parent.name, model, split, model.split("seed=")[-1] if "seed=" in model else "—",
                "{:.3f}".format(train) if train is not None else "—", val["balanced_accuracy"],
                val["retract_recall"], val["advance_recall"],
                "{:.2f}".format(latency) if latency is not None else "—"))
        lines.extend(["", "Run `{}`: encoder `{}`, checkpoint selection `{}`, git `{}`, dirty `{}`.".format(
            path.parent.name, config.get("encoder", report.get("encoders", "unknown")),
            "validation BA / early stopping" if config else "validation C selection",
            report.get("git_commit", "unknown"), report.get("git_dirty", "unknown")), ""])
        if config:
            lines.append('Classifier activation: {}.'.format(config.get('classifier_activation','relu')))
            lines.append("Budget: epochs={}, patience={}, lr={}, weight_decay={}, width={}, dropout={}.".format(
                *(config.get(k, "unknown") for k in ("epochs", "patience", "lr", "weight_decay", "width", "dropout"))))
    lines.extend(["", "Latency entries are synchronized same-case microbenchmarks, not workload-wide p95.",
                  "Old static encoder-only timing is omitted because it excludes the classifier.",
                  "Seeds remain separate; do not select a lucky seed or call selected validation scores final performance."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    content = summarize(args.metrics)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(content)
    else:
        print(content)


if __name__ == "__main__":
    main()
