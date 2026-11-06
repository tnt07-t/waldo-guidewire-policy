"""Export one run's metrics.csv to a Google Sheets–friendly CSV.

Single input → single output. One output row per recorded epoch.

Output columns (exact order):
  Epoch,Training Accuracy,Training Loss,Validation Accuracy,Validation Loss

Uses ordinary accuracy columns ``train_acc`` and ``val_acc`` (and
``train_loss`` / ``val_loss``). Does not substitute balanced accuracy.

Known source layouts:
  - train_cnngru.py (e.g. H2): docs/grp1/results/<run>/metrics.csv
    columns include epoch, train_acc, train_loss, val_acc, val_loss, ...
  - probe_cnn_features.py (newer): docs/grp1/results/<run>/<design>/metrics.csv
    may include train_acc / val_acc
  - Older probe CSVs: train_bal_acc / val_bal_acc only, no per-epoch confusion
    or predictions → this tool refuses to invent ordinary accuracy

Examples (Perlmutter, repo root):

  python export_metrics_sheets.py \\
    docs/grp1/results/h2-lr3e4-split0-20261104-232604/metrics.csv \\
    -o sheets/h2-lr3e4.csv

  python export_metrics_sheets.py \\
    docs/grp1/results/probe-cnn-nomaxpool-seed1-split0-20261105-132018/goal_only/metrics.csv \\
    -o sheets/probe-goalonly-nomaxpool-seed1.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

OUT_FIELDS = (
    "Epoch",
    "Training Accuracy",
    "Training Loss",
    "Validation Accuracy",
    "Validation Loss",
)

REQUIRED_LOSS = ("train_loss", "val_loss")
REQUIRED_ACC = ("train_acc", "val_acc")


def _parse_epoch(raw, row_index):
    text = (raw or "").strip()
    if text == "":
        raise ValueError("Row {}: missing epoch value".format(row_index))
    try:
        # Allow "1" or "1.0" but require an integral epoch index.
        value = float(text)
    except ValueError as exc:
        raise ValueError("Row {}: epoch is not numeric: {!r}".format(
            row_index, raw)) from exc
    if not value.is_integer():
        raise ValueError("Row {}: epoch must be an integer, got {!r}".format(
            row_index, raw))
    return int(value)


def _accuracy_from_confusion(raw, row_index, field_name):
    """correct / total from a 2x2 confusion matrix saved in a CSV cell."""
    if raw is None or str(raw).strip() == "":
        return None
    text = str(raw).strip()
    try:
        matrix = json.loads(text)
    except json.JSONDecodeError:
        try:
            matrix = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "Row {}: cannot parse {}: {!r}".format(
                    row_index, field_name, raw)) from exc
    try:
        total = 0
        correct = 0
        for i, row in enumerate(matrix):
            for j, count in enumerate(row):
                n = int(count)
                total += n
                if i == j:
                    correct += n
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Row {}: {} is not a numeric confusion matrix: {!r}".format(
                row_index, field_name, raw)) from exc
    if total <= 0:
        raise ValueError(
            "Row {}: {} has no predictions (total=0)".format(
                row_index, field_name))
    # Same float division as (preds == labels).double().mean().item()
    return correct / total


def _accuracy_from_predictions(pred_raw, label_raw, row_index):
    if pred_raw is None or label_raw is None:
        return None
    pred_text = str(pred_raw).strip()
    label_text = str(label_raw).strip()
    if pred_text == "" or label_text == "":
        return None
    try:
        preds = json.loads(pred_text)
        labels = json.loads(label_text)
    except json.JSONDecodeError:
        try:
            preds = ast.literal_eval(pred_text)
            labels = ast.literal_eval(label_text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "Row {}: cannot parse predictions/labels".format(
                    row_index)) from exc
    if len(preds) != len(labels) or len(preds) == 0:
        raise ValueError(
            "Row {}: predictions/labels length mismatch or empty".format(
                row_index))
    correct = sum(int(p) == int(y) for p, y in zip(preds, labels))
    return correct / len(preds)


def resolve_accuracy(row, row_index, split):
    """Ordinary accuracy for train or val; never balanced accuracy."""
    acc_key = "{}_acc".format(split)
    if acc_key in row and str(row[acc_key]).strip() != "":
        return row[acc_key]

    confusion_keys = (
        "{}_confusion".format(split),
        "{}_confusion_matrix".format(split),
    )
    for key in confusion_keys:
        if key in row:
            value = _accuracy_from_confusion(row[key], row_index, key)
            if value is not None:
                # Preserve full float repr without forcing percent scale.
                return repr(value) if isinstance(value, float) else value

    pred_key = "{}_preds".format(split)
    label_keys = ("{}_labels".format(split), "label", "labels")
    if pred_key in row:
        for label_key in label_keys:
            if label_key in row:
                value = _accuracy_from_predictions(
                    row[pred_key], row[label_key], row_index)
                if value is not None:
                    return repr(value) if isinstance(value, float) else value

    bal_key = "{}_bal_acc".format(split)
    has_bal = bal_key in row and str(row[bal_key]).strip() != ""
    raise ValueError(
        "Cannot supply ordinary {split} accuracy for row {row}: "
        "column {acc!r} is missing, and this metrics.csv has no per-epoch "
        "confusion counts or predictions to recompute it"
        "{bal}. "
        "Balanced accuracy ({bal_key!r}) will not be substituted, and a "
        "best-epoch summary.json entry cannot fill every epoch. "
        "Re-run the experiment with a writer that logs {acc!r}, or export "
        "from a metrics.csv that already contains it "
        "(e.g. train_cnngru.py runs)."
        .format(
            split=split,
            row=row_index,
            acc=acc_key,
            bal=(" (only {0} is present)".format(bal_key) if has_bal else ""),
            bal_key=bal_key,
        )
    )


def convert_file(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Empty CSV (no header): {}".format(path))
        fields = set(reader.fieldnames)
        if "epoch" not in fields:
            raise ValueError(
                "{}: missing required column 'epoch'; got {}".format(
                    path, sorted(fields)))
        for key in REQUIRED_LOSS:
            if key not in fields:
                raise ValueError(
                    "{}: missing required column {!r}; got {}".format(
                        path, key, sorted(fields)))

        seen = {}
        rows = []
        for row_index, row in enumerate(reader, start=2):  # header is line 1
            epoch = _parse_epoch(row.get("epoch"), row_index)
            if epoch in seen:
                raise ValueError(
                    "{}: duplicate epoch {} (rows {} and {})".format(
                        path, epoch, seen[epoch], row_index))
            seen[epoch] = row_index

            train_loss = row.get("train_loss")
            val_loss = row.get("val_loss")
            if train_loss is None or str(train_loss).strip() == "":
                raise ValueError(
                    "{}: row {}: missing train_loss".format(path, row_index))
            if val_loss is None or str(val_loss).strip() == "":
                raise ValueError(
                    "{}: row {}: missing val_loss".format(path, row_index))

            rows.append({
                "Epoch": str(epoch),
                "Training Accuracy": resolve_accuracy(row, row_index, "train"),
                "Training Loss": train_loss,
                "Validation Accuracy": resolve_accuracy(row, row_index, "val"),
                "Validation Loss": val_loss,
            })

    if not rows:
        raise ValueError("{}: no epoch rows found".format(path))
    rows.sort(key=lambda item: int(item["Epoch"]))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "metrics",
        type=Path,
        help="Path to one metrics.csv from a single run",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output CSV path (Sheets-importable)",
    )
    args = parser.parse_args()

    if not args.metrics.is_file():
        sys.exit("Missing file: {}".format(args.metrics))

    try:
        rows = convert_file(args.metrics)
    except ValueError as exc:
        sys.exit(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("Source: {}".format(args.metrics))
    print("Output: {}".format(args.output))
    print("Epochs exported: {}".format(len(rows)))


if __name__ == "__main__":
    main()
