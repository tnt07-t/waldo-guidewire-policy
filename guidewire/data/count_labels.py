"""Count ADVANCE / RETRACT cases for one split (no images loaded).

Reads only cases.jsonl and splits.json. Run on Perlmutter with DATA_DIR set.

Example:
  python3 -m guidewire.data.count_labels --split-position 0
"""

import argparse
import json
import os
from pathlib import Path


def load_labels(data_dir):
    path = Path(data_dir) / "cases.jsonl"
    labels = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            label = record["label"]
            if label not in (0, 1):
                raise ValueError(
                    "Unexpected label {} at cases.jsonl line {}".format(
                        label, line_number
                    )
                )
            labels.append(label)
    return labels


def count_indices(labels, indices):
    retract = sum(1 for i in indices if labels[i] == 0)
    advance = sum(1 for i in indices if labels[i] == 1)
    return {
        "n": len(indices),
        "RETRACT_0": retract,
        "ADVANCE_1": advance,
        "frac_advance": advance / len(indices) if indices else float("nan"),
    }


def format_row(name, stats):
    return (
        "{:<12} n={:<4}  RETRACT(0)={:<4}  ADVANCE(1)={:<4}  "
        "frac_ADVANCE={:.3f}".format(
            name,
            stats["n"],
            stats["RETRACT_0"],
            stats["ADVANCE_1"],
            stats["frac_advance"],
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    parser.add_argument(
        "--split-position",
        type=int,
        required=True,
        help="Zero-based position in splits.json",
    )
    args = parser.parse_args()
    if not args.data_dir:
        parser.error("Set DATA_DIR or provide --data-dir")

    root = Path(args.data_dir).expanduser().resolve()
    labels = load_labels(root)
    with (root / "splits.json").open(encoding="utf-8") as handle:
        splits = json.load(handle)
    if not 0 <= args.split_position < len(splits):
        parser.error("split_position out of range")

    split = splits[args.split_position]
    print("DATA_DIR:", root)
    print("split_position:", args.split_position, "fold:", split.get("fold"))
    print(format_row("all_cases", count_indices(labels, range(len(labels)))))
    for name in ("train", "validation", "test"):
        print(format_row(name, count_indices(labels, split[name])))


if __name__ == "__main__":
    main()
