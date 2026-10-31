"""Read original-resolution guidewire cases without model preprocessing.

Real dataset access must run on Perlmutter. Only Pillow is required; PyTorch
is not required for this initial reader.
"""

import argparse
import json
import operator
import os
from pathlib import Path

from PIL import Image


class GuidewireDataset:
    """An indexable view of one partition of one supplied split object.

    ``split_position`` is a zero-based position in splits.json, NOT the
    object's ``fold`` value. No split is chosen implicitly.
    """

    def __init__(self, data_dir, *, split_position, partition):
        self.root = Path(data_dir).expanduser().resolve()
        if partition not in ("train", "validation", "test"):
            raise ValueError("partition must be train, validation, or test")
        self.records = []
        with (self.root / "cases.jsonl").open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except ValueError as exc:
                    raise ValueError(
                        "Invalid JSON at cases.jsonl line {}".format(line_number)
                    ) from exc
                self.records.append(record)
        with (self.root / "splits.json").open(encoding="utf-8") as handle:
            splits = json.load(handle)
        if type(split_position) is not int or not 0 <= split_position < len(splits):
            raise ValueError("split_position is outside the supplied split list")
        self.split_position = split_position
        self.partition = partition
        self.fold = splits[split_position]["fold"]
        # Partition entries are row positions in cases.jsonl, not case IDs.
        self.indices = tuple(splits[split_position][partition])
        if any(type(i) is not int or not 0 <= i < len(self.records)
               for i in self.indices):
            raise ValueError("Selected partition contains an invalid record index")

    def __len__(self):
        return len(self.indices)

    def _read_image(self, relative_path, row_index):
        relative_path = Path(relative_path)
        if relative_path.is_absolute():
            raise ValueError("Image paths must be relative to the dataset root")
        # Resolve .. and symlinks before checking that the file stays inside root.
        path = (self.root / relative_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Image path escapes the dataset root") from exc
        try:
            with Image.open(path) as image:
                image.load()
                # Detach pixels from the file so no open file handle is returned.
                # Preserve resolution, color mode, and pixel values.
                return image.copy()
        except (OSError, ValueError) as exc:
            raise OSError(
                "Cannot decode image for record row {}: {}".format(
                    row_index, relative_path
                )
            ) from exc

    def __getitem__(self, index):
        row_index = self.indices[operator.index(index)]
        record = self.records[row_index]
        paths = record["paths"]
        if len(paths) != 7:
            raise ValueError("Expected goal plus six history paths")
        depth = record["depth_relative_mm"]
        time = record["history_offsets_s"]
        if len(depth) != 6 or len(time) != 6:
            raise ValueError("Expected six aligned depths and time offsets")
        # paths[0] is the marked goal; paths[1:] are oldest-to-newest history.
        images = [self._read_image(path, row_index) for path in paths]
        return {
            "goal": images[0],
            "history": images[1:],
            "depth_relative_mm": list(depth),
            "history_offsets_s": list(time),
            "label": record["label"],
            "case_id": record["case_id"],
            "episode": record["episode"],
            "row_index": row_index,
        }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    parser.add_argument("--split-position", type=int, required=True)
    parser.add_argument("--partition", choices=("train", "validation", "test"),
                        required=True)
    parser.add_argument("--count", type=int, default=1,
                        help="Number of cases to decode, not a training batch size")
    args = parser.parse_args()
    if not args.data_dir:
        parser.error("set DATA_DIR or provide --data-dir")
    if args.count <= 0:
        parser.error("--count must be positive")
    dataset = GuidewireDataset(args.data_dir, split_position=args.split_position,
                               partition=args.partition)
    print("Selected partition cases:", len(dataset))
    count = min(args.count, len(dataset))
    for i in range(count):
        case = dataset[i]
        images = [case["goal"]] + case["history"]
        print("Sample {}: {} images; sizes={}; modes={}; depths={}; times={}".format(
            i, len(images), sorted({image.size for image in images}),
            sorted({image.mode for image in images}),
            len(case["depth_relative_mm"]), len(case["history_offsets_s"])))
    print("Decoded {} cases; no preprocessing applied.".format(count))


if __name__ == "__main__":
    main()
