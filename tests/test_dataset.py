"""Synthetic-only loader checks: no real records or images are needed."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dataset import GuidewireDataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Distinct pixel values reveal wrong case selection or frame ordering.
        records = []
        for row in range(3):
            paths = []
            for frame in range(7):
                path = "row{}-image{}.png".format(row, frame)
                Image.new("L", (13, 11), row * 10 + frame).save(self.root / path)
                paths.append(path)
            records.append({
                "case_id": "invented-case-{}".format(row),
                "episode": "invented-episode-{}".format(row),
                "paths": paths,
                "depth_relative_mm": [row, -4, -3, -2, -1, 0],
                "history_offsets_s": [-9, -7, -4, -2, -1, 0],
                "label": row % 2,
            })
        (self.root / "cases.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        # Nonsequential rows and fold numbers catch split-index confusion.
        (self.root / "splits.json").write_text(json.dumps([
            {"fold": 42, "train": [2, 0], "validation": [1], "test": []},
            {"fold": 99, "train": [1], "validation": [], "test": [0, 2]},
        ]), encoding="utf-8")

    def dataset(self, **kwargs):
        return GuidewireDataset(self.root, split_position=kwargs.get("split_position", 0),
                                partition=kwargs.get("partition", "train"))

    def test_selection_alignment_and_original_pixels(self):
        dataset = self.dataset()
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.fold, 42)
        case = dataset[0]
        self.assertEqual(case["row_index"], 2)
        self.assertEqual(case["case_id"], "invented-case-2")
        self.assertEqual(case["episode"], "invented-episode-2")
        self.assertEqual(case["label"], 0)
        images = [case["goal"]] + case["history"]
        self.assertEqual([im.getpixel((0, 0)) for im in images], list(range(20, 27)))
        self.assertTrue(all(im.size == (13, 11) and im.mode == "L" for im in images))
        self.assertEqual(case["depth_relative_mm"], [2, -4, -3, -2, -1, 0])
        self.assertEqual(case["history_offsets_s"], [-9, -7, -4, -2, -1, 0])
        # The loaded images remain usable after their source files are removed.
        for path in self.root.glob("row2-*.png"):
            path.unlink()
        self.assertEqual(case["goal"].getpixel((0, 0)), 20)

    def test_other_split_and_partition(self):
        dataset = self.dataset(split_position=1, partition="test")
        self.assertEqual([dataset[i]["row_index"] for i in range(len(dataset))], [0, 2])
        self.assertEqual(len(self.dataset(partition="test")), 0)

    def test_color_images_are_not_converted(self):
        Image.new("RGBA", (17, 19), (11, 22, 33, 44)).save(
            self.root / "row2-image0.png")
        image = self.dataset()[0]["goal"]
        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(image.size, (17, 19))
        self.assertEqual(image.getpixel((0, 0)), (11, 22, 33, 44))

    def test_missing_or_corrupt_image_fails_instead_of_skipping(self):
        path = self.root / "row2-image3.png"
        path.unlink()
        with self.assertRaisesRegex(OSError, "record row 2"):
            self.dataset()[0]
        path.write_bytes(b"not an image")
        with self.assertRaisesRegex(OSError, "record row 2"):
            self.dataset()[0]

    def test_unsafe_paths_are_rejected(self):
        dataset = self.dataset()
        for path in ("../outside.png", "/outside.png"):
            dataset.records[2]["paths"][0] = path
            with self.assertRaises(ValueError):
                dataset[0]

    def test_invalid_selection_fails(self):
        for position in (-1, 2, True):
            with self.assertRaises(ValueError):
                self.dataset(split_position=position)
        with self.assertRaises(ValueError):
            self.dataset(partition="typo")
        with self.assertRaises(IndexError):
            self.dataset()[2]


if __name__ == "__main__":
    unittest.main()
