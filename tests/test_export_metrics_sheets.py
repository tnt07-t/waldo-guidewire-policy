"""Checks for Sheets export of a single-run metrics.csv."""

import csv
import tempfile
import unittest
from pathlib import Path

from export_metrics_sheets import convert_file


class ExportMetricsSheetsTests(unittest.TestCase):
    def _write(self, header, rows):
        path = Path(tempfile.mkdtemp()) / "metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_h2_style_columns_sorted_and_preserved(self):
        path = self._write(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best"],
            [
                {"epoch": "3", "train_loss": "0.50001", "train_acc": "0.712345",
                 "val_loss": "0.80001", "val_acc": "0.612345", "best": "False"},
                {"epoch": "1", "train_loss": "0.90000", "train_acc": "0.500000",
                 "val_loss": "0.95000", "val_acc": "0.500000", "best": "True"},
                {"epoch": "2", "train_loss": "0.70000", "train_acc": "0.600000",
                 "val_loss": "0.85000", "val_acc": "0.550000", "best": "False"},
            ],
        )
        rows = convert_file(path)
        self.assertEqual([r["Epoch"] for r in rows], ["1", "2", "3"])
        self.assertEqual(rows[0]["Training Accuracy"], "0.500000")
        self.assertEqual(rows[2]["Training Loss"], "0.50001")
        self.assertEqual(list(rows[0]), [
            "Epoch", "Training Accuracy", "Training Loss",
            "Validation Accuracy", "Validation Loss",
        ])

    def test_duplicate_epoch_rejected(self):
        path = self._write(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
            [
                {"epoch": "1", "train_loss": "0.1", "train_acc": "0.5",
                 "val_loss": "0.2", "val_acc": "0.4"},
                {"epoch": "1", "train_loss": "0.1", "train_acc": "0.5",
                 "val_loss": "0.2", "val_acc": "0.4"},
            ],
        )
        with self.assertRaisesRegex(ValueError, "duplicate epoch"):
            convert_file(path)

    def test_old_probe_bal_acc_only_rejected(self):
        path = self._write(
            ["epoch", "train_loss", "train_bal_acc", "val_loss",
             "val_bal_acc", "val_auc", "best"],
            [{
                "epoch": "23",
                "train_loss": "0.69",
                "train_bal_acc": "0.512",
                "val_loss": "0.6777",
                "val_bal_acc": "0.664",
                "val_auc": "0.702",
                "best": "True",
            }],
        )
        with self.assertRaisesRegex(ValueError, "ordinary train accuracy"):
            convert_file(path)

    def test_accuracy_from_confusion_when_acc_missing(self):
        path = self._write(
            ["epoch", "train_loss", "val_loss",
             "train_confusion", "val_confusion"],
            [{
                "epoch": "1",
                "train_loss": "0.5",
                "val_loss": "0.6",
                # 3/4 correct
                "train_confusion": "[[1, 1], [0, 2]]",
                # 2/4 correct
                "val_confusion": "[[1, 1], [1, 1]]",
            }],
        )
        rows = convert_file(path)
        self.assertEqual(rows[0]["Training Accuracy"], repr(0.75))
        self.assertEqual(rows[0]["Validation Accuracy"], repr(0.5))


if __name__ == "__main__":
    unittest.main()
