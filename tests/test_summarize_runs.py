"""Invented-input checks for summarize_runs.py (temporary run folders only)."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from summarize_runs import load_group, load_run, mean_std, paired_differences

FIELDS = ("epoch", "train_bal_acc", "val_bal_acc", "val_auc", "val_loss")


def write_run(root, name, split, seed, val_auc, null=False, epochs=5):
    """Fake run folder: constant val AUC per epoch, best epoch 2."""
    run_dir = Path(root) / name
    run_dir.mkdir()
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            writer.writerow({"epoch": epoch, "train_bal_acc": 0.9, "val_bal_acc": 0.55,
                             "val_auc": val_auc, "val_loss": 0.7})
    summary = {"args": {"split_position": split, "seed": seed,
                        "shuffle_train_labels": null},
               "best": {"epoch": 2, "val": {"bal_acc": 0.6, "auc": val_auc}}}
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


class SummarizeRunsTests(unittest.TestCase):
    def test_load_run_recomputes_window(self):
        with tempfile.TemporaryDirectory() as root:
            run = load_run(write_run(root, "a", 1, 2, 0.6, null=True), window=3)
        self.assertEqual(run["key"], (1, 2))
        self.assertTrue(run["null"])
        self.assertEqual(run["epochs_run"], 5)
        self.assertEqual(run["window"]["first_epoch"], 3)
        self.assertAlmostEqual(run["window"]["val_auc"]["mean"], 0.6)
        self.assertAlmostEqual(run["final_val_auc"], 0.6)

    def test_older_summary_without_null_flag(self):
        with tempfile.TemporaryDirectory() as root:
            path = write_run(root, "old", 0, 0, 0.5)
            summary = json.loads((path / "summary.json").read_text())
            del summary["args"]["shuffle_train_labels"]
            (path / "summary.json").write_text(json.dumps(summary))
            self.assertFalse(load_run(path, window=20)["null"])

    def test_paired_differences_by_split_and_seed(self):
        with tempfile.TemporaryDirectory() as root:
            for split, seed, auc in ((0, 0, 0.50), (0, 1, 0.52), (1, 0, 0.48)):
                write_run(root, "null-{}-{}".format(split, seed), split, seed, auc, null=True)
            # Real runs beat null by 0.10 on two shared keys; key (2, 0) is unpaired.
            for split, seed, auc in ((0, 0, 0.60), (0, 1, 0.62), (2, 0, 0.90)):
                write_run(root, "real-{}-{}".format(split, seed), split, seed, auc)
            null = load_group(str(Path(root) / "null-*"), window=20)
            real = load_group(str(Path(root) / "real-*"), window=20)
        result = paired_differences(null, real, "last-N val auc")
        self.assertEqual(result["pairs"], 2)
        self.assertEqual(result["keys"], [(0, 0), (0, 1)])
        self.assertAlmostEqual(result["mean"], 0.10)
        self.assertAlmostEqual(result["std"], 0.0)
        self.assertEqual(result["wins"], 2)
        self.assertTrue(result["passes"])
        reverse = paired_differences(real, null, "last-N val auc")
        self.assertFalse(reverse["passes"])

    def test_duplicate_key_in_group_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            write_run(root, "r1", 0, 0, 0.5)
            write_run(root, "r2", 0, 0, 0.6)
            with self.assertRaises(SystemExit):
                load_group(str(Path(root) / "r*"), window=20)

    def test_mean_std_skips_nan(self):
        self.assertEqual(mean_std([1.0, math.nan, 3.0]), (2.0, 1.0))
        self.assertTrue(all(math.isnan(v) for v in mean_std([math.nan])))


if __name__ == "__main__":
    unittest.main()
