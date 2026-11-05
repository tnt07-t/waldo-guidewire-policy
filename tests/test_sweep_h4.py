"""Invented-input checks for sweep_h4.py and the in-process train_cnngru.run path."""

import csv
import functools
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import sweep_h4
import train_cnngru
from model_cnngru import CNNGRUModel


def invented_split(n_train=24, n_val=12, seed=0):
    generator = torch.Generator().manual_seed(seed)

    def part(n):
        return {"history": torch.randn(n, 6, 512, generator=generator).abs(),
                "goal": torch.randn(n, 512, generator=generator),
                "depth": torch.randn(n, 6, generator=generator),
                "offsets": torch.randn(n, 6, generator=generator),
                "label": torch.arange(n) % 2}

    return {"train": part(n_train), "validation": part(n_val)}


class RunListTests(unittest.TestCase):
    def test_full_sweep_has_270_unique_runs(self):
        runs = sweep_h4.build_runs([0, 1, 2], [0, 1, 2])
        names = [name for name, _, _ in runs]
        self.assertEqual(len(runs), 15 * 3 * 3 * 2)
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(sum(name.split("-")[2] == "null" for name in names), 135)

    def test_every_run_parses_with_its_setting(self):
        for name, split, argv in sweep_h4.build_runs([1], [2]):
            with self.subTest(run=name):
                args = train_cnngru.parse_args(["--data-dir", "/data"] + argv)
                self.assertEqual(args.run_name, name)
                self.assertEqual((args.split_position, args.seed, split), (1, 2, 1))
                self.assertEqual((args.epochs, args.patience), (200, 0))
                self.assertEqual(args.shuffle_train_labels, "-null-" in name)
        wd = train_cnngru.parse_args(
            ["--data-dir", "/data"] + dict((n, a) for n, _, a in sweep_h4.build_runs([0], [0]))
            ["h4-wd0.3-split0-seed0"])
        self.assertEqual(wd.weight_decay, 0.3)

    def test_shards_cover_every_run_once(self):
        runs = sweep_h4.build_runs([0, 1, 2], [0, 1, 2])
        shards = [sweep_h4.shard(runs, i, 8) for i in range(8)]
        self.assertEqual(sorted(sum(shards, [])), sorted(runs))
        self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)
        with self.assertRaises(ValueError):
            sweep_h4.shard(runs, 8, 8)

    def test_parse_shard(self):
        self.assertEqual(sweep_h4.parse_shard("3/8"), (3, 8))
        for text in ("8/8", "x/2", "1", "-1/4"):
            with self.subTest(text=text), self.assertRaises(Exception):
                sweep_h4.parse_shard(text)


class RunStateTests(unittest.TestCase):
    def test_done_partial_todo_and_clear(self):
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(sweep_h4, "RESULTS_DIR", Path(root)):
            (Path(root) / "h4-a").mkdir()
            (Path(root) / "h4-a" / "summary.json").write_text("{}")
            (Path(root) / "h4-b").mkdir()
            self.assertEqual([sweep_h4.run_state(n) for n in ("h4-a", "h4-b", "h4-c")],
                             ["done", "partial", "todo"])
            sweep_h4.clear_partial("h4-b")
            self.assertEqual(sweep_h4.run_state("h4-b"), "todo")
            (Path(root) / "phase1-x").mkdir()
            with self.assertRaises(ValueError):
                sweep_h4.clear_partial("phase1-x")


class InProcessRunTests(unittest.TestCase):
    """train_cnngru.run inside one process must match separate processes."""

    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        root = Path(self.root.name)
        # Random ResNet (no weight download) and a temporary results folder.
        self.patches = [
            mock.patch.object(train_cnngru, "CNNGRUModel",
                              functools.partial(CNNGRUModel, pretrained=False)),
            mock.patch.object(train_cnngru, "REPO_ROOT", root),
            mock.patch.object(train_cnngru, "RESULTS_DIR", root / "results"),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.root.cleanup()

    def args(self, name, *extra):
        return train_cnngru.parse_args(["--data-dir", "/data", "--split-position", "0",
                                        "--seed", "1", "--epochs", "3", "--run-name", name,
                                        *extra])

    def metrics(self, name):
        """metrics.csv rows without the wall-clock column."""
        with (train_cnngru.RESULTS_DIR / name / "metrics.csv").open(newline="") as handle:
            return [{k: v for k, v in row.items() if k != "seconds"}
                    for row in csv.DictReader(handle)]

    def test_run_after_other_runs_matches_run_alone(self):
        features = invented_split()
        snapshot = {p: {k: v.clone() for k, v in d.items()} for p, d in features.items()}
        train_cnngru.run(self.args("alone", "--feature-noise", "0.5"), features, echo=False)
        # Other runs first (one of them shuffles labels), as inside a worker.
        train_cnngru.run(self.args("other-null", "--shuffle-train-labels", "--dropout", "0.5"),
                         features, echo=False)
        train_cnngru.run(self.args("other", "--frame-dropout", "0.3"), features, echo=False)
        train_cnngru.run(self.args("after", "--feature-noise", "0.5"), features, echo=False)
        self.assertEqual(self.metrics("alone"), self.metrics("after"))
        for part, tensors in features.items():
            for key, value in tensors.items():
                self.assertTrue(torch.equal(value, snapshot[part][key]), (part, key))

    def test_summary_written_last_and_paths_kept_out(self):
        summary = train_cnngru.run(self.args("r"), invented_split(), echo=False)
        saved = json.loads((train_cnngru.RESULTS_DIR / "r" / "summary.json").read_text())
        self.assertEqual(saved["epochs_run"], 3)
        self.assertEqual(summary["epochs_run"], 3)
        self.assertNotIn("data_dir", saved["args"])
        self.assertNotIn("feature_cache", saved["args"])

    def test_cache_only_load_requires_cache(self):
        with self.assertRaises(ValueError):
            train_cnngru.load_features(self.args("x"), None, None, log=lambda line: None)


if __name__ == "__main__":
    unittest.main()
