"""Invented-array evaluation checks, independent of real images and weights."""

from types import SimpleNamespace
import unittest

import numpy as np

from evaluation import (check_partitions, design_matrix, episode_interval,
                        benchmark_encoders, collect, diagnostic_notes, evaluate, fit_probe, metrics)


class EvaluationTests(unittest.TestCase):
    def test_metrics_expose_majority_failure(self):
        result = metrics([0, 0, 1, 1, 1], [1, 1, 1, 1, 1])
        self.assertEqual(result["accuracy"], 0.6)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertEqual(result["confusion_matrix"], [[0, 2], [0, 3]])
        with self.assertRaises(ValueError):
            metrics([0, 1], [0, float("nan")])

    def test_episode_and_row_leakage_rejected(self):
        records = [{"episode": str(i), "label": i % 2} for i in range(6)]
        parts = {name: SimpleNamespace(indices=indices, records=records) for name, indices in
                 (("train", [0, 1]), ("validation", [2, 3]), ("test", [4, 5]))}
        check_partitions(parts)
        records[2]["episode"] = records[0]["episode"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            check_partitions(parts)
        records[2]["episode"] = "2"
        parts["validation"].indices = [0, 3]
        with self.assertRaisesRegex(ValueError, "leakage"):
            check_partitions(parts)

    def test_scaler_never_fits_validation(self):
        x = np.array([[-2.], [-1.], [1.], [2.]])
        model, trials = fit_probe(x, [0, 0, 1, 1], x + 100, [0, 0, 1, 1])
        np.testing.assert_equal(model[0].mean_, [0.])
        self.assertEqual(len(trials), 3)
        # Equal validation scores use the smallest C, i.e. strongest penalty.
        self.assertEqual(model[-1].C, 0.01)

    def test_feature_roles_and_order(self):
        f = {"history": np.arange(24).reshape(2, 6, 2),
             "goal": np.full((2, 2), 99), "metadata": np.zeros((2, 12))}
        np.testing.assert_equal(design_matrix(f, "latest"), [[10, 11], [22, 23]])
        np.testing.assert_equal(design_matrix(f, "mean_goal")[:, :2], [[5, 6], [17, 18]])
        self.assertEqual(design_matrix(f, "latest_goal_metadata").shape, (2, 16))
        f["goal"][0, 0] = 12
        self.assertEqual(design_matrix(f, "goal_only")[0, 0], 12)

    def test_episode_interval_handles_correlated_cases(self):
        result = episode_interval([0, 1, 0, 1], [0.1, .9, .1, .9],
                                  ["a", "a", "b", "b"], repeats=100)
        self.assertEqual(result["interval"], [1., 1.])
        self.assertEqual(result["valid_resamples"], 100)

    def test_complete_synthetic_comparison(self):
        def data(prefix):
            y = np.array([0, 1] * 8)
            return {"y": y, "metadata": np.tile(y[:, None], (1, 12)),
                    "episode": np.array([prefix + str(i // 2) for i in range(16)]),
                    "history": np.tile(y[:, None, None], (1, 6, 3)),
                    "goal": np.zeros((16, 3))}
        train, val = data("train"), data("val")
        results, predictions = evaluate(train, val, {"fake": train}, {"fake": val}, repeats=100)
        self.assertEqual(len(results), 7)
        self.assertEqual(results["fake:latest_goal"]["validation"]["balanced_accuracy"], 1.)
        self.assertEqual(results["fake:goal_only"]["validation"]["balanced_accuracy"], .5)
        self.assertEqual(len(predictions["metadata"]), 16)
        self.assertTrue(any("goal_only" in note for note in diagnostic_notes(results)))

    def test_collection_and_benchmark_use_separate_goal_and_history(self):
        import torch

        class FakeEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.tensor(0.))
                self.seen = []

            def forward(self, images):
                self.seen.extend(images[:, 0, 0, 0].tolist())
                return images[:, :, 0, 0]

        class Preparation:
            def prepare_image(self, image):
                return {"tensor": torch.full((3, 2, 2), float(image))}

            def prepare_case(self, case):
                return {"goal": self.prepare_image(case["goal"])["tensor"],
                        "history": torch.stack([self.prepare_image(i)["tensor"]
                                                for i in case["history"]])}

        case = {"goal": 99, "history": list(range(6)), "depth_relative_mm": [0] * 6,
                "history_offsets_s": [-5, -4, -3, -2, -1, 0], "label": 1,
                "episode": "invented", "case_id": "invented", "row_index": 0}
        encoder, prepare = FakeEncoder(), Preparation()
        shared, features = collect([case], prepare, {"resnet18": encoder})
        self.assertEqual(encoder.seen, [99, 0, 1, 2, 3, 4, 5])
        self.assertEqual(features["resnet18"]["history"].shape, (1, 6, 3))
        self.assertEqual(shared["metadata"].shape, (1, 12))
        encoder.seen.clear()
        timing = benchmark_encoders(case, prepare, {"resnet18": encoder}, "cpu",
                                    warmup=1, repeats=3)
        self.assertEqual(encoder.seen, [99, 5] * 4)
        self.assertIsNone(timing["resnet18"]["peak_cuda_bytes"])


if __name__ == "__main__":
    unittest.main()
