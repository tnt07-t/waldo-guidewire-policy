"""Invented-array evaluation checks, independent of real images and weights."""

from types import SimpleNamespace
import unittest

import numpy as np

from evaluation import (check_partitions, design_matrix, episode_interval,
                        benchmark_encoders, collect, diagnostic_notes, evaluate, fit_probe, metrics)
from evaluation import (all_splits_summary, control_lines, grouped_fold_index, inner_cv,
                        label_permutations, shuffled_label_control)


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


def separable_features(episodes=10):
    """Two cases per episode, one per class; the latest frame carries the label."""
    y = np.array([0, 1] * episodes)
    return {"y": y, "episode": np.repeat(["e" + str(i) for i in range(episodes)], 2),
            "metadata": np.zeros((2 * episodes, 12)),
            "history": np.tile(y[:, None, None], (1, 6, 3)).astype(float)
            + np.linspace(0, 0.1, 2 * episodes)[:, None, None],
            "goal": np.zeros((2 * episodes, 3))}


class ControlAndSelectionTests(unittest.TestCase):
    def test_label_permutations_keep_class_counts_and_are_seeded(self):
        y = np.array([0] * 7 + [1] * 13)
        permutations = label_permutations(y, 5, seed=3)
        self.assertEqual(len(permutations), 5)
        for permuted in permutations:
            self.assertEqual(np.bincount(permuted).tolist(), [7, 13])
        self.assertFalse(all(np.array_equal(p, y) for p in permutations))
        for again, first in zip(label_permutations(y, 5, seed=3), permutations):
            np.testing.assert_array_equal(again, first)

    def test_shuffled_label_control_reports_distribution(self):
        train, val = separable_features(), separable_features()
        result = shuffled_label_control(design_matrix(train, "latest_goal"), train["y"],
                                        design_matrix(val, "latest_goal"), val["y"],
                                        permutations=3, seed=0)
        self.assertEqual(len(result["validation_balanced_accuracy"]), 3)
        self.assertTrue(all(0 <= s <= 1 for s in result["validation_balanced_accuracy"]))
        self.assertLessEqual(result["mean"], result["max"])

    def test_grouped_folds_never_share_an_episode(self):
        episodes = np.array(["e{}".format(i // 3) for i in range(60)])
        fold = grouped_fold_index(episodes, 5)
        self.assertEqual(sorted(set(fold.tolist())), [0, 1, 2, 3, 4])
        for k in range(5):
            scored = set(episodes[fold == k])
            fitted = set(episodes[fold != k])
            self.assertFalse(scored & fitted)
            # Inner C selection inside rotation k must not see fold k either.
            for j in set(range(5)) - {k}:
                inner_fit = set(episodes[(fold != k) & (fold != j)])
                self.assertFalse(inner_fit & scored)
                self.assertFalse(inner_fit & set(episodes[fold == j]))

    def test_inner_cv_scores_each_fold(self):
        result = inner_cv(separable_features(), "latest_goal", n_splits=5)
        self.assertEqual(len(result["fold_balanced_accuracy"]), 5)
        self.assertEqual(len(result["selected_C"]), 5)
        self.assertEqual(result["mean"], 1.0)
        self.assertEqual(result["std"], 0.0)

    def test_all_splits_summary_and_control_lines(self):
        def result(score):
            return {"validation": {"balanced_accuracy": score}}
        summary = all_splits_summary({0: {"a:latest_goal": result(.5)},
                                      1: {"a:latest_goal": result(.7)}})
        self.assertEqual(summary["a:latest_goal"]["validation_balanced_accuracy"],
                         {"0": .5, "1": .7})
        self.assertAlmostEqual(summary["a:latest_goal"]["mean"], .6)
        controls = {"shuffled_labels": {"a:latest_goal": {"mean": .5, "max": .55,
                                                          "permutations": 20}},
                    "random_center": {"results": {"a:latest_goal": result(.52)}}}
        lines = control_lines(controls, {"a:latest_goal": result(.7)})
        self.assertEqual(len(lines), 2)
        self.assertIn("real BA 0.700", lines[0])
        self.assertIn("random_center", lines[1])


class WirePreprocessorEvaluationTests(unittest.TestCase):
    """Invented cases through the real WirePreprocessor and an untrained ResNet."""

    @staticmethod
    def wire_case(row):
        from PIL import Image

        frame = Image.new("L", (96, 96), 100 + row)
        return {"goal": Image.new("RGB", (96, 96), (255, 0, 0)), "goal_raw": frame,
                "goal_xy": [48.0, 48.0], "history": [frame] * 6,
                "depth_relative_mm": [-5, -4, -3, -2, -1, 0],
                "history_offsets_s": [-4.0, -3.2, -2.4, -1.6, -0.8, 0.0],
                "label": row % 2, "episode": "invented", "case_id": "invented-" + str(row),
                "row_index": row}

    def test_collect_with_wire_preprocessor_and_frame_encoder(self):
        import torch
        from cnn import FrameEncoder
        from evaluation import static_images
        from preprocessing import WirePreprocessor

        cases = [self.wire_case(0), self.wire_case(1)]
        encoder = FrameEncoder(pretrained=False).eval()
        for channels in ("raw", "enhanced"):
            with self.subTest(channels=channels):
                prepare = WirePreprocessor(crop_size=64, channels=channels)
                shared, features = collect(cases, prepare, {"resnet18": encoder})
                self.assertEqual(features["resnet18"]["goal"].shape, (2, 512))
                self.assertEqual(features["resnet18"]["history"].shape, (2, 6, 512))
                self.assertTrue(np.isfinite(features["resnet18"]["history"]).all())
                self.assertEqual(shared["metadata"].shape, (2, 12))
                images, prepared_count = static_images(prepare, cases[0])
                self.assertEqual((tuple(images.shape), prepared_count), ((2, 3, 64, 64), 7))
                prepared = prepare.prepare_case(cases[0])
                self.assertTrue(torch.equal(images[1], prepared["history"][-1]))
        timing = benchmark_encoders(cases[0], prepare, {"resnet18": encoder}, "cpu",
                                    warmup=1, repeats=2)
        self.assertEqual(timing["resnet18"]["prepared_images_per_case"], 7)


if __name__ == "__main__":
    unittest.main()
