"""Invented-input checks for the CNN-GRU training pieces (no dataset needed)."""

import math
import unittest

import torch

from model_cnngru import FEATURE_NORMS, CNNGRUModel, feature_norm_from_state_dict
from train_cnngru import (binary_metrics, class_weights, feature_scale_report, roc_auc,
                          train_model)


def tiny_model(dropout=0.3, feature_norm="none"):
    torch.manual_seed(0)
    # pretrained=False: random ResNet-18, no weight download.
    return CNNGRUModel(pretrained=False, dropout=dropout, feature_norm=feature_norm).eval()


class FeatureNormTests(unittest.TestCase):
    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            tiny_model(feature_norm="batch")

    def test_none_adds_no_state(self):
        model = tiny_model()
        self.assertFalse(any(k.startswith("feature_norm.") for k in model.state_dict()))
        with self.assertRaises(ValueError):
            model.set_feature_stats(torch.randn(2, 6, 512), torch.randn(2, 512))

    def test_layer_norm_centers_each_vector(self):
        model = tiny_model(feature_norm="layer")
        features = torch.rand(3, 6, 512) * 5 + 2  # positive, uncentered like post-ReLU
        out = model.feature_norm(features)
        self.assertTrue(torch.allclose(out.mean(dim=-1), torch.zeros(3, 6), atol=1e-4))
        self.assertTrue(torch.allclose(out.std(dim=-1, unbiased=False),
                                       torch.ones(3, 6), atol=1e-3))

    def test_zscore_is_identity_until_fitted(self):
        model = tiny_model(feature_norm="zscore")
        features = torch.rand(4, 512)
        self.assertTrue(torch.equal(model.feature_norm(features), features))

    def test_zscore_pools_history_and_goal(self):
        model = tiny_model(feature_norm="zscore")
        scale = torch.linspace(0.1, 10.0, 512)  # per-dimension scale spread 100x
        history = torch.rand(5, 6, 512) * scale + 3
        goal = torch.rand(5, 512) * scale + 3
        model.set_feature_stats(history, goal)
        pooled = model.feature_norm(torch.cat([history.reshape(-1, 512), goal]))
        self.assertTrue(torch.allclose(pooled.mean(dim=0), torch.zeros(512), atol=1e-3))
        self.assertTrue(torch.allclose(pooled.std(dim=0, unbiased=False),
                                       torch.ones(512), atol=1e-3))

    def test_zscore_constant_dim_keeps_unit_std(self):
        model = tiny_model(feature_norm="zscore")
        history, goal = torch.rand(3, 6, 512), torch.rand(3, 512)
        history[..., 7] = 0.0
        goal[:, 7] = 0.0
        model.set_feature_stats(history, goal)
        self.assertEqual(model.feature_norm.std[7].item(), 1.0)

    def test_checkpoint_round_trip_and_detection(self):
        for kind in FEATURE_NORMS:
            with self.subTest(kind=kind):
                model = tiny_model(feature_norm=kind)
                if kind == "zscore":
                    model.set_feature_stats(torch.rand(3, 6, 512) * 4, torch.rand(3, 512))
                state = model.state_dict()
                self.assertEqual(feature_norm_from_state_dict(state), kind)
                restored = tiny_model(feature_norm=kind)
                restored.load_state_dict(state)
                for key, value in state.items():
                    self.assertTrue(torch.equal(restored.state_dict()[key], value))

    def test_feature_scale_report(self):
        history = torch.zeros(2, 3, 4)
        history[..., 0] = torch.tensor([[1.0, 3.0, 1.0], [3.0, 1.0, 3.0]])
        history[..., 1] = torch.tensor([[10.0, 30.0, 10.0], [30.0, 10.0, 30.0]])
        goal = torch.tensor([[1.0, 10.0, 0.0, 0.0], [3.0, 30.0, 0.0, 0.0]])
        report = feature_scale_report(history, goal)
        self.assertEqual(report["constant_dims"], 2)
        self.assertAlmostEqual(report["std_min"], 1.0)
        self.assertAlmostEqual(report["std_max"], 10.0)
        self.assertAlmostEqual(report["std_ratio"], 10.0)
        self.assertAlmostEqual(report["zero_fraction"], 0.5)


class MetadataStatsTests(unittest.TestCase):
    def test_standardizes_with_training_stats(self):
        encoder = tiny_model().encoder
        depth = torch.tensor([[-10.0, -5.0, 0.0], [-20.0, -10.0, 0.0]])
        offsets = torch.tensor([[-3.0, -1.0, 0.0], [-6.0, -2.0, 0.0]])
        encoder.set_metadata_stats(depth, offsets)
        meta = torch.stack([depth, offsets], dim=-1).reshape(-1, 2)
        normalized = (meta - encoder.meta_mean) / encoder.meta_std
        self.assertTrue(torch.allclose(normalized.mean(dim=0), torch.zeros(2), atol=1e-5))
        self.assertTrue(torch.allclose(normalized.std(dim=0, unbiased=False),
                                       torch.ones(2), atol=1e-5))

    def test_constant_channel_keeps_unit_std(self):
        encoder = tiny_model().encoder
        encoder.set_metadata_stats(torch.zeros(4, 6), torch.randn(4, 6))
        self.assertEqual(encoder.meta_std[0].item(), 1.0)

    def test_stats_are_saved_in_state_dict(self):
        model = tiny_model()
        model.encoder.set_metadata_stats(torch.randn(4, 6) * 7 + 3, torch.randn(4, 6))
        restored = tiny_model()
        restored.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(restored.encoder.meta_mean, model.encoder.meta_mean))
        self.assertTrue(torch.equal(restored.encoder.meta_std, model.encoder.meta_std))


class ForwardPathTests(unittest.TestCase):
    def test_feature_path_matches_image_path(self):
        history = torch.randn(2, 6, 3, 32, 32)
        goal = torch.randn(2, 3, 32, 32)
        depth, offsets = torch.randn(2, 6), torch.randn(2, 6)
        for kind in FEATURE_NORMS:
            with self.subTest(kind=kind):
                model = tiny_model(feature_norm=kind)
                with torch.no_grad():
                    history_features = model.encoder.cnn(history)
                    goal_features = model.encoder.cnn(goal)
                    if kind == "zscore":
                        model.set_feature_stats(history_features, goal_features)
                    direct = model(history, depth, offsets, goal)
                    cached = model.forward_features(history_features, depth, offsets,
                                                    goal_features)
                self.assertEqual(tuple(direct.shape), (2, 2))
                self.assertTrue(torch.isfinite(direct).all())
                self.assertTrue(torch.allclose(direct, cached, atol=1e-5))

    def test_backbone_stays_frozen(self):
        model = tiny_model()
        model.train()
        self.assertFalse(model.encoder.cnn.backbone.training)
        self.assertFalse(any(p.requires_grad for p in model.encoder.cnn.parameters()))


class MetricTests(unittest.TestCase):
    def test_class_weights_inverse_frequency(self):
        weights = class_weights(torch.tensor([0, 1, 1, 1]))
        self.assertTrue(torch.allclose(weights, torch.tensor([2.0, 2.0 / 3.0])))
        with self.assertRaises(ValueError):
            class_weights(torch.tensor([1, 1]))

    def test_roc_auc(self):
        labels = torch.tensor([0, 0, 1, 1])
        self.assertEqual(roc_auc(torch.tensor([0.1, 0.2, 0.8, 0.9]), labels), 1.0)
        self.assertEqual(roc_auc(torch.tensor([0.9, 0.8, 0.2, 0.1]), labels), 0.0)
        self.assertEqual(roc_auc(torch.full((4,), 0.5), labels), 0.5)
        self.assertTrue(math.isnan(roc_auc(torch.tensor([0.1, 0.2]), torch.tensor([1, 1]))))

    def test_binary_metrics_known_confusion(self):
        # true: R R R A ; pred: R A A A
        labels = torch.tensor([0, 0, 0, 1])
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        metrics = binary_metrics(logits, labels, torch.ones(2))
        self.assertEqual(metrics["confusion"], [[1, 2], [0, 1]])
        self.assertAlmostEqual(metrics["acc"], 0.5)
        self.assertAlmostEqual(metrics["recall_retract"], 1 / 3)
        self.assertAlmostEqual(metrics["recall_advance"], 1.0)
        self.assertAlmostEqual(metrics["bal_acc"], (1 / 3 + 1.0) / 2)


class TrainLoopTests(unittest.TestCase):
    def test_learns_separable_invented_features(self):
        generator = torch.Generator().manual_seed(1)
        n = 40
        labels = torch.arange(n) % 2
        goal = torch.randn(n, 512, generator=generator)
        goal[:, 0] = labels.float() * 6 - 3  # label is readable from one goal feature
        data = {
            "history": torch.randn(n, 6, 512, generator=generator).abs(),
            "goal": goal,
            "depth": torch.randn(n, 6, generator=generator),
            "offsets": torch.randn(n, 6, generator=generator),
            "label": labels,
        }
        for kind in FEATURE_NORMS:
            with self.subTest(kind=kind):
                model = tiny_model(dropout=0.1, feature_norm=kind)
                model.encoder.set_metadata_stats(data["depth"], data["offsets"])
                if kind == "zscore":
                    model.set_feature_stats(data["history"], data["goal"])
                lines = []
                result = train_model(model, data, data, epochs=60, batch_size=8, lr=1e-2,
                                     weight_decay=1e-2, grad_clip=1.0, patience=60, seed=0,
                                     log=lines.append)
                self.assertEqual(result["best"]["val"]["bal_acc"], 1.0)
                self.assertEqual(len(lines), len(result["rows"]))
                self.assertIn("val loss", lines[0])

    def test_early_stopping_restores_best(self):
        model = tiny_model()
        n = 8
        data = {"history": torch.zeros(n, 6, 512), "goal": torch.zeros(n, 512),
                "depth": torch.zeros(n, 6), "offsets": torch.zeros(n, 6),
                "label": torch.arange(n) % 2}
        lines = []
        # lr=0 keeps weights fixed, so validation never improves after epoch 1.
        result = train_model(model, data, data, epochs=50, batch_size=4, lr=0.0,
                             weight_decay=0.0, grad_clip=1.0, patience=3, seed=0,
                             log=lines.append)
        self.assertLess(len(result["rows"]), 50)
        self.assertTrue(lines[-1].startswith("Early stop"))
        for key, value in result["best_state"].items():
            self.assertTrue(torch.equal(model.state_dict()[key], value))


if __name__ == "__main__":
    unittest.main()
