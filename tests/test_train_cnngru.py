"""Invented-input checks for the CNN-GRU training pieces (no dataset needed)."""

import math
import unittest

import torch

from model_cnngru import CAPACITY_RUNGS, CNNGRUModel, describe_capacity
from train_cnngru import (binary_metrics, class_weights, roc_auc, shuffled_labels,
                          train_model, window_summary)


def tiny_model(dropout=0.3, capacity="M"):
    torch.manual_seed(0)
    # pretrained=False: random ResNet-18, no weight download.
    return CNNGRUModel(pretrained=False, dropout=dropout, capacity=capacity).eval()


def trainable_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CapacityTests(unittest.TestCase):
    # Hand-computed head sizes (LayerNorm 1,024 + projection + GRU + MLP).
    EXPECTED_PARAMETERS = {"XS": 26_114, "S": 67_586, "M": 330_626, "L": 875_074}

    def test_rung_parameter_counts(self):
        for name, expected in self.EXPECTED_PARAMETERS.items():
            with self.subTest(capacity=name):
                self.assertEqual(trainable_count(tiny_model(capacity=name)), expected)

    def test_rung_m_matches_pre_ladder_head(self):
        # Same keys as checkpoints saved before the ladder (no projection
        # weights, MLP Linear layers at Sequential indices 1 and 4).
        keys = {k for k in tiny_model().state_dict() if not k.startswith("encoder.cnn.")}
        self.assertEqual(keys, {
            "encoder.meta_mean", "encoder.meta_std",
            "encoder.gru.weight_ih_l0", "encoder.gru.weight_hh_l0",
            "encoder.gru.bias_ih_l0", "encoder.gru.bias_hh_l0",
            "feature_norm.weight", "feature_norm.bias",
            "classifier.mlp.1.weight", "classifier.mlp.1.bias",
            "classifier.mlp.4.weight", "classifier.mlp.4.bias",
        })

    def test_every_rung_feature_path_matches_image_path(self):
        history = torch.randn(2, 6, 3, 32, 32)
        goal = torch.randn(2, 3, 32, 32)
        depth, offsets = torch.randn(2, 6), torch.randn(2, 6)
        for name in CAPACITY_RUNGS:
            with self.subTest(capacity=name), torch.no_grad():
                model = tiny_model(capacity=name)
                direct = model(history, depth, offsets, goal)
                cached = model.forward_features(model.encoder.cnn(history), depth, offsets,
                                                model.encoder.cnn(goal))
                self.assertEqual(tuple(direct.shape), (2, 2))
                self.assertTrue(torch.isfinite(direct).all())
                self.assertTrue(torch.allclose(direct, cached, atol=1e-5))

    def test_projection_is_shared_by_history_and_goal(self):
        model = tiny_model(capacity="S")
        self.assertEqual(tuple(model.projection.weight.shape), (64, 512))
        self.assertEqual(model.encoder.gru.input_size, 64 + 2)
        self.assertEqual(model.classifier.mlp[1].in_features, 64 + 64)

    def test_rung_l_structure(self):
        model = tiny_model(capacity="L")
        self.assertEqual((model.encoder.gru.num_layers, model.encoder.gru.dropout), (2, 0.2))
        self.assertEqual(model.classifier.mlp[1].in_features, 256 + 128)
        self.assertIsInstance(model.classifier.mlp[2], torch.nn.LayerNorm)
        self.assertEqual(sum(isinstance(m, torch.nn.LayerNorm) for m in model.classifier.mlp), 1)

    def test_describe_and_unknown_rung(self):
        self.assertIn("MLP 640 -> 128 -> 2", describe_capacity("M"))
        self.assertIn("MLP 384 -> 256 -> 64 -> 2", describe_capacity("L"))
        with self.assertRaises(ValueError):
            CNNGRUModel(pretrained=False, capacity="XL")

    def test_smallest_rung_learns_separable_invented_features(self):
        generator = torch.Generator().manual_seed(2)
        n = 40
        labels = torch.arange(n) % 2
        goal = torch.randn(n, 512, generator=generator)
        goal[:, :16] += (labels.float() * 6 - 3)[:, None]  # survives LayerNorm + projection
        data = {"history": torch.randn(n, 6, 512, generator=generator).abs(), "goal": goal,
                "depth": torch.randn(n, 6, generator=generator),
                "offsets": torch.randn(n, 6, generator=generator), "label": labels}
        model = tiny_model(dropout=0.1, capacity="XS")
        model.encoder.set_metadata_stats(data["depth"], data["offsets"])
        result = train_model(model, data, data, epochs=60, batch_size=8, lr=1e-2,
                             weight_decay=1e-2, grad_clip=1.0, patience=0, seed=0,
                             log=lambda line: None)
        self.assertEqual(result["best"]["val"]["bal_acc"], 1.0)


class FeatureNormTests(unittest.TestCase):
    def test_layer_norm_centers_each_vector(self):
        model = tiny_model()
        features = torch.rand(3, 6, 512) * 5 + 2  # positive, uncentered like post-ReLU
        out = model.feature_norm(features)
        self.assertTrue(torch.allclose(out.mean(dim=-1), torch.zeros(3, 6), atol=1e-4))
        self.assertTrue(torch.allclose(out.std(dim=-1, unbiased=False),
                                       torch.ones(3, 6), atol=1e-3))

    def test_layer_norm_is_trainable_and_saved(self):
        model = tiny_model()
        self.assertTrue(all(p.requires_grad for p in model.feature_norm.parameters()))
        with torch.no_grad():
            model.feature_norm.weight.mul_(2.0)
            model.feature_norm.bias.add_(0.5)
        restored = tiny_model()
        restored.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(restored.feature_norm.weight, model.feature_norm.weight))
        self.assertTrue(torch.equal(restored.feature_norm.bias, model.feature_norm.bias))


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
        model = tiny_model()
        history = torch.randn(2, 6, 3, 32, 32)
        goal = torch.randn(2, 3, 32, 32)
        depth, offsets = torch.randn(2, 6), torch.randn(2, 6)
        with torch.no_grad():
            direct = model(history, depth, offsets, goal)
            cached = model.forward_features(model.encoder.cnn(history), depth, offsets,
                                            model.encoder.cnn(goal))
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


class NullAndWindowTests(unittest.TestCase):
    def test_shuffled_labels_keep_counts_and_weights(self):
        labels = torch.tensor([0] * 30 + [1] * 70)
        shuffled = shuffled_labels(labels, seed=0)
        self.assertTrue(torch.equal(torch.bincount(shuffled), torch.bincount(labels)))
        self.assertTrue(torch.equal(class_weights(shuffled), class_weights(labels)))
        self.assertFalse(torch.equal(shuffled, labels))

    def test_shuffled_labels_seeded(self):
        labels = torch.arange(50) % 2
        self.assertTrue(torch.equal(shuffled_labels(labels, 3), shuffled_labels(labels, 3)))
        self.assertFalse(torch.equal(shuffled_labels(labels, 0), shuffled_labels(labels, 1)))

    def test_window_summary_last_epochs(self):
        rows = [{"epoch": e, "train_bal_acc": 1.0, "val_bal_acc": e / 10,
                 "val_auc": 0.5, "val_loss": float(e)} for e in range(1, 11)]
        summary = window_summary(rows, 4)  # epochs 7-10
        self.assertEqual((summary["window"], summary["first_epoch"],
                          summary["last_epoch"]), (4, 7, 10))
        self.assertAlmostEqual(summary["val_bal_acc"]["mean"], 0.85)
        self.assertAlmostEqual(summary["val_loss"]["std"], math.sqrt(1.25))
        self.assertEqual(summary["train_bal_acc"]["std"], 0.0)

    def test_window_summary_short_run_strings_and_nan(self):
        # metrics.csv rows come back as strings; a NaN AUC is skipped.
        rows = [{"epoch": "1", "train_bal_acc": "0.5", "val_bal_acc": "0.5",
                 "val_auc": "nan", "val_loss": "0.7"},
                {"epoch": "2", "train_bal_acc": "0.7", "val_bal_acc": "0.6",
                 "val_auc": "0.6", "val_loss": "0.6"}]
        summary = window_summary(rows, 20)
        self.assertEqual(summary["window"], 2)
        self.assertAlmostEqual(summary["val_auc"]["mean"], 0.6)
        self.assertEqual(summary["val_auc"]["std"], 0.0)
        only_nan = window_summary(rows[:1], 20)
        self.assertTrue(math.isnan(only_nan["val_auc"]["mean"]))
        with self.assertRaises(ValueError):
            window_summary([], 20)


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
        model = tiny_model(dropout=0.1)
        model.encoder.set_metadata_stats(data["depth"], data["offsets"])
        lines = []
        result = train_model(model, data, data, epochs=60, batch_size=8, lr=1e-2,
                             weight_decay=1e-2, grad_clip=1.0, patience=0, seed=0,
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

    def test_patience_zero_runs_all_epochs(self):
        model = tiny_model()
        n = 8
        data = {"history": torch.zeros(n, 6, 512), "goal": torch.zeros(n, 512),
                "depth": torch.zeros(n, 6), "offsets": torch.zeros(n, 6),
                "label": torch.arange(n) % 2}
        lines = []
        # Validation never improves after epoch 1, yet no early stop happens.
        result = train_model(model, data, data, epochs=12, batch_size=4, lr=0.0,
                             weight_decay=0.0, grad_clip=1.0, patience=0, seed=0,
                             log=lines.append)
        self.assertEqual(len(result["rows"]), 12)
        self.assertFalse(any(line.startswith("Early stop") for line in lines))
        self.assertEqual(result["best"]["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
