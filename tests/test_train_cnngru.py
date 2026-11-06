"""Invented-input checks for the CNN-GRU training pieces (no dataset needed)."""

import math
import tempfile
import unittest
from pathlib import Path

import torch

from model_cnngru import CNNGRUModel
from train_cnngru import (REPO_ROOT, binary_metrics, cache_metadata, cache_path,
                          check_cache_dir, class_weights, evaluate, parse_args, run,
                          load_cached_features, roc_auc, save_cached_features,
                          shuffled_labels, train_model, window_summary)


def tiny_model(dropout=0.3, **regularizers):
    torch.manual_seed(0)
    # pretrained=False: random ResNet-18, no weight download.
    return CNNGRUModel(pretrained=False, dropout=dropout, **regularizers).eval()


def invented_features(n=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {"history": torch.randn(n, 6, 512, generator=generator).abs(),
            "goal": torch.randn(n, 512, generator=generator),
            "depth": torch.randn(n, 6, generator=generator),
            "offsets": torch.randn(n, 6, generator=generator),
            "label": torch.arange(n) % 2}


def trainable_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class HeadTests(unittest.TestCase):
    def test_parameter_count(self):
        # LayerNorm 1,024 + projection 32,832 + GRU 25,344 + MLP 8,386.
        self.assertEqual(trainable_count(tiny_model()), 67_586)

    def test_state_dict_layout(self):
        # Retain the small head's existing parameter layout.
        keys = {k for k in tiny_model().state_dict() if not k.startswith("encoder.cnn.")}
        self.assertEqual(keys, {
            "encoder.meta_mean", "encoder.meta_std",
            "encoder.gru.weight_ih_l0", "encoder.gru.weight_hh_l0",
            "encoder.gru.bias_ih_l0", "encoder.gru.bias_hh_l0",
            "feature_norm.weight", "feature_norm.bias",
            "projection.weight", "projection.bias",
            "classifier.mlp.1.weight", "classifier.mlp.1.bias",
            "classifier.mlp.4.weight", "classifier.mlp.4.bias",
        })

    def test_projection_is_shared_by_history_and_goal(self):
        model = tiny_model()
        self.assertEqual(tuple(model.projection.weight.shape), (64, 512))
        self.assertEqual(model.encoder.gru.input_size, 64 + 2)
        self.assertEqual(model.encoder.gru.num_layers, 1)
        self.assertEqual(model.classifier.mlp[1].in_features, 64 + 64)


class RegularizerTests(unittest.TestCase):
    ALL_ON = {"feature_noise": 0.5}

    def forward(self, model, data):
        return model.forward_features(data["history"], data["depth"], data["offsets"],
                                      data["goal"])

    def test_defaults_draw_no_random_numbers(self):
        # Disabled feature noise returns its inputs without drawing randomness.
        model = tiny_model().train()
        history, goal = torch.randn(4, 6, 512), torch.randn(4, 512)
        torch.manual_seed(5)
        out_history, out_goal = model.regularize(history, goal)
        after = torch.rand(3)
        self.assertIs(out_history, history)
        self.assertIs(out_goal, goal)
        torch.manual_seed(5)
        self.assertTrue(torch.equal(after, torch.rand(3)))

    def test_inactive_in_eval_mode(self):
        data = invented_features()
        plain, regularized = tiny_model(), tiny_model(**self.ALL_ON)
        with torch.no_grad():
            self.assertTrue(torch.equal(self.forward(plain, data),
                                        self.forward(regularized, data)))

    def test_active_in_training_mode(self):
        model = tiny_model(dropout=0.0, **self.ALL_ON).train()
        data = invented_features()
        with torch.no_grad():
            first, second = self.forward(model, data), self.forward(model, data)
        self.assertFalse(torch.equal(first, second))

    def test_no_new_parameters_or_state(self):
        self.assertEqual(set(tiny_model().state_dict()),
                         set(tiny_model(**self.ALL_ON).state_dict()))

    def test_invalid_settings_rejected(self):
        for value in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CNNGRUModel(pretrained=False, feature_noise=value)


class FeatureCacheTests(unittest.TestCase):
    def test_round_trip_and_metadata_check(self):
        features = invented_features()
        metadata = cache_metadata("/data/copy", 2, "train", 518)
        with tempfile.TemporaryDirectory() as root:
            path = cache_path(root, metadata)
            self.assertIsNone(load_cached_features(path, metadata))
            save_cached_features(path, features, metadata)
            loaded = load_cached_features(path, metadata)
            for key, value in features.items():
                self.assertTrue(torch.equal(loaded[key], value))
            other = cache_metadata("/data/copy", 2, "train", 224)
            save_cached_features(path, features, other)  # same name, wrong content
            with self.assertRaises(ValueError):
                load_cached_features(path, metadata)

    def test_file_name_separates_split_partition_and_size(self):
        names = {cache_path("/c", cache_metadata("/d", split, part, size)).name
                 for split in (0, 1) for part in ("train", "validation") for size in (224, 518)}
        self.assertEqual(len(names), 8)

    def test_cache_inside_repository_refused(self):
        with self.assertRaises(ValueError):
            check_cache_dir(REPO_ROOT / "cache")
        with self.assertRaises(ValueError):
            check_cache_dir(REPO_ROOT)
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(check_cache_dir(root), Path(root).resolve())


class RunTests(unittest.TestCase):
    def test_presets_and_explicit_zero_overrides(self):
        for config, noise, epochs in (("baseline", 0.0, 200), ("moderate", 1.0, 50),
                                      ("strict", 2.0, 50)):
            with self.subTest(config=config):
                argv = ["--data-dir", "/invented", "--split-position", "0", "--config", config]
                args = parse_args(argv)
                self.assertEqual((args.weight_decay, args.feature_noise, args.epochs),
                                 (1.0, noise, epochs))
                custom = parse_args(argv + ["--weight-decay", "0", "--feature-noise", "0",
                                            "--epochs", "3"])
                self.assertEqual((custom.weight_decay, custom.feature_noise, custom.epochs),
                                 (0.0, 0.0, 3))

    def test_configs_keep_identical_model_and_increase_training_noise_only(self):
        models = []
        for config in ("moderate", "strict"):
            args = parse_args(["--data-dir", "/invented", "--split-position", "0",
                               "--config", config])
            models.append(tiny_model(feature_noise=args.feature_noise))
        moderate, strict = models
        self.assertEqual(trainable_count(moderate), trainable_count(strict))
        for key, value in moderate.state_dict().items():
            self.assertTrue(torch.equal(value, strict.state_dict()[key]))
        data = invented_features()
        def forward(model):
            return model.forward_features(data["history"], data["depth"], data["offsets"], data["goal"])
        with torch.no_grad():
            self.assertTrue(torch.equal(forward(moderate), forward(strict)))
        perturbed = []
        for model in models:
            model.train()
            torch.manual_seed(7)
            perturbed.append(model.regularize(data["history"], data["goal"]))
        for i, original in enumerate((data["history"], data["goal"])):
            self.assertTrue(torch.allclose(perturbed[1][i] - original,
                                           2 * (perturbed[0][i] - original), atol=1e-6))

    def test_terminal_run_is_reproducible_without_writing_artifacts(self):
        from unittest import mock
        import functools
        args = parse_args(["--data-dir", "/invented", "--split-position", "0",
                           "--epochs", "3", "--config", "strict"])
        self.assertEqual(args.weight_decay, 1.0)
        features = {"train": invented_features(16), "validation": invented_features(8, seed=1)}
        snapshot = {p: {k: v.clone() for k, v in data.items()} for p, data in features.items()}
        with mock.patch("train_cnngru.CNNGRUModel", functools.partial(CNNGRUModel, pretrained=False)), \
                mock.patch("torch.save") as save, \
                mock.patch.object(Path, "mkdir") as mkdir, \
                mock.patch("builtins.open", side_effect=AssertionError("Unexpected disk access")):
            first = run(args, features, echo=False)
            args.shuffle_train_labels = True
            run(args, features, echo=False)
            args.shuffle_train_labels = False
            second = run(args, features, echo=False)
        save.assert_not_called()
        mkdir.assert_not_called()
        clean = lambda result: [{k: v for k, v in row.items() if k != "seconds"}
                                for row in result["rows"]]
        self.assertEqual(clean(first), clean(second))
        for part, data in features.items():
            for key, value in data.items():
                self.assertTrue(torch.equal(value, snapshot[part][key]))
        self.assertEqual(first["final_epoch"]["epoch"], 3)
        self.assertEqual(first["model"].feature_noise, 2.0)
        self.assertAlmostEqual(first["generalization_gap"],
                               first["last_window"]["train_bal_acc"]["mean"] -
                               first["last_window"]["val_bal_acc"]["mean"])


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
        goal[:, :16] += (labels.float() * 6 - 3)[:, None]  # survives LayerNorm + projection
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
                             weight_decay=1e-2, grad_clip=1.0, seed=0,
                             log=lines.append)
        self.assertEqual(result["rows"][-1]["val_bal_acc"], 1.0)
        self.assertEqual(len(lines), len(result["rows"]))
        self.assertIn("val loss", lines[0])

    def test_runs_all_epochs_and_retains_final_weights(self):
        model = tiny_model()
        n = 8
        data = {"history": torch.zeros(n, 6, 512), "goal": torch.zeros(n, 512),
                "depth": torch.zeros(n, 6), "offsets": torch.zeros(n, 6),
                "label": torch.arange(n) % 2}
        lines = []
        # Validation never improves after epoch 1, yet no early stop happens.
        result = train_model(model, data, data, epochs=12, batch_size=4, lr=0.0,
                             weight_decay=0.0, grad_clip=1.0, seed=0,
                             log=lines.append)
        self.assertEqual(len(result["rows"]), 12)
        self.assertFalse(any(line.startswith("Early stop") for line in lines))
        final = evaluate(model, data, class_weights(data["label"]))
        self.assertEqual(final["loss"], result["rows"][-1]["val_loss"])
        self.assertNotIn("best_state", result)


if __name__ == "__main__":
    unittest.main()
