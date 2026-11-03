"""Invented-feature checks; no real data or pretrained weights."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from torch import nn

from temporal_models import FeatureHead, FrozenPolicy
from temporal_evaluation import metadata_scale, tensors, fit, probabilities, main, benchmark, experiment_variants
from compare_runs import summarize


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def data(self, n=8):
        torch.manual_seed(4)
        return (torch.randn(n, 6, 8), torch.randn(n, 8), torch.randn(n, 6, 2),
                torch.arange(n) % 2)

    def test_all_heads_backward_and_case_independence(self):
        h, g, m, _ = self.data()
        for kind in ("latest_mlp", "mean_mlp", "attention", "gru", "lstm"):
            head = FeatureHead(kind, 8, 8, 0).eval()
            out = head(h, g, m)
            self.assertEqual(out.shape, (8, 2))
            torch.testing.assert_close(out[:1], head(h[:1], g[:1], m[:1]))
            out.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_requested_sweep_and_no_position_control(self):
        variants = experiment_variants(SimpleNamespace(attention_sweep=True))
        self.assertEqual(len(variants), 8)
        self.assertEqual(len({v['name'] for v in variants}), 8)
        h, g, m, _ = self.data(2)
        for v in variants:
            head = FeatureHead(v['kind'], 8, v['width'], 0,
                               position_embeddings=v['position_embeddings'], classifier_width=64).eval()
            self.assertEqual(head.classifier[0].out_features, 64)
            out = head(h, g, m)
            out.square().sum().backward()
            self.assertTrue(all(p.grad is not None for p in head.parameters()))
            if v['kind'] == 'attention' and not v['position_embeddings']:
                self.assertIsNone(head.position)
                # Earlier aligned pairs are permutation-equivariant without
                # learned positions. Latest query and time values are retained.
                order = [4, 3, 2, 1, 0, 5]
                torch.testing.assert_close(out, head(h[:, order], g, m[:, order]))

    def test_attention_uses_all_steps_goal_metadata_and_order(self):
        h, g, m, _ = self.data(2)
        h.requires_grad_(); g.requires_grad_(); m.requires_grad_()
        head = FeatureHead("attention", 8, 8, 0).eval()
        original = head(h, g, m)
        original.square().sum().backward()
        self.assertTrue((h.grad.abs().sum((0, 2)) > 0).all())
        self.assertGreater(g.grad.abs().sum(), 0)
        self.assertTrue((m.grad.abs().sum((0, 1)) > 0).all())
        # Preserve the latest step; permute preceding aligned image/metadata pairs.
        order = [4, 3, 2, 1, 0, 5]
        self.assertFalse(torch.allclose(original, head(h[:, order], g, m[:, order])))

    def test_latest_ignores_old_images_but_uses_historical_metadata(self):
        h, g, m, _ = self.data()
        head = FeatureHead("latest_mlp", 8, 8, 0).eval()
        other = h.clone(); other[:, :-1] += 10
        torch.testing.assert_close(head(h, g, m), head(other, g, m))
        self.assertFalse(torch.allclose(head(h, g, m), head(h, g, m + 10)))

    def test_train_only_scaling_and_metadata_alignment(self):
        import numpy as np
        f = {"history": np.zeros((2, 6, 8)), "goal": np.zeros((2, 8)),
             "metadata": np.array([list(range(6)) + list(range(10, 16))] * 2), "y": [0, 1]}
        data = tensors(f, "cpu")
        torch.testing.assert_close(data[2][0, 2], torch.tensor([2., 12.]))
        scale = metadata_scale(data[2])
        self.assertTrue(torch.isfinite(scale).all())
        torch.testing.assert_close(metadata_scale(torch.zeros(2, 6, 2)), torch.ones(2))
        saved = scale.clone()
        _ = (data[2] + 1000) / scale
        torch.testing.assert_close(scale, saved)

    def test_actual_training_checkpoint_reload(self):
        data = self.data(16)
        data[0][:, -1, 0] = data[3].float() * 6 - 3
        head = FeatureHead("latest_mlp", 8, 8, 0)
        result = fit(head, data, data, epochs=60, patience=15, lr=.03, weight_decay=0, seed=1)
        self.assertGreater(result["best_epoch"], 0)
        self.assertGreater(((probabilities(head, data) >= .5) == data[3].numpy()).mean(), .9)
        fresh = FeatureHead("latest_mlp", 8, 8, 0)
        fresh.load_state_dict(head.state_dict())
        torch.testing.assert_close(torch.tensor(probabilities(head, data)), torch.tensor(probabilities(fresh, data)))

    def test_frozen_policy_keeps_backbone_fixed(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)
                self.bn = nn.BatchNorm1d(8)
            def forward(self, x):
                shape = x.shape
                return self.bn(self.linear(x.reshape(-1, 8))).reshape(shape)
        encoder = Encoder()
        policy = FrozenPolicy(encoder, FeatureHead("attention", 8, 8, 0)).train()
        before = {k: v.clone() for k, v in encoder.state_dict().items()}
        h, g, m, _ = self.data()
        policy(g, h, m).sum().backward()
        self.assertFalse(encoder.training)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        for k, v in encoder.state_dict().items():
            torch.testing.assert_close(v, before[k])

    def test_ledger_reads_static_and_temporal_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(2):
                p = Path(directory) / (str(i) + '.json')
                p.write_text(json.dumps({"split_position": 0, "results": {"example": {
                    "validation": {"balanced_accuracy": .6, "retract_recall": .5, "advance_recall": .7}}}}))
                paths.append(p)
            result = summarize(paths)
            self.assertIn("0.600", result)
            self.assertIn("No automatic ranking", result)

    def test_full_runner_invented_images_and_private_reports(self):
        from PIL import Image
        import os

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(3, 512)
            def forward(self, x):
                return self.backbone(x.mean((-1, -2)))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"; data.mkdir()
            rows = []
            for i in range(16):
                name = str(i) + '.png'
                Image.new('RGB', (8, 8), (i * 12, 40, 70)).save(data / name)
                rows.append({"paths": [name] * 7, "depth_relative_mm": [-5, -4, -3, -2, -1, 0],
                             "history_offsets_s": [-5, -4, -3, -2, -1, 0], "label": i % 2,
                             "case_id": 'invented-' + str(i), "episode": 'invented-' + str(i)})
            (data / 'cases.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (data / 'splits.json').write_text(json.dumps([{"fold": 0, "train": list(range(8)),
                                                         "validation": list(range(8, 12)), "test": list(range(12, 16))}]))
            output = root / 'run'
            with patch.dict(os.environ, {"PSCRATCH": str(root)}), patch.dict(
                    'sys.modules', {"cnn": SimpleNamespace(FrameEncoder=Encoder)}):
                main(['--data-dir', str(data), '--output-dir', str(output), '--split-position', '0',
                      '--device', 'cpu', '--epochs', '2', '--patience', '1', '--attention-sweep'])
            report = json.loads((output / 'metrics.json').read_text())
            self.assertTrue(report['encoder_unchanged'])
            self.assertEqual(len(report['results']), 9)
            self.assertTrue((output / 'attention-w32-pos0-seed0.pt').exists())
            checkpoint = torch.load(output / 'attention-w32-pos0-seed0.pt', weights_only=True)
            head = FeatureHead('attention', 512, 32, position_embeddings=False, classifier_width=64)
            head.load_state_dict(checkpoint['head_state'])
            private = json.loads((output / 'validation_predictions.private.json').read_text())
            self.assertEqual(len(private['y']), 4)
            self.assertEqual(private['row_index'], [8, 9, 10, 11])
            self.assertEqual(json.loads((output / 'status.json').read_text())['status'], 'completed')

    def test_benchmark_counts_images_for_static_and_attention(self):
        from PIL import Image
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.seen = 0
            def forward(self, x):
                self.seen += len(x)
                return x.mean((-1, -2)).repeat(1, 3)[:, :8]
        class Prepare:
            def prepare_image(self, image):
                return {'tensor': torch.ones(3, 8, 8)}
        case = {'goal': Image.new('RGB', (8, 8)), 'history': [Image.new('RGB', (8, 8))] * 6,
                'depth_relative_mm': [0] * 6, 'history_offsets_s': [-5, -4, -3, -2, -1, 0]}
        for kind, images in [('latest_mlp', 2), ('attention', 7)]:
            encoder = Encoder()
            report = benchmark(case, Prepare(), encoder, FeatureHead(kind, 8, 8, 0),
                               torch.ones(2), 'cpu', warmup=1, repeats=2)
            self.assertEqual(encoder.seen, images * 3)
            self.assertEqual(report['images'], images)


if __name__ == "__main__":
    unittest.main()
