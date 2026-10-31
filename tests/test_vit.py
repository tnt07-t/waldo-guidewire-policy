"""Test extraction bookkeeping with an invented backbone, never real weights."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from preprocessing import CHECKPOINT, MEAN, STD
from vit import DINOv2Encoder


# Echo image markers as features to test wiring without real weights.
class InventedBackbone(torch.nn.Module):
    num_features = 384

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward_features(self, images):
        self.calls.append((len(images), self.training, torch.is_grad_enabled()))
        return images[:, 0, 0, 0].reshape(-1, 1, 1).expand(-1, 2, 384)

    def forward_head(self, tokens, pre_logits=False):
        assert pre_logits
        return tokens[:, 0]


def invented_inputs(batch=1):
    # Separate goal/history markers expose mixing or reordering across cases.
    goal = torch.arange(batch, dtype=torch.float32).add(100)
    history = torch.arange(batch * 6, dtype=torch.float32).reshape(batch, 6)
    return (goal[:, None, None, None].expand(-1, 3, 518, 518),
            history[:, :, None, None, None].expand(-1, -1, 3, 518, 518))


class EncoderTests(unittest.TestCase):
    def test_single_case_separation_order_and_inference(self):
        backbone = InventedBackbone()
        encoder = DINOv2Encoder(backbone=backbone)
        backbone.train()
        goals, histories = invented_inputs()
        result = encoder.encode(goals[0], histories[0])
        self.assertEqual(result["goal_features"].shape, (384,))
        self.assertEqual(result["history_features"].shape, (6, 384))
        self.assertEqual(result["goal_features"][0].item(), 100)
        torch.testing.assert_close(result["history_features"][:, 0], torch.arange(6.))
        self.assertEqual(backbone.calls, [(1, False, False)] * 7)
        self.assertFalse(result["history_features"].requires_grad)

    def test_two_cases_chunking_matches_individual_encoding(self):
        encoder = DINOv2Encoder(backbone=InventedBackbone())
        goals, histories = invented_inputs(2)
        # Uneven chunks exercise the final partial group and case reconstruction.
        grouped = encoder.encode(goals, histories, image_chunk_size=5)
        self.assertEqual(grouped["history_features"].shape, (2, 6, 384))
        for i in range(2):
            individual = encoder.encode(goals[i], histories[i])
            for key in individual:
                torch.testing.assert_close(grouped[key][i], individual[key], rtol=0, atol=0)

    def test_invalid_inputs_rejected(self):
        encoder = DINOv2Encoder(backbone=InventedBackbone())
        goals, histories = invented_inputs()
        for size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                encoder.encode(goals, histories, image_chunk_size=size)
        with self.assertRaises(ValueError):
            encoder.encode(goals, histories[:, :5])
        with self.assertRaises(ValueError):
            encoder.encode(goals[..., :224, :224], histories)
        with self.assertRaises(ValueError):
            encoder.encode(goals.double(), histories)
        with self.assertRaises(ValueError):
            encoder.encode(goals * float("nan"), histories)

    def test_non_finite_features_rejected(self):
        backbone = InventedBackbone()
        backbone.forward_head = lambda tokens, **kwargs: tokens[:, 0] * float("nan")
        goals, histories = invented_inputs()
        with self.assertRaisesRegex(ValueError, "non-finite features"):
            DINOv2Encoder(backbone=backbone).encode(goals, histories)

    def test_pretrained_loader_and_config_guard_without_network(self):
        calls = []
        config = {"input_size": (3, 518, 518), "mean": MEAN, "std": STD}
        def create_model(name, **kwargs):
            calls.append((name, kwargs))
            return InventedBackbone()
        stub = SimpleNamespace(create_model=create_model,
                               data=SimpleNamespace(resolve_model_data_config=lambda _: config))
        # Mock the loader, not evidence of actual checkpoint compatibility.
        with patch.dict(sys.modules, {"timm": stub}):
            DINOv2Encoder()
            self.assertEqual(calls, [(CHECKPOINT, {"pretrained": True, "num_classes": 0})])
            config["mean"] = (0., 0., 0.)
            with self.assertRaisesRegex(ValueError, "preparation differs"):
                DINOv2Encoder()


if __name__ == "__main__":
    unittest.main()
