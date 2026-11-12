"""Invented-input checks for goal_classifier goal→ResNet encode helper."""

import unittest

import torch
from torch import nn

from guidewire.cnn_gru.goal_classifier import GOAL_FEATURE_DIM, encode_goal


class FakeEncoder(nn.Module):
    """Deterministic stand-in for FrameEncoder without torchvision."""

    feature_dim = GOAL_FEATURE_DIM

    def forward(self, images):
        batch = images.shape[0]
        features = torch.zeros(batch, self.feature_dim, dtype=images.dtype)
        features[:, 0] = images.mean(dim=(1, 2, 3))
        return features


class MlpCnnGoalTests(unittest.TestCase):
    def test_encode_goal_single_and_batch(self):
        encoder = FakeEncoder().eval()
        out_single = encode_goal(encoder, torch.randn(3, 56, 56))
        out_batch = encode_goal(encoder, torch.randn(2, 3, 56, 56))
        self.assertEqual(tuple(out_single.shape), (GOAL_FEATURE_DIM,))
        self.assertEqual(tuple(out_batch.shape), (2, GOAL_FEATURE_DIM))
        self.assertTrue(torch.isfinite(out_single).all())
        self.assertTrue(torch.isfinite(out_batch).all())

    def test_encode_goal_rejects_bad_shapes(self):
        encoder = FakeEncoder().eval()
        with self.assertRaises(ValueError):
            encode_goal(encoder, torch.randn(1, 6, 3, 56, 56))
        with self.assertRaises(ValueError):
            encode_goal(encoder, torch.randn(56, 56))
        with self.assertRaises(ValueError):
            encode_goal(encoder, torch.randn(2, 1, 56, 56))


if __name__ == "__main__":
    unittest.main()
