"""Invented-input checks for frozen CNN linear-probe helpers."""

import unittest

import torch
from torch import nn

from probe_cnn_features import BEST_PRESET, build_features, train_linear_probe


class BuildFeaturesTests(unittest.TestCase):
    def test_design_shapes(self):
        history = torch.randn(5, 6, 512)
        goal = torch.randn(5, 512)
        self.assertEqual(tuple(build_features("latest_goal", history, goal).shape), (5, 1024))
        self.assertEqual(tuple(build_features("mean_goal", history, goal).shape), (5, 1024))
        self.assertEqual(tuple(build_features("goal_only", history, goal).shape), (5, 512))
        torch.testing.assert_close(build_features("latest_goal", history, goal)[:, :512],
                                   history[:, -1])

    def test_best_preset_is_goal_only_nomaxpool(self):
        self.assertTrue(BEST_PRESET["remove_maxpool"])
        self.assertEqual(BEST_PRESET["designs"], ["goal_only"])


class ProbeTrainTests(unittest.TestCase):
    def test_separable_goal_feature_reaches_perfect_val(self):
        n = 40
        labels = torch.arange(n) % 2
        goal = torch.zeros(n, 512)
        goal[:, 0] = labels.float() * 4 - 2
        history = torch.randn(n, 6, 512)
        train_x = build_features("goal_only", history, goal)
        result = train_linear_probe(
            train_x, labels, train_x, labels,
            epochs=40, lr=1e-1, weight_decay=0.0)
        self.assertEqual(result["best"]["val"]["bal_acc"], 1.0)

    def test_linear_state_loads_into_classifier(self):
        # Avoid importing torchvision: exercise the Linear interface only.
        head = nn.Linear(512, 2)
        with torch.no_grad():
            head.weight.fill_(0.01)
            head.bias.fill_(0.0)
        restored = nn.Linear(512, 2)
        restored.load_state_dict(head.state_dict())
        torch.testing.assert_close(restored.weight, head.weight)


if __name__ == "__main__":
    unittest.main()
