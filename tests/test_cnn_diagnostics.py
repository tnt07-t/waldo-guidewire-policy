"""Invented-input checks for CNN 512-d diagnostics (no real images/weights)."""

import unittest

import torch
from torch import nn
from PIL import Image

from cnn_diagnostics import (
    cosine_distance,
    direction_from_history_features,
    equal_area_control_mask,
    goal_marker_probe,
    goal_vs_history_separation,
    grid_occlusion_deltas,
    l2_distance,
    occlude_square,
    red_marker_mask,
    summarize_occlusion_grid,
    wire_history_probe,
)


class BrightnessEncoder(nn.Module):
    """Maps mean brightness into the first feature dim; rest zeros."""

    feature_dim = 512

    def forward(self, images):
        single = images.ndim == 4
        if single:
            images = images.unsqueeze(1)
        batch, steps, _, _, _ = images.shape
        brightness = images.mean(dim=(2, 3, 4))
        features = torch.zeros(batch, steps, self.feature_dim, dtype=images.dtype)
        features[..., 0] = brightness
        if single:
            return features[:, 0]
        return features


def solid_image(value, size=56):
    return torch.full((3, size, size), float(value))


class DiagnosticsTests(unittest.TestCase):
    def test_distances_and_occlusion_change_features(self):
        encoder = BrightnessEncoder()
        image = solid_image(1.0)
        baseline = encoder(image.unsqueeze(0))[0]
        occluded = occlude_square(image, 0, 0, 28, fill=0.0)
        changed = encoder(occluded.unsqueeze(0))[0]
        # Brightness-only features keep the same direction; L2 still moves.
        self.assertGreater(l2_distance(baseline, changed), 0.0)
        self.assertEqual(cosine_distance(baseline, changed), 0.0)
        opposite = baseline.clone()
        opposite[0] = -baseline[0]
        self.assertGreater(cosine_distance(baseline, opposite), 0.0)

    def test_grid_occlusion_prefers_bright_patch(self):
        encoder = BrightnessEncoder()
        image = solid_image(0.0, size=56)
        # Bright square in the upper-left; occluding it should dominate.
        image[:, 0:28, 0:28] = 1.0
        tops, lefts, deltas = grid_occlusion_deltas(
            encoder, image, patch=28, stride=28
        )
        summary = summarize_occlusion_grid(tops, lefts, deltas, 56, 56, 28)
        self.assertEqual(summary["n_patches"], 4)
        self.assertGreater(summary["max_l2"], summary["median_l2"])
        x, y = summary["argmax_center_xy_norm"]
        self.assertLess(x, 0.5)
        self.assertLess(y, 0.5)

    def test_wire_history_probe_shapes(self):
        encoder = BrightnessEncoder()
        history = torch.zeros(6, 3, 56, 56)
        for t in range(6):
            history[t, :, 0:14, 0:14] = 0.2 * (t + 1)
        report = wire_history_probe(encoder, history, patch=28, stride=28)
        self.assertEqual(len(report["frames"]), 6)
        self.assertEqual(len(report["wire_scores"]), 6)
        self.assertGreaterEqual(report["score_range"], 0.0)

    def test_red_marker_and_goal_probe(self):
        encoder = BrightnessEncoder()
        preview = Image.new("RGB", (56, 56), (0, 0, 0))
        for y in range(10, 20):
            for x in range(10, 20):
                preview.putpixel((x, y), (220, 30, 30))
        mask = red_marker_mask(preview)
        self.assertGreater(int(mask.sum()), 0)
        control = equal_area_control_mask(mask)
        self.assertEqual(int(control.sum()), int(mask.sum()))

        goal = solid_image(0.0, size=56)
        # Bright block under the red preview mark so occlusion changes mean.
        goal[:, 10:20, 10:20] = 1.0
        probe = goal_marker_probe(encoder, goal, preview)
        self.assertGreater(probe["marker_pixels"], 0)
        self.assertGreater(probe["marker_l2"], 0.0)

    def test_direction_from_features_uses_sequence_only(self):
        features = torch.zeros(6, 512)
        for t in range(6):
            features[t, 0] = float(t)
        depths = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        scores = [1.0, 1.0, 1.2, 1.4, 1.5, 1.7]
        report = direction_from_history_features(
            features, wire_scores=scores, depths=depths
        )
        self.assertEqual(len(report["step_l2"]), 5)
        self.assertAlmostEqual(report["first_last_l2"], 5.0, places=5)
        self.assertIn("feature_depth_step_corr", report)
        sep = goal_vs_history_separation(features[5], features)
        self.assertEqual(len(sep["goal_history_l2"]), 6)


if __name__ == "__main__":
    unittest.main()
