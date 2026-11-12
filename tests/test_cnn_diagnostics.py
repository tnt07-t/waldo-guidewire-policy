"""Invented-input checks for CNN 512-d diagnostics (no real images/weights)."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from PIL import Image

from guidewire.diagnostics.cnn import (
    content_bounds,
    cosine_distance,
    default_occlusion_size,
    goal_marker_sensitivity,
    goal_vs_history_separation,
    grid_occlusion_deltas,
    l2_distance,
    marker_probe_inputs,
    occlusion_sensitivity_history,
    pearson_corr,
    prepared_to_original_xy,
    red_marker_mask_from_rgb,
    replace_masked_rgb,
    run_case_diagnostics,
    safe_ratio,
    summarize_occlusion_grid,
    temporal_feature_diagnostics,
    translated_control_masks,
)
from guidewire.data.preprocessing import DINOv2Preprocessor, WirePreprocessor


class BrightnessEncoder(nn.Module):
    """Maps mean brightness into the first feature dim; rest zeros."""

    feature_dim = 512

    def __init__(self):
        super().__init__()
        # Present so frozen-state checks can inspect a submodule.
        self.backbone = nn.Identity()
        self.backbone.requires_grad_(False)
        self.backbone.eval()

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

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self


class PassthroughPreprocessor:
    """Minimal prep: RGB -> float tensor /255 without ImageNet norm."""

    def prepare_image(self, image):
        rgb = image.convert("RGB")
        pixels = torch.tensor(list(rgb.getdata()), dtype=torch.float32)
        pixels = pixels.reshape(rgb.height, rgb.width, 3).permute(2, 0, 1) / 255.0
        return {
            "tensor": pixels,
            "preview": rgb,
            "geometry": {
                "original_size_wh": rgb.size,
                "resized_size_wh": rgb.size,
                "padding_ltrb": (0, 0, 0, 0),
            },
        }

    def prepare_case(self, case):
        goal = self.prepare_image(case["goal"])
        history = [self.prepare_image(frame) for frame in case["history"]]
        return {
            "goal": goal["tensor"],
            "history": torch.stack([frame["tensor"] for frame in history]),
            "previews": {
                "goal": goal["preview"],
                "history": [frame["preview"] for frame in history],
            },
            "preparation": {
                "goal": goal["geometry"],
                "history": [frame["geometry"] for frame in history],
            },
            "depth_relative_mm": list(case["depth_relative_mm"]),
            "history_offsets_s": list(case["history_offsets_s"]),
        }


def solid_tensor(value, size=56):
    return torch.full((3, size, size), float(value))


def solid_rgb(color, size=56):
    return Image.new("RGB", (size, size), color)


class DiagnosticsTests(unittest.TestCase):
    def test_red_mask_rejects_bright_gray_accepts_red(self):
        gray = solid_rgb((230, 230, 230))
        red = solid_rgb((220, 30, 30))
        self.assertEqual(int(red_marker_mask_from_rgb(gray).sum()), 0)
        self.assertGreater(int(red_marker_mask_from_rgb(red).sum()), 0)
        # Overflow regression: uint8 (230+40) would wrap; int16 must not mark gray.
        mixed = solid_rgb((0, 0, 0))
        mixed.putpixel((10, 10), (230, 230, 230))
        mixed.putpixel((20, 20), (200, 20, 20))
        mask = red_marker_mask_from_rgb(mixed)
        self.assertFalse(bool(mask[10, 10]))
        self.assertTrue(bool(mask[20, 20]))

    def test_empty_marker_reported_unavailable(self):
        encoder = BrightnessEncoder().eval()
        prep = PassthroughPreprocessor()
        result = goal_marker_sensitivity(encoder, prep, solid_rgb((0, 0, 0)))
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["marker_pixels"], 0)

    def test_translated_controls_preserve_shape_without_overlap(self):
        mask = torch.zeros(40, 40, dtype=torch.bool)
        mask[5:15, 5:15] = True
        controls = translated_control_masks(mask, max_controls=4)
        self.assertGreater(len(controls), 0)
        area = int(mask.sum())
        for control in controls:
            self.assertEqual(int(control.sum()), area)
            self.assertFalse(bool((control & mask).any()))

    def test_safe_ratio_and_zero_distance(self):
        self.assertIsNone(safe_ratio(1.0, 0.0))
        self.assertIsNone(safe_ratio(1.0, 1e-12))
        self.assertAlmostEqual(safe_ratio(4.0, 2.0), 2.0)

    def test_goal_marker_sensitivity_with_controls(self):
        encoder = BrightnessEncoder().eval()
        prep = PassthroughPreprocessor()
        goal = solid_rgb((0, 0, 0), size=64)
        for y in range(8, 18):
            for x in range(8, 18):
                goal.putpixel((x, y), (220, 30, 30))
        with tempfile.TemporaryDirectory() as tmp:
            result = goal_marker_sensitivity(
                encoder, prep, goal, output_dir=tmp
            )
            self.assertEqual(result["status"], "ok")
            self.assertGreater(result["n_controls"], 1)
            self.assertIn("control_l2_values", result)
            self.assertTrue((Path(tmp) / "goal_red_marker_overlay.png").is_file())
            # Replacement uses the same helper for marker and controls.
            marker = red_marker_mask_from_rgb(goal)
            edited = replace_masked_rgb(goal, marker)
            self.assertNotEqual(list(goal.getdata())[8 * 64 + 8], list(edited.getdata())[8 * 64 + 8])

    def test_content_padding_excluded_from_occlusion_grid(self):
        encoder = BrightnessEncoder().eval()
        image = solid_tensor(0.0, size=56)
        image[:, 14:42, 14:42] = 1.0
        geometry = {
            "original_size_wh": (28, 28),
            "resized_size_wh": (28, 28),
            "padding_ltrb": (14, 14, 14, 14),
        }
        tops, lefts, deltas = grid_occlusion_deltas(
            encoder, image, patch=14, stride=14, geometry=geometry
        )
        bounds = content_bounds(geometry)
        self.assertEqual(bounds, (14, 14, 42, 42))
        for top, left in zip(tops.tolist(), lefts.tolist()):
            self.assertGreaterEqual(top, 14)
            self.assertGreaterEqual(left, 14)
            self.assertLessEqual(top + 14, 42)
            self.assertLessEqual(left + 14, 42)
        summary = summarize_occlusion_grid(tops, lefts, deltas, 14, geometry)
        self.assertEqual(summary["status"], "ok")
        self.assertNotIn("wire", summary)

    def test_occlusion_history_writes_heatmaps(self):
        encoder = BrightnessEncoder().eval()
        history = torch.zeros(6, 3, 56, 56)
        rgbs = []
        geometries = []
        for t in range(6):
            history[t, :, 0:20, 0:20] = 0.2 * (t + 1)
            rgbs.append(solid_rgb((10, 10, 10), size=56))
            geometries.append({
                "original_size_wh": (56, 56),
                "resized_size_wh": (56, 56),
                "padding_ltrb": (0, 0, 0, 0),
            })
        with tempfile.TemporaryDirectory() as tmp:
            report = occlusion_sensitivity_history(
                encoder,
                history,
                rgbs,
                geometries,
                patch=28,
                stride=28,
                output_dir=tmp,
            )
            self.assertEqual(report["name"], "occlusion_sensitivity")
            self.assertTrue(report["not_wire_detection"])
            self.assertTrue((Path(tmp) / "history_frame_0_occlusion_heatmap.png").is_file())

    def test_temporal_reports_signed_and_abs_depth_no_signed_corr(self):
        features = torch.zeros(6, 512)
        for t in range(6):
            features[t, 0] = float(t)
        depths = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        report = temporal_feature_diagnostics(features, depths)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(len(report["step_feature_l2"]), 5)
        self.assertEqual(len(report["step_depth_signed"]), 5)
        self.assertEqual(len(report["step_depth_abs"]), 5)
        self.assertNotIn("feature_depth_step_corr", report)
        self.assertIn("feature_l2_vs_abs_depth_corr", report)
        self.assertIn("wire-motion", report["interpretation"])

    def test_temporal_constant_and_missing(self):
        features = torch.zeros(6, 512)
        constant = temporal_feature_diagnostics(features, [0.0] * 6)
        self.assertEqual(constant["corr_status"], "unavailable")
        missing = temporal_feature_diagnostics(features, None)
        self.assertEqual(missing["status"], "unavailable")
        bad = temporal_feature_diagnostics(features, [1.0, 2.0])
        self.assertEqual(bad["status"], "unavailable")
        self.assertIsNone(pearson_corr([1.0], [1.0]))

    def test_distances_and_eval_encode(self):
        from guidewire.diagnostics.cnn import encode_image

        encoder = BrightnessEncoder()
        encoder.train()
        image = solid_tensor(1.0)
        feature = encode_image(encoder, image)
        self.assertFalse(encoder.backbone.training)
        self.assertEqual(feature.shape, (512,))
        occluded = image.clone()
        occluded[:, :28, :28] = 0.0
        self.assertGreater(l2_distance(feature, encode_image(encoder, occluded)), 0.0)
        self.assertEqual(cosine_distance(feature, feature), 0.0)
        sep = goal_vs_history_separation(feature, torch.stack([feature] * 6))
        self.assertEqual(len(sep["goal_history_l2"]), 6)


class WireDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        # 96 px frames, goal at the center: a 64 crop covers x and y 16..79.
        frame = Image.new("RGB", (96, 96), (90, 90, 90))
        marked = frame.copy()
        for x in range(46, 51):
            for y in range(46, 51):
                marked.putpixel((x, y), (220, 30, 30))
        self.case = {
            "goal": marked,
            "goal_raw": frame,
            "goal_xy": [48, 48],
            "history": [frame.copy() for _ in range(6)],
            "depth_relative_mm": [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0],
            "history_offsets_s": [-4.0, -3.2, -2.4, -1.6, -0.8, 0.0],
        }

    def test_crop_coordinates_map_back_by_box_offset(self):
        geometry = {"original_size_wh": (96, 96), "crop_box_ltrb": (-10, 16, 54, 80),
                    "resized_size_wh": (54, 64), "padding_ltrb": (10, 0, 0, 0)}
        self.assertEqual(prepared_to_original_xy(32, 5, geometry), (22, 21))
        self.assertEqual(content_bounds(geometry), (0, 10, 64, 64))

    def test_marker_probe_uses_marked_goal_crop(self):
        prep = WirePreprocessor(crop_size=64)
        goal, marker_prep = marker_probe_inputs(prep, self.case)
        self.assertEqual((goal.size, goal.mode), ((64, 64), "RGB"))
        # The marker sits at the crop center: 46..50 minus the 16 px box offset.
        self.assertEqual(int(red_marker_mask_from_rgb(goal).sum()), 25)
        self.assertTrue(bool(red_marker_mask_from_rgb(goal)[32, 32]))
        prepared = marker_prep.prepare_image(goal)
        self.assertEqual(prepared["geometry"]["crop_box_ltrb"], (0, 0, 64, 64))
        self.assertEqual(prepared["geometry"]["padding_ltrb"], (0, 0, 0, 0))
        dinov2 = DINOv2Preprocessor(14)
        self.assertEqual(marker_probe_inputs(dinov2, self.case), (self.case["goal"], dinov2))

    def test_run_case_diagnostics_with_wire_preprocessor(self):
        prep = WirePreprocessor(crop_size=64)
        patch = default_occlusion_size(prep)
        self.assertEqual(patch, 8)
        self.assertEqual(default_occlusion_size(DINOv2Preprocessor()), 74)
        report = run_case_diagnostics(BrightnessEncoder().eval(), prep, self.case,
                                      patch=patch, stride=patch)
        goal = report["goal_marker_sensitivity"]
        self.assertEqual(goal["status"], "ok")
        self.assertEqual(goal["input"], "marked goal cropped with the wire box")
        self.assertGreater(goal["marker_l2"], 0.0)
        frames = report["occlusion_sensitivity"]["frames"]
        self.assertEqual(frames[0]["n_patches"], 64)
        cx, cy = frames[0]["argmax_center_original_xy"]
        self.assertTrue(16 <= cx <= 80 and 16 <= cy <= 80)
        self.assertEqual(report["history_features_shape"], (6, 512))


if __name__ == "__main__":
    unittest.main()
