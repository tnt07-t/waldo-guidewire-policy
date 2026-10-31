"""Invented images only; no real data or pretrained weights are needed."""

import unittest

from PIL import Image, ImageDraw
import torch

from preprocessing import DINOv2Preprocessor


class PreprocessingTests(unittest.TestCase):
    def test_original_geometry_and_padding(self):
        image = Image.new("RGB", (1065, 1050), (255, 0, 0))
        before = image.tobytes()
        result = DINOv2Preprocessor().prepare_image(image)
        self.assertEqual(result["geometry"]["resized_size_wh"], (518, 511))
        self.assertEqual(result["geometry"]["padding_ltrb"], (0, 3, 0, 4))
        self.assertEqual(result["tensor"].shape, (3, 518, 518))
        self.assertEqual(result["preview"].getpixel((0, 3)), (255, 0, 0))
        self.assertEqual(result["preview"].getpixel((517, 513)), (255, 0, 0))
        self.assertEqual(image.size, (1065, 1050))
        self.assertEqual(image.tobytes(), before)

    def test_all_four_edges_survive_landscape_and_portrait_resize(self):
        for size in ((112, 56), (56, 112)):
            with self.subTest(size=size):
                image = Image.new("RGB", size)
                draw = ImageDraw.Draw(image)
                w, h = size
                draw.rectangle((0, 0, w - 1, 7), fill=(255, 0, 0))
                draw.rectangle((0, h - 8, w - 1, h - 1), fill=(0, 255, 0))
                draw.rectangle((0, 8, 7, h - 9), fill=(0, 0, 255))
                draw.rectangle((w - 8, 8, w - 1, h - 9), fill=(255, 255, 0))
                result = DINOv2Preprocessor(56).prepare_image(image)
                preview = result["preview"]
                rw, rh = result["geometry"]["resized_size_wh"]
                left, top, _, _ = result["geometry"]["padding_ltrb"]
                self.assertEqual(preview.getpixel((left + rw // 2, top)), (255, 0, 0))
                self.assertEqual(preview.getpixel((left + rw // 2, top + rh - 1)), (0, 255, 0))
                self.assertEqual(preview.getpixel((left, top + rh // 2)), (0, 0, 255))
                self.assertEqual(preview.getpixel((left + rw - 1, top + rh // 2)), (255, 255, 0))

    def test_rgb_scaling_normalization_and_preview_agree(self):
        prep = DINOv2Preprocessor(14)
        result = prep.prepare_image(Image.new("RGB", (14, 14), (255, 128, 0)))
        expected = torch.tensor([(1 - .485) / .229,
                                 (128 / 255 - .456) / .224,
                                 (0 - .406) / .225])
        torch.testing.assert_close(result["tensor"][:, 0, 0], expected)
        self.assertEqual(result["tensor"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(result["tensor"]).all())
        self.assertEqual(result["preview"].getpixel((0, 0)), (255, 128, 0))
        gray = Image.new("L", (14, 7), 80)
        converted = prep.prepare_image(gray)
        self.assertEqual(converted["preview"].getpixel((0, 3)), (80, 80, 80))
        self.assertEqual(converted["preview"].getpixel((0, 0)), (124, 116, 104))
        self.assertLess(converted["tensor"][:, 0, 0].abs().max().item(), .01)
        self.assertEqual(gray.mode, "L")

    def test_case_order_metadata_separation_and_repeatability(self):
        case = {
            "goal": Image.new("RGB", (14, 14), (255, 0, 0)),
            "history": [Image.new("RGB", (14, 14), (i * 30,) * 3) for i in range(6)],
            "depth_relative_mm": [-5, -4, -3, -2, -1, 0],
            "history_offsets_s": [-10, -8, -5, -3, -1, 0],
            "label": 1, "case_id": "invented", "episode": "synthetic", "row_index": 7,
        }
        prep = DINOv2Preprocessor(14)
        result = prep.prepare_case(case)
        self.assertEqual(result["goal"].shape, (3, 14, 14))
        self.assertEqual(result["history"].shape, (6, 3, 14, 14))
        for key in ("depth_relative_mm", "history_offsets_s", "label", "case_id", "episode", "row_index"):
            self.assertEqual(result[key], case[key])
        for i, frame in enumerate(case["history"]):
            torch.testing.assert_close(result["history"][i], prep.prepare_image(frame)["tensor"])
        self.assertEqual(result["previews"]["goal"].getpixel((0, 0)), (255, 0, 0))
        torch.testing.assert_close(result["history"], prep.prepare_case(case)["history"], rtol=0, atol=0)
        result["depth_relative_mm"][0] = 999
        self.assertEqual(case["depth_relative_mm"][0], -5)
        case["history"].pop()
        with self.assertRaisesRegex(ValueError, "six aligned"):
            prep.prepare_case(case)

    def test_invalid_sizes_rejected(self):
        for size in (0, -14, 15, True, 518.0):
            with self.subTest(size=size), self.assertRaises(ValueError):
                DINOv2Preprocessor(size)


if __name__ == "__main__":
    unittest.main()
