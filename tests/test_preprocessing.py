"""Invented images only; no real data or pretrained weights are needed."""

from copy import deepcopy
import unittest

from PIL import Image, ImageDraw
import torch

from preprocessing import (CHECKPOINT, MEAN, STD, DINOv2Preprocessor,
                           WirePreprocessor, black_tophat, goal_crop)


def legacy_prepare_case(prep, case):
    """Frozen copy of DINOv2Preprocessor.prepare_case before it was refactored."""
    if any(len(case[key]) != 6 for key in
           ("history", "depth_relative_mm", "history_offsets_s")):
        raise ValueError("Expected six aligned history images, depths, and times")
    goal = prep.prepare_image(case["goal"])
    history = [prep.prepare_image(frame) for frame in case["history"]]
    prepared = {key: deepcopy(value) for key, value in case.items()
                if key not in ("goal", "history")}
    prepared.update({
        "goal": goal["tensor"],
        "history": torch.stack([frame["tensor"] for frame in history]),
        "previews": {
            "goal": goal["preview"],
            "history": [frame["preview"] for frame in history],
        },
        "preparation": {
            "checkpoint": CHECKPOINT,
            "size": prep.size,
            "interpolation": "Pillow bicubic",
            "padding_rgb": prep.padding_rgb,
            "mean": MEAN,
            "std": STD,
            "pixel_scale": "uint8 / 255",
            "goal": goal["geometry"],
            "history": [frame["geometry"] for frame in history],
        },
    })
    return prepared


def reference_black_tophat(rows, size):
    """Slow closing minus image on nested lists, clamping at the borders."""
    height, width, radius = len(rows), len(rows[0]), size // 2

    def window(values, y, x, pick):
        return pick(values[min(max(y + dy, 0), height - 1)][min(max(x + dx, 0), width - 1)]
                    for dy in range(-radius, radius + 1)
                    for dx in range(-radius, radius + 1))

    dilated = [[window(rows, y, x, max) for x in range(width)] for y in range(height)]
    closed = [[window(dilated, y, x, min) for x in range(width)] for y in range(height)]
    return [[closed[y][x] - rows[y][x] for x in range(width)] for y in range(height)]


def unnormalize(tensor):
    """Undo ImageNet normalization on [..., 3, H, W] to recover 0-1 pixels."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return tensor * std + mean


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
        # Colored borders test cropping, not real thin-wire visibility.
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
        # Distinct history brightness and a red goal reveal ordering/mixing errors.
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

    def test_case_preparation_matches_frozen_legacy_version(self):
        # Uneven shapes and drawn rectangles make resize and padding matter.
        def frame(i):
            image = Image.new("RGB", (37, 23), (i * 30, 90, 255 - i * 30))
            ImageDraw.Draw(image).rectangle((3 + i, 2, 12 + i, 9), fill=(250, 250, 250))
            return image
        case = {
            "goal": frame(9),
            "history": [frame(i) for i in range(6)],
            "depth_relative_mm": [-5.5, -4, -3, -2, -1, 0],
            "history_offsets_s": [-4.0, -3.2, -2.4, -1.6, -0.8, 0.0],
            "goal_xy": [18.5, 11.25],
            "label": 0, "case_id": "invented", "episode": "synthetic", "row_index": 3,
        }
        prep = DINOv2Preprocessor(28)
        result = prep.prepare_case(case)
        expected = legacy_prepare_case(prep, case)
        self.assertEqual(set(result), set(expected))
        self.assertTrue(torch.equal(result["goal"], expected["goal"]))
        self.assertTrue(torch.equal(result["history"], expected["history"]))
        self.assertEqual(result["previews"]["goal"].tobytes(),
                         expected["previews"]["goal"].tobytes())
        self.assertEqual([p.tobytes() for p in result["previews"]["history"]],
                         [p.tobytes() for p in expected["previews"]["history"]])
        self.assertEqual(result["preparation"], expected["preparation"])
        for key in expected:
            if key not in ("goal", "history", "previews", "preparation"):
                self.assertEqual(result[key], expected[key])

    def test_unmarked_goal_image_is_not_copied_as_metadata(self):
        case = {
            "goal": Image.new("RGB", (14, 14)),
            "goal_raw": Image.new("RGB", (14, 14)),
            "history": [Image.new("RGB", (14, 14))] * 6,
            "depth_relative_mm": [0] * 6, "history_offsets_s": [0] * 6,
        }
        self.assertNotIn("goal_raw", DINOv2Preprocessor(14).prepare_case(case))


class BlackTophatTests(unittest.TestCase):
    def test_thin_dark_line_kept_and_gradient_removed(self):
        image = torch.linspace(0.5, 0.7, 64).expand(64, 64).clone()
        image[:, 30:34] = 0.05
        result = black_tophat(image, 11)
        self.assertGreater(result[:, 30:34].min().item(), 0.3)
        background = torch.ones(64, 64, dtype=torch.bool)
        background[:, 30:34] = False
        # Only the left border differs from zero, by five pixels of slope (~0.016).
        self.assertLess(result[background].max().item(), 0.02)

    def test_dark_blob_wider_than_window_is_removed(self):
        image = torch.full((48, 48), 0.8)
        image[10:31, 10:31] = 0.1
        self.assertLess(black_tophat(image, 11).max().item(), 1e-6)

    def test_constant_image_gives_zeros(self):
        image = torch.full((5, 7), 0.4)
        self.assertTrue(torch.equal(black_tophat(image, 3), torch.zeros(5, 7)))

    def test_shape_batching_and_input_unchanged(self):
        torch.manual_seed(0)
        batch = torch.rand(2, 1, 9, 11)
        before = batch.clone()
        result = black_tophat(batch, 5)
        self.assertEqual(result.shape, batch.shape)
        self.assertTrue(torch.equal(batch, before))
        self.assertGreaterEqual(result.min().item(), 0)
        for i in range(2):
            self.assertTrue(torch.equal(result[i, 0], black_tophat(batch[i, 0], 5)))

    def test_matches_slow_reference(self):
        torch.manual_seed(1)
        image = torch.rand(9, 11)
        for size in (3, 5, 7):
            with self.subTest(size=size):
                expected = torch.tensor(reference_black_tophat(image.tolist(), size))
                torch.testing.assert_close(black_tophat(image, size), expected)

    def test_invalid_inputs_rejected(self):
        for size in (1, 2, 4, -3, True, 3.0):
            with self.subTest(size=size), self.assertRaises(ValueError):
                black_tophat(torch.rand(9, 9), size)
        for image in (torch.rand(3, 9, 9), torch.rand(1, 2, 9, 9),
                      torch.zeros(9, 9, dtype=torch.uint8)):
            with self.subTest(shape=tuple(image.shape)), self.assertRaises(ValueError):
                black_tophat(image, 3)


class GoalCropTests(unittest.TestCase):
    @staticmethod
    def pattern(size=64):
        # Every pixel's value encodes its position, so replication is checkable.
        image = Image.new("L", (size, size))
        image.putdata([x + 2 * y for y in range(size) for x in range(size)])
        return image

    def test_goal_lands_at_crop_center(self):
        image = Image.new("L", (1065, 1050), 50)
        image.putpixel((532, 345), 255)
        crop, geometry = goal_crop(image, (532.04, 345.01), 256)
        self.assertEqual(crop.size, (256, 256))
        self.assertEqual(crop.mode, "L")
        self.assertEqual(crop.getpixel((128, 128)), 255)
        self.assertEqual(crop.getpixel((127, 128)), 50)
        self.assertEqual(geometry, {
            "original_size_wh": (1065, 1050),
            "crop_box_ltrb": (404, 217, 660, 473),
            "resized_size_wh": (256, 256),
            "padding_ltrb": (0, 0, 0, 0),
        })
        # Halves round up, not to the nearest even pixel.
        crop, _ = goal_crop(image, (531.5, 344.5), 256)
        self.assertEqual(crop.getpixel((128, 128)), 255)

    def test_borders_replicate_edge_pixels(self):
        image = self.pattern()
        cases = {
            "left": ((3, 30), (13, 0, 0, 0)),
            "top": ((30, 5), (0, 11, 0, 0)),
            "right": ((60, 30), (0, 0, 12, 0)),
            "bottom": ((30, 62), (0, 0, 0, 14)),
            "corner": ((0, 63), (16, 0, 0, 15)),
        }
        for name, (goal_xy, padding) in cases.items():
            with self.subTest(name):
                crop, geometry = goal_crop(image, goal_xy, 32)
                self.assertEqual(geometry["padding_ltrb"], padding)
                left, top, right, bottom = geometry["crop_box_ltrb"]
                self.assertEqual((right - left, bottom - top), (32, 32))
                self.assertEqual(geometry["resized_size_wh"],
                                 (32 - padding[0] - padding[2], 32 - padding[1] - padding[3]))
                for j in range(32):
                    for i in range(32):
                        source = (min(max(left + i, 0), 63), min(max(top + j, 0), 63))
                        self.assertEqual(crop.getpixel((i, j)), image.getpixel(source))

    def test_color_input_converted_and_unchanged(self):
        image = Image.new("RGB", (64, 64), (90, 90, 90))
        before = image.tobytes()
        crop, _ = goal_crop(image, (32, 32), 32)
        self.assertEqual(crop.mode, "L")
        self.assertEqual(crop.getpixel((0, 0)), 90)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.tobytes(), before)

    def test_invalid_sizes_and_goals_rejected(self):
        image = self.pattern()
        for size in (0, -32, 31, 33, 100, True, 256.0):
            with self.subTest(size=size), self.assertRaises(ValueError):
                goal_crop(image, (32, 32), size)
        for goal_xy in ((-1, 32), (32, 64), (70, 70)):
            with self.subTest(goal_xy=goal_xy), self.assertRaises(ValueError):
                goal_crop(image, goal_xy, 32)


class WirePreprocessorTests(unittest.TestCase):
    def setUp(self):
        # A bright pixel at goal_xy on every frame; distinct backgrounds per frame.
        def frame(value, mode="L"):
            image = Image.new(mode, (1065, 1050), value if mode == "L" else (value,) * 3)
            image.putpixel((532, 345), 255 if mode == "L" else (255,) * 3)
            return image
        self.case = {
            "goal": Image.new("RGB", (1065, 1050), (255, 0, 0)),
            "goal_raw": frame(60, "RGB"),
            "goal_xy": [532.04, 345.01],
            "history": [frame(10 + 20 * i) for i in range(6)],
            "depth_relative_mm": [-5, -4, -3, -2, -1, 0],
            "history_offsets_s": [-4.0, -3.2, -2.4, -1.6, -0.8, 0.0],
            "label": 1, "case_id": "invented", "episode": "synthetic", "row_index": 7,
        }

    def test_shapes_center_and_raw_channels(self):
        result = WirePreprocessor().prepare_case(self.case)
        self.assertEqual(result["goal"].shape, (3, 256, 256))
        self.assertEqual(result["history"].shape, (6, 3, 256, 256))
        self.assertEqual(result["goal"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(result["goal"]).all())
        self.assertTrue(torch.isfinite(result["history"]).all())
        pixels = unnormalize(torch.cat([result["goal"][None], result["history"]]))
        torch.testing.assert_close(pixels[:, 0], pixels[:, 1])
        torch.testing.assert_close(pixels[:, 0], pixels[:, 2])
        torch.testing.assert_close(pixels[:, 0, 128, 128], torch.ones(7))
        # The goal comes from the unmarked image, not the red marked one.
        backgrounds = torch.tensor([60] + [10 + 20 * i for i in range(6)]) / 255
        torch.testing.assert_close(pixels[:, 0, 0, 0], backgrounds)
        self.assertEqual(result["previews"]["goal"].getpixel((128, 128)), 255)

    def test_preparation_record_and_metadata(self):
        result = WirePreprocessor(crop_size=192).prepare_case(self.case)
        preparation = result["preparation"]
        self.assertEqual(preparation["preprocessor"], "WirePreprocessor")
        self.assertEqual(preparation["crop_size"], 192)
        self.assertEqual(preparation["channels"], "raw")
        self.assertIsNone(preparation["resize"])
        self.assertEqual((preparation["mean"], preparation["std"]), (MEAN, STD))
        self.assertNotIn("checkpoint", preparation)
        self.assertEqual(preparation["goal"]["crop_box_ltrb"], (436, 249, 628, 441))
        self.assertEqual(len(preparation["history"]), 6)
        self.assertEqual(result["goal_xy"], self.case["goal_xy"])
        self.assertNotIn("goal_raw", result)
        for key in ("depth_relative_mm", "history_offsets_s", "label", "case_id", "row_index"):
            self.assertEqual(result[key], self.case[key])

    def test_raw_case_not_modified(self):
        images = [self.case["goal"], self.case["goal_raw"]] + self.case["history"]
        before = [(image.mode, image.size, image.tobytes()) for image in images]
        metadata = {key: deepcopy(value) for key, value in self.case.items()
                    if key not in ("goal", "goal_raw", "history")}
        result = WirePreprocessor().prepare_case(self.case)
        result["goal_xy"][0] = 0
        self.assertEqual([(image.mode, image.size, image.tobytes()) for image in images], before)
        for key, value in metadata.items():
            self.assertEqual(self.case[key], value)

    def test_frame_encoder_accepts_output(self):
        from cnn import FrameEncoder  # needs torchvision, so imported only here
        result = WirePreprocessor().prepare_case(self.case)
        encoder = FrameEncoder(pretrained=False)
        with torch.no_grad():
            history = encoder(result["history"].unsqueeze(0))
            goal = encoder(result["goal"].unsqueeze(0))
        self.assertEqual(history.shape, (1, 6, 512))
        self.assertEqual(goal.shape, (1, 512))

    def test_invalid_settings_and_cases_rejected(self):
        for kwargs in ({"crop_size": 250}, {"crop_size": 0}, {"channels": "enhanced"}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                WirePreprocessor(**kwargs)
        self.case["history"].pop()
        with self.assertRaisesRegex(ValueError, "six aligned"):
            WirePreprocessor().prepare_case(self.case)


if __name__ == "__main__":
    unittest.main()
