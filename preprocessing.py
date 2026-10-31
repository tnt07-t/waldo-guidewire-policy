"""Full-view preparation for the initial DINOv2 Small/14 trial.

Real images and previews must stay on Perlmutter. This module downloads no
weights, writes no files, and leaves the raw reader's images unchanged.
"""

from copy import deepcopy

from PIL import Image
import torch


CHECKPOINT = "vit_small_patch14_dinov2.lvd142m"
# Checkpoint config: https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/blob/main/config.json
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class DINOv2Preprocessor:
    """RGB, proportional bicubic resize, centered padding, then normalization.

    ``size`` is the square trial input size, not a training-size decision.
    Padding uses the nearest 8-bit RGB representation of the checkpoint mean,
    so its normalized value is approximately (rather than exactly) zero.
    """

    def __init__(self, size=518):
        if type(size) is not int or size <= 0 or size % 14:
            raise ValueError("size must be a positive integer divisible by 14")
        self.size = size
        self.padding_rgb = tuple(round(value * 255) for value in MEAN)
        self.mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)

    def prepare_image(self, image):
        """Return a normalized [3,H,W] tensor, RGB preview, and geometry.

        Preview pixels are exactly those converted to the model tensor. The
        geometry uses Pillow's (width, height) convention; padding is L,T,R,B.
        """
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        # Fit the longer side to the square; one shared scale preserves shape.
        scale = self.size / max(width, height)
        resized_size = (max(1, round(width * scale)),
                        max(1, round(height * scale)))
        resized = rgb.resize(resized_size, resample=Image.Resampling.BICUBIC)
        # Center without cropping. An odd leftover pixel goes right or below.
        left = (self.size - resized_size[0]) // 2
        top = (self.size - resized_size[1]) // 2
        preview = Image.new("RGB", (self.size, self.size), self.padding_rgb)
        preview.paste(resized, (left, top))

        # A writable byte copy avoids sharing storage with the Pillow image.
        pixels = torch.frombuffer(bytearray(preview.tobytes()), dtype=torch.uint8)
        # Pillow stores [height,width,RGB]; PyTorch expects [RGB,height,width].
        pixels = pixels.reshape(self.size, self.size, 3).permute(2, 0, 1)
        tensor = (pixels.to(torch.float32).div(255) - self.mean) / self.std
        return {
            "tensor": tensor.contiguous(),
            "preview": preview,
            "geometry": {
                "original_size_wh": image.size,
                "resized_size_wh": resized_size,
                "padding_ltrb": (left, top, self.size - resized_size[0] - left,
                                 self.size - resized_size[1] - top),
            },
        }

    def prepare_case(self, case):
        """Prepare one raw case; goal [3,H,W], history [6,3,H,W].

        Metadata is copied unchanged, not normalized or used as image input.
        Previews and geometry retain separate goal/history roles as well.
        """
        if any(len(case[key]) != 6 for key in
               ("history", "depth_relative_mm", "history_offsets_s")):
            raise ValueError("Expected six aligned history images, depths, and times")
        goal = self.prepare_image(case["goal"])
        history = [self.prepare_image(frame) for frame in case["history"]]
        # Copy metadata to keep the raw case independent of prepared outputs.
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
                "size": self.size,
                "interpolation": "Pillow bicubic",
                "padding_rgb": self.padding_rgb,
                "mean": MEAN,
                "std": STD,
                "pixel_scale": "uint8 / 255",
                "goal": goal["geometry"],
                "history": [frame["geometry"] for frame in history],
            },
        })
        return prepared
