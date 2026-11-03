"""Model-input preparation: DINOv2 full view and goal-centered wire crops.

Real images and previews must stay on Perlmutter. This module downloads no
weights, writes no files, and leaves the raw reader's images unchanged.
"""

from copy import deepcopy
import math

from PIL import Image
import torch
import torch.nn.functional as F


CHECKPOINT = "vit_small_patch14_dinov2.lvd142m"
# Checkpoint config: https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/blob/main/config.json
# These are also the ImageNet constants torchvision's ResNet weights expect.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Raw images are replaced by prepared tensors; every other key is metadata.
IMAGE_KEYS = ("goal", "goal_raw", "history")
WIRE_CHANNEL_MODES = ("raw",)


def check_history(case):
    """Require six aligned history images, depths, and time offsets."""
    if any(len(case[key]) != 6 for key in
           ("history", "depth_relative_mm", "history_offsets_s")):
        raise ValueError("Expected six aligned history images, depths, and times")


def assemble_case(case, goal, history, preparation):
    """Combine ``prepare_image`` results with copied case metadata.

    ``preparation`` holds the preprocessor-specific settings; the goal and
    history geometry are added to it here.
    """
    # Copy metadata to keep the raw case independent of prepared outputs.
    prepared = {key: deepcopy(value) for key, value in case.items()
                if key not in IMAGE_KEYS}
    prepared.update({
        "goal": goal["tensor"],
        "history": torch.stack([frame["tensor"] for frame in history]),
        "previews": {
            "goal": goal["preview"],
            "history": [frame["preview"] for frame in history],
        },
        "preparation": {
            **preparation,
            "goal": goal["geometry"],
            "history": [frame["geometry"] for frame in history],
        },
    })
    return prepared


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
        check_history(case)
        goal = self.prepare_image(case["goal"])
        history = [self.prepare_image(frame) for frame in case["history"]]
        return assemble_case(case, goal, history, {
            "checkpoint": CHECKPOINT,
            "size": self.size,
            "interpolation": "Pillow bicubic",
            "padding_rgb": self.padding_rgb,
            "mean": MEAN,
            "std": STD,
            "pixel_scale": "uint8 / 255",
        })


def _dilate(x, size):
    """Square max filter on [N,1,H,W] as two 1-D passes; edges replicate."""
    radius = size // 2
    x = F.max_pool2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"),
                     (1, size), stride=1)
    return F.max_pool2d(F.pad(x, (0, 0, radius, radius), mode="replicate"),
                        (size, 1), stride=1)


def black_tophat(image, size):
    """Return closing(image) - image for a ``size`` square window.

    Keeps dark structures narrower than the window (the wire) and removes
    smooth background. ``image`` is a float [H,W] or [N,1,H,W] tensor; the
    result has the same shape, is non-negative, and runs on the input device.
    """
    if type(size) is not int or size < 3 or size % 2 == 0:
        raise ValueError("size must be an odd integer of at least 3")
    if not torch.is_floating_point(image):
        raise ValueError("image must be a floating-point tensor")
    if image.ndim == 2:
        x = image[None, None]
    elif image.ndim == 4 and image.shape[1] == 1:
        x = image
    else:
        raise ValueError("Expected image [H, W] or [N, 1, H, W]")
    # Closing is erosion of the dilation; erosion is a max filter on -x.
    closing = -_dilate(-_dilate(x, size), size)
    # Closing is never below the input, so the difference is non-negative.
    return (closing - x).reshape(image.shape)


def check_crop_size(size):
    """Crops must be a positive multiple of ResNet's total stride of 32."""
    if type(size) is not int or size <= 0 or size % 32:
        raise ValueError("crop size must be a positive multiple of 32")


def goal_crop(image, goal_xy, size):
    """Crop a grayscale ``size`` square centered on ``goal_xy`` at 1x scale.

    Returns the Pillow "L" crop and its geometry. Box parts outside the image
    repeat the nearest edge pixel; ``padding_ltrb`` counts them per side and
    ``resized_size_wh`` is the size of the content taken from the image.
    """
    check_crop_size(size)
    gray = image.convert("L")
    width, height = gray.size
    # Round half up; Python's round() sends .5 to the nearest even pixel.
    cx, cy = (math.floor(value + 0.5) for value in goal_xy)
    if not (0 <= cx < width and 0 <= cy < height):
        raise ValueError("goal_xy is outside the image")
    left, top = cx - size // 2, cy - size // 2
    right, bottom = left + size, top + size
    pad_l, pad_t = max(0, -left), max(0, -top)
    pad_r, pad_b = max(0, right - width), max(0, bottom - height)
    inner = gray.crop((left + pad_l, top + pad_t, right - pad_r, bottom - pad_b))
    inner_w, inner_h = inner.size

    crop = Image.new("L", (size, size))
    crop.paste(inner, (pad_l, pad_t))
    # Stretch the outermost content column or row over each padding strip.
    # Rows go last and span the full width, so corners take the corner pixel.
    nearest = Image.Resampling.NEAREST
    if pad_l:
        edge = crop.crop((pad_l, pad_t, pad_l + 1, pad_t + inner_h))
        crop.paste(edge.resize((pad_l, inner_h), nearest), (0, pad_t))
    if pad_r:
        edge = crop.crop((size - pad_r - 1, pad_t, size - pad_r, pad_t + inner_h))
        crop.paste(edge.resize((pad_r, inner_h), nearest), (size - pad_r, pad_t))
    if pad_t:
        edge = crop.crop((0, pad_t, size, pad_t + 1))
        crop.paste(edge.resize((size, pad_t), nearest), (0, 0))
    if pad_b:
        edge = crop.crop((0, size - pad_b - 1, size, size - pad_b))
        crop.paste(edge.resize((size, pad_b), nearest), (0, size - pad_b))
    return crop, {
        "original_size_wh": image.size,
        "crop_box_ltrb": (left, top, right, bottom),
        "resized_size_wh": (inner_w, inner_h),
        "padding_ltrb": (pad_l, pad_t, pad_r, pad_b),
    }


class WirePreprocessor:
    """Goal-centered native-resolution grayscale crops for the frozen ResNet.

    One crop box around ``goal_xy`` serves the unmarked goal and all six
    history frames, since the view is fixed within a case. ``raw`` repeats
    the grayscale crop to three channels with ImageNet normalization.
    """

    def __init__(self, crop_size=256, channels="raw"):
        check_crop_size(crop_size)
        if channels not in WIRE_CHANNEL_MODES:
            raise ValueError("channels must be one of {}".format(WIRE_CHANNEL_MODES))
        self.crop_size = crop_size
        self.channels = channels
        self.mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)

    def prepare_image(self, image, goal_xy):
        """Return a normalized [3,S,S] tensor, grayscale preview, and geometry."""
        crop, geometry = goal_crop(image, goal_xy, self.crop_size)
        # A writable byte copy avoids sharing storage with the Pillow image.
        pixels = torch.frombuffer(bytearray(crop.tobytes()), dtype=torch.uint8)
        gray = pixels.reshape(self.crop_size, self.crop_size).to(torch.float32).div(255)
        tensor = (gray.expand(3, -1, -1) - self.mean) / self.std
        return {"tensor": tensor.contiguous(), "preview": crop, "geometry": geometry}

    def prepare_case(self, case):
        """Prepare one raw case; goal [3,S,S], history [6,3,S,S].

        The goal comes from ``goal_raw`` so the target marking is not cropped.
        Metadata is copied unchanged, as in ``DINOv2Preprocessor``.
        """
        check_history(case)
        if len(case["goal_xy"]) != 2:
            raise ValueError("Expected goal_xy to hold two coordinates")
        goal = self.prepare_image(case["goal_raw"], case["goal_xy"])
        history = [self.prepare_image(frame, case["goal_xy"])
                   for frame in case["history"]]
        return assemble_case(case, goal, history, {
            "preprocessor": type(self).__name__,
            "crop_size": self.crop_size,
            "channels": self.channels,
            "crop_center": "goal_xy rounded half up",
            "outside_fill": "edge replication",
            "resize": None,
            "grayscale": "Pillow L",
            "mean": MEAN,
            "std": STD,
            "pixel_scale": "uint8 / 255",
        })
