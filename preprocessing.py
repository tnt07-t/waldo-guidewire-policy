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
WIRE_CHANNEL_MODES = ("raw", "enhanced")
DIFFERENCE_MODES = ("median", "previous")
# Context padding: the 8-bit gray nearest the average ImageNet channel mean.
CONTEXT_FILL = round(sum(MEAN) / 3 * 255)


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


def fit_square(image, size, fill):
    """Proportional bicubic resize into a ``size`` square on ``fill`` padding.

    Returns the square image and its geometry in Pillow's (width, height)
    convention; padding is L,T,R,B.
    """
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    # Fit the longer side to the square; one shared scale preserves shape.
    scale = size / max(width, height)
    resized_size = (max(1, round(width * scale)),
                    max(1, round(height * scale)))
    resized = image.resize(resized_size, resample=Image.Resampling.BICUBIC)
    # Center without cropping. An odd leftover pixel goes right or below.
    left = (size - resized_size[0]) // 2
    top = (size - resized_size[1]) // 2
    square = Image.new(image.mode, (size, size), fill)
    square.paste(resized, (left, top))
    return square, {
        "original_size_wh": image.size,
        "resized_size_wh": resized_size,
        "padding_ltrb": (left, top, size - resized_size[0] - left,
                         size - resized_size[1] - top),
    }


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
        preview, geometry = fit_square(image.convert("RGB"), self.size, self.padding_rgb)
        # A writable byte copy avoids sharing storage with the Pillow image.
        pixels = torch.frombuffer(bytearray(preview.tobytes()), dtype=torch.uint8)
        # Pillow stores [height,width,RGB]; PyTorch expects [RGB,height,width].
        pixels = pixels.reshape(self.size, self.size, 3).permute(2, 0, 1)
        tensor = (pixels.to(torch.float32).div(255) - self.mean) / self.std
        return {"tensor": tensor.contiguous(), "preview": preview, "geometry": geometry}

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


def goal_crop(image, goal_xy, size, *, mode="L"):
    """Crop a ``size`` square centered on ``goal_xy`` at 1x scale.

    Returns the crop, converted to Pillow ``mode`` (grayscale by default), and
    its geometry. Box parts outside the image repeat the nearest edge pixel;
    ``padding_ltrb`` counts them per side and ``resized_size_wh`` is the size
    of the content taken from the image.
    """
    check_crop_size(size)
    converted = image.convert(mode)
    width, height = converted.size
    # Round half up; Python's round() sends .5 to the nearest even pixel.
    cx, cy = (math.floor(value + 0.5) for value in goal_xy)
    if not (0 <= cx < width and 0 <= cy < height):
        raise ValueError("goal_xy is outside the image")
    left, top = cx - size // 2, cy - size // 2
    right, bottom = left + size, top + size
    pad_l, pad_t = max(0, -left), max(0, -top)
    pad_r, pad_b = max(0, right - width), max(0, bottom - height)
    inner = converted.crop((left + pad_l, top + pad_t, right - pad_r, bottom - pad_b))
    inner_w, inner_h = inner.size

    crop = Image.new(mode, (size, size))
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


def _check_stats(stats):
    mean, std = stats
    if not (math.isfinite(mean) and math.isfinite(std) and std > 0):
        raise ValueError("channel stats must be a finite mean and positive std")
    return (float(mean), float(std))


def _gray_tensor(images, size):
    """Stack ``size`` square Pillow "L" images into floats [N,S,S] in 0-1."""
    # A writable byte copy avoids sharing storage with the Pillow images.
    data = bytearray(b"".join(image.tobytes() for image in images))
    pixels = torch.frombuffer(data, dtype=torch.uint8)
    return pixels.reshape(len(images), size, size).to(torch.float32).div(255)


class WirePreprocessor:
    """Goal-centered native-resolution grayscale crops for the frozen ResNet.

    One crop box around ``goal_xy`` serves the unmarked goal and all six
    history frames, since the view is fixed within a case. ``raw`` repeats
    the grayscale crop to three channels with ImageNet normalization.
    ``enhanced`` uses channels raw, black top-hat, and difference; top-hat
    and difference are in 0-1 pixel units, standardized by the given stats
    (see ``channel_stats``). ``zero_channels`` are zeroed after normalization
    for ablations. ``context_size`` adds a downscaled full view per image.
    """

    def __init__(self, crop_size=256, channels="raw", *, tophat_size=11,
                 difference="median", tophat_stats=(0.0, 1.0),
                 difference_stats=(0.0, 1.0), zero_channels=(), context_size=None):
        check_crop_size(crop_size)
        if channels not in WIRE_CHANNEL_MODES:
            raise ValueError("channels must be one of {}".format(WIRE_CHANNEL_MODES))
        if type(tophat_size) is not int or tophat_size < 3 or tophat_size % 2 == 0:
            raise ValueError("tophat_size must be an odd integer of at least 3")
        if difference not in DIFFERENCE_MODES:
            raise ValueError("difference must be one of {}".format(DIFFERENCE_MODES))
        if any(type(c) is not int or c not in (0, 1, 2) for c in zero_channels):
            raise ValueError("zero_channels may only contain 0, 1, and 2")
        if context_size is not None:
            check_crop_size(context_size)
        self.crop_size = crop_size
        self.channels = channels
        self.tophat_size = tophat_size
        self.difference = difference
        self.tophat_stats = _check_stats(tophat_stats)
        self.difference_stats = _check_stats(difference_stats)
        self.zero_channels = tuple(sorted(set(zero_channels)))
        self.context_size = context_size
        self.mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)

    def _normalize(self, gray, tophat=None, difference=None):
        """Model tensors [N,3,S,S] from 0-1 gray crops and wire channels [N,S,S]."""
        if self.channels == "raw":
            tensor = (gray[:, None].expand(-1, 3, -1, -1) - self.mean) / self.std
        else:
            tensor = torch.stack([
                (gray - MEAN[0]) / STD[0],
                (tophat - self.tophat_stats[0]) / self.tophat_stats[1],
                (difference - self.difference_stats[0]) / self.difference_stats[1],
            ], dim=1)
        for channel in self.zero_channels:
            tensor[:, channel] = 0
        return tensor

    def _crop_case(self, case):
        """Crop goal_raw and history with one box; returns crops and gray [7,S,S]."""
        check_history(case)
        if len(case["goal_xy"]) != 2:
            raise ValueError("Expected goal_xy to hold two coordinates")
        crops = [goal_crop(image, case["goal_xy"], self.crop_size)
                 for image in [case["goal_raw"], *case["history"]]]
        return crops, _gray_tensor([crop for crop, _ in crops], self.crop_size)

    def _wire_channels(self, gray):
        """Unnormalized top-hat and difference for gray [goal, *history] crops."""
        tophat = black_tophat(gray[:, None], self.tophat_size)[:, 0]
        history = gray[1:]
        if self.difference == "median":
            # Six frames have two middle values; their mean is the median.
            difference = history - history.sort(dim=0).values[2:4].mean(dim=0)
        else:
            difference = torch.zeros_like(history)
            difference[1:] = history[1:] - history[:-1]
        # The goal has no timeline, so its difference channel is zero.
        return tophat, torch.cat([torch.zeros_like(gray[:1]), difference])

    def _context(self, image):
        """Downscaled full view: normalized [3,C,C] tensor, preview, geometry."""
        preview, geometry = fit_square(image.convert("L"), self.context_size, CONTEXT_FILL)
        gray = _gray_tensor([preview], self.context_size)[0]
        tensor = (gray.expand(3, -1, -1) - self.mean) / self.std
        return {"tensor": tensor.contiguous(), "preview": preview, "geometry": geometry}

    def prepare_image(self, image, goal_xy):
        """Return a normalized [3,S,S] tensor, grayscale preview, and geometry.

        A single image has no timeline, so in ``enhanced`` mode its difference
        channel is zero, as for the goal.
        """
        crop, geometry = goal_crop(image, goal_xy, self.crop_size)
        gray = _gray_tensor([crop], self.crop_size)
        if self.channels == "raw":
            tensor = self._normalize(gray)
        else:
            tophat = black_tophat(gray[:, None], self.tophat_size)[:, 0]
            tensor = self._normalize(gray, tophat, torch.zeros_like(gray))
        return {"tensor": tensor[0].contiguous(), "preview": crop, "geometry": geometry}

    def prepare_case(self, case):
        """Prepare one raw case; goal [3,S,S], history [6,3,S,S].

        The goal comes from ``goal_raw`` so the target marking is not cropped.
        Metadata is copied unchanged, as in ``DINOv2Preprocessor``. With
        ``context_size`` set, ``goal_context`` [3,C,C] and ``history_context``
        [6,3,C,C] are added; ``zero_channels`` does not apply to them.
        """
        crops, gray = self._crop_case(case)
        if self.channels == "raw":
            tensors = self._normalize(gray)
        else:
            tensors = self._normalize(gray, *self._wire_channels(gray))
        goal, *history = [
            {"tensor": tensor.clone(), "preview": crop, "geometry": geometry}
            for tensor, (crop, geometry) in zip(tensors, crops)]
        preparation = {
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
            "zero_channels": self.zero_channels,
        }
        if self.channels == "enhanced":
            preparation.update({
                "channel_order": ("raw", "black_tophat", "difference"),
                "raw_normalization": "ImageNet channel 0",
                "tophat_size": self.tophat_size,
                "difference": self.difference,
                "median": "mean of the two middle history values",
                "goal_difference": "zeros (the goal has no timeline)",
                "wire_pixel_scale": "uint8 / 255, then (value - mean) / std",
                "tophat_stats": self.tophat_stats,
                "difference_stats": self.difference_stats,
            })
        prepared = assemble_case(case, goal, history, preparation)
        if self.context_size is not None:
            goal = self._context(case["goal_raw"])
            history = [self._context(frame) for frame in case["history"]]
            prepared["goal_context"] = goal["tensor"]
            prepared["history_context"] = torch.stack([frame["tensor"] for frame in history])
            prepared["previews"]["goal_context"] = goal["preview"]
            prepared["previews"]["history_context"] = [frame["preview"] for frame in history]
            prepared["preparation"].update({
                "context_size": self.context_size,
                "context": "Pillow L, proportional bicubic resize, centered padding, "
                           "repeated to three channels, ImageNet normalization",
                "context_fill": CONTEXT_FILL,
                "goal_context": goal["geometry"],
                "history_context": [frame["geometry"] for frame in history],
            })
        return prepared


# Wire-only flags, keyed by their WirePreprocessor argument names.
WIRE_FLAGS = {"crop_size": "--crop-size", "channels": "--channels",
              "tophat_size": "--tophat-size", "difference": "--difference",
              "context_size": "--context-size"}


def add_preprocessor_arguments(parser):
    """Add the shared model-input flags; ``build_preprocessor`` reads them.

    Wire defaults are applied there, so flags given for the other
    preprocessor can be rejected instead of silently ignored.
    """
    group = parser.add_argument_group("model input preparation")
    group.add_argument("--preprocessor", choices=("dinov2", "wire"), default="dinov2")
    group.add_argument("--size", type=int, default=None,
                       help="dinov2 input size, a multiple of 14 (default 518)")
    group.add_argument("--crop-size", type=int, default=None,
                       help="wire crop size, a multiple of 32 (default 256)")
    group.add_argument("--channels", choices=WIRE_CHANNEL_MODES, default=None,
                       help="wire channel mode (default raw)")
    group.add_argument("--tophat-size", type=int, default=None,
                       help="enhanced top-hat window, odd (default 11)")
    group.add_argument("--difference", choices=DIFFERENCE_MODES, default=None,
                       help="enhanced difference channel (default median)")
    group.add_argument("--context-size", type=int, default=None,
                       help="wire full-view context size, a multiple of 32 (default off)")


def build_preprocessor(args):
    """Build the preprocessor selected by ``add_preprocessor_arguments`` flags.

    Raises ValueError for flags that do not apply to the selected options.
    """
    given = {name: getattr(args, name) for name in WIRE_FLAGS
             if getattr(args, name) is not None}
    if args.preprocessor == "dinov2":
        if given:
            raise ValueError("{} only apply to --preprocessor wire".format(
                ", ".join(WIRE_FLAGS[name] for name in given)))
        return DINOv2Preprocessor(size=518 if args.size is None else args.size)
    if args.size is not None:
        raise ValueError("--size applies to --preprocessor dinov2; use --crop-size with wire")
    if given.get("channels") != "enhanced":
        unused = [WIRE_FLAGS[name] for name in ("tophat_size", "difference") if name in given]
        if unused:
            raise ValueError("{} only apply to --channels enhanced".format(", ".join(unused)))
    return WirePreprocessor(**given)


def channel_stats(preprocessor, cases):
    """Top-hat and difference (mean, std) over the history crops of ``cases``.

    Call this on TRAINING cases only; validation or test statistics would
    leak into preprocessing. Uses the preprocessor's crop, top-hat, and
    difference settings and ignores its current stats. The goal is excluded
    because its difference channel is zero by design. Returns keyword
    arguments for ``WirePreprocessor``.
    """
    totals = {"tophat": [0.0, 0.0, 0], "difference": [0.0, 0.0, 0]}
    for case in cases:
        _, gray = preprocessor._crop_case(case)
        tophat, difference = preprocessor._wire_channels(gray)
        for name, values in (("tophat", tophat[1:]), ("difference", difference[1:])):
            values = values.double()
            totals[name][0] += values.sum().item()
            totals[name][1] += values.square().sum().item()
            totals[name][2] += values.numel()
    if not totals["tophat"][2]:
        raise ValueError("channel_stats needs at least one case")
    stats = {}
    for name, (total, squares, count) in totals.items():
        mean = total / count
        stats[name + "_stats"] = (mean, math.sqrt(max(squares / count - mean ** 2, 0.0)))
    return stats
