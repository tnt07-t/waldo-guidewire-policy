"""Diagnostics for frozen ResNet-18 512-d features (not part of the policy CNN).

Probes whether pooled vectors a later GRU would receive change under controlled
edits. They do not train, do not add heads to ``cnn.py``, and do not claim wire
detection, target understanding, or advance/retract recognition.

Probes (all scored with L2 distance in 512-d space):
1. Occlusion sensitivity on history frames (not verified wire detection).
2. Goal red-marker region perturbation vs translated same-shape controls,
   with edits applied on RGB before the shared preprocessor. With
   ``--preprocessor wire`` the probe uses the marked goal cropped with the
   wire box, so the marker and controls stay inside the model input.
3. Temporal feature variation vs aligned depth changes (magnitude only).

Real cases run on Perlmutter. Optional ``--output-dir`` writes verification
overlays that must stay on Perlmutter. Invented-input unit tests cover
bookkeeping.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from preprocessing import MEAN, STD, WirePreprocessor, goal_crop


# Red-marker thresholds on uint8 RGB in [0, 255], before ImageNet normalization.
# A pixel counts as marker-like when R is bright and dominates G and B.
RED_MIN = 150
RED_DOMINANCE = 40
FEATURE_DISTANCE = "L2"
OCCLUDE_FILL_NORMALIZED = 0.0  # ~ImageNet mean after DINOv2Preprocessor
REPLACE_RGB = tuple(round(value * 255) for value in MEAN)


def l2_distance(a, b):
    """Scalar L2 between two feature vectors."""
    return torch.linalg.vector_norm(a - b).item()


def cosine_distance(a, b):
    """1 - cosine similarity; 0 means identical direction."""
    a = a / torch.linalg.vector_norm(a).clamp_min(1e-8)
    b = b / torch.linalg.vector_norm(b).clamp_min(1e-8)
    return (1.0 - torch.dot(a, b)).item()


def encode_image(encoder, image):
    """Encode one [3,H,W] tensor to a detached [512] feature (eval, no grad)."""
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        features = encoder(image.unsqueeze(0))
    if was_training:
        encoder.train()
    if features.shape != (1, encoder.feature_dim):
        raise ValueError("Expected a single 512-d feature vector")
    if not torch.isfinite(features).all():
        raise ValueError("Encoder returned non-finite features")
    return features[0].detach()


def rgb_uint8_array(image_rgb):
    """Return HxWx3 uint8 tensor from a PIL RGB image (copy, writable)."""
    rgb = image_rgb.convert("RGB")
    pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
    return pixels.reshape(rgb.height, rgb.width, 3).clone()


def red_marker_mask_from_rgb(image_rgb, *, min_red=RED_MIN, dominance=RED_DOMINANCE):
    """Boolean [H,W] red-ish mask on pre-normalization RGB in [0, 255].

    Arithmetic uses int16 so ``green + dominance`` cannot wrap as uint8.
    Thresholds: R >= min_red and R >= G + dominance and R >= B + dominance.
    """
    if min_red < 0 or dominance < 0:
        raise ValueError("min_red and dominance must be non-negative")
    pixels = rgb_uint8_array(image_rgb).to(torch.int16)
    red, green, blue = pixels.unbind(-1)
    return (
        (red >= min_red)
        & (red >= green + dominance)
        & (red >= blue + dominance)
    )


def _pil_from_rgb_tensor(pixels_u8):
    height, width, _ = pixels_u8.shape
    flat = pixels_u8.contiguous().reshape(-1).cpu().to(torch.uint8)
    return Image.frombytes("RGB", (width, height), bytes(flat.tolist()))


def replace_masked_rgb(image_rgb, mask, fill_rgb=REPLACE_RGB):
    """Copy RGB image with masked pixels set to ``fill_rgb`` (before prep)."""
    pixels = rgb_uint8_array(image_rgb)
    if mask.shape != pixels.shape[:2]:
        raise ValueError("mask spatial size must match the RGB image")
    out = pixels.clone()
    out[mask] = torch.tensor(fill_rgb, dtype=torch.uint8)
    return _pil_from_rgb_tensor(out)


def translated_control_masks(marker_mask, *, max_controls=8):
    """Same-shape masks translated to valid non-overlapping locations.

    Returns an empty list when the marker is empty or no valid placement exists.
    """
    if marker_mask.dtype != torch.bool:
        marker_mask = marker_mask.bool()
    height, width = marker_mask.shape
    area = int(marker_mask.sum())
    if area <= 0:
        return []

    ys, xs = torch.where(marker_mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    box_h = y1 - y0 + 1
    box_w = x1 - x0 + 1
    local = marker_mask[y0:y1 + 1, x0:x1 + 1]

    # Deterministic scan of top-left placements for the marker bounding box.
    controls = []
    step_y = max(1, box_h // 2)
    step_x = max(1, box_w // 2)
    for top in range(0, height - box_h + 1, step_y):
        for left in range(0, width - box_w + 1, step_x):
            if top == y0 and left == x0:
                continue
            candidate = torch.zeros_like(marker_mask)
            candidate[top:top + box_h, left:left + box_w] = local
            if int(candidate.sum()) != area:
                continue
            if bool((candidate & marker_mask).any()):
                continue
            controls.append(candidate)
            if len(controls) >= max_controls:
                return controls
    return controls


def save_mask_overlay(image_rgb, mask, path, *, title_note="detected red mask"):
    """Save RGB overlay with mask pixels highlighted (verification only)."""
    base = image_rgb.convert("RGB").copy()
    draw = ImageDraw.Draw(base, "RGBA")
    ys, xs = torch.where(mask.bool())
    for y, x in zip(ys.tolist(), xs.tolist()):
        draw.point((x, y), fill=(0, 255, 0, 160))
    draw.text((8, 8), title_note, fill=(255, 255, 0, 255))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.save(path)


def encoder_device(encoder):
    """Device of the first parameter, or CPU when the module is parameter-free."""
    try:
        return next(encoder.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def prepare_and_encode(encoder, preprocessor, image_rgb):
    """Shared prep then encode; returns 512-d feature on encoder device."""
    tensor = preprocessor.prepare_image(image_rgb)["tensor"]
    return encode_image(encoder, tensor.to(encoder_device(encoder)))


def safe_ratio(numerator, denominator):
    """Marker/control style ratio; None when the denominator is ~0."""
    if abs(denominator) < 1e-8:
        return None
    return float(numerator / denominator)


def goal_marker_sensitivity(encoder, preprocessor, goal_rgb, *, output_dir=None):
    """Sensitivity to replacing the red-marker region before preprocessing.

    Compares the marker edit against several same-shape translated controls.
    Does not prove the encoder understands the target or its meaning.
    """
    marker = red_marker_mask_from_rgb(goal_rgb)
    marker_pixels = int(marker.sum())
    result = {
        "status": "ok",
        "marker_pixels": marker_pixels,
        "thresholds": {"min_red": RED_MIN, "dominance": RED_DOMINANCE, "range": "[0,255]"},
        "replacement_rgb": REPLACE_RGB,
        "feature_distance": FEATURE_DISTANCE,
        "interpretation": (
            "Sensitivity to a marker-region RGB perturbation before preprocessing. "
            "Does not prove target understanding."
        ),
    }
    if marker_pixels <= 0:
        result["status"] = "unavailable"
        result["reason"] = "empty red-marker mask"
        return result

    if output_dir is not None:
        save_mask_overlay(
            goal_rgb,
            marker,
            Path(output_dir) / "goal_red_marker_overlay.png",
            title_note="red-marker mask (pre-normalization RGB)",
        )
        result["marker_overlay"] = "goal_red_marker_overlay.png"

    controls = translated_control_masks(marker)
    if not controls:
        result["status"] = "unavailable"
        result["reason"] = "no valid non-overlapping translated control masks"
        return result

    baseline = prepare_and_encode(encoder, preprocessor, goal_rgb)
    marker_image = replace_masked_rgb(goal_rgb, marker)
    marker_delta = l2_distance(
        baseline, prepare_and_encode(encoder, preprocessor, marker_image)
    )
    control_deltas = []
    for control in controls:
        edited = replace_masked_rgb(goal_rgb, control)
        control_deltas.append(
            l2_distance(baseline, prepare_and_encode(encoder, preprocessor, edited))
        )
    control_tensor = torch.tensor(control_deltas, dtype=torch.float32)
    control_mean = float(control_tensor.mean())
    result.update({
        "n_controls": len(control_deltas),
        "marker_l2": marker_delta,
        "control_l2_values": control_deltas,
        "control_l2_mean": control_mean,
        "control_l2_median": float(control_tensor.median()),
        "control_l2_min": float(control_tensor.min()),
        "control_l2_max": float(control_tensor.max()),
        "marker_over_control_mean": safe_ratio(marker_delta, control_mean),
        "marker_over_control_median": safe_ratio(
            marker_delta, float(control_tensor.median())
        ),
    })
    return result


def content_bounds(geometry):
    """Inclusive-exclusive content box (top, left, bottom, right) in prepared px."""
    left, top, right, bottom_pad = geometry["padding_ltrb"]
    width, height = geometry["resized_size_wh"]
    return top, left, top + height, left + width


def patch_inside_content(top, left, patch, bounds):
    content_top, content_left, content_bottom, content_right = bounds
    return (
        top >= content_top
        and left >= content_left
        and top + patch <= content_bottom
        and left + patch <= content_right
    )


def occlude_square(image, top, left, patch, fill=OCCLUDE_FILL_NORMALIZED):
    """Copy ``image`` [3,H,W] with one square replaced by ``fill``."""
    _, height, width = image.shape
    if patch <= 0:
        raise ValueError("patch must be positive")
    out = image.clone()
    bottom = min(height, top + patch)
    right = min(width, left + patch)
    if top >= height or left >= width or bottom <= top or right <= left:
        raise ValueError("occlusion square is outside the image")
    out[:, top:bottom, left:right] = fill
    return out


def grid_occlusion_deltas(encoder, image, *, patch, stride, geometry):
    """L2 feature change for content-only grid occlusions on [3,H,W]."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected image [3, H, W]")
    _, height, width = image.shape
    if patch <= 0 or stride <= 0:
        raise ValueError("patch and stride must be positive")

    bounds = content_bounds(geometry)
    baseline = encode_image(encoder, image)
    tops, lefts, deltas = [], [], []
    for top in range(0, max(1, height - patch + 1), stride):
        for left in range(0, max(1, width - patch + 1), stride):
            if not patch_inside_content(top, left, patch, bounds):
                continue
            occluded = occlude_square(image, top, left, patch)
            tops.append(top)
            lefts.append(left)
            deltas.append(l2_distance(baseline, encode_image(encoder, occluded)))

    return (
        torch.tensor(tops, dtype=torch.int64),
        torch.tensor(lefts, dtype=torch.int64),
        torch.tensor(deltas, dtype=torch.float32),
    )


def prepared_to_original_xy(x_prep, y_prep, geometry):
    """Map prepared-pixel coordinates into original image coordinates."""
    if "crop_box_ltrb" in geometry:
        # Wire crops are 1x: prepared pixels are offsets from the box corner.
        box_left, box_top, _, _ = geometry["crop_box_ltrb"]
        return box_left + x_prep, box_top + y_prep
    left, top, _, _ = geometry["padding_ltrb"]
    orig_w, orig_h = geometry["original_size_wh"]
    resized_w, resized_h = geometry["resized_size_wh"]
    scale_x = resized_w / orig_w
    scale_y = resized_h / orig_h
    return (x_prep - left) / scale_x, (y_prep - top) / scale_y


def save_occlusion_heatmap(original_rgb, geometry, tops, lefts, deltas, patch, path):
    """Overlay occlusion L2 scores on the original frame; outline the max patch."""
    orig_w, orig_h = original_rgb.size
    heat = torch.zeros(orig_h, orig_w, dtype=torch.float32)
    weight = torch.zeros(orig_h, orig_w, dtype=torch.float32)
    best = int(torch.argmax(deltas)) if len(deltas) else None
    best_box = None
    for index, (top, left, delta) in enumerate(zip(tops.tolist(), lefts.tolist(), deltas.tolist())):
        x0, y0 = prepared_to_original_xy(left, top, geometry)
        x1, y1 = prepared_to_original_xy(left + patch, top + patch, geometry)
        xa, xb = sorted((int(round(x0)), int(round(x1))))
        ya, yb = sorted((int(round(y0)), int(round(y1))))
        xa, xb = max(0, xa), min(orig_w, xb)
        ya, yb = max(0, ya), min(orig_h, yb)
        if xb <= xa or yb <= ya:
            continue
        heat[ya:yb, xa:xb] += float(delta)
        weight[ya:yb, xa:xb] += 1.0
        if index == best:
            best_box = (xa, ya, xb, yb)

    base = original_rgb.convert("RGB").copy()
    if float(weight.max()) > 0:
        norm = heat / weight.clamp_min(1.0)
        norm = norm / norm.max().clamp_min(1e-8)
        pixels = rgb_uint8_array(base).to(torch.float32)
        # Red tint proportional to occlusion sensitivity.
        pixels[..., 0] = torch.clamp(pixels[..., 0] + 180 * norm, max=255)
        pixels[..., 1] = torch.clamp(pixels[..., 1] * (1.0 - 0.4 * norm), min=0)
        pixels[..., 2] = torch.clamp(pixels[..., 2] * (1.0 - 0.4 * norm), min=0)
        base = _pil_from_rgb_tensor(pixels.to(torch.uint8))
    draw = ImageDraw.Draw(base)
    if best_box is not None:
        draw.rectangle(best_box, outline=(0, 255, 255), width=3)
        draw.text((8, 8), "max occlusion-sensitivity patch (not verified wire)", fill=(255, 255, 0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.save(path)


def summarize_occlusion_grid(tops, lefts, deltas, patch, geometry):
    """Aggregates for one frame's content-only occlusion grid."""
    if len(deltas) == 0:
        return {
            "status": "unavailable",
            "reason": "no content patches for patch/stride and padding",
            "n_patches": 0,
        }
    best = int(torch.argmax(deltas))
    top = tops[best].item()
    left = lefts[best].item()
    cx_prep = left + patch / 2
    cy_prep = top + patch / 2
    cx_orig, cy_orig = prepared_to_original_xy(cx_prep, cy_prep, geometry)
    median = float(deltas.median())
    maximum = float(deltas.max())
    return {
        "status": "ok",
        "n_patches": len(deltas),
        "max_l2": maximum,
        "median_l2": median,
        "mean_l2": float(deltas.mean()),
        "max_over_median": safe_ratio(maximum, median),
        "argmax_center_prepared_xy": (cx_prep, cy_prep),
        "argmax_center_original_xy": (cx_orig, cy_orig),
        "patch": patch,
        "feature_distance": FEATURE_DISTANCE,
        "replacement": OCCLUDE_FILL_NORMALIZED,
    }


def occlusion_sensitivity_history(
    encoder,
    history_tensors,
    history_rgb,
    geometries,
    *,
    patch,
    stride,
    output_dir=None,
):
    """Content-only occlusion sensitivity on six history frames.

    Highest-change patches are not labeled as wire without ground truth.
    """
    if history_tensors.ndim != 4 or history_tensors.shape[0] != 6:
        raise ValueError("Expected history tensors [6, 3, H, W]")
    if len(history_rgb) != 6 or len(geometries) != 6:
        raise ValueError("Expected six history RGB frames and geometries")

    frames = []
    scores = []
    for index in range(6):
        tops, lefts, deltas = grid_occlusion_deltas(
            encoder,
            history_tensors[index],
            patch=patch,
            stride=stride,
            geometry=geometries[index],
        )
        summary = summarize_occlusion_grid(
            tops, lefts, deltas, patch, geometries[index]
        )
        summary["frame"] = index
        summary["chronological_order"] = "oldest_to_newest"
        if summary["status"] == "ok" and output_dir is not None:
            name = f"history_frame_{index}_occlusion_heatmap.png"
            save_occlusion_heatmap(
                history_rgb[index],
                geometries[index],
                tops,
                lefts,
                deltas,
                patch,
                Path(output_dir) / name,
            )
            summary["heatmap"] = name
        frames.append(summary)
        if summary["status"] == "ok":
            scores.append(summary["max_l2"])

    report = {
        "name": "occlusion_sensitivity",
        "not_wire_detection": True,
        "patch": patch,
        "stride": stride,
        "replacement_normalized": OCCLUDE_FILL_NORMALIZED,
        "feature_distance": FEATURE_DISTANCE,
        "padding_excluded": True,
        "frames": frames,
        "interpretation": (
            "Occlusion sensitivity of pooled 512-d features. "
            "Max-change patches are not verified wire locations."
        ),
    }
    if scores:
        report["max_l2_scores"] = scores
        report["score_range"] = float(max(scores) - min(scores))
        report["score_std"] = float(torch.tensor(scores).std(unbiased=False))
    else:
        report["status"] = "unavailable"
        report["reason"] = "no valid content patches on any frame"
    return report


def pearson_corr(x, y):
    """Pearson correlation, or None if undefined (constant / too short)."""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return None
    if float(x.std(unbiased=False)) < 1e-12 or float(y.std(unbiased=False)) < 1e-12:
        return None
    x_n = (x - x.mean()) / x.std(unbiased=False)
    y_n = (y - y.mean()) / y.std(unbiased=False)
    return float((x_n * y_n).mean())


def temporal_feature_diagnostics(history_features, depths):
    """Consecutive feature L2 vs signed/absolute depth changes.

    Feature variation does not establish wire-motion understanding,
    advance/retract recognition, or downstream prediction quality.
    Correlation uses absolute depth steps only (magnitude association).
    """
    report = {
        "chronological_order": "oldest_to_newest (dataset paths[1:])",
        "feature_distance": FEATURE_DISTANCE,
        "interpretation": (
            "Feature variation over history does not establish wire-motion "
            "understanding, advance/retract recognition, or prediction quality."
        ),
    }
    if history_features is None or depths is None:
        report["status"] = "unavailable"
        report["reason"] = "missing history features or depths"
        return report
    if history_features.ndim != 2 or history_features.shape[0] != 6:
        raise ValueError("Expected history features [6, D]")
    depth = torch.as_tensor(depths, dtype=torch.float32).reshape(-1)
    if depth.numel() != 6:
        report["status"] = "unavailable"
        report["reason"] = "depths are not length 6; cannot align with history"
        return report
    if not torch.isfinite(depth).all() or not torch.isfinite(history_features).all():
        report["status"] = "unavailable"
        report["reason"] = "non-finite features or depths"
        return report

    step_l2 = [
        l2_distance(history_features[t], history_features[t + 1])
        for t in range(5)
    ]
    depth_signed = (depth[1:] - depth[:-1]).tolist()
    depth_abs = torch.abs(depth[1:] - depth[:-1]).tolist()
    report.update({
        "status": "ok",
        "n_steps": 5,
        "aligned_depths": depth.tolist(),
        "step_feature_l2": step_l2,
        "step_depth_signed": depth_signed,
        "step_depth_abs": depth_abs,
        "mean_step_feature_l2": float(sum(step_l2) / len(step_l2)),
        "first_last_feature_l2": l2_distance(history_features[0], history_features[5]),
        "first_last_cosine_distance": cosine_distance(
            history_features[0], history_features[5]
        ),
        "depth_first_last_signed": float(depth[5] - depth[0]),
        "feature_l2_vs_abs_depth_corr": pearson_corr(step_l2, depth_abs),
        "corr_note": (
            "Correlation is between nonnegative feature L2 and absolute depth "
            "change only (magnitude association). It is not direction recognition."
        ),
    })
    if report["feature_l2_vs_abs_depth_corr"] is None:
        report["corr_status"] = "unavailable"
        report["corr_reason"] = "constant sequence, missing variance, or insufficient spread"
    else:
        report["corr_status"] = "ok"
    return report


def goal_vs_history_separation(goal_features, history_features):
    """How distinct the goal 512-d vector is from each history frame."""
    distances = [
        l2_distance(goal_features, history_features[t]) for t in range(6)
    ]
    return {
        "goal_history_l2": distances,
        "goal_history_l2_mean": float(sum(distances) / len(distances)),
        "goal_history_l2_min": float(min(distances)),
    }


def default_occlusion_size(preprocessor):
    """Occlusion patch/stride default: 74 for 518 inputs, crop/8 for wire crops."""
    if isinstance(preprocessor, WirePreprocessor):
        return preprocessor.crop_size // 8
    return 74


class _FixedCenterCrop:
    """Single-image ``prepare_image`` for wire crops already centered on the goal."""

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        half = preprocessor.crop_size // 2
        self.center = (half, half)

    def prepare_image(self, image):
        return self.preprocessor.prepare_image(image, self.center)


def marker_probe_inputs(preprocessor, case):
    """Marked goal and single-image preprocessor for the marker probe.

    Wire crops ignore everything outside the goal box, so the probe runs on
    the MARKED goal cropped with that box (never the unmarked goal); the
    marker mask and its translated controls then stay inside the model input.
    """
    if not isinstance(preprocessor, WirePreprocessor):
        return case["goal"], preprocessor
    crop, _ = goal_crop(case["goal"], case["goal_xy"], preprocessor.crop_size, mode="RGB")
    return crop, _FixedCenterCrop(preprocessor)


def run_case_diagnostics(
    encoder,
    preprocessor,
    case,
    *,
    patch,
    stride,
    output_dir=None,
):
    """Run diagnostics on one raw case; optional overlays under ``output_dir``."""
    encoder.eval()
    prepared = preprocessor.prepare_case(case)
    device = encoder_device(encoder)
    history = prepared["history"].to(device)
    goal = prepared["goal"].to(device)

    with torch.no_grad():
        history_features = encoder(history.unsqueeze(0))[0].detach()
        goal_features = encoder(goal.unsqueeze(0))[0].detach()

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    occlusion = occlusion_sensitivity_history(
        encoder,
        history,
        case["history"],
        prepared["preparation"]["history"],
        patch=patch,
        stride=stride,
        output_dir=output_dir,
    )
    marker_goal, marker_preprocessor = marker_probe_inputs(preprocessor, case)
    goal_probe = goal_marker_sensitivity(
        encoder, marker_preprocessor, marker_goal, output_dir=output_dir
    )
    if marker_preprocessor is not preprocessor:
        goal_probe["input"] = "marked goal cropped with the wire box"
    separation = goal_vs_history_separation(goal_features, history_features)
    temporal = temporal_feature_diagnostics(
        history_features.cpu(),
        prepared.get("depth_relative_mm"),
    )
    # Confirm depth list aligns with the six chronological history frames.
    temporal["history_depth_lengths"] = {
        "history_frames": len(case["history"]),
        "depth_relative_mm": len(case["depth_relative_mm"]),
        "history_offsets_s": len(case["history_offsets_s"]),
        "aligned": (
            len(case["history"])
            == len(case["depth_relative_mm"])
            == len(case["history_offsets_s"])
            == 6
        ),
    }
    return {
        "occlusion_sensitivity": occlusion,
        "goal_marker_sensitivity": goal_probe,
        "goal_separation": separation,
        "temporal": temporal,
        "history_features_shape": tuple(history_features.shape),
        "goal_features_shape": tuple(goal_features.shape),
        "encoder_training": encoder.training,
        "backbone_training": encoder.backbone.training,
    }


def _fmt_optional(value, digits=6):
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return round(value, digits)
    return value


def _print_report(report):
    """Print permitted aggregates only (no feature vectors)."""
    occ = report["occlusion_sensitivity"]
    print("Occlusion sensitivity (history; not verified wire detection):")
    print("  patch/stride:", occ["patch"], occ["stride"])
    print("  replacement_normalized:", occ["replacement_normalized"])
    print("  feature_distance:", occ["feature_distance"])
    if "score_range" in occ:
        print("  score_range:", round(occ["score_range"], 6))
        print("  score_std:", round(occ["score_std"], 6))
    for frame in occ["frames"]:
        if frame["status"] != "ok":
            print(f"  frame {frame['frame']}: unavailable ({frame.get('reason')})")
            continue
        cx, cy = frame["argmax_center_original_xy"]
        print(
            f"  frame {frame['frame']}: max_l2={frame['max_l2']:.6f} "
            f"max/median={_fmt_optional(frame['max_over_median'], 3)} "
            f"argmax_original_xy=({cx:.1f},{cy:.1f})"
        )

    goal = report["goal_marker_sensitivity"]
    print("Goal marker-region perturbation sensitivity:")
    print("  status:", goal["status"])
    if "input" in goal:
        print("  input:", goal["input"])
    if goal["status"] != "ok":
        print("  reason:", goal.get("reason"))
    else:
        print("  marker_pixels:", goal["marker_pixels"])
        print("  marker_l2:", round(goal["marker_l2"], 6))
        print("  control_l2_mean:", round(goal["control_l2_mean"], 6))
        print("  control_l2_median:", round(goal["control_l2_median"], 6))
        print("  control_l2_min/max:",
              round(goal["control_l2_min"], 6), round(goal["control_l2_max"], 6))
        print(
            "  marker/control_mean:",
            _fmt_optional(goal["marker_over_control_mean"], 3),
        )
        print(
            "  marker/control_median:",
            _fmt_optional(goal["marker_over_control_median"], 3),
        )
        print(" ", goal["interpretation"])

    sep = report["goal_separation"]
    print("Goal vs history L2:")
    print("  mean:", round(sep["goal_history_l2_mean"], 6))
    print("  min:", round(sep["goal_history_l2_min"], 6))

    temporal = report["temporal"]
    print("Temporal feature variation (not direction/action recognition):")
    print("  status:", temporal.get("status"))
    print("  aligned:", temporal.get("history_depth_lengths", {}).get("aligned"))
    if temporal.get("status") == "ok":
        print("  mean_step_feature_l2:", round(temporal["mean_step_feature_l2"], 6))
        print("  first_last_feature_l2:", round(temporal["first_last_feature_l2"], 6))
        print(
            "  first_last_cosine_distance:",
            round(temporal["first_last_cosine_distance"], 6),
        )
        print("  step_depth_signed:", [round(v, 6) for v in temporal["step_depth_signed"]])
        print("  step_depth_abs:", [round(v, 6) for v in temporal["step_depth_abs"]])
        print(
            "  feature_l2_vs_abs_depth_corr:",
            _fmt_optional(temporal["feature_l2_vs_abs_depth_corr"]),
            f"({temporal.get('corr_status')})",
        )
        print(" ", temporal["interpretation"])


def main():
    # Real-data entrypoint only: keep top-level imports free of torchvision so
    # invented-input unit tests can run without the CNN dependency stack.
    from cnn import FrameEncoder
    from dataset import GuidewireDataset
    from preprocessing import add_preprocessor_arguments, build_preprocessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    parser.add_argument("--split-position", type=int, required=True)
    parser.add_argument(
        "--partition",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--case-index", type=int, default=0)
    add_preprocessor_arguments(parser)
    parser.add_argument("--remove-maxpool", action="store_true")
    parser.add_argument(
        "--patch",
        type=int,
        default=None,
        help="Occlusion square side in prepared pixels (default 74; crop/8 for wire)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Occlusion grid stride in prepared pixels (default 74; crop/8 for wire)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for mask/heatmap overlays (keep on Perlmutter)",
    )
    args = parser.parse_args()

    if not args.data_dir:
        parser.error("Set DATA_DIR or provide --data-dir")
    if args.case_index < 0:
        parser.error("--case-index must be non-negative")
    try:
        preprocessor = build_preprocessor(args)
    except ValueError as exc:
        parser.error(str(exc))
    # Defaults were sized for 518 inputs; wire crops scale them down.
    if args.patch is None:
        args.patch = default_occlusion_size(preprocessor)
    if args.stride is None:
        args.stride = default_occlusion_size(preprocessor)

    dataset = GuidewireDataset(
        args.data_dir,
        split_position=args.split_position,
        partition=args.partition,
    )
    if len(dataset) == 0:
        parser.error("Selected partition is empty")
    if args.case_index >= len(dataset):
        parser.error("--case-index out of range for partition")

    case = dataset[args.case_index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrameEncoder(remove_maxpool=args.remove_maxpool).to(device)
    encoder.eval()

    report = run_case_diagnostics(
        encoder,
        preprocessor,
        case,
        patch=args.patch,
        stride=args.stride,
        output_dir=args.output_dir,
    )

    print("Device:", device)
    print("History features:", report["history_features_shape"])
    print("Goal features:", report["goal_features_shape"])
    print("Encoder training mode:", report["encoder_training"])
    print("Backbone training mode:", report["backbone_training"])
    if args.output_dir:
        print("Overlays written under:", args.output_dir)
    _print_report(report)
    print("CNN 512-d diagnostics finished.")


if __name__ == "__main__":
    main()
