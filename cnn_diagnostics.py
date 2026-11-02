"""Diagnostics for frozen ResNet-18 512-d features (not part of the policy CNN).

These checks ask whether the pooled vectors a later GRU would receive still
carry coarse wire, goal-marker, and motion signal. They do not train, do not
add localization heads to ``cnn.py``, and must not write features or images.

Three probes (all scored in 512-d space):
1. Wire-ish sensitivity: grid occlusion on each history frame; larger feature
   change when a patch matters more to the encoder.
2. Goal marker: occlude red-ish preview pixels versus an equal-area control.
3. Motion: calculations on the history feature sequence from (1), optionally
   compared with relative depth — not a CNN direction head.

Real cases run on Perlmutter only. Invented-input unit tests cover bookkeeping.
"""

from __future__ import annotations

import argparse
import os

import torch

from preprocessing import MEAN, STD


def l2_distance(a, b):
    """Scalar L2 between two feature vectors."""
    return torch.linalg.vector_norm(a - b).item()


def cosine_distance(a, b):
    """1 - cosine similarity; 0 means identical direction."""
    a = a / torch.linalg.vector_norm(a).clamp_min(1e-8)
    b = b / torch.linalg.vector_norm(b).clamp_min(1e-8)
    return (1.0 - torch.dot(a, b)).item()


def encode_image(encoder, image):
    """Encode one [3,H,W] tensor to a detached [512] feature."""
    with torch.no_grad():
        features = encoder(image.unsqueeze(0))
    if features.shape != (1, encoder.feature_dim):
        raise ValueError("Expected a single 512-d feature vector")
    if not torch.isfinite(features).all():
        raise ValueError("Encoder returned non-finite features")
    return features[0]


def occlude_square(image, top, left, patch, fill=0.0):
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


def grid_occlusion_deltas(encoder, image, *, patch, stride):
    """L2 feature change for each grid occlusion on one [3,H,W] image.

    Fill uses 0.0 in normalized space (approx. ImageNet mean after prep).
    Returns tops, lefts, and deltas as 1-d CPU float tensors.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected image [3, H, W]")
    _, height, width = image.shape
    if patch <= 0 or stride <= 0:
        raise ValueError("patch and stride must be positive")

    baseline = encode_image(encoder, image)
    tops = []
    lefts = []
    deltas = []
    for top in range(0, max(1, height - patch + 1), stride):
        for left in range(0, max(1, width - patch + 1), stride):
            occluded = occlude_square(image, top, left, patch)
            delta = l2_distance(baseline, encode_image(encoder, occluded))
            tops.append(top)
            lefts.append(left)
            deltas.append(delta)

    return (
        torch.tensor(tops, dtype=torch.int64),
        torch.tensor(lefts, dtype=torch.int64),
        torch.tensor(deltas, dtype=torch.float32),
    )


def summarize_occlusion_grid(tops, lefts, deltas, height, width, patch):
    """Permitted aggregates for one frame's occlusion grid."""
    if len(deltas) == 0:
        raise ValueError("occlusion grid is empty")
    best = int(torch.argmax(deltas))
    # Patch center in normalized image coordinates (0 at top/left, 1 at bottom/right).
    y = (tops[best].item() + patch / 2) / height
    x = (lefts[best].item() + patch / 2) / width
    return {
        "n_patches": len(deltas),
        "max_l2": float(deltas.max()),
        "median_l2": float(deltas.median()),
        "mean_l2": float(deltas.mean()),
        "max_over_median": float(deltas.max() / deltas.median().clamp_min(1e-8)),
        "argmax_center_xy_norm": (x, y),
    }


def red_marker_mask(preview_rgb, *, min_red=150, margin=40):
    """Boolean [H,W] mask of red-ish pixels from a prepared RGB preview.

    Heuristic for the painted goal circle; not a clinical detector.
    """
    pixels = torch.frombuffer(bytearray(preview_rgb.tobytes()), dtype=torch.uint8)
    pixels = pixels.reshape(preview_rgb.height, preview_rgb.width, 3)
    red, green, blue = pixels.unbind(-1)
    return (
        (red >= min_red)
        & (red >= green + margin)
        & (red >= blue + margin)
    )


def equal_area_control_mask(marker_mask):
    """Axis-aligned box with the same pixel count as ``marker_mask``, if possible.

    Prefer the bottom-right corner so the control rarely overlaps a centered mark.
    """
    height, width = marker_mask.shape
    area = int(marker_mask.sum())
    control = torch.zeros_like(marker_mask)
    if area <= 0:
        return control

    side = max(1, int(area ** 0.5))
    while side * side < area:
        side += 1
    side = min(side, height, width)
    top = height - side
    left = width - side
    control[top:top + side, left:left + side] = True
    # Trim to exact area without growing outside the box.
    flat = control.flatten()
    idx = flat.nonzero(as_tuple=False).flatten()
    if len(idx) > area:
        flat[idx[area:]] = False
    return flat.reshape(height, width)


def occlude_mask(image, mask, fill=0.0):
    """Fill masked pixels on a [3,H,W] tensor."""
    if mask.shape != image.shape[-2:]:
        raise ValueError("mask spatial size must match the image")
    out = image.clone()
    out[:, mask] = fill
    return out


def goal_marker_probe(encoder, goal, preview):
    """Compare 512-d change from marker occlusion vs equal-area control."""
    baseline = encode_image(encoder, goal)
    marker = red_marker_mask(preview)
    control = equal_area_control_mask(marker)
    marker_delta = l2_distance(
        baseline, encode_image(encoder, occlude_mask(goal, marker))
    )
    control_delta = l2_distance(
        baseline, encode_image(encoder, occlude_mask(goal, control))
    )
    return {
        "marker_pixels": int(marker.sum()),
        "control_pixels": int(control.sum()),
        "marker_l2": marker_delta,
        "control_l2": control_delta,
        "marker_over_control": marker_delta / max(control_delta, 1e-8),
    }


def denormalize_stats_note():
    """Document fill convention; values stay in normalized tensor space."""
    return {
        "fill": 0.0,
        "meaning": "approx ImageNet mean after DINOv2Preprocessor normalization",
        "mean": MEAN,
        "std": STD,
    }


def wire_history_probe(encoder, history, *, patch, stride):
    """Occlusion summaries and max-L2 wire scores for history [6,3,H,W]."""
    if history.ndim != 4 or history.shape[0] != 6 or history.shape[1] != 3:
        raise ValueError("Expected history [6, 3, H, W]")
    _, _, height, width = history.shape
    frames = []
    wire_scores = []
    for index in range(6):
        tops, lefts, deltas = grid_occlusion_deltas(
            encoder, history[index], patch=patch, stride=stride
        )
        summary = summarize_occlusion_grid(
            tops, lefts, deltas, height, width, patch
        )
        summary["frame"] = index
        frames.append(summary)
        wire_scores.append(summary["max_l2"])
    return {
        "frames": frames,
        "wire_scores": wire_scores,
        "score_range": float(max(wire_scores) - min(wire_scores)),
        "score_std": float(torch.tensor(wire_scores).std(unbiased=False)),
    }


def direction_from_history_features(history_features, wire_scores=None, depths=None):
    """Motion diagnostics from the 6×512 sequence (and optional wire scores).

    No learnable direction head: only feature deltas and simple correlations.
    """
    if history_features.ndim != 2 or history_features.shape != (6, 512):
        raise ValueError("Expected history features [6, 512]")

    step_l2 = [
        l2_distance(history_features[t], history_features[t + 1])
        for t in range(5)
    ]
    first_last_l2 = l2_distance(history_features[0], history_features[5])
    first_last_cos = cosine_distance(history_features[0], history_features[5])
    report = {
        "step_l2": step_l2,
        "mean_step_l2": float(sum(step_l2) / len(step_l2)),
        "first_last_l2": first_last_l2,
        "first_last_cosine_distance": first_last_cos,
    }

    if wire_scores is not None:
        if len(wire_scores) != 6:
            raise ValueError("Expected six wire scores")
        score = torch.tensor(wire_scores, dtype=torch.float32)
        report["wire_score_first_last_delta"] = float(score[5] - score[0])
        report["wire_score_path"] = float(
            torch.sum(torch.abs(score[1:] - score[:-1]))
        )

    if depths is not None:
        depth = torch.as_tensor(depths, dtype=torch.float32).reshape(6)
        depth_steps = torch.abs(depth[1:] - depth[:-1])
        step = torch.tensor(step_l2, dtype=torch.float32)
        # Pearson correlation between feature step size and |depth| step.
        if float(step.std(unbiased=False)) > 0 and float(depth_steps.std(unbiased=False)) > 0:
            step_n = (step - step.mean()) / step.std(unbiased=False)
            depth_n = (depth_steps - depth_steps.mean()) / depth_steps.std(unbiased=False)
            report["feature_depth_step_corr"] = float((step_n * depth_n).mean())
        else:
            report["feature_depth_step_corr"] = 0.0
        report["depth_first_last_delta"] = float(depth[5] - depth[0])

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


def run_case_diagnostics(encoder, prepared, *, patch, stride):
    """Run all three probes on one prepared case; return printable aggregates."""
    history = prepared["history"]
    goal = prepared["goal"]
    device = next(encoder.parameters()).device
    history = history.to(device)
    goal = goal.to(device)

    with torch.no_grad():
        history_features = encoder(history.unsqueeze(0))[0]
        goal_features = encoder(goal.unsqueeze(0))[0]

    wire = wire_history_probe(encoder, history, patch=patch, stride=stride)
    goal_probe = goal_marker_probe(encoder, goal, prepared["previews"]["goal"])
    separation = goal_vs_history_separation(goal_features, history_features)
    direction = direction_from_history_features(
        history_features.cpu(),
        wire_scores=wire["wire_scores"],
        depths=prepared.get("depth_relative_mm"),
    )
    return {
        "fill_convention": denormalize_stats_note(),
        "wire": wire,
        "goal_marker": goal_probe,
        "goal_separation": separation,
        "direction": direction,
        "history_features_shape": tuple(history_features.shape),
        "goal_features_shape": tuple(goal_features.shape),
    }


def _print_report(report):
    """Print permitted aggregates only (no feature values or paths)."""
    wire = report["wire"]
    print("Wire occlusion (history):")
    print("  score_range:", round(wire["score_range"], 6))
    print("  score_std:", round(wire["score_std"], 6))
    for frame in wire["frames"]:
        x, y = frame["argmax_center_xy_norm"]
        print(
            f"  frame {frame['frame']}: max_l2={frame['max_l2']:.6f} "
            f"max/median={frame['max_over_median']:.3f} "
            f"argmax_xy_norm=({x:.3f},{y:.3f})"
        )

    goal = report["goal_marker"]
    print("Goal marker occlusion:")
    print("  marker_pixels:", goal["marker_pixels"])
    print("  marker_l2:", round(goal["marker_l2"], 6))
    print("  control_l2:", round(goal["control_l2"], 6))
    print("  marker/control:", round(goal["marker_over_control"], 3))

    sep = report["goal_separation"]
    print("Goal vs history L2:")
    print("  mean:", round(sep["goal_history_l2_mean"], 6))
    print("  min:", round(sep["goal_history_l2_min"], 6))

    direction = report["direction"]
    print("Direction from history features (not a CNN head):")
    print("  mean_step_l2:", round(direction["mean_step_l2"], 6))
    print("  first_last_l2:", round(direction["first_last_l2"], 6))
    print(
        "  first_last_cosine_distance:",
        round(direction["first_last_cosine_distance"], 6),
    )
    if "wire_score_first_last_delta" in direction:
        print(
            "  wire_score_first_last_delta:",
            round(direction["wire_score_first_last_delta"], 6),
        )
    if "feature_depth_step_corr" in direction:
        print(
            "  feature_depth_step_corr:",
            round(direction["feature_depth_step_corr"], 6),
        )


def main():
    # Real-data entrypoint only: keep top-level imports free of torchvision so
    # invented-input unit tests can run without the CNN dependency stack.
    from cnn import FrameEncoder
    from dataset import GuidewireDataset
    from preprocessing import DINOv2Preprocessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    parser.add_argument("--split-position", type=int, required=True)
    parser.add_argument(
        "--partition",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--remove-maxpool", action="store_true")
    parser.add_argument(
        "--patch",
        type=int,
        default=74,
        help="Occlusion square side in prepared pixels",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=74,
        help="Occlusion grid stride in prepared pixels",
    )
    args = parser.parse_args()

    if not args.data_dir:
        parser.error("Set DATA_DIR or provide --data-dir")
    if args.case_index < 0:
        parser.error("--case-index must be non-negative")

    dataset = GuidewireDataset(
        args.data_dir,
        split_position=args.split_position,
        partition=args.partition,
    )
    if len(dataset) == 0:
        parser.error("Selected partition is empty")
    if args.case_index >= len(dataset):
        parser.error("--case-index out of range for partition")

    prepare = DINOv2Preprocessor(size=args.size)
    prepared = prepare.prepare_case(dataset[args.case_index])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrameEncoder(remove_maxpool=args.remove_maxpool).to(device)
    report = run_case_diagnostics(
        encoder, prepared, patch=args.patch, stride=args.stride
    )

    print("Device:", device)
    print("History features:", report["history_features_shape"])
    print("Goal features:", report["goal_features_shape"])
    _print_report(report)
    print("CNN 512-d diagnostics finished.")


if __name__ == "__main__":
    main()
