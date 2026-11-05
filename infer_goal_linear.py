"""Inference for the goal-only linear baseline (best probe config).

Loads ``best_model.pt`` from ``probe_cnn_features.py --preset best`` (or any
goal_only probe that wrote the bundle). Encodes the goal with a frozen
ResNet-18 (``remove_maxpool`` from the checkpoint) and applies the saved
linear head. History is not used. The test partition may be scored only for
final reporting — do not use it for tuning.
"""

import argparse
import os

import torch

from dataset import GuidewireDataset
from model_goal_linear import GoalOnlyLinearModel
from preprocessing import DINOv2Preprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    parser.add_argument("--split-position", type=int, required=True)
    parser.add_argument("--partition", choices=("train", "validation", "test"),
                        default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint", required=True,
                        help="best_model.pt from a goal_only probe run")
    parser.add_argument("--size", type=int, default=None,
                        help="Override prep size; default = checkpoint size")
    args = parser.parse_args()

    if not args.data_dir:
        parser.error("Set DATA_DIR or provide --data-dir")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.checkpoint, map_location=device)
    if bundle.get("design") != "goal_only" or "linear" not in bundle:
        parser.error("Checkpoint must be a goal_only best_model.pt bundle")

    size = args.size if args.size is not None else int(bundle.get("size", 518))
    model = GoalOnlyLinearModel(
        pretrained=True, remove_maxpool=bool(bundle.get("remove_maxpool", True))
    ).to(device)
    model.load_linear_state_dict(bundle["linear"])
    model.eval()

    dataset = GuidewireDataset(
        args.data_dir, split_position=args.split_position, partition=args.partition)
    if len(dataset) < args.batch_size:
        parser.error("Partition has fewer cases than --batch-size")

    prepare = DINOv2Preprocessor(size=size)
    goals, labels = [], []
    for i in range(args.batch_size):
        prepared = prepare.prepare_case(dataset[i])
        goals.append(prepared["goal"])
        labels.append(prepared["label"])
    goal = torch.stack(goals).to(device)
    label = torch.tensor(labels, device=device)

    with torch.no_grad():
        logits = model(goal)
    preds = logits.argmax(dim=-1)

    print("Device:", device)
    print("remove_maxpool:", model.remove_maxpool, "| size:", size)
    print("Logits shape:", tuple(logits.shape))
    print("Predictions:", preds.tolist())
    print("Batch accuracy: {:.3f}".format((preds == label).double().mean().item()))


if __name__ == "__main__":
    main()
