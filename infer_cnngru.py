"""Run CNNGRUModel inference on one batch of cases.

Loads ``--batch-size`` cases (each one goal + six history images),
prepares them with the shared preprocessor, and prints probability shape,
predictions and batch accuracy. Without ``--checkpoint`` the GRU and
classifier are untrained, so accuracy only confirms data flows through.

Real images and derived tensors must remain on Perlmutter.
"""

import argparse
import os

import torch

from dataset import GuidewireDataset
from model_cnngru import CNNGRUModel
from preprocessing import DINOv2Preprocessor


def collate(cases, device):
    """Stack prepared cases into model inputs and labels."""
    f32 = dict(dtype=torch.float32, device=device)
    return {
        "history": torch.stack([c["history"] for c in cases]).to(device),       # [B, 6, 3, H, W]
        "depth": torch.tensor([c["depth_relative_mm"] for c in cases], **f32),  # [B, 6]
        "offsets": torch.tensor([c["history_offsets_s"] for c in cases], **f32),# [B, 6]
        "goal": torch.stack([c["goal"] for c in cases]).to(device),             # [B, 3, H, W]
        "label": torch.tensor([c["label"] for c in cases], device=device),      # [B]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    parser.add_argument("--split-position", type=int, required=True,
                        help="Zero-based position in splits.json")
    parser.add_argument("--partition", choices=("train", "validation", "test"),
                        default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=518,
                        help="Input size; shared preprocessor requires a multiple of 14")
    parser.add_argument("--checkpoint", help="Optional state_dict for CNNGRUModel")
    args = parser.parse_args()

    if not args.data_dir:
        parser.error("Set DATA_DIR or provide --data-dir")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    dataset = GuidewireDataset(
        args.data_dir,
        split_position=args.split_position,
        partition=args.partition,
    )
    if len(dataset) < args.batch_size:
        parser.error("Partition has fewer cases than --batch-size")

    prepare = DINOv2Preprocessor(size=args.size)
    cases = [prepare.prepare_case(dataset[i]) for i in range(args.batch_size)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = collate(cases, device)

    model = CNNGRUModel().to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        probabilities = model(batch["history"], batch["depth"], batch["offsets"], batch["goal"])
    preds = probabilities.argmax(dim=-1)  # 0 = RETRACT, 1 = ADVANCE

    assert probabilities.shape == (args.batch_size, 2)
    assert torch.isfinite(probabilities).all()

    print("Device:", device)
    print("Probability shape:", tuple(probabilities.shape))
    print("Predictions:", preds.tolist())
    print("Batch accuracy{}: {:.3f}".format(
        "" if args.checkpoint else " (untrained)",
        (preds == batch["label"]).float().mean().item(),
    ))


if __name__ == "__main__":
    main()
