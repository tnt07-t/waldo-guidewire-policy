"""Frozen ResNet-18 feature encoder for guidewire action prediction.

Implemented:
- ImageNet-pretrained ResNet-18 by default, with fc replaced by Identity.
- 512 features per image; final average pooling retained.
- Standard maxpool retained by default, with optional removal.
- Backbone parameters frozen and BatchNorm kept in evaluation mode,
  including when the surrounding policy enters training mode.
- History: (B, T, 3, H, W) -> (B, T, 512), with T=6 for this dataset.
- Goal: (B, 3, H, W) -> (B, 512), using the same encoder weights.
  The goal is encoded separately, not as another history step.
- Existing DINOv2Preprocessor supplies RGB conversion, proportional
  resizing, padding, and ImageNet normalization. Its checkpoint metadata
  refers to DINOv2 and must not be used to identify this ResNet model.

Running this file checks output shapes and finite features for one case.
It does not train, evaluate predictive accuracy, or save features.
Real images and derived features must remain on Perlmutter.

Remaining work:
- Visually inspect preprocessed images for guidewire/target visibility.
- Integrate GRU/LSTM, aligned depth/time inputs, and goal fusion.
- Train and evaluate the fully frozen baseline.
- Implement layer4 fine-tuning with a smaller backbone learning rate;
  consider layer3 only if validation results justify it.
"""

"""Frozen ResNet-18 encoder using the existing shared preprocessing."""

import argparse
import os

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from dataset import GuidewireDataset
from preprocessing import DINOv2Preprocessor


class FrameEncoder(nn.Module):
    """Encode history and goal images into 512-dimensional features."""

    feature_dim = 512

    def __init__(self, pretrained=True, remove_maxpool=False):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # Return features instead of ImageNet class predictions.
        # Keep the final average-pooling layer.
        self.backbone.fc = nn.Identity()

        # Optional experiment: preserve larger internal feature maps.
        # uuseful bc guidewire small
        if remove_maxpool:
            self.backbone.maxpool = nn.Identity()

        # Start with a fully frozen encoder.
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        """Keep ResNet in eval mode while the GRU/classifier train."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images):
        """
        Accept tensors from the existing preprocessor.

        Images:  [B, 3, H, W]    -> [B, 512]
        History: [B, T, 3, H, W] -> [B, T, 512]

        T = 6 for this dataset.
        Encode the goal separately; it is not a seventh history step.
        """
        if images.ndim not in (4, 5):
            raise ValueError(
                "Expected [B, 3, H, W] or [B, T, 3, H, W]"
            )

        if images.shape[-3] != 3:
            raise ValueError("Expected 3 RGB channels")

        if images.ndim == 4:
            return self.backbone(images)

        batch_size, steps, channels, height, width = images.shape

        # Temporarily combine batch and time to process individual frames.
        frames = images.reshape(
            batch_size * steps, channels, height, width
        )
        features = self.backbone(frames)

        # Restore the sequence structure without changing frame order.
        return features.reshape(batch_size, steps, self.feature_dim)


def main():
    """Check one real case on Perlmutter without training or saving data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR"),
    )
    parser.add_argument(
        "--split-position",
        type=int,
        required=True,
        help="Zero-based position in splits.json",
    )
    parser.add_argument(
        "--partition",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=518,
        help="Input size; shared preprocessor requires a multiple of 14",
    )
    parser.add_argument(
        "--remove-maxpool",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.data_dir:
        parser.error("Set DATA_DIR or provide --data-dir")

    dataset = GuidewireDataset(
        args.data_dir,
        split_position=args.split_position,
        partition=args.partition,
    )
    if len(dataset) == 0:
        parser.error("Selected partition is empty")

    # Reuse RGB conversion, proportional resizing, padding, and normalization.
    # Its DINOv2 checkpoint metadata does not describe this ResNet encoder.
    prepare = DINOv2Preprocessor(size=args.size)
    prepared = prepare.prepare_case(dataset[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrameEncoder(
        remove_maxpool=args.remove_maxpool,
    ).to(device)

    # Add a batch dimension to the prepared case.
    history = prepared["history"].unsqueeze(0).to(device)
    goal = prepared["goal"].unsqueeze(0).to(device)

    with torch.no_grad():
        history_features = encoder(history)
        goal_features = encoder(goal)

    assert history_features.shape == (1, 6, 512)
    assert goal_features.shape == (1, 512)
    assert torch.isfinite(history_features).all()
    assert torch.isfinite(goal_features).all()

    print("Device:", device)
    print("History features:", tuple(history_features.shape))
    print("Goal features:", tuple(goal_features.shape))
    print("Frozen encoder check passed.")


if __name__ == "__main__":
    main()