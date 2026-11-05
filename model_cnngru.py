"""Full CNN-GRU model: history + goal in, yes / no logits out.

Data (one batch of B, after prepare_case + collate):
  history            [B, T, 3, H, W]
  depth_relative_mm  [B, T]
  history_offsets_s  [B, T]
  goal               [B, 3, H, W]
  label              [B]  (0 = no, 1 = yes)

Depth and offsets are standardized inside the encoder with training-set
statistics (``model.encoder.set_metadata_stats``); the statistics are
buffers saved in the state_dict, so inference reuses them automatically.

The 512-d frozen-ResNet features can be normalized before the GRU and the
MLP (``feature_norm``). One shared module is applied to every history
frame and to the goal, so both stay in the same feature space:
  "none"    identity (baseline H0)
  "layer"   nn.LayerNorm(512), per sample across the 512 dimensions (H1)
  "zscore"  per-dimension training-set mean / std, fitted with
            ``model.set_feature_stats`` and saved as buffers (H1b)
Depth/offset metadata is not part of this normalization; it is already
standardized and is concatenated after the 512 features are normalized.

The ResNet is frozen, so training (train_cnngru.py) encodes every image
once and then calls ``forward_features`` on the cached 512-d vectors.
``forward`` gives identical outputs from raw prepared images.

Prediction: logits.argmax(dim=-1)  # 0 = no, 1 = yes
"""

import torch
from torch import nn

from cnn import FrameEncoder
from gru import CNNGRUEncoder
from mlp_cnn import GoalClassifier, encode_goal

FEATURE_NORMS = ("none", "layer", "zscore")


class FeatureStandardizer(nn.Module):
    """Per-dimension z-score with fixed training-set statistics.

    Identity until ``set_stats`` is called. Buffers, so the statistics
    travel with the checkpoint, like the metadata statistics in gru.py.
    """

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    @torch.no_grad()
    def set_stats(self, features):
        """Fit mean/std per dimension from TRAINING features [N, dim].

        A constant dimension keeps std 1 instead of dividing by ~0.
        """
        features = features.reshape(-1, self.mean.numel()).double()
        mean = features.mean(dim=0)
        std = features.std(dim=0, unbiased=False)
        std = torch.where(std > 1e-6, std, torch.ones_like(std))
        self.mean.copy_(mean.to(self.mean))
        self.std.copy_(std.to(self.std))

    def forward(self, features):
        return (features - self.mean) / self.std


def build_feature_norm(kind, dim=FrameEncoder.feature_dim):
    if kind == "none":
        return nn.Identity()
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "zscore":
        return FeatureStandardizer(dim)
    raise ValueError("feature_norm must be one of {}".format(FEATURE_NORMS))


def feature_norm_from_state_dict(state_dict):
    """Which ``feature_norm`` a saved CNNGRUModel state_dict was trained with."""
    if "feature_norm.std" in state_dict:
        return "zscore"
    if "feature_norm.weight" in state_dict:
        return "layer"
    return "none"


class CNNGRUModel(nn.Module):
    def __init__(self, pretrained=True, dropout=0.3, feature_norm="none"):
        super().__init__()
        self.feature_norm_kind = feature_norm
        self.encoder = CNNGRUEncoder(pretrained=pretrained)  # history path
        # Shared by history frames and goal; built after the encoder so the
        # default "none" model initializes exactly like before this option.
        self.feature_norm = build_feature_norm(feature_norm)
        self.classifier = GoalClassifier(dropout=dropout)    # 640 -> 128 -> 2

    @torch.no_grad()
    def set_feature_stats(self, history_features, goal_features):
        """Fit the "zscore" statistics from TRAINING features.

        history_features [N, T, 512] and goal_features [N, 512] are pooled:
        every history frame and every goal is one sample, matching the one
        shared normalization applied to both.
        """
        if self.feature_norm_kind != "zscore":
            raise ValueError("set_feature_stats needs feature_norm='zscore'")
        dim = goal_features.shape[-1]
        self.feature_norm.set_stats(torch.cat([history_features.reshape(-1, dim),
                                               goal_features.reshape(-1, dim)]))

    def forward(self, history, depth, offsets, goal):
        history_features = self.encoder.cnn(history)           # [B, T, 512]
        goal_features = encode_goal(self.encoder.cnn, goal)    # [B, 512] same CNN
        return self.forward_features(history_features, depth, offsets, goal_features)

    def forward_features(self, history_features, depth, offsets, goal_features):
        """Head only: frozen-CNN features in, [B, 2] logits out."""
        history_features = self.feature_norm(history_features)  # [B, T, 512]
        goal_features = self.feature_norm(goal_features)        # [B, 512]
        gru_summary = self.encoder.summarize(history_features, depth, offsets)  # [B, 128]
        return self.classifier(gru_summary, goal_features)
