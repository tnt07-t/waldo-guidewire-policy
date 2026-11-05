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

The 512-d frozen-ResNet features are post-ReLU (non-negative, uncentered),
so one shared ``nn.LayerNorm(512)`` normalizes every history frame and the
goal before the GRU and the MLP. Without it the head underfits (split-0
train loss stays near 0.65, see docs/grp1/PROGRESS.md). Depth/offset
metadata is concatenated after this normalization, since it is already
standardized.

The head is size S from the Phase 1 capacity ladder (docs/grp1/PROGRESS.md):
one shared Linear(512 -> 64) projects every normalized history frame and
the goal into one 64-d space, a one-layer GRU (hidden 64) summarizes the
history, and the MLP maps 128 -> 64 -> 2. 67,586 trainable parameters.
XS, M and L performed no better on validation, so S is the only option.

The ResNet is frozen, so training (train_cnngru.py) encodes every image
once and then calls ``forward_features`` on the cached 512-d vectors.
``forward`` gives identical outputs from raw prepared images.

Prediction: logits.argmax(dim=-1)  # 0 = no, 1 = yes
"""

import torch
from torch import nn
from torch.nn import functional as F

from cnn import FrameEncoder
from gru import CNNGRUEncoder
from mlp_cnn import GoalClassifier, encode_goal

PROJECTION_DIM = 64
GRU_HIDDEN = 64
MLP_HIDDEN = 64


class CNNGRUModel(nn.Module):
    """CNN-GRU policy with optional H4 feature-space regularizers.

    The regularizers (docs/grp1/NEXT_EXPERIMENTS.md, H4) act on the
    LayerNormed 512-d features in training mode only and add no parameters:

    - ``frame_dropout``: each history step of each case is replaced by zeros
      (the LayerNorm mean) with this probability; depth/offset stay.
    - ``feature_dropout``: element-wise dropout on history and goal features.
    - ``feature_noise``: std of Gaussian noise added to history and goal
      features, in LayerNorm units.

    At 0 (the default) a regularizer is skipped entirely and draws no random
    numbers, so default training is unchanged.
    """

    def __init__(self, pretrained=True, dropout=0.3, feature_noise=0.0,
                 feature_dropout=0.0, frame_dropout=0.0):
        super().__init__()
        if feature_noise < 0:
            raise ValueError("feature_noise must be non-negative")
        for name, p in (("feature_dropout", feature_dropout), ("frame_dropout", frame_dropout)):
            if not 0 <= p < 1:
                raise ValueError("{} must be in [0, 1)".format(name))
        self.feature_noise = feature_noise
        self.feature_dropout = feature_dropout
        self.frame_dropout = frame_dropout
        # Construction order fixes the seeded initialization; keep it so runs
        # reproduce the Phase 1 rung-S results and checkpoints keep their keys.
        self.encoder = CNNGRUEncoder(input_dim=PROJECTION_DIM, hidden_dim=GRU_HIDDEN,
                                     pretrained=pretrained)  # history path
        # Shared by history frames and goal so both stay in one feature space.
        self.feature_norm = nn.LayerNorm(FrameEncoder.feature_dim)
        self.projection = nn.Linear(FrameEncoder.feature_dim, PROJECTION_DIM)
        self.classifier = GoalClassifier(summary_dim=GRU_HIDDEN, goal_dim=PROJECTION_DIM,
                                         hidden_dim=MLP_HIDDEN, dropout=dropout)

    def forward(self, history, depth, offsets, goal):
        history_features = self.encoder.cnn(history)           # [B, T, 512]
        goal_features = encode_goal(self.encoder.cnn, goal)    # [B, 512] same CNN
        return self.forward_features(history_features, depth, offsets, goal_features)

    def regularize(self, history_features, goal_features):
        """Training-only H4 regularizers on normalized [B, T, 512] / [B, 512]."""
        if not self.training:
            return history_features, goal_features
        if self.frame_dropout:
            keep = torch.rand(history_features.shape[:2], device=history_features.device)
            keep = (keep >= self.frame_dropout).to(history_features.dtype)
            history_features = history_features * keep[..., None]
        if self.feature_dropout:
            history_features = F.dropout(history_features, self.feature_dropout)
            goal_features = F.dropout(goal_features, self.feature_dropout)
        if self.feature_noise:
            history_features = history_features + self.feature_noise * torch.randn_like(
                history_features)
            goal_features = goal_features + self.feature_noise * torch.randn_like(goal_features)
        return history_features, goal_features

    def forward_features(self, history_features, depth, offsets, goal_features):
        """Head only: frozen-CNN features in, [B, 2] logits out."""
        history_features, goal_features = self.regularize(
            self.feature_norm(history_features), self.feature_norm(goal_features))
        history_features = self.projection(history_features)  # [B, T, 64]
        goal_features = self.projection(goal_features)        # [B, 64]
        gru_summary = self.encoder.summarize(history_features, depth, offsets)  # [B, 64]
        return self.classifier(gru_summary, goal_features)
