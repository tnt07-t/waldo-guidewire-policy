"""Full CNN-GRU model: history + goal in, RETRACT / ADVANCE probabilities out.

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
goal before the GRU and the MLP. Depth/offset
metadata is concatenated after this normalization, since it is already
standardized.

The small head uses a shared Linear(512 -> 64) projection, a one-layer
GRU (hidden 64), and an MLP mapping 128 -> 64 -> 2: 67,586 trainable parameters.

The ResNet is frozen, so training (train_cnngru.py) encodes every image
once and then calls ``forward_features(return_logits=True)`` on cached vectors
for cross-entropy training. Default outputs apply Softmax over the two classes.
``forward`` gives identical outputs from raw prepared images.

Prediction: probabilities.argmax(dim=-1)  # 0 = RETRACT, 1 = ADVANCE
"""

import torch
from torch import nn

from cnn import FrameEncoder
from gru import CNNGRUEncoder
from mlp_cnn import GoalClassifier, encode_goal

PROJECTION_DIM = 64
GRU_HIDDEN = 64
MLP_HIDDEN = 64


class CNNGRUModel(nn.Module):
    """CNN-GRU policy with optional training-only Gaussian feature noise.

    Noise is added after LayerNorm and before the shared projection. Zero
    noise leaves the features and random state unchanged. The constructor
    retains legacy dropout 0.3; the trainer passes its selected preset's
    dropout explicitly (0 for baseline/light/medium/heavy).
    """

    def __init__(self, pretrained=True, dropout=0.3, feature_noise=0.0):
        super().__init__()
        if not 0 <= feature_noise < float("inf"):
            raise ValueError("feature_noise must be finite and non-negative")
        self.feature_noise = feature_noise
        # Keep construction order stable for seeded initialization.
        self.encoder = CNNGRUEncoder(input_dim=PROJECTION_DIM, hidden_dim=GRU_HIDDEN,
                                     pretrained=pretrained)  # history path
        # Shared by history frames and goal so both stay in one feature space.
        self.feature_norm = nn.LayerNorm(FrameEncoder.feature_dim)
        self.projection = nn.Linear(FrameEncoder.feature_dim, PROJECTION_DIM)
        self.classifier = GoalClassifier(summary_dim=GRU_HIDDEN, goal_dim=PROJECTION_DIM,
                                         hidden_dim=MLP_HIDDEN, dropout=dropout)

    def forward(self, history, depth, offsets, goal, *, return_logits=False):
        history_features = self.encoder.cnn(history)           # [B, T, 512]
        goal_features = encode_goal(self.encoder.cnn, goal)    # [B, 512] same CNN
        return self.forward_features(history_features, depth, offsets, goal_features,
                                     return_logits=return_logits)

    def regularize(self, history_features, goal_features):
        """Training-only noise on normalized [B, T, 512] / [B, 512] features."""
        if not self.training:
            return history_features, goal_features
        if self.feature_noise:
            history_features = history_features + self.feature_noise * torch.randn_like(
                history_features)
            goal_features = goal_features + self.feature_noise * torch.randn_like(goal_features)
        return history_features, goal_features

    def forward_features(self, history_features, depth, offsets, goal_features, *, return_logits=False):
        """Return [B, 2] probabilities, or raw logits for cross-entropy."""
        history_features, goal_features = self.regularize(
            self.feature_norm(history_features), self.feature_norm(goal_features))
        history_features = self.projection(history_features)  # [B, T, 64]
        goal_features = self.projection(goal_features)        # [B, 64]
        gru_summary = self.encoder.summarize(history_features, depth, offsets)  # [B, 64]
        return self.classifier(gru_summary, goal_features, return_logits=return_logits)
