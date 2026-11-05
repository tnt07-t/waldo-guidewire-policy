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

The ResNet is frozen, so training (train_cnngru.py) encodes every image
once and then calls ``forward_features`` on the cached 512-d vectors.
``forward`` gives identical outputs from raw prepared images.

Prediction: logits.argmax(dim=-1)  # 0 = no, 1 = yes
"""

from torch import nn

from cnn import FrameEncoder
from gru import CNNGRUEncoder
from mlp_cnn import GoalClassifier, encode_goal


class CNNGRUModel(nn.Module):
    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        self.encoder = CNNGRUEncoder(pretrained=pretrained)  # history path
        # Shared by history frames and goal so both stay in one feature space.
        self.feature_norm = nn.LayerNorm(FrameEncoder.feature_dim)
        self.classifier = GoalClassifier(dropout=dropout)    # 640 -> 128 -> 2

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
