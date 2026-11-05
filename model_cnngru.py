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

Head size is chosen by a capacity rung (``CAPACITY_RUNGS``, the Phase 1
ladder in docs/grp1/NEXT_EXPERIMENTS.md). Rungs with a projection apply
one shared Linear(512 -> p) to every normalized history frame and the
goal, so both stay in one p-d space and the goal no longer outweighs the
GRU summary in the MLP. Rung M is the original head: no projection, GRU
128, MLP 640 -> 128 -> 2, with the same modules, state_dict keys and
seeded initialization as before the ladder existed.

The ResNet is frozen, so training (train_cnngru.py) encodes every image
once and then calls ``forward_features`` on the cached 512-d vectors.
``forward`` gives identical outputs from raw prepared images.

Prediction: logits.argmax(dim=-1)  # 0 = no, 1 = yes
"""

from typing import NamedTuple, Optional, Tuple

from torch import nn

from cnn import FrameEncoder
from gru import CNNGRUEncoder
from mlp_cnn import GoalClassifier, encode_goal


class HeadConfig(NamedTuple):
    """Trainable head size: everything after the frozen ResNet and LayerNorm."""

    projection_dim: Optional[int]   # None: GRU and MLP see the 512-d features
    gru_hidden: int
    gru_layers: int
    gru_dropout: float              # between stacked GRU layers only
    mlp_hidden: Tuple[int, ...]
    mlp_norm_first_hidden: bool


CAPACITY_RUNGS = {
    "XS": HeadConfig(32, 32, 1, 0.0, (32,), False),
    "S": HeadConfig(64, 64, 1, 0.0, (64,), False),
    "M": HeadConfig(None, 128, 1, 0.0, (128,), False),
    "L": HeadConfig(128, 256, 2, 0.2, (256, 64), True),
}
DEFAULT_CAPACITY = "M"


def describe_capacity(name):
    """One-line human description of a rung, e.g. for train.log."""
    config = CAPACITY_RUNGS[name]
    width = config.projection_dim or FrameEncoder.feature_dim
    projection = ("512 -> {} shared".format(config.projection_dim)
                  if config.projection_dim else "none")
    mlp = " -> ".join(str(d) for d in (config.gru_hidden + width,) + config.mlp_hidden + (2,))
    return "capacity {}: projection {} | GRU {} x {} layer(s){} | MLP {}{}".format(
        name, projection, config.gru_hidden, config.gru_layers,
        ", dropout {}".format(config.gru_dropout) if config.gru_layers > 1 else "",
        mlp, " (LayerNorm after first hidden)" if config.mlp_norm_first_hidden else "")


class CNNGRUModel(nn.Module):
    def __init__(self, pretrained=True, dropout=0.3, capacity=DEFAULT_CAPACITY):
        super().__init__()
        if capacity not in CAPACITY_RUNGS:
            raise ValueError("Unknown capacity {!r}; choose from {}".format(
                capacity, ", ".join(CAPACITY_RUNGS)))
        config = CAPACITY_RUNGS[capacity]
        self.capacity = capacity
        width = config.projection_dim or FrameEncoder.feature_dim

        # Construction order matters for rung M: encoder, LayerNorm, then
        # classifier reproduces the pre-ladder seeded initialization.
        self.encoder = CNNGRUEncoder(pretrained=pretrained, input_dim=width,
                                     hidden_dim=config.gru_hidden,
                                     num_layers=config.gru_layers,
                                     dropout=config.gru_dropout)  # history path
        # Shared by history frames and goal so both stay in one feature space.
        self.feature_norm = nn.LayerNorm(FrameEncoder.feature_dim)
        # Identity has no parameters, so rung M's state_dict keys are unchanged.
        self.projection = (nn.Linear(FrameEncoder.feature_dim, config.projection_dim)
                           if config.projection_dim else nn.Identity())
        self.classifier = GoalClassifier(dropout=dropout, summary_dim=config.gru_hidden,
                                         goal_dim=width, hidden_dims=config.mlp_hidden,
                                         norm_first_hidden=config.mlp_norm_first_hidden)

    def forward(self, history, depth, offsets, goal):
        history_features = self.encoder.cnn(history)           # [B, T, 512]
        goal_features = encode_goal(self.encoder.cnn, goal)    # [B, 512] same CNN
        return self.forward_features(history_features, depth, offsets, goal_features)

    def forward_features(self, history_features, depth, offsets, goal_features):
        """Head only: frozen-CNN features in, [B, 2] logits out."""
        history_features = self.projection(self.feature_norm(history_features))  # [B, T, w]
        goal_features = self.projection(self.feature_norm(goal_features))        # [B, w]
        gru_summary = self.encoder.summarize(history_features, depth, offsets)  # [B, gru]
        return self.classifier(gru_summary, goal_features)
