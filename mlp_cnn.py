"""CNN–GRU policy pieces that feed the MLP classifier.

Goal path (this file):
  prepared goal [B, 3, H, W]  --FrameEncoder (ResNet-18)-->  goal_features [B, 512]

History path (elsewhere):
  history + depth/time  --CNNGRUEncoder-->  gru_summary [B, 128]

Teammate: concat([gru_summary, goal_features], dim=-1) -> [B, 640] -> MLP -> action.
"""

import torch
from torch import nn

# Matches FrameEncoder.feature_dim / CNNGRUEncoder hidden size.
GOAL_FEATURE_DIM = 512
GRU_SUMMARY_DIM = 128

def encode_goal(encoder, goal):
    """Run the prepared goal through the shared frozen ResNet (FrameEncoder).

    goal: [3, H, W] -> [512]
          [B, 3, H, W] -> [B, 512]

    ``encoder`` should be the same CNN used for history (e.g. ``model.cnn``).
    Expect a tensor already prepared like history (e.g. ``prepared["goal"]``
    from DINOv2Preprocessor); this function does not resize/normalize.

    Goal is not a GRU time step — only a 512-d vector for MLP fusion.
    """
    single = goal.ndim == 3
    if single:
        goal = goal.unsqueeze(0)
    if goal.ndim != 4 or goal.shape[1] != 3:
        raise ValueError("Expected goal [3, H, W] or [B, 3, H, W]")

    # Same ResNet forward as a single history frame.
    features = encoder(goal)
    if features.shape != (len(goal), GOAL_FEATURE_DIM):
        raise ValueError("Expected one 512-d goal feature per image")
    if not torch.isfinite(features).all():
        raise ValueError("Goal encoder returned non-finite features")
    if single:
        return features[0]
    return features


class GoalClassifier(nn.Module):
    """MLP classifier (yes / no) over the GRU summary and goal features.

    gru_summary [B, summary_dim] + goal_features [B, goal_dim] -> logits [B, 2]
    (no, yes). Defaults give the original 640 -> 128 -> ReLU -> 2 head.
    Train with CrossEntropyLoss, predict with argmax.

    Each hidden layer is Linear -> (LayerNorm, first hidden layer only, when
    ``norm_first_hidden``) -> ReLU -> Dropout. Dropout is also applied to the
    fused input; with ~270 training cases it is the main regularizer of the
    head. It is inactive in eval mode.

    With the defaults the Sequential indices (Linear layers at 1 and 4) match
    checkpoints saved before the width options existed.
    """

    def __init__(self, dropout=0.3, summary_dim=GRU_SUMMARY_DIM,
                 goal_dim=GOAL_FEATURE_DIM, hidden_dims=(128,), norm_first_hidden=False):
        super().__init__()
        layers = [nn.Dropout(dropout)]
        width = summary_dim + goal_dim
        for i, hidden in enumerate(hidden_dims):
            layers.append(nn.Linear(width, hidden))
            if norm_first_hidden and i == 0:
                layers.append(nn.LayerNorm(hidden))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        layers.append(nn.Linear(width, 2))
        self.mlp = nn.Sequential(*layers)

    def forward(self, gru_summary, goal_features):
        return self.mlp(torch.cat([gru_summary, goal_features], dim=-1))
