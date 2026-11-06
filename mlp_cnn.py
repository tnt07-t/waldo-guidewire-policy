"""CNN–GRU policy pieces that feed the MLP classifier.

Goal path (this file):
  prepared goal [B, 3, H, W]  --FrameEncoder (ResNet-18)-->  goal_features [B, 512]

History path (gru.py):
  history + depth/time  --CNNGRUEncoder-->  gru_summary [B, 64]

model_cnngru.py projects history and goal features to 64-d, then
GoalClassifier maps concat([gru_summary, goal_projection]) [B, 128] -> logits.
"""

import torch
from torch import nn

# Matches FrameEncoder.feature_dim.
GOAL_FEATURE_DIM = 512

def encode_goal(encoder, goal):
    """Run the prepared goal through the shared frozen ResNet (FrameEncoder).

    goal: [3, H, W] -> [512]
          [B, 3, H, W] -> [B, 512]

    ``encoder`` should be the same CNN used for history (e.g. ``model.cnn``).
    Expect a tensor already prepared like history (e.g. ``prepared["goal"]``
    from DINOv2Preprocessor); this function does not resize/normalize.

    Goal is not a GRU time step - only a 512-d vector for MLP fusion.
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
    (no, yes): Dropout -> Linear -> ReLU -> Dropout -> Linear.
    Train with CrossEntropyLoss, predict with argmax.

    Dropout on the fused input and the hidden layer is the main regularizer
    of the head with ~270 training cases. It is inactive in eval mode.
    """ 
    def __init__(self, summary_dim, goal_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(summary_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, gru_summary, goal_features):
        return self.mlp(torch.cat([gru_summary, goal_features], dim=-1))
