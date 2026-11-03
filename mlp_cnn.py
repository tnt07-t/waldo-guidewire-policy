"""CNN–GRU policy pieces that feed the MLP classifier.

Goal path (this file):
  prepared goal [B, 3, H, W]  --FrameEncoder (ResNet-18)-->  goal_features [B, 512]

History path (elsewhere):
  history + depth/time  --CNNGRUEncoder-->  gru_summary [B, 128]

Teammate: concat([gru_summary, goal_features], dim=-1) -> [B, 640] -> MLP -> action.
"""

import torch

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


# TODO: add MLP here, e.g.
#   x = torch.cat([gru_summary, encode_goal(model.cnn, goal)], dim=-1)  # [B, 640]
#   return mlp(x)
