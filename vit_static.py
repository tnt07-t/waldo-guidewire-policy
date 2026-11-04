"""Latest visual observation + goal + historical movement metadata; no recurrence."""
import torch
from torch import nn


class ViTStaticHead(nn.Module):
    """Frozen features -> 780 -> 64 -> ReLU -> 2 (RETRACT=0, ADVANCE=1).

    Order: latest image384, goal384, then six chronological [depth,time] pairs.
    Accept all six image features for comparisons, or just the latest for efficient
    extraction. Earlier visual frames never enter the classifier.
    """
    model = 'static_latest_goal_metadata'
    architecture = 'late_goal'  # Shared CLI default; no GRU or goal-fusion variant.

    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(780, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, history, goal, metadata):
        if (history.ndim != 3 or history.shape[1:] not in ((1, 384), (6, 384))
                or len(history) == 0 or goal.shape != (len(history), 384)
                or metadata.shape != (len(history), 6, 2)):
            raise ValueError('Expected history [B,1 or 6,384], goal [B,384], metadata [B,6,2]')
        for value in (history, goal, metadata):
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('Inputs must be finite float32 tensors')
        return self.classifier(torch.cat((history[:, -1], goal, metadata.flatten(1)), dim=1))
