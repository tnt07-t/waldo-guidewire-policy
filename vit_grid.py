"""Compact spatial features for latest + goal; pretrained ViT stays frozen."""
import torch
from torch import nn


class ViTGridHead(nn.Module):
    """Shared 384→16 regional projection; 16 regions/image; 524→64→2 head.

    Extraction pools the 37×37 contextual patch grid to 4×4 regions. Preserve
    row-major locations instead of averaging the entire image into one vector.
    This module learns action prediction, not explicit tip/target localization.
    """
    model = 'static_grid_goal_metadata'
    architecture = 'late_goal'  # Compatibility field; this head has no GRU.

    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(384, 16), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(524, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, history, goal, metadata):
        if (history.ndim != 4 or history.shape[1:] != (1, 16, 384)
                or len(history) == 0 or goal.shape != (len(history), 16, 384)
                or metadata.shape != (len(history), 6, 2)):
            raise ValueError('Expected latest grid [B,1,16,384], goal [B,16,384], metadata [B,6,2]')
        for value in (history, goal, metadata):
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('Inputs must be finite float32 tensors')
        # One projection shared across all regions and both image roles.
        latest = self.projection(history[:, 0]).flatten(1)
        reference = self.projection(goal).flatten(1)
        return self.classifier(torch.cat((latest, reference, metadata.flatten(1)), dim=1))
