"""Trainable head for frozen DINOv2 features; no encoder or data access."""

import torch
from torch import nn


class ViTGRUHead(nn.Module):
    """Six chronological image/depth/time steps plus a separate goal.

    Metadata is already scaled, ordered [depth, time] at each step.
    Outputs are raw scores in label order RETRACT=0, ADVANCE=1.
    """

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(386, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(448, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, history, goal, metadata):
        if (history.ndim != 3 or history.shape[1:] != (6, 384)
                or len(history) == 0 or goal.shape != (len(history), 384)
                or metadata.shape != (len(history), 6, 2)):
            raise ValueError("Expected history [B,6,384], goal [B,384], metadata [B,6,2]")
        for value in (history, goal, metadata):
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError("Inputs must be finite float32 tensors")
        # Omitting h0 starts each independent case at zero; no state is retained.
        _, hidden = self.gru(torch.cat((history, metadata), dim=-1))
        return self.classifier(torch.cat((hidden[-1], goal), dim=-1))
