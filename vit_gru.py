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
        # Each time step has 384 image features + depth + time = 386 numbers.
        # One forward-reading layer keeps a 64-number memory of the six steps.
        # batch_first means inputs are [cases, time steps, features].
        self.gru = nn.GRU(386, 64, num_layers=1, batch_first=True)
        # Combine the final 64-number memory with the separate 384-number goal.
        # The classifier maps these 448 inputs through 64 units to two action scores.
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
        # Append each step's depth/time to its matching image, preserving order.
        # Ignore the per-step outputs; hidden contains the final layer memories.
        _, hidden = self.gru(torch.cat((history, metadata), dim=-1))
        # hidden[-1] selects the final (only) layer, not the previous case.
        # Goal enters here, after the history; raw scores feed cross-entropy.
        return self.classifier(torch.cat((hidden[-1], goal), dim=-1))
