"""Trainable head for frozen DINOv2 features; no encoder or data access."""

import torch
from torch import nn


class ViTGRUHead(nn.Module):
    """Six chronological image/depth/time steps plus a separate goal.

    Metadata is already scaled, ordered [depth, time] at each step.
    Outputs are raw scores in label order RETRACT=0, ADVANCE=1.
    """

    def __init__(self, architecture="late_goal"):
        super().__init__()
        if architecture not in ("late_goal", "early_goal"):
            raise ValueError("Unsupported architecture")
        self.architecture = architecture
        # Ordered inputs: [history384, depth, time], then goal384 for early_goal.
        # The same goal conditions every memory update, not an extra time step.
        input_size = 770 if architecture == "early_goal" else 386
        # One forward-reading layer keeps a 64-number memory of the six steps.
        # batch_first means inputs are [cases, time steps, features].
        self.gru = nn.GRU(input_size, 64, num_layers=1, batch_first=True)
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
        steps = torch.cat((history, metadata), dim=-1)
        if self.architecture == "early_goal":
            steps = torch.cat((steps, goal[:, None, :].expand(-1, 6, -1)), dim=-1)
        _, hidden = self.gru(steps)
        # hidden[-1] selects the final (only) layer, not the previous case.
        # Both variants retain direct goal fusion; raw scores feed cross-entropy.
        return self.classifier(torch.cat((hidden[-1], goal), dim=-1))
