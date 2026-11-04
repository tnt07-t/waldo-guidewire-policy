"""CNN-GRU encoder: six history frames in, 128-value summary out."""

import torch
from torch import nn

from cnn import FrameEncoder

class CNNGRUEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.cnn = FrameEncoder(pretrained=pretrained)
        self.gru = nn.GRU(FrameEncoder.feature_dim + 2, 128, batch_first=True)

    def forward(self, history, depth, offsets):
        # history (B, T, 3, H, W), depth (B, T), offsets (B, T)
        features = self.cnn(history)                    # (B, T, 512)
        meta = torch.stack([depth, offsets], dim=-1)    # (B, T, 2)
        steps = torch.cat([features, meta], dim=-1)     # (B, T, 514)
        _, h = self.gru(steps)                          # h: (1, B, 128)
        return h[-1]                                    # (B, 128)