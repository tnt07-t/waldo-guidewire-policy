"""CNN-GRU encoder: six history frames in, one GRU summary vector out."""

import torch
from torch import nn

from cnn import FrameEncoder

HIDDEN_DIM = 128


class CNNGRUEncoder(nn.Module):
    """Frozen ResNet-18 per frame, then a GRU over the frame sequence.

    ``input_dim`` is the width of each frame feature the GRU receives (512
    for raw ResNet features, smaller when the model projects them first);
    two standardized metadata channels [depth_mm, offset_s] are appended
    at every step. ``dropout`` acts between stacked GRU layers only.
    """

    def __init__(self, pretrained=True, input_dim=FrameEncoder.feature_dim,
                 hidden_dim=HIDDEN_DIM, num_layers=1, dropout=0.0):
        super().__init__()
        self.cnn = FrameEncoder(pretrained=pretrained)
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim + 2, hidden_dim, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0.0,
                          batch_first=True)
        # Standardization for the [depth_mm, offset_s] pair at every step.
        # Identity until set_metadata_stats() is called with training data.
        # Buffers, so the statistics travel with the checkpoint.
        self.register_buffer("meta_mean", torch.zeros(2))
        self.register_buffer("meta_std", torch.ones(2))

    @torch.no_grad()
    def set_metadata_stats(self, depth, offsets):
        """Fit mean/std per channel from TRAINING depth and offsets [N, T].

        Statistics pool all time steps of a channel, so the always-zero last
        step is included exactly as the GRU will see it. A constant channel
        keeps std 1 instead of dividing by ~0.
        """
        meta = torch.stack([depth, offsets], dim=-1).reshape(-1, 2).double()
        mean = meta.mean(dim=0)
        std = meta.std(dim=0, unbiased=False)
        std = torch.where(std > 1e-6, std, torch.ones_like(std))
        self.meta_mean.copy_(mean.to(self.meta_mean))
        self.meta_std.copy_(std.to(self.meta_std))

    def summarize(self, features, depth, offsets):
        """GRU over frame features (B, T, input_dim) -> last-layer state (B, hidden_dim)."""
        meta = torch.stack([depth, offsets], dim=-1)               # (B, T, 2)
        meta = (meta - self.meta_mean) / self.meta_std
        steps = torch.cat([features, meta.to(features.dtype)], dim=-1)  # (B, T, input_dim + 2)
        _, h = self.gru(steps)                                     # h: (layers, B, hidden)
        return h[-1]                                               # (B, hidden)

    def forward(self, history, depth, offsets):
        # history (B, T, 3, H, W), depth (B, T), offsets (B, T)
        return self.summarize(self.cnn(history), depth, offsets)
