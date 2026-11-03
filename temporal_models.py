"""Trainable heads on frozen image features; no dataset access or downloads."""

import torch
from torch import nn


class FeatureHead(nn.Module):
    """Common late-goal fusion for static and temporal comparisons."""

    def __init__(self, kind, feature_dim=512, width=64, dropout=0.1,
                 position_embeddings=True, classifier_width=None):
        super().__init__()
        if kind not in ("latest_mlp", "mean_mlp", "attention", "gru", "lstm"):
            raise ValueError("Unknown head: " + kind)
        if width <= 0 or width % 4 or feature_dim <= 0:
            raise ValueError("Positive width divisible by four and feature_dim required")
        self.kind, self.feature_dim = kind, feature_dim
        classifier_width = width if classifier_width is None else classifier_width
        if classifier_width <= 0:
            raise ValueError("classifier_width must be positive")
        if kind == "latest_mlp":
            fused = 2 * feature_dim + 12
        elif kind == "mean_mlp":
            fused = 2 * feature_dim + 12
        else:
            self.project = nn.Linear(feature_dim + 2, width)
            if kind == "attention":
                if position_embeddings:
                    self.position = nn.Parameter(torch.empty(1, 6, width))
                    nn.init.normal_(self.position, std=0.02)
                else:
                    self.register_parameter("position", None)
                layer = nn.TransformerEncoderLayer(
                    width, 4, dim_feedforward=2 * width, dropout=dropout,
                    activation="gelu", batch_first=True)
                self.temporal = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            else:
                cls = nn.GRU if kind == "gru" else nn.LSTM
                self.temporal = cls(width, width, batch_first=True)
            fused = width + feature_dim
        self.classifier = nn.Sequential(nn.Linear(fused, classifier_width), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(classifier_width, 2))

    def forward(self, history, goal, metadata):
        if (history.ndim != 3 or history.shape[1:] != (6, self.feature_dim)
                or len(history) == 0 or goal.shape != (len(history), self.feature_dim)
                or metadata.shape != (len(history), 6, 2)):
            raise ValueError("Expected history [B,6,F], goal [B,F], metadata [B,6,2]")
        if any(x.dtype != torch.float32 or not torch.isfinite(x).all()
               for x in (history, goal, metadata)):
            raise ValueError("Inputs must be finite float32 tensors")
        if self.kind in ("latest_mlp", "mean_mlp"):
            image = history[:, -1] if self.kind == "latest_mlp" else history.mean(1)
            fused = torch.cat((image, goal, metadata.flatten(1)), -1)
        else:
            sequence = self.project(torch.cat((history, metadata), -1))
            if self.kind == "attention":
                # All six observations are available at decision time. No future
                # observations are present; within-window attention is unmasked.
                summary = self.temporal(sequence if self.position is None
                                        else sequence + self.position)[:, -1]
            else:
                _, state = self.temporal(sequence)
                summary = (state[0] if self.kind == "lstm" else state)[-1]
            fused = torch.cat((summary, goal), -1)
        return self.classifier(fused)


class FrozenPolicy(nn.Module):
    """Reusable image-to-logit CNN policy; head runner may reuse RAM features."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, goal, history, metadata):
        with torch.no_grad():
            goal_features = self.encoder(goal)
            history_features = self.encoder(history)
        return self.head(history_features, goal_features, metadata)
