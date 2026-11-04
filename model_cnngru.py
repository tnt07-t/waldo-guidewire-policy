"""Full CNN-GRU model: history + goal in, yes / no logits out.

Data (one batch of B, after prepare_case + collate):
  history            [B, T, 3, H, W]
  depth_relative_mm  [B, T]
  history_offsets_s  [B, T]
  goal               [B, 3, H, W]
  label              [B]  (0 = no, 1 = yes)

Training step:
  logits = model(batch["history"], batch["depth_relative_mm"],
                 batch["history_offsets_s"], batch["goal"])
  loss = nn.functional.cross_entropy(logits, batch["label"])
  loss.backward(); optimizer.step(); optimizer.zero_grad()

Prediction: logits.argmax(dim=-1)  # 0 = no, 1 = yes
"""

from torch import nn

from gru import CNNGRUEncoder
from mlp_cnn import GoalClassifier, encode_goal


class CNNGRUModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = CNNGRUEncoder(pretrained=pretrained)  # history path
        self.classifier = GoalClassifier()                   # 640 -> 128 -> 2

    def forward(self, history, depth, offsets, goal):
        gru_summary = self.encoder(history, depth, offsets)    # [B, 128]
        goal_features = encode_goal(self.encoder.cnn, goal)    # [B, 512] same CNN
        return self.classifier(gru_summary, goal_features)     # [B, 2] logits
