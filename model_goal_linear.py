"""Best frozen-probe baseline: goal image → ResNet-18 → Linear(512, 2).

Matches the strongest linear-probe configuration from the split-0 study
(``goal_only`` + ``remove_maxpool``; see docs/grp1/PROBE_LINEAR.md). History,
depth, and offsets are ignored. The CNN stays frozen in eval mode.
"""

from torch import nn

from cnn import FrameEncoder


class GoalOnlyLinearModel(nn.Module):
    """Frozen FrameEncoder + linear ADVANCE/RETRACT head on the goal only."""

    def __init__(self, pretrained=True, remove_maxpool=True):
        super().__init__()
        self.remove_maxpool = remove_maxpool
        self.cnn = FrameEncoder(pretrained=pretrained, remove_maxpool=remove_maxpool)
        self.classifier = nn.Linear(FrameEncoder.feature_dim, 2)

    def train(self, mode=True):
        """Keep the ResNet frozen/eval while the linear head trains."""
        super().train(mode)
        self.cnn.eval()
        return self

    def forward(self, goal):
        """goal: [B, 3, H, W] images or [B, 512] cached features → [B, 2] logits."""
        if goal.ndim == 4:
            goal = self.cnn(goal)
        elif goal.ndim != 2:
            raise ValueError("Expected goal [B, 3, H, W] or [B, 512]")
        return self.classifier(goal)

    def load_linear_state_dict(self, state_dict):
        """Load weights saved by probe_cnn_features (nn.Linear state_dict)."""
        self.classifier.load_state_dict(state_dict)
