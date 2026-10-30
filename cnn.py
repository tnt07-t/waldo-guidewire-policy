"""Planned per-image CNN encoder; no model is implemented yet.

Origin commit 77947cd proposes pretrained ResNet-18. Backbone selection,
preprocessing, and freezing still require team agreement.
"""

# TODO: agree on the backbone, weights, input preparation, and freezing policy.
# TODO: define FrameEncoder(torch.nn.Module) once model work is authorized.
# TODO: replace the classification head with per-image feature output.
# TODO: encode history (B, 6, C, H, W) into features (B, 6, D).
# TODO: define goal encoding separately: the goal is not a seventh history step.
# TODO: agree on goal/history fusion and shared weights with the temporal models.
# Depth and time stay aligned with the six history steps; fusion belongs in the
# policy model. The raw dataset reader returns Pillow images, not these tensors.
