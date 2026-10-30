"""Pretrained ResNet-18 encoder producing per-frame feature vectors."""

# TODO: define FrameEncoder(torch.nn.Module).
# TODO: load torchvision ResNet-18 with pretrained weights.
# TODO: replace the final classification layer so the output is a feature vector.
# TODO: forward - map frames of shape (B, T, C, H, W) to features of shape (B, T, D).
# TODO: decide whether to freeze the backbone.
