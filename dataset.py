"""PyTorch Dataset for guidewire decision points.

Each item is one decision point: a sequence of image frames, a depth history,
and an advance/retract label.
"""

# TODO: define GuidewireDataset(torch.utils.data.Dataset).
# TODO: __init__ - take the data root and a split, index decision points.
# TODO: __len__ - number of decision points.
# TODO: __getitem__ - return (frames, depth_history, label).
# TODO: frames - load and stack the image frames for the decision point.
# TODO: depth_history - collect the preceding depth values.
# TODO: label - encode advance/retract.
