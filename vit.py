"""Inference-only DINOv2 Small/14 features; real inputs stay on Perlmutter."""

import torch

from preprocessing import CHECKPOINT


class DINOv2Encoder:
    """Encode normalized 518-square inputs with separate goal/history outputs.

    The pretrained model is loaded only at construction, through timm. Loading
    may download public weights; image tensors are processed locally. This
    wrapper performs no training and does not select a future freezing policy.
    ``backbone`` injection supports synthetic tests without timm or weights.
    """

    def __init__(self, *, device="cpu", backbone=None):
        # Load pretrained image features without a category-classification head.
        if backbone is None:
            import timm

            backbone = timm.create_model(CHECKPOINT, pretrained=True, num_classes=0)
            # Check that the loaded weights expect our preparation settings.
            config = timm.data.resolve_model_data_config(backbone)
            from preprocessing import MEAN, STD

            if (tuple(config["input_size"]) != (3, 518, 518)
                    or tuple(config["mean"]) != MEAN or tuple(config["std"]) != STD):
                raise ValueError("Loaded checkpoint preparation differs from preprocessing.py")
        self.device = torch.device(device)
        self.backbone = backbone.to(self.device).eval()
        if self.backbone.num_features != 384:
            raise ValueError("Expected DINOv2 Small feature width 384")

    def _encode_images(self, images, image_chunk_size):
        outputs = []
        # Process groups of whole images to limit memory; no image is cropped.
        for chunk in images.split(image_chunk_size):
            tokens = self.backbone.forward_features(chunk.to(self.device))
            # Select the whole-image summary (class token), before prediction.
            features = self.backbone.forward_head(tokens, pre_logits=True)
            if features.shape != (len(chunk), 384):
                raise ValueError("Expected one 384-value feature vector per image")
            if not torch.isfinite(features).all():
                raise ValueError("Encoder returned non-finite features")
            outputs.append(features)
        return torch.cat(outputs)

    @torch.inference_mode()
    def encode_latest_grid(self, goal, latest, *, image_chunk_size=1):
        """Frozen full-view patch features → row-major 4×4 regional means.

        No input crop/resolution change. Adaptive averaging uses the entire
        37×37 grid; boundary bins can overlap because 37 is not divisible by 4.
        Features include padding and global context, not isolated crop features.
        """
        if type(image_chunk_size) is not int or image_chunk_size <= 0:
            raise ValueError('image_chunk_size must be a positive integer')
        for value in (goal, latest):
            if value.shape != (3, 518, 518):
                raise ValueError('Expected a 518-square RGB goal and latest image')
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('Inputs must be finite normalized float32 tensors')
        # The chosen non-register DINOv2 Small/14 has one prefix (CLS) token.
        # Reject unexpected layouts rather than silently treating prefixes as patches.
        if getattr(self.backbone, 'num_prefix_tokens', None) != 1:
            raise ValueError('Expected one CLS prefix token for the selected checkpoint')
        self.backbone.eval()
        grids = []
        for chunk in torch.stack((goal, latest)).split(image_chunk_size):
            tokens = self.backbone.forward_features(chunk.to(self.device))
            if tokens.shape != (len(chunk), 1370, 384) or not torch.isfinite(tokens).all():
                raise ValueError('Expected finite CLS + 37x37 patch tokens of width 384')
            spatial = tokens[:, 1:].reshape(len(chunk), 37, 37, 384).permute(0, 3, 1, 2)
            pooled = torch.nn.functional.adaptive_avg_pool2d(spatial, (4, 4))
            grids.append(pooled.permute(0, 2, 3, 1).reshape(len(chunk), 16, 384))
        features = torch.cat(grids)
        return {'goal_features': features[0], 'history_features': features[1:2]}

    @torch.inference_mode()
    def encode_latest(self, goal, latest, *, image_chunk_size=1):
        """Encode only one goal and latest image; preserve the shared preparation."""
        if type(image_chunk_size) is not int or image_chunk_size <= 0:
            raise ValueError("image_chunk_size must be a positive integer")
        for value in (goal, latest):
            if value.shape != (3, 518, 518):
                raise ValueError("Expected a 518-square RGB goal and latest image")
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError("Inputs must be finite normalized float32 tensors")
        self.backbone.eval()
        features = self._encode_images(torch.stack((goal, latest)), image_chunk_size)
        return {"goal_features": features[0], "history_features": features[1:2]}

    # Skip learning bookkeeping; this wrapper is for extraction only.
    @torch.inference_mode()
    def encode(self, goal, history, *, image_chunk_size=1):
        """Return goal_features [384], history_features [6,384] for one case.

        Also accepts a case batch: goal [B,3,518,518] and history
        [B,6,3,518,518], returning [B,384] and [B,6,384]. Chunk size counts
        individual images, not cases. Labels and metadata are never accepted.
        """
        if type(image_chunk_size) is not int or image_chunk_size <= 0:
            raise ValueError("image_chunk_size must be a positive integer")
        # Treat one case as a batch of one, then remove that dimension at return.
        single_case = goal.ndim == 3
        if single_case:
            goal, history = goal.unsqueeze(0), history.unsqueeze(0)
        if (goal.ndim != 4 or tuple(goal.shape[1:]) != (3, 518, 518)
                or history.shape != (len(goal), 6, 3, 518, 518) or len(goal) == 0):
            raise ValueError("Expected separate 518-square RGB goal and six history tensors")
        for inputs in (goal, history):
            if inputs.dtype != torch.float32 or not torch.isfinite(inputs).all():
                raise ValueError("Inputs must be finite normalized float32 tensors")
        self.backbone.eval()
        # Use shared weights while keeping goal separate from chronological history.
        goal_features = self._encode_images(goal, image_chunk_size)
        # [B,6,3,H,W] -> [B*6,3,H,W] lets ViT read individual images.
        # Restore [B,6,384] afterward: same case order and same frame order.
        history_features = self._encode_images(
            history.reshape(-1, 3, 518, 518), image_chunk_size
        ).reshape(len(goal), 6, 384)
        if single_case:
            goal_features, history_features = goal_features[0], history_features[0]
        return {"goal_features": goal_features, "history_features": history_features}
