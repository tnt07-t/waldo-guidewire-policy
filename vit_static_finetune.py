"""Differentiable last-block DINOv2 features and the unchanged static head.

Accepts already normalized inputs; metadata scaling stays with the caller.
This module does not use the frozen runner's detached cache or head checkpoints.
"""
import torch
from torch import nn

from preprocessing import CHECKPOINT, MEAN, STD
from vit_static import ViTStaticHead


class DINOv2LastBlockEncoder(nn.Module):
    """Shared Small/14 backbone; only blocks[-1] learns, always in eval mode.

    Inject a compatible backbone for synthetic checks without timm/downloads.
    Forward accepts goal/latest [B,3,518,518] and returns separate summaries.
    Standard torch.no_grad() remains available for validation. eval() alone
    preserves gradients, including through the frozen final output norm.
    """

    def __init__(self, *, device='cpu', backbone=None):
        super().__init__()
        if backbone is None:
            import timm
            backbone = timm.create_model(CHECKPOINT, pretrained=True, num_classes=0)
            config = timm.data.resolve_model_data_config(backbone)
            if (tuple(config['input_size']) != (3, 518, 518)
                    or tuple(config['mean']) != MEAN or tuple(config['std']) != STD):
                raise ValueError('Loaded checkpoint preparation differs from preprocessing.py')
        self._validate_layout(backbone)
        self.backbone = backbone.to(device)
        self.backbone.requires_grad_(False)
        self.backbone.blocks[-1].requires_grad_(True)
        self.train()

    @staticmethod
    def _validate_layout(backbone):
        blocks = getattr(backbone, 'blocks', None)
        if (getattr(backbone, 'num_features', None) != 384
                or not isinstance(blocks, (nn.Sequential, nn.ModuleList)) or not len(blocks)
                or not isinstance(getattr(backbone, 'norm', None), nn.LayerNorm)
                or not isinstance(getattr(backbone, 'patch_embed', None), nn.Module)
                or not isinstance(getattr(backbone, 'pos_embed', None), nn.Parameter)
                or not callable(getattr(backbone, 'forward_features', None))
                or not callable(getattr(backbone, 'forward_head', None))):
            raise ValueError('Unsupported DINOv2 backbone layout')
        final = blocks[-1]
        if any(not isinstance(getattr(final, name, None), nn.Module)
               for name in ('attn', 'mlp', 'norm1', 'norm2')) or not list(final.parameters()):
            raise ValueError('Expected a full final transformer block')
        prefix = f'blocks.{len(blocks) - 1}.'
        selected = {id(p) for p in final.parameters()}
        # Shared aliases outside the final block would silently unfreeze more.
        if any(id(p) in selected and not name.startswith(prefix)
               for name, p in backbone.named_parameters(remove_duplicate=False)):
            raise ValueError('Final-block parameters must not be shared outside that block')

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # Disable dropout/stochastic depth and buffer updates.
        return self

    def trainable_parameter_counts(self):
        """Exact names -> element counts, relative to this encoder."""
        return {name: p.numel() for name, p in self.named_parameters() if p.requires_grad}

    def forward(self, goal, latest):
        if (goal.ndim != 4 or tuple(goal.shape[1:]) != (3, 518, 518)
                or len(goal) == 0 or latest.shape != goal.shape):
            raise ValueError('Expected goal and latest [B,3,518,518]')
        for value in (goal, latest):
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('Inputs must be finite normalized float32 tensors')
        self.backbone.eval()
        device = next(self.backbone.parameters()).device
        features = []
        for images in (goal, latest):
            tokens = self.backbone.forward_features(images.to(device))
            summary = self.backbone.forward_head(tokens, pre_logits=True)
            if summary.shape != (len(goal), 384) or not torch.isfinite(summary).all():
                raise ValueError('Expected finite [B,384] features')
            features.append(summary)
        return dict(goal_features=features[0], history_features=features[1].unsqueeze(1))


class ViTStaticFinetune(nn.Module):
    """Raw prepared goal/latest -> shared encoder -> fresh ViTStaticHead.

    forward(goal, latest, metadata) takes metadata [B,6,2] in chronological
    depth/time order, with caller-supplied training-only scaling. Move metadata
    and the policy to the same device. No earlier visual frames are accepted.
    """

    def __init__(self, *, device='cpu', backbone=None):
        super().__init__()
        self.encoder = DINOv2LastBlockEncoder(device=device, backbone=backbone)
        self.head = ViTStaticHead().to(device)

    def trainable_parameter_counts(self):
        """Exact policy names/counts for checking a later optimizer's scope."""
        return {name: p.numel() for name, p in self.named_parameters() if p.requires_grad}

    def forward(self, goal, latest, metadata):
        features = self.encoder(goal, latest)
        return self.head(features['history_features'], features['goal_features'], metadata)
