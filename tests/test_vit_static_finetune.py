"""Invented CPU backbones only: gradients, parity and immutable frozen state."""
from copy import deepcopy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from preprocessing import CHECKPOINT, MEAN, STD
from vit import DINOv2Encoder
from vit_static import ViTStaticHead
from vit_static_finetune import DINOv2LastBlockEncoder, ViTStaticFinetune


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(384), nn.LayerNorm(384)
        self.attn, self.mlp = nn.Linear(384, 384), nn.Linear(384, 384)
        self.ls1, self.ls2 = nn.Parameter(torch.ones(384)), nn.Parameter(torch.ones(384))
        self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x))) * self.ls1
        return x + self.dropout(torch.tanh(self.mlp(self.norm2(x)))) * self.ls2


class Backbone(nn.Module):
    num_features = 384

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(3, 384)
        self.pos_embed = nn.Parameter(torch.randn(1, 384))
        self.blocks = nn.Sequential(Block(), Block())
        self.norm = nn.LayerNorm(384)
        self.buffer_norm = nn.BatchNorm1d(384)
        self.calls = []

    def forward_features(self, images):
        self.calls.append((self.training, torch.is_grad_enabled()))
        x = self.patch_embed(images[:, :, 0, 0]) + self.pos_embed
        return self.norm(self.blocks(self.buffer_norm(x))).unsqueeze(1)

    def forward_head(self, tokens, pre_logits=False):
        assert pre_logits
        return tokens[:, 0]


def inputs(batch=3):
    # Only tiny invented markers are materialized; expand to the input contract.
    return tuple(torch.randn(batch, 3, 1, 1).expand(-1, -1, 518, 518)
                 for _ in range(2))


class FinetuneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)

    def test_exact_scope_and_fresh_unchanged_head(self):
        backbone = Backbone()
        torch.manual_seed(9)
        expected_head = ViTStaticHead()
        torch.manual_seed(9)
        model = ViTStaticFinetune(backbone=backbone)
        expected = {f'encoder.backbone.blocks.1.{n}': p.numel()
                    for n, p in backbone.blocks[-1].named_parameters()}
        expected.update({f'head.{n}': p.numel() for n, p in model.head.named_parameters()})
        self.assertEqual(model.trainable_parameter_counts(), expected)
        self.assertEqual(model.encoder.trainable_parameter_counts(),
                         {n.removeprefix('encoder.'): c for n, c in expected.items()
                          if n.startswith('encoder.')})
        self.assertEqual(sum(p.numel() for p in model.head.parameters()), 50114)
        self.assertIs(type(model.head), ViTStaticHead)
        for name, value in expected_head.state_dict().items():
            torch.testing.assert_close(model.head.state_dict()[name], value, rtol=0, atol=0)
        self.assertFalse(backbone.norm.weight.requires_grad)
        self.assertTrue(all(p.requires_grad for p in backbone.blocks[-1].parameters()))

    def test_frozen_parity_case_order_and_metadata_packing(self):
        backbone = Backbone()
        old = DINOv2Encoder(backbone=deepcopy(backbone))
        model = ViTStaticFinetune(backbone=backbone)
        model.encoder.requires_grad_(False)
        goal, latest = inputs()
        with torch.no_grad():
            features = model.encoder(goal, latest)
        self.assertEqual(features['history_features'].shape, (3, 1, 384))
        for i in range(3):
            reference = old.encode_latest(goal[i], latest[i])
            for key in reference:
                torch.testing.assert_close(features[key][i], reference[key], rtol=1e-5, atol=1e-6)
        meta = torch.arange(36.).reshape(3, 6, 2)
        captured = []
        handle = model.head.classifier[0].register_forward_pre_hook(lambda m, x: captured.append(x[0]))
        logits = model(goal, latest, meta)
        handle.remove()
        torch.testing.assert_close(captured[0], torch.cat((features['history_features'][:, 0],
                                                         features['goal_features'], meta.flatten(1)), 1))
        self.assertEqual(logits.shape, (3, 2))
        self.assertTrue(torch.isfinite(logits).all())

    def test_both_branch_gradients_and_full_loss_update_only_selected_state(self):
        model = ViTStaticFinetune(backbone=Backbone())
        goal, latest = inputs()
        features = model.encoder(goal, latest)
        block = tuple(model.encoder.backbone.blocks[-1].parameters())
        for feature in features.values():
            # A nonconstant projection avoids the zero derivative of LayerNorm's sum.
            loss = (feature * torch.randn_like(feature)).sum()
            grads = torch.autograd.grad(loss, block, retain_graph=True)
            self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))
        meta = torch.randn(3, 6, 2, requires_grad=True)
        before = {n: v.clone() for n, v in model.state_dict().items()}
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.001)
        loss = nn.functional.cross_entropy(model(goal, latest, meta), torch.tensor([0, 1, 0]))
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertTrue((meta.grad.abs().sum(0) > 0).all())
        optimizer.step()
        trainable = model.trainable_parameter_counts()
        for name, value in model.state_dict().items():
            if name in trainable:
                self.assertFalse(torch.equal(value, before[name]), name)
            else:
                torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)

    def test_train_eval_determinism_buffers_and_gradient_modes(self):
        model = ViTStaticFinetune(backbone=Backbone())
        goal, latest = inputs(1)
        meta = torch.randn(1, 6, 2)
        expected = model(goal, latest, meta)
        for mode in (False, True, False, True):
            self.assertIs(model.train(mode), model)
            self.assertEqual(model.head.training, mode)
            self.assertTrue(all(not m.training for m in model.encoder.backbone.modules()))
            output = model(goal, latest, meta)
            self.assertTrue(output.requires_grad)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertTrue(all(call == (False, True) for call in model.encoder.backbone.calls))
        with torch.no_grad():
            output = model(goal, latest, meta)
        self.assertFalse(output.requires_grad)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_invalid_inputs_outputs_and_layouts_rejected(self):
        goal, latest = inputs(1)
        encoder = DINOv2LastBlockEncoder(backbone=Backbone())
        for bad in (goal[0], goal[:, :, :224, :224], goal[:0], goal.double(), goal * float('nan')):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                encoder(bad, latest)
        with self.assertRaises(ValueError):
            ViTStaticFinetune(backbone=Backbone())(goal, latest, torch.zeros(1, 12))
        for output in (torch.zeros(1, 383), torch.full((1, 384), float('nan'))):
            with patch.object(encoder.backbone, 'forward_head', return_value=output):
                with self.assertRaisesRegex(ValueError, 'finite'):
                    encoder(goal, latest)
        for mutate in (lambda b: setattr(b, 'blocks', nn.Sequential()),
                       lambda b: setattr(b, 'norm', nn.Identity()),
                       lambda b: setattr(b.blocks[-1], 'attn', None),
                       lambda b: setattr(b, 'alias', b.blocks[-1].attn),
                       lambda b: setattr(b, 'num_features', 768)):
            backbone = Backbone(); mutate(backbone)
            with self.assertRaises(ValueError):
                DINOv2LastBlockEncoder(backbone=backbone)

    def test_mocked_pretrained_loader_and_config_guard(self):
        config = dict(input_size=(3, 518, 518), mean=MEAN, std=STD)
        with patch.dict(sys.modules, timm=SimpleNamespace(
                create_model=lambda name, **kw: self._loader(name, kw),
                data=SimpleNamespace(resolve_model_data_config=lambda b: config))):
            DINOv2LastBlockEncoder()
            config['std'] = (1., 1., 1.)
            with self.assertRaisesRegex(ValueError, 'preparation differs'):
                DINOv2LastBlockEncoder()

    def _loader(self, name, kwargs):
        self.assertEqual(name, CHECKPOINT)
        self.assertEqual(kwargs, dict(pretrained=True, num_classes=0))
        return Backbone()


if __name__ == '__main__':
    unittest.main()
