"""Synthetic CPU checks; no timm downloads, GPU, or real cases."""
from copy import deepcopy
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

import torch
from torch import nn
from test_vit_static_finetune import Backbone
from vit_static_finetune import ViTStaticFinetune
import vit_static_finetune_train as runner
from preprocessing import CHECKPOINT


class Cases:
    def __init__(self, count=5):
        self.values = torch.randn(count, 2, 3, 1, 1)
        self.reads = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        self.reads.append(i)
        return dict(goal=self.values[i, 0], history=[self.values[i, 1]])


class Prepare:
    def prepare_image(self, image):
        return dict(tensor=image.expand(3, 518, 518))


class TrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2)
        self.base = Backbone()
        self.factory = lambda device='cpu': ViTStaticFinetune(device=device, backbone=deepcopy(self.base))
        self.model = self.factory()
        self.cases = Cases()
        self.metadata = torch.randn(5, 6, 2)
        self.labels = torch.tensor([0, 1, 0, 1, 0])
        self.data = (self.cases, self.metadata, self.labels)
        self.prepare = Prepare()
        self.config = runner.resolve_config(max_epochs=2, batch_size=3)
        self.context = dict(config=self.config, checkpoint=CHECKPOINT,
                            base_encoder_sha256=runner.state_hash(self.model.encoder),
                            frozen_sha256=runner.state_hash(self.model, frozen_only=True),
                            trainable_parameters=self.model.trainable_parameter_counts(),
                            metadata_scale_depth_time=[1., 2.])

    def test_exact_optimizer_groups_and_accumulation_equivalence(self):
        other = deepcopy(self.model)
        first = runner.make_optimizer(self.model, self.config)
        second = runner.make_optimizer(other, self.config)
        self.assertEqual([g['lr'] for g in first.param_groups], [1e-5, 1e-3])
        # SGD makes gradient equivalence visible without Adam's first-step normalization.
        for group in first.param_groups:
            group['lr'] = .01
        for group in second.param_groups:
            group['lr'] = .01
        first = torch.optim.SGD(first.param_groups)
        second = torch.optim.SGD(second.param_groups)
        indices = torch.tensor([4, 0, 2, 1, 3])
        runner.optimizer_step(self.model, first, indices, *self.data, self.prepare, 2, lambda: None)
        runner.optimizer_step(other, second, indices, *self.data, self.prepare, 5, lambda: None)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[name], atol=1e-7, rtol=1e-5)
        self.assertEqual(runner.state_hash(self.model, frozen_only=True), self.context['frozen_sha256'])
        self.assertNotEqual(runner.state_hash(self.model.encoder), self.context['base_encoder_sha256'])

    def test_validation_no_updates_or_gradients(self):
        before = runner.state_hash(self.model)
        self.model.train()
        scores, logits = runner.evaluate(self.model, *self.data, self.prepare, 2, lambda: None)
        self.assertEqual(logits.shape, (5, 2))
        self.assertEqual(scores['n'], 5)
        self.assertTrue(self.model.training)
        self.assertEqual(before, runner.state_hash(self.model))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertEqual(self.cases.reads, list(range(5)))

    def test_checkpoint_roundtrip_and_wrong_base_rejection(self):
        optimizer = runner.make_optimizer(self.model, self.config)
        runner.optimizer_step(self.model, optimizer, torch.arange(5), *self.data, self.prepare, 2, lambda: None)
        scores, expected = runner.evaluate(self.model, *self.data, self.prepare, 2, lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            runner.save_checkpoint(directory, self.model, self.context, 1, 1, scores)
            restored, saved = runner.load_checkpoint(Path(directory)/'best.pt', factory=self.factory)
            _, actual = runner.evaluate(restored, *self.data, self.prepare, 2, lambda: None)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(saved['updates'], 1)
            wrong = deepcopy(self.base)
            with torch.no_grad():
                wrong.patch_embed.weight.add_(1)
            with self.assertRaisesRegex(ValueError, 'content mismatch'):
                runner.load_checkpoint(Path(directory)/'best.pt', factory=lambda **kw: ViTStaticFinetune(backbone=wrong))
            original_save = runner.save_bytes
            with self.assertRaisesRegex(RuntimeError, 'storage limit'):
                with patch('vit_static_finetune_train.save_bytes', side_effect=lambda d, n, b: original_save(d, n, b, limit=1)):
                    runner.save_checkpoint(directory, self.model, self.context, 1, 1, scores)

    def test_complete_training_strict_ties_reload_and_last_batch(self):
        # Keep real loss/logits, force tied selection scores only.
        original = runner.evaluate
        def tied(*args, **kwargs):
            scores, logits = original(*args, **kwargs)
            return dict(scores, balanced_accuracy=.5), logits
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'evaluate', side_effect=tied):
            result = runner.train(self.model, self.data, self.data, self.prepare, directory, self.context, factory=self.factory)
            self.assertEqual(result['updates'], 4)  # two batches per epoch, including final two cases
            self.assertEqual(result['completed_epochs'], 2)
            self.assertEqual(result['best_epoch'], 1)
            self.assertTrue(result['reload_verified'])
            self.assertTrue(result['frozen_unchanged'])
            self.assertTrue((Path(directory)/'selected_validation.pt').exists())

    def test_timeout_and_update_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = iter([0., 121., 122.])
            result = runner.train(self.model, self.data, self.data, self.prepare, directory, self.context,
                                  clock=lambda: next(clock), factory=self.factory)
            self.assertEqual(result['status'], 'budget_limited')
            self.assertEqual(result['updates'], 0)
            self.assertFalse(result['reload_verified'])
        context = dict(self.context, config=runner.resolve_config(max_updates=1, batch_size=3))
        with tempfile.TemporaryDirectory() as directory:
            result = runner.train(self.model, self.data, self.data, self.prepare, directory, context, factory=self.factory)
            self.assertEqual(result['stop_reason'], 'max_updates')
            self.assertEqual(result['updates'], 1)
            self.assertEqual(result['completed_epochs'], 0)

    def test_early_stopping_and_rejected_configuration(self):
        original = runner.evaluate
        def tied(*args, **kwargs):
            scores, logits = original(*args, **kwargs)
            return dict(scores, balanced_accuracy=.5), logits
        context = dict(self.context, config=runner.resolve_config(max_epochs=5, patience=1))
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'evaluate', side_effect=tied):
            result = runner.train(self.model, self.data, self.data, self.prepare, directory, context, factory=self.factory)
            self.assertEqual(result['stop_reason'], 'early_stopping')
            self.assertEqual(result['completed_epochs'], 2)
        for overrides in (dict(training_seconds=121), dict(overall_seconds=601), dict(microbatch_size=33),
                          dict(block_lr=.001), dict(seed=-1), dict(max_updates=901), dict(training_seconds=float('nan'))):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                runner.resolve_config(**overrides)
        self.model.encoder.backbone.patch_embed.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'Optimizer scope'):
            runner.make_optimizer(self.model, self.config)


if __name__ == '__main__':
    unittest.main()
