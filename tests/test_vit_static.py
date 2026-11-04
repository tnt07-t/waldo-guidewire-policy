"""Static comparison contracts and matched training on invented CPU inputs only."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from vit import DINOv2Encoder
from vit_static import ViTStaticHead
from vit_gru_debug import freeze_and_snapshot, verify_frozen
from vit_gru_train import (CONFIG, evaluate, extract, load_checkpoint, main, resolve_config,
                           save_checkpoint, selected_config, train_fresh)
from test_vit_gru_train import inputs


class StaticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.config = selected_config('static_latest_goal_metadata')
        self.context = dict(config=self.config, metadata_scale_depth_time=[2., 3.])

    def test_feature_order_dimensions_and_visual_history_invariance(self):
        head = ViTStaticHead()
        self.assertEqual(sum(p.numel() for p in head.parameters()), 50114)
        self.assertFalse(any(isinstance(m, torch.nn.GRU) for m in head.modules()))
        history = torch.randn(3, 6, 384)
        goal = torch.randn(3, 384)
        meta = torch.arange(36, dtype=torch.float32).reshape(3, 6, 2)
        captured = []
        handle = head.classifier[0].register_forward_pre_hook(lambda m, x: captured.append(x[0]))
        expected = head(history, goal, meta)
        handle.remove()
        torch.testing.assert_close(captured[0][:, :384], history[:, -1])
        torch.testing.assert_close(captured[0][:, 384:768], goal)
        torch.testing.assert_close(captured[0][:, 768:], meta.flatten(1))
        history[:, :-1] = torch.randn_like(history[:, :-1]) * 100
        torch.testing.assert_close(head(history, goal, meta), expected, rtol=0, atol=0)
        torch.testing.assert_close(head(history[:, -1:], goal, meta), expected, rtol=0, atol=0)

    def test_all_used_inputs_receive_gradients_and_actual_update(self):
        head = ViTStaticHead()
        history = torch.randn(8, 6, 384, requires_grad=True)
        goal = torch.randn(8, 384, requires_grad=True)
        metadata = torch.randn(8, 6, 2, requires_grad=True)
        before = [p.detach().clone() for p in head.parameters()]
        optimizer = torch.optim.Adam(head.parameters(), lr=.001)
        torch.nn.functional.cross_entropy(head(history, goal, metadata), torch.arange(8) % 2).backward()
        self.assertEqual(history.grad[:, :-1].abs().sum().item(), 0)
        for grad in (history.grad[:, -1], goal.grad, metadata.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertTrue((grad.abs().sum(0) > 0).all())
        for parameter in head.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        self.assertTrue(all(not torch.equal(p, q) for p, q in zip(head.parameters(), before)))

    def test_invalid_inputs_and_selectors(self):
        data = inputs(8)
        for index, bad in [(0, torch.zeros(8, 5, 384)), (1, torch.zeros(8, 383)),
                           (2, torch.zeros(8, 12)), (0, data[0].double()),
                           (2, torch.full((8, 6, 2), float('nan')))]:
            values = list(data[:3]); values[index] = bad
            with self.subTest(index=index), self.assertRaises(ValueError):
                ViTStaticHead()(*values)
        for config in [dict(CONFIG, model='unknown'), dict(self.config, architecture='early_goal')]:
            with self.assertRaises(ValueError):
                resolve_config(config)
        self.assertEqual(selected_config(), CONFIG)
        self.assertEqual(self.config['optimizer'], 'Adam')
        self.assertEqual(self.config['weight_decay'], 0)
        self.assertEqual(selected_config('static_latest_goal_metadata', optimizer='AdamW', weight_decay=.02)['weight_decay'], .02)

    def test_fresh_fit_partial_batches_loss_reduction_and_exact_reload(self):
        data = inputs(34)
        trained = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory, patch('torch.optim.Adam', wraps=torch.optim.Adam) as optimizer:
                result = train_fresh(data, inputs(8), directory, self.context, max_epochs=20)
                optimizer.assert_called_once()
                self.assertEqual(optimizer.call_args.kwargs, dict(lr=.001, weight_decay=0.))
                self.assertEqual(result['updates'], 40)
                self.assertTrue(result['reload_verified'])
                head, checkpoint = load_checkpoint(Path(directory) / 'latest.pt')
                scores, logits = evaluate(head, data)
                self.assertLess(scores['loss'], result['initial_train']['loss'])
                self.assertIsInstance(head, ViTStaticHead)
                self.assertEqual(checkpoint['context']['head']['trainable_parameters'], 50114)
                self.assertEqual(checkpoint['context']['head']['visual_history_frames'], 1)
                self.assertEqual(checkpoint['context']['config']['model'], self.config['model'])
                self.assertEqual(checkpoint['context']['metadata_scale_depth_time'], [2., 3.])
                torch.manual_seed(0)
                fresh = ViTStaticHead()
                self.assertEqual(evaluate(fresh, data)[0], result['initial_train'])
                self.assertTrue(any(not torch.equal(p, q) for p, q in zip(head.parameters(), fresh.parameters())))
                save_checkpoint(directory, 'roundtrip.pt', head, self.context, 20, 40, scores)
                restored, _ = load_checkpoint(Path(directory) / 'roundtrip.pt')
                torch.testing.assert_close(restored(*data[:3]), logits, rtol=0, atol=0)
                trained.append(head.state_dict())
        for name in trained[0]:
            torch.testing.assert_close(trained[0][name], trained[1][name], rtol=0, atol=0)

    def test_checkpoint_model_identity_and_legacy_gru_default(self):
        from vit_gru import ViTGRUHead
        with tempfile.TemporaryDirectory() as directory:
            for head, config in [(ViTStaticHead(), CONFIG), (ViTGRUHead(), self.config)]:
                with self.assertRaisesRegex(ValueError, 'model does not match'):
                    save_checkpoint(directory, 'bad.pt', head, dict(config=config), 1, 1, {})
            path = Path(directory) / 'legacy.pt'
            config = dict(CONFIG); del config['model']; del config['architecture']
            head = ViTGRUHead()
            torch.save(dict(format_version=1, context=dict(config=config, metadata_scale_depth_time=[1., 1.]),
                            state_dict=head.state_dict()), path)
            restored, _ = load_checkpoint(path)
            torch.testing.assert_close(restored(*inputs(8)[:3]), head(*inputs(8)[:3]), rtol=0, atol=0)
            for model, error in [('unknown', ValueError), ('gru', RuntimeError)]:
                torch.save(dict(format_version=1, context=dict(config=dict(self.config, model=model),
                                metadata_scale_depth_time=[1., 1.]), state_dict=ViTStaticHead().state_dict()), path)
                with self.assertRaises(error):
                    load_checkpoint(path)

    def test_shared_ties_patience_validation_isolation_and_limits(self):
        original = evaluate
        data, validation = inputs(8), inputs(10)
        training_batches = []
        def forward(module, values):
            if module.training and torch.is_grad_enabled():
                training_batches.append(len(values[0]))
        def tied(head, values):
            scores, logits = original(head, values)
            scores['balanced_accuracy'] = .5
            return scores, logits
        with tempfile.TemporaryDirectory() as directory, patch('vit_gru_train.evaluate', side_effect=tied):
            handle = torch.nn.modules.module.register_module_forward_pre_hook(
                lambda module, values: forward(module, values) if isinstance(module, ViTStaticHead) else None)
            try:
                result = train_fresh(data, validation, directory, self.context, patience=2)
            finally:
                handle.remove()
            self.assertEqual(training_batches, [8, 8, 8])
            self.assertEqual(result['best_epoch'], 1)
            self.assertEqual(result['completed_epochs'], 3)
            self.assertEqual(result['stop_reason'], 'early_stopping')
        for options, reason, updates in [({'max_updates': 1}, 'max_updates', 1),
                                         ({'clock': iter([0, 121, 121, 121]).__next__}, 'head_time_limit', 0)]:
            with tempfile.TemporaryDirectory() as directory:
                result = train_fresh(data, validation, directory, self.context, **options)
                self.assertEqual(result['stop_reason'], reason)
                self.assertEqual(result['updates'], updates)
        with tempfile.TemporaryDirectory() as directory:
            # save_bytes has a bound default; override its call to impose a tiny cap.
            from vit_gru_train import save_bytes
            with patch('vit_gru_train.save_bytes', side_effect=lambda d, n, b: save_bytes(d, n, b, limit=1)):
                with self.assertRaisesRegex(RuntimeError, 'storage limit'):
                    train_fresh(data, validation, directory, self.context)

    def test_only_latest_and_goal_are_prepared_encoded_and_encoder_stays_frozen(self):
        from PIL import Image
        from preprocessing import DINOv2Preprocessor
        class Backbone(torch.nn.Module):
            num_features = 384
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(384))
                self.register_buffer('unchanged', torch.tensor(1.))
                self.images = 0
            def forward_features(self, images):
                self.images += len(images)
                return images.mean((1, 2, 3))[:, None] * self.weight[None]
            def forward_head(self, tokens, pre_logits): return tokens
        backbone = Backbone()
        encoder = DINOv2Encoder(backbone=backbone)
        snapshot = freeze_and_snapshot(backbone)
        prepare = DINOv2Preprocessor()
        # Deliberately unusable earlier images prove that the static path never prepares them.
        case = dict(goal=Image.new('RGB', (8, 8), (100, 0, 0)),
                    history=[object()] * 5 + [Image.new('RGB', (8, 8), (0, 100, 0))])
        with patch.object(prepare, 'prepare_image', wraps=prepare.prepare_image) as prep:
            features = extract(([case] * 2, [case] * 2), prepare, encoder, model=self.config['model'])
        self.assertEqual(prep.call_count, 8)
        self.assertEqual(backbone.images, 8)  # two, rather than seven, images per case
        self.assertEqual(features[0][0].shape, (2, 1, 384))
        self.assertFalse(features[0][0].is_inference())
        with tempfile.TemporaryDirectory() as directory:
            data = (*features[0], torch.zeros(2, 6, 2), torch.tensor([0, 1]))
            result = train_fresh(data, data, directory, self.context, max_epochs=1)
            self.assertTrue(result['reload_verified'])
        verify_frozen(backbone, snapshot)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in backbone.parameters()))
        with self.assertRaises(TimeoutError):
            extract(([case],), prepare, encoder, model=self.config['model'], seconds=1, clock=iter([0, 2]).__next__)

    def test_cli_forwards_static_recipe_and_requires_fresh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); data = root / 'data'; data.mkdir()
            output = root / 'static-latest-goal-meta-h64-adam-wd0-split0-seed0-01'
            command = ['--data-dir', str(data), '--output-dir', str(output), '--model', self.config['model']]
            with patch.dict(os.environ, PSCRATCH=str(root), SLURM_JOB_ID='synthetic'), patch(
                    'vit_gru_train.supervise', return_value=(124, True)) as supervisor:
                self.assertEqual(main(command), 124)
                args = supervisor.call_args.args[0]
                for flag, value in [('--model', self.config['model']), ('--optimizer', 'Adam'), ('--weight-decay', '0.0')]:
                    self.assertEqual(args[args.index(flag) + 1], value)
                self.assertEqual(supervisor.call_args.kwargs['seconds'], 600)
                self.assertEqual(json.loads((output / 'summary.json').read_text())['status'], 'overall_time_limit')
                with self.assertRaises(SystemExit):
                    main(command)
                self.assertEqual(supervisor.call_count, 1)


if __name__ == '__main__':
    unittest.main()
