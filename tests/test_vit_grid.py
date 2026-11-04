"""Spatial extraction and bounded action training on invented inputs only."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from vit import DINOv2Encoder
from vit_grid import ViTGridHead
from vit_gru_debug import freeze_and_snapshot, verify_frozen
from vit_gru_train import (CONFIG, evaluate, extract, head_description, load_checkpoint,
                           main, resolve_config, save_checkpoint, selected_config, train_fresh)


def grid_inputs(n=8):
    rng = torch.Generator().manual_seed(19)
    labels = torch.arange(n) % 2
    latest = torch.randn(n, 1, 16, 384, generator=rng) * .01
    goal = torch.randn(n, 16, 384, generator=rng) * .01
    goal[:, :, 0] = labels[:, None].float() * 2 - 1
    return latest, goal, torch.zeros(n, 6, 2), labels


class PatchBackbone(torch.nn.Module):
    num_features = 384
    num_prefix_tokens = 1

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(384))
        self.register_buffer('fixed', torch.tensor(1.))
        self.images = 0

    def forward_features(self, images):
        self.images += len(images)
        rows = torch.arange(37, device=images.device)[:, None] * 100
        cols = torch.arange(37, device=images.device)[None, :]
        locations = (rows + cols).reshape(1, 1369, 1)
        patches = (locations + images.mean((1, 2, 3))[:, None, None]) * self.weight
        # Large prefix distinguishes accidental CLS inclusion in spatial pooling.
        return torch.cat((torch.full((len(images), 1, 384), 100000., device=images.device), patches), dim=1)

    def forward_head(self, *args, **kwargs):
        raise AssertionError('Spatial extraction must not use the whole-image head')


class GridTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.config = selected_config('static_grid_goal_metadata')
        self.context = dict(config=self.config, metadata_scale_depth_time=[2., 3.])

    def test_spatial_pooling_roles_prefix_exclusion_and_chunk_equivalence(self):
        backbone = PatchBackbone()
        encoder = DINOv2Encoder(backbone=backbone)
        goal, latest = torch.ones(3, 518, 518), torch.zeros(3, 518, 518)
        one = encoder.encode_latest_grid(goal, latest)
        two = encoder.encode_latest_grid(goal, latest, image_chunk_size=2)
        self.assertEqual(one['history_features'].shape, (1, 16, 384))
        self.assertEqual(one['goal_features'].shape, (16, 384))
        # Independently average adaptive bins; verifies row/column order and every region.
        for row in range(4):
            r0, r1 = row * 37 // 4, ((row + 1) * 37 + 3) // 4
            for col in range(4):
                c0, c1 = col * 37 // 4, ((col + 1) * 37 + 3) // 4
                expected = 100 * (r0 + r1 - 1) / 2 + (c0 + c1 - 1) / 2
                self.assertEqual(one['history_features'][0, row * 4 + col, 0], expected)
        for key in one:
            torch.testing.assert_close(one[key], two[key], rtol=0, atol=0)
        torch.testing.assert_close(one['goal_features'], one['history_features'][0] + 1)

    def test_encoder_rejects_wrong_token_layout_and_nonfinite_inputs(self):
        backbone = PatchBackbone()
        encoder = DINOv2Encoder(backbone=backbone)
        image = torch.zeros(3, 518, 518)
        for bad in (image.double(), torch.zeros(3, 224, 224), image + float('nan')):
            with self.assertRaises(ValueError):
                encoder.encode_latest_grid(bad, image)
        for chunk in (0, True, 1.5):
            with self.assertRaises(ValueError):
                encoder.encode_latest_grid(image, image, image_chunk_size=chunk)
        backbone.num_prefix_tokens = 2
        with self.assertRaises(ValueError):
            encoder.encode_latest_grid(image, image)
        backbone.num_prefix_tokens = 1
        for tokens in (torch.zeros(1, 1369, 384), torch.zeros(1, 1370, 383),
                       torch.full((1, 1370, 384), float('nan'))):
            with patch.object(backbone, 'forward_features', return_value=tokens), self.assertRaises(ValueError):
                encoder.encode_latest_grid(image, image)

    def test_projection_sharing_feature_order_and_bounded_parameter_count(self):
        head = ViTGridHead()
        self.assertEqual(sum(p.numel() for p in head.parameters()), 39890)
        self.assertFalse(any(isinstance(m, torch.nn.GRU) for m in head.modules()))
        history, goal, metadata = grid_inputs()[:3]
        metadata = torch.arange(96, dtype=torch.float32).reshape(8, 6, 2)
        captured = []
        hook = head.classifier[0].register_forward_pre_hook(lambda m, x: captured.append(x[0]))
        head(history, goal, metadata)
        hook.remove()
        torch.testing.assert_close(captured[0][:, :256], head.projection(history[:, 0]).flatten(1))
        torch.testing.assert_close(captured[0][:, 256:512], head.projection(goal).flatten(1))
        torch.testing.assert_close(captured[0][:, 512:], metadata.flatten(1))
        description = head_description(head)
        self.assertEqual(description['spatial_grid'], [4, 4])
        self.assertEqual(description['visual_history_frames'], 1)
        self.assertEqual(description['regional_projection_width'], 16)

    def test_spatial_arrangements_can_change_predictions_despite_same_global_mean(self):
        head = ViTGridHead()
        with torch.no_grad():
            for p in head.parameters():
                p.zero_()
            head.projection[0].weight[0, 0] = 1
            head.classifier[0].weight[0, 0] = 1  # Read upper-left region only.
            head.classifier[2].weight[1, 0] = 1
        history, goal, metadata, _ = grid_inputs(2)
        history.zero_(); goal.zero_()
        history[0, 0, 0, 0] = 2
        history[1, 0, 15, 0] = 2
        torch.testing.assert_close(history[0].mean(1), history[1].mean(1))
        logits = head(history, goal, metadata)
        self.assertEqual(logits[:, 1].tolist(), [2., 0.])

    def test_inputs_and_parameters_receive_finite_gradients_and_update(self):
        head = ViTGridHead()
        history = torch.randn(8, 1, 16, 384, requires_grad=True)
        goal = torch.randn(8, 16, 384, requires_grad=True)
        metadata = torch.randn(8, 6, 2, requires_grad=True)
        before = [p.detach().clone() for p in head.parameters()]
        optimizer = torch.optim.Adam(head.parameters(), lr=.001)
        torch.nn.functional.cross_entropy(head(history, goal, metadata), torch.arange(8) % 2).backward()
        for value in (history, goal, metadata):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertTrue((value.grad.abs().sum(0) > 0).all())
        for p in head.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        optimizer.step()
        self.assertTrue(all(not torch.equal(p, q) for p, q in zip(head.parameters(), before)))

    def test_invalid_head_inputs_and_selector_defaults(self):
        data = grid_inputs()
        for index, bad in [(0, torch.zeros(8, 6, 16, 384)), (1, torch.zeros(8, 384)),
                           (2, torch.zeros(8, 12)), (0, data[0].double()),
                           (1, data[1] + float('inf'))]:
            values = list(data[:3]); values[index] = bad
            with self.subTest(index=index), self.assertRaises(ValueError):
                ViTGridHead()(*values)
        self.assertEqual(selected_config(), CONFIG)
        self.assertEqual(self.config['optimizer'], 'Adam')
        self.assertEqual(self.config['weight_decay'], 0)
        with self.assertRaises(ValueError):
            resolve_config(dict(self.config, architecture='early_goal'))

    def test_real_extraction_to_backward_and_reload_with_frozen_invented_backbone(self):
        from PIL import Image
        from preprocessing import DINOv2Preprocessor
        backbone = PatchBackbone()
        encoder = DINOv2Encoder(backbone=backbone)
        snapshot = freeze_and_snapshot(backbone)
        prepare = DINOv2Preprocessor()
        case = dict(goal=Image.new('RGB', (8, 8), (100, 0, 0)),
                    history=[object()] * 5 + [Image.new('RGB', (8, 8), (0, 100, 0))])
        with patch.object(prepare, 'prepare_image', wraps=prepare.prepare_image) as prep:
            features = extract(([case] * 2, [case] * 2), prepare, encoder, model=self.config['model'])
        self.assertEqual(prep.call_count, 8)
        self.assertEqual(backbone.images, 8)
        self.assertEqual(features[0][0].shape, (2, 1, 16, 384))
        self.assertFalse(features[0][0].is_inference())
        with tempfile.TemporaryDirectory() as directory:
            data = (*features[0], torch.zeros(2, 6, 2), torch.tensor([0, 1]))
            result = train_fresh(data, data, directory, self.context, max_epochs=1)
            self.assertTrue(result['reload_verified'])
        verify_frozen(backbone, snapshot)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in backbone.parameters()))
        with self.assertRaises(TimeoutError):
            extract(([case],), prepare, encoder, model=self.config['model'], seconds=1, clock=iter([0, 2]).__next__)

    def test_fresh_learning_partial_batches_and_checkpoint_identity(self):
        data = grid_inputs(34)
        trained = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                result = train_fresh(data, grid_inputs(), directory, self.context, max_epochs=20)
                self.assertEqual(result['updates'], 40)
                self.assertTrue(result['reload_verified'])
                head, saved = load_checkpoint(Path(directory) / 'latest.pt')
                scores, logits = evaluate(head, data)
                self.assertLess(scores['loss'], result['initial_train']['loss'])
                self.assertIsInstance(head, ViTGridHead)
                self.assertEqual(saved['context']['head']['trainable_parameters'], 39890)
                self.assertEqual(saved['context']['metadata_scale_depth_time'], [2., 3.])
                save_checkpoint(directory, 'roundtrip.pt', head, self.context, 20, 40, scores)
                restored, _ = load_checkpoint(Path(directory) / 'roundtrip.pt')
                torch.testing.assert_close(restored(*data[:3]), logits, rtol=0, atol=0)
                trained.append(head.state_dict())
                with self.assertRaisesRegex(ValueError, 'model does not match'):
                    save_checkpoint(directory, 'wrong.pt', head, dict(config=CONFIG), 1, 1, {})
                saved['context']['config']['model'] = 'static_latest_goal_metadata'
                torch.save(saved, Path(directory) / 'wrong.pt')
                with self.assertRaises(RuntimeError):
                    load_checkpoint(Path(directory) / 'wrong.pt')
        for name in trained[0]:
            torch.testing.assert_close(trained[0][name], trained[1][name], rtol=0, atol=0)

    def test_selection_validation_isolation_and_training_limits(self):
        data, validation = grid_inputs(), grid_inputs(10)
        original, batches = evaluate, []
        def tied(head, values):
            scores, logits = original(head, values)
            scores['balanced_accuracy'] = .5
            return scores, logits
        def forward(module, values):
            if isinstance(module, ViTGridHead) and module.training and torch.is_grad_enabled():
                batches.append(len(values[0]))
        with tempfile.TemporaryDirectory() as directory, patch('vit_gru_train.evaluate', side_effect=tied):
            hook = torch.nn.modules.module.register_module_forward_pre_hook(forward)
            try:
                result = train_fresh(data, validation, directory, self.context, patience=2)
            finally:
                hook.remove()
            self.assertEqual(batches, [8, 8, 8])
            self.assertEqual(result['best_epoch'], 1)
            self.assertEqual(result['completed_epochs'], 3)
            self.assertEqual(result['stop_reason'], 'early_stopping')
        with tempfile.TemporaryDirectory() as directory:
            result = train_fresh(data, validation, directory, self.context, max_updates=1)
            self.assertEqual(result['updates'], 1)
            self.assertEqual(result['stop_reason'], 'max_updates')
        head = ViTGridHead()
        head(*data[:3]).sum().backward()
        before, grads = [p.clone() for p in head.parameters()], [p.grad.clone() for p in head.parameters()]
        evaluate(head, validation)
        for p, value, grad in zip(head.parameters(), before, grads):
            torch.testing.assert_close(p, value, rtol=0, atol=0)
            torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)

    def test_cli_forwards_grid_recipe_and_requires_distinct_fresh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); data = root / 'data'; data.mkdir()
            output = root / 'grid4x4-p16-h64-adam-wd0-split0-seed0-01'
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
