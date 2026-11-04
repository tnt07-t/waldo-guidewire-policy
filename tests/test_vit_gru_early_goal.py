"""Early goal wiring and compatibility, using invented CPU tensors only."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from vit_gru import ViTGRUHead
from vit_gru_train import (CONFIG, evaluate, load_checkpoint, main, resolve_config,
                           save_checkpoint, train_fresh)


class EarlyGoalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        torch.set_num_threads(2)
        self.data = (torch.randn(8, 6, 384), torch.randn(8, 384),
                     torch.randn(8, 6, 2), torch.tensor([0, 1] * 4))

    def test_goal_conditions_every_memory_update_and_preserves_input_order(self):
        history, goal, metadata, _ = self.data
        for architecture in ('late_goal', 'early_goal'):
            with self.subTest(architecture=architecture):
                head = ViTGRUHead(architecture)
                states, steps = [], []
                def capture(module, values, output):
                    steps.append(values[0].detach().clone())
                    states.append(output[0])  # all six recurrent states
                handle = head.gru.register_forward_hook(capture)
                try:
                    head(history, goal, metadata)
                    head(history, goal + 1, metadata)
                finally:
                    handle.remove()
                torch.testing.assert_close(steps[0][..., :384], history)
                torch.testing.assert_close(steps[0][..., 384:386], metadata)
                if architecture == 'early_goal':
                    self.assertEqual(steps[0].shape, (8, 6, 770))
                    torch.testing.assert_close(steps[0][..., 386:], goal[:, None].expand(-1, 6, -1))
                    # Each memory update changes with the goal, before classification.
                    self.assertTrue(((states[0] - states[1]).abs().sum((0, 2)) > 0).all())
                else:
                    self.assertEqual(steps[0].shape, (8, 6, 386))
                    torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)

    def test_early_goal_backward_and_update_without_direct_classifier_goal_path(self):
        head = ViTGRUHead('early_goal')
        self.assertEqual(sum(p.numel() for p in head.parameters()), 189378)
        # Disable direct goal fusion to prove gradients can travel through the GRU.
        with torch.no_grad():
            head.classifier[0].weight[:, 64:] = 0
        history, goal, metadata, labels = self.data
        for value in (history, goal, metadata):
            value.requires_grad_()
        before = head.gru.weight_ih_l0.detach().clone()
        optimizer = torch.optim.Adam(head.parameters(), lr=.001)
        logits = head(history, goal, metadata)
        self.assertEqual(logits.shape, (8, 2))
        torch.nn.functional.cross_entropy(logits, labels).backward()
        for value in (history, goal, metadata):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        self.assertTrue((history.grad.abs().sum((0, 2)) > 0).all())
        for parameter in head.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before, head.gru.weight_ih_l0))
        self.assertTrue(all(torch.isfinite(p).all() for p in head.parameters()))

    def test_early_goal_cases_do_not_share_hidden_state(self):
        head = ViTGRUHead('early_goal')
        expected = head(*self.data[:3])
        head(*(x.flip(0) for x in self.data[:3]))
        actual = torch.cat([head(*(x[i:i+1] for x in self.data[:3])) for i in range(8)])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_early_goal_training_records_recipe_and_reloads_predictions(self):
        config = dict(CONFIG, architecture='early_goal', optimizer='Adam', weight_decay=0.)
        ctx = dict(config=config, metadata_scale_depth_time=[1., 1.])
        with tempfile.TemporaryDirectory() as directory, patch(
                'torch.optim.Adam', wraps=torch.optim.Adam) as optimizer:
            result = train_fresh(self.data, self.data, directory, ctx, max_epochs=2)
            self.assertTrue(result['reload_verified'])
            optimizer.assert_called_once()
            self.assertEqual(optimizer.call_args.kwargs, dict(lr=.001, weight_decay=0.))
            head, saved = load_checkpoint(Path(directory) / 'latest.pt')
            self.assertEqual(head.architecture, 'early_goal')
            self.assertEqual(saved['context']['config'], dict(config, max_epochs=2))
            scores, expected = evaluate(head, self.data)
            save_checkpoint(directory, 'roundtrip.pt', head, ctx, 2, 2, scores)
            restored, _ = load_checkpoint(Path(directory) / 'roundtrip.pt')
            torch.testing.assert_close(restored(*self.data[:3]), expected, rtol=0, atol=0)

    def test_legacy_missing_architecture_loads_exact_predictions(self):
        head = ViTGRUHead()
        legacy = dict(CONFIG, optimizer='Adam', weight_decay=0.)
        del legacy['architecture']
        expected = head(*self.data[:3])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pt'
            # Construct the old format directly: new saving must not supply the field.
            torch.save(dict(format_version=1, state_dict=head.state_dict(),
                            context=dict(config=legacy, metadata_scale_depth_time=[1., 1.])), path)
            restored, saved = load_checkpoint(path)
            self.assertNotIn('architecture', saved['context']['config'])
            self.assertEqual(restored.architecture, 'late_goal')
            torch.testing.assert_close(restored(*self.data[:3]), expected, rtol=0, atol=0)
        self.assertEqual(resolve_config(legacy)['architecture'], 'late_goal')

    def test_unknown_or_mismatched_architecture_rejected(self):
        with self.assertRaises(ValueError):
            ViTGRUHead('unknown')
        with self.assertRaises(ValueError):
            resolve_config(dict(CONFIG, architecture='unknown'))
        with tempfile.TemporaryDirectory() as directory:
            head = ViTGRUHead('early_goal')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                save_checkpoint(directory, 'bad.pt', head,
                                dict(config=CONFIG), 1, 1, {})
            path = Path(directory) / 'invalid.pt'
            for architecture, error in [('unknown', ValueError), ('late_goal', RuntimeError)]:
                torch.save(dict(format_version=1, state_dict=head.state_dict(), context=dict(
                    config=dict(CONFIG, architecture=architecture),
                    metadata_scale_depth_time=[1., 1.])), path)
                with self.assertRaises(error):
                    load_checkpoint(path)

    def test_cli_forwards_explicit_comparison_and_preserves_defaults(self):
        for flags, architecture, optimizer, decay in [
                ([], 'late_goal', 'AdamW', '0.01'),
                (['--architecture', 'early_goal', '--optimizer', 'Adam', '--weight-decay', '0'],
                 'early_goal', 'Adam', '0.0')]:
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                data, output = root / 'data', root / 'run'
                data.mkdir()
                with patch.dict(os.environ, PSCRATCH=str(root), SLURM_JOB_ID='synthetic'), patch(
                        'vit_gru_train.supervise', return_value=(124, True)) as supervisor:
                    self.assertEqual(main(['--data-dir', str(data), '--output-dir', str(output)] + flags), 124)
                command = supervisor.call_args.args[0]
                for flag, value in [('--architecture', architecture), ('--optimizer', optimizer),
                                    ('--weight-decay', decay)]:
                    self.assertEqual(command[command.index(flag) + 1], value)
                self.assertEqual(supervisor.call_args.kwargs['seconds'], 600)
                self.assertEqual(json.loads((output / 'summary.json').read_text())['status'], 'overall_time_limit')


if __name__ == '__main__':
    unittest.main()
