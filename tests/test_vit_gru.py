"""Synthetic checks only: no dataset, pretrained weights, GPU or network."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from vit import DINOv2Encoder
from vit_gru import ViTGRUHead
from vit_gru_debug import (extract_selected, fit_head, freeze_and_snapshot, main,
                           select_eight, supervise, training_inputs, verify_frozen)


def fake_dataset():
    records = [{"label": i % 2, "depth_relative_mm": [-5, -4, -3, -2, -1, 0],
                "history_offsets_s": [-10, -8, -6, -4, -2, 0]} for i in range(12)]
    # A malformed held-out row must never be read for scaling or selection.
    records.append({"label": "held out", "depth_relative_mm": "must not read"})
    return SimpleNamespace(partition="train", split_position=0,
                           indices=tuple(range(12)), records=records)


def learnable_batch():
    labels = torch.tensor([0, 1] * 4)
    goal = (labels.float() * 2 - 1)[:, None].expand(8, 384).clone()
    return torch.randn(8, 6, 384), goal, torch.randn(8, 6, 2), labels


class ViTGRUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(0)

    def test_shapes_parameter_count_and_all_input_paths_receive_gradients(self):
        head = ViTGRUHead()
        history, goal, metadata, labels = learnable_batch()
        for value in (history, goal, metadata):
            value.requires_grad_()
        logits = head(history, goal, metadata)
        self.assertEqual(logits.shape, (8, 2))
        self.assertEqual(sum(p.numel() for p in head.parameters()), 115650)
        torch.nn.functional.cross_entropy(logits, labels).backward()
        for value in (history, goal, metadata):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        self.assertTrue((history.grad.abs().sum(dim=(0, 2)) > 0).all())

    def test_metadata_alignment_and_goal_are_not_history_steps(self):
        head = ViTGRUHead()
        history, goal, metadata, _ = learnable_batch()
        captured = {}
        def capture(module, values):
            captured["steps"] = values[0].detach().clone()
        handle = head.gru.register_forward_pre_hook(capture)
        head(history, goal, metadata)
        handle.remove()
        self.assertEqual(captured["steps"].shape, (8, 6, 386))
        torch.testing.assert_close(captured["steps"][..., :384], history)
        torch.testing.assert_close(captured["steps"][..., 384:], metadata)

    def test_independent_cases_have_no_carried_hidden_state(self):
        head = ViTGRUHead()
        history, goal, metadata, _ = learnable_batch()
        batched = head(history, goal, metadata)
        head(history.flip(1), goal + 2, metadata)
        separate = torch.cat([head(history[i:i+1], goal[i:i+1], metadata[i:i+1])
                              for i in range(8)])
        torch.testing.assert_close(batched, separate, atol=1e-6, rtol=1e-5)
        self.assertFalse(torch.allclose(batched, head(history.flip(1), goal, metadata.flip(1))))

    def test_training_only_scales_preserve_zero_sign_and_channel_order(self):
        dataset = fake_dataset()
        normalized, labels, scales = training_inputs(dataset)
        expected = torch.tensor([-5., -4., -3., -2., -1., 0.], dtype=torch.float64).std(correction=0)
        torch.testing.assert_close(scales, torch.stack((expected, expected * 2)))
        self.assertTrue((normalized[:, -1] == 0).all())
        self.assertTrue((normalized[:, :-1] < 0).all())
        torch.testing.assert_close(normalized[..., 0], normalized[..., 1])
        for row in dataset.records[:12]:
            row["depth_relative_mm"] = [0] * 6
        _, _, scales = training_inputs(dataset)
        self.assertEqual(scales[0], 1)
        self.assertEqual(labels.shape, (12,))

    def test_balanced_selection_is_deterministic_unique_and_training_only(self):
        _, labels, _ = training_inputs(fake_dataset())
        selected = select_eight(labels)
        self.assertEqual(selected, select_eight(labels))
        self.assertEqual(len(set(selected)), 8)
        self.assertEqual(labels[selected].tolist().count(0), 4)
        self.assertEqual(labels[selected].tolist().count(1), 4)
        with self.assertRaisesRegex(ValueError, "four"):
            select_eight(torch.tensor([0] * 5 + [1] * 3))

    def test_reject_bad_metadata_and_nontraining_partition(self):
        for field, value in [("partition", "validation"), ("split_position", 1)]:
            dataset = fake_dataset()
            setattr(dataset, field, value)
            with self.assertRaises(ValueError):
                training_inputs(dataset)
        for key, value in [("depth_relative_mm", [float("nan")] * 6),
                           ("history_offsets_s", [-1, -2, -3, -4, -5, 0]),
                           ("depth_relative_mm", [0] * 5), ("label", 2)]:
            dataset = fake_dataset()
            dataset.records[0][key] = value
            with self.assertRaises(ValueError):
                training_inputs(dataset)

    def test_inference_features_can_train_head_without_updating_encoder(self):
        class Backbone(torch.nn.Module):
            num_features = 384
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(384))
                self.register_buffer("offset", torch.zeros(384))
                self.calls = []

            def forward_features(self, images):
                self.calls.append((len(images), self.training, torch.is_grad_enabled()))
                return images[:, 0, 0, 0, None] * self.weight + self.offset

            def forward_head(self, features, **kwargs):
                return features

        class Dataset:
            seen = []
            def __getitem__(self, index):
                self.seen.append(index)
                return index

        class Prepare:
            def prepare_case(self, index):
                marker = torch.tensor(float(index % 2 * 2 - 1))
                return {"goal": marker.expand(3, 518, 518),
                        "history": torch.arange(6.).view(6, 1, 1, 1).expand(6, 3, 518, 518)}

        backbone, dataset = Backbone(), Dataset()
        encoder = DINOv2Encoder(backbone=backbone)
        before = freeze_and_snapshot(backbone)
        history, goal = extract_selected(dataset, list(range(8)), Prepare(), encoder)
        self.assertEqual(dataset.seen, list(range(8)))
        self.assertEqual(backbone.calls, [(1, False, False)] * 56)
        self.assertFalse(torch.is_inference(history))
        self.assertFalse(torch.is_inference(goal))
        result = fit_head(ViTGRUHead(), history, goal, torch.zeros(8, 6, 2),
                          torch.tensor([0, 1] * 4), max_updates=40)
        self.assertTrue(result["finite_gradients"])
        self.assertTrue(result["head_weights_changed"])
        self.assertLess(result["final"]["loss"], result["initial"]["loss"] * 0.2)
        self.assertEqual(result["final"]["accuracy"], 1)
        verify_frozen(backbone, before)
        backbone.offset.add_(1)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            verify_frozen(backbone, before)

    def test_update_and_time_caps_are_enforced(self):
        batch = learnable_batch()
        result = fit_head(ViTGRUHead(), *batch, max_updates=2)
        self.assertEqual(result["updates"], 2)
        self.assertEqual(result["stop_reason"], "max_updates")
        ticks = iter([0, 10, 20])
        result = fit_head(ViTGRUHead(), *batch, head_seconds=1, clock=lambda: next(ticks))
        self.assertEqual(result["updates"], 0)
        self.assertFalse(result["head_weights_changed"])
        self.assertIsNone(result["finite_gradients"])
        self.assertEqual(result["stop_reason"], "head_time_limit")
        for kwargs in ({"max_updates": 301}, {"head_seconds": 121}, {"head_seconds": float("nan")}):
            with self.assertRaises(ValueError):
                fit_head(ViTGRUHead(), *batch, **kwargs)

    def test_nonfinite_inputs_and_gradients_fail(self):
        history, goal, metadata, labels = learnable_batch()
        with self.assertRaises(ValueError):
            ViTGRUHead()(history[:, :5], goal, metadata)
        with self.assertRaises(ValueError):
            ViTGRUHead()(history, goal * float("nan"), metadata)
        head = ViTGRUHead()
        next(head.parameters()).register_hook(lambda gradient: gradient * float("nan"))
        with self.assertRaisesRegex(RuntimeError, "gradients"):
            fit_head(head, history, goal, metadata, labels, max_updates=1)

    def test_supervisor_terminates_worker_at_deadline(self):
        code, timed_out = supervise([sys.executable, "-c", "import time; time.sleep(10)"], seconds=0.05)
        self.assertEqual(code, 124)
        self.assertTrue(timed_out)

    def test_timeout_writes_private_report_and_existing_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data, output = Path(directory) / "data", Path(directory) / "run"
            data.mkdir()
            argv = ["--data-dir", str(data), "--output-dir", str(output)]
            with patch.dict(os.environ, {"PSCRATCH": directory}):
                with patch("vit_gru_debug.supervise", return_value=(124, True)):
                    self.assertEqual(main(argv), 124)
                report = json.loads((output / "debug_summary.json").read_text())
                self.assertEqual(report["status"], "overall_time_limit")
                self.assertTrue(report["overall_limit_reached"])
                with self.assertRaises(SystemExit):
                    main(argv)


if __name__ == "__main__":
    unittest.main()
