"""Tiny-check integration on invented inputs; no dataset/weights downloaded."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from cnn_attention_debug import (training_inputs, select_eight, fit_head,
                                 freeze_and_snapshot, verify_frozen, run_worker, main)
from temporal_models import FeatureHead


class TinyAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_balanced_selection_and_training_only_metadata(self):
        records = [{'label': i % 2, 'depth_relative_mm': [-5,-4,-3,-2,-1,0],
                    'history_offsets_s': [-5,-4,-3,-2,-1,0]} for i in range(10)]
        data = SimpleNamespace(partition='train', split_position=0, indices=range(10), records=records)
        meta, labels, scale = training_inputs(data)
        indices = select_eight(labels)
        self.assertEqual(indices, select_eight(labels))
        self.assertEqual(labels[indices].tolist().count(0), 4)
        self.assertEqual(meta.shape, (10,6,2))
        self.assertTrue((meta[:, -1] == 0).all())
        data.partition = 'validation'
        with self.assertRaises(ValueError):
            training_inputs(data)

    def test_updates_caps_and_frozen_state(self):
        torch.manual_seed(0)
        h, g, m = torch.randn(8,6,512), torch.randn(8,512), torch.randn(8,6,2)
        head = FeatureHead('attention')
        result = fit_head(head, h, g, m, torch.arange(8) % 2, max_updates=3)
        self.assertEqual(result['updates'], 3)
        self.assertTrue(result['finite_gradients'])
        self.assertTrue(result['head_weights_changed'])
        ticks = iter([0., 121., 122.])
        stopped = fit_head(head,h,g,m,torch.arange(8)%2,clock=lambda: next(ticks))
        self.assertEqual(stopped['updates'],0)
        self.assertEqual(stopped['stop_reason'],'head_time_limit')
        with self.assertRaises(ValueError):
            fit_head(head, h,g,m,torch.arange(8)%2,max_updates=301)
        backbone = nn.Linear(2,2)
        snap = freeze_and_snapshot(backbone)
        verify_frozen(backbone, snap)
        with torch.no_grad():
            backbone.weight.add_(1)
        with self.assertRaises(RuntimeError):
            verify_frozen(backbone,snap)

    def test_worker_raw_loader_to_report(self):
        from PIL import Image
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(3,512)
            def forward(self,x):
                return self.backbone(x.mean((-1,-2)))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory); data = root/'data'; data.mkdir()
            Image.new('RGB',(8,8),(100,20,50)).save(data/'image.png')
            records = [{'label': i%2, 'case_id': str(i), 'episode': str(i),
                        'paths':['image.png']*7, 'depth_relative_mm':[-5,-4,-3,-2,-1,0],
                        'history_offsets_s':[-5,-4,-3,-2,-1,0]} for i in range(8)]
            (data/'cases.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            (data/'splits.json').write_text(json.dumps([{'fold':0,'train':list(range(8)),
                                                        'validation':[],'test':[]}]))
            out=root/'out'; out.mkdir()
            args=SimpleNamespace(device='cpu',data_dir=data,output_dir=out,max_updates=2,head_seconds=120)
            with patch.dict('sys.modules', {'cnn':SimpleNamespace(FrameEncoder=Encoder)}), patch(
                    'importlib.metadata.version', return_value='synthetic-test'):
                self.assertEqual(run_worker(args),0)
            report=json.loads((out/'debug_summary.json').read_text())
            self.assertTrue(report['encoder_unchanged'])
            self.assertEqual(report['updates'],2)
            self.assertEqual(report['trainable_parameters'],103874)
            self.assertEqual(len(report['selected_training_row_indices']),8)
            self.assertFalse(list(out.glob('*.pt')))

