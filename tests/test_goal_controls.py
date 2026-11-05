"""Input-role controls: exclusion, training and minimal image IO."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from torch import nn

from temporal_models import FeatureHead, FrozenPolicy
from temporal_evaluation import main
from test_temporal_diagnostics import injected_cnn


class GoalControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)

    def test_exact_input_roles_and_parameter_counts(self):
        h=torch.randn(8,6,512,requires_grad=True)
        g=torch.randn(8,512,requires_grad=True)
        m=torch.randn(8,6,2,requires_grad=True)
        for kind,parameters in [('latest_goal_mlp',65730),('goal_metadata_mlp',33730)]:
            head=FeatureHead(kind,dropout=0).eval()
            out=head(h,g,m)
            edited=h.detach().clone(); edited[:,:-1]+=100
            torch.testing.assert_close(out,head(edited,g,m))
            grads=torch.autograd.grad(out.square().sum(),(h,g,m),allow_unused=True)
            self.assertGreater(grads[1].abs().sum(),0)
            if kind=='latest_goal_mlp':
                self.assertGreater(grads[0][:,-1].abs().sum(),0)
                self.assertEqual(float(grads[0][:,:-1].abs().sum()),0)
                self.assertIsNone(grads[2])
                torch.testing.assert_close(out,head(h,g,m+100))
            else:
                self.assertIsNone(grads[0])
                self.assertGreater(grads[2].abs().sum(),0)
            self.assertEqual(sum(p.numel() for p in head.parameters()),parameters)

    def test_frozen_policy_does_not_encode_history(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__(); self.calls=0
            def forward(self,x):
                self.calls+=1
                return torch.ones(len(x),8)
        for kind in ('latest_goal_mlp','goal_metadata_mlp'):
            enc=Encoder(); policy=FrozenPolicy(enc,FeatureHead(kind,8,8)).eval()
            policy(torch.zeros(2,3,8,8),torch.zeros(2,6,3,8,8),torch.zeros(2,6,2))
            self.assertEqual(enc.calls,2 if kind=='latest_goal_mlp' else 1)

    def test_real_runner_with_missing_history_images(self):
        from PIL import Image
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__(); self.backbone=nn.Linear(3,512)
            def forward(self,x):
                return self.backbone(x.mean((-1,-2)))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root=Path(directory); data=root/'data'; data.mkdir()
            Image.new('RGB',(8,8),(100,20,30)).save(data/'goal.png')
            Image.new('RGB',(8,8),(50,40,90)).save(data/'latest.png')
            rows=[{'paths':['goal.png']+['unread-history.png']*5+['latest.png'],'case_id':str(i),'episode':str(i),
                   'label':i%2,'depth_relative_mm':[-5,-4,-3,-2,-1,0],'history_offsets_s':[-5,-4,-3,-2,-1,0]}
                  for i in range(16)]
            (data/'cases.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (data/'splits.json').write_text(json.dumps([{'fold':0,'train':list(range(8)),
                                                        'validation':list(range(8,12)),'test':list(range(12,16))}]))
            with patch.dict(os.environ,{'PSCRATCH':str(root)}),injected_cnn(Encoder):
                main(['--data-dir',str(data),'--output-dir',str(root/'run'),'--split-position','0',
                      '--models','latest_goal_mlp','goal_metadata_mlp','--device','cpu','--epochs','2','--benchmark'])
            report=json.loads((root/'run'/'metrics.json').read_text())
            self.assertTrue(report['encoder_unchanged'])
            self.assertEqual(report['config']['extraction_visual_roles'],'goal + latest')
            for kind in ('latest_goal_mlp','goal_metadata_mlp'):
                r=report['results'][kind+':seed=0']
                self.assertEqual(r['latency']['images'],2 if kind=='latest_goal_mlp' else 1)
                saved=torch.load(root/'run'/(kind+'-seed0.pt'),weights_only=True)
                head=FeatureHead(kind)
                head.load_state_dict(saved['head_state'])
                self.assertEqual(r['epochs_run'],2)
                self.assertTrue((root/'run'/(kind+'-seed0-final.pt')).exists())


if __name__=='__main__':
    unittest.main()
