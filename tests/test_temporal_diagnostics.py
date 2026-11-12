"""Invented inputs only: diagnostic consistency, sensitivity and probe controls."""
import unittest
import json
import copy
import os
from pathlib import Path
import tempfile
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from temporal_models import FeatureHead
from temporal_evaluation import fit, assess
from temporal_evaluation import main as train_main
from temporal_diagnostics import (stages, routing_summary, interventions,
                                  representation_stats, metadata_recoverability,
                                  spatial_sensitivity, classifier_health)
from temporal_diagnostics import main as diagnostic_main


@contextmanager
def injected_cnn(encoder):
    # Restore only this fake module. Restoring the entire sys.modules mapping
    # can unload lazily imported torch modules while leaving native registrations.
    previous = sys.modules.get('cnn')
    sys.modules['cnn'] = SimpleNamespace(FrameEncoder=encoder)
    try:
        yield
    finally:
        if previous is None:
            del sys.modules['cnn']
        else:
            sys.modules['cnn'] = previous


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)
        self.data = (torch.randn(16,6,8),torch.randn(16,8),torch.randn(16,6,2),torch.arange(16)%2)

    def test_inspection_preserves_forward_and_state(self):
        for position in (True,False):
            head = FeatureHead('attention',8,8,position_embeddings=position).eval()
            before = {k:v.clone() for k,v in head.state_dict().items()}
            normal = head(*self.data[:3]).detach()
            observed = stages(head,self.data)
            torch.testing.assert_close(normal,observed['logits'])
            self.assertEqual(observed['attention'].shape,(16,4,6,6))
            torch.testing.assert_close(observed['attention'].sum(-1),torch.ones(16,4,6))
            for k,v in head.state_dict().items():
                torch.testing.assert_close(v,before[k])
            self.assertEqual(len(routing_summary(observed['attention'])['latest_query_mean_per_head']),4)

    def test_uniform_routing_and_collapsed_representation(self):
        summary = routing_summary(torch.full((2,4,6,6),1/6))
        np.testing.assert_allclose(summary['latest_query_normalized_entropy_per_head'],1.,atol=1e-6)
        stats = representation_stats(torch.ones(8,6,4))
        self.assertEqual(stats['effective_rank'],0)
        self.assertEqual(stats['between_case_std_mean'],0)

    def test_dead_relu_health_control_and_no_gradient_mutation(self):
        head=FeatureHead('latest_goal_mlp',8,8,dropout=0)
        with torch.no_grad():
            head.classifier[0].weight.zero_()
            head.classifier[0].bias.fill_(-1)
            head.classifier[3].bias.copy_(torch.tensor([-.1,.1]))
        before={k:v.clone() for k,v in head.state_dict().items()}
        r=classifier_health(head,self.data,self.data)
        self.assertEqual(r['units_zero_for_all_cases'],8)
        self.assertEqual(r['fraction_zero_activations'],1)
        self.assertLess(r['advance_probability_std'],1e-6)
        self.assertEqual(r['roc_auc'],.5)
        self.assertEqual(r['gradient_l2']['classifier.0.weight'],0)
        self.assertEqual(r['gradient_l2']['classifier.3.weight'],0)
        self.assertGreater(r['gradient_l2']['classifier.3.bias'],0)
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        for k,v in head.state_dict().items():
            torch.testing.assert_close(v,before[k])

    def test_active_relu_health_control(self):
        head=FeatureHead('latest_goal_mlp',8,8,dropout=0)
        with torch.no_grad():
            head.classifier[0].bias.fill_(3)
        r=classifier_health(head,self.data,self.data)
        self.assertLess(r['units_zero_for_all_cases'],8)
        self.assertGreater(r['gradient_l2']['classifier.0.weight'],0)
        self.assertGreater(r['advance_probability_std'],0)

    def test_leaky_relu_restores_gradients_for_same_negative_preactivations(self):
        relu=FeatureHead('latest_goal_mlp',8,8,dropout=0)
        with torch.no_grad():
            relu.classifier[0].weight.zero_()
            relu.classifier[0].bias.fill_(-1)
            relu.classifier[3].bias.copy_(torch.tensor([-.1,.1]))
        leaky=FeatureHead('latest_goal_mlp',8,8,dropout=0,classifier_activation='leaky_relu')
        leaky.load_state_dict(relu.state_dict())
        dead=classifier_health(relu,self.data,self.data)
        active=classifier_health(leaky,self.data,self.data)
        self.assertEqual(dead['gradient_l2']['classifier.0.weight'],0)
        self.assertGreater(active['gradient_l2']['classifier.0.weight'],0)
        self.assertEqual(active['activation'],'LeakyReLU')
        self.assertAlmostEqual(active['negative_slope'],.01)
        before=leaky.classifier[0].weight.detach().clone()
        fit(leaky,self.data,self.data,epochs=1,patience=1)
        self.assertFalse(torch.equal(before,leaky.classifier[0].weight))
        self.assertEqual(sum(p.numel() for p in leaky.parameters()),sum(p.numel() for p in relu.parameters()))

    def test_intervention_negative_control_latest_ignores_old_images(self):
        head = FeatureHead('latest_mlp',8,8,dropout=0)
        r = interventions(head,self.data,self.data)
        for i in range(1,6):
            self.assertEqual(r['controls']['replace_visual_f'+str(i)]['mean_absolute_probability_change'],0)
        self.assertEqual(r['controls']['repeat_latest_visual']['prediction_flip_fraction'],0)

    def test_ridge_probe_recovers_constructed_metadata_not_constant_summary(self):
        torch.manual_seed(3)
        meta = torch.randn(80,6,2)
        meta[:,-1] = 0
        result = metadata_recoverability({'raw':meta,'constant':torch.zeros(80,4)},meta,
                                         [str(i//2) for i in range(80)])
        self.assertEqual(result['status'],'completed')
        for channel in ('depth','time'):
            self.assertGreater(result['results']['raw'][channel]['mean_r2_nonconstant_steps'],.9)
            self.assertLess(result['results']['constant'][channel]['mean_r2_nonconstant_steps'],.2)
            self.assertEqual(result['results']['raw'][channel]['variable_steps'],5)
        self.assertEqual(metadata_recoverability({'raw':meta},meta,['same']*80)['status'],'unavailable')

    def test_epoch_metrics_eval_counts_and_selected_vs_final(self):
        head = FeatureHead('attention',8,8)
        callbacks=[]
        result=fit(head,self.data,self.data,epochs=3,patience=2,epoch_callback=lambda h:callbacks.append(len(h)))
        self.assertEqual(len(callbacks),result['epochs_run'])
        for row in result['history']:
            for partition in ('train','validation'):
                m=row[partition]
                self.assertEqual(sum(m['prediction_counts'].values()),16)
                self.assertAlmostEqual(m['balanced_accuracy'],(m['retract_recall']+m['advance_recall'])/2)
                self.assertTrue(np.isfinite(m['loss']))
        selected=result['history'][result['best_epoch']-1]['train']
        self.assertAlmostEqual(assess(head,self.data)['balanced_accuracy'],selected['balanced_accuracy'])
        self.assertEqual(result['final_epoch_metrics']['train'],result['history'][-1]['train'])
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_eval_metrics_preserve_weights_and_rng(self):
        head=FeatureHead('attention',8,8,dropout=.5).train()
        state={k:v.clone() for k,v in head.state_dict().items()}
        rng=torch.random.get_rng_state().clone()
        first=assess(head,self.data)
        self.assertEqual(first,assess(head,self.data))
        torch.testing.assert_close(torch.random.get_rng_state(),rng)
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        for k,v in head.state_dict().items():
            torch.testing.assert_close(v,state[k])

    def test_final_state_is_saved_before_selected_state_restoration(self):
        head=FeatureHead('latest_mlp',8,8,dropout=0)
        captured={}
        def save(state,epoch):
            captured['state']=copy.deepcopy(state)
            captured['epoch']=epoch
        result=fit(head,self.data,self.data,epochs=3,patience=2,
                   final_checkpoint_callback=save)
        final=FeatureHead('latest_mlp',8,8,dropout=0)
        final.load_state_dict(captured['state'])
        self.assertEqual(captured['epoch'],result['epochs_run'])
        self.assertAlmostEqual(assess(final,self.data)['loss'],result['final_epoch_metrics']['train']['loss'])
        self.assertAlmostEqual(assess(head,self.data)['loss'],result['history'][result['best_epoch']-1]['train']['loss'])

    def test_spatial_edits_reencode_one_image_at_a_time(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__(); self.calls=0
            def forward(self,x):
                self.calls+=1
                return x.mean((-1,-2)).repeat(1,3)[:,:8]
        encoder=Encoder()
        geometry={'padding_ltrb':(0,1,0,1),'resized_size_wh':(9,7)}
        prepared={'goal':torch.ones(3,9,9),'history':torch.ones(6,3,9,9),
                  'preparation':{'goal':geometry,'history':[geometry]*6}}
        data=tuple(x[:1] for x in self.data)
        head=FeatureHead('attention',8,8).eval()
        r=spatial_sensitivity(head,encoder,prepared,data)
        self.assertEqual(encoder.calls,63)
        self.assertEqual(len(r['roles']),7)
        self.assertTrue(all(len(v)==9 for v in r['roles'].values()))

    def test_checkpoint_to_diagnostic_report_cli(self):
        from PIL import Image
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone=nn.Linear(3,512)
                with torch.no_grad():
                    self.backbone.weight.fill_(.1); self.backbone.bias.zero_()
            def forward(self,x):
                return self.backbone(x.mean((-1,-2)))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root=Path(directory); data=root/'data'; data.mkdir()
            Image.new('RGB',(8,8),(100,30,70)).save(data/'image.png')
            rows=[{'paths':['image.png']*7,'label':i%2,'episode':str(i),'case_id':str(i),
                   'depth_relative_mm':[-5,-4,-3,-2,-1,0],'history_offsets_s':[-5,-4,-3,-2,-1,0]}
                  for i in range(16)]
            (data/'cases.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (data/'splits.json').write_text(json.dumps([{'fold':0,'train':list(range(8)),
                                                        'validation':list(range(8,12)),'test':list(range(12,16))}]))
            with patch.dict(os.environ,{'PSCRATCH':str(root)}),injected_cnn(Encoder):
                train_main(['--data-dir',str(data),'--output-dir',str(root/'run'),'--split-position','0',
                            '--models','attention','--device','cpu','--epochs','2','--width','8'])
                diagnostic_main(['--data-dir',str(data),'--checkpoint',str(root/'run'/'attention-seed0.pt'),
                                 '--output-dir',str(root/'inspect'),'--device','cpu','--spatial-case-index','0'])
                diagnostic_main(['--data-dir',str(data),'--checkpoint',str(root/'run'/'attention-seed0-final.pt'),
                                 '--output-dir',str(root/'final-inspect'),'--device','cpu'])
            report=json.loads((root/'inspect'/'diagnostics.json').read_text())
            self.assertEqual(report['partition'],'train')
            self.assertEqual(report['cases'],8)
            self.assertTrue(report['encoder_unchanged'])
            self.assertIn('spatial_sensitivity',report)
            self.assertTrue((root/'inspect'/'summary.md').exists())
            self.assertTrue((root/'inspect'/'spatial_sensitivity.private.md').exists())
            self.assertNotIn('Signed ADVANCE-probability changes',(root/'inspect'/'summary.md').read_text())
            self.assertNotIn('row_index',report)
            self.assertTrue((root/'run'/'learning_curves.csv').exists())
            self.assertTrue((root/'run'/'learning_summary.md').exists())
            final_report=json.loads((root/'final-inspect'/'diagnostics.json').read_text())
            self.assertEqual(final_report['checkpoint_role'],'final')
            self.assertEqual(final_report['checkpoint_epoch'],2)


if __name__ == '__main__':
    unittest.main()
