import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from vit_gru import ViTGRUHead
from vit_gru_train import (CONFIG, check_partitions, evaluate, extract, load_checkpoint,
                           metrics, save_bytes, train_fresh, validation_inputs)
from vit_gru_debug import training_inputs

torch.set_num_threads(2)


def inputs(n=34):
    generator = torch.Generator().manual_seed(8)
    labels = torch.arange(n) % 2
    history = torch.randn(n, 6, 384, generator=generator) * .01
    goal = torch.zeros(n, 384)
    goal[:, 0] = labels.float() * 2 - 1
    return history, goal, torch.zeros(n, 6, 2), labels


def context():
    return dict(config=CONFIG, metadata_scale_depth_time=[1., 1.])


class TrainingTests(unittest.TestCase):
    def test_metrics_known_counts_and_ties(self):
        result = metrics(torch.tensor([[2.,0.],[0.,2.],[0.,2.],[2.,0.],[0.,0.]]),
                         torch.tensor([0,0,1,1,1]))
        self.assertEqual(result['confusion_matrix'], [[1,1],[1,2]])
        self.assertAlmostEqual(result['balanced_accuracy'], (0.5+2/3)/2)
        with self.assertRaises(ValueError):
            metrics(torch.full((2,2), float('nan')), torch.tensor([0,1]))

    def test_validation_does_not_mutate_weights_or_gradients(self):
        head = ViTGRUHead()
        head(*inputs()[:3]).sum().backward()
        before = {k:v.clone() for k,v in head.state_dict().items()}
        grads = [p.grad.clone() for p in head.parameters()]
        evaluate(head, inputs())
        self.assertTrue(head.training)
        for k,v in head.state_dict().items():
            self.assertTrue(torch.equal(v,before[k]))
        for p,g in zip(head.parameters(),grads):
            self.assertTrue(torch.equal(p.grad,g))

    def test_fresh_training_partial_batch_and_reload(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = train_fresh(inputs(), inputs(8), a, context(), max_epochs=20)
            self.assertEqual(first['updates'], 40)  # 32 plus final 2, every epoch
            self.assertTrue(first['reload_verified'])
            head, checkpoint = load_checkpoint(Path(a)/'latest.pt')
            self.assertLess(evaluate(head,inputs())[0]['loss'], first['initial_train']['loss'])
            self.assertEqual(checkpoint['context']['metadata_scale_depth_time'], [1.,1.])
            # A second run is freshly seeded, not continued from the first.
            second = train_fresh(inputs(), inputs(8), b, context(), max_epochs=20)
            other, _ = load_checkpoint(Path(b)/'latest.pt')
            self.assertEqual(first['initial_train'], second['initial_train'])
            for p,q in zip(head.parameters(),other.parameters()):
                self.assertTrue(torch.equal(p,q))

    def test_tied_validation_keeps_earlier_checkpoint_and_stops(self):
        original = evaluate
        def tied(head, data):
            result, logits = original(head,data)
            result['balanced_accuracy'] = 0.5
            return result, logits
        with tempfile.TemporaryDirectory() as directory, patch('vit_gru_train.evaluate', side_effect=tied):
            result = train_fresh(inputs(),inputs(8),directory,context(), patience=2)
            self.assertEqual(result['stop_reason'],'early_stopping')
            self.assertEqual(result['best_epoch'],1)
            self.assertEqual(result['completed_epochs'],3)

    def test_limits_and_nonfinite_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = train_fresh(inputs(),inputs(8),directory,context(),max_updates=1)
            self.assertEqual(result['updates'],1)
            self.assertEqual(result['status'],'incomplete')
            self.assertFalse((Path(directory)/'best.pt').exists())
        with tempfile.TemporaryDirectory() as directory:
            ticks = iter([0,121,121])
            result = train_fresh(inputs(),inputs(8),directory,context(),clock=lambda:next(ticks,121))
            self.assertEqual(result['updates'],0)
            self.assertEqual(result['stop_reason'],'head_time_limit')
            bad = list(inputs()); bad[0][0,0,0] = float('nan')
            with self.assertRaises(ValueError):
                train_fresh(bad,inputs(8),directory,context())
            with self.assertRaises(RuntimeError):
                save_bytes(directory,'too_big',b'12345',limit=4)
            self.assertFalse((Path(directory)/'too_big').exists())

    def test_training_only_scales_and_episode_overlap(self):
        record = dict(depth_relative_mm=[-5,-4,-3,-2,-1,0],
                      history_offsets_s=[-5,-4,-3,-2,-1,0], label=0, episode='a')
        records = [record,dict(record,episode='b',label=1)]
        train = SimpleNamespace(partition='train',split_position=0,indices=[0],records=records)
        _,_,scales = training_inputs(train)
        val = SimpleNamespace(indices=[1],records=records)
        before = scales.clone()
        records[1]['depth_relative_mm'] = [-500,-400,-300,-200,-100,0]
        normalized,_ = validation_inputs(val,scales)
        self.assertTrue(torch.equal(scales,before))
        self.assertAlmostEqual(normalized[0,0,0].item(), -500/scales[0].item(),places=4)
        class Dataset:
            def __init__(self,partition,indices):
                self.partition,self.indices = partition,indices
                self.records,self.split_position = records,0
            def __len__(self): return len(self.indices)
        a,b=Dataset('train',[0]),Dataset('validation',[1])
        check_partitions(a,b)
        records[1]['episode']='a'
        with self.assertRaises(ValueError): check_partitions(a,b)

    def test_real_loader_preparation_to_checkpoint_with_invented_images(self):
        from PIL import Image
        from dataset import GuidewireDataset
        from preprocessing import DINOv2Preprocessor
        class Encoder:
            def encode(self, goal, history, image_chunk_size):
                self.chunk = image_chunk_size
                with torch.inference_mode():
                    return dict(goal_features=torch.full((384,),goal.mean().item()),
                                history_features=torch.ones(6,384))
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            root=Path(root)
            Image.new('RGB',(8,8),(120,0,0)).save(root/'invented.png')
            records=[dict(paths=['invented.png']*7,depth_relative_mm=[-5,-4,-3,-2,-1,0],
                          history_offsets_s=[-5,-4,-3,-2,-1,0],label=i%2,
                          episode=str(i),case_id=str(i)) for i in range(4)]
            (root/'cases.jsonl').write_text('\n'.join(map(json.dumps,records)))
            (root/'splits.json').write_text(json.dumps([dict(fold=0,train=[0,1],validation=[2,3],test=[])]))
            train=GuidewireDataset(root,split_position=0,partition='train')
            val=GuidewireDataset(root,split_position=0,partition='validation')
            check_partitions(train,val)
            encoder=Encoder()
            feature=extract((train,val),DINOv2Preprocessor(),encoder)
            self.assertEqual(encoder.chunk,1)
            self.assertFalse(feature[0][0].is_inference())
            meta,labels,scales=training_inputs(train)
            vm,vl=validation_inputs(val,scales)
            result=train_fresh((*feature[0],meta,labels),(*feature[1],vm,vl),output,context(),max_epochs=2)
            self.assertTrue(result['reload_verified'])
            with self.assertRaises(TimeoutError):
                extract((train,val),DINOv2Preprocessor(),encoder,seconds=1,clock=iter([0,2]).__next__)


if __name__=='__main__':
    unittest.main()
