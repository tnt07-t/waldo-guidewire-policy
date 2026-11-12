"""Training-only feature normalization, persistence and experimental isolation."""
import unittest
import torch

from guidewire.temporal.models import FeatureHead
from guidewire.temporal.evaluation import fit


class FeatureNormalizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(8)
        self.h=torch.randn(16,6,8)*3+20
        self.g=torch.randn(16,8)*2-10
        self.m=torch.randn(16,6,2)

    def head(self):
        return FeatureHead('latest_goal_mlp',8,8,classifier_activation='leaky_relu',feature_normalization='standard')

    def test_training_only_role_statistics_and_constant_guard(self):
        self.h[:,:-1]=100000
        self.h[:,-1,0]=5
        head=self.head()
        with self.assertRaises(ValueError):
            head(self.h,self.g,self.m)
        head.fit_feature_normalization(self.h,self.g)
        torch.testing.assert_close(head.latest_mean,self.h[:,-1].mean(0))
        torch.testing.assert_close(head.goal_mean,self.g.mean(0))
        self.assertEqual(float(head.latest_scale[0]),1)
        normalized=(self.h[:,-1]-head.latest_mean)/head.latest_scale
        torch.testing.assert_close(normalized[:,1:].mean(0),torch.zeros(7),atol=1e-5,rtol=0)
        torch.testing.assert_close(normalized[:,1:].std(0,unbiased=False),torch.ones(7))
        before={k:v.clone() for k,v in head.named_buffers()}
        head(self.h+1000,self.g-1000,self.m)
        for k,v in head.named_buffers():
            torch.testing.assert_close(v,before[k])
        with self.assertRaises(ValueError):
            head.fit_feature_normalization(self.h+1000,self.g-1000)

    def test_normalization_does_not_change_initialization_or_parameter_count(self):
        torch.manual_seed(0)
        raw=FeatureHead('latest_goal_mlp',8,8,classifier_activation='leaky_relu')
        torch.manual_seed(0)
        normalized=self.head()
        for (name,p),(other,q) in zip(raw.named_parameters(),normalized.named_parameters()):
            self.assertEqual(name,other)
            torch.testing.assert_close(p,q)
        self.assertEqual(sum(p.numel() for p in raw.parameters()),sum(p.numel() for p in normalized.parameters()))

    def test_checkpoint_buffers_reload_and_survive_training(self):
        head=self.head()
        head.fit_feature_normalization(self.h,self.g)
        before={k:v.clone() for k,v in head.named_buffers()}
        data=(self.h,self.g,self.m,torch.arange(16)%2)
        fit(head,data,data,epochs=2,patience=1)
        for k,v in head.named_buffers():
            torch.testing.assert_close(v,before[k])
        fresh=self.head()
        fresh.load_state_dict(head.state_dict())
        head.eval(); fresh.eval()
        torch.testing.assert_close(head(*data[:3]),fresh(*data[:3]))
        legacy=FeatureHead('latest_goal_mlp',8,8)
        FeatureHead('latest_goal_mlp',8,8).load_state_dict(legacy.state_dict())

    def test_standardization_scope_and_nonfinite_rejection(self):
        with self.assertRaises(ValueError):
            FeatureHead('attention',8,8,feature_normalization='standard')
        bad=self.g.clone(); bad[0,0]=float('nan')
        with self.assertRaises(ValueError):
            self.head().fit_feature_normalization(self.h,bad)


if __name__=='__main__':
    unittest.main()
