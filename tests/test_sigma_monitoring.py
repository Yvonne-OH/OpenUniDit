"""Coverage, sample weighting and noninterference of diagnostic statistics."""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import optimizations
optimizations.__path__ = [str(ROOT / 'optimizations'), *list(optimizations.__path__)]
from optimizations.sigma_monitoring import SigmaMonitor


def config(interval=2, tensor_interval=2):
    return {'dataset': {'buckets': ['256_square', '512x384']},
            'monitoring': {'enabled': True, 'sigma': {'enabled': True,
                'every_n_steps': interval, 'tensor_every_n_steps': tensor_interval}}}


class SigmaTests(unittest.TestCase):
    def test_sample_weighting_resolution_cfg_and_text_strata(self):
        monitor = SigmaMonitor(config(), 'cpu')
        for step, losses, size, drop in ((1, [1.], (256,256), [False]),
                                        (2, [3.,5.,7.], (512,384), [True,False,True])):
            monitor.begin_update(step)
            n = len(losses)
            mask = torch.ones(n, 513, dtype=torch.bool)
            if step == 1: mask[:, 2:] = False
            else: mask[0,128:] = False; mask[1,512:] = False
            sample = torch.ones(n,1,1,1,1)
            monitor.record_micro(torch.full((n,),.25),torch.tensor(losses),mask,drop,size,
                                 prediction=sample,latents=sample,noise=sample)
            result = monitor.end_update(step)
            if step == 1: self.assertEqual(result,{})
        prefix='diagnostics/all/sigma_0.2-0.3'
        self.assertEqual(result[prefix+'/samples'],4)
        self.assertEqual(result[prefix+'/loss_mean'],4)
        self.assertAlmostEqual(result[prefix+'/loss_std'],math.sqrt(20/3))
        self.assertAlmostEqual(result[prefix+'/loss_variance'],20/3)
        self.assertEqual(result[prefix+'/loss_contribution'],4)
        self.assertAlmostEqual(result[prefix+'/sigma_mean'],.25)
        self.assertEqual(result[prefix+'/text_tokens_mean'],(2+128+512+513)/4)
        self.assertEqual(result['diagnostics/resolution/256x256/loss_mean'],1)
        self.assertEqual(result['diagnostics/resolution/512x384/loss_mean'],5)
        self.assertEqual(result['diagnostics/cfg/dropped/loss_mean'],5)
        self.assertEqual(result['diagnostics/cfg/kept/loss_mean'],3)
        self.assertEqual(result['diagnostics/text_tokens/0-128/samples'],2)
        self.assertEqual(result['diagnostics/text_tokens/129-512/samples'],1)
        self.assertEqual(result['diagnostics/text_tokens/513-plus/samples'],1)
        self.assertEqual(result['diagnostics/output/sigma_0.2-0.3/samples'],3)
        self.assertEqual(result['diagnostics/output_sampled_updates'],1)

    def test_output_moments_have_known_scale_and_alignment(self):
        monitor=SigmaMonitor(config(1,1),'cpu');monitor.begin_update(1)
        p=torch.ones(2,1,1,1,2);x=torch.zeros_like(p);eps=2*torch.ones_like(p)
        monitor.record_micro(torch.tensor([.05,.05]),torch.ones(2),torch.ones(2,3),[False,False],
                             (256,256),prediction=p,latents=x,noise=eps)
        values=monitor.end_update(1);prefix='diagnostics/output/sigma_0.0-0.1/'
        for key,value in {'samples':2,'pred_rms':1,'target_rms':2,'latent_rms':0,
                          'noise_rms':2,'error_mean':-1,'error_rms':1,'error_variance':0,
                          'pred_variance':0,'target_variance':0,'latent_variance':0,'noise_variance':0,
                          'pred_mean':1,'target_mean':2,'normalized_mse':.25,
                          'projection_gain':.5,'pred_target_cosine':1}.items():
            self.assertEqual(values[prefix+key],value)

    def test_variance_distinguishes_bias_from_spread(self):
        monitor=SigmaMonitor(config(1,1),'cpu');monitor.begin_update(1)
        p=torch.tensor([1.,3.]).view(1,1,1,1,2)
        x=torch.tensor([0.,2.]).view_as(p);eps=torch.tensor([2.,0.]).view_as(p)
        loss=(p-(eps-x)).square().flatten(1).mean(1)
        monitor.record_micro(torch.tensor([.35]),loss,torch.ones(1,3),[False],(256,256),
                             prediction=p,latents=x,noise=eps)
        result=monitor.end_update(1);prefix='diagnostics/output/sigma_0.3-0.4/'
        for key,value in {'pred_mean':2,'pred_variance':1,'target_mean':0,'target_variance':4,
                          'latent_variance':1,'noise_variance':1,'error_mean':2,'error_variance':9}.items():
            self.assertAlmostEqual(result[prefix+key],value)

    def test_nonfinite_samples_are_visible_and_loss_is_not_faked(self):
        monitor=SigmaMonitor(config(1,1),'cpu');monitor.begin_update(1)
        p=torch.tensor([1.,float('inf')]).view(2,1,1,1,1)
        monitor.record_micro(torch.tensor([.4,.4]),torch.tensor([1.,float('nan')]),torch.ones(2,2),
                             [False,True],(7,9),prediction=p,latents=p*0,noise=p*0)
        values=monitor.end_update(1);prefix='diagnostics/all/sigma_0.4-0.5/'
        self.assertEqual(values['diagnostics/nonfinite_samples'],1)
        self.assertEqual(values[prefix+'samples'],2)
        self.assertEqual(values[prefix+'nonfinite_samples'],1)
        self.assertNotIn(prefix+'loss_mean',values)
        self.assertNotIn('diagnostics/output/sigma_0.4-0.5/pred_rms',values)
        self.assertEqual(values['diagnostics/resolution/other/samples'],2)
        self.assertTrue(all(math.isfinite(v) for v in values.values()))

    def test_no_host_sync_rng_or_gradient_mutation_in_record_micro(self):
        monitor=SigmaMonitor(config(1,1),'cpu');monitor.begin_update(1)
        p=torch.nn.Parameter(torch.ones(2,1,1,1,2));x=torch.zeros_like(p);eps=2*torch.ones_like(p)
        losses=(p-eps).square().flatten(1).mean(1)
        expected=torch.autograd.grad(losses.sum(),p,retain_graph=True)[0]
        rng=torch.get_rng_state().clone()
        with patch.object(torch.Tensor,'item',side_effect=AssertionError('host sync')),\
             patch.object(torch.Tensor,'tolist',side_effect=AssertionError('host sync')),\
             patch.object(torch.Tensor,'cpu',side_effect=AssertionError('host copy')),\
             patch('optimizations.sigma_monitoring.dist.all_reduce',side_effect=AssertionError('per-micro collective')):
            monitor.record_micro(torch.tensor([.1,.9]),losses,torch.ones(2,4),[True,False],
                                 (256,256),prediction=p,latents=x,noise=eps)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        losses.sum().backward();torch.testing.assert_close(p.grad,expected)
        self.assertFalse(monitor._base.requires_grad)
        self.assertFalse(monitor._output.requires_grad)

    def test_empty_bins_reset_and_unsampled_outputs_do_not_need_tensors(self):
        monitor=SigmaMonitor(config(2,2),'cpu');monitor.begin_update(1)
        monitor.record_micro(torch.tensor([.9]),torch.tensor([2.]),torch.ones(1,2),[False],(256,256))
        self.assertEqual(monitor.end_update(1),{})
        monitor.begin_update(2);values=monitor.end_update(2)
        self.assertEqual(values['diagnostics/all/sigma_0.0-0.1/samples'],0)
        self.assertNotIn('diagnostics/all/sigma_0.0-0.1/loss_mean',values)
        monitor.begin_update(4);empty=monitor.end_update(4)
        self.assertEqual(empty['diagnostics/samples'],0)
        self.assertNotIn('diagnostics/all/sigma_0.9-1.0/loss_mean',empty)

    def test_disabled_and_configuration_validation(self):
        with patch('optimizations.sigma_monitoring.torch.zeros',side_effect=AssertionError('disabled allocation')):
            monitor=SigmaMonitor({},'cpu');monitor.begin_update(1)
            monitor.record_micro(None,None,None,None,None)
            self.assertEqual(monitor.end_update(1),{})
        for interval,tensor in ((0,1),(10,0),(10,3)):
            with self.assertRaises(ValueError):SigmaMonitor(config(interval,tensor),'cpu')


if __name__=='__main__':unittest.main()
