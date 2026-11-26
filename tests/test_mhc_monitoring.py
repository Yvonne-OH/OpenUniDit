"""Monitoring must preserve weighted statistics and avoid per-micro host syncs."""
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import optimizations
optimizations.__path__=[str(Path(__file__).resolve().parents[1]/'optimizations'),*list(optimizations.__path__)]
from optimizations.mhc_monitoring import TrainingMonitor


class Site(torch.nn.Module):
    def __init__(self):
        super().__init__();self.collect_stats=False;self.last_stats=None


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.mhc_attn=Site();self.mhc_ffn=Site()


class Model(torch.nn.Module):
    n_streams=2
    mhc_stat_keys=('pre_mean','sinkhorn_row_error','res_min','nonfinite_count')
    def __init__(self):
        super().__init__();self.blocks=torch.nn.ModuleList([Block()]);self.read_calls=0
    def set_mhc_collect_stats(self,enabled):
        for m in (self.blocks[0].mhc_attn,self.blocks[0].mhc_ffn):m.collect_stats=enabled
    def set_stats(self,mean,error,minimum,nonfinite):
        for m in (self.blocks[0].mhc_attn,self.blocks[0].mhc_ffn):
            m.last_stats={k:torch.tensor(v) for k,v in zip(self.mhc_stat_keys,(mean,error,minimum,nonfinite))}
    def mhc_stats(self):
        self.read_calls+=1
        result={}
        for name,m in self.named_modules():
            if isinstance(m,Site) and m.last_stats:
                result.update({name+'/'+k:v for k,v in m.last_stats.items()})
        result.update({'readout/'+k:torch.tensor(v) for k,v in {'mean':.25,'min':.1,'max':.4,'sum':1.,'rms':.3}.items()})
        return result


class MonitorTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('GIGA_MHC_MODEL_ROOT'),'needs staged current mHC model')
    def test_actual_mhc_contract_forward_backward(self):
        import models
        models.__path__=[str(Path(os.environ['GIGA_MHC_MODEL_ROOT'])/'models'),*list(models.__path__)]
        from models.giga_transformer_mhc_transfer import GigaExpTransformerMHC
        spec={'dim':32,'ffn_dim':64,'layers':2,'heads':2,'text_dim':16,
            'in_channels':4,'out_channels':4,'rope_axes_dims':[4,4,4,4],
            'adaln_type':'basis','adaln_rank':8,'text_trans':{'enabled':True,'layers':1,'heads':2}}
        model=GigaExpTransformerMHC(**spec,n_streams=4,sinkhorn_iters=10)
        config={'model':spec,'monitoring':{'enabled':True,'every_n_steps':1,'peak_tflops_per_gpu':989}}
        monitor=TrainingMonitor(config,'cpu',model,True)
        monitor.begin_update(1)
        mask=torch.tensor([[1,1,1],[1,1,0]],dtype=torch.bool)
        out=model(torch.randn(2,4,1,2,2),torch.tensor([200.,600.]),torch.randn(2,3,16),mask)
        out.square().mean().backward()
        monitor.record_micro(2,4,mask)
        payload=monitor.end_update(1,grad_norm=1.)
        self.assertGreater(len(payload),60)
        self.assertLess(payload['mhc/all/sinkhorn_row_error'],1e-4)
        self.assertEqual(payload['mhc/all/nonfinite_count'],0)
        self.assertTrue(all(math.isfinite(x) for x in payload.values()))
        self.assertIn('performance/mfu_dit_estimate',payload)

    def exercise(self,distributed=False):
        clock=iter([0.,2.,2.])
        model=Model();monitor=TrainingMonitor({'monitoring':{'enabled':True,'every_n_steps':2}},'cpu',model,True,lambda:next(clock))
        self.assertFalse(monitor.begin_update(1))
        monitor.record_micro(1,2,torch.tensor([[1,1,0,0]],dtype=torch.bool))
        self.assertEqual(monitor.end_update(1,skipped_batches=1,
            data_wait_seconds=.2,data_wait_max_seconds=.15,data_requests=2,
            data_read_seconds=.1,data_read_max_seconds=.07),{})
        self.assertEqual(model.read_calls,0)
        self.assertTrue(monitor.begin_update(2))
        self.assertTrue(model.blocks[0].mhc_attn.collect_stats)
        model.set_stats(.2,.01,.1,0.)
        monitor.record_micro(1,2,torch.ones(1,4,dtype=torch.bool))
        model.set_stats(.8,.04,.02,3.)
        monitor.record_micro(2,2,torch.tensor([[1,1],[1,0]],dtype=torch.bool))
        if not distributed:
            payload=monitor.end_update(2,lr=3e-5,grad_norm=torch.tensor(5.),skipped_batches=2,
                data_wait_seconds=.4,data_wait_max_seconds=.25,data_requests=3,
                data_read_seconds=.3,data_read_max_seconds=.2)
        else:
            def reduce(tensor,op):
                if op==torch.distributed.ReduceOp.SUM:tensor.mul_(2)
            with patch('optimizations.mhc_monitoring.dist.is_available',return_value=True),\
                 patch('optimizations.mhc_monitoring.dist.is_initialized',return_value=True),\
                 patch('optimizations.mhc_monitoring.dist.get_world_size',return_value=2),\
                 patch('optimizations.mhc_monitoring.dist.all_reduce',side_effect=reduce) as collective:
                payload=monitor.end_update(2,lr=3e-5,grad_norm=torch.tensor(5.),skipped_batches=2,
                    data_wait_seconds=.4,data_wait_max_seconds=.25,data_requests=3,
                    data_read_seconds=.3,data_read_max_seconds=.2)
                self.assertEqual(collective.call_count,3)
        self.assertFalse(model.blocks[0].mhc_attn.collect_stats)
        self.assertAlmostEqual(payload['mhc/all/pre_mean'],(.2*6+.8*8)/14,places=6)
        self.assertAlmostEqual(payload['mhc/all/sinkhorn_row_error'],.04,places=6)
        self.assertAlmostEqual(payload['mhc/all/res_min'],.02,places=6)
        self.assertEqual(payload['performance/skipped_batches_window'],3)
        self.assertEqual(payload['train/grad_norm'],5)
        self.assertEqual(payload['performance/updates_per_sec'],1)
        self.assertEqual(payload['performance/micro_batches_per_update'],1.5)
        self.assertEqual(payload['performance/text_padding_fraction'],.25)
        world=2 if distributed else 1
        self.assertEqual(payload['performance/samples_per_sec'],2*world)
        self.assertEqual(payload['performance/text_tokens_per_sec'],4.5*world)
        self.assertEqual(payload['mhc/all/nonfinite_count'],6*world)
        self.assertEqual(payload['data/requests_global_sum'],5*world)
        self.assertEqual(payload['data/requests_mean_per_rank'],5)
        for name,total,longest in [('fetch_and_sync',.6,.25),('read_wait',.4,.2)]:
            self.assertAlmostEqual(payload[f'data/{name}_seconds_global_sum'],total*world)
            self.assertAlmostEqual(payload[f'data/{name}_seconds_mean_per_rank'],total)
            self.assertAlmostEqual(payload[f'data/{name}_seconds_slowest_rank'],total)
            self.assertAlmostEqual(payload[f'data/{name}_seconds_max_request'],longest)
            self.assertAlmostEqual(payload[f'data/{name}_wall_fraction_per_rank'],total/2)
        return monitor,payload

    def test_token_weighting_mixed_batches_and_extrema(self):self.exercise()
    def test_distributed_sums_means_extrema_and_shared_skip(self):self.exercise(True)

    def test_disabled_does_not_traverse_or_allocate(self):
        class Poison:
            def named_modules(self):raise AssertionError('must not traverse disabled model')
        with patch('optimizations.mhc_monitoring.torch.zeros',side_effect=AssertionError('allocation forbidden')):
            monitor=TrainingMonitor({'monitoring':{'enabled':False}},'cpu',Poison(),False)
            self.assertFalse(monitor.begin_update(50))
            with monitor.phase('vae'):pass
            monitor.record_micro(2,10,None)
            self.assertEqual(monitor.end_update(50,data_wait_seconds=object(),data_read_seconds=object()),{})
        self.assertFalse(TrainingMonitor({},'cpu',Poison(),True).enabled)

    def test_dense_has_no_mhc_contract_and_reports_general_metrics(self):
        class Dense(torch.nn.Linear):
            def named_modules(self, *args, **kwargs):
                raise AssertionError('dense monitor must not inspect mHC sites')
        model=Dense(2,2)
        clock=iter([0.,2.,2.])
        monitor=TrainingMonitor({'monitoring':{'enabled':True,'every_n_steps':2}},
                                'cpu',model,False,lambda:next(clock))
        for step in (1,2):
            monitor.begin_update(step)
            model(torch.ones(2,2)).square().mean().backward()
            monitor.record_micro(2,4,torch.tensor([[1,1,0],[1,0,0]],dtype=torch.bool))
            payload=monitor.end_update(step,grad_norm=3.,loss=float('nan') if step==1 else .2,
                                       data_requests=1,data_read_seconds=.1)
            if step==1:self.assertEqual(payload,{})
        self.assertEqual(payload['train/grad_norm'],3.)
        self.assertEqual(payload['health/grad_norm_missing'],0.)
        self.assertEqual(payload['health/nonfinite_loss_updates_window'],1.)
        self.assertEqual(payload['performance/samples_per_sec'],2.)
        self.assertEqual(payload['performance/text_padding_fraction'],.5)
        self.assertAlmostEqual(payload['data/read_wait_seconds_global_sum'],.2)
        self.assertIn('memory/gpu_peak_allocated_gib',payload)
        self.assertFalse(any(key.startswith('mhc/') for key in payload))

    def test_dense_uses_two_nonempty_collectives_and_excludes_mhc_flops(self):
        config={'monitoring':{'enabled':True,'every_n_steps':1,'peak_tflops_per_gpu':1},
            'model':{'dim':4,'ffn_dim':8,'layers':2,'text_dim':3,'in_channels':1,'out_channels':1,
                     'patch_size':[1,1,1],'text_trans':{'enabled':True,'layers':1}}}
        model=torch.nn.Linear(1,1);clock=iter([0.,2.,2.])
        monitor=TrainingMonitor(config,'cpu',model,False,lambda:next(clock))
        monitor.begin_update(1)
        monitor.record_micro(2,2,torch.tensor([[1,1,1],[1,0,0]],dtype=torch.bool))
        def reduce(tensor,op):
            self.assertGreater(tensor.numel(),0)
            if op==torch.distributed.ReduceOp.SUM:tensor.mul_(2)
        with patch('optimizations.mhc_monitoring.dist.is_available',return_value=True),\
             patch('optimizations.mhc_monitoring.dist.is_initialized',return_value=True),\
             patch('optimizations.mhc_monitoring.dist.get_world_size',return_value=2),\
             patch('optimizations.mhc_monitoring.dist.all_reduce',side_effect=reduce) as collective:
            payload=monitor.end_update(1)
        self.assertEqual(collective.call_count,2)
        self.assertEqual(payload['health/grad_norm_missing'],1)
        self.assertNotIn('train/grad_norm',payload)
        # Same tiny architecture as the mHC test, minus its 15360 mapping FLOPs.
        self.assertAlmostEqual(payload['performance/dit_tflops_per_gpu'],25312/2/1e12,places=15)

    def test_data_cpu_accumulation_before_report_and_window_reset(self):
        model=Model();clock=iter([0.,2.,2.,4.,4.])
        monitor=TrainingMonitor({'monitoring':{'enabled':True,'every_n_steps':2}},'cpu',model,True,lambda:next(clock))
        monitor.begin_update(1)
        # Non-reporting data accumulation must neither allocate tensors nor read GPU memory.
        with patch('optimizations.mhc_monitoring.torch.tensor',side_effect=AssertionError('per-update allocation forbidden')),\
             patch('optimizations.mhc_monitoring.torch.as_tensor',side_effect=AssertionError('per-update allocation forbidden')):
            self.assertEqual(monitor.end_update(1,data_wait_seconds=.5,data_wait_max_seconds=.3,
                data_read_seconds=.2,data_read_max_seconds=.15,data_requests=4),{})
        monitor.begin_update(2)
        payload=monitor.end_update(2)
        self.assertEqual(payload['data/requests_global_sum'],4)
        self.assertEqual(payload['data/fetch_and_sync_seconds_global_sum'],.5)
        self.assertEqual(payload['data/read_wait_wall_fraction_per_rank'],.1)
        monitor.begin_update(4)
        fresh=monitor.end_update(4)
        for key,value in fresh.items():
            if key.startswith('data/'):
                self.assertEqual(value,0.,key)

    def test_nonfinite_values_are_counted_not_published_as_finite(self):
        model=Model();monitor=TrainingMonitor({'monitoring':{'enabled':True,'every_n_steps':1}},'cpu',model,True)
        monitor.begin_update(1);model.set_stats(float('nan'),.01,.1,7.)
        monitor.record_micro(1,2,torch.ones(1,3,dtype=torch.bool))
        payload=monitor.end_update(1,grad_norm=float('inf'))
        self.assertEqual(payload['health/nonfinite_monitor_values'],2)
        self.assertEqual(payload['health/nonfinite_grad_norm'],1)
        self.assertEqual(payload['mhc/all/nonfinite_count'],14)
        self.assertNotIn('mhc/all/pre_mean',payload)
        self.assertNotIn('train/grad_norm',payload)
        self.assertTrue(all(math.isfinite(x) for x in payload.values()))

    def test_pending_phase_events_never_wait(self):
        model=Model();monitor=TrainingMonitor({'monitoring':{'enabled':True}},'cpu',model,True)
        class Pending:
            def query(self):return False
            def elapsed_time(self,other):raise AssertionError('must not access pending event')
            def synchronize(self):raise AssertionError('must never synchronize')
        monitor._events=[(0,Pending(),Pending())]
        sums,counts=monitor._take_phase_events()
        self.assertEqual(sums,[0.]*5);self.assertEqual(counts,[0.]*5)
        self.assertEqual(len(monitor._events),1)

    def test_analytical_mfu_known_tiny_dimensions(self):
        config={'monitoring':{'enabled':True,'every_n_steps':1,'peak_tflops_per_gpu':1},
            'model':{'dim':4,'ffn_dim':8,'layers':2,'text_dim':3,'in_channels':1,'out_channels':1,
                     'patch_size':[1,1,1],'text_trans':{'enabled':True,'layers':1}}}
        clock=iter([0.,2.,2.]);model=Model()
        monitor=TrainingMonitor(config,'cpu',model,True,lambda:next(clock))
        monitor.begin_update(1);model.set_stats(.5,.01,.1,0.)
        monitor.record_micro(2,2,torch.tensor([[1,1,1],[1,0,0]],dtype=torch.bool))
        payload=monitor.end_update(1)
        # Main linear incl mHC: 2*5*(1536+1536); text linear:
        # 2*3*(960+48); image: 2*2*40; attention:96*(25+9)+48*(9+1).
        flops=30720+6048+160+3264+480
        self.assertEqual(flops,40672)
        self.assertAlmostEqual(payload['performance/dit_tflops_per_gpu'],flops/2/1e12,places=15)
        self.assertAlmostEqual(payload['performance/mfu_dit_estimate'],flops/2/1e12,places=15)
        self.assertAlmostEqual(payload['performance/mfu_dit_estimate_percent'],100*flops/2/1e12,places=15)

    def test_mfu_absent_without_explicit_peak(self):
        _,payload=self.exercise()
        self.assertNotIn('performance/mfu_dit_estimate',payload)

    def test_sampled_clipping_scale_estimate(self):
        monitor=TrainingMonitor({'monitoring':{'enabled':True,'every_n_steps':1}},
                                'cpu',torch.nn.Linear(1,1),grad_clip_threshold=1.)
        monitor.begin_update(1)
        payload=monitor.end_update(1,grad_norm=4.)
        self.assertEqual(payload['train/grad_clip_active'],1)
        self.assertAlmostEqual(payload['train/grad_clip_scale_estimate'],.25,places=6)
        monitor.begin_update(2)
        payload=monitor.end_update(2,grad_norm=.5)
        self.assertEqual(payload['train/grad_clip_active'],0)
        self.assertEqual(payload['train/grad_clip_scale_estimate'],1)


if __name__=='__main__':unittest.main()
