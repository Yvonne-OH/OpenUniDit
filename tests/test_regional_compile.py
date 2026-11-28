"""Run CPU regressions, or --gpu for a bounded BF16/FA3 numerical check."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(os.environ.get('GIGA_REPO_ROOT', Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT/'thirdparty'), str(ROOT)]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


regional = load('regional_test_ops', os.environ.get('GIGA_REGIONAL_SOURCE', ROOT/'optimizations/regional_compile.py'))
model_module = load('models.regional_test_model', os.environ.get('GIGA_MODEL_SOURCE', ROOT/'models/giga_transformer.py'))


def make_model():
    return model_module.GigaExpTransformer(
        in_channels=4, out_channels=4, text_dim=32, dim=256, ffn_dim=768,
        heads=2, layers=2, rope_axes_dims=(16,48,48,16), adaln_type='basis',
        adaln_rank=8, text_trans=False, final_image_only=True,
        activation_checkpointing=True, activation_checkpointing_ratio=0.5)


def inputs(device, dtype, height, width, length):
    return (torch.randn(2,4,1,height,width,device=device,dtype=dtype),
            torch.tensor([120.,850.],device=device),
            torch.randn(2,length,32,device=device,dtype=dtype),
            torch.arange(length,device=device)[None,:] < torch.tensor([max(2,length//2),length],device=device)[:,None])


def compare(a, b, args, tolerance):
    a.zero_grad(set_to_none=True); b.zero_grad(set_to_none=True)
    y = a(*args); z = b(*args)
    y.float().square().mean().backward(); z.float().square().mean().backward()
    errors=[]
    for (name,p), (other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name == other
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            assert torch.isfinite(q.grad).all(), name
            error=((p.grad.float()-q.grad.float()).norm()/p.grad.float().norm().clamp_min(1e-8)).item()
            errors.append((error,name))
    out=((y.float()-z.float()).norm()/y.float().norm().clamp_min(1e-8)).item()
    assert out <= tolerance, ('output',out)
    assert max(errors)[0] <= tolerance, max(errors)
    return {'output_relative_l2':out,'worst_gradient':max(errors)}


class RegionalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch._dynamo.reset()
        torch.manual_seed(42)

    def test_model_gradients_state_dict_and_sealed_guard_misses(self):
        eager=make_model(); candidate=copy.deepcopy(eager); compiled=[]
        def backend(gm, example_inputs):
            compiled.append(gm)
            return gm.forward
        ops=regional.attach_regional_ops(candidate, backend)
        assert list(eager.state_dict()) == list(candidate.state_dict())
        args=inputs('cpu',torch.float32,4,4,9)
        compare(eager,candidate,args,1e-5)
        self.assertGreater(len(compiled),0)
        ops.seal(); before=len(compiled)
        for h,w,length in [(4,3,13),(3,4,17),(4,4,9)]:
            compare(eager,candidate,inputs('cpu',torch.float32,h,w,length),1e-5)
        self.assertEqual(len(compiled),before)
        self.assertGreater(sum(v for k,v in ops.metrics().items() if '/regional_calls/' in k),0)

    def test_plan_only_uses_configured_shapes_and_real_batches(self):
        cfg={'model':{'patch_size':[1,1,1]},'training':{'batch_size_per_device':4},
             'components':{'text_encoder':{'max_length':1280}},
             'dataset':{'buckets':['256_square','256x192','192x256','512_square','512x384','384x512'],
                        'bucket_batch_sizes':{k:8 for k in ['256_square','256x192','192x256','512_square','512x384','384x512']}}}
        plan=regional.warmup_plan(cfg)
        self.assertEqual({b for b,_,_ in plan},{8})
        self.assertEqual({n for _,n,_ in plan},{192,256,768,1024})
        self.assertEqual({n for _,_,n in plan},{2,128,512,1280})
        self.assertEqual(len(plan),16)

    def test_moe_is_rejected(self):
        from types import SimpleNamespace
        model=SimpleNamespace(has_moe=True, mod_adaln=True)
        with self.assertRaises(ValueError):regional.attach_regional_ops(model)

    def test_node_leader_warms_before_followers(self):
        from unittest.mock import patch
        for local_rank, expected in [('0', ['warm', 'barrier', 'barrier']),
                                     ('1', ['barrier', 'warm', 'barrier'])]:
            events=[]
            with patch.dict(os.environ, {'LOCAL_RANK':local_rank}), \
                 patch.object(torch.distributed, 'is_initialized', return_value=True), \
                 patch.object(torch.distributed, 'barrier', side_effect=lambda: events.append('barrier')), \
                 patch.object(regional, 'warmup_regional', side_effect=lambda *a: events.append('warm')):
                regional.warmup_regional_distributed(None,None,None,None)
            self.assertEqual(events,expected)


def gpu_check():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.1)
    assert model_module._FA3, 'FA3 must be available for the GPU path'
    torch.manual_seed(42)
    eager=make_model().to('cuda',dtype=torch.bfloat16)
    candidate=copy.deepcopy(eager)
    before={k:v.clone() for k,v in candidate.state_dict().items()}
    ops=regional.attach_regional_ops(candidate)
    cfg={'model':{'patch_size':[1,1,1]},'training':{'batch_size_per_device':2},
         'components':{'text_encoder':{'max_length':32}},
         'dataset':{'buckets':['64_square','64x48'],'bucket_batch_sizes':{'64_square':2,'64x48':2}}}
    import logging
    logging.basicConfig(level=logging.INFO)
    rng=torch.cuda.get_rng_state().clone()
    regional.warmup_regional(candidate,cfg,ops,logging.getLogger('regional-test'))
    assert torch.equal(rng,torch.cuda.get_rng_state()), 'warmup modified RNG'
    assert all(torch.equal(before[k],v) for k,v in candidate.state_dict().items())
    assert all(p.grad is None for p in candidate.parameters()), 'warmup touched live gradients'
    results=[]
    for h,w,length in [(4,4,9),(4,3,13),(3,4,31)]:
        results.append(compare(eager,candidate,inputs('cuda',torch.bfloat16,h,w,length),0.04))
    print(json.dumps({'gpu_numerics':results,'metrics':ops.metrics(),
                      'peak_gib':torch.cuda.max_memory_allocated()/2**30},indent=2))


if __name__=='__main__':
    if '--gpu' in sys.argv:gpu_check()
    else:unittest.main()
