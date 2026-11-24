"""mHC manifold, non-saturated transfer, learnability, and sampled diagnostics."""
from pathlib import Path
import os,sys,unittest
REAL_ROOT=Path(os.environ.get('GIGA_REPO_ROOT',Path(__file__).resolve().parents[1]))
sys.path[:0]=[str(REAL_ROOT),str(REAL_ROOT/'thirdparty')]
import models
if os.environ.get('GIGA_MHC_STAGE'):models.__path__.insert(0,str(Path(os.environ['GIGA_MHC_STAGE'])/'models'))
import torch
from models.mhc_residual_transfer import MHCResidual,MHC_STAT_KEYS,sinkhorn_log_domain,stack_stream,read_out

torch.set_num_threads(1)

class ResidualTests(unittest.TestCase):
 def test_manifold_and_bf16_arithmetic(self):
  for dtype in [torch.float32,torch.bfloat16]:
   m=MHCResidual(16,4).to(dtype=dtype)
   pre,post,res=m._mappings(torch.randn(2,7,4,16,dtype=dtype))
   self.assertEqual(res.dtype,torch.float32)
   self.assertEqual(pre.dtype,torch.float32)
   self.assertTrue((res>=0).all())
   self.assertLess(float((res.sum(-1)-1).abs().max()),1e-5)
   self.assertLess(float((res.sum(-2)-1).abs().max()),1e-5)
   self.assertTrue((pre>1e-3).all() and (pre<1-1e-3).all())
   self.assertTrue((post>.01).all() and (post<1.99).all())
 def test_transfer_mean_and_asymmetric_learnable_streams(self):
  torch.manual_seed(8)
  m=MHCResidual(16,4);x=torch.randn(2,5,16);fn=torch.nn.Linear(16,16,bias=False)
  h=stack_stream(x,4);out=m(h,fn)
  torch.testing.assert_close(out.mean(2),x+fn(x),rtol=2e-6,atol=2e-6)
  self.assertGreater(float((out[:,:,0]-out[:,:,1]).abs().max()),.01)
  loss=m(out,fn).mean(2).square().mean();loss.backward()
  s=4
  gradient=m.mapping.weight.grad
  self.assertTrue(torch.isfinite(gradient).all())
  self.assertGreater(float(gradient[:s].abs().sum()),1e-5)
  self.assertGreater(float(gradient[s:2*s].abs().sum()),1e-5)
  # Once pre routing has moved from uniform, residual routes receive gradients.
  downstream=MHCResidual(16,4)
  with torch.no_grad():downstream.mapping.bias[0].add_(.01)
  m.zero_grad();downstream(m(out.detach(),fn),fn).mean(2).square().mean().backward()
  self.assertGreater(float(m.mapping.weight.grad[2*s:].abs().sum()),1e-7)
 def test_diagnostics_opt_in_and_schema(self):
  m=MHCResidual(8,4);h=stack_stream(torch.randn(2,3,8),4)
  m(h,lambda x:x);self.assertIsNone(m.last_stats)
  m.collect_stats=True;out=m(h,lambda x:x)
  self.assertEqual(set(m.last_stats),set(MHC_STAT_KEYS))
  self.assertTrue(all(v.ndim==0 and not v.requires_grad for v in m.last_stats.values()))
  self.assertEqual(float(m.last_stats['nonfinite_count']),0)
 def test_invalid_configuration(self):
  for streams in (0,1,9):
   with self.assertRaises(ValueError):MHCResidual(8,streams)
  with self.assertRaises(ValueError):MHCResidual(8,4,sinkhorn_iters=0)
  with self.assertRaises(ValueError):MHCResidual(8,4,mapping_fp32=False)

if __name__=='__main__':unittest.main()
