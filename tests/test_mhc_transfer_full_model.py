"""Dense -> mHC migration, base-feature compatibility and full-model gradients."""
from pathlib import Path
import os,sys,unittest
REAL_ROOT=Path(os.environ.get('GIGA_REPO_ROOT',Path(__file__).resolve().parents[1]))
sys.path[:0]=[str(REAL_ROOT),str(REAL_ROOT/'thirdparty')]
import models
if os.environ.get('GIGA_MHC_STAGE'):models.__path__.insert(0,str(Path(os.environ['GIGA_MHC_STAGE'])/'models'))
import torch
from models.giga_transformer import GigaExpTransformer
from models.giga_transformer_mhc_transfer import GigaExpTransformerMHC,load_mhc_compatible_state_dict,is_mhc_parameter_name

torch.set_num_threads(1)
KW=dict(in_channels=4,out_channels=4,text_dim=12,dim=32,ffn_dim=64,heads=4,layers=2,
        rope_axes_dims=[2,2,2,2],text_trans={'enabled':True,'layers':1,'heads':4},
        adaln_type='basis',adaln_rank=8)

class FullModelTests(unittest.TestCase):
 def inputs(self,dtype=torch.float32):
  return (torch.randn(2,4,1,4,6,dtype=dtype),torch.tensor([0.,650.]),
          torch.randn(2,7,12,dtype=dtype),torch.tensor([[1]*7,[1]*4+[0]*3],dtype=torch.bool))
 def test_all_dense_keys_retained_and_output_transfer(self):
  for adaln in ['shared','dual','basis']:
   for streams in [2,4]:
    torch.manual_seed(42);kw={**KW,'adaln_type':adaln}
    base=GigaExpTransformer(**kw).eval();mhc=GigaExpTransformerMHC(**kw,n_streams=streams).eval()
    matched,missing,ignored=load_mhc_compatible_state_dict(mhc,base.state_dict(),base.semantic_metadata())
    self.assertEqual(matched,len(base.state_dict()));self.assertEqual(ignored,0)
    self.assertTrue(all(is_mhc_parameter_name(k) for k in set(mhc.state_dict())-set(base.state_dict())))
    args=self.inputs()
    with torch.no_grad():a=base(*args);b=mhc(*args)
    torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-5)
 def test_explicit_text_input_resize_and_strict_own_resume(self):
  base=GigaExpTransformer(**KW);model=GigaExpTransformerMHC(**{**KW,'text_dim':20})
  state=base.state_dict();matched,missing,ignored=load_mhc_compatible_state_dict(model,state,base.semantic_metadata())
  self.assertEqual(matched,len(state)-1);self.assertEqual(ignored,1)
  self.assertEqual(model.last_migration_report['reinitialized'],['text_in.weight'])
  bad=dict(state);del bad['patch_in.weight']
  with self.assertRaises(ValueError):load_mhc_compatible_state_dict(model,bad,base.semantic_metadata())
  bad=dict(state);bad['head_mod.1.weight']=bad['head_mod.1.weight'][:1]
  with self.assertRaises(ValueError):load_mhc_compatible_state_dict(model,bad,base.semantic_metadata())
  own=model.state_dict();metadata=model.semantic_metadata()
  load_mhc_compatible_state_dict(model,own,metadata)
  bad=dict(own);del bad['blocks.0.mhc_attn.mapping.weight']
  with self.assertRaises(ValueError):load_mhc_compatible_state_dict(model,bad,metadata)
  with self.assertRaises(ValueError):load_mhc_compatible_state_dict(model,own,None)
  incomplete={k:v for k,v in own.items() if not is_mhc_parameter_name(k)}
  with self.assertRaises(ValueError):load_mhc_compatible_state_dict(model,incomplete,metadata)
 def test_checkpointed_backward_and_stats(self):
  torch.manual_seed(42)
  model=GigaExpTransformerMHC(**{**KW,'activation_checkpointing':True}).train()
  model.set_mhc_collect_stats(True)
  args=self.inputs();loss=model(*args).square().mean();loss.backward()
  self.assertTrue(torch.isfinite(loss))
  for name,p in model.named_parameters():
   if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),name)
  self.assertGreater(float(model.blocks[1].mhc_attn.mapping.weight.grad.abs().sum()),1e-8)
  stats=model.mhc_stats();self.assertTrue(stats)
  self.assertIn('readout/sum',stats)
  model.set_mhc_collect_stats(False);self.assertEqual(model.mhc_stats(),{})
 def test_state_roundtrip(self):
  model=GigaExpTransformerMHC(**KW);other=GigaExpTransformerMHC(**KW)
  load_mhc_compatible_state_dict(other,model.state_dict(),model.semantic_metadata())
  args=self.inputs()
  with torch.no_grad():torch.testing.assert_close(model(*args),other(*args),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
