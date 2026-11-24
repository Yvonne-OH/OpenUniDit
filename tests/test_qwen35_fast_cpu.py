"""CPU integration tests for dispatch, final features, and checkpoint bootstrap."""
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT=Path('/home/jovyan/pfs/users/zhanqian.wu/code/model_exp')
STAGE=Path(os.environ.get('GIGA_FAST_SOURCE_ROOT','/tmp/qwen35_throughput_optimization'))
sys.path[:0]=[str(STAGE),str(ROOT),str(ROOT/'thirdparty')]
import torch
from safetensors.torch import save_file
from optimizations.qwen35_fast import (compatible_dispatch,choose_weight_source,final_features,
    install_resume_bootstrap,kernel_mode,resolve_kernels)

torch.set_num_threads(1)


class FastTests(unittest.TestCase):
    def test_dispatch_preserves_math_arguments_and_filters_model_kwargs(self):
        calls=[]
        def fast(q,k,v,g=None,beta=None,use_qk_l2norm_in_kernel=False):
            calls.append((q,k,v,g,beta,use_qk_l2norm_in_kernel));return q,None
        wrapped=compatible_dispatch(fast)
        self.assertEqual(wrapped(1,2,3,g=4,beta=5,use_qk_l2norm_in_kernel=True,output_attentions=True),(1,None))
        self.assertEqual(calls,[(1,2,3,4,5,True)])

    def test_final_features_and_mode_restore(self):
        from transformers import Qwen3_5TextConfig,Qwen3_5TextModel
        raw=json.loads(Path('/mnt/nfs/models/huggingface/Qwen/Qwen3.5-4B/config.json').read_text())['text_config']
        raw.update(hidden_size=64,intermediate_size=128,vocab_size=128,num_hidden_layers=4,
            num_attention_heads=2,num_key_value_heads=1,head_dim=32,linear_num_key_heads=2,
            linear_num_value_heads=4,linear_key_head_dim=16,linear_value_head_dim=16,
            layer_types=['linear_attention']*3+['full_attention'],bos_token_id=1,eos_token_id=2,pad_token_id=0)
        raw['rope_parameters']['mrope_section']=[1,1,2]
        model=Qwen3_5TextModel(Qwen3_5TextConfig(**raw)).eval().requires_grad_(False)
        inputs={'input_ids':torch.randint(1,128,(2,20)),
                'attention_mask':torch.tensor([[1]*20,[1]*11+[0]*9])}
        modeling,*_=resolve_kernels();before=modeling.torch_chunk_gated_delta_rule
        with torch.no_grad(),kernel_mode('reference'):
            with patch.object(model,'forward',wraps=model.forward) as forward:
                actual=final_features(model,inputs,-1)
                self.assertIs(forward.call_args.kwargs['output_hidden_states'],False)
            reference=model(**inputs,use_cache=False,return_dict=True,output_hidden_states=True)
            torch.testing.assert_close(actual,reference.hidden_states[-1],rtol=0,atol=0)
            torch.testing.assert_close(final_features(model,inputs,-2),reference.hidden_states[-2],rtol=0,atol=0)
        self.assertIs(modeling.torch_chunk_gated_delta_rule,before)

    def test_bootstrap_preserves_progress_then_prefers_own_checkpoint(self):
        spec=importlib.util.spec_from_file_location('runtime_test_helpers',ROOT/'tests/test_stage1_training_runtime.py')
        helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
        ns=helpers.helper_namespace()
        def load_compatible(model,state,metadata):
            result=model.load_state_dict(state,strict=True);return len(state),len(result.missing_keys),len(result.unexpected_keys)
        ns['load_compatible_state_dict']=load_compatible
        model=torch.nn.Linear(1,1,bias=False)
        model.semantic_metadata=lambda:{}
        engine=SimpleNamespace(module=model,device=torch.device('cpu'),optimizer=SimpleNamespace(refresh_fp32_params=Mock()))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir();output=root/'output'
            cursor={'epoch':1,'batch_offset':23,'sampler_signature':'example'}
            def save(folder,step,value,samples):
                save_file({'weight':torch.tensor([[value]])},str(folder/f'step-{step:08d}.safetensors'),
                    metadata={'step':str(step),'samples_seen':str(samples),'data_cursor':json.dumps(cursor)})
            save(source,10000,2.,100)
            training=SimpleNamespace(load_pure_weights=ns['load_pure_weights'])
            install_resume_bootstrap(training,source)
            self.assertEqual(training.load_pure_weights(engine,output),10000)
            self.assertEqual(engine.giga_samples_seen,100)
            self.assertEqual(engine.giga_data_cursor,cursor)
            self.assertEqual(model.weight.item(),2.)
            own=output/'checkpoints/weights';own.mkdir(parents=True)
            save(own,15000,3.,200)
            (own/'step-00016000.safetensors').write_bytes(b'bad snapshot')
            self.assertEqual(choose_weight_source(output,source),own)
            self.assertEqual(training.load_pure_weights(engine,output),15000)
            self.assertEqual(engine.giga_samples_seen,200)
            self.assertEqual(model.weight.item(),3.)

    def test_timer_is_disabled(self):
        from deepspeed.utils.config import get_timers_config
        from deepspeed.utils.timer import ThroughputTimer
        config=json.loads((STAGE/'configs/zero1_qwen35_fast.json').read_text())
        with patch('deepspeed.utils.timer.get_accelerator') as accelerator:
            timer=ThroughputTimer(get_timers_config(config),batch_size=16,start_step=0)
            for _ in range(4):timer.start();timer.stop(global_step=True)
            accelerator.assert_not_called()


if __name__=='__main__':unittest.main()
