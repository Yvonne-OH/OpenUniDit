"""Run with torchrun --standalone --nproc_per_node=2 for ZeRO coverage."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from optimizations import mhc_readout_monitoring as monitoring


class ReadoutMonitoringTests(unittest.TestCase):
    def test_snapshots_and_missing_gradient(self):
        parameter = torch.nn.Parameter(torch.full((4,), .25, dtype=torch.bfloat16))
        master = parameter.float().detach().clone()
        with patch.object(monitoring, 'safe_get_full_fp32_param', return_value=master), \
             patch.object(monitoring, 'safe_get_full_grad', return_value=None):
            before = monitoring.capture_readout_before_step(parameter)
            master.add_(3e-5)
            metrics = monitoring.readout_update_metrics(parameter, before)
        self.assertEqual(metrics['mhc/readout_grad/available'], 0)
        self.assertGreater(metrics['mhc/readout_update/l2'], 0)
        self.assertGreater(metrics['mhc/readout_quantization_gap/l2'], 0)
        self.assertEqual(metrics['mhc/readout_model/min'], .25)

    @unittest.skipUnless(torch.cuda.is_available() and 'LOCAL_RANK' in os.environ,
                         'requires CUDA torchrun')
    def test_zero_partitioned_master_and_accumulated_gradient(self):
        import deepspeed
        from scripts.train.train_giga_exp import build_optimizer
        from deepspeed.utils import safe_get_full_fp32_param
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.readout = torch.nn.Parameter(torch.full((4,), .25))

            def forward(self, direction):
                return (self.readout.float() * direction).sum()

        model = Model().to(device=f'cuda:{local_rank}', dtype=torch.bfloat16)
        optimizer = build_optimizer(model, {
            'learning_rate': 3e-5, 'muon': {'enabled': False},
            'adam': {'betas': [.9, .95], 'epsilon': 1e-8, 'weight_decay': .005},
        })
        config = {
            'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 2,
            'bf16': {'enabled': True}, 'zero_optimization': {'stage': 1},
            'zero_allow_untested_optimizer': True, 'steps_per_print': 10000,
        }
        engine, _, _, _ = deepspeed.initialize(
            model=model, model_parameters=model.parameters(), optimizer=optimizer, config=config)
        direction = torch.tensor([1., -1., 2., -2.], device=model.readout.device)
        samples = 0
        for micro in range(4):
            engine.backward(engine(direction))
            boundary = engine.is_gradient_accumulation_boundary()
            before = monitoring.capture_readout_before_step(model.readout) if boundary else None
            engine.step()
            if not boundary:
                continue
            metrics = monitoring.readout_update_metrics(model.readout, before)
            master = safe_get_full_fp32_param(model.readout).detach().clone()
            samples += 1
            if torch.distributed.get_rank() == 0:
                self.assertEqual(metrics['mhc/readout_grad/available'], 1)
                self.assertGreater(metrics['mhc/readout_grad/l2'], 0)
                self.assertGreater(metrics['mhc/readout_update/l2'], 0)
                self.assertEqual(metrics['mhc/readout_model/min'], .25)
                self.assertEqual(metrics['mhc/readout_model/max'], .25)
                for index, value in enumerate(master.tolist()):
                    self.assertEqual(metrics[f'mhc/readout_fp32/stream_{index}'], value)
                print('READOUT_ZERO_PASS', engine.global_steps, metrics, flush=True)
            else:
                self.assertEqual(metrics, {})
        self.assertEqual(samples, 2)
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
