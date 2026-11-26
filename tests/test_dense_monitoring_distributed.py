"""Run with torchrun --standalone --nproc_per_node=2 (CPU/Gloo)."""
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'thirdparty')]

import torch
import torch.distributed as dist
from models.giga_transformer import GigaExpTransformer
from optimizations.mhc_monitoring import TrainingMonitor


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    torch.manual_seed(42 + rank)
    spec = dict(dim=32, ffn_dim=64, layers=2, heads=2, text_dim=16,
                in_channels=4, out_channels=4, rope_axes_dims=[4, 4, 4, 4],
                adaln_type='basis', adaln_rank=8,
                text_trans={'enabled': True, 'layers': 1, 'heads': 2},
                activation_checkpointing=True)
    model = GigaExpTransformer(**spec).train()
    cfg = {'model': spec, 'monitoring': {'enabled': True, 'every_n_steps': 2}}
    clock = iter([0., 1. + rank, 1. + rank])
    monitor = TrainingMonitor(cfg, 'cpu', model, False, lambda: next(clock))
    expected_samples = expected_text = 0
    for step in (1, 2):
        monitor.begin_update(step)
        model.zero_grad(set_to_none=True)
        for micro in range(2):
            batch = rank + 1
            mask = torch.zeros(batch, 5, dtype=torch.bool)
            mask[:, :3 + rank] = True
            output = model(torch.randn(batch, 4, 1, 2, 2), torch.full((batch,), 400.),
                           torch.randn(batch, 5, 16), mask)
            output.square().mean().backward()
            monitor.record_micro(batch, 4, mask)
            expected_samples += batch
            expected_text += batch * (3 + rank)
        payload = monitor.end_update(step, grad_norm=rank + 1., loss=.3,
                                     data_requests=rank + 1, data_read_seconds=.1 * (rank + 1))
        if step == 1:
            assert payload == {}
    expected = torch.tensor([expected_samples, expected_text], dtype=torch.float64)
    dist.all_reduce(expected)
    assert payload['performance/samples_per_sec'] == expected[0] / 2
    assert payload['performance/text_tokens_per_sec'] == expected[1] / 2
    assert payload['performance/micro_batches_per_update'] == 2
    assert payload['train/grad_norm'] == 2
    assert payload['health/nonfinite_loss_updates_window'] == 0
    assert payload['health/grad_norm_missing'] == 0
    assert payload['data/requests_global_sum'] == 6
    assert math.isclose(payload['data/read_wait_seconds_global_sum'], .6)
    assert all(math.isfinite(value) for value in payload.values())
    assert not any(key.startswith('mhc/') for key in payload)
    gathered = [None, None]
    dist.all_gather_object(gathered, payload)
    assert gathered[0] == gathered[1]
    if rank == 0:
        print('PASS dense 2-rank monitoring: actual checkpointed DiT forward/backward, '
              'unequal rank batch sizes, global throughput/data timings/grad norm; '
              f'{len(payload)} metrics', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
