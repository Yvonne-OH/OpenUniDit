"""ZeRO-2 与 Muon/AdamW 的两卡更新和断点恢复回归测试。"""

import json
import os
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
from torch import nn

from scripts.train.train_giga_exp import build_optimizer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.extra = nn.Linear(8, 8)
        self.bias = nn.Parameter(torch.zeros(8))

    def forward(self, x):
        return (self.extra(self.linear(x)) + self.bias).square().mean()


def make_engine(config):
    model = TinyModel().cuda().bfloat16()
    optimizer = build_optimizer(model, {"learning_rate": 1e-3, "muon": {"enabled": True}})
    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), optimizer=optimizer, config=config
    )
    return engine


def test_partitioned_muon_cpu():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    torch.manual_seed(31)
    matrix = [nn.Parameter(torch.randn(5, 7)), nn.Parameter(torch.randn(3, 4))]
    reference = [nn.Parameter(p.detach().clone()) for p in matrix]
    optimizer = build_optimizer(nn.ParameterList(matrix), {"learning_rate": 1e-3, "muon": {"enabled": True}})
    original = torch.optim.Muon(reference, lr=1e-3, weight_decay=0.0, adjust_lr_fn="match_rms_adamw")
    size = sum(p.numel() for p in matrix)
    shard_size = (size + world - 1) // world
    shard = nn.Parameter(torch.cat([p.detach().flatten() for p in matrix]).new_zeros(shard_size * world)[rank * shard_size:(rank + 1) * shard_size].clone())
    full = torch.cat([p.detach().flatten() for p in matrix])
    shard.data[:min(shard_size, size - rank * shard_size)].copy_(full[rank * shard_size:(rank + 1) * shard_size])
    optimizer.muon.param_groups[0]["params"] = [shard]
    for _ in range(3):
        gradients = [torch.randn_like(p) for p in matrix]
        for p, grad in zip(reference, gradients):
            p.grad = grad.clone()
        grad_flat = torch.cat([grad.flatten() for grad in gradients])
        shard.grad = torch.zeros_like(shard)
        count = min(shard_size, size - rank * shard_size)
        shard.grad[:count] = grad_flat[rank * shard_size:rank * shard_size + count]
        original.step()
        optimizer.step()
        gathered = [torch.empty_like(shard) for _ in range(world)]
        dist.all_gather(gathered, shard.detach())
        actual = torch.cat(gathered)[:size]
        expected = torch.cat([p.detach().flatten() for p in reference])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    saved = optimizer.state_dict()
    restored = build_optimizer(nn.ParameterList([nn.Parameter(torch.zeros_like(p)) for p in matrix]), {"learning_rate": 1e-3, "muon": {"enabled": True}})
    restored.muon.param_groups[0]["params"] = [nn.Parameter(shard.detach().clone())]
    restored.load_state_dict(saved)
    torch.testing.assert_close(
        restored.muon.state[restored.muon.param_groups[0]["params"][0]]["momentum_buffer"],
        optimizer.muon.state[shard]["momentum_buffer"], rtol=0, atol=0
    )
    if rank == 0:
        print("Partitioned Muon matches full-matrix Muon for 3 steps", flush=True)
    dist.destroy_process_group()


def main():
    if os.environ.get("GIGA_CPU_TEST") == "1":
        test_partitioned_muon_cpu()
        return
    deepspeed.init_distributed(dist_backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(17)
    config = json.loads((Path(__file__).resolve().parents[2] / "configs/zero2.json").read_text())
    config.update(train_batch_size=dist.get_world_size() * 2, train_micro_batch_size_per_gpu=2, gradient_accumulation_steps=1)
    engine = make_engine(config)
    path = Path(os.environ["GIGA_TEST_CKPT_DIR"])
    for step in range(2):
        x = torch.arange(16, device="cuda", dtype=torch.bfloat16).view(2, 8) / 16
        engine.backward(engine(x))
        engine.step()
        if step == 0:
            before = {k: v.detach().clone() for k, v in engine.module.state_dict().items()}
            engine.save_checkpoint(str(path), tag="smoke", client_state={"step": 1})
    reference = {k: v.detach().clone() for k, v in engine.module.state_dict().items()}
    dist.barrier()
    del engine
    torch.manual_seed(17)
    engine = make_engine(config)
    loaded, state = engine.load_checkpoint(str(path), tag="smoke")
    assert loaded and state["step"] == 1
    for k, value in before.items():
        torch.testing.assert_close(engine.module.state_dict()[k], value, rtol=0, atol=0)
    x = torch.arange(16, device="cuda", dtype=torch.bfloat16).view(2, 8) / 16
    engine.backward(engine(x))
    engine.step()
    for k, value in reference.items():
        torch.testing.assert_close(engine.module.state_dict()[k], value, rtol=0, atol=0)
    if dist.get_rank() == 0:
        print("ZeRO-2 Muon+AdamW two-rank step and checkpoint resume OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
