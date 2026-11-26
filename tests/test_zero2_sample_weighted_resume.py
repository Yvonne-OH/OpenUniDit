"""Run with torchrun --standalone --nproc-per-node=2 on a CPU host (Gloo)."""
from pathlib import Path
import sys
import tempfile

import deepspeed
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train.train_giga_exp import (
    build_optimizer, load_training_checkpoint, sample_loss_scale, set_lr,
)


class Linear(torch.nn.Linear):
    def semantic_metadata(self):
        return {}


def make_engine(training, bf16):
    torch.manual_seed(12)
    model = Linear(4, 1, bias=False)
    optimizer = build_optimizer(model, training)
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "bf16": {"enabled": bf16},
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {"stage": 2, "overlap_comm": False},
    }
    engine, zero, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
    assert engine.zero_optimization_stage() == 2
    return engine, optimizer, zero


def batches(step, rank):
    # Different local sample counts across both microbatches and ranks.
    for micro, count in enumerate(((1, 2), (4, 1))[rank]):
        generator = torch.Generator().manual_seed(100 * step + 10 * rank + micro)
        yield torch.randn(count, 4, generator=generator), torch.randn(count, 1, generator=generator)


def update(state, step, training):
    engine, optimizer, zero = state
    samples = list(batches(step, dist.get_rank()))
    scale = sample_loss_scale(sum(len(x) for x, _ in samples), len(samples), engine.device)
    lr = set_lr(optimizer, zero, step, training)
    assert optimizer.adam.param_groups[0]["lr"] == lr
    for x, target in samples:
        per_image = (engine(x.to(engine.module.weight.dtype)).float() - target).square().flatten(1).mean(1)
        engine.backward(per_image.sum() * scale)
        engine.step()
    return lr


def main():
    # Native Gloo collectives suffice; the optional shared-memory extension needs a compiler.
    deepspeed.ops.__compatible_ops__["deepspeed_shm_comm"] = False
    deepspeed.init_distributed(dist_backend="gloo")
    assert dist.get_world_size() == 2
    training = {"learning_rate": 0.03, "lr_warmup_steps": 4, "total_steps": 10,
                "muon": {"enabled": False}, "adam": {"betas": [0.9, 0.95], "weight_decay": 0.005}}
    bf16 = "--bf16" in sys.argv
    continuous = make_engine(training, bf16)
    reference = Linear(4, 1, bias=False)
    reference.load_state_dict(continuous[0].module.state_dict())
    adam = torch.optim.AdamW(reference.parameters(), lr=0.03, betas=(0.9, 0.95), weight_decay=0.005)
    with tempfile.TemporaryDirectory() as local_tmp:
        shared = [local_tmp]
        dist.broadcast_object_list(shared, src=0)
        directory = Path(shared[0])
        for step in (1, 2, 3):
            lr = update(continuous, step, training)
            samples = [batch for rank in (0, 1) for batch in batches(step, rank)]
            x, target = (torch.cat(parts) for parts in zip(*samples))
            adam.param_groups[0]["lr"] = lr
            adam.zero_grad(set_to_none=True)
            (reference(x) - target).square().mean().backward()
            adam.step()
            # BF16 forward and gradient reductions round differently from the FP32 oracle.
            torch.testing.assert_close(continuous[0].module.weight.float(), reference.weight,
                                       rtol=0, atol=0.002 if bf16 else 1e-7)
            if step == 1:
                continuous[0].save_checkpoint(str(directory), tag="step-1", client_state={"step": 1})
                resumed = make_engine(training, bf16)
                _, client = load_training_checkpoint(resumed[0], directory, "step-1")
                assert client["step"] == 1
            else:
                update(resumed, step, training)
                torch.testing.assert_close(resumed[0].module.weight, continuous[0].module.weight, rtol=0, atol=0)
                left, right = (state[1].adam.state_dict()["state"] for state in (continuous, resumed))
                for key in left:
                    for name in ("step", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(left[key][name], right[key][name], rtol=0, atol=0)
        dist.barrier()
    if dist.get_rank() == 0:
        print("PASS ZeRO-2 mixed batches vs global mean; continued vs resumed weights and Adam moments", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
