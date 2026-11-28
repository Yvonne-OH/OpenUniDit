"""Opt-in compilation of dense DiT tensor regions; attention stays eager.

No modules/parameters are replaced. Warmup uses private Parameter copies, never
production parameters or the DeepSpeed engine, and preserves RNG state.
"""
from collections import Counter
from contextlib import nullcontext
import os
import re
import time

import torch
from torch.nn import functional as F

_EAGER_CALLS = Counter()


def _norm_mod(x, weight, scale, shift, eps):
    if not torch.compiler.is_compiling():
        _EAGER_CALLS['norm_mod'] += 1
    return F.rms_norm(x, (x.shape[-1],), weight, eps) * (1 + scale[:, None]) + shift[:, None]


def _ffn(x, up, down):
    if not torch.compiler.is_compiling():
        _EAGER_CALLS['ffn'] += 1
    return F.linear(F.silu(F.linear(x, up)), down)


def _residual(x, update, gate):
    if not torch.compiler.is_compiling():
        _EAGER_CALLS['residual'] += 1
    return x + update * gate[:, None]


class RegionalOps:
    def __init__(self, backend='inductor'):
        self.functions = {
            'norm_mod': torch.compile(_norm_mod, backend=backend, fullgraph=True),
            'ffn': torch.compile(_ffn, backend=backend, fullgraph=True),
            'residual': torch.compile(_residual, backend=backend, fullgraph=True),
        }
        self.sealed = False
        self.calls = Counter()
        self.eager_baseline = Counter()

    def _call(self, name, *args):
        self.calls[name] += 1
        # This policy catches guard misses, not runtime/compile exceptions.
        # Never suppress OOM, kernel errors, or numerical failures.
        stance = torch.compiler.set_stance('eager_on_recompile') if self.sealed else nullcontext()
        with stance:
            return self.functions[name](*args)

    def norm_mod(self, x, norm, scale, shift):
        # Normalize slice strides outside the graph; otherwise each image/text
        # combination can specialize the batch stride and exhaust the cache.
        x = x.contiguous()
        torch._dynamo.mark_dynamic(x, 1)
        return self._call('norm_mod', x, norm.weight, scale, shift, norm.eps)

    def ffn(self, x, ff):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        torch._dynamo.mark_dynamic(flat, 0)
        return self._call('ffn', flat, ff[0].weight, ff[2].weight).reshape(shape)

    def residual(self, x, update, gate):
        x, update = x.contiguous(), update.contiguous()
        torch._dynamo.mark_dynamic(x, 1)
        torch._dynamo.mark_dynamic(update, 1)
        return self._call('residual', x, update, gate)

    def seal(self):
        self.sealed = True
        self.calls.clear()
        self.eager_baseline = _EAGER_CALLS.copy()

    def metrics(self):
        return {**{f'compile/regional_calls/{k}': v for k, v in self.calls.items()},
                **{f'compile/regional_eager_calls/{k}': _EAGER_CALLS[k] - self.eager_baseline[k]
                   for k in self.functions}}


def attach_regional_ops(model, backend='inductor'):
    if getattr(model, 'has_moe', False) or not getattr(model, 'mod_adaln', False):
        raise ValueError('regional_compile currently supports dense dual/basis DiT only')
    ops = RegionalOps(backend)
    for block in model.blocks:
        block._regional_ops = ops  # plain Python object: state_dict stays unchanged
    model._regional_ops = ops
    return ops


def warmup_plan(cfg):
    """Use only configured Wan image buckets, actual batch sizes and text limit."""
    dataset = cfg['dataset']
    patch = cfg['model'].get('patch_size', (1, 1, 1))
    if tuple(patch) != (1, 1, 1):
        raise ValueError('regional warmup currently requires Wan image patch_size=[1,1,1]')
    pairs = set()
    for bucket in dataset['buckets']:
        name = bucket.split(':')[-1]
        match = re.fullmatch(r'(\d+)(?:x(\d+)|_square)', name)
        if match is None:
            raise ValueError(f'Unsupported regional warmup image bucket: {bucket}')
        width, height = int(match[1]), int(match[2] or match[1])
        if width % 16 or height % 16:
            raise ValueError(f'Wan image bucket must be divisible by 16: {bucket}')
        sizes = dataset.get('bucket_batch_sizes', {})
        batch = int(sizes.get(bucket, sizes.get(name, cfg['training']['batch_size_per_device'])))
        pairs.add((batch, (width // 16) * (height // 16)))
    maximum = int(cfg['components']['text_encoder'].get('max_length', 1280))
    lengths = sorted({max(2, min(n, maximum)) for n in (2, 128, 512, maximum)})
    return [(b, image, text) for b, image in sorted(pairs) for text in lengths]


def warmup_regional(model, cfg, ops, logger):
    block = model.blocks[0]
    device, dtype = block.ff[0].weight.device, block.ff[0].weight.dtype
    if device.type != 'cuda':
        raise ValueError('Production regional warmup requires CUDA')
    plan = warmup_plan(cfg)
    start = time.perf_counter()
    # Copies avoid DeepSpeed's gradient hooks and all training-state mutations.
    up = torch.nn.Parameter(block.ff[0].weight.detach().clone())
    down = torch.nn.Parameter(block.ff[2].weight.detach().clone())
    weight = torch.nn.Parameter(block.norm1.weight.detach().clone())
    from types import SimpleNamespace
    norm = SimpleNamespace(weight=weight, eps=block.norm1.eps)
    ff = [SimpleNamespace(weight=up), None, SimpleNamespace(weight=down)]
    dim = up.shape[1]
    with torch.random.fork_rng(devices=[device.index]):
        for index, (batch, image, text) in enumerate(plan):
            logger.info('regional compile warmup %d/%d B=%d image=%d text=%d',
                        index + 1, len(plan), batch, image, text)
            x = torch.randn(batch, image + text, dim, device=device, dtype=dtype, requires_grad=True)
            mods = torch.randn(batch, dim * 6, device=device, dtype=dtype, requires_grad=True)
            shift, scale, gate, _, _, _ = mods.chunk(6, -1)
            img, txt = x[:, :image], x[:, image:]
            y = torch.cat((ops.norm_mod(img, norm, scale, shift),
                           ops.norm_mod(txt, norm, scale, shift)), 1)
            f = ops.ffn(y, ff)
            a = ops.residual(img, f[:, :image], gate)
            b = ops.residual(txt, f[:, image:], gate)
            # Also cover contiguous image-only output of the final block.
            final = ops.ffn(ops.norm_mod(a, norm, scale, shift), ff)
            out = ops.residual(a, final, gate)
            (out.float().square().mean() + b.float().square().mean()).backward()
            up.grad = down.grad = weight.grad = None
            del x, mods, shift, scale, gate, img, txt, y, f, a, b, final, out
    torch.cuda.synchronize(device)
    ops.seal()
    logger.info('regional compile warmup complete: %.1fs; guard misses use eager; regions=%s',
                time.perf_counter() - start, ','.join(ops.functions))


def warmup_regional_distributed(model, cfg, ops, logger):
    """Populate node-local disk caches before the remaining local ranks warm up.

    Every process must still warm its own Dynamo/AOT state. Only disk artifacts
    are shared. All ranks enter the same two barriers; warmup itself has no
    distributed calls and operates solely on private parameter copies.
    """
    dist = torch.distributed
    if not dist.is_available() or not dist.is_initialized():
        return warmup_regional(model, cfg, ops, logger)
    leader = int(os.environ.get('LOCAL_RANK', '0')) == 0
    if leader:
        warmup_regional(model, cfg, ops, logger)
    dist.barrier()
    if not leader:
        warmup_regional(model, cfg, ops, logger)
    dist.barrier()
