"""Fused FP32 4x4 log-Sinkhorn with its exact finite-iteration reverse pass.

This module is imported lazily by the opt-in fused mHC routing backend.
The reference backend and CPU imports do not require Triton.
"""
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _sinkhorn4_forward(LOGITS, OUTPUT, PROBS, N,
                       ITERS: tl.constexpr, SAVE: tl.constexpr,
                       BLOCK: tl.constexpr):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = tl.arange(0, 4), tl.arange(0, 4)
    offsets = token[:, None, None]*16 + row[None, :, None]*4 + col[None, None, :]
    mask = token[:, None, None] < N
    work = tl.load(LOGITS + offsets, mask=mask, other=0).to(tl.float32)
    for iteration in range(ITERS):
        maximum = tl.max(work, 2)
        exponential = tl.exp(work - maximum[:, :, None])
        work = work - maximum[:, :, None] - tl.log(tl.sum(exponential, 2))[:, :, None]
        if SAVE:
            saved = ((token[:, None, None]*ITERS + iteration)*2)*16 + row[None, :, None]*4 + col[None, None, :]
            tl.store(PROBS + saved, tl.exp(work), mask=mask)
        maximum = tl.max(work, 1)
        exponential = tl.exp(work - maximum[:, None, :])
        work = work - maximum[:, None, :] - tl.log(tl.sum(exponential, 1))[:, None, :]
        if SAVE:
            saved = ((token[:, None, None]*ITERS + iteration)*2 + 1)*16 + row[None, :, None]*4 + col[None, None, :]
            tl.store(PROBS + saved, tl.exp(work), mask=mask)
    tl.store(OUTPUT + offsets, tl.exp(work), mask=mask)


@triton.jit
def _sinkhorn4_backward(GRAD_OUT, OUTPUT, PROBS, GRAD_IN, N,
                        ITERS: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    row, col = tl.arange(0, 4), tl.arange(0, 4)
    offsets = token[:, None, None]*16 + row[None, :, None]*4 + col[None, None, :]
    mask = token[:, None, None] < N
    grad = tl.load(GRAD_OUT + offsets, mask=mask, other=0).to(tl.float32)
    grad = grad * tl.load(OUTPUT + offsets, mask=mask, other=0).to(tl.float32)
    for reverse in range(ITERS):
        iteration = ITERS - 1 - reverse
        saved = ((token[:, None, None]*ITERS + iteration)*2 + 1)*16 + row[None, :, None]*4 + col[None, None, :]
        probability = tl.load(PROBS + saved, mask=mask, other=0)
        grad = grad - probability*tl.sum(grad, 1)[:, None, :]
        probability = tl.load(PROBS + saved - 16, mask=mask, other=0)
        grad = grad - probability*tl.sum(grad, 2)[:, :, None]
    tl.store(GRAD_IN + offsets, grad, mask=mask)


class _Sinkhorn4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, iterations):
        value = logits.contiguous()
        n = value.numel()//16
        output = torch.empty_like(value)
        save = ctx.needs_input_grad[0]
        probabilities = torch.empty((n, iterations, 2, 4, 4), device=value.device, dtype=torch.float32) if save else output
        _sinkhorn4_forward[(triton.cdiv(n, 16),)](
            value, output, probabilities, n, iterations, save, 16,
            num_warps=4, enable_fp_fusion=False,
        )
        ctx.iterations = iterations
        if save:
            ctx.save_for_backward(output, probabilities)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        output, probabilities = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        gradient = torch.empty_like(output)
        n = output.numel()//16
        _sinkhorn4_backward[(triton.cdiv(n, 16),)](
            grad_output, output, probabilities, gradient, n, ctx.iterations, 16,
            num_warps=4, enable_fp_fusion=False,
        )
        return gradient, None


def sinkhorn4_triton(logits, n_iters=20):
    if not logits.is_cuda or logits.dtype != torch.float32 or logits.shape[-2:] != (4, 4):
        raise ValueError('Fused Sinkhorn requires CUDA FP32 [...,4,4] logits')
    if not isinstance(n_iters, int) or n_iters < 1:
        raise ValueError('n_iters must be a positive integer')
    return _Sinkhorn4.apply(logits, n_iters)
