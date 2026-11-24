"""DeepSeek-style mHC maps with log-Sinkhorn projection and transfer initialization.

H_next = H_res @ H + H_post.T * F(H_pre @ H).
H_pre=sigmoid, H_post=2*sigmoid, H_res is nonnegative/doubly stochastic.
The transfer initializer is deliberately distinct from the paper's from-scratch init.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

MHC_STAT_KEYS = (
    'pre_mean','pre_min','pre_max','pre_saturation_fraction',
    'post_mean','post_min','post_max','post_saturation_fraction',
    'res_diag_mean','res_offdiag_mean','res_min','res_entropy',
    'sinkhorn_row_error','sinkhorn_col_error',
    'stream_rms_mean','stream_rms_min','stream_rms_max','stream_rms_cv','nonfinite_count',
)


def sinkhorn_log_domain(logits, n_iters=20, eps=1e-6):
    if n_iters < 1:
        raise ValueError('sinkhorn_iters must be positive')
    work = logits.float()
    for _ in range(int(n_iters)):
        work = work - torch.logsumexp(work, -1, keepdim=True)
        work = work - torch.logsumexp(work, -2, keepdim=True)
    return work.exp()


class MHCResidual(nn.Module):
    def __init__(self, dim, n_streams=4, sinkhorn_iters=20, eps=1e-6,
                 mapping_fp32=True, init_mode='transfer', routing_backend='eager'):
        super().__init__()
        if not isinstance(n_streams, int) or not 2 <= n_streams <= 8:
            raise ValueError('mHC requires 2..8 streams')
        if sinkhorn_iters < 1 or eps <= 0:
            raise ValueError('sinkhorn_iters and eps must be positive')
        if init_mode not in ('transfer','paper'):
            raise ValueError('init_mode must be transfer or paper')
        self.dim, self.n_streams = int(dim), n_streams
        self.sinkhorn_iters, self.eps = int(sinkhorn_iters), float(eps)
        if not mapping_fp32:
            raise ValueError('mHC mapping_fp32 must be true for stable manifold arithmetic')
        self.mapping_fp32, self.init_mode = True, init_mode
        self.set_routing_backend(routing_backend)
        self.mapping = nn.Linear(n_streams*dim, 2*n_streams+n_streams*n_streams)
        self.dynamic_alpha = nn.Parameter(torch.ones(3))
        self.collect_stats = False
        self.last_stats = None
        self.reset_mhc_parameters()

    def set_routing_backend(self, backend):
        if backend not in ('eager', 'fused'):
            raise ValueError('mHC routing_backend must be eager or fused')
        if backend == 'fused' and self.n_streams != 4:
            raise ValueError('Fused mHC routing currently requires four streams')
        self.routing_backend = backend

    @torch.no_grad()
    def reset_mhc_parameters(self):
        s = self.n_streams
        if self.init_mode == 'paper':
            nn.init.normal_(self.mapping.weight, std=1/math.sqrt(s*self.dim))
            nn.init.zeros_(self.mapping.bias)
            self.dynamic_alpha.fill_(0.01)
            return
        # Uniform pre preserves the stream mean. Asymmetric post with mean one
        # breaks stream symmetry without changing the collapsed dense function.
        nn.init.zeros_(self.mapping.weight)
        self.dynamic_alpha.fill_(1.)
        pre = torch.full((s,), 1/s)
        post = torch.ones(s)
        for i in range(s//2):
            offset = .25/(2**i)
            post[2*i], post[2*i+1] = 1-offset, 1+offset
        self.mapping.bias[:s].copy_(torch.logit(pre))
        self.mapping.bias[s:2*s].copy_(torch.logit(post/2))
        # A positive, well-conditioned doubly stochastic residual map, without
        # saturated +/-12 logits. Mean preservation holds for any such map.
        residual = torch.eye(s)*2
        self.mapping.bias[2*s:].copy_(residual.flatten())

    @torch.no_grad()
    def init_paper_mode(self, scale=.02):
        self.init_mode = 'paper'
        self.reset_mhc_parameters()

    def _mappings(self, h):
        b,n,s,c = h.shape
        if self.routing_backend == 'fused':
            from models.mhc_routing_fused import mapping_logits
            from models.mhc_sinkhorn_triton import sinkhorn4_triton
            pre,post,logits = mapping_logits(h,self.mapping.weight,self.mapping.bias,
                                           self.dynamic_alpha,s,self.eps)
            return pre,post,sinkhorn4_triton(logits,self.sinkhorn_iters)
        vec = h.reshape(b,n,s*c)
        # All nonlinear map construction and Sinkhorn operations run in FP32.
        # Casts remain differentiable when model parameters are stored as BF16.
        vec = F.rms_norm(vec.float(), (s*c,), eps=self.eps)
        dynamic = F.linear(vec, self.mapping.weight.float(), None)
        pieces = dynamic.split((s,s,s*s),-1)
        dynamic = torch.cat([part*self.dynamic_alpha[i].float() for i,part in enumerate(pieces)],-1)
        logits = dynamic + self.mapping.bias.float()
        pre = logits[...,:s].sigmoid()
        post = 2*logits[...,s:2*s].sigmoid()
        residual = sinkhorn_log_domain(logits[...,2*s:].reshape(b,n,s,s),self.sinkhorn_iters,self.eps)
        return pre,post,residual

    def forward(self,h,block_fn):
        if h.ndim != 4 or h.shape[-2:] != (self.n_streams,self.dim):
            raise ValueError(f'Expected [B,N,{self.n_streams},{self.dim}], got {tuple(h.shape)}')
        pre,post,residual = self._mappings(h)
        if self.routing_backend == 'fused':
            from models.mhc_routing_fused import pre_mix,post_mix
            mixed = pre_mix(h,pre)
        else:
            mixed = torch.einsum('bns,bnsc->bnc',pre,h.float()).to(h.dtype)
        update = block_fn(mixed)
        if self.routing_backend == 'fused':
            out = post_mix(h,update,post,residual)
        else:
            out = (torch.einsum('bnij,bnjc->bnic',residual,h.float())
                   +torch.einsum('bns,bnc->bnsc',post,update.float())).to(h.dtype)
        if self.collect_stats:
            with torch.no_grad():
                diagonal = residual.diagonal(dim1=-2,dim2=-1)
                rms = out.float().square().mean(dim=(0,1,3)).sqrt()
                nonfinite = sum((~torch.isfinite(t)).sum() for t in (pre,post,residual,out))
                self.last_stats = {
                    'pre_mean':pre.mean(),'pre_min':pre.amin(),'pre_max':pre.amax(),
                    'pre_saturation_fraction':((pre<1e-3)|(pre>1-1e-3)).float().mean(),
                    'post_mean':post.mean(),'post_min':post.amin(),'post_max':post.amax(),
                    'post_saturation_fraction':((post<2e-3)|(post>2-2e-3)).float().mean(),
                    'res_diag_mean':diagonal.mean(),
                    'res_offdiag_mean':(residual.sum(-1)-diagonal).mean()/(self.n_streams-1),
                    'res_min':residual.amin(),
                    'res_entropy':-(residual*residual.clamp_min(self.eps).log()).sum(-1).mean(),
                    'sinkhorn_row_error':(residual.sum(-1)-1).abs().amax(),
                    'sinkhorn_col_error':(residual.sum(-2)-1).abs().amax(),
                    'stream_rms_mean':rms.mean(),'stream_rms_min':rms.amin(),'stream_rms_max':rms.amax(),
                    'stream_rms_cv':rms.std(unbiased=False)/rms.mean().clamp_min(self.eps),
                    'nonfinite_count':nonfinite.float(),
                }
                self.last_stats = {k:v.detach() for k,v in self.last_stats.items()}
        return out


def stack_stream(x,n_streams):
    if n_streams < 2:
        raise ValueError('mHC requires at least two streams')
    return x.unsqueeze(2).expand(*x.shape[:-1],n_streams,x.shape[-1]).clone()


def read_out(h,weight):
    return torch.einsum('bnsc,s->bnc',h.float(),weight.float()).to(h.dtype)
