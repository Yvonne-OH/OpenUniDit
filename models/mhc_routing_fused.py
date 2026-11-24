"""Opt-in routing fusion; diffusion attention and FFN stay outside compilation."""
from functools import lru_cache
import torch
import torch.nn.functional as F


def _mapping_logits(h, weight, bias, alpha, n_streams, eps):
    b, n, s, c = h.shape
    vec = F.rms_norm(h.reshape(b, n, s*c).float(), (s*c,), eps=eps)
    dynamic = F.linear(vec, weight.float(), None)
    pieces = dynamic.split((n_streams, n_streams, n_streams*n_streams), -1)
    dynamic = torch.cat([part*alpha[i].float() for i, part in enumerate(pieces)], -1)
    logits = dynamic + bias.float()
    return (logits[..., :s].sigmoid(), 2*logits[..., s:2*s].sigmoid(),
            logits[..., 2*s:].reshape(b, n, s, s))


def _pre_mix(h, pre):
    return (pre.unsqueeze(-1)*h.float()).sum(-2).to(h.dtype)


def _post_mix(h, update, post, residual):
    return ((residual.unsqueeze(-1)*h.float().unsqueeze(-3)).sum(-2)
            + post.unsqueeze(-1)*update.float().unsqueeze(-2)).to(h.dtype)


@lru_cache(maxsize=1)
def compiled_routing():
    # Dynamic token counts prevent a compilation for each caption length.
    # B=1 and B>1 may still need separate first-use compilations.
    return tuple(torch.compile(fn, fullgraph=True, dynamic=True)
                 for fn in (_mapping_logits, _pre_mix, _post_mix))


def mapping_logits(*args):
    return compiled_routing()[0](*args)


def pre_mix(*args):
    return compiled_routing()[1](*args)


def post_mix(*args):
    return compiled_routing()[2](*args)
