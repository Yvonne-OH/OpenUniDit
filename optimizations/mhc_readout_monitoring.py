"""Readout diagnostics at sampled optimizer boundaries, including ZeRO masters."""
import math

import torch
import torch.distributed as dist
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad


@torch.no_grad()
def capture_readout_before_step(parameter):
    """Call on every rank after the final accumulated backward, before step()."""
    master = safe_get_full_fp32_param(parameter)
    gradient = safe_get_full_grad(parameter)
    # ZeRO may return views that step() updates or clears in place.
    return (
        None if master is None else master.detach().float().clone(),
        None if gradient is None else gradient.detach().float().clone(),
    )


@torch.no_grad()
def readout_update_metrics(parameter, before):
    """Gather on every rank after step(); only rank zero materializes metrics."""
    previous, gradient = before
    current = safe_get_full_fp32_param(parameter)
    if dist.is_initialized() and dist.get_rank() != 0:
        return {}
    payload = {
        'mhc/readout_fp32/available': int(current is not None),
        'mhc/readout_grad/available': int(gradient is not None),
        'mhc/readout_update/available': int(previous is not None and current is not None),
    }
    vectors = {'readout_model': parameter.detach().float().reshape(-1)}
    if current is not None:
        vectors['readout_fp32'] = current.detach().float().reshape(-1)
        vectors['readout_quantization_gap'] = vectors['readout_fp32'] - vectors['readout_model']
    if gradient is not None:
        vectors['readout_grad'] = gradient.reshape(-1)
    if previous is not None and current is not None:
        vectors['readout_update'] = vectors['readout_fp32'] - previous.reshape(-1)
    size = parameter.numel()
    values = torch.cat(list(vectors.values())).cpu().tolist()
    for index, name in enumerate(vectors):
        vector = values[index * size:(index + 1) * size]
        prefix = f'mhc/{name}'
        payload[f'{prefix}/nonfinite_count'] = sum(not math.isfinite(v) for v in vector)
        if payload[f'{prefix}/nonfinite_count']:
            continue
        payload.update({f'{prefix}/stream_{i}': v for i, v in enumerate(vector)})
        payload[f'{prefix}/min'] = min(vector)
        payload[f'{prefix}/max'] = max(vector)
        payload[f'{prefix}/mean'] = sum(vector) / size
        payload[f'{prefix}/sum'] = sum(vector)
        payload[f'{prefix}/l2'] = math.sqrt(sum(v * v for v in vector))
    return payload
