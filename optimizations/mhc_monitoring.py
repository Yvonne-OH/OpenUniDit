"""Sampled mHC diagnostics and distributed training throughput.

No CPU reads or collective operations occur in ``record_micro``. At reporting
boundaries three fixed-shape all-reduces are followed by one host transfer. CUDA
phase events are queried later; this module never synchronizes an accelerator.
Stage event durations are sampled GPU-stream intervals, not exclusive kernel time.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import time

import torch
import torch.distributed as dist


PHASES = ('vae', 'text_encode', 'dit_forward', 'backward', 'optimizer')
READOUT_KEYS = ('mean', 'min', 'max', 'sum', 'rms')


def dit_flop_coefficients(config, n_streams):
    """Analytical trainable-DiT matmul FLOPs, excluding encoder/VAE/Sinkhorn.

    Linear layers use forward+backward (6 per weight), except frozen-input patch
    and text projections (4: no gradient to input). Main/refiner attention uses
    12*d*L*S^2. mHC includes its per-token mapping projections only. Scalar
    activations, time/AdaLN projections, stream einsums and normalization are
    omitted, so this is an explicit approximation, not profiler FLOPs/HFU.
    """
    spec = config['model']
    dim, ffn, depth = int(spec['dim']), int(spec['ffn_dim']), int(spec['layers'])
    refiner = spec.get('text_trans', {})
    ref_depth = int(refiner.get('layers', 2)) if isinstance(refiner, dict) and refiner.get('enabled', False) else 0
    patch = math.prod(spec.get('patch_size', [1, 1, 1]))
    mapping = 6 * 2 * depth * n_streams * dim * (2*n_streams+n_streams*n_streams)
    return {
        'main_linear': 6*depth*(4*dim*dim+2*dim*ffn) + mapping,
        'main_attention': 12*depth*dim,
        'text_linear': 6*ref_depth*(4*dim*dim+3*dim*ffn) + 4*int(spec['text_dim'])*dim,
        'text_attention': 12*ref_depth*dim,
        'image_linear': patch*dim*(4*int(spec['in_channels'])+6*int(spec['out_channels'])),
    }


def _reducer(key):
    if key.endswith('_count'):
        return 'sum'
    if key.endswith('_min') or key == 'min':
        return 'min'
    if key.endswith(('_max', '_error', '_cv')) or key == 'max':
        return 'max'
    return 'mean'


class TrainingMonitor:
    """Opt-in monitor for a model exposing the mHC diagnostics contract.

    ``config.monitoring.enabled`` and ``mhc_enabled`` must both be true.
    ``every_n_steps`` defaults to 50, ``cuda_phase_timing`` to false.
    ``per_layer`` may be a list of selected layer indices or true (all layers).

    Call begin_update before consuming micro-batches; record_micro after every
    successful backward/engine.step; end_update after the ordinary loss/sigma
    host reads. Pass skipped_batches as the delta for this update, not cumulative.
    Every rank must call end_update, even though only rank zero publishes its
    returned payload. Means are weighted by processed tokens across sites,
    micro-batches and ranks; errors/ranges use global extrema.

    Data timings are CPU floats for this update, including successful/skipped
    requests. data_read_* times only stream.next_batch (I/O, decode, queue wait);
    data_wait_* additionally includes rank coordination and worker recovery.
    They are accumulated on CPU and transferred only at reporting boundaries.
    """

    def __init__(self, config, device, model, mhc_enabled=False, clock=None):
        options = config.get('monitoring', {})
        self.enabled = bool(mhc_enabled and options.get('enabled', False))
        self.model = model
        self.device = torch.device(device)
        self.clock = clock or time.perf_counter
        self.interval = max(1, int(options.get('every_n_steps', 50)))
        self.should_report = False
        self._collecting = False
        self._window_start = None
        self._events = []
        self._cuda_timing = bool(options.get('phase_timings', options.get('cuda_phase_timing', False))) and self.device.type == 'cuda'
        if not self.enabled:
            return
        if not callable(getattr(model, 'mhc_stats', None)) or not callable(getattr(model, 'set_mhc_collect_stats', None)):
            raise ValueError('mHC monitoring requires mhc_stats() and set_mhc_collect_stats()')
        keys = tuple(model.mhc_stat_keys)
        sites = [name for name, module in model.named_modules()
                 if name and hasattr(module, 'collect_stats') and hasattr(module, 'last_stats')]
        if not sites:
            raise ValueError('mHC monitoring found no residual sites')
        self._schema = [(site + '/' + key, key, site) for site in sites for key in keys]
        self._schema += [('readout/' + key, key, 'readout') for key in READOUT_KEYS]
        self._size = len(self._schema)
        self._peak_tflops = float(options.get('peak_tflops_per_gpu') or 0.)
        if not math.isfinite(self._peak_tflops) or self._peak_tflops < 0:
            raise ValueError('peak_tflops_per_gpu must be a finite positive dense-BF16 peak')
        self._flop_coefficients = dit_flop_coefficients(config, int(getattr(model, 'n_streams', 4))) if self._peak_tflops else None
        self._zero = torch.zeros((), device=self.device, dtype=torch.float64)
        self._zero_stat = torch.zeros((), device=self.device, dtype=torch.float32)
        self._one = torch.ones((), device=self.device, dtype=torch.float64)
        self._sum_mask = torch.tensor([_reducer(key) == 'sum' for _, key, _ in self._schema], device=self.device)
        self._groups = {'all': [], 'attn': [], 'ffn': [], 'readout': []}
        selected = options.get('per_layer', False)
        for i, (_, key, site) in enumerate(self._schema):
            if site == 'readout':
                self._groups['readout'].append(i)
                continue
            self._groups['all'].append(i)
            if site.endswith('mhc_attn'):
                self._groups['attn'].append(i)
            elif site.endswith('mhc_ffn'):
                self._groups['ffn'].append(i)
            layer = next((int(part) for part in site.split('.') if part.isdigit()), None)
            if selected is True or (isinstance(selected, (list, tuple)) and layer in selected):
                self._groups.setdefault(site.replace('.', '/'), []).append(i)
        self._reset_window()

    def _reset_window(self):
        self._totals = torch.zeros(self._size, device=self.device, dtype=torch.float64)
        self._counts = torch.zeros_like(self._totals)
        self._mins = torch.full_like(self._totals, math.inf)
        self._maxs = torch.full_like(self._totals, -math.inf)
        self._invalid_metrics = self._zero.clone()
        self._sample_text_lengths = []
        self._sample_image_lengths = []
        self._linear_flops = 0
        self._samples = self._image_tokens = self._padded_text_tokens = self._micros = 0
        self._updates = self._skipped = 0
        self._data_wait = self._data_read = 0.
        self._data_wait_max = self._data_read_max = 0.
        self._data_requests = 0

    def begin_update(self, step):
        if not self.enabled:
            return False
        if self._window_start is None:
            self._window_start = self.clock()
        self.should_report = int(step) % self.interval == 0
        if self._collecting != self.should_report:
            self.model.set_mhc_collect_stats(self.should_report)
            self._collecting = self.should_report
        return self.should_report

    @contextmanager
    def phase(self, name):
        if not self.enabled or not self.should_report or not self._cuda_timing:
            yield
            return
        if name not in PHASES:
            raise ValueError(f'Unknown sampled phase: {name}')
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            stop.record()
            self._events.append((PHASES.index(name), start, stop))

    def record_micro(self, batch_size, image_tokens_per_sample, text_mask, padded_text_length=None):
        if not self.enabled:
            return
        batch_size = int(batch_size)
        padded_text_length = int(text_mask.shape[-1] if padded_text_length is None else padded_text_length)
        self._samples += batch_size
        self._image_tokens += batch_size * int(image_tokens_per_sample)
        self._padded_text_tokens += batch_size * padded_text_length
        self._micros += 1
        # Only one small reduction per micro-batch. Retain per-sample int32
        # lengths, not text masks/features, and batch the token/FLOPs arithmetic
        # at the reporting boundary instead of launching scalar ops each micro.
        self._sample_text_lengths.append(text_mask.detach().sum(dim=-1, dtype=torch.int32).to(self.device))
        if self._flop_coefficients is not None:
            c = self._flop_coefficients
            image = int(image_tokens_per_sample)
            self._linear_flops += batch_size*((image+padded_text_length)*c['main_linear']
                + padded_text_length*c['text_linear']+image*c['image_linear'])
            self._sample_image_lengths.extend([image]*batch_size)
        if not self.should_report:
            return
        stats = self.model.mhc_stats()
        present = [name in stats for name, _, _ in self._schema]
        # Stack FP32 diagnostics first, then cast the entire vector once. Casting
        # each site's scalar separately would launch thousands of tiny CUDA ops.
        values = torch.stack([stats[name].detach().to(device=self.device)
                              if found else self._zero_stat for (name, _, _), found in zip(self._schema, present)]).to(torch.float64)
        presence = torch.tensor(present, device=self.device, dtype=torch.bool)
        finite = torch.isfinite(values)
        valid = presence & finite
        self._invalid_metrics += (presence & ~finite).sum()
        weight = torch.where(self._sum_mask, self._one,
                             self._one * (batch_size * (int(image_tokens_per_sample) + padded_text_length)))
        self._totals += torch.where(valid, values * weight, self._zero)
        self._counts += valid * weight
        self._mins = torch.minimum(self._mins, torch.where(valid, values, math.inf))
        self._maxs = torch.maximum(self._maxs, torch.where(valid, values, -math.inf))

    def _take_phase_events(self):
        sums = [0.] * len(PHASES)
        counts = [0.] * len(PHASES)
        pending = []
        for phase, start, stop in self._events:
            if not stop.query():
                pending.append((phase, start, stop))
                continue
            sums[phase] += start.elapsed_time(stop)
            counts[phase] += 1
        self._events = pending
        return sums, counts

    def _window_token_totals(self):
        if not self._sample_text_lengths:
            return self._zero, self._zero
        lengths = torch.cat(self._sample_text_lengths).to(torch.float64)
        tokens = lengths.sum()
        if self._flop_coefficients is None:
            return tokens, self._zero
        images = torch.tensor(self._sample_image_lengths,device=self.device,dtype=torch.float64)
        c = self._flop_coefficients
        flops = self._linear_flops + c['main_attention']*(lengths+images).square().sum() + c['text_attention']*lengths.square().sum()
        return tokens, flops

    def end_update(self, step, lr=None, grad_norm=None, skipped_batches=0,
                   data_wait_seconds=0., data_wait_max_seconds=0., data_requests=0,
                   data_read_seconds=0., data_read_max_seconds=0.):
        if not self.enabled:
            return {}
        self._updates += 1
        self._skipped += int(skipped_batches)
        self._data_wait += float(data_wait_seconds)
        self._data_read += float(data_read_seconds)
        self._data_wait_max = max(self._data_wait_max, float(data_wait_max_seconds))
        self._data_read_max = max(self._data_read_max, float(data_read_max_seconds))
        self._data_requests += int(data_requests)
        if not self.should_report:
            return {}
        # Existing training metrics already require a host read at this boundary.
        # No event wait/synchronize is introduced here, including for pending events.
        elapsed = max(self.clock() - self._window_start, 1e-9)
        phase_sums, phase_counts = self._take_phase_events()
        text_tokens, flops = self._window_token_totals()
        extra = torch.tensor([self._samples, self._image_tokens, self._padded_text_tokens,
                              self._micros, *phase_sums, *phase_counts,
                              self._data_wait, self._data_read, self._data_requests], device=self.device, dtype=torch.float64)
        extra = torch.cat((extra, text_tokens.reshape(1), self._invalid_metrics.reshape(1), flops.reshape(1)))
        summed = torch.cat((self._totals, self._counts, extra))
        grad = torch.as_tensor(float('nan') if grad_norm is None else grad_norm,
                               device=self.device, dtype=torch.float64).detach().reshape(())
        grad_valid = torch.isfinite(grad)
        bad_grad = (~grad_valid).to(torch.float64) if grad_norm is not None else self._zero
        peak = torch.cuda.max_memory_allocated(self.device) / 2**30 if self.device.type == 'cuda' else 0.
        allocated = torch.cuda.memory_allocated(self.device) / 2**30 if self.device.type == 'cuda' else 0.
        cpu_maximum = torch.tensor([elapsed, self._updates, self._skipped, peak, allocated,
            self._data_wait, self._data_read, self._data_wait_max, self._data_read_max],
            device=self.device, dtype=torch.float64)
        maximum = torch.cat((self._maxs, cpu_maximum,
            torch.stack((torch.where(grad_valid, grad, -math.inf), bad_grad))))
        minimum = self._mins.clone()
        distributed = dist.is_available() and dist.is_initialized()
        world = dist.get_world_size() if distributed else 1
        if distributed:
            dist.all_reduce(summed, op=dist.ReduceOp.SUM)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        # A single host transfer, not one .item() per metric/site.
        host = torch.cat((summed, maximum, minimum)).cpu().tolist()
        n = self._size
        totals, counts = host[:n], host[n:2*n]
        offset = 2*n
        samples, images, padded, micros = host[offset:offset+4]
        offset += 4
        durations = host[offset:offset+len(PHASES)]; offset += len(PHASES)
        event_counts = host[offset:offset+len(PHASES)]; offset += len(PHASES)
        data_wait, data_read, data_requests = host[offset:offset+3]; offset += 3
        text_tokens, invalid, flops = host[offset:offset+3]; offset += 3
        maxima = host[offset:offset+n]; offset += n
        (seconds, updates, skipped, peak, allocated, data_wait_slowest, data_read_slowest,
            data_wait_max, data_read_max, grad_value, bad_grad) = host[offset:offset+11]; offset += 11
        minima = host[offset:offset+n]
        updates = max(updates, 1.)
        payload = {
            'performance/seconds_per_update': seconds / updates,
            'performance/updates_per_sec': updates / seconds,
            'performance/samples_per_sec': samples / seconds,
            'performance/samples_per_update': samples / updates,
            'performance/micro_batches_per_update': micros / (updates * world),
            'performance/image_tokens_per_sec': images / seconds,
            'performance/text_tokens_per_sec': text_tokens / seconds,
            'performance/padded_text_tokens_per_sec': padded / seconds,
            'performance/text_padding_fraction': max(0., 1. - text_tokens/max(padded, 1.)),
            'performance/skipped_batches_window': skipped,
            'memory/gpu_peak_allocated_gib': peak,
            'memory/gpu_allocated_gib': allocated,
            'health/nonfinite_monitor_values': invalid,
            'health/nonfinite_grad_norm': bad_grad,
            'data/requests_global_sum': data_requests,
            'data/requests_mean_per_rank': data_requests / world,
        }
        for prefix, total, slowest, longest in (
            ('fetch_and_sync', data_wait, data_wait_slowest, data_wait_max),
            ('read_wait', data_read, data_read_slowest, data_read_max)):
            payload.update({
                f'data/{prefix}_seconds_global_sum': total,
                f'data/{prefix}_seconds_mean_per_rank': total / world,
                f'data/{prefix}_seconds_slowest_rank': slowest,
                f'data/{prefix}_seconds_max_request': longest,
                f'data/{prefix}_wall_fraction_per_rank': total / (seconds * world),
            })
        if math.isfinite(grad_value):
            payload['train/grad_norm'] = grad_value
        if lr is not None:
            payload['train/lr'] = float(lr)
        if self._flop_coefficients is not None:
            tflops = flops / (seconds * world * 1e12)
            estimate = tflops / self._peak_tflops
            # Never clamp to 100%: an impossible result should expose a wrong
            # timing/count/peak assumption instead of hiding it.
            payload.update({'performance/dit_tflops_per_gpu': tflops,
                'performance/mfu_dit_estimate': estimate,
                'performance/mfu_dit_estimate_percent': 100*estimate})
        for name, duration, count in zip(PHASES, durations, event_counts):
            if count:
                payload[f'performance/gpu_phase_ms/{name}'] = duration/count
        for group, indices in self._groups.items():
            by_key = {}
            for i in indices:
                by_key.setdefault(self._schema[i][1], []).append(i)
            for key, positions in by_key.items():
                valid_positions = [i for i in positions if counts[i] > 0]
                if not valid_positions:
                    continue
                reducer = _reducer(key)
                if reducer == 'max':
                    value = max(maxima[i] for i in valid_positions)
                elif reducer == 'min':
                    value = min(minima[i] for i in valid_positions)
                elif reducer == 'sum':
                    value = sum(totals[i] for i in valid_positions)
                else:
                    value = sum(totals[i] for i in valid_positions) / sum(counts[i] for i in valid_positions)
                if math.isfinite(value):
                    payload[f'mhc/{group}/{key}'] = value
        self.model.set_mhc_collect_stats(False)
        self._collecting = False
        self.should_report = False
        self._reset_window()
        self._window_start = self.clock()
        return payload
