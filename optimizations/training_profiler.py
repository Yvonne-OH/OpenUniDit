"""Bounded, opt-in per-rank CUDA traces. No profiling overhead when disabled.

NCCL-only device intervals are an overlap diagnostic, NOT critical-path wait.
Do not add overlapping kernel durations or call this quantity exposed latency.
"""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import socket
import uuid
import torch
import torch.distributed as dist


def merge_intervals(intervals):
    result = []
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(stop, result[-1][1]))
        else:
            result.append((start, stop))
    return result


def duration(intervals):
    return sum(b - a for a, b in merge_intervals(intervals))


def overlap(a, b):
    a, b = merge_intervals(a), merge_intervals(b)
    i = j = 0; total = 0.
    while i < len(a) and j < len(b):
        total += max(0., min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return total


def summarize_trace(path):
    events = json.loads(Path(path).read_text())['traceEvents']
    groups = {}
    for e in events:
        if e.get('ph') != 'X' or e.get('cat') != 'kernel':
            continue
        device = str(e.get('args', {}).get('device', e.get('pid')))
        g = groups.setdefault(device, {'comm': [], 'compute': []})
        name = e.get('name', '').lower()
        key = 'comm' if 'nccl' in name or 'rccl' in name else 'compute'
        g[key].append((e['ts'], e['ts'] + e.get('dur', 0)))
    result = {}
    for device, g in groups.items():
        comm, compute = duration(g['comm']), duration(g['compute'])
        shared = overlap(g['comm'], g['compute'])
        result[device] = {'nccl_union_ms': comm / 1000, 'compute_union_ms': compute / 1000,
                          'overlap_ms': shared / 1000, 'nccl_without_compute_ms': (comm - shared) / 1000}
    return {'devices': result, 'interpretation': 'NCCL without concurrent compute is NOT measured critical-path wait; inspect stream dependencies and stragglers in the trace.'}


class TrainingProfiler:
    def __init__(self, cfg, output_dir, model):
        options = cfg.get('monitoring', {}).get('profiler', {})
        self.enabled = bool(options.get('enabled', False))
        self.prof = None; self.scope = None; self.handles = []; self.stack = {}
        self.count = 0; self.done = False
        self.wait = int(options.get('wait_steps', 5))
        self.warmup = int(options.get('warmup_steps', 1))
        self.active = int(options.get('active_steps', 3))
        if min(self.wait, self.warmup) < 0 or self.active < 1:
            raise ValueError('Invalid profiler window')
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        ranks = options.get('ranks', 'all')
        self.selected = ranks == 'all' or self.rank in ranks
        self.backend = cfg.get('distributed', {}).get('backend', 'deepspeed')
        self.path = Path(output_dir) / 'profiles' / f'{self.backend}-{socket.gethostname()}-rank{self.rank}-{os.getpid()}-{uuid.uuid4().hex[:8]}'
        self.model = model
        self.layer_hooks = bool(options.get('layer_hooks', True))
        self.record_shapes = bool(options.get('record_shapes', False))
        self.profile_memory = bool(options.get('profile_memory', False))

    def _install_hooks(self):
        if not self.layer_hooks:
            return
        for index, block in enumerate(getattr(self.model, 'blocks', [])):
            label = f'giga/block_{index}'
            def pre(module, args, label=label):
                scope = torch.profiler.record_function(label)
                scope.__enter__(); self.stack.setdefault(module, []).append(scope)
            def post(module, args, output):
                scopes = self.stack.get(module, [])
                if scopes: scopes.pop().__exit__(None, None, None)
            self.handles.extend([block.register_forward_pre_hook(pre),
                                 block.register_forward_hook(post, always_call=True)])

    def begin_update(self, step):
        if not self.enabled or self.done or not self.selected:
            return
        self.count += 1
        if self.count == self.wait + 1:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available(): activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.prof = torch.profiler.profile(activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=self.warmup, active=self.active, repeat=1),
                record_shapes=self.record_shapes, profile_memory=self.profile_memory,
                on_trace_ready=self._export)
            self.prof.__enter__(); self._install_hooks()
        if self.prof is not None:
            self.scope = torch.profiler.record_function(f'giga/update_{step}')
            self.scope.__enter__()

    def _export(self, prof):
        trace = self.path.with_suffix('.json'); prof.export_chrome_trace(str(trace))
        summary = summarize_trace(trace)
        summary.update(backend=self.backend, rank=self.rank, hostname=socket.gethostname(), active_steps=self.active)
        self.path.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))

    def end_update(self):
        if self.prof is None:
            return
        if self.scope is not None:
            self.scope.__exit__(None, None, None); self.scope = None
        self.prof.step()
        if self.count >= self.wait + self.warmup + self.active:
            self.close(); self.done = True

    def close(self):
        if self.scope is not None:
            self.scope.__exit__(None, None, None); self.scope = None
        for handle in self.handles: handle.remove()
        self.handles.clear()
        if self.prof is not None:
            self.prof.__exit__(None, None, None); self.prof = None

    def phase(self, name):
        return torch.profiler.record_function('giga/' + name) if self.prof is not None else nullcontext()
