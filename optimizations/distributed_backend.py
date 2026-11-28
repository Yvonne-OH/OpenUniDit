"""Optional FSDP2/HSDP engine with the stage-1 loop's accumulation contract.

DeepSpeed remains the default. FP32 sharded parameters/Adam moments are kept;
only forward/backward parameters are cast to the requested compute dtype.
"""
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import uuid

import torch
import torch.distributed as dist


def backend_name(cfg):
    name = str(cfg.get('distributed', {}).get('backend', 'deepspeed')).lower()
    if name not in {'deepspeed', 'fsdp2', 'hsdp'}:
        raise ValueError(f'Unknown distributed.backend: {name}')
    return name


def validate_backend(cfg):
    name = backend_name(cfg)
    if name == 'deepspeed':
        return
    train = cfg['training']
    if train.get('muon', {}).get('enabled', True):
        raise ValueError('FSDP2 currently supports AdamW; set training.muon.enabled: false')
    if train.get('torch_compile') or train.get('regional_compile'):
        raise ValueError('FSDP2 compilation is not validated; disable both compile options')
    hc = cfg.get('mhc', False)
    if (cfg.get('model', {}).get('moe') or {}).get('enabled', False) or (hc if isinstance(hc, bool) else hc.get('enabled', False)):
        raise ValueError('FSDP2 currently supports the dense model only')


class FSDP2Engine:
    def __init__(self, model, cfg, dtype):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        validate_backend(cfg)
        if getattr(model, 'has_moe', False) or getattr(model, 'is_mhc_model', False):
            raise ValueError('FSDP2 currently supports dense models only')
        if not torch.cuda.is_available():
            raise RuntimeError('This FSDP2 training backend requires CUDA')
        self.backend = backend_name(cfg)
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.gas = int(cfg['training'].get('gradient_accumulation_steps', 1))
        if self.gas < 1:
            raise ValueError('gradient_accumulation_steps must be positive')
        options = cfg.get('distributed', {}).get('fsdp2', {})
        self.gradient_clipping = float(options.get('gradient_clipping', 1.0))
        self.sync_each_micro = bool(options.get('sync_each_micro', True))
        world = dist.get_world_size()
        shard_size = int(options.get('shard_size', int(os.environ.get('LOCAL_WORLD_SIZE', 8))))
        if self.backend == 'hsdp':
            if shard_size < 1 or world % shard_size:
                raise ValueError('HSDP shard_size must divide world_size')
            local_world = int(os.environ.get('LOCAL_WORLD_SIZE', shard_size))
            if shard_size > local_world or local_world % shard_size:
                raise ValueError('HSDP shard groups must fit within a node and divide LOCAL_WORLD_SIZE')
            mesh = init_device_mesh('cuda', (world // shard_size, shard_size), mesh_dim_names=('replicate', 'shard'))
        else:
            mesh = init_device_mesh('cuda', (world,), mesh_dim_names=('shard',))
        self.mesh = mesh
        reduce_dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}.get(options.get('reduce_dtype', 'fp32'))
        if reduce_dtype is None:
            raise ValueError('fsdp2.reduce_dtype must be fp32 or bf16')
        # The DiT deliberately passes FP32 RoPE/sigma tensors alongside BF16
        # activations. Recursive input casting changes these tensors and makes
        # checkpoint recomputation disagree with the original forward.
        policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=False)
        # Converting a model to BF16 before fully_shard would lose FP32 master updates.
        model.float()
        for value in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(value.data, src=0)
        self.units = []
        reshard = bool(options.get('reshard_after_forward', True))
        for block in getattr(model, 'blocks', []):
            fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard)
            self.units.append((block, reshard))
        refiner = getattr(model, 'text_refiner', None)
        if refiner is not None:
            fully_shard(refiner, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard)
            self.units.append((refiner, reshard))
        fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
        self.units.append((model, False))
        self.module = model
        train = cfg['training']; adam = train.get('adam', {})
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=float(train.get('learning_rate', 1e-4)),
            betas=tuple(adam.get('betas', [0.9, 0.95])), eps=float(adam.get('epsilon', 1e-8)),
            weight_decay=float(adam.get('weight_decay', 0.0)), foreach=False)
        self.global_steps = self.micro_steps = 0
        self._grad_norm = None
        self._validation = False

    def __call__(self, *args, **kwargs):
        if self.module.training and torch.is_grad_enabled():
            self.module.set_requires_gradient_sync(self.sync_each_micro or self.is_gradient_accumulation_boundary())
        return self.module(*args, **kwargs)

    def backward(self, loss):
        # sample_loss_scale in the production loop already accounts for GAS/DP.
        (loss / self.gas).backward()

    def is_gradient_accumulation_boundary(self):
        return (self.micro_steps + 1) % self.gas == 0

    def step(self):
        boundary = self.is_gradient_accumulation_boundary()
        self.micro_steps += 1
        if not boundary:
            return
        norm = torch.nn.utils.clip_grad_norm_(self.module.parameters(),
            self.gradient_clipping if self.gradient_clipping > 0 else float('inf'), foreach=False)
        self._grad_norm = norm.full_tensor() if hasattr(norm, 'full_tensor') else norm
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_steps += 1

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def get_global_grad_norm(self):
        return self._grad_norm

    def train(self):
        self.module.train(); return self

    def eval(self):
        self.module.eval(); return self

    def zero_optimization_stage(self):
        # Existing fixed validation can only run inside the collective materialization context.
        return 2 if self._validation else 3

    @contextmanager
    def validation_context(self):
        # All ranks enter/exit, even ranks assigned no fixed prompts. Keep weights
        # materialized so unequal local inference counts perform no collectives.
        for unit, _ in self.units:
            unit.set_reshard_after_forward(False, recurse=False)
        for unit, _ in self.units:
            unit.unshard()
        self._validation = True
        try:
            yield
        finally:
            self._validation = False
            for unit, original in reversed(self.units):
                unit.reshard()
                unit.set_reshard_after_forward(original, recurse=False)
            dist.barrier()

    def full_weights(self):
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        return get_model_state_dict(self.module, options=StateDictOptions(full_state_dict=True, cpu_offload=True))

    def load_full_weights(self, state):
        from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
        # Resumes must have the same architecture. Cross-branch migrations use
        # init_from, which is applied before sharding by the existing loader.
        result = set_model_state_dict(self.module, state,
            options=StateDictOptions(full_state_dict=True, strict=True))
        self.optimizer.state.clear()
        return len(state), len(result.missing_keys), len(result.unexpected_keys)

    def save_checkpoint(self, directory, tag, client_state):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict
        target = Path(directory) / tag
        # DCP publishes .metadata only after all ranks finish writing shards.
        model_state, optim_state = get_state_dict(self.module, self.optimizer)
        dcp.save({'model': model_state, 'optimizer': optim_state}, checkpoint_id=str(target / 'dcp'))
        rng_dir = target / 'rng'; rng_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}, rng_dir / f'rank-{dist.get_rank()}.pt')
        dist.barrier()
        if dist.get_rank() == 0:
            meta = dict(backend=self.backend, world_size=dist.get_world_size(), gas=self.gas,
                        global_steps=self.global_steps, micro_steps=self.micro_steps, client_state=client_state)
            temporary = target / f'.metadata-{uuid.uuid4().hex}.tmp'
            temporary.write_text(json.dumps(meta)); os.replace(temporary, target / 'fsdp2.json')
        dist.barrier()
        return True

    def load_checkpoint(self, directory, tag, **kwargs):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
        target = Path(directory) / tag
        meta = json.loads((target / 'fsdp2.json').read_text())
        if meta['world_size'] != dist.get_world_size() or meta['gas'] != self.gas:
            raise ValueError('Exact FSDP2 state resume requires unchanged world_size and accumulation')
        model_state, optim_state = get_state_dict(self.module, self.optimizer)
        state = {'model': model_state, 'optimizer': optim_state}
        dcp.load(state, checkpoint_id=str(target / 'dcp'))
        set_state_dict(self.module, self.optimizer, model_state_dict=state['model'], optim_state_dict=state['optimizer'])
        self.global_steps, self.micro_steps = meta['global_steps'], meta['micro_steps']
        rng = torch.load(target / 'rng' / f'rank-{dist.get_rank()}.pt', map_location='cpu', weights_only=True)
        torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state(rng['cuda'])
        return str(target), meta['client_state']


def validation_context(engine):
    return engine.validation_context() if isinstance(engine, FSDP2Engine) else nullcontext()
