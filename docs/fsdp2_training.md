# Optional FSDP2 / HSDP training

The existing launcher and training loop are shared. Omitted `distributed.backend`
means `deepspeed`; existing YAMLs keep their existing behavior. New backends are
opt-in for dense AdamW training. No production YAML is switched automatically.

```yaml
distributed:
  backend: hsdp  # fsdp2 = global sharding; hsdp = 2D replicate/shard mesh
  fsdp2:
    shard_size: 8  # production: 8 per node; single-node 2x4 test: set 4
    reshard_after_forward: true
    sync_each_micro: true
    reduce_dtype: fp32
    gradient_clipping: 1.0
training:
  muon:
    enabled: false
  torch_compile: false
  regional_compile: false
monitoring:
  profiler:
    enabled: true
    wait_steps: 5
    warmup_steps: 1
    active_steps: 3
    ranks: all
    layer_hooks: true
```

Use a new output directory and SwanLab identity. `init_from` supports existing
weight migrations before sharding. `resume_mode: auto` imports same-architecture
pure weights and resets optimizer, matching existing semantics. `state` restores
FSDP2 DCP optimizer/model state and per-rank torch RNG with unchanged world size
and accumulation; an incompatible state raises rather than silently resetting
Adam. DeepSpeed optimizer checkpoint files cannot be loaded as FSDP2 checkpoints.

FP32 master parameters and Adam moments are sharded; BF16 forward parameters are
materialized on demand. Global per-image loss scaling and GAS division are the
same as the DeepSpeed loop. `sync_each_micro: false` delays gradient synchronization
until the final microbatch, using more gradient memory. Measure both memory and
speed; do not conflate this change with replacing the backend.

Validation collectively materializes full compute weights on every rank before
running unequal per-rank prompt counts, then reshards on every rank. This costs
full-model compute-weight memory; it is required to avoid unequal collective
counts. Both fixed and next-batch validation use this context.

Profiles are bounded windows relative to process start, excluding checkpoint and
validation. CPU phase and per-layer forward/recompute hooks are included, along
with CUDA/NCCL kernels. `profiles/*.summary.json` computes device-interval unions
and overlap, not sums of concurrent kernels. `nccl_without_compute_ms` is NOT
critical-path communication wait. Use Chrome/Perfetto trace stream dependencies
to separate exposed communication, launch gaps, and stragglers. Run throughput
measurement separately with profiling disabled, at matching data/token shapes,
GAS, precision, AC and optimizer. Verify multi-host placement before interpreting
multi-node scaling results.

A single 8-GPU host with `backend: hsdp` and `shard_size: 4` tests two
replicas of four-way shards. It is NOT an inter-node bandwidth experiment.

The provided YAMLs use `resume_mode: auto` and `initial_resume_from` pointing to
step 40000. On first launch only, if the new output has no weights, the fallback
preserves the source step/sample counters. Subsequent restarts prefer the new
output's latest weights. Cross-backend Adam state is not migrated.

## Developer-host comparison

Use eight ranks on one host; HSDP uses `--shard-size 4`. Run one backend at a
time with the same fixed `--weights`, `--batch 8 --accum 2`, image shape, text
length, and `--ac`. `tests/benchmark_distributed_backend.py --help` lists all
options. The default benchmark includes frozen Qwen/VAE with synthetic repeated
inputs, excluding PFS, data loading and tokenization. `--dit-only` explicitly
excludes the encoders. `--profile` is a separate diagnostic run: its wall time
includes trace export and MUST NOT be compared with unprofiled throughput.

Peak memory is PyTorch allocated/reserved memory, not a sampled nvidia-smi peak.
Full-size benchmarking requires a fixed checkpoint. `--tiny` is a functional
smoke test, not a performance proxy for the full model.

The cluster suite (`tests/benchmark_backend_suite.py`) launches each case in a
separate process group and waits for both nodes to finish before advancing.
Numerical preflights are required; an OOM in a performance case is recorded and
does not silently reduce the batch. A stale GPU process aborts the suite.
The two-node matrix uses HSDP `shard_size: 8`, unlike the developer-host 2x4 test.
