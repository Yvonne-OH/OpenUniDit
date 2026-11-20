"""本机数据管线压测: 复用训练真实路径 GigaLMDBDataset(tar_parquet) + collate_image_batch.

用法: python stress_data.py <config.yaml> <index_cache_dir> <num_workers> <persistent_decoder> <seconds> <batch_size>
env YISU_IMAGE_PERSISTENT_DECODER 必须在启动前设置 (worker 子进程继承)。
"""
import os, sys, time, statistics, yaml

config_path, cache_dir, workers, persistent, seconds, batch_size = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], float(sys.argv[5]), int(sys.argv[6]))
os.environ["YISU_IMAGE_PERSISTENT_DECODER"] = persistent
# 与训练脚本一致的超时预算
os.environ.setdefault("YISU_IMAGE_TIMEOUT_SECONDS", "48")
os.environ.setdefault("YISU_IMAGE_SAMPLE_TIMEOUT_SECONDS", "120")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import torch  # noqa: E402
from data.lmdb_dataset import GigaLMDBDataset, collate_image_batch  # noqa: E402
from data.samplers.bucket_sampler import BucketBatchSampler  # noqa: E402

cfg = yaml.safe_load(open(config_path).read())
dcfg, trcfg = cfg["dataset"], cfg["training"]
t0 = time.perf_counter()
ds = GigaLMDBDataset(dcfg, seed=42, index_cache_dir=cache_dir)
print(f"[init] dataset ready in {time.perf_counter()-t0:.1f}s, len={len(ds.records)} records", flush=True)

# 与训练完全一致的 bucket 采样 (rank0/单机视角): batch_size=1 + 256桶 b=2
batch_sampler = BucketBatchSampler(
    ds.bucket_indices, trcfg["batch_size_per_device"],
    weights=dcfg.get("bucket_sampling_weights"),
    bucket_batch_sizes=dcfg.get("bucket_batch_sizes"),
    seed=42, rank=0, world_size=1, randomize_order_on_start=True, drop_last=True)
batch_size = None  # 由 batch_sampler 决定 (bucket 混合 1/2)

loader = torch.utils.data.DataLoader(
    ds, batch_sampler=batch_sampler, num_workers=workers,
    prefetch_factor=4, persistent_workers=True, collate_fn=collate_image_batch)

t_start = time.perf_counter()
t0 = t_start
fetch_times, n_samples, n_batches = [], 0, 0
slow_batches = 0
try:
    for batch in loader:
        t_fetch = time.perf_counter() - t0
        fetch_times.append(t_fetch)
        if t_fetch > 5.0:
            slow_batches += 1
        n_samples += len(batch["pixel_values"])
        n_batches += 1
        if time.perf_counter() - t_start >= seconds:
            break
        t0 = time.perf_counter()
except KeyboardInterrupt:
    pass

total = time.perf_counter() - t_start if fetch_times else 0
ft = sorted(fetch_times)
if ft:
    n = len(ft)
    # 稳态速率: 去掉首 10% (worker 冷启动/NFS page cache 预热)
    warm = ft[int(n*0.1):]
    print(f"[result] workers={workers} persistent={persistent} wall={total:.1f}s")
    print(f"[result] batches={n} samples={n_samples}")
    print(f"[result] 吞吐: overall={n_samples/total:.2f} samples/s  ({n/total:.2f} batches/s)")
    print(f"[result] batch fetch: p50={ft[n//2]:.3f}s p90={ft[int(n*0.9)]:.3f}s p99={ft[int(n*0.99)]:.3f}s max={ft[-1]:.2f}s")
    print(f"[result] >5s batch 比例: {slow_batches}/{n} = {100*slow_batches/n:.1f}%")
    sw = sum(warm)
    warm_samples = n_samples * len(warm) / n
    print(f"[result] 稳态(去首10%): {warm_samples/sw:.2f} samples/s")
