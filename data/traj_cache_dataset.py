"""Teacher Trajectory Replay 缓存读取 (KD traj 模式)。

缓存由 scripts/cache/gen_teacher_traj_cache.py 离线生成:
    cache_dir/
      index_rank{r}.json     # 每个生成 rank 一份: {"meta": {...}, "shards": [{"dir":..., "rows":...}, ...]}
      shard_r{r}_{seq}/
        x_start.npy          # [N, C, h, w] fp16  teacher 轨迹状态 x_k^T (σ_k)
        x_end.npy            # [N, C, h, w] fp16  teacher 轨迹状态 x_{k+2}^T (σ_{k+2})
        sigmas.npy           # [N, 3] fp32       (σ_k, σ_{k+1}, σ_{k+2}), 推理 scheduler 真实值
        captions.json        # list[str] len N   生成该轨迹的 caption (teacher chat-template 条件)

每个样本是一个 2-step 轨迹区间: student 从 x_k^T 出发用自身速度场 (CFG 组合) 走两步
(必须经过自己预测的 x̂_{k+1}, 不许重塞 teacher 中间状态), 对齐 x_{k+2}^T。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TeacherTrajCacheDataset(Dataset):
    """把分片缓存呈现为平面样本流: item = {x_start, x_end, sigmas, caption}。"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        index_files = sorted(self.cache_dir.glob("index_rank*.json"))
        if not index_files:
            raise FileNotFoundError(f"{self.cache_dir} 下没有 index_rank*.json (缓存未生成或路径错误)")
        self.shards: list[dict] = []
        self.meta: dict = {}
        for fp in index_files:
            data = json.loads(fp.read_text())
            if not self.meta:
                self.meta = data.get("meta", {})
            for shard in data.get("shards", []):
                if shard["rows"] > 0 and Path(shard["dir"]).is_dir():
                    self.shards.append(shard)
        if not self.shards:
            raise FileNotFoundError(f"{self.cache_dir} 的 index 里没有任何有效 shard")
        # 前缀和 → 全局 idx → (shard, row)
        self._cum = []
        total = 0
        for shard in self.shards:
            total += shard["rows"]
            self._cum.append(total)
        self.total_rows = total
        self._mmap_cache: dict[str, dict] = {}
        self._caption_cache: dict[str, list[str]] = {}

    def __len__(self):
        return self.total_rows

    def _locate(self, idx):
        lo, hi = 0, len(self._cum)
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        shard = self.shards[lo]
        row = idx - (self._cum[lo - 1] if lo > 0 else 0)
        return shard["dir"], row

    def _load_shard(self, shard_dir: str) -> dict:
        if shard_dir not in self._mmap_cache:
            if len(self._mmap_cache) >= 4:  # 每 worker 最多驻留 4 个 shard 的 mmap 句柄
                self._mmap_cache.pop(next(iter(self._mmap_cache)))
            self._mmap_cache[shard_dir] = {
                "x_start": np.load(os.path.join(shard_dir, "x_start.npy"), mmap_mode="r"),
                "x_end": np.load(os.path.join(shard_dir, "x_end.npy"), mmap_mode="r"),
                "sigmas": np.load(os.path.join(shard_dir, "sigmas.npy"), mmap_mode="r"),
            }
        return self._mmap_cache[shard_dir]

    def _load_captions(self, shard_dir: str) -> list[str]:
        if shard_dir not in self._caption_cache:
            if len(self._caption_cache) >= 8:
                self._caption_cache.pop(next(iter(self._caption_cache)))
            self._caption_cache[shard_dir] = json.loads(
                Path(shard_dir, "captions.json").read_text())
        return self._caption_cache[shard_dir]

    def __getitem__(self, idx):
        shard_dir, row = self._locate(idx)
        shard = self._load_shard(shard_dir)
        return {
            # fp16 numpy 传输 (同 collate_image_batch 的 shm 考量), 训练侧转 bf16;
            # np.array 显式拷贝避免 mmap 只读视图触发 torch warning
            "x_start": torch.from_numpy(np.array(shard["x_start"][row])).float(),
            "x_end": torch.from_numpy(np.array(shard["x_end"][row])).float(),
            "sigmas": torch.from_numpy(np.array(shard["sigmas"][row])).float(),
            "caption": self._load_captions(shard_dir)[row],
        }


def collate_traj_batch(samples):
    """list[dict] → numpy dict (避免 torch tensor 走 /dev/shm); 训练侧再转 torch。"""
    return {
        "x_start": np.stack([s["x_start"].numpy() for s in samples]).astype(np.float16),
        "x_end": np.stack([s["x_end"].numpy() for s in samples]).astype(np.float16),
        "sigmas": np.stack([s["sigmas"].numpy() for s in samples]).astype(np.float32),
        "captions": [s["caption"] for s in samples],
    }


__all__ = ["TeacherTrajCacheDataset", "collate_traj_batch"]
