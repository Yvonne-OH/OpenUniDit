"""Weighted bucket batch sampler for image/video datasets."""
import math
import random
from collections.abc import Iterator
from torch.utils.data import Sampler


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(self, bucket_indices: dict[str, list[int]], batch_size: int,
                 weights: dict[str, float] | None = None, seed: int = 42,
                 drop_last: bool = False, rank: int = 0, world_size: int = 1,
                 randomize_order_on_start: bool = True,
                 bucket_batch_sizes: dict[str, int] | None = None,
                 max_repeat_per_epoch: float | None = None):
        self.bucket_indices = {key: list(value) for key, value in bucket_indices.items() if value}
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size 必须为正整数, 当前: {batch_size}")
        self.bucket_batch_sizes = {str(key): int(value) for key, value in (bucket_batch_sizes or {}).items()}
        for key, value in self.bucket_batch_sizes.items():
            if value <= 0:
                raise ValueError(f"bucket batch size 必须为正整数: {key}={value}")
        self.weights = dict(weights) if weights else None
        self.seed = seed
        self.drop_last = drop_last
        self.rank = int(rank)
        self.world_size = int(world_size)
        # False 时每个 epoch 使用完全相同的采样顺序 (seed 不叠加 epoch); True 时逐 epoch reshuffle
        self.randomize_order_on_start = bool(randomize_order_on_start)
        # 数据量感知配额 (bucket_max_repeat_per_epoch):
        # - 权重=0 的桶明确不加载; 样本数×repeat 不足一个全局 batch 的稀疏桶自动跳过 (避免反复抽同几条);
        # - 其余桶配额上限 = ceil(pool×repeat/全局batch), 权重配额超出时削顶, 余量按权重转移给数据充足的桶;
        # - epoch 总步数随可用数据量收缩, __len__ 与实际迭代严格一致; None=旧行为 (纯权重分配)。
        if max_repeat_per_epoch is not None and max_repeat_per_epoch < 1.0:
            raise ValueError(f"max_repeat_per_epoch 必须 >= 1.0, 当前: {max_repeat_per_epoch}")
        self.max_repeat_per_epoch = max_repeat_per_epoch
        self.start_batch = 0
        self.epoch = 0
        self._start_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))
        if self.epoch != self._start_epoch:
            self.start_batch = 0

    def set_start_batch(self, start_batch: int, epoch: int = 0) -> None:
        self.start_batch = max(0, int(start_batch))
        self._start_epoch = max(0, int(epoch))
        self.epoch = self._start_epoch

    def _batch_size_for_bucket(self, bucket: str) -> int:
        if bucket in self.bucket_batch_sizes:
            return self.bucket_batch_sizes[bucket]
        tail = bucket.split(":")[-1]
        if tail in self.bucket_batch_sizes:
            return self.bucket_batch_sizes[tail]
        return self.batch_size

    def _num_steps_for_bucket(self, bucket: str, items: list[int]) -> int:
        global_batch_size = self._batch_size_for_bucket(bucket) * self.world_size
        if self.drop_last:
            return len(items) // global_batch_size
        return (len(items) + global_batch_size - 1) // global_batch_size

    def _weight_value(self, key: str) -> float:
        """桶权重查表: 显式 key → 冒号后最后一段 (空间 bucket 名, 使同一空间桶的
        image/edit 拿相同权重) → 1.0; weights 未设置时按数据量 (pool 大小) 作为权重。"""
        if self.weights is None:
            self.weights = {k: len(v) for k, v in self.bucket_indices.items()}
        if key in self.weights:
            return max(0.0, float(self.weights[key]))
        tail = key.split(":")[-1]
        if tail in self.weights:
            return max(0.0, float(self.weights[tail]))
        return 1.0

    def _bucket_limits(self, keys: list[str], values: list[float]) -> list[int] | None:
        """数据量感知的每桶 batch 配额硬上限 (max_repeat_per_epoch=None 时返回 None)。

        - 权重≤0: 0 (明确不加载);
        - pool×repeat 不足一个全局 batch (稀疏桶, 任一 batch 都要把同几条样本重复多倍): 0 (自动跳过);
        - 其余: ceil(pool×repeat/全局batch) — 恰好完整覆盖一遍数据 (尾批少量 wrap, 与旧行为一致)。
        """
        if self.max_repeat_per_epoch is None:
            return None
        limits = []
        for key, weight in zip(keys, values):
            if weight <= 0:
                limits.append(0)
                continue
            pool = len(self.bucket_indices[key])
            global_bs = self._batch_size_for_bucket(key) * self.world_size
            raw = pool * self.max_repeat_per_epoch / global_bs
            limits.append(int(math.ceil(raw - 1e-9)) if raw >= 1.0 else 0)
        return limits

    def _total_steps(self) -> int:
        keys = list(self.bucket_indices)
        values = [self._weight_value(key) for key in keys]
        limits = self._bucket_limits(keys, values)
        if limits is None:
            return sum(self._num_steps_for_bucket(key, items) for key, items in self.bucket_indices.items())
        # 配额感知: 总步数 = 权重+上限联合分配后的实际配额和, 与 __iter__ 生成的 bucket_order 严格一致
        return sum(self._allocate(sum(limits), values, limits))

    def _weighted_bucket_order(self, rng: random.Random, total: int) -> list[str]:
        keys = list(self.bucket_indices)
        values = [self._weight_value(key) for key in keys]
        weight_sum = sum(values)
        if weight_sum <= 0:
            raise ValueError("bucket weights 至少需要一个正值")
        limits = self._bucket_limits(keys, values)
        if limits is None:
            quotas = self._integerize(total, [total * v / weight_sum for v in values])
        else:
            # caps 模式下 total 由 _total_steps 按 limits 推导, 这里用 sum(limits) 保证两处一致
            quotas = self._allocate(sum(limits), values, limits)
        buckets = [key for key, quota in zip(keys, quotas) for _ in range(quota)]
        rng.shuffle(buckets)
        return buckets

    def _allocate(self, total: int, values: list[float], limits: list[int] | None) -> list[int]:
        """把 total 个 batch 按权重分给各桶; 有 limits 时超限桶削到上限, 余量按权重转移给未达限桶。"""
        n = len(values)
        if limits is None:
            weight_sum = sum(values)
            return self._integerize(total, [total * v / weight_sum for v in values])
        locked = [False] * n
        quotas = [0] * n
        for _ in range(n + 1):
            # 未锁死桶的可承受余量
            room = [limits[i] - quotas[i] if not locked[i] else 0 for i in range(n)]
            deficit = total - sum(quotas)
            if deficit <= 0:
                break
            active = [i for i in range(n) if room[i] > 0]
            if not active:
                break  # 全部达到上限; total = Σlimits 时理论上不会发生 (权重=0 的桶 limits 已为 0)
            # 把 deficit 按权重分给仍有 room 的桶
            wsum = sum(values[i] for i in active)
            if wsum <= 0:
                share = {i: deficit / len(active) for i in active}
            else:
                share = {i: deficit * values[i] / wsum for i in active}
            newly_locked = False
            for i in active:
                quotas[i] += share[i]
                if quotas[i] > limits[i]:
                    quotas[i] = limits[i]
                    locked[i] = True
                    newly_locked = True
            if not newly_locked:
                return self._integerize(total, quotas, limits)
        return self._integerize(total, quotas, limits)

    @staticmethod
    def _integerize(total: int, exact: list[float], limits: list[int] | None = None) -> list[int]:
        """浮点配额取整: floor + 余数按小数部分降序补 1 (limits 限制下余数只分给未达限的桶)。"""
        quotas = [int(v) for v in exact]
        remainder = total - sum(quotas)
        if remainder > 0:
            order = sorted(range(len(exact)), key=lambda i: (-(exact[i] - quotas[i]), i))
            if limits is not None:
                order = [i for i in order if quotas[i] < limits[i]]
            for i in order:
                if remainder <= 0:
                    break
                quotas[i] += 1
                remainder -= 1
        return quotas

    def __iter__(self) -> Iterator[list[int]]:
        # randomize_order_on_start=False 时 epoch 不参与 seed, 各 epoch 顺序完全相同
        rng_seed = self.seed + self.epoch if self.randomize_order_on_start else self.seed
        rng = random.Random(rng_seed)
        pools = {key: list(value) for key, value in self.bucket_indices.items()}
        for pool in pools.values():
            rng.shuffle(pool)
        offsets = {key: 0 for key in pools}
        steps = self._total_steps()
        bucket_order = self._weighted_bucket_order(rng, steps)

        for step, bucket in enumerate(bucket_order):
            batch_size = self._batch_size_for_bucket(bucket)
            global_batch_size = batch_size * self.world_size
            global_batch = []
            while len(global_batch) < global_batch_size:
                pool, start = pools[bucket], offsets[bucket]
                take = min(global_batch_size - len(global_batch), len(pool) - start)
                global_batch.extend(pool[start:start + take])
                offsets[bucket] += take
                if offsets[bucket] == len(pool):
                    rng.shuffle(pool)
                    offsets[bucket] = 0
            if step < self.start_batch:
                continue
            start = self.rank * batch_size
            yield global_batch[start:start + batch_size]

    def __len__(self) -> int:
        # 返回完整 epoch 的 batch 数 (start_batch 跳过只影响 __iter__, 不影响长度)。
        # 之前减去 start_batch 会让 resume 后 total_steps = max_epochs*(L-K) 随恢复点缩水。
        return self._total_steps()
