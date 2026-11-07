"""评测 scorer 统一接口.

所有 scorer 实现:
    score(images: list[PIL.Image.Image], prompts: list[str] | None) -> list[float]
返回与 images 等长的分数列表 (越高越好, 具体量纲见各类 docstring).

带结构化输出的 scorer (如 vlm_judge) 额外提供:
    score_detailed(...) -> list[dict]
"""
from __future__ import annotations

import os
from pathlib import Path

# 评测 ckpt 根目录 (可用环境变量 BENCH_CKPT_ROOT 覆盖), 子目录约定见 benchmark/README.md
# base.py 位于 <user_root>/code/model_exp/benchmark/scorers/, parents[4] = <user_root>
CKPT_ROOT = Path(os.environ.get(
    "BENCH_CKPT_ROOT",
    Path(__file__).resolve().parents[4] / "ckpt",
))


class Scorer:
    """scorer 基类: 惰性加载, 首次 score() 时初始化模型."""

    name: str = "base"

    def __init__(self, device: str = "cpu", **kwargs):
        self.device = device
        self.kwargs = kwargs
        self._loaded = False

    def load(self):
        """加载模型权重 (子类实现, 可重入)."""
        raise NotImplementedError

    def ensure_loaded(self):
        if not self._loaded:
            self.load()
            self._loaded = True

    def score(self, images, prompts=None):
        """images: list[PIL.Image]; prompts: 与 images 等长或 None. 返回 list[float]."""
        raise NotImplementedError

    def score_detailed(self, images, prompts=None):
        """带结构化输出的 scorer 覆写此方法; 缺省退化为 score()."""
        values = self.score(images, prompts)
        return [{"score": v} for v in values]
