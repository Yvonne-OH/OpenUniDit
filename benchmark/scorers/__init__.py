"""benchmark.scorers: 各评测 scorer 注册表."""
from __future__ import annotations

from .base import CKPT_ROOT, Scorer
from .aesthetic import AestheticScorer
from .hpsv2 import HPSv2Scorer
from .image_reward import ImageRewardScorer
from .vlm_judge import VLMJudgeScorer

# name → 类; 每类均为惰性加载, 实例化不开销
SCORER_REGISTRY: dict[str, type] = {
    AestheticScorer.name: AestheticScorer,
    HPSv2Scorer.name: HPSv2Scorer,
    ImageRewardScorer.name: ImageRewardScorer,
    VLMJudgeScorer.name: VLMJudgeScorer,
}


def make_scorer(name: str, **kwargs) -> Scorer:
    """按名称构建 scorer; vlm_judge 支持额外 mode 参数 (geneval|score)."""
    if name not in SCORER_REGISTRY:
        raise KeyError(f"未知 scorer: {name}, 可选 {sorted(SCORER_REGISTRY)}")
    return SCORER_REGISTRY[name](**kwargs)


__all__ = ["CKPT_ROOT", "Scorer", "SCORER_REGISTRY", "make_scorer",
           "AestheticScorer", "HPSv2Scorer", "ImageRewardScorer", "VLMJudgeScorer"]
