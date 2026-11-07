"""ImageReward v1.0 scorer (CPU 可用).

权重: <CKPT_ROOT>/imagereward/{ImageReward.pt, med_config.json}
实现取舍: 官方 pip 包 image-reward 依赖 pytorch-lightning 1.x / 旧 transformers, 与当前环境
(transformers 5.12.1 + torch 2.10) 冲突, 故 vendor 官方源码中最核心的 BLIP 模块
(scorers/_vendor/image_reward_blip/, 改自 THUDM/ImageReward, 仅修 file_utils import 与
tokenizer 本地化), 加载与打分逻辑按官方 ImageReward.py 等价重写.

分数: ImageReward 奖励分 (原始分约 -2 ~ +2, 经 (x-mean)/std 归一到 ~N(0,1) 量纲), 越高越好.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor
from torchvision.transforms import InterpolationMode

from .base import CKPT_ROOT, Scorer

# vendor 的 BLIP 模块作为独立包加载: 把 _vendor 目录加入 sys.path 后
# `import image_reward_blip` (其内部 `from .med import ...` 相对导入可正常解析)
_VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

from image_reward_blip.blip_pretrain import BLIP_Pretrain  # noqa: E402


def _transform(n_px: int = 224):
    """BLIP/CLIP 标准预处理 (与官方 ImageReward.py 一致)."""
    bicubic = InterpolationMode.BICUBIC
    return Compose([
        Resize(n_px, interpolation=bicubic),
        CenterCrop(n_px),
        lambda im: im.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class _RewardHead(nn.Module):
    """官方 ImageReward.py 的 MLP(768) 头: 768→1024→128→64→16→1."""

    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16), nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class ImageRewardScorer(Scorer):
    """ImageReward-v1.0: 人类偏好奖励模型 (paper: arXiv 2304.05977)."""

    name = "image_reward"
    # 官方标定的分数归一化常数
    MEAN = 0.16717362830052426
    STD = 1.0333394966054072

    def load(self):
        model_dir = CKPT_ROOT / "imagereward"
        med_config = model_dir / "med_config.json"
        ckpt = model_dir / "ImageReward.pt"
        if not med_config.is_file() or not ckpt.is_file():
            raise FileNotFoundError(
                f"缺少 ImageReward 权重, 请先下载到 {model_dir} "
                f"(来源: hf-mirror.com/zai-org/ImageReward)")
        # med_config 描述 BLIP 文本侧 BERT (带 cross-attention) 结构
        self.blip = BLIP_Pretrain(image_size=224, vit="large", med_config=str(med_config))
        self.mlp = _RewardHead(768)
        state = torch.load(str(ckpt), map_location="cpu")
        # 只取 ImageReward 用到的子集 (官方 load 即 strict=False; ckpt 里可能含未用键)
        missing, unexpected = self.blip.load_state_dict(
            {k: v for k, v in state.items() if k.startswith("blip.")}, strict=False)
        self.mlp.load_state_dict(
            {k[len("mlp."):]: v for k, v in state.items() if k.startswith("mlp.")})
        self.blip.eval().to(self.device).float()
        self.mlp.eval().to(self.device).float()
        self.preprocess = _transform(224)

    @torch.no_grad()
    def score(self, images, prompts=None):
        """返回 list[float]: ImageReward 分数 (归一化后, 越高越好)."""
        assert prompts is not None and len(prompts) == len(images), \
            "ImageReward 需要 image 与 prompt 一一对应"
        self.ensure_loaded()
        out = []
        for prompt, img in zip(prompts, images):
            text_input = self.blip.tokenizer(
                [prompt], padding="max_length", truncation=True, max_length=35,
                return_tensors="pt").to(self.device)
            pixel = self.preprocess(img).unsqueeze(0).to(self.device)
            image_embeds = self.blip.visual_encoder(pixel)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)
            text_output = self.blip.text_encoder(
                text_input.input_ids, attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts,
                return_dict=True)
            # 取 [CLS] token 的隐状态 → MLP 头 → 官方归一化
            feat = text_output.last_hidden_state[:, 0, :].float()
            reward = self.mlp(feat)
            reward = (reward - self.MEAN) / self.STD
            out.append(reward.item())
        return out
