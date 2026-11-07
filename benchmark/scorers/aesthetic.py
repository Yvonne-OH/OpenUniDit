"""LAION aesthetic predictor scorer (CPU 可用).

权重:
- 底座 CLIP ViT-L/14 (OpenAI 权重, HF 格式): <CKPT_ROOT>/clip-vit-large-patch14/
- 线性头 MLP: <CKPT_ROOT>/aesthetic/sac+logos+ava1-l14-linearMSE.pth
  (即社区常说的 sac_logos_v2, LAION improved-aesthetic-predictor v2; 两者为同一 MLP 的不同格式)

实现取舍: 不依赖 openai/clip 官方包, 用 transformers 的 CLIPVisionModel 加载本地权重,
图像 embedding 归一化后送入 MLP 头, 与 ImageReward 官方 AestheticScore 数值路径一致.

分数: 1 ~ 10 (美学评分, 越高越好).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from torchvision.transforms import InterpolationMode
from transformers import CLIPModel

from .base import CKPT_ROOT, Scorer

# 与 improved-aesthetic-predictor 相同的 MLP 结构 (键名 layers.0/2/4/6/7)
class _AestheticMLP(nn.Module):
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


class AestheticScorer(Scorer):
    """LAION aesthetic predictor v2 (sac_logos_v2)."""

    name = "aesthetic"

    def load(self):
        clip_dir = CKPT_ROOT / "clip-vit-large-patch14"
        head_path = CKPT_ROOT / "aesthetic" / "sac+logos+ava1-l14-linearMSE.pth"
        if not clip_dir.is_dir():
            raise FileNotFoundError(f"缺少 CLIP 权重: {clip_dir}")
        if not head_path.is_file():
            raise FileNotFoundError(f"缺少 aesthetic 头: {head_path}")
        # OpenAI CLIP ViT-L/14 (HF openai/clip-vit-large-patch14 转换格式);
        # 用 CLIPModel.get_image_features 取 visual_projection 后的 768 维 embedding
        # (与 openai CLIP encode_image 输出一致; pooler_output 是未投影的 1024 维)
        self.clip = CLIPModel.from_pretrained(str(clip_dir)).eval()
        self.mlp = _AestheticMLP(768)
        state = torch.load(str(head_path), map_location="cpu")
        self.mlp.load_state_dict(state)
        self.clip.to(self.device).float()
        self.mlp.to(self.device).float().eval()
        bicubic = InterpolationMode.BICUBIC
        self.preprocess = Compose([
            Resize(224, interpolation=bicubic),
            CenterCrop(224),
            lambda im: im.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    @torch.no_grad()
    def score(self, images, prompts=None):
        """返回 list[float]: 美学评分 (1-10, 越高越好). prompt 不使用."""
        self.ensure_loaded()
        pixel = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        feats = self.clip.get_image_features(pixel)             # [B, 768] projection 后 embedding
        if not torch.is_tensor(feats):  # transformers 5.x 返回 ModelOutput (pooler_output 已投影 768 维)
            feats = feats.pooler_output
        feats = F.normalize(feats.float(), dim=-1)             # 归一化后输入 MLP (与官方一致)
        ratings = self.mlp(feats).squeeze(-1)
        return [float(x) for x in ratings.cpu()]
