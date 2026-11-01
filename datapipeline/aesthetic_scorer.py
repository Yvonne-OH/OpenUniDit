#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aesthetic_scorer.py — LAION Aesthetic Predictor V2 批量打分。

头: sac+logos+ava1-l14-linearMSE.pth
    结构 (improved-aesthetic-predictor 官方 MLP, 无激活函数, Dropout 仅训练期生效):
    Linear(768,1024) → Linear(1024,128) → Linear(128,64) → Linear(64,16) → Linear(16,1)
    输入 = L2 归一化的 CLIP 图像 embedding, 输出 ≈ 1-10 美学分。

backbone 两种格式 (config: aesthetic.clip_backbone):
    openclip:ViT-L-14:laion2b_s32b_b82k   官方训练所用 backbone (推荐, 分数在标准量纲)
    hf:/abs/path/to/clip-vit-large-patch14  本地 OpenAI CLIP (embedding 空间不匹配,
                                            分数整体偏低且压缩, 阈值需自行校准)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image


def load_aesthetic_head(head_path: str, device: str = "cuda") -> nn.Module:
    sd = torch.load(head_path, map_location="cpu", weights_only=True)
    sd = {k.replace("layers.", "", 1): v for k, v in sd.items()}
    head = nn.Sequential(
        nn.Linear(768, 1024), nn.Dropout(0.2),
        nn.Linear(1024, 128), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.Dropout(0.2),
        nn.Linear(64, 16), nn.Linear(16, 1),
    )
    head.load_state_dict(sd, strict=True)
    return head.half().to(device).eval()


class AestheticScorer:
    def __init__(self, backbone: str, head_path: str, device: str = "cuda:0"):
        self.device = device
        self.head = load_aesthetic_head(head_path, device)
        if backbone.startswith("openclip:"):
            import open_clip
            _, arch, pretrained = backbone.split(":", 2)
            model, _, pre = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
            self.pretrained_tag = pretrained
            self.model = model.half().to(device).eval()
            self._pre = pre
            self.preprocess = self._openclip_preprocess
        elif backbone.startswith("hf:"):
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
            path = backbone[3:]
            self.model = CLIPVisionModelWithProjection.from_pretrained(
                path, dtype=torch.float16).to(device).eval()
            self._proc = CLIPImageProcessor.from_pretrained(path)
            self.pretrained_tag = "hf:" + path.rsplit("/", 1)[-1]
            self.preprocess = self._hf_preprocess
        else:
            raise ValueError(f"未知 backbone 格式: {backbone}")

    def _openclip_preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        return torch.stack([self._pre(im) for im in images])

    def _hf_preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        return self._proc(images=images, return_tensors="pt")["pixel_values"]

    @torch.inference_mode()
    def score_batch(self, images: list[Image.Image]) -> list[float]:
        """PIL 列表 → 美学分列表 (失败项 NaN)。"""
        pixel = self.preprocess(images).half().to(self.device)
        if hasattr(self.model, "encode_image"):            # openclip
            emb = self.model.encode_image(pixel).float()
        else:                                              # transformers CLIP
            emb = self.model(pixel_values=pixel).image_embeds.float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        scores = self.head(emb.half()).squeeze(-1).float().cpu()
        return [float(s) for s in scores]
