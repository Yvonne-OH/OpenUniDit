"""HPS v2 scorer (CPU 可用).

权重: <CKPT_ROOT>/hpsv2/HPS_v2_compressed.pt (v2.0 compressed, 含 CLIP ViT-H/14 全部权重 + 偏好头)
来源: HF space xswu/HPSv2 (官方发布渠道, 与 pip 包 hpsv2 的 hf_hub_download 目标一致)

实现取舍: pip 包 hpsv2 依赖从 git 安装的 openai/CLIP + pytorch_lightning 1.x, 与当前环境冲突;
但其核心只是 open_clip 的 ViT-H-14 结构 + 官方 compressed checkpoint. 故直接使用标准
open_clip_torch (纯 torch 生态, 不影响 transformers) 构建模型并加载官方权重.

分数: 原始 logits (约 0 ~ 0.35, 越高越好; 官方 benchmark 也直接用该原始值).
"""
from __future__ import annotations

import torch
from PIL import Image

from .base import CKPT_ROOT, Scorer


class HPSv2Scorer(Scorer):
    """HPS v2 (Human Preference Score v2, paper: arXiv 2306.09341)."""

    name = "hpsv2"
    MODEL_NAME = "ViT-H-14"
    CKPT_NAME = "HPS_v2_compressed.pt"   # v2.0; v2.1 换成 HPS_v2.1_compressed.pt

    def load(self):
        import open_clip
        ckpt_path = CKPT_ROOT / "hpsv2" / self.CKPT_NAME
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"缺少 HPS 权重: {ckpt_path} (来源: hf-mirror.com/spaces/xswu/HPSv2)")
        # 与官方 hpsv2/img_score.py 相同的结构参数; pretrained=None 仅构建结构,
        # 权重随后由 compressed checkpoint 完整覆盖 (其内含 CLIP + 偏好头)
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            self.MODEL_NAME, pretrained=None, precision="fp32", device="cpu",
            jit=False, force_quick_gelu=False, output_dict=True,
        )
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        self.model = model.eval().to(self.device).float()
        self.preprocess = preprocess_val
        # tokenizer: open_clip 内置 CLIP BPE; laion ViT-H-14 词表与 openai CLIP 相同
        self.tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)

    @torch.no_grad()
    def score(self, images, prompts=None):
        """返回 list[float]: HPS 原始分数 (越高越好)."""
        assert prompts is not None and len(prompts) == len(images), \
            "HPSv2 需要 image 与 prompt 一一对应"
        self.ensure_loaded()
        pixel = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        text = self.tokenizer(prompts).to(self.device)
        out = self.model(pixel, text)
        image_features = out["image_features"] if isinstance(out, dict) else out[0]
        text_features = out["text_features"] if isinstance(out, dict) else out[1]
        # 官方 img_score.py: image_features @ text_features.T 的对角线 (逐对相似度)
        logits = image_features @ text_features.T
        scores = logits.diag()
        return [float(x) for x in scores.cpu()]
