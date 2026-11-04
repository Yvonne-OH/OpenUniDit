"""图像质量指标计算。

CPU (cv2): 清晰度 (Laplacian/Tenengrad)、曝光 (过曝/欠曝比例)、色彩饱和度、
colorfulness、JPEG 块效应 (8x8 块边界能量比, 压缩伪影近似)。
GPU (批量): LAION aesthetic (open_clip ViT-L/14 + sac+logos+ava 线性头)、
CLIPScore (transformers CLIP, 2.5 × cosine, 对齐 image_eval_scorers 约定)。

水印/Logo 检测暂无本地模型, 未纳入; NIMA/BRISQUE 由 LAION aesthetic 代替。
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DEFAULT_CLIP_DIR = "/home/jovyan/pfs/users/zhanqian.wu/ckpt/clip-vit-large-patch14"
DEFAULT_AES_HEAD = "/home/jovyan/pfs/users/zhanqian.wu/ckpt/laion-aesthetic-heads/sac+logos+ava1-l14-linearMSE.pth"
DEFAULT_OPENCLIP_DIR = "/home/jovyan/pfs/users/zhanqian.wu/ckpt/CLIP-ViT-L-14-laion2B-s32B-b82K"


METRICS_MAX_SIDE = 1024  # 清晰度/曝光/色彩在降采样图上计算 (吞吐优化, 阈值为该尺度下的经验值)


def cv_metrics(img_bytes: bytes) -> dict | None:
    """单图 CPU 指标; 解码失败返回 None。

    laplacian/tenengrad 等在 METRICS_MAX_SIDE 降采样图上计算 (数值与全分辨率
    不同, 但单调性一致, 过滤阈值按此尺度标定); jpeg_block_score 需要 8px 块
    网格, 在原图灰度上计算。
    """
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # JPEG 块效应: 8px 块边界列的梯度能量 / 非边界列 (比值高 ⇒ 压缩伪影明显)
    dx = np.abs(np.diff(gray.astype(np.float32), axis=1))
    boundary = np.zeros(dx.shape[1], dtype=bool)
    boundary[7::8] = True
    if (~boundary).sum() > 0 and boundary.sum() > 0:
        jpeg_block_score = float(dx[:, boundary].mean() /
                                 max(dx[:, ~boundary].mean(), 1e-6) - 1.0)
    else:
        jpeg_block_score = 0.0

    # 其余指标在降采样图上算 (清晰度等变换是 O(像素), 4K 图 25x 提速)
    h, w = gray.shape
    scale = METRICS_MAX_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    small_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if scale < 1.0 else gray

    laplacian_var = float(cv2.Laplacian(small_gray, cv2.CV_32F).var())
    sobel_x = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sobel_x ** 2 + sobel_y ** 2))

    hist = np.bincount(small_gray.ravel(), minlength=256).astype(np.float64)
    n_pixels = hist.sum()
    overexposed_ratio = float(hist[251:].sum() / n_pixels)
    underexposed_ratio = float(hist[:6].sum() / n_pixels)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation_mean = float(hsv[:, :, 1].mean() / 255.0)
    brightness_mean = float(small_gray.mean() / 255.0)

    # Hasler-Süsstrunk colorfulness
    rgb = img[:, :, ::-1].astype(np.float32)
    rg = rgb[:, :, 0] - rgb[:, :, 1]
    yb = 0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2]
    colorfulness = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))

    return {
        "laplacian_var": round(laplacian_var, 2),
        "tenengrad": round(tenengrad, 2),
        "jpeg_block_score": round(jpeg_block_score, 4),
        "brightness_mean": round(brightness_mean, 4),
        "overexposed_ratio": round(overexposed_ratio, 4),
        "underexposed_ratio": round(underexposed_ratio, 4),
        "saturation_mean": round(saturation_mean, 4),
        "colorfulness": round(colorfulness, 2),
    }


def _decode_pil(img_bytes: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


class AestheticScorer:
    """LAION aesthetic: open_clip ViT-L/14 图像嵌入 + 线性头 (量程 ~1-10)。"""

    def __init__(self, openclip_dir: str = DEFAULT_OPENCLIP_DIR,
                 head_path: str = DEFAULT_AES_HEAD, device: str = "cuda:0",
                 batch_size: int = 32):
        import open_clip
        import torch

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=str(Path(openclip_dir) / "open_clip_pytorch_model.bin"))
        self.model = self.model.to(device).eval()
        # LAION improved-aesthetic-predictor MLP (sac+logos+ava, state_dict 加载)
        head = torch.nn.Sequential(
            torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16), torch.nn.Linear(16, 1))
        sd = torch.load(head_path, map_location=device, weights_only=True)
        head.load_state_dict({k.removeprefix("layers."): v for k, v in sd.items()})
        self.head = head.to(device).eval()

    @torch.no_grad()
    def scores(self, images: list[Image.Image]) -> list[float | None]:
        out: list[float | None] = []
        for i in range(0, len(images), self.batch_size):
            batch = [img for img in images[i:i + self.batch_size] if img is not None]
            if not batch:
                out.extend([None] * len(images[i:i + self.batch_size]))
                continue
            x = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
            emb = self.model.encode_image(x).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            s = self.head(emb).squeeze(-1).float().cpu().tolist()
            # 有效图数与批量不一致时 (含 None), 按原位回填
            batch_scores = iter(s)
            out.extend(next(batch_scores) if img is not None else None
                       for img in images[i:i + self.batch_size])
        return out


class CLIPScoreScorer:
    """CLIPScore = 2.5 × cos(image, caption), ViT-L/14 (对齐 Hessel et al. 2021)。"""

    def __init__(self, clip_dir: str = DEFAULT_CLIP_DIR, device: str = "cuda:0",
                 batch_size: int = 32):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.model = CLIPModel.from_pretrained(clip_dir).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(clip_dir)

    @torch.no_grad()
    def scores(self, images: list[Image.Image | None],
               captions: list[str]) -> list[float | None]:
        out: list[float | None] = []
        for i in range(0, len(images), self.batch_size):
            b_imgs = images[i:i + self.batch_size]
            b_caps = captions[i:i + self.batch_size]
            valid = [j for j, img in enumerate(b_imgs) if img is not None]
            res: list[float | None] = [None] * len(b_imgs)
            if valid:
                try:
                    inputs = self.processor(
                        text=[b_caps[j] for j in valid], images=[b_imgs[j] for j in valid],
                        return_tensors="pt", padding=True, truncation=True).to(self.device)
                    outputs = self.model(**inputs)
                    cos = (outputs.image_embeds.float() @ outputs.text_embeds.float().T) \
                        .diagonal().cpu()
                    for j, c in zip(valid, cos.tolist()):
                        res[j] = round(2.5 * c, 4)
                except Exception as exc:  # 单批失败不拖垮整体
                    print(f"⚠️ [clip_score] 批内推理失败: {exc}", flush=True)
            out.extend(res)
        return out
