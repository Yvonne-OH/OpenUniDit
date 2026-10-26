#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quality_filter.py — 画质过滤 (无重依赖: numpy 实现 Laplacian/亮度/对比度)。

指标:
  blur_var    3x3 Laplacian 方差 (清晰度, 低=模糊)
  brightness  灰度均值 (过暗/过曝)
  contrast    灰度标准差 (低对比)
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# 3x3 Laplacian 核
_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def _laplacian_var(gray: np.ndarray) -> float:
    # 边界用零填充, 手动卷积 (图像尺寸 ~640, 开销可忽略)
    p = np.pad(gray.astype(np.float32), 1, mode="constant")
    h, w = gray.shape
    win = np.stack([
        p[0:h, 1:w + 1], p[1:h + 1, 0:w], p[1:h + 1, 1:w + 1],
        p[1:h + 1, 2:w + 2], p[2:h + 2, 1:w + 1],
    ])
    lap = win.sum(axis=0) - 4 * p[1:h + 1, 1:w + 1]
    return float(lap.var())


def quality_check(img: Image.Image, cfg: dict) -> dict:
    """返回 {'pass': bool, 指标, 'fails': [原因]}。cfg 为 quality_filter 段。"""
    w, h = img.size
    fails = []
    if min(w, h) < cfg["min_side"]:
        fails.append(f"min_side<{cfg['min_side']}")
    if max(w, h) > cfg["max_side"]:
        fails.append(f"max_side>{cfg['max_side']}")

    gray = np.asarray(img.convert("L"), dtype=np.float32)
    blur_var = _laplacian_var(gray)
    brightness = float(gray.mean())
    contrast = float(gray.std())

    if blur_var < cfg["blur_var_min"]:
        fails.append(f"blur<{cfg['blur_var_min']}")
    if brightness < cfg["brightness_min"]:
        fails.append(f"dark<{cfg['brightness_min']}")
    if brightness > cfg["brightness_max"]:
        fails.append(f"bright>{cfg['brightness_max']}")
    if contrast < cfg["contrast_std_min"]:
        fails.append(f"low_contrast<{cfg['contrast_std_min']}")

    return {
        "pass": len(fails) == 0,
        "width": w, "height": h,
        "blur_var": round(blur_var, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "fails": fails,
    }
