#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bbox_filter.py — bbox 过滤: conf 阈值 + 类无关 IoU NMS。

conf 过滤: 标注含数值 score 字段且 < conf_min 时滤除 (LVIS GT 无 score → 全保留)。
NMS: 按面积降序贪心, IoU > 阈值者抑制 (面积大者视为更完整, 保留)。
"""
from __future__ import annotations

from typing import Any


def iou_xywh(a: list, b: list) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def filter_annotations(
    annotations: list[dict[str, Any]],
    conf_key: str = "score",
    conf_min: float = 0.0,
    iou_threshold: float = 0.7,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """返回 (保留的标注列表(保持原相对顺序), 过滤统计)。"""
    stats = {"n_input": len(annotations), "n_conf_dropped": 0, "n_nms_dropped": 0}

    # 1) conf 过滤
    kept_conf = []
    for ann in annotations:
        score = ann.get(conf_key)
        if isinstance(score, (int, float)) and score < conf_min:
            stats["n_conf_dropped"] += 1
        else:
            kept_conf.append(ann)

    # 2) 类无关 NMS (面积降序贪心)
    order = sorted(range(len(kept_conf)), key=lambda i: -kept_conf[i]["area"])
    suppressed = set()
    for ii, i in enumerate(order):
        if i in suppressed:
            continue
        for j in order[ii + 1:]:
            if j in suppressed:
                continue
            if iou_xywh(kept_conf[i]["bbox"], kept_conf[j]["bbox"]) > iou_threshold:
                suppressed.add(j)
    kept = [ann for i, ann in enumerate(kept_conf) if i not in suppressed]
    stats["n_nms_dropped"] = len(kept_conf) - len(kept)
    stats["n_kept"] = len(kept)
    return kept, stats
