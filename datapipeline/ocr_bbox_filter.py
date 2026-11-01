#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ocr_bbox_filter.py — OCR 文字框过滤 (xyxy 坐标)。

(1) 小框过滤:   w/h < min_box_side 或面积 < min_box_area 的框丢弃 (噪声/不可读)。
(2) 重合过滤:   两框 IoU > overlap_iou 记为重合; 重合框占比 > max_overlap_ratio 整图丢弃。
    空文本 ('' 或纯空白) 的框同步丢弃。
返回 (kept_boxes, fail_reasons, stats)。
"""
from __future__ import annotations


def _iou(a: list, b: list) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def filter_ocr_boxes(
    texts: list[str],
    bboxes: list[list[float]],
    min_box_side: float = 12.0,
    min_box_area: float = 400.0,
    overlap_iou: float = 0.7,
    max_overlap_ratio: float = 0.3,
) -> tuple[list[dict], list[str], dict]:
    """→ (kept: [{'bbox','text'}], fails: 整图丢弃原因, stats)"""
    stats = {"n_input": len(bboxes), "n_empty_text": 0, "n_small": 0}
    kept = []
    for text, bb in zip(texts, bboxes):
        if not text or not str(text).strip():
            stats["n_empty_text"] += 1
            continue
        x0, y0, x1, y1 = bb
        w, h = x1 - x0, y1 - y0
        if w < min_box_side or h < min_box_side or w * h < min_box_area:
            stats["n_small"] += 1
            continue
        kept.append({"bbox": [float(x0), float(y0), float(x1), float(y1)],
                     "text": str(text).strip()})
    stats["n_kept"] = len(kept)

    fails = []
    if not kept:
        fails.append("no_valid_boxes")
        return kept, fails, stats

    if max_overlap_ratio > 0:  # 重合检测 (kept 已很小, O(n²) 可忽略)
        overlapped = set()
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                if _iou(kept[i]["bbox"], kept[j]["bbox"]) > overlap_iou:
                    overlapped.update((i, j))
        stats["n_overlapped"] = len(overlapped)
        ratio = len(overlapped) / len(kept)
        stats["overlap_ratio"] = round(ratio, 4)
        if ratio > max_overlap_ratio:
            fails.append(f"overlap_ratio>{max_overlap_ratio}")
    return kept, fails, stats
