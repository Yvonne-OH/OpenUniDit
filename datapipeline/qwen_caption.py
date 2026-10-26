#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qwen_caption.py — Qwen3-VL 结构化 caption。

做法: 在原图上绘制编号框 (R0, R1, ...), 让 VLM 输出严格 JSON:
  {"scene": "<全图描述>", "regions": [{"id": 0, "description": "..."}, ...]}
再与过滤后标注的 bbox/segmentation/category 合并成最终 record。

密集图保护: region 数 > max_regions 时仅描述面积前 N 大的 (overlay 仍画全部框,
未描述的 region description 置 null, bbox/mask 坐标不丢失)。
"""
from __future__ import annotations

import io
import json
import re

import torch
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

PALETTE = [
    (230, 57, 70), (29, 53, 87), (42, 157, 143), (231, 111, 81),
    (0, 121, 140), (255, 183, 3), (106, 76, 147), (188, 71, 73),
    (60, 98, 155), (244, 162, 97), (38, 70, 83), (168, 218, 220),
]

_PROMPT_EN = (
    "You are annotating an image for a detection dataset. "
    "The image contains {n} numbered regions, each marked by a colored rectangle labeled R0..R{n_mm1}.\n"
    "Return STRICT JSON only (no markdown, no extra text):\n"
    '{{"scene": "<2-4 sentence description of the overall image: setting, main objects, action, style>", '
    '"regions": [{{"id": <int>, "description": "<what OBJECT is inside this region: category, attributes, state, spatial relation to the scene>"}}]}}\n'
    "Rules: cover every region id {ids_hint} exactly once; do NOT invent or merge regions; "
    "be factual and specific; describe the CONTENT inside each rectangle, NEVER mention the "
    "rectangle, its color, or the R-label itself.{terse}"
)
_TERSE = "\nIMPORTANT: the description of each region must be AT MOST 12 words."

_PROMPT_EN_NOREGION = (
    "Describe this image for a captioning dataset. Return STRICT JSON only (no markdown): "
    '{"scene": "<2-4 sentence description: setting, main objects, action, style>"}'
)


def build_overlay(img: Image.Image, annotations: list[dict], max_regions: int) -> Image.Image:
    """绘制编号框 overlay; 返回 (overlay 图, 被选中描述的 region 下标列表)。"""
    base = img.convert("RGB").copy()
    draw = ImageDraw.Draw(base)
    lw = max(2, min(base.size) // 320)
    f = ImageFont.truetype(FONT, max(14, min(base.size) // 36)) if _font_ok() else ImageFont.load_default()

    # 密集图: 只对面积前 max_regions 大的做逐区域描述 (框全画)
    order = sorted(range(len(annotations)), key=lambda i: -annotations[i]["area"])
    described = set(order[:max_regions])

    for i, ann in enumerate(annotations):
        color = PALETTE[i % len(PALETTE)]
        x, y, w, h = ann["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=lw)
        label = f"R{i}"
        ty = y - f.size - 2 if y - f.size - 2 > 0 else y + 2
        tw = f.getlength(label) + 6
        draw.rectangle([x, ty, x + tw, ty + f.size + 2], fill=color)
        draw.text((x + 3, ty), label, fill="white", font=f)
    return base, sorted(described)


def _font_ok() -> bool:
    import os
    return os.path.isfile(FONT)


def extract_json(text: str) -> dict | None:
    """从模型输出中稳健提取 JSON (容忍 code fence / 前后噪声)。"""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    raw = m.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return json.loads(raw.replace("\n", " ").replace(",}", "}").replace(",]", "]"))
        except json.JSONDecodeError:
            return None


class QwenCaptioner:
    def __init__(self, model_path: str, dtype: str = "bfloat16",
                 max_new_tokens: int = 768, language: str = "en",
                 max_regions: int = 64):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"  # 生成式 batch 必须左 padding
        torch_dtype = getattr(torch, dtype)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch_dtype, device_map="cuda:0",
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.language = language
        self.max_regions = max_regions

    def _make_prompt(self, n_regions: int) -> str:
        if self.language == "zh":
            raise NotImplementedError("zh prompt 待补充; 默认 en")
        if n_regions == 0:
            return _PROMPT_EN_NOREGION
        ids_hint = "0.." + str(n_regions - 1)
        terse = _TERSE if n_regions > 24 else ""
        return _PROMPT_EN.format(n=n_regions, n_mm1=n_regions - 1, ids_hint=ids_hint, terse=terse)

    def _token_budget(self, n_regions: int) -> int:
        """动态生成预算: 密集图按 region 数放大, 封顶 3072。"""
        m = min(n_regions, self.max_regions)
        return min(3072, max(self.max_new_tokens, 160 + 20 * m))

    @torch.inference_mode()
    def caption_batch(self, items: list[dict]) -> list[dict]:
        """items: [{'img': PIL.Image, 'annotations': [...]}] → [{'scene', 'regions', 'raw', 'ok'}]"""
        batch_msgs, metas = [], []
        for it in items:
            overlay, described = build_overlay(it["img"], it["annotations"], self.max_regions)
            n = len(it["annotations"])
            prompt = self._make_prompt(n)
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": overlay},
                {"type": "text", "text": prompt},
            ]}]
            batch_msgs.append(msgs)
            metas.append((n, described))

        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in batch_msgs]
        inputs = self.processor(
            text=texts, images=[[b[0]["content"][0]["image"] for b in batch_msgs]],
            padding=True, return_tensors="pt",
        ).to("cuda:0")
        # 按 batch 内最大 region 数给足生成预算, 避免密集图 JSON 截断
        budget = max(self._token_budget(n) for n, _ in metas)
        out = self.model.generate(
            **inputs, max_new_tokens=budget, do_sample=False,
        )
        gen = out[:, inputs.input_ids.shape[1]:]
        decoded = self.processor.batch_decode(gen, skip_special_tokens=True)

        results = []
        for text, (n, described) in zip(decoded, metas):
            data = extract_json(text)
            if data is None:
                results.append({"scene": None, "regions": [], "raw": text, "ok": False})
                continue
            scene = data.get("scene")
            rmap = {}
            for r in data.get("regions") or []:
                if isinstance(r, dict) and isinstance(r.get("id"), int):
                    rmap[r["id"]] = str(r.get("description", "")).strip()
            # 组装: 未描述的 region → None (坐标仍保留)
            described_set = set(described)
            regions = [
                {"id": i, "description": (rmap.get(i) if (n == 0 or i in described_set) else None)}
                for i in range(n)
            ]
            missing = [r["id"] for r in regions
                       if r["description"] is None and (n <= self.max_regions or r["id"] in described_set)]
            results.append({"scene": scene, "regions": regions, "raw": text,
                            "ok": len(missing) == 0, "missing": missing})
        return results


def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")
