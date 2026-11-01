#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ocr_caption.py — Qwen2.5-VL 整图描述: 场景 + 文字风格 (中英双语, 单次生成)。

输出严格 JSON:
  {"scene": "...", "text_style": "...", "scene_zh": "...", "text_style_zh": "..."}
不画 overlay、不做逐区域描述 —— OCR 内容已在标注里, VLM 只补全图语义。
"""
from __future__ import annotations

import torch
from PIL import Image

from qwen_caption import extract_json  # 复用稳健 JSON 提取

_PROMPT = (
    "You are annotating an image that contains visible text, for a text-aware "
    "image generation dataset.\n"
    "Return STRICT JSON only (no markdown, no extra text), with BOTH English and "
    "Chinese versions of each field:\n"
    '{{"scene": "<2-4 sentences in ENGLISH: image type, setting, main objects, '
    'composition, color and lighting>", '
    '"text_style": "<1-3 sentences in ENGLISH describing the visible TEXT: '
    'language(s)/script, font style (handwritten/serif/sans-serif/decorative/neon/'
    'signage...), color, size, and layout/placement>", '
    '"scene_zh": "<scene 描述的中文版本>", '
    '"text_style_zh": "<text_style 的中文版本>"}}\n'
    "Rules: be factual and specific; describe the text's VISUAL STYLE only, do NOT "
    'transcribe the text content; if there is no visible text, set text_style and '
    'text_style_zh to "".'
)

_FIELDS = ("scene", "text_style", "scene_zh", "text_style_zh")


class Qwen25VLCaptioner:
    def __init__(self, model_path: str, dtype: str = "bfloat16",
                 max_new_tokens: int = 640, max_pixels: int = 4194304,
                 language: str = "both"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(
            model_path, max_pixels=max_pixels, min_pixels=3136)
        self.processor.tokenizer.padding_side = "left"  # 生成式 batch 必须左 padding
        torch_dtype = getattr(torch, dtype)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch_dtype, device_map="cuda:0",
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def caption_batch(self, images: list[Image.Image]) -> list[dict]:
        """PIL 列表 → [{'scene','text_style','scene_zh','text_style_zh','raw','ok'}]"""
        batch_msgs = [[{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": _PROMPT},
        ]}] for img in images]
        texts = [self.processor.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True) for m in batch_msgs]
        inputs = self.processor(
            text=texts,
            images=[[m[0]["content"][0]["image"] for m in batch_msgs]],
            padding=True, return_tensors="pt",
        ).to("cuda:0")
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                  do_sample=False)
        gen = out[:, inputs.input_ids.shape[1]:]
        decoded = self.processor.batch_decode(gen, skip_special_tokens=True)

        results = []
        for text in decoded:
            data = extract_json(text)
            if data is None:
                results.append({**{k: None for k in _FIELDS}, "raw": text, "ok": False})
                continue
            fields = {k: str(data.get(k) or "").strip() for k in _FIELDS}
            # ok: 任一语言的 scene 非空即可
            ok = bool(fields["scene"] or fields["scene_zh"])
            results.append({**fields, "raw": text, "ok": ok})
        return results
