"""VLM judge scorer: 基于本地 Qwen3-VL-4B-Instruct 的视觉判定/打分 (CPU 可用, 较慢).

模型: <text_encoder 路径> (configs/stage_1.yaml components.text_encoder),
      缺省 /mnt/nfs/models/huggingface/Qwen/Qwen3-VL-4B-Instruct, 强制 local_files_only.

两种 judge 模式:
- mode="geneval": GenEval 协议判定. 对每条 prompt 按 meta(tag/include) 生成判定清单
  (对象存在/计数/颜色/位置), VLM 输出 JSON verdict, score() = 通过率 [0,1];
  score_detailed() 额外返回 checks 与各维度结果, 供 run.py 汇总各维度正确率.
- mode="score": 多维度 1-10 打分 (alignment 文图对齐 / aesthetics 美学 / detail 细节),
  用于 heim 与 qwen_image_bench; score() = 三维均值, score_detailed() 返回各维分数.
"""
from __future__ import annotations

import json
import os
import re

import torch
from PIL import Image

from .base import Scorer

DEFAULT_MODEL_PATH = "/mnt/nfs/models/huggingface/Qwen/Qwen3-VL-4B-Instruct"

# GenEval tag → 判定维度构造方式
_GENEVAL_TAGS = {"single_object", "two_objects", "counting", "colors", "position",
                 "color_attr", "attributes", "complex"}


def _extract_json(text: str) -> dict:
    """从模型输出中提取第一个 JSON 对象 (容忍 markdown 代码块/前后缀文本)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"VLM 输出中没有 JSON: {text[:200]!r}")
    return json.loads(match.group(0))


def _build_geneval_checks(prompt: str, meta: dict) -> list[dict]:
    """按 GenEval 元数据构造判定清单.

    include 元素形如 {"class": "bench", "count": 1, "color": "red"?}; tag 决定关注维度:
    existence 恒查; count 查计数; color 查颜色; position 用 prompt 原文的方位关系让 VLM 判定.
    """
    checks = []
    tag = meta.get("tag", "")
    for obj in meta.get("include", []):
        cls, count = obj.get("class"), int(obj.get("count", 1))
        checks.append({"type": "existence", "expect": cls,
                       "question": f"Does the image clearly contain '{cls}'? Answer true or false."})
        if tag in ("counting", "complex"):
            checks.append({"type": "count", "expect": f"{cls} x{count}",
                           "question": f"How many '{cls}' are in the image? Is the count exactly {count}?"})
        color = obj.get("color")
        if color or tag in ("colors", "color_attr"):
            color = color or re.search(rf"\b(red|orange|yellow|green|blue|purple|pink|brown|black|white|gray|grey)\b\s+\w*\s*{cls}", prompt)
            if color:
                color = color if isinstance(color, str) else color.group(1)
                checks.append({"type": "color", "expect": f"{color} {cls}",
                               "question": f"Is the '{cls}' in the image {color} (or a plausible shade of it)?"})
    if tag == "position":
        checks.append({"type": "position", "expect": prompt,
                       "question": "Describe the relative positions of the main objects. Does it match "
                                   f"this description: \"{prompt}\"? Answer true or false."})
    if not checks:  # 兜底: complex/未知 tag 至少判一次整体对齐
        checks.append({"type": "existence", "expect": prompt,
                       "question": f"Does the image match the request \"{prompt}\"? Answer true or false."})
    return checks


class VLMJudgeScorer(Scorer):
    """Qwen3-VL 本地判定器. mode: "geneval" | "score"."""

    name = "vlm_judge"

    def __init__(self, device: str = "cpu", mode: str = "score",
                 model_path: str | None = None, max_new_tokens: int = 256, **kwargs):
        super().__init__(device=device, **kwargs)
        self.mode = mode
        self.model_path = model_path or os.environ.get("BENCH_VLM_PATH", DEFAULT_MODEL_PATH)
        self.max_new_tokens = max_new_tokens

    def load(self):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"本地 VLM 路径不存在: {self.model_path}")
        dtype = torch.bfloat16  # CPU 上 bf16 省内存 (4B ≈ 8GB), 数值对判定足够
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, dtype=dtype, local_files_only=True).eval().to(self.device)

    # ── 底层: 单次图文对话 ────────────────────────────────────────────────
    @torch.no_grad()
    def _ask(self, question: str, image: Image.Image) -> str:
        """对单张图提问, 返回模型文本输出."""
        self.ensure_loaded()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.device)
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    # ── geneval judge ────────────────────────────────────────────────────
    def judge_geneval(self, image: Image.Image, prompt: str, meta: dict) -> dict:
        """返回 {"score": pass_rate, "checks": [...]}; checks 每项含 type/pass/reason."""
        checks = _build_geneval_checks(prompt, meta or {})
        question = (
            "You are a strict image evaluator. Answer with ONLY a JSON object, no extra text.\n"
            "For each question in the list, output its index, a boolean 'pass', and a short 'reason'.\n"
            "Format: {\"checks\": [{\"index\": 0, \"pass\": true, \"reason\": \"...\"}, ...]}\n\n"
            "Questions:\n" + "\n".join(f"{i}. {c['question']}" for i, c in enumerate(checks))
        )
        try:
            data = _extract_json(self._ask(question, image))
            raw = data.get("checks", [])
        except (ValueError, json.JSONDecodeError) as exc:
            return {"score": 0.0, "checks": [{"type": "parse_error", "pass": False, "reason": str(exc)}]}
        results = []
        for i, c in enumerate(checks):
            verdict = raw[i] if i < len(raw) else {}
            results.append({
                "type": c["type"], "expect": c["expect"],
                "pass": bool(verdict.get("pass", False)),
                "reason": str(verdict.get("reason", ""))[:200],
            })
        passed = sum(1 for r in results if r["pass"])
        return {"score": passed / len(results) if results else 0.0, "checks": results}

    # ── score judge (heim / qwen_image_bench) ────────────────────────────
    def judge_score(self, image: Image.Image, prompt: str) -> dict:
        """返回 {"alignment": a, "aesthetics": b, "detail": c} (1-10)."""
        question = (
            "You are an expert text-to-image evaluator. Rate the image on three dimensions, "
            "each an integer from 1 to 10:\n"
            "- alignment: how well the image matches the prompt\n"
            "- aesthetics: overall visual beauty and composition\n"
            "- detail: clarity, realism and richness of details\n\n"
            f"Prompt: \"{prompt}\"\n\n"
            "Answer with ONLY a JSON object: "
            "{\"alignment\": <1-10>, \"aesthetics\": <1-10>, \"detail\": <1-10>}"
        )
        try:
            data = _extract_json(self._ask(question, image))
            scores = {k: float(data[k]) for k in ("alignment", "aesthetics", "detail")}
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            scores = {"alignment": 0.0, "aesthetics": 0.0, "detail": 0.0, "error": str(exc)[:200]}
        return scores

    # ── 统一接口 ─────────────────────────────────────────────────────────
    def score(self, images, prompts=None, metas=None):
        if self.mode == "geneval":
            assert metas is not None, "geneval judge 需要 prompts jsonl 的 meta"
            return [self.judge_geneval(im, p, m)["score"]
                    for im, p, m in zip(images, prompts, metas)]
        return [sum(v for k, v in self.judge_score(im, p).items()
                    if k in ("alignment", "aesthetics", "detail")) / 3.0
                for im, p in zip(images, prompts)]

    def score_detailed(self, images, prompts=None, metas=None):
        if self.mode == "geneval":
            return [self.judge_geneval(im, p, m) for im, p, m in zip(images, prompts, metas)]
        return [self.judge_score(im, p) for im, p in zip(images, prompts)]
