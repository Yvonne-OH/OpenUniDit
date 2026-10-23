# -*- coding: utf-8 -*-
"""
VLM 评价（走 OpenAI 兼容 API）。

配置方式（优先级：命令行参数 > 环境变量）：
    --vlm-base   / VLM_API_BASE    API base url，如 https://api.example.com/v1
    --vlm-key    / VLM_API_KEY     API key
    --vlm-model  / VLM_MODEL       模型名，如 qwen-vl-max / glm-4v / gpt-4o-mini ...

输出：每张图一个 dict：
    {"aesthetics": 1-10, "technical_quality": 1-10,
     "prompt_alignment": 1-10|null, "overall": 1-10, "comment": "..."}
解析失败时保留 {"raw": "<模型原始返回>"}。
"""
import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

VL_SYSTEM_PROMPT = (
    "你是严格的图像质量评审。请仅输出一个 JSON 对象，禁止输出任何其他文字或代码块标记。"
    "JSON 字段：\n"
    '{"aesthetics": <1-10 整数, 画面美学：构图/光影/色彩/氛围>,\n'
    ' "technical_quality": <1-10 整数, 技术质量：清晰度/伪影/畸变/噪点/结构合理性>,\n'
    ' "prompt_alignment": <1-10 整数或 null, 与给定 prompt 的语义一致性；未给 prompt 时必须为 null>,\n'
    ' "overall": <1-10 整数, 综合评价>,\n'
    ' "comment": "<不超过 30 字的中文简评>"}'
)


class VLMScorer:
    name = "vlm"
    needs_prompt = False  # 无 prompt 也能评（prompt_alignment 为 null）

    def __init__(self, base_url=None, api_key=None, model=None,
                 workers: int = 8, max_retries: int = 4, timeout: int = 120,
                 max_side: int = 768, jpeg_quality: int = 88):
        self.base_url = base_url or os.environ.get("VLM_API_BASE") \
            or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("VLM_API_KEY") \
            or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("VLM_MODEL")
        self.workers = workers
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_side = max_side
        self.jpeg_quality = jpeg_quality
        if not (self.base_url and self.api_key and self.model):
            raise ValueError(
                "VLM API 配置不完整：需要 --vlm-base/--vlm-key/--vlm-model "
                "或环境变量 VLM_API_BASE/VLM_API_KEY/VLM_MODEL")
        from openai import OpenAI
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                             timeout=self.timeout)

    def _encode_image(self, path: str) -> str:
        """图片压到 max_side 内再转 base64 data url，控制请求体积。"""
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = self.max_side / max(w, h)
        if scale < 1.0:
            img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=self.jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _parse_json(text: str):
        """从返回文本中鲁棒地提取第一个 JSON 对象（先剥离 <think> 段）。"""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def score_one(self, path: str, prompt=None) -> dict:
        user_lines = ["请评审这张图片。"]
        if prompt:
            user_lines.append(f"生成该图使用的 prompt：\"\"\"{prompt}\"\"\"")
        user_lines.append("请按系统要求仅输出 JSON。")
        messages = [
            {"role": "system", "content": VL_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": self._encode_image(path)}},
                {"type": "text", "text": "\n".join(user_lines)},
            ]},
        ]
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=0.1, max_tokens=1024)  # thinking 模型需覆盖推理 token
                text = resp.choices[0].message.content or ""
                parsed = self._parse_json(text)
                if parsed is not None:
                    out = {}
                    for k in ("aesthetics", "technical_quality",
                              "prompt_alignment", "overall"):
                        v = parsed.get(k)
                        try:
                            out[k] = int(v) if v is not None else None
                        except (TypeError, ValueError):
                            out[k] = None
                    out["comment"] = str(parsed.get("comment", ""))[:60]
                    return out
                last_err = f"JSON 解析失败: {text[:120]}"
            except Exception as e:
                last_err = str(e)[:160]
            time.sleep(min(2 ** attempt, 8))  # 指数退避
        return {"error": last_err or "unknown"}

    def score_batch(self, items, progress=True):
        """items: [{"path":..., "prompt":...}, ...] → 与输入等长的结果列表（并发调用）。"""
        results = [None] * len(items)
        with ThreadPoolExecutor(self.workers) as pool:
            futures = {pool.submit(self.score_one, it["path"], it.get("prompt")): i
                       for i, it in enumerate(items)}
            iterator = as_completed(futures)
            if progress:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(iterator, total=len(futures),
                                    ncols=100, desc="  🤖 VLM API", leave=False)
                except ImportError:
                    pass
            for f in iterator:
                results[futures[f]] = f.result()
        return results
