# -*- coding: utf-8 -*-
"""Qwen3-VL prompt 润色 worker (独占 1 张卡)。

输入 opencole 原始标注 -> 中文润色、概括成通用海报 prompt,
每个请求输出 2 个海报 layout JSON (MOSS-Image JSON Layout Prompt 格式,
文字内容用中文引号 "" 框住), 推入渲染队列。
"""
import json
import re
import sys
from pathlib import Path

MODEL_PATH = "/home/jovyan/pfs/users/zhanqian.wu/ckpt/Qwen3-VL-4B-Instruct"

# 供 Qwen3-VL 仿写的 layout JSON 范例 (用户给的示例)
EXAMPLE_LAYOUT = {
    "summary": {
        "meta": ["poster"],
        "canvas": "single",
        "caption": "A minimalist travel poster with a cream background, a blue mountain illustration, and the title EXPLORE at the top.",
    },
    "items": [
        {
            "label": "text",
            "caption": "Large dark-blue uppercase title centered at the top.",
            "text": "\u201cEXPLORE\u201d",
            "bbox": [-416, -384, -288, 384],
        },
        {
            "label": "object",
            "caption": "A flat blue mountain illustration in the lower center.",
            "bbox": [-192, -288, 384, 288],
        },
    ],
    "bbox_coord_space": "abs_centralized",
    "bbox_format": "yxyx",
}

SYSTEM_PROMPT = """你是资深广告海报创意设计师和 prompt 工程师。给你一份海报设计的原始标注(intention/description/keywords/captions/headings), 你需要:

1. 中文润色: 把原始标注提炼润色成一份简洁的中文海报创意描述。
2. 概括成通用语言的 prompt: 不要照搬原始事件的具体细节(如具体地址、日期、店名), 把它概括成可复用的通用海报主题描述。
3. 基于该创意生成恰好 2 个不同风格的海报 layout JSON, 每个海报输出 2~4 个元素(标题文字 + 画面主体, 可选副标题/装饰文字)。

每个海报 layout JSON 必须严格遵守以下格式(这是一个范例):
""" + json.dumps(EXAMPLE_LAYOUT, ensure_ascii=False) + """

格式规则:
- summary.meta 固定为 ["poster"]; canvas 固定为 "single"。
- summary.caption: 一句话概括整张海报(中文润色后的通用描述, 中英混合的美术风格描述皆可)。
- items 中文字元素用 label="text" 且必须带 text 字段; 文字内容必须用中文引号 “” 框住, 例如 "text": "“探索山海”"。文字要精炼(2~10 个汉字), 中文为主。
- items 中画面/插图元素用 label="object", 不带 text 字段。
- 每个 item 带一句 caption 说明其外观和位置。
- bbox 为 [y1, x1, y2, x2] (yxyx), abs_centralized 坐标系: 画布中心为原点, 各方向范围约 [-512, 512] (对应 1024 画布)。bbox 数值之间不能重叠太多, 标题在上、主体居中或居下。
- 最外层必须带 "bbox_coord_space": "abs_centralized" 和 "bbox_format": "yxyx"。

输出要求: 只输出一个 JSON 数组, 包含恰好 2 个海报 layout JSON, 不要输出任何其他解释文字。"""


def _extract_json_array(text: str):
    text = text.strip()
    # 优先取 ```json ... ``` 代码块
    m = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidates = m if m else [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            # 截取首个 [ 到最后一个 ]
            i, j = cand.find("["), cand.rfind("]")
            if i == -1 or j <= i:
                continue
            try:
                data = json.loads(cand[i:j + 1])
            except json.JSONDecodeError:
                continue
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list) and data and all(isinstance(d, dict) for d in data):
            return data
    return None


def build_user_message(record: dict) -> str:
    kw = ", ".join(k for k in record.get("keywords", []) if k)
    headings = "; ".join(record.get("headings_heading", []) or [])
    sub = "; ".join(record.get("headings_sub_heading", []) or [])
    parts = [
        f"intention: {record.get('intention', '')}",
        f"description: {record.get('description', '')}",
        f"keywords: {kw}",
        f"captions_background: {record.get('captions_background', '')}",
        f"captions_objects: {record.get('captions_objects', '')}",
        f"headings_heading: {headings}",
        f"headings_sub_heading: {sub}",
    ]
    return "原始海报标注:\n" + "\n".join(parts)


def generate_layouts(processor, model, record: dict, num_posters: int = 2,
                     max_new_tokens: int = 4096):
    """对一个请求生成 num_posters 个海报 layout, 失败重试一次。"""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": build_user_message(record)}]},
    ]
    last_err = None
    for _ in range(3):
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.95,
            repetition_penalty=1.1,
        )
        reply = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        layouts = _extract_json_array(reply)
        if layouts and len(layouts) >= 1:
            return layouts[:num_posters], reply
        last_err = reply
        # 把失败回复拼回去再问一次
        messages.append({"role": "assistant",
                         "content": [{"type": "text", "text": reply}]})
        messages.append({"role": "user",
                         "content": [{"type": "text",
                                      "text": "上面的输出不是合法 JSON 数组, 请重新只输出一个包含 2 个海报 layout JSON 的 JSON 数组。"}]})
    raise ValueError(f"Qwen3-VL 输出无法解析为 JSON: {str(last_err)[:500]}")


def load_vlm():
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    # 先做一次最小 CUDA 初始化, 避免大文件 mmap 加载后延迟初始化报 Error 304
    from render_worker import ensure_cuda
    ensure_cuda()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    return processor, model


def refiner_worker(vlm_in_q, img_q, num_posters=2):
    """消费 vlm_in_q 中的 (req_id, record), 润色后把渲染任务推入 img_q。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    processor, model = load_vlm()
    print("[refiner] Qwen3-VL ready", flush=True)
    while True:
        task = vlm_in_q.get()
        if task == "STOP":
            img_q.put("STOP")
            break
        req_id, record = task
        try:
            layouts, reply = generate_layouts(processor, model, record, num_posters)
        except Exception as e:
            print(f"[refiner] req {req_id} FAILED: {e}", flush=True)
            img_q.put((req_id, record, None, {"error": str(e)}))
            continue
        for gi, layout in enumerate(layouts):
            layout.setdefault("summary", {})["meta"] = ["poster"]
            layout.setdefault("summary", {})["canvas"] = "single"
            layout["bbox_coord_space"] = "abs_centralized"
            layout["bbox_format"] = "yxyx"
            img_q.put((req_id, record, gi, layout))
        img_q.put((req_id, record, "num_posters", len(layouts)))
        print(f"[refiner] req {req_id} -> {len(layouts)} layouts", flush=True)
