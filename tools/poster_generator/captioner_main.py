# -*- coding: utf-8 -*-
"""Qwen3-VL caption worker 主进程 (独立进程, CUDA_VISIBLE_DEVICES 由启动方设置)。

轮询各分片目录: 渲染完成的 *_p<j>.png -> Qwen3-VL 识别图片
  1. 检测异常文字 (乱码/无意义字符/明显错字): 有 -> 删除该海报 (png/json) 并写 .bad 标记;
  2. 正常 -> 写自然语言 prompt 到 *_p<j>_caption.txt。
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CAPTION_SYSTEM = """你是海报质检员和描述专家。仔细看这张海报图片:

第一步, 检查图中是否出现异常文字: 乱码、无意义字符、明显错误的拼写或汉字、\
扭曲变形不可读的文字、重复堆叠的字母等。标题/正文中个别风格化设计不算异常, \
无法辨认的乱码才算。

第二步, 若文字正常, 写一段简洁的自然语言 prompt (中文, 60~120字), \
描述画面主体、图中出现的文字内容(原样引用)、排版位置和整体风格, \
适合作为文生图模型的输入 prompt。

严格按以下格式输出两行, 不要输出任何其他内容:
ABNORMAL: <yes|no>
CAPTION: <一段自然语言 prompt>"""


def load_captioner():
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_path = "/home/jovyan/pfs/users/zhanqian.wu/ckpt/Qwen3-VL-4B-Instruct"
    from render_worker import ensure_cuda
    ensure_cuda()
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    return processor, model


def check_image(processor, model, image_path: Path):
    """返回 (abnormal: bool, caption: str|None)。解析失败时 abnormal=False, caption=None。"""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": CAPTION_SYSTEM}]},
        {"role": "user", "content": [{"type": "image", "image": img}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    reply = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    m = re.search(r"ABNORMAL:\s*(yes|no)", reply, re.I)
    c = re.search(r"CAPTION:\s*(.+)", reply, re.S)
    if not m:
        return False, None  # 格式异常: 不误删, 留待重试
    abnormal = m.group(1).lower() == "yes"
    caption = c.group(1).strip() if c else None
    return abnormal, caption


def find_pending(shard_dirs):
    pending = []
    for d in shard_dirs:
        for png in d.glob("req_*_p*.png"):
            if png.stem.rsplit("_p", 1)[-1].isdigit() and \
                    not png.with_name(png.stem + "_caption.txt").exists() and \
                    not png.with_name(png.stem + ".bad").exists():
                pending.append(png)
    return sorted(pending)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out_dir = args.out_dir

    processor, model = load_captioner()
    print("[captioner] Qwen3-VL ready", flush=True)

    stop_flag = out_dir / "tasks" / "ALL_STOP"
    while True:
        pending = find_pending(out_dir.glob("shard_*"))
        if pending:
            png = pending[0]
            cap_path = png.with_name(png.stem + "_caption.txt")
            try:
                cap_path.touch(exist_ok=False)  # 原子占位, 防止重复
            except FileExistsError:
                continue
            try:
                abnormal, caption = check_image(processor, model, png)
            except Exception as e:
                cap_path.unlink(missing_ok=True)
                print(f"[captioner] {png.name} FAILED: {e}", flush=True)
                time.sleep(2)
                continue
            if abnormal:
                # 异常文字: 删除海报及其 layout json、caption 占位, 写 .bad 标记防止重新渲染
                cap_path.unlink(missing_ok=True)
                png.with_suffix(".json").unlink(missing_ok=True)
                png.unlink(missing_ok=True)
                png.with_name(png.stem + ".bad").write_text(
                    "abnormal text detected", encoding="utf-8")
                print(f"[captioner] {png.name} ABNORMAL TEXT -> deleted", flush=True)
                continue
            if not caption:
                # 解析失败: 释放占位, 稍后重试 (最多靠占用-重试循环)
                cap_path.unlink(missing_ok=True)
                time.sleep(2)
                continue
            cap_path.write_text(caption, encoding="utf-8")
            print(f"[captioner] {png.name} -> {caption[:60]}...", flush=True)
            continue
        if stop_flag.exists():
            break
        time.sleep(5)
    print("[captioner] exit", flush=True)


if __name__ == "__main__":
    main()
