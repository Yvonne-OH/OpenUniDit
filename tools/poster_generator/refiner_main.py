# -*- coding: utf-8 -*-
"""Qwen3-VL 润色 worker 主进程 (独立进程, CUDA_VISIBLE_DEVICES 由启动方设置)。

轮询 tasks 目录: tasks/req_<id>.json -> tasks/req_<id>_layouts.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_refiner import generate_layouts, load_vlm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", type=Path, required=True)
    ap.add_argument("--num-posters", type=int, default=2)
    args = ap.parse_args()
    tasks_dir = args.tasks_dir
    tasks_dir.mkdir(parents=True, exist_ok=True)

    processor, model = load_vlm()
    print("[refiner] Qwen3-VL ready", flush=True)

    stop_flag = tasks_dir / "ALL_STOP"
    # 只取原始任务文件 req_<id>.json (排除 refiner 自己产出的 *_layouts.json)
    pending = lambda: sorted(p for p in tasks_dir.glob("req_*.json")
                             if not p.name.endswith("_layouts.json")
                             and not p.with_name(p.stem + "_layouts.json").exists())
    while True:
        done = True
        for task_file in pending():
            if stop_flag.exists():
                break
            done = False
            req_id = task_file.stem.replace("req_", "")
            try:
                record = json.loads(task_file.read_text(encoding="utf-8"))
                layouts, _ = generate_layouts(processor, model, record,
                                              args.num_posters)
            except Exception as e:
                print(f"[refiner] req {req_id} FAILED: {e}", flush=True)
                # 写一个空的 layouts 文件避免无限重试
                task_file.with_name(task_file.stem + "_layouts.json").write_text(
                    json.dumps({"error": str(e)}, ensure_ascii=False),
                    encoding="utf-8")
                continue
            out = []
            for layout in layouts:
                layout.setdefault("summary", {})["meta"] = ["poster"]
                layout.setdefault("summary", {})["canvas"] = "single"
                layout["bbox_coord_space"] = "abs_centralized"
                layout["bbox_format"] = "yxyx"
                out.append(layout)
            task_file.with_name(task_file.stem + "_layouts.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[refiner] req {req_id} -> {len(out)} layouts", flush=True)
        if done and stop_flag.exists():
            break
        time.sleep(3)
    print("[refiner] exit", flush=True)


if __name__ == "__main__":
    main()
