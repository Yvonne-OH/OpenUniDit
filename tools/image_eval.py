# -*- coding: utf-8 -*-
"""
无 GT 图片评价器 CLI（模型评价 + VLM API 评价）。

用法示例
────────────────────────────────────────────────────────────────
# 全指标 + VLM（API 走环境变量 VLM_API_BASE / VLM_API_KEY / VLM_MODEL）
python tools/image_eval.py --images outputs/pretrain/vis_token_exp/step-00165000 \
    --metrics all --vlm-workers 8

# 只跑免 prompt 指标，抽样 100 张
python tools/image_eval.py --images <dir> --metrics musiq,liqe,laion_aes --sample 100

# 指定 VLM API
python tools/image_eval.py --images <dir> --vlm-base https://api.xxx.com/v1 \
    --vlm-key sk-xxx --vlm-model qwen-vl-max

# 查看指标清单
python tools/image_eval.py --list-metrics

输出
────────────────────────────────────────────────────────────────
<out>/results.jsonl   每张图一条：{file, prompt, scores:{...}, vlm:{...}, error}
<out>/summary.json    配置 + 各指标统计（count/mean/std/min/max）
控制台                彩色汇总表

prompt 解析（--prompts auto，默认）：
    1) 图片目录下 manifest.json 的 images[].file/prompt（infer 输出格式）
    2) 同名 .txt（整文件内容即 prompt）
    3) 同名 .json 的 prompt/caption/text 字段
也可传 {"<文件名或绝对路径>": "<prompt>"} 的 json 文件，或 none 关闭。
"""
import argparse
import glob
import json
import os
import random
import re
import statistics
import sys
import time

# ---- H800 容器 CUDA 兼容：libcuda 被 ldconfig 解析到旧版 compat 库导致 error 803，
# ---- 若 CUDA 不可用则用正确的 libcuda 自动重拉起自身（一次性到位）
_LIBCUDA = "/usr/lib/x86_64-linux-gnu/libcuda.so.570.172.08"


def _ensure_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            return
    except Exception:
        pass
    if os.path.exists(_LIBCUDA) and os.environ.get("LD_PRELOAD", "") != _LIBCUDA:
        env = dict(os.environ, LD_PRELOAD=_LIBCUDA)
        print(f"\033[33m⚠️ CUDA 不可用，自动以 LD_PRELOAD={_LIBCUDA} 重新拉起\033[0m")
        os.execve(sys.executable, [sys.executable] + sys.argv, env)


G = "\033[92m"   # 绿
B = "\033[94m"   # 蓝
Y = "\033[93m"   # 黄
R = "\033[91m"   # 红
W = "\033[0m"    # 复位
BOLD = "\033[1m"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
PROMPT_JSON_KEYS = ("prompt", "caption", "text")


def log_info(msg):
    print(f"{G}[INFO]{W} {msg}", flush=True)


def log_warn(msg):
    print(f"{Y}[WARN]{W} {msg}", flush=True)


def log_err(msg):
    print(f"{R}[ERR ]{W} {msg}", flush=True)


def list_metrics():
    from image_eval_scorers import METRIC_INFO
    print(f"\n{BOLD}可用指标（无 GT，reference-free）{W}")
    print(f"{'名称':<14}{'需要prompt':<10}{'类别':<28}{'推荐度':<10}{'量程'}")
    print("─" * 90)
    needs = {"clip": "✅", "siglip": "✅", "imagereward": "✅", "pickscore": "✅",
             "musiq": "❌", "liqe": "❌", "laion_aes": "❌", "vlm": "可选"}
    from image_eval_scorers import SCORER_REGISTRY
    hints = {k: cls.range_hint for k, cls in SCORER_REGISTRY.items()}
    hints["vlm"] = "1-10 四维 + 简评"
    for key, (disp, cat, stars) in METRIC_INFO.items():
        print(f"{B}{key:<14}{W}{needs[key]:<10}{cat:<28}{stars:<12}{hints[key]}")
    print()
    print(f"VLM 评价需配置：--vlm-base/--vlm-key/--vlm-model 或环境变量 "
          f"VLM_API_BASE / VLM_API_KEY / VLM_MODEL\n")


def collect_images(images: str, recursive: bool):
    """返回排序后的图片绝对路径列表。"""
    if os.path.isdir(images):
        pattern = "**/*" if recursive else "*"
        paths = [p for p in glob.glob(os.path.join(images, pattern), recursive=recursive)
                 if p.lower().endswith(IMG_EXTS) and os.path.isfile(p)]
    elif any(ch in images for ch in "*?["):
        paths = [p for p in glob.glob(images) if os.path.isfile(p)]
    elif os.path.isfile(images) and images.endswith(".txt"):
        with open(images, encoding="utf-8") as f:
            paths = [ln.strip() for ln in f if ln.strip()]
        paths = [p for p in paths if os.path.isfile(p)]
    elif os.path.isfile(images):
        paths = [images]
    else:
        raise FileNotFoundError(f"--images 不存在或不支持: {images}")
    return sorted(os.path.abspath(p) for p in paths)


def _read_sidecar(path: str):
    base = os.path.splitext(path)[0]
    txt = base + ".txt"
    if os.path.isfile(txt):
        try:
            with open(txt, encoding="utf-8") as f:
                s = f.read().strip()
            if s:
                return s
        except Exception:
            pass
    js = base + ".json"
    if os.path.isfile(js):
        try:
            with open(js, encoding="utf-8") as f:
                d = json.load(f)
            for k in PROMPT_JSON_KEYS:
                if isinstance(d, dict) and d.get(k):
                    return str(d[k])
        except Exception:
            pass
    return None


def load_prompts(mode: str, paths):
    """返回 {abspath: prompt}。mode: none / auto / json 文件路径"""
    if mode == "none":
        return {}
    mapping = {}
    if mode == "auto":
        # 1) manifest.json（infer 输出目录格式）
        manifests = []
        for p in paths:
            d = os.path.dirname(p)
            m = os.path.join(d, "manifest.json")
            if m not in manifests and os.path.isfile(m):
                manifests.append(m)
        for m in manifests:
            try:
                with open(m, encoding="utf-8") as f:
                    d = json.load(f)
                for it in d.get("images", []):
                    fp = it.get("file")
                    pr = it.get("prompt") or it.get("caption")
                    if fp and pr:
                        mapping[os.path.abspath(os.path.join(os.path.dirname(m), fp))] = str(pr)
            except Exception as e:
                log_warn(f"manifest.json 解析失败 {m}: {e}")
        # 2) 同名 .txt / .json sidecar 兜底
        for p in paths:
            if p not in mapping:
                s = _read_sidecar(p)
                if s:
                    mapping[p] = s
    else:  # 显式 json 映射文件
        with open(mode, encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            mapping[os.path.abspath(k)] = str(v)
        # 允许用纯文件名匹配
        by_name = {os.path.basename(k): v for k, v in mapping.items()}
        for p in paths:
            if p not in mapping and os.path.basename(p) in by_name:
                mapping[p] = by_name[os.path.basename(p)]
    return {p: mapping[p] for p in paths if p in mapping}


def sample_paths(paths, n_ratio, n_count):
    if n_count:
        return paths[:n_count]
    if n_ratio:
        k = max(1, int(round(len(paths) * n_ratio)))
        rnd = random.Random(42)  # 固定种子保证可复现
        return sorted(rnd.sample(paths, min(k, len(paths))))
    return paths


def stats_block(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "mean": round(statistics.fmean(vals), 4),
        "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def parse_args():
    ap = argparse.ArgumentParser(
        description="无 GT 图片评价器：模型评价（CLIPScore/SigLIP/ImageReward/PickScore/"
                    "MUSIQ/LIQE/LAION-Aes）+ VLM API 评价",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--images", help="图片目录 / glob / txt 列表文件")
    ap.add_argument("--recursive", action="store_true", help="递归扫描目录")
    ap.add_argument("--prompts", default="auto",
                    help="auto | none | json 映射文件路径（见 docstring）")
    ap.add_argument("--metrics", default="all",
                    help="all 或逗号分隔: clip,siglip,imagereward,pickscore,musiq,"
                         "liqe,laion_aes,vlm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--sample", type=int, default=0, help="只评前 N 张")
    ap.add_argument("--sample-ratio", type=float, default=0.0,
                    help="随机抽样比例（seed=42），如 0.01")
    ap.add_argument("--out-dir", default=None, help="结果输出目录")
    ap.add_argument("--resume", action="store_true",
                    help="跳过 results.jsonl 中已完成的图片")
    ap.add_argument("--list-metrics", action="store_true", help="打印指标清单退出")
    # VLM API
    ap.add_argument("--vlm-base", default=None)
    ap.add_argument("--vlm-key", default=None)
    ap.add_argument("--vlm-model", default=None)
    ap.add_argument("--vlm-workers", type=int, default=8, help="VLM 并发数")
    ap.add_argument("--vlm-timeout", type=int, default=120)
    args = ap.parse_args()
    if not args.list_metrics and not args.images:
        ap.error("--images 必填（或用 --list-metrics 查看指标）")
    return args


def main():
    args = parse_args()
    if args.list_metrics:
        list_metrics()
        return
    _ensure_cuda()
    t0 = time.time()

    # ── 1. 收集图片与 prompt ──────────────────────────────────────────
    paths = collect_images(args.images, args.recursive)
    if not paths:
        log_err("未找到任何图片")
        sys.exit(1)
    paths = sample_paths(paths, args.sample_ratio, args.sample)
    prompt_map = load_prompts(args.prompts, paths)
    n_prompt = len(prompt_map)
    log_info(f"🖼️  图片 {len(paths)} 张（有 prompt: {n_prompt}，缺失: "
             f"{len(paths) - n_prompt}）")

    # ── 2. 解析指标 ──────────────────────────────────────────────────
    from image_eval_scorers import (ALL_MODEL_METRICS, METRIC_INFO,
                                    SCORER_REGISTRY)
    if args.metrics.strip().lower() == "all":
        metrics = list(ALL_MODEL_METRICS)
        vlm_on = True
    else:
        req = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
        vlm_on = "vlm" in req
        metrics = [m for m in req if m in SCORER_REGISTRY]
        bad = [m for m in req if m != "vlm" and m not in SCORER_REGISTRY]
        if bad:
            log_err(f"未知指标: {bad}（用 --list-metrics 查看）")
            sys.exit(1)
    # 无 prompt 时自动剔除需要 prompt 的指标
    if n_prompt == 0:
        drop = [m for m in metrics if SCORER_REGISTRY[m].needs_prompt]
        if drop:
            log_warn(f"没有任何可用 prompt，跳过需要 prompt 的指标: {drop}")
            metrics = [m for m in metrics if m not in drop]

    vlm_cfg = None
    if vlm_on:
        from image_eval_vlm import VLMScorer
        try:
            vlm = VLMScorer(base_url=args.vlm_base, api_key=args.vlm_key,
                            model=args.vlm_model, workers=args.vlm_workers,
                            timeout=args.vlm_timeout)
            metrics.append("vlm")
            vlm_cfg = {"base": vlm.base_url, "model": vlm.model,
                       "workers": args.vlm_workers}
        except ValueError as e:
            log_warn(f"VLM 评价不可用: {e}")

    if not metrics:
        log_err("没有可执行的指标")
        sys.exit(1)
    log_info(f"📊 启用指标: {metrics} @ {args.device}")

    # ── 3. 输出目录 & 断点续评 ────────────────────────────────────────
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(paths[0]) if len(paths) == 1 else args.images
        if os.path.isdir(args.images) else os.getcwd(), "image_eval_results")
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "results.jsonl")
    records = {}
    if args.resume and os.path.isfile(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    records[r["file"]] = r
                except Exception:
                    pass
        log_info(f"断点续评：已有 {len(records)} 条记录")

    todo = [p for p in paths if p not in records]

    # ── 4. 模型指标逐个跑（每个指标一条 tqdm）────────────────────────
    def _flush(rec):
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from tqdm import tqdm
    for m in [x for x in metrics if x != "vlm"]:
        cls = SCORER_REGISTRY[m]
        scorer = cls(device=args.device, batch_size=args.batch_size)
        sub = [p for p in todo] if not cls.needs_prompt else \
            [p for p in todo if p in prompt_map]
        if cls.needs_prompt:
            miss = len(todo) - len(sub)
            if miss:
                log_warn(f"[{m}] {miss} 张缺 prompt，跳过")
        if not sub:
            continue
        print(f"\n{B}▶ {m}  {METRIC_INFO[m][0]}（{METRIC_INFO[m][1]}）"
              f"  量程 {cls.range_hint}{W}")
        prompts_sub = [prompt_map.get(p) for p in sub]
        if hasattr(scorer, "_encode"):  # CLIP/SigLIP/PickScore：内置批量
            _scores = scorer(sub, prompts_sub)
        else:  # ImageReward / pyiqa 系：逐图推理，分块跑并显示进度
            _scores = []
            chunk = 64
            with tqdm(total=len(sub), ncols=100, desc=f"  🎨 {m}", leave=False) as bar:
                for i in range(0, len(sub), chunk):
                    _scores.extend(scorer(sub[i:i + chunk],
                                          prompts_sub[i:i + chunk]))
                    bar.update(min(chunk, len(sub) - i))
        for p, s in zip(sub, _scores):
            if p not in records:
                records[p] = {"file": p, "prompt": prompt_map.get(p), "scores": {},
                              "vlm": None, "error": None}
            records[p]["scores"][m] = s
        _flush_records(records, paths, jsonl_path)
        n_fail = sum(1 for s in _scores if s is None)
        if n_fail:
            log_warn(f"[{m}] 失败 {n_fail}/{len(sub)}")

    # ── 5. VLM 评价（并发 API）──────────────────────────────────────
    if "vlm" in metrics:
        from image_eval_vlm import VLMScorer
        vlm = VLMScorer(base_url=args.vlm_base, api_key=args.vlm_key,
                        model=args.vlm_model, workers=args.vlm_workers,
                        timeout=args.vlm_timeout)
        items = [{"path": p, "prompt": prompt_map.get(p)} for p in todo]
        print(f"\n{B}▶ vlm  VLM API 评价（{vlm.model} @ {vlm.base_url}，"
              f"{args.vlm_workers} 并发）{W}")
        outs = vlm.score_batch(items)
        for p, o in zip(todo, outs):
            if p not in records:
                records[p] = {"file": p, "prompt": prompt_map.get(p), "scores": {},
                              "vlm": None, "error": None}
            records[p]["vlm"] = o
        _flush_records(records, paths, jsonl_path)
        n_err = sum(1 for o in outs if o and "error" in o)
        if n_err:
            log_warn(f"[vlm] 失败 {n_err}/{len(outs)}")

    # ── 6. 汇总 ─────────────────────────────────────────────────────
    summary = {
        "config": {
            "images": args.images, "n_images": len(paths),
            "metrics": metrics, "device": args.device,
            "batch_size": args.batch_size,
            "prompts_mode": args.prompts, "n_with_prompt": n_prompt,
            "vlm": vlm_cfg,
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "stats": {},
    }
    for m in [x for x in metrics if x != "vlm"]:
        summary["stats"][m] = stats_block([records[p]["scores"].get(m) for p in paths])
    # VLM 各维度统计
    vlm_keys = ("aesthetics", "technical_quality", "prompt_alignment", "overall")
    for k in vlm_keys:
        vals = [records[p]["vlm"].get(k) for p in paths
                if records[p].get("vlm") and records[p]["vlm"].get(k) is not None]
        if vals:
            summary["stats"][f"vlm/{k}"] = stats_block(vals)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 控制台汇总表
    print(f"\n{BOLD}{'═' * 78}")
    print(f"{B}  评价汇总（n={len(paths)}, 有 prompt={n_prompt}, "
          f"耗时 {summary['config']['elapsed_sec']}s）{W}")
    print(f"{'═' * 78}{W}")
    print(f"{'指标':<16}{'count':>7}{'mean':>10}{'std':>9}{'min':>9}{'max':>9}")
    print("─" * 78)
    for k, s in summary["stats"].items():
        if not s.get("count"):
            print(f"{Y}{k:<16}{W}{'0':>7}")
            continue
        print(f"{G}{k:<16}{W}{s['count']:>7}{s['mean']:>10.3f}{s['std']:>9.3f}"
              f"{s['min']:>9.3f}{s['max']:>9.3f}")
    print("─" * 78)
    log_info(f"✅ 完成：{jsonl_path}")
    log_info(f"✅ 完成：{os.path.join(out_dir, 'summary.json')}")


def _flush_records(records, paths, jsonl_path):
    """按 paths 顺序全量重写 results.jsonl（断点续评时保留旧记录）。"""
    tmp = jsonl_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for p in paths:
            if p in records:
                f.write(json.dumps(records[p], ensure_ascii=False) + "\n")
    os.replace(tmp, jsonl_path)


if __name__ == "__main__":
    main()
