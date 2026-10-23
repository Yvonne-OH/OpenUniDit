# -*- coding: utf-8 -*-
"""
批量评测 infer 输出根目录下所有 step-* 文件夹，每个 step 生成一份 JSON 报告。

每个 step 文件夹产出（直接写入该文件夹）：
    summary.json    配置 + 各指标统计（count/mean/std/min/max）
    results.jsonl   每张图一条：{file, prompt, scores:{...}, vlm:{...}, error}

根目录额外产出：
    eval_steps_summary.json   所有 step 的汇总（step → stats + 耗时）
    eval_steps_table.txt      step × 指标 mean 对比表（文本）

用法
────────────────────────────────────────────────────────────────
# 全指标 + VLM（MiniMax 国内站），用最后一张卡避免与训练冲突
python tools/image_eval_steps.py --root outputs/pretrain/vis_token_exp \
    --device cuda:7 \
    --vlm-base https://api.minimaxi.com/v1 --vlm-key sk-xxx \
    --vlm-model MiniMax-M3

# 只跑部分 step（调试用，逗号分隔 step 数字）
python tools/image_eval_steps.py --root <root> --only 16500,17000

# 不带 VLM 参数则只跑 7 个模型指标；已完成的 step 自动跳过，--force 重跑
"""
import argparse
import json
import os
import re
import sys
import time

G = "\033[92m"
B = "\033[94m"
Y = "\033[93m"
R = "\033[91m"
W = "\033[0m"
BOLD = "\033[1m"

_TOOLS = os.path.dirname(os.path.abspath(__file__))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from image_eval import (collect_images, load_prompts, log_info, log_warn,
                        stats_block, _ensure_cuda)


def find_step_dirs(root: str, only=None):
    """返回 [(step_num, dir), ...] 按数字升序。"""
    out = []
    for name in os.listdir(root):
        m = re.fullmatch(r"step-(\d+)", name)
        if m and os.path.isdir(os.path.join(root, name)):
            n = int(m.group(1))  # 显式十进制，避免前导零陷阱
            if only is None or n in only:
                out.append((n, os.path.join(root, name)))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(
        description="批量评测所有 step-* 文件夹（每 step 一份 summary.json）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", required=True, help="infer 输出根目录")
    ap.add_argument("--metrics", default="all",
                    help="all 或逗号分隔: clip,siglip,imagereward,pickscore,"
                         "musiq,liqe,laion_aes,vlm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--only", default=None,
                    help="只评测指定 step（逗号分隔数字）")
    ap.add_argument("--force", action="store_true", help="重跑已完成 step")
    # VLM API
    ap.add_argument("--vlm-base", default=None)
    ap.add_argument("--vlm-key", default=None)
    ap.add_argument("--vlm-model", default=None)
    ap.add_argument("--vlm-workers", type=int, default=8)
    ap.add_argument("--vlm-timeout", type=int, default=180)
    args = ap.parse_args()

    _ensure_cuda()
    t_all = time.time()
    only = ({int(x) for x in args.only.split(",") if x.strip()}
            if args.only else None)
    steps = find_step_dirs(args.root, only)
    if not steps:
        print(f"{R}[ERR ]{W} 在 {args.root} 下未找到 step-* 文件夹")
        sys.exit(1)

    # ── 指标解析 ──────────────────────────────────────────────────
    from image_eval_scorers import ALL_MODEL_METRICS, SCORER_REGISTRY
    if args.metrics.strip().lower() == "all":
        metrics = list(ALL_MODEL_METRICS)
        vlm_on = True
    else:
        req = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
        vlm_on = "vlm" in req
        metrics = [m for m in req if m in SCORER_REGISTRY]

    # ── VLM（可用性 ping 一次，额度/配置问题尽早暴露）──────────────
    vlm = None
    if vlm_on:
        try:
            from image_eval_vlm import VLMScorer
            vlm = VLMScorer(base_url=args.vlm_base, api_key=args.vlm_key,
                            model=args.vlm_model, workers=args.vlm_workers,
                            timeout=args.vlm_timeout)
            ping_path = None
            for _, d in steps:
                imgs = collect_images(d, recursive=False)
                if imgs:
                    ping_path = imgs[0]
                    break
            ping = vlm.score_one(ping_path, None)
            if "error" in ping:
                log_warn(f"VLM ping 失败，本批禁用 VLM: {ping['error']}")
                vlm = None
            else:
                log_info(f"🤖 VLM 就绪: {vlm.model} @ {vlm.base_url} "
                         f"(ping: overall={ping.get('overall')})")
        except ValueError as e:
            log_warn(f"VLM 评价不可用: {e}")
            vlm = None
    active_metrics = metrics + (["vlm"] if vlm else [])
    if not active_metrics:
        print(f"{R}[ERR ]{W} 没有可执行的指标")
        sys.exit(1)
    print(f"{B}📊 指标: {active_metrics} @ {args.device} | "
          f"共 {len(steps)} 个 step{W}")

    # ── 逐 step 评测（评分器进程内复用）───────────────────────────
    scorers = {}
    rows = []  # (step_num, step_dir, stats, elapsed)
    for si, (step_num, step_dir) in enumerate(steps, 1):
        summary_path = os.path.join(step_dir, "summary.json")
        paths = collect_images(step_dir, recursive=False)
        # 断点跳过：已有 summary 且图片数一致
        if not args.force and os.path.isfile(summary_path):
            try:
                with open(summary_path, encoding="utf-8") as f:
                    old = json.load(f)
                if old.get("config", {}).get("n_images") == len(paths):
                    log_info(f"⏭️  [{si}/{len(steps)}] step-{step_num} "
                             f"已完成，跳过（--force 重跑）")
                    rows.append((step_num, step_dir, old.get("stats", {}),
                                 old.get("config", {}).get("elapsed_sec", 0)))
                    continue
            except Exception:
                pass
        if not paths:
            log_warn(f"[{si}/{len(steps)}] step-{step_num} 无图片，跳过")
            continue
        prompt_map = load_prompts("auto", paths)
        t0 = time.time()
        print(f"\n{BOLD}{'═' * 78}")
        print(f"{B}▶ [{si}/{len(steps)}] step-{step_num}  "
              f"🖼️ {len(paths)} 张（prompt {len(prompt_map)}）{W}")
        print(f"{'═' * 78}{W}")

        records = {p: {"file": p, "prompt": prompt_map.get(p),
                       "scores": {}, "vlm": None, "error": None} for p in paths}
        for m in metrics:
            cls = SCORER_REGISTRY[m]
            if cls.needs_prompt and not prompt_map:
                log_warn(f"[{m}] 无可用 prompt，跳过该指标")
                continue
            if m not in scorers:
                scorers[m] = cls(device=args.device,
                                 batch_size=args.batch_size)
            sub = [p for p in paths if not cls.needs_prompt
                   or p in prompt_map]
            prompts_sub = [prompt_map.get(p) for p in sub]
            print(f"{B}  ▶ {m}{W}  量程 {cls.range_hint}")
            scores = scorers[m](sub, prompts_sub)
            for p, s in zip(sub, scores):
                records[p]["scores"][m] = s
            n_fail = sum(1 for s in scores if s is None)
            if n_fail:
                log_warn(f"  [{m}] 失败 {n_fail}/{len(sub)}")

        if vlm:
            items = [{"path": p, "prompt": prompt_map.get(p)} for p in paths]
            print(f"{B}  ▶ vlm  {vlm.model}（{args.vlm_workers} 并发）{W}")
            outs = vlm.score_batch(items)
            for p, o in zip(paths, outs):
                records[p]["vlm"] = o

        # 写 results.jsonl + summary.json（每个 step 文件夹一份报告）
        jsonl_path = os.path.join(step_dir, "results.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for p in paths:
                f.write(json.dumps(records[p], ensure_ascii=False) + "\n")
        elapsed = round(time.time() - t0, 1)
        stats = {}
        for m in metrics:
            stats[m] = stats_block([records[p]["scores"].get(m) for p in paths])
        if vlm:
            for k in ("aesthetics", "technical_quality",
                      "prompt_alignment", "overall"):
                vals = [records[p]["vlm"].get(k) for p in paths
                        if records[p].get("vlm")
                        and records[p]["vlm"].get(k) is not None]
                if vals:
                    stats[f"vlm/{k}"] = stats_block(vals)
        summary = {
            "config": {"root": args.root, "n_images": len(paths),
                       "n_with_prompt": len(prompt_map),
                       "metrics": active_metrics, "device": args.device,
                       "batch_size": args.batch_size,
                       "vlm": ({"base": vlm.base_url, "model": vlm.model,
                                "workers": args.vlm_workers} if vlm else None),
                       "elapsed_sec": elapsed},
            "stats": stats,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        rows.append((step_num, step_dir, stats, elapsed))
        heads = [m for m in active_metrics if stats.get(m, {}).get("count")]
        line = "  ".join(
            f"{m}={stats[m]['mean']:.3f}" if m != "vlm" else
            f"vlm/overall={stats['vlm/overall']['mean']:.2f}"
            if stats.get("vlm/overall", {}).get("count") else "vlm=err"
            for m in heads)
        print(f"{G}  ✅ step-{step_num} 完成（{elapsed}s）{W}  {line}")

    # ── 根目录汇总报告 ────────────────────────────────────────────
    all_metrics = []
    for _, _, stats, _ in rows:
        for m in stats:
            if m not in all_metrics:
                all_metrics.append(m)
    root_summary = {
        "root": args.root,
        "n_steps": len(rows),
        "total_elapsed_sec": round(time.time() - t_all, 1),
        "device": args.device,
        "steps": {str(n): {"dir": d, "stats": s, "elapsed_sec": e}
                  for n, d, s, e in rows},
    }
    root_json = os.path.join(args.root, "eval_steps_summary.json")
    with open(root_json, "w", encoding="utf-8") as f:
        json.dump(root_summary, f, ensure_ascii=False, indent=2)

    # 文本对比表：行=step，列=指标 mean
    tbl = os.path.join(args.root, "eval_steps_table.txt")
    with open(tbl, "w", encoding="utf-8") as f:
        f.write(f"{'step':<14}" + "".join(f"{m:>16}" for m in all_metrics)
                + f"{'elapsed_s':>12}\n")
        f.write("─" * (14 + 16 * len(all_metrics) + 12) + "\n")
        for n, _, stats, e in rows:
            cells = ""
            for m in all_metrics:
                v = stats.get(m, {})
                cells += f"{v['mean']:>16.4f}" if v.get("count") \
                    else f"{'-':>16}"
            f.write(f"{n:<14}" + cells + f"{e:>12}\n")

    # 控制台总表
    print(f"\n{BOLD}{'═' * 78}")
    print(f"{B}  全部 step 评测汇总（{len(rows)} 个 step，"
          f"总耗时 {root_summary['total_elapsed_sec']}s）{W}")
    print(f"{'═' * 78}{W}")
    print(f"{'step':<14}" + "".join(f"{m:>15}" for m in all_metrics))
    print("─" * (14 + 15 * len(all_metrics)))
    for n, _, stats, _ in rows:
        cells = ""
        for m in all_metrics:
            v = stats.get(m, {})
            cells += f"{v['mean']:>15.3f}" if v.get("count") else f"{'-':>15}"
        print(f"{G}{n:<14}{W}" + cells)
    print("─" * (14 + 15 * len(all_metrics)))
    log_info(f"✅ 总报告: {root_json}")
    log_info(f"✅ 对比表: {tbl}")


if __name__ == "__main__":
    main()
