#!/usr/bin/env python3
"""benchmark.run: 对 generate.py 产出的图片目录打分并汇总报告 (json + markdown).

输入目录约定 (与 benchmark.generate 输出一致):
    <images-dir>/<bench_name>/<ckpt_tag>/<id>_seed{seed}.png + manifest.json

各 benchmark 的默认评分方式:
    geneval          → VLMJudgeScorer(mode="geneval"): GenEval 协议判定, 输出各维度正确率
    heim             → VLMJudgeScorer(mode="score"): alignment/aesthetics/detail 各维 1-10 均分
    qwen_image_bench → VLMJudgeScorer(mode="score"): 同上
    hpsv2            → HPSv2Scorer: HPS v2 原始分均值

也可用 --scorer 覆盖为 image_reward / aesthetic 等标量 scorer 对任意 bench 打分.

示例 (冒烟):
    python -m benchmark.run --bench hpsv2 --images-dir outputs/bench/smoke_hpsv2 \
        --ckpt-tag smoke --out report.json --md report.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHES = ["geneval", "heim", "hpsv2", "qwen_image_bench"]

# 各 benchmark 的默认 scorer 配置
DEFAULT_SCORERS = {
    "geneval": {"scorer": "vlm_judge", "mode": "geneval"},
    "heim": {"scorer": "vlm_judge", "mode": "score"},
    "qwen_image_bench": {"scorer": "vlm_judge", "mode": "score"},
    "hpsv2": {"scorer": "hpsv2"},
}
GENEVAL_DIMS = ["existence", "count", "color", "position"]
SCORE_DIMS = ["alignment", "aesthetics", "detail"]


def parse_args():
    ap = argparse.ArgumentParser(description="评测打分与报告")
    ap.add_argument("--bench", required=True, choices=BENCHES + ["all"])
    ap.add_argument("--images-dir", type=Path, required=True,
                    help="图片根目录 (<images-dir>/<bench>/<ckpt_tag>/*.png)")
    ap.add_argument("--ckpt-tag", default=None,
                    help="只评这个 ckpt 子目录; 缺省评 images-dir 下发现的全部 bench/ckpt 组合")
    ap.add_argument("--prompts-dir", type=Path, default=Path(__file__).parent / "prompts",
                    help="prompts jsonl 目录 (图片目录无 manifest 时按 id 回查 prompt/meta)")
    ap.add_argument("--scorer", default=None,
                    choices=["hpsv2", "image_reward", "aesthetic", "vlm_geneval", "vlm_score"],
                    help="覆盖默认 scorer")
    ap.add_argument("--limit", type=int, default=None, help="每个组合最多评 N 张图 (冒烟用)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True, help="report.json 输出路径")
    ap.add_argument("--md", type=Path, default=None, help="report.md 输出路径 (缺省与 --out 同名 .md)")
    return ap.parse_args()


def discover_runs(images_dir: Path, bench: str, ckpt_tag=None):
    """枚举 <images-dir>/<bench>/<ckpt_tag> 下有图片的组合."""
    base = images_dir / bench
    if not base.is_dir():
        return []
    tags = [ckpt_tag] if ckpt_tag else sorted({p.parent.name for p in base.glob("*/*.png")})
    return [(bench, t, base / t) for t in tags if any((base / t).glob("*.png"))]


def load_meta_map(run_dir: Path, bench: str, prompts_dir: Path):
    """id → {"prompt", "meta"}: 优先读 run_dir/manifest.json, 否则回查 prompts jsonl."""
    meta_map = {}
    manifest = run_dir / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for im in data.get("images", []):
            meta_map[im.get("id")] = {"prompt": im.get("prompt", ""),
                                      "meta": im.get("meta", {})}
    jsonl = prompts_dir / f"{bench}_prompts.jsonl"
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            it = json.loads(line)
            meta_map.setdefault(it["id"], {"prompt": it["prompt"], "meta": it.get("meta", {})})
    return meta_map


def make_scorer(name: str, device: str):
    """构建 scorer 实例 (延迟 import, 避免未用到的重依赖被拉起)."""
    from benchmark.scorers import make_scorer as _make
    if name == "vlm_geneval":
        return _make("vlm_judge", device=device, mode="geneval")
    if name == "vlm_score":
        return _make("vlm_judge", device=device, mode="score")
    return _make(name, device=device)


def scalar_report(values: list[dict]) -> dict:
    """标量 scorer 的汇总: 每图 {"id","seed","score"} → 均值/极值 + prompt 级均值."""
    scores = [v["score"] for v in values]
    by_prompt = defaultdict(list)
    for v in values:
        by_prompt[v["id"]].append(v["score"])
    return {
        "num_images": len(scores),
        "mean": statistics.fmean(scores) if scores else None,
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
        "prompt_mean": statistics.fmean(statistics.fmean(s) for s in by_prompt.values()) if by_prompt else None,
        "per_image": values,
    }


def geneval_report(values: list[dict]) -> dict:
    """GenEval 汇总: overall 正确率 + 各维度正确率 (existence/count/color/position)."""
    all_checks = [c for v in values for c in v["detailed"].get("checks", [])]
    by_dim = {}
    for dim in GENEVAL_DIMS:
        checks = [c for c in all_checks if c["type"] == dim]
        by_dim[dim] = {
            "n": len(checks),
            "pass_rate": (sum(1 for c in checks if c["pass"]) / len(checks)) if checks else None,
        }
    overall = sum(1 for c in all_checks if c["pass"]) / len(all_checks) if all_checks else 0.0
    # prompt 级通过率 (同 id 多 seed 取均值)
    by_prompt = defaultdict(list)
    for v in values:
        by_prompt[v["id"]].append(v["detailed"]["score"])
    return {
        "num_images": len(values),
        "overall_pass_rate": overall,
        "prompt_mean_pass_rate": statistics.fmean(
            statistics.fmean(s) for s in by_prompt.values()) if by_prompt else None,
        "by_dim": by_dim,
        "per_image": values,
    }


def score_report(values: list[dict]) -> dict:
    """VLM 多维打分汇总: 各维均分 (1-10) + 总均值."""
    by_dim = {}
    for dim in SCORE_DIMS:
        vals = [v["detailed"][dim] for v in values if dim in v["detailed"]
                and isinstance(v["detailed"][dim], (int, float))]
        by_dim[dim] = {
            "n": len(vals),
            "mean": statistics.fmean(vals) if vals else None,
        }
    means = [d["mean"] for d in by_dim.values() if d["mean"] is not None]
    return {
        "num_images": len(values),
        "mean": statistics.fmean(means) if means else None,
        "by_dim": by_dim,
        "per_image": values,
    }


def eval_run(bench: str, ckpt_tag: str, run_dir: Path, scorer_name: str,
             device: str, limit=None) -> dict:
    """评一个 bench/ckpt 组合, 返回该组合的报告 dict."""
    from PIL import Image

    scorer = make_scorer(scorer_name, device=device)
    meta_map = load_meta_map(run_dir, bench, Path(__file__).parent / "prompts")
    pngs = sorted(run_dir.glob("*.png"))
    if limit:
        pngs = pngs[:limit]
    values = []
    batch_img, batch_meta = [], []
    png_names = []

    def flush():
        if not batch_img:
            return
        prompts = [m["prompt"] for m in batch_meta]
        metas = [m["meta"] for m in batch_meta]
        if scorer_name.startswith("vlm_"):
            detailed = scorer.score_detailed(batch_img, prompts, metas=metas)
            for m, d in zip(batch_meta, detailed):
                if scorer_name == "vlm_geneval":
                    values.append({"id": m["id"], "seed": m["seed"], "detailed": d})
                else:
                    dims = [v for k, v in d.items()
                            if k in SCORE_DIMS and isinstance(v, (int, float))]
                    values.append({"id": m["id"], "seed": m["seed"], "detailed": d,
                                   "score": round(statistics.fmean(dims), 4) if dims else 0.0})
        else:
            scores = scorer.score(batch_img, prompts if any(prompts) else None)
            for m, s in zip(batch_meta, scores):
                values.append({"id": m["id"], "seed": m["seed"], "score": round(float(s), 4)})
        batch_img.clear(), batch_meta.clear(), png_names.clear()

    for png in pngs:
        stem = png.stem  # <id>_seed<seed>
        seed_part = stem.rsplit("_seed", 1)
        img_id = seed_part[0] if len(seed_part) == 2 else stem
        seed = int(seed_part[1]) if len(seed_part) == 2 and seed_part[1].isdigit() else 0
        meta = meta_map.get(img_id, {"prompt": "", "meta": {}})
        batch_img.append(Image.open(png).convert("RGB"))
        batch_meta.append({"id": img_id, "seed": seed, **meta})
        png_names.append(png.name)
        if len(batch_img) >= 4:  # 小批量: CPU 下控制内存/便于增量输出
            flush()
    flush()

    if scorer_name == "vlm_geneval":
        body = geneval_report(values)
    elif scorer_name == "vlm_score":
        body = score_report(values)
    else:
        body = scalar_report(values)
    body.update({"bench": bench, "ckpt_tag": ckpt_tag, "scorer": scorer_name,
                 "images_dir": str(run_dir)})
    return body


def render_md(report: dict, bench_order: list[str]) -> str:
    """汇总 markdown: 总表 + 各 benchmark 明细表."""
    lines = ["# T2I Benchmark Report", ""]
    lines.append(f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 图片目录: `{report.get('images_root', '')}`")
    lines.append("")

    # 总表
    lines.append("## 总览")
    lines.append("")
    lines.append("| benchmark | ckpt_tag | scorer | 指标 | 数值 | 图片数 |")
    lines.append("|---|---|---|---|---|---|")
    for bench in bench_order:
        for tag, r in report.get(bench, {}).items():
            if r.get("scorer") == "vlm_geneval":
                metric, val = "GenEval pass_rate", r.get("overall_pass_rate")
            elif r.get("scorer") == "vlm_score":
                metric, val = "VLM 3-dim mean (1-10)", r.get("mean")
            else:
                metric, val = f"{r.get('scorer')} mean", r.get("mean")
            lines.append(f"| {bench} | {tag} | {r.get('scorer')} | {metric} | "
                         f"{val:.4f} | {r.get('num_images')} |" if val is not None else
                         f"| {bench} | {tag} | {r.get('scorer')} | {metric} | - | {r.get('num_images')} |")
    lines.append("")

    # 明细表
    for bench in bench_order:
        for tag, r in report.get(bench, {}).items():
            lines.append(f"## {bench} · {tag}")
            lines.append("")
            by_dim = r.get("by_dim") or {}
            if r.get("scorer") == "vlm_geneval":
                lines.append(f"总体通过率 **{r.get('overall_pass_rate'):.2%}** "
                             f"({r.get('num_images')} 张图)")
                lines.append("")
                lines.append("| 维度 | n | 正确率 |")
                lines.append("|---|---|---|")
                for dim, d in by_dim.items():
                    pr = f"{d['pass_rate']:.2%}" if d.get("pass_rate") is not None else "-"
                    lines.append(f"| {dim} | {d['n']} | {pr} |")
            elif by_dim:
                lines.append(f"总均分 **{r.get('mean'):.2f}** (1-10, {r.get('num_images')} 张图)")
                lines.append("")
                lines.append("| 维度 | n | 均分 |")
                lines.append("|---|---|---|")
                for dim, d in by_dim.items():
                    m = f"{d['mean']:.2f}" if d.get("mean") is not None else "-"
                    lines.append(f"| {dim} | {d['n']} | {m} |")
            else:
                lines.append(f"mean **{r.get('mean'):.4f}** · min {r.get('min')} · max {r.get('max')} "
                             f"({r.get('num_images')} 张图)")
            lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    benches = BENCHES if args.bench == "all" else [args.bench]
    report: dict = {"images_root": str(args.images_dir), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    bench_order = []
    for bench in benches:
        runs = discover_runs(args.images_dir, bench, args.ckpt_tag)
        if not runs:
            print(f"[skip] {bench}: {args.images_dir / bench} 下没有图片")
            continue
        report[bench] = {}
        for bench_name, tag, run_dir in runs:
            scorer_name = args.scorer or DEFAULT_SCORERS[bench_name]["scorer"]
            # vlm scorer 需要按 bench 决定 mode
            if args.scorer is None:
                scorer_name = {"geneval": "vlm_geneval", "hpsv2": "hpsv2"}.get(
                    bench_name, "vlm_score")
            print(f"[run] {bench_name}/{tag}: scorer={scorer_name} dir={run_dir}")
            body = eval_run(bench_name, tag, run_dir, scorer_name, args.device, limit=args.limit)
            report[bench_name][tag] = body
            bench_order.append(bench_name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # per_image 体积可能很大: 报告里保留, json 原样写出
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = args.md or args.out.with_suffix(".md")
    md_path.write_text(render_md(report, bench_order), encoding="utf-8")
    print(f"✅ report.json → {args.out}")
    print(f"✅ report.md   → {md_path}")


if __name__ == "__main__":
    main()
