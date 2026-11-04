#!/usr/bin/env python
"""缩句工具: 读取 unified_images_lmdb 的 parquet caption, 用 LLM 压缩为短句。

默认只做对比 (compare): 输出原文/缩句 JSONL 报告, 不写任何数据。
确认效果后再用 --apply 改写 parquet (mdb 改写待确认后另行实现)。

单卡:
  python compress_captions.py --lmdb_root ... --source fine_t2i --max-samples 32
多卡 (按分片切分, 各 rank 处理不同 parquet 分片):
  python -m torch.distributed.run --nproc_per_node=8 compress_captions.py ... --apply
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq


# ─────────────────────────── 数据读取 ───────────────────────────

def list_shards(lmdb_root: str, sources: list[str]) -> list[Path]:
    """枚举各 source 下的 meta_XXXX.parquet 分片 (跳过汇总 meta.parquet)。"""
    root = Path(lmdb_root)
    shards: list[Path] = []
    for source in sources:
        parquet_dir = root / source / "parquet"
        if not parquet_dir.is_dir():
            print(f"⚠️ 跳过不存在的 source: {parquet_dir}", flush=True)
            continue
        files = sorted(p for p in parquet_dir.glob("*.parquet") if p.name != "meta.parquet")
        if not files:
            print(f"⚠️ source 无分片: {parquet_dir}", flush=True)
        shards.extend(files)
    return shards


def iter_captions(shard_path: Path, source: str, max_samples: int | None,
                  stride: int = 1) -> list[dict]:
    """读取单个分片的 caption 行。stride>1 时均匀抽样, 避免大分片全量读取慢。"""
    table = pq.read_table(shard_path, columns=["sample_id", "caption"])
    n = table.num_rows
    ids = table.column("sample_id").to_pylist()
    caps = table.column("caption").to_pylist()
    rows = []
    for i in range(0, n, stride):
        cap = caps[i]
        if isinstance(cap, str) and cap.strip():
            rows.append({"source": source, "shard": shard_path.name,
                         "sample_id": ids[i], "caption": cap})
        if max_samples and len(rows) >= max_samples:
            break
    return rows


# ─────────────────────────── LLM 推理 ───────────────────────────

SYSTEM_PROMPT = (
    "你是图像描述压缩助手。将给定的图像 caption 压缩为一句简短描述："
    "1) 保留主体、场景、动作、风格、文字内容等关键信息；"
    "2) 删除冗余修饰、氛围描写和重复表达；"
    "3) 输出语言与输入一致 (英文输入输出英文, 中文输入输出中文)；"
    "4) 长度控制在 15~40 个词以内, 原文已短于 25 词则原样返回；"
    "5) 只输出压缩后的 caption 本身, 不要任何解释、引号或前后缀。"
)


class CaptionCompressor:
    def __init__(self, model_path: str, max_new_tokens: int = 128,
                 batch_size: int = 32):
        import torch
        from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                                  AutoTokenizer)

        self.torch = torch
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        except (ValueError, KeyError):  # qwen3_5 等多模态包装类
            model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype)
        self.model = model.to(self.device).eval()
        print(f"✅ 模型加载完成: {model_path} @ {self.device} (bf16)", flush=True)

    def _build_inputs(self, captions: list[str]) -> dict:
        torch = self.torch
        prompts = []
        for cap in captions:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": cap.strip()},
            ]
            try:
                prompts.append(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False))
            except TypeError:  # 旧模板无 enable_thinking 参数
                prompts.append(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True))
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def compress_batch(self, captions: list[str]) -> list[str]:
        torch = self.torch
        inputs = self._build_inputs(captions)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return [self.tokenizer.decode(out[i, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()
                for i in range(len(captions))]

    def compress_all(self, rows: list[dict], progress_every: int = 10) -> list[str]:
        """按长度排序分批推理 (减少 padding), 再还原原顺序。"""
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["caption"]))
        results: list[str | None] = [None] * len(rows)
        t0 = time.time()
        for bi in range(0, len(order), self.batch_size):
            idx = order[bi:bi + self.batch_size]
            batch = [rows[i]["caption"] for i in idx]
            outs = self.compress_batch(batch)
            for i, o in zip(idx, outs):
                results[i] = o
            done = min(bi + self.batch_size, len(order))
            if done % progress_every < self.batch_size or done == len(order):
                speed = done / max(time.time() - t0, 1e-6)
                print(f"  ⏳ {done}/{len(order)} 条 ({speed:.1f} 条/s)", flush=True)
        return results  # type: ignore[return-value]


# ─────────────────────────── 主流程 ───────────────────────────

def pick_samples(args) -> list[dict]:
    shards = list_shards(args.lmdb_root, args.sources)
    if not shards:
        sys.exit("❌ 没有可用 parquet 分片")
    if args.shards:
        keep = set(args.shards)
        shards = [s for s in shards if s.stem in keep or s.name in keep]
    rng = random.Random(args.seed)
    shards.sort()
    if args.sample_shards:
        rng.shuffle(shards)
    rows: list[dict] = []
    per_shard = args.max_samples
    for shard in shards:
        if per_shard and len(rows) >= per_shard:
            break
        rows.extend(iter_captions(
            shard, shard.parent.parent.name,
            max_samples=None if per_shard is None else per_shard - len(rows),
            stride=args.stride))
        if per_shard is None and args.stride <= 1 and len(rows) > 2_000_000:
            break  # 全量模式的安全上限
    if args.sample_rows and per_shard is None:
        rng.shuffle(rows)
        rows = rows[:args.sample_rows]
    print(f"✅ 共采样 {len(rows):,} 条 caption (来自 {len(set(r['source'] for r in rows))} 个 source)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="caption 缩句工具 (默认只对比, 不写数据)")
    ap.add_argument("--lmdb_root", default="/home/jovyan/pfs/users/zhanqian.wu/pfs_data/unified_images_lmdb")
    ap.add_argument("--source", action="append", default=None,
                    help="source 名, 可多次指定; 默认取 fine_t2i")
    ap.add_argument("--model", default="/home/jovyan/pfs/users/zhanqian.wu/ckpt/Qwen3.5-4B")
    ap.add_argument("--max-samples", type=int, default=32,
                    help="总采样条数; 0 表示全量")
    ap.add_argument("--sample-rows", type=int, default=0,
                    help="配合 stride 全量扫描后随机抽 N 条")
    ap.add_argument("--stride", type=int, default=1, help="分片内均匀抽样步长")
    ap.add_argument("--sample-shards", action="store_true", help="随机打乱分片顺序再采样")
    ap.add_argument("--shards", nargs="*", default=None, help="只处理指定分片名")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="/home/jovyan/pfs/users/zhanqian.wu/exp/giga_exp_image/compress_captions")
    ap.add_argument("--apply", action="store_true",
                    help="确认后使用: 将缩写结果写入新 parquet (不碰 mdb)")
    args = ap.parse_args()

    args.sources = args.source or ["fine_t2i"]
    if args.max_samples and args.max_samples <= 0:
        args.max_samples = None

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rows = pick_samples(args)
    # 多卡: 按 rank 轮转切分样本 (单卡 world_size=1 不受影响)
    rows = rows[rank::world_size]
    print(f"[rank {rank}/{world_size}] 处理 {len(rows):,} 条", flush=True)
    if not rows:
        return

    compressor = CaptionCompressor(args.model, args.max_new_tokens,
                                   args.batch_size)
    compressed = compressor.compress_all(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if world_size == 1 else f"_rank{rank}"
    report_path = out_dir / f"compare{tag}.jsonl"
    with open(report_path, "w", encoding="utf-8") as f:
        for row, short in zip(rows, compressed):
            f.write(json.dumps({
                "source": row["source"], "shard": row["shard"],
                "sample_id": row["sample_id"],
                "original": row["caption"], "compressed": short,
            }, ensure_ascii=False) + "\n")
    print(f"✅ 对比报告已写入: {report_path}", flush=True)

    if args.apply:
        apply_parquet(args.lmdb_root, rows, compressed, out_dir)


def apply_parquet(lmdb_root: str, rows: list[dict], compressed: list[str],
                  out_dir: Path):
    """按 (source, shard) 分组, 原样复制分片后仅改写 caption 列, 写到 out_dir 下镜像目录。"""
    groups: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault((row["source"], row["shard"]), []).append(i)
    for (source, shard_name), idxs in groups.items():
        src_path = Path(lmdb_root) / source / "parquet" / shard_name
        table = pq.read_table(src_path)
        caps = table.column("caption").to_pylist()
        id2short = {rows[i]["sample_id"]: compressed[i] for i in idxs}
        n_changed = 0
        for r, sid in enumerate(table.column("sample_id").to_pylist()):
            if sid in id2short:
                caps[r] = id2short[sid]
                n_changed += 1
        dst_dir = out_dir / source / "parquet"
        dst_dir.mkdir(parents=True, exist_ok=True)
        import pyarrow as pa
        new_table = table.set_column(
            table.schema.get_field_index("caption"),
            "caption", pa.array(caps, type=table.schema.field("caption").type))
        dst_path = dst_dir / shard_name
        pq.write_table(new_table, dst_path)
        print(f"  ✍️ {source}/{shard_name}: 改写 {n_changed:,} 条 -> {dst_path}", flush=True)


if __name__ == "__main__":
    main()
