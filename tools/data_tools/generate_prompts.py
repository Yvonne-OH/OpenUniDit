#!/usr/bin/env python
"""多语言 caption 变体 + 图像质量指标生成工具。

读取 unified_images_lmdb 各 source 的 parquet caption 与 mdb 图像字节, 产出:
- prompt: 规范化 JSON 变体 {"short","zh","jp","de","fr"}
  short = 压缩短句 (与原文同语言); zh/jp/de/fr = 完整原文的翻译,
  画面内文字 (招牌/标语等) 保持原语言;
- metrics: 清晰度/曝光/色彩/JPEG块效应 (cv2) + LAION aesthetic + CLIPScore。

逐条追加写入 {lmdb_root}/{source}/enchanced_prompt/{分片stem}.jsonl,
记录 source/shard/sample_id 来源, 支持断点续跑 (跳过已完成 sample_id)。
本脚本只写外部文件; 回写 parquet/mdb 用 apply_prompts.py。

单卡:  python generate_prompts.py --source fine_t2i --max-per-shard 100
多卡: python generate_prompts.py ... (检测到多卡自动 python -m torch.distributed.run 重启, 按分片切分)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from lmdb_index import list_shards  # noqa: E402

PROMPT_KEYS = ("short", "zh", "jp", "de", "fr")

SYSTEM_PROMPT = (
    "你是多语言图像 caption 变体生成器。对给定的图像 caption 输出一个严格的 JSON 对象:\n"
    '{"short": "...", "zh": "...", "jp": "...", "de": "...", "fr": "..."}\n'
    "规则:\n"
    "1) short: 将 caption 压缩为一句简短描述, 语言与原文一致 (英文输入输出英文, 中文输入输出中文); "
    "保留主体、场景、动作、风格、画面内文字等关键信息, 删除冗余修饰与重复表达, 目标 15~40 词; "
    "short 必须比原文显著更短, 严禁原样复制原文; 仅当原文本身短于 25 词时才原样返回。\n"
    "2) zh/jp/de/fr: 分别是【完整原文 caption】的简体中文、日语、德语、法语翻译"
    "(不是 short 的翻译, 尽量保留原文的信息量)。\n"
    "3) 所有跨语言输出中, 画面内文字 (招牌、标语、服装字样、UI 文本等) "
    "必须保持原语言原样, 不翻译, 用引号保留 (如 \"40% off\" 保持英文)。\n"
    "4) 只输出上述 JSON 对象本身: 不要 markdown 代码块, 不要解释, 不要多余键, 5 个键缺一不可。"
)

_KEY_ALIASES = {"short": "short",
                "zh": "zh", "chinese": "zh", "cn": "zh",
                "jp": "jp", "ja": "jp", "japanese": "jp",
                "de": "de", "german": "de",
                "fr": "fr", "french": "fr"}


def normalize_prompt(obj) -> dict | None:
    """规范化 LLM 输出: 别名归并/去引号/去空, 5 键齐全才返回, 否则 None。"""
    if not isinstance(obj, dict):
        return None
    out: dict[str, str] = {}
    for k, v in obj.items():
        key = _KEY_ALIASES.get(str(k).strip().lower())
        if key is None or key in out:
            continue
        if not isinstance(v, str):
            v = "" if v is None else str(v)
        v = v.strip().strip('"“”').strip()
        if v:
            out[key] = v
    return out if all(k in out for k in PROMPT_KEYS) else None


def parse_prompt_json(text: str) -> dict | None:
    """从 LLM 输出中提取第一个 JSON 对象并规范化。"""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return normalize_prompt(json.loads(text[start:end + 1]))
    except json.JSONDecodeError:
        return None


class PromptGenerator:
    def __init__(self, model_path: str, max_new_tokens: int, batch_size: int):
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

    def _generate(self, user_contents: list[str], system: str) -> list[str]:
        torch = self.torch
        prompts = []
        for content in user_contents:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": content}]
            try:
                prompts.append(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False))
            except TypeError:
                prompts.append(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True))
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tokenizer.pad_token_id)
        return [self.tokenizer.decode(out[i, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()
                for i in range(len(user_contents))]

    def generate(self, captions: list[str]) -> tuple[list[dict | None], list[str]]:
        """生成 + 规范化, 失败项重试一次; 返回 (结果, 失败项原始输出片段)。"""
        outputs = self._generate(captions, SYSTEM_PROMPT)
        results = [parse_prompt_json(o) for o in outputs]
        retry_idx = [i for i, r in enumerate(results) if r is None]
        raw_errors: list[str] = []
        if retry_idx:
            fix_prompt = ("你上次的输出无法解析为规定格式的 JSON。请重新处理下面的 caption, "
                          "只输出包含 short/zh/jp/de/fr 五个键的合法 JSON 对象。caption:\n")
            retried = self._generate([fix_prompt + captions[i].strip() for i in retry_idx],
                                     SYSTEM_PROMPT)
            for j, i in enumerate(retry_idx):
                results[i] = parse_prompt_json(retried[j])
        raw_errors = [outputs[i][:300] for i, r in enumerate(results) if r is None]
        return results, raw_errors


# ─────────────────────────── 图像字节与指标 ───────────────────────────

_IMG_FIELDS = ("img_bytes", "image_bytes", "img", "jpg", "data")


class MdbReader:
    """按 sample_id 读 mdb 记录的图像字节 (key scheme 自动探测)。"""

    def __init__(self, mdb_path: Path):
        import lmdb

        self.env = lmdb.open(str(mdb_path), readonly=True, lock=False, subdir=False)
        self.field = None

    def image_bytes(self, sample_id: int) -> bytes | None:
        with self.env.begin(buffers=True) as txn:
            for key in (_pack_id(sample_id), str(sample_id).encode()):
                raw = txn.get(key)
                if raw is not None:
                    rec = pickle_loads(raw)
                    if self.field is None:
                        self.field = next((f for f in _IMG_FIELDS if f in rec), None)
                    if self.field is None:
                        return None
                    return rec.get(self.field)
        return None

    def close(self):
        self.env.close()


def _pack_id(sample_id: int) -> bytes:
    import struct
    return struct.pack("<Q", int(sample_id))


def pickle_loads(raw) -> dict:
    import pickle
    return pickle.loads(raw)


# ─────────────────────────── 主流程 ───────────────────────────

def iter_caption_rows(shard_path: Path, source: str, stride: int,
                      max_per_shard: int | None):
    """流式读取分片的 (sample_id, caption), 均匀抽样用 stride 控制。"""
    pf = pq.ParquetFile(shard_path)
    taken = 0
    for batch in pf.iter_batches(batch_size=4096, columns=["sample_id", "caption"]):
        ids = batch.column("sample_id").to_pylist()
        caps = batch.column("caption").to_pylist()
        for i in range(0, len(ids), stride):
            cap = caps[i]
            if isinstance(cap, str) and cap.strip():
                yield {"source": source, "shard": shard_path.stem,
                       "sample_id": ids[i], "caption": cap}
                taken += 1
                if max_per_shard and taken >= max_per_shard:
                    return


def load_done_ids(jsonl_path: Path, retry_failed: bool) -> set[int]:
    done: set[int] = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # 写入中断的残行, 视为未完成
                continue
            if rec.get("status") == "ok" or (not retry_failed and rec.get("status") == "failed"):
                done.add(int(rec["sample_id"]))
    return done


def _compute_cpu_metrics(chunk, mdb, qm):
    """线程池任务: mdb 读字节 + PIL 解码 + cv2 指标 (纯 CPU, 与 GPU 流水线重叠)。"""
    img_bytes = [mdb.image_bytes(r["sample_id"]) for r in chunk]
    images = [qm._decode_pil(b) if b else None for b in img_bytes]
    metrics = [qm.cv_metrics(b) if b else None for b in img_bytes]
    return images, metrics


def process_shard(generator: PromptGenerator, scorers, shard_path: Path,
                  source: str, out_dir: Path, args) -> tuple[int, int]:
    """处理单个分片: CPU 指标 (线程池, 预取下一块) 与 GPU LLM 生成流水线重叠。"""
    import quality_metrics as qm
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    jsonl_path = out_dir / f"{shard_path.stem}.jsonl"
    done = load_done_ids(jsonl_path, args.retry_failed)
    pending = [row for row in iter_caption_rows(shard_path, source, args.stride,
                                                args.max_per_shard)
               if row["sample_id"] not in done]
    if not pending:
        print(f"  ⏭️ {source}/{shard_path.stem}: 无待处理样本 (断点续跑跳过)", flush=True)
        return 0, 0
    print(f"  ⏳ {source}/{shard_path.stem}: 待处理 {len(pending):,} 条 (已跳过 {len(done):,})", flush=True)

    mdb = None
    if scorers:
        from lmdb_index import shard_mdb_map
        mdb_path = shard_mdb_map(shard_path.parent.parent).get(shard_path.stem)
        if mdb_path is None:
            print(f"  ⚠️ {shard_path.stem}: 无对应 mdb, 指标将只有 prompt 部分", flush=True)
        else:
            mdb = MdbReader(mdb_path)

    chunks = [pending[i:i + args.chunk_size] for i in range(0, len(pending), args.chunk_size)]
    pool = ThreadPoolExecutor(max_workers=args.metric_workers) if mdb else None
    futures = deque()
    if pool:
        futures.append(pool.submit(_compute_cpu_metrics, chunks[0], mdb, qm))

    ok = fail = 0
    t_shard = time.time()
    t_llm = t_gpu_metrics = 0.0
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for ci, chunk in enumerate(chunks):
            t0 = time.time()
            images, metrics_list = (futures.popleft().result() if pool
                                    else ([None] * len(chunk), [None] * len(chunk)))
            if ci + 1 < len(chunks) and pool:  # 预取下一块的 CPU 指标
                futures.append(pool.submit(_compute_cpu_metrics, chunks[ci + 1], mdb, qm))
            if ci + 2 < len(chunks) and pool and len(futures) < 2:  # 深度预取, 掩盖 PFS 读图延迟
                futures.append(pool.submit(_compute_cpu_metrics, chunks[ci + 2], mdb, qm))

            prompts, raw_errors = generator.generate([r["caption"] for r in chunk])
            t_llm += time.time() - t0

            t1 = time.time()
            if scorers and mdb is not None:
                aes = scorers["aes"].scores(images)
                clip = scorers["clip"].scores(images, [r["caption"] for r in chunk])
                for m, a, c in zip(metrics_list, aes, clip):
                    if m is not None:
                        m["aesthetic_score"] = a
                        m["clip_score"] = c
            t_gpu_metrics += time.time() - t1

            now = time.strftime("%Y-%m-%d %H:%M:%S")
            err_i = 0
            for row, prompt, metrics in zip(chunk, prompts, metrics_list):
                rec = {"source": row["source"], "shard": row["shard"],
                       "sample_id": row["sample_id"], "status": "ok" if prompt else "failed",
                       "model": Path(args.model).name, "ts": now,
                       "enhanced_prompt": prompt, "metrics": metrics}
                if prompt is None:
                    rec["error"] = raw_errors[err_i] if err_i < len(raw_errors) else ""
                    err_i += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok += prompt is not None
                fail += prompt is None
            f.flush()
            os.fsync(f.fileno())  # 逐块落盘, 中断可断点续跑
            done_n = min((ci + 1) * args.chunk_size, len(pending))
            rate = done_n / (time.time() - t_shard)
            print(f"    ⏳ {done_n:,}/{len(pending):,} (ok={ok:,} fail={fail:,}, "
                  f"{rate:.1f} 条/s, llm={t_llm:.0f}s gpu_metrics={t_gpu_metrics:.0f}s)",
                  flush=True)
    if pool:
        pool.shutdown(wait=False)
    if mdb is not None:
        mdb.close()
    return ok, fail


def main():
    ap = argparse.ArgumentParser(description="多语言 caption 变体 + 质量指标生成 (只写外部文件)")
    ap.add_argument("--lmdb_root", default="/home/jovyan/pfs/users/zhanqian.wu/pfs_data/unified_images_lmdb")
    ap.add_argument("--source", action="append", required=True, help="source 名, 可多次指定")
    ap.add_argument("--model", default="/home/jovyan/pfs/users/zhanqian.wu/ckpt/Qwen3.5-4B")
    ap.add_argument("--ngpu", type=int, default=0, help="自动多卡卡数; 0=全部可见 GPU (默认 8 卡 torchrun)")
    ap.add_argument("--max-per-shard", type=int, default=None, help="每个分片最多处理条数")
    ap.add_argument("--stride", type=int, default=1, help="分片内均匀抽样步长")
    ap.add_argument("--batch-size", type=int, default=64, help="LLM 批大小 (建议=chunk-size, 每 chunk 单次 generate)")
    ap.add_argument("--chunk-size", type=int, default=64, help="指标/落盘分块大小")
    ap.add_argument("--metric-workers", type=int, default=16, help="CPU 指标线程数")
    ap.add_argument("--scorer-batch-size", type=int, default=32, help="CLIP/aesthetic 批大小")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--shards", nargs="*", default=None, help="只处理指定分片 stem")
    ap.add_argument("--no-metrics", action="store_true", help="跳过图像质量指标计算")
    ap.add_argument("--retry-failed", action="store_true", help="续跑时重试上次 failed 的样本")
    args = ap.parse_args()

    # 默认 8 卡: 未在 torchrun 中且可见 GPU > 1 时, 自动以 torchrun 重启自身
    if os.environ.get("WORLD_SIZE") is None:
        import torch as _t
        n_gpus = args.ngpu or _t.cuda.device_count()
        if n_gpus > 1:
            rest, skip = [], False
            for a in sys.argv[1:]:  # 去掉 --ngpu 及其值, 避免 torchrun 子进程再触发
                if skip:
                    skip = False
                    continue
                if a == "--ngpu":
                    skip = True
                    continue
                rest.append(a)
            # 用当前解释器 -m 启动, 避免 PATH 上其它 conda 环境的 torchrun (如 /opt/conda/bin)
            cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={n_gpus}",
                   "--master_port", str(29500 + int(time.time()) % 1000),
                   str(Path(__file__).resolve()), *rest]
            print(f"🚀 自动启用 {n_gpus} 卡: {' '.join(cmd)}", flush=True)
            os.execvp(sys.executable, cmd)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    shards = list_shards(args.lmdb_root, args.source)
    if args.shards:
        keep = set(args.shards)
        shards = [s for s in shards if s.stem in keep]
    # 多卡按分片轮转切分, 单 writer 独占一个分片的 jsonl
    shards = shards[rank::world_size]
    print(f"[rank {rank}/{world_size}] 共 {len(shards)} 个分片待处理", flush=True)
    if not shards:
        return

    generator = PromptGenerator(args.model, args.max_new_tokens, args.batch_size)
    scorers = None
    if not args.no_metrics:
        import quality_metrics as qm
        scorers = {
            "aes": qm.AestheticScorer(device=generator.device,
                                      batch_size=args.scorer_batch_size),
            "clip": qm.CLIPScoreScorer(device=generator.device,
                                       batch_size=args.scorer_batch_size),
        }
        print("✅ 质量指标模型就绪 (LAION aesthetic + CLIPScore)", flush=True)

    total_ok = total_fail = 0
    t0 = time.time()
    for shard in shards:
        source = shard.parent.parent.name
        out_dir = shard.parent.parent / "enchanced_prompt"
        out_dir.mkdir(parents=True, exist_ok=True)
        ok, fail = process_shard(generator, scorers, shard, source, out_dir, args)
        total_ok += ok
        total_fail += fail
    print(f"✅ 全部完成: ok={total_ok:,} fail={total_fail:,} "
          f"({time.time() - t0:.0f}s), 输出于各 source 的 enchanced_prompt/", flush=True)


if __name__ == "__main__":
    main()
