# -*- coding: utf-8 -*-
"""MOSS-Image-SFT 渲染 worker (每张卡一个进程)。"""
import json
import sys
import time
import zlib
from pathlib import Path

MOSS_MODEL_PATH = "/home/jovyan/pfs/users/zhanqian.wu/ckpt/MOSS-Image-SFT"
MODEL_EXP_ROOT = Path(__file__).resolve().parents[2]  # code/model_exp


def ensure_cuda(retries=12, delay=5.0):
    """多进程同时初始化 CUDA 时集群驱动偶发 Error 304, 重试即可。"""
    import time
    import torch
    for _ in range(retries):
        try:
            torch.zeros(1).cuda()
            torch.cuda.mem_get_info(0)
            return
        except RuntimeError as e:
            if "304" not in str(e):
                raise
            time.sleep(delay)
    raise RuntimeError("CUDA init kept failing with Error 304")


def load_moss_pipeline():
    import torch
    from transformers import AutoTokenizer

    # 先做一次最小 CUDA 初始化: 集群环境下大文件 mmap 加载后延迟初始化 CUDA
    # 可能触发 Error 304 (OS call failed)
    ensure_cuda()

    sys.path.insert(0, str(MODEL_EXP_ROOT / "diffusers" / "src"))
    sys.path.insert(0, MOSS_MODEL_PATH)
    from pipeline_moss_image import MOSSImagePipeline

    pipe = MOSSImagePipeline.from_components_pretrained(
        transformer_path=Path(MOSS_MODEL_PATH) / "transformer",
        text_encoder_path=Path(MOSS_MODEL_PATH) / "text_encoder",
        vae_path=Path(MOSS_MODEL_PATH) / "vae",
        torch_dtype=torch.bfloat16,
        vae_dtype=torch.float32,
        local_files_only=True,
    )
    pipe.tokenizer = AutoTokenizer.from_pretrained(
        Path(MOSS_MODEL_PATH) / "tokenizer", local_files_only=True)
    if pipe.tokenizer.chat_template is None:
        template_path = Path(MOSS_MODEL_PATH) / "tokenizer" / "chat_template.jinja"
        pipe.tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    pipe.to("cuda")
    return pipe


def render_one(pipe, layout: dict, out_path: Path, height=1024, width=1024,
               steps=50, guidance_scale=3.0, seed=42, max_sequence_length=4096):
    import torch

    prompt = json.dumps(layout, ensure_ascii=False)
    images = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        negative_prompt="",
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        uncond_skip_layers=[22],
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out_path)
    return out_path


def render_worker(gpu_id, img_q, out_dir: Path, meta_path: Path, render_args: dict):
    import torch  # noqa: F401  确保子进程内初始化 CUDA

    pipe = load_moss_pipeline()
    print(f"[render gpu{gpu_id}] MOSS-Image ready", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        task = img_q.get()
        if task == "STOP":
            break
        req_id, record, poster_idx, layout = task
        if layout is None:  # refiner 失败
            continue
        if poster_idx == "num_posters":  # 元信息占位, 无需渲染
            continue
        seed = (zlib.crc32(f"{req_id}_p{poster_idx}".encode()) % (2 ** 31)) + gpu_id * 100003
        out_png = out_dir / f"{req_id}_p{poster_idx}.png"
        out_json = out_dir / f"{req_id}_p{poster_idx}.json"
        out_json.write_text(json.dumps(layout, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        t0 = time.time()
        try:
            render_one(pipe, layout, out_png, seed=seed, **render_args)
        except Exception as e:
            print(f"[render gpu{gpu_id}] req {req_id} p{poster_idx} FAILED: {e}",
                  flush=True)
            continue
        dt = time.time() - t0
        meta = {
            "req_id": req_id,
            "poster_idx": poster_idx,
            "image": str(out_png),
            "layout": str(out_json),
            "seed": seed,
            "seconds": round(dt, 1),
            "source_record": record,
        }
        with meta_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        print(f"[render gpu{gpu_id}] req {req_id} p{poster_idx} done in {dt:.1f}s -> {out_png}",
              flush=True)
