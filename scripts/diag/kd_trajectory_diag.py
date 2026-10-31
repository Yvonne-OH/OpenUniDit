"""KD 轨迹诊断: 固定 prompt / noise / solver, teacher 与 student 各自采样, 逐步对比。

观测量 (每步、每 prompt, latent 网格逐位对齐):
  1. State Drift                  ||x_t^S - x_t^T||               两条轨迹偏离速度
  2. Teacher-state Velocity Error ||v_S(x_t^T) - v_T(x_t^T)||     teacher forcing 下的局部差距
  3. Student-state Velocity Error ||v_S(x_t^S) - v_T(x_t^S)||     student 走偏后能否修正
  4. Endpoint Drift               ||x_0^S - x_0^T||               最终累计误差

口径:
  - 统一 Euler solver + 统一 σ schedule (shift_sigma, 与训练验证一致), 同 seed 同 x_1
  - v 统一到 student 约定 (noise - z0): teacher 原始输出 = z0 - noise, 取负号
  - 默认 guidance=1.0 (纯条件, 无 CFG), 消除 CFG 组合对 velocity 定义的干扰;
    传 --guidance >1 时两侧各自按约定加 negative 分支后再比较
  - 数值一律 RMS (L2/sqrt(N)) + 相对值 (除以对应参照的 RMS)

用法:
    python scripts/diag/kd_trajectory_diag.py \
        --config configs/stage_1_kd_v2.yaml \
        --checkpoint exp/giga_exp_image/stage_1_kd_28layers_v2/checkpoints/weights/step-00247500.safetensors \
        --prompt "..." --prompt "..."
输出: <output-dir>/metrics.json / metrics.csv / curves.png / endpoints.png / per-prompt png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from diffusers import AutoencoderKLWan
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from models.giga_transformer import GigaExpTransformer
from pipeline.giga_pipeline_6b_teacher import build_gwp6b_teacher, encode_prompts_6b_teacher

DEFAULT_PROMPTS = [
    "a photo of a red fox sitting on a mossy rock in a sunlit forest",
    "a neon sign reading OPEN 24H on a rainy city street at night",
    "an oil painting of a sailboat on a calm sea at golden hour",
    "a white teacup and an open book on a wooden table by a window",
]


def parse_args():
    p = argparse.ArgumentParser(description="KD teacher/student 采样轨迹对比诊断")
    p.add_argument("--config", default="configs/stage_1_kd_v2.yaml")
    p.add_argument("--checkpoint", default=None,
                   help="student 权重 (默认取 config output_dir 下最新 step-*)")
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=20, help="采样步数 (与训练验证一致)")
    p.add_argument("--shift", type=float, default=None, help="缺省取 scheduler.validation_shift")
    p.add_argument("--guidance", type=float, default=1.0, help="1.0 = 无 CFG (纯条件)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attention-backend", default="auto", help="teacher attention: auto|flash3|flash|native")
    p.add_argument("--teacher-checkpoint", default=None, help="覆盖 config 里的 teacher.checkpoint")
    p.add_argument("--negative-prompt", default=(
        "blurry, low quality, low resolution, distorted, deformed, disfigured, "
        "bad anatomy, extra limbs, malformed hands, text, watermark, logo, artifacts"),
        help="guidance>1 时两侧共用的 negative 文本 (对齐 gigaworld 标准推理); 传空串则回退各自空条件约定")
    return p.parse_args()


def rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt())


def main():
    args = parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    device = torch.device(args.device)
    dtype = torch.bfloat16
    prompts = args.prompt or DEFAULT_PROMPTS
    shift = args.shift if args.shift is not None else float(
        cfg["scheduler"].get("validation_shift") or cfg["scheduler"].get("shift") or 1.0)

    out_dir = Path(args.output_dir) if args.output_dir else Path(
        cfg["output_dir"]) / "diag" / f"traj_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- student ckpt ----
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        wdir = Path(cfg["output_dir"]) / "checkpoints" / "weights"
        ckpt = max(wdir.glob("step-*.safetensors"), key=lambda q: int(q.name.split("-")[1].split(".")[0]))
    print(f"🎓 student: {ckpt}")
    print(f"🧑‍🏫 teacher: {cfg['teacher']['checkpoint']}")
    print(f"⚙️  {len(prompts)} prompts | {args.width}x{args.height} | steps={args.steps} "
          f"| shift={shift} | guidance={args.guidance} | seed={args.seed}")

    # ---- 组件 (共享一份 Qwen3-VL + VAE) ----
    tc, vc = cfg["components"]["text_encoder"], cfg["components"]["wan_vae"]
    tokenizer = AutoTokenizer.from_pretrained(tc["tokenizer_path"], local_files_only=True)
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        tc["pretrained_path"], torch_dtype=dtype, local_files_only=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(tc["pretrained_path"], local_files_only=True)
    vae = AutoencoderKLWan.from_pretrained(
        vc["pretrained_path"], subfolder=vc.get("subfolder", "vae"),
        torch_dtype=torch.float32, local_files_only=True).eval().to(device)

    student = GigaExpTransformer(**cfg["model"]).to(device, dtype=dtype)
    state = load_file(str(ckpt)) if ckpt.suffix == ".safetensors" else torch.load(str(ckpt), map_location="cpu")
    current = student.state_dict()
    matched = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    student.load_state_dict(matched, strict=False)
    student.eval().requires_grad_(False)
    print(f"✅ student 权重: matched={len(matched)}/{len(current)}")

    teacher_cfg = dict(cfg["teacher"])
    if args.teacher_checkpoint:
        teacher_cfg["checkpoint"] = args.teacher_checkpoint
    teacher_cfg["attention_backend"] = args.attention_backend
    teacher, _ = build_gwp6b_teacher(teacher_cfg, device, dtype)
    t_patch = int(cfg["teacher"].get("patch_size", 2))
    t_fpatch = int(cfg["teacher"].get("f_patch_size", 1))
    t_layer = int(cfg["teacher"].get("layer", -2))
    t_maxcond = int(cfg["teacher"].get("max_condition_tokens", 4064))

    z_dim, compress = vae.config.z_dim, vae.config.scale_factor_spatial
    lat_h, lat_w = args.height // compress, args.width // compress
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, z_dim, 1, 1, 1).float()
    std = torch.tensor(vae.config.latents_std, device=device).view(1, z_dim, 1, 1, 1).float()

    # ---- σ schedule (student 验证口径, 两侧共用) ----
    def shift_sigma(u):
        return shift * u / (1.0 + (shift - 1.0) * u)
    sigmas = [shift_sigma(1.0 - i / args.steps) for i in range(args.steps + 1)]

    # ---- 条件编码 ----
    layer, max_length = tc.get("layer"), tc.get("max_length", 512)
    s_conds = []  # (text, mask) per prompt
    t_caps = []
    neg_text = neg_mask = None
    neg_caps = None
    for pr in prompts:
        filled = pr.strip() if pr.strip() else tokenizer.eos_token
        enc = tokenizer([filled], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length).to(device)
        out = encoder(**enc, use_cache=False, return_dict=True, output_hidden_states=True)
        s_conds.append((out.hidden_states[layer], enc.attention_mask.bool()))
        t_caps.append(encode_prompts_6b_teacher(
            processor, encoder, [pr], device, layer=t_layer, max_condition_tokens=t_maxcond)[0].to(dtype))
        if args.guidance > 1.0:
            neg_text_src = args.negative_prompt.strip() if args.negative_prompt.strip() else tokenizer.eos_token
            enc_n = tokenizer([neg_text_src], return_tensors="pt", padding=True, truncation=True,
                              max_length=max_length).to(device)
            out_n = encoder(**enc_n, use_cache=False, return_dict=True, output_hidden_states=True)
            if neg_text is None:
                neg_text, neg_mask = out_n.hidden_states[layer], enc_n.attention_mask.bool()
            if neg_caps is None:
                neg_src = args.negative_prompt.strip() if args.negative_prompt.strip() else ""
                neg_caps = encode_prompts_6b_teacher(
                    processor, encoder, [neg_src], device, layer=t_layer,
                    max_condition_tokens=t_maxcond)[0].to(dtype)

    @torch.no_grad()
    def v_student(x, sigma):
        """student 约定: 输出 = noise - z0。x [1,C,1,h,w] fp32 → v fp32。"""
        ts = torch.full((1,), sigma * 1000.0, device=device)
        def fwd(text, mask):
            return student(x.to(dtype), ts, text, mask).float()
        v = fwd(*s_conds[pi])
        if args.guidance > 1.0:
            v_un = fwd(neg_text, neg_mask)
            v = v_un + args.guidance * (v - v_un)
        return v

    @torch.no_grad()
    def v_teacher(x, sigma):
        """teacher 原始输出 = z0 - noise, 取负对齐 student 约定。"""
        t = torch.full((1,), 1.0 - sigma, device=device)
        def fwd(cap):
            out = teacher(x=[x[0].to(dtype)], t=t, cap_feats=[cap],
                          patch_size=t_patch, f_patch_size=t_fpatch)
            mo = out.sample if hasattr(out, "sample") else out[0]
            if isinstance(mo, list):
                mo = torch.stack(mo)
            return -mo.float()
        v = fwd(t_caps[pi])
        if args.guidance > 1.0:
            v_un = fwd(neg_caps)
            v = v_un + args.guidance * (v - v_un)
        return v

    # ---- 主循环: 每条 prompt 独立双轨迹 ----
    all_records = []
    for pi in range(len(prompts)):
        gen = torch.Generator(device=device).manual_seed(args.seed)
        x1 = torch.randn(1, z_dim, 1, lat_h, lat_w, generator=gen, device=device, dtype=torch.float32)
        xT, xS = x1.clone(), x1.clone()
        steps_rec = []

        def _cos(a, b):
            na, nb = float(a.norm()), float(b.norm())
            return float((a * b).sum() / (na * nb)) if na > 1e-12 and nb > 1e-12 else float("nan")

        e_prev, dv_prev = None, None
        t0 = time.time()
        for i in range(args.steps):
            sigma, next_sigma = sigmas[i], sigmas[i + 1]
            vT_xT = v_teacher(xT, sigma)   # teacher 自身步 (用于 teacher 更新)
            vS_xT = v_student(xT, sigma)   # teacher forcing: student 在 teacher 状态
            vS_xS = v_student(xS, sigma)   # student 自身步 (用于 student 更新)
            vT_xS = v_teacher(xS, sigma)   # student 状态: teacher 参照
            e = xS - xT                    # 当前状态偏差
            dv = vS_xT - vT_xT             # teacher forcing 下的速度误差
            dt = next_sigma - sigma        # <0, ODE 时间步长
            rec = {
                "step": i, "sigma": round(sigma, 6),
                "state_drift_rms": rms(xS - xT),
                "state_drift_rel": rms(xS - xT) / max(rms(xT), 1e-12),
                "vel_err_teacher_rms": rms(vS_xT - vT_xT),
                "vel_err_teacher_rel": rms(vS_xT - vT_xT) / max(rms(vT_xT), 1e-12),
                "vel_err_student_rms": rms(vS_xS - vT_xS),
                "vel_err_student_rel": rms(vS_xS - vT_xS) / max(rms(vT_xS), 1e-12),
                "vT_rms": rms(vT_xT), "vS_rms": rms(vS_xS),
                # ---- 误差方向性指标 ----
                "cos_e_dv": _cos(e, dv),                    # cos(e_k, Δv_k)
                "cos_e_dvdt": _cos(e, dt * dv),             # cos(e_k, Δt_k·Δv_k), >0 = 本步速度误差沿现有偏差方向继续推
                "cos_e_prev": _cos(e_prev, e) if e_prev is not None else float("nan"),
                "cos_dv_prev": _cos(dv_prev, dv) if dv_prev is not None else float("nan"),
            }
            steps_rec.append(rec)
            xT = xT - vT_xT * (sigma - next_sigma)
            xS = xS - vS_xS * (sigma - next_sigma)
            e_prev, dv_prev = e.clone(), dv.clone()
        endpoint = {
            "endpoint_drift_rms": rms(xS - xT),
            "endpoint_drift_rel": rms(xS - xT) / max(rms(xT), 1e-12),
            "xT_rms": rms(xT), "xS_rms": rms(xS),
            "seconds": round(time.time() - t0, 1),
        }
        all_records.append({"prompt_index": pi, "prompt": prompts[pi],
                            "steps": steps_rec, "endpoint": endpoint})
        print(f"  p{pi} done in {endpoint['seconds']}s | state_drift_rel: "
              f"{steps_rec[0]['state_drift_rel']:.4f} → {steps_rec[-1]['state_drift_rel']:.4f} | "
              f"endpoint_rel={endpoint['endpoint_drift_rel']:.4f}")

        # 解码两端点图
        with torch.no_grad():
            imgT = vae.decode(xT * std + mean).sample[0, :, 0]
            imgS = vae.decode(xS * std + mean).sample[0, :, 0]
        def to_u8(t):
            return ((t.float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        from PIL import Image
        Image.fromarray(to_u8(imgT)).save(out_dir / f"p{pi}_teacher.png")
        Image.fromarray(to_u8(imgS)).save(out_dir / f"p{pi}_student.png")
        diff = (imgS - imgT).abs().mean(0)
        d = diff / max(float(diff.max()), 1e-6)
        heat = (torch.stack([d, 1 - d, torch.zeros_like(d)], -1).clamp(0, 1) * 255).byte().cpu().numpy()
        Image.fromarray(heat).save(out_dir / f"p{pi}_diff.png")

    # ---- 落盘 ----
    meta = {
        "config": args.config, "student_checkpoint": str(ckpt),
        "teacher_checkpoint": cfg["teacher"]["checkpoint"],
        "height": args.height, "width": args.width, "steps": args.steps,
        "shift": shift, "guidance": args.guidance, "seed": args.seed,
        "sigma_schedule": [round(s, 6) for s in sigmas],
        "note": "v 统一 student 约定 (noise-z0); teacher 输出取负; RMS=L2/sqrt(N); rel 以 teacher 为参照",
        "records": all_records,
    }
    (out_dir / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_index", "step", "sigma", "state_drift_rms", "state_drift_rel",
                    "vel_err_teacher_rms", "vel_err_teacher_rel",
                    "vel_err_student_rms", "vel_err_student_rel", "vT_rms", "vS_rms",
                    "cos_e_dv", "cos_e_dvdt", "cos_e_prev", "cos_dv_prev"])
        for rec in all_records:
            for s in rec["steps"]:
                w.writerow([rec["prompt_index"], s["step"], s["sigma"],
                            *(f"{s[k]:.6f}" for k in ("state_drift_rms", "state_drift_rel",
                                  "vel_err_teacher_rms", "vel_err_teacher_rel",
                                  "vel_err_student_rms", "vel_err_student_rel", "vT_rms", "vS_rms",
                                  "cos_e_dv", "cos_e_dvdt", "cos_e_prev", "cos_dv_prev"))])

    # ---- 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    panels = [
        ("state_drift_rel", "State Drift  ||x_S - x_T|| / ||x_T||", "per-step, 轨迹偏离"),
        ("vel_err_teacher_rel", "Teacher-state Velocity Error  rel", "v_S(x_T) vs v_T(x_T), teacher forcing"),
        ("vel_err_student_rel", "Student-state Velocity Error  rel", "v_S(x_S) vs v_T(x_S), 走偏后参照"),
        ("cos_e_dvdt", r"cos$(e_k,\ \Delta t_k\,\Delta v_k)$", ">0: 速度误差沿现有偏差方向推 (系统性放大)"),
        ("cos_e_prev", r"cos$(e_k,\ e_{k+1})$", "状态误差方向持续性 (>0 = 同方向漂移)"),
        ("cos_dv_prev", r"cos$(\Delta v_k,\ \Delta v_{k+1})$", "速度误差方向持续性 (>0 = 低频偏置)"),
    ]
    for ax, (key, title, sub) in zip(axes.flat, panels):
        for rec in all_records:
            ax.plot([s["step"] for s in rec["steps"]], [s[key] for s in rec["steps"]],
                    marker=".", ms=4, label=f"p{rec['prompt_index']}")
        if key.startswith("cos"):
            ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"{title}\n{sub}", fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"KD trajectory diag | student={ckpt.stem} | {args.steps} steps, "
                 f"shift={shift}, cfg={args.guidance}, seed={args.seed}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=130)

    fig2, ax = plt.subplots(figsize=(7, 4))
    eps = [r["endpoint"]["endpoint_drift_rel"] for r in all_records]
    ax.bar([f"p{r['prompt_index']}" for r in all_records], eps, color="#5fb8ff")
    for x_, v in zip(range(len(eps)), eps):
        ax.text(x_, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Endpoint Drift  ||x0_S - x0_T|| / ||x0_T||")
    ax.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(out_dir / "endpoint.png", dpi=130)

    print(f"\n✅ 输出目录: {out_dir}")
    print("   metrics.json / metrics.csv / curves.png / endpoint.png / p*_teacher|student|diff.png")


if __name__ == "__main__":
    main()
