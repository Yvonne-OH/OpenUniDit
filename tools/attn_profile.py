"""DiT Attention Profiling: 推理期旁路统计 Q/K/V 激活与 attention 分布。

零侵入原则
----------
只挂 forward hook 读取张量, **不替换任何算子**, 因此模型 forward 的数值与 FA3 路径
完全不变; profiling 额外重算一次 softmax(QK^T/√d) 用于统计分布。

数值一致性
----------
- hook 取到 norm 后 / RoPE 前的 Q、K (RMSNorm 逐 head) 与原始 V;
- 用模型 rope 输出的 fp32 cos/sin 复现同样的旋转 (dense 路径等价于 FA3 varlen
  的逐 token 位置旋转);
- 有效 key 掩码取模型自身的 attention_mask, 并与 FA3 varlen 的"有效前缀"语义取交集,
  因此统计到的分布与模型实际计算的一致。

显存
----
按 query 分块计算 attention 概率, 峰值 ~[B,H,qc,N], 不物化整层 O(BHN²)。
推理分辨率下 N 通常只有几百, 实际开销可忽略。

指标 (per step × layer × head)
------------------------------
q_raw_rms / k_raw_rms / v_rms   norm 前 Q/K 与 V 的 RMS (激活尺度)
q_rms / k_rms                  RoPE 后 Q/K 的 RMS (真正参与 attention 的尺度)
logit_std                      Std(QK^T/√d) (是否容易过尖锐)
entropy                        每 query 熵 H=-Σ p log p 后平均 (nats)
entropy_norm                   H / log N
eff_rank                       exp(H) (有效关注 token 数)
max_attention                  每 query 最大注意力后平均
attn_out_rms                   RMS(AV) (attention 输出激活强度)
residual_ratio                 RMS(AV) / RMS(残差流)
img_to_text_mass               image query → text key 注意力占比
img_to_image_mass              image query → image key
txt_to_image_mass              text  query → image key
txt_to_text_mass               text  query → text key

冗余 / 空间
-----------
head_cos        层内 head 两两余弦相似度 (head 冗余)
layer_cos       层间签名余弦相似度 (层冗余)
img_token_mass  每个 image token 被所有 query 关注的总量 (主体定位)
img_to_text     每个 image token 投给 text 的注意力总量
txt_to_image    每个 text token 投给 image 的注意力总量
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

_EPS = 1e-8
# query chunk 的 logits 峰值显存上限 (fp32)
_CHUNK_BYTES = 256 * 1024 * 1024


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class DiTAttentionProfiler:
    """在模型 forward 过程中旁路采集 attention 统计。

    Parameters
    ----------
    model:
        GigaExpTransformer / GigaExpTransformerMHC 实例 (需含 .rope 与 .blocks)。
    layers:
        需要统计的层号 (0-based); None = 全部层。
    branch:
        CFG batch 分支: "cond" 只统计有条件分支 (batch 后半), "uncond" 只统计
        无条件分支, "both" 全统计。默认 "cond"。
    stride:
        每 stride 个 model forward 采一次 (1 = 每次都采)。
    grid:
        image token 的 (t, h, w) 网格, 仅写入结果元信息 (便于空间可视化)。
    """

    def __init__(self, model, *, layers=None, branch="cond", stride=1, grid=None, meta=None):
        self.model = model
        self.layers = None if layers is None else sorted({int(x) for x in layers})
        self.branch = branch
        self.stride = max(1, int(stride))
        self.grid = tuple(grid) if grid else None
        self.meta = dict(meta or {})
        self._steps: list[dict] = []
        self._forward_count = 0
        self._handles: list = []
        self._ctx: dict = {}

    # ------------------------------------------------------------------ #
    # hook 管理
    # ------------------------------------------------------------------ #
    def install(self) -> None:
        if self._handles:
            return
        m = self.model
        self._handles.append(m.register_forward_pre_hook(self._pre_forward, with_kwargs=True))
        self._handles.append(m.register_forward_hook(self._post_forward))
        self._handles.append(m.rope.register_forward_hook(self._rope_hook))
        for idx, block in enumerate(m.blocks):
            if self.layers is not None and idx not in self.layers:
                continue
            # mHC 变体的 block 内联计算 attention (无独立 qkv/q_norm/k_norm/out 子模块), 跳过
            if not all(hasattr(block, name) for name in ("qkv", "q_norm", "k_norm", "out")):
                continue
            self._handles.append(block.register_forward_pre_hook(
                lambda mod, args, kwargs, i=idx: self._block_pre_hook(i, args, kwargs),
                with_kwargs=True))
            self._handles.append(block.qkv.register_forward_hook(
                lambda mod, args, out: self._ctx.__setitem__("qkv", out)))
            self._handles.append(block.q_norm.register_forward_hook(
                lambda mod, args, out: self._ctx.__setitem__("q", out)))
            self._handles.append(block.k_norm.register_forward_hook(
                lambda mod, args, out: self._ctx.__setitem__("k", out)))
            # out 投影在 attention 之后触发, 此时 q/k/v/rope/mask 仍在 ctx 中
            self._handles.append(block.out.register_forward_hook(
                lambda mod, args, out, i=idx: self._out_hook(i, out)))

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *exc):
        self.uninstall()
        return False

    # ------------------------------------------------------------------ #
    # hooks
    # ------------------------------------------------------------------ #
    def _pre_forward(self, module, args, kwargs):
        """model.forward(x, timestep, text, text_mask) → 抓 sigma 与文本长度."""
        timestep = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        text = kwargs.get("text", args[2] if len(args) > 2 else None)
        self._ctx = {
            "sigma": float(timestep.reshape(-1)[0].item()) / 1000.0 if timestep is not None else None,
            "text_len": int(text.shape[1]) if text is not None else None,
            "collect": self._forward_count % self.stride == 0,
            "layers": {},
        }

    def _rope_hook(self, module, args, output):
        cos, sin = output
        self._ctx["cos"], self._ctx["sin"] = cos, sin

    def _block_pre_hook(self, layer_idx, args, kwargs):
        """抓 block 输入 (残差流) 与 attention_mask。

        block.forward(x|h, cond, cos, sin, attention_mask, pack, n_img) —— 全位置参数,
        因此 attention_mask 固定在 args[4]。
        """
        x = args[0] if len(args) > 0 else None
        if x is not None:
            if x.dim() == 4:            # mHC: [B, N, S, C] → 取 stream0
                x = x[:, :, 0]
            self._ctx["res_in"] = x
        mask = args[4] if len(args) > 4 else kwargs.get("attention_mask")
        if mask is not None and mask.dim() == 4:
            self._ctx["attn_mask"] = mask.bool()

    def _out_hook(self, layer_idx, output):
        """attention 分支 out 投影之后触发: 此刻做本层统计 (q/k/v/rope/mask 齐备)."""
        self._ctx["branch_out"] = output
        ctx = self._ctx
        if not ctx.get("collect", True):
            return
        q, k = ctx.get("q"), ctx.get("k")
        qkv, cos, sin = ctx.get("qkv"), ctx.get("cos"), ctx.get("sin")
        if q is None or k is None or qkv is None or cos is None:
            return
        stats = self._layer_stats(q, k, qkv, cos, sin, ctx)
        if stats is not None:
            ctx["layers"][layer_idx] = stats

    def _post_forward(self, module, args, output):
        ctx = self._ctx
        idx = self._forward_count
        self._forward_count += 1
        try:
            if idx % self.stride == 0 and ctx.get("layers"):
                self._steps.append({
                    "step": idx,
                    "sigma": ctx.get("sigma"),
                    "layers": ctx["layers"],
                })
        finally:
            self._ctx = {}

    # ------------------------------------------------------------------ #
    # 逐层统计
    # ------------------------------------------------------------------ #
    def _layer_stats(self, q, k, qkv, cos, sin, ctx):
        b_all, n, heads, head_dim = q.shape
        dim = heads * head_dim
        text_len = ctx.get("text_len") or 0
        n_img = n - text_len
        if n_img <= 0 or text_len <= 0:
            return None

        # --- RoPE 复现 (与 block._attend 的 fp32 旋转一致) ---
        cos_b = cos[None, :, None, :].float()   # [1, N, 1, Dh]
        sin_b = sin[None, :, None, :].float()
        qf, kf = q.float(), k.float()
        q_rot = qf * cos_b + _rotate_half(qf) * sin_b
        k_rot = kf * cos_b + _rotate_half(kf) * sin_b

        q_raw = qkv[..., :dim].float().reshape(b_all, n, heads, head_dim)
        k_raw = qkv[..., dim:2 * dim].float().reshape(b_all, n, heads, head_dim)
        v = qkv[..., 2 * dim:].float().reshape(b_all, n, heads, head_dim)

        # --- CFG 分支选择 (batch 前半 = uncond, 后半 = cond) ---
        if self.branch == "both" or b_all % 2 != 0:
            sel = slice(0, b_all)
        elif self.branch == "uncond":
            sel = slice(0, b_all // 2)
        else:
            sel = slice(b_all // 2, b_all)
        q_rot, k_rot = q_rot[sel], k_rot[sel]
        q_raw, k_raw, v = q_raw[sel], k_raw[sel], v[sel]
        b = q_rot.shape[0]

        # --- 有效 token 掩码 ---
        # attention_mask = [ones(n_img), valid_text]; FA3 varlen 按"有效前缀"计算,
        # 两者取交集 → 与模型实际计算的可见性一致。
        attn_mask = ctx.get("attn_mask")
        if attn_mask is not None:
            valid = attn_mask[sel].reshape(b, -1)
        else:
            valid = torch.ones((b, n), dtype=torch.bool, device=q_rot.device)
        n_valid_text = int(attn_mask[sel, 0, 0, n_img:].sum(-1).max().item()) \
            if attn_mask is not None else text_len
        prefix_len = n_img + n_valid_text
        pos = torch.arange(n, device=q_rot.device)
        valid = valid & (pos[None] < prefix_len)
        max_valid = int(valid.sum(-1).max().item())
        if max_valid <= 0:
            return None

        scale = 1.0 / math.sqrt(head_dim)
        per_q = b * heads * n * 4
        chunk = max(1, min(n, _CHUNK_BYTES // max(per_q, 1)))

        z = torch.zeros(b, heads, device=q_rot.device, dtype=torch.float32)
        acc = {
            "entropy": z.clone(), "q_cnt": z.clone(), "max_attn": z.clone(),
            "ent_i": z.clone(), "ent_t": z.clone(),
            "logit_sum": z.clone(), "logit_sq": z.clone(), "logit_cnt": z.clone(),
            "av_sq": z.clone(),
            "i2t": z.clone(), "i2i": z.clone(), "t2i": z.clone(), "t2t": z.clone(),
            "img_q": z.clone(), "txt_q": z.clone(),
            "head_sig": torch.zeros(b, heads, n, device=q_rot.device),
            "img_tok": torch.zeros(b, heads, n_img, device=q_rot.device),
            "i2t_tok": torch.zeros(b, heads, n_img, device=q_rot.device),
            "t2i_tok": torch.zeros(b, heads, text_len, device=q_rot.device),
        }

        for start in range(0, max_valid, chunk):
            end = min(start + chunk, max_valid)
            qc = q_rot[:, start:end]
            logits = torch.einsum("bqhd,bkhd->bhqk", qc, k_rot) * scale   # [b,H,q,N]
            valid_k = valid[:, None, None, :]
            p = torch.softmax(logits.masked_fill(~valid_k, float("-inf")), dim=-1)
            qv = valid[:, start:end].float()               # [b, qc]

            ent = -(p * p.clamp_min(_EPS).log()).sum(-1)   # [b,H,qc]
            acc["entropy"] += (ent * qv[:, None, :]).sum(-1)
            acc["q_cnt"] += qv.sum(-1)[:, None].expand_as(acc["q_cnt"])
            acc["max_attn"] += (p.amax(-1) * qv[:, None, :]).sum(-1)

            lg = logits.masked_fill(~valid_k, 0.0) * qv[:, None, :, None]
            acc["logit_sum"] += lg.sum(dim=(-1, -2))
            acc["logit_sq"] += lg.pow(2).sum(dim=(-1, -2))
            acc["logit_cnt"] += (valid_k.sum(dim=(-1, -2)).reshape(b) * qv.sum(-1))[:, None]

            av = torch.einsum("bhqk,bkhd->bqhd", p, v)    # [b,qc,H,Dh]
            acc["av_sq"] += (av.pow(2).sum(-1) * qv[:, :, None]).sum(1)

            # --- 跨模态注意力质量 ---
            img_lo = max(0, min(start, n_img) - start)
            img_hi = max(0, min(end, n_img) - start)
            if img_hi > img_lo:
                p_img = p[:, :, img_lo:img_hi]                      # [b,H,qi,N]
                m = qv[:, img_lo:img_hi][:, None, :]
                acc["ent_i"] += (ent[:, :, img_lo:img_hi] * m).sum(-1)
                acc["i2i"] += (p_img[..., :n_img].sum(-1) * m).sum(-1)
                acc["i2t"] += (p_img[..., n_img:].sum(-1) * m).sum(-1)
                acc["img_q"] += m.sum(-1).expand_as(acc["img_q"])
                acc["img_tok"] += p_img[..., :n_img].sum(2)         # [b,H,n_img]
                acc["i2t_tok"][:, :, start + img_lo:start + img_hi] += p_img[..., n_img:].sum(-1)
            txt_lo = max(0, max(start, n_img) - start)
            txt_hi = end - start
            if txt_hi > txt_lo:
                p_txt = p[:, :, txt_lo:txt_hi]
                m = qv[:, txt_lo:txt_hi][:, None, :]
                acc["ent_t"] += (ent[:, :, txt_lo:txt_hi] * m).sum(-1)
                acc["t2i"] += (p_txt[..., :n_img].sum(-1) * m).sum(-1)
                acc["t2t"] += (p_txt[..., n_img:].sum(-1) * m).sum(-1)
                acc["txt_q"] += m.sum(-1).expand_as(acc["txt_q"])
                t0 = start + txt_lo - n_img
                acc["t2i_tok"][:, :, t0:t0 + (txt_hi - txt_lo)] += p_txt[..., :n_img].sum(-1)

            # head 签名 = 每 head 对各 key 的注意力列质量 (全 key 宽度, 不随 query chunk 切)
            acc["head_sig"] += (p * qv[:, None, :, None]).sum(2)   # [b,H,N]

        q_cnt = acc["q_cnt"].clamp_min(1)
        img_q = acc["img_q"].clamp_min(1)
        txt_q = acc["txt_q"].clamp_min(1)
        logit_mean = acc["logit_sum"] / acc["logit_cnt"].clamp_min(1)
        logit_var = (acc["logit_sq"] / acc["logit_cnt"].clamp_min(1) - logit_mean ** 2).clamp_min(0)
        ent_mean = acc["entropy"] / q_cnt
        # res_in / branch_out 是全 batch (uncond+cond), 与 valid 的分支切片对齐
        res_in, branch_out = ctx.get("res_in"), ctx.get("branch_out")
        if sel != slice(0, b_all):
            if res_in is not None and res_in.shape[0] == b_all:
                res_in = res_in[sel]
            if branch_out is not None and branch_out.shape[0] == b_all:
                branch_out = branch_out[sel]
        res_rms = self._masked_rms(res_in, valid)
        branch_rms = self._masked_rms(branch_out, valid)

        def head_mean(x):
            return x.mean(0).cpu().numpy()

        sig = acc["head_sig"].sum(0)                       # [H, N]
        sig = sig / sig.sum(-1, keepdim=True).clamp_min(_EPS)
        attn_out_rms = (acc["av_sq"] / q_cnt).sqrt()       # [b,H]

        return {
            "q_raw_rms": head_mean(q_raw.pow(2).mean(dim=(0, 1, 3)).sqrt()),
            "k_raw_rms": head_mean(k_raw.pow(2).mean(dim=(0, 1, 3)).sqrt()),
            "v_rms": head_mean(v.pow(2).mean(dim=(0, 1, 3)).sqrt()),
            "q_rms": head_mean(q_rot.pow(2).mean(dim=(0, 1, 3)).sqrt()),
            "k_rms": head_mean(k_rot.pow(2).mean(dim=(0, 1, 3)).sqrt()),
            "logit_std": head_mean(logit_var.sqrt()),
            "entropy": head_mean(ent_mean),
            "entropy_img": head_mean(acc["ent_i"] / img_q),
            "entropy_txt": head_mean(acc["ent_t"] / txt_q),
            "entropy_norm": head_mean(ent_mean / math.log(max(n, 2))),
            "eff_rank": head_mean(torch.exp(ent_mean)),
            "max_attention": head_mean(acc["max_attn"] / q_cnt),
            "attn_out_rms": head_mean(attn_out_rms),
            "img_to_text_mass": head_mean(acc["i2t"] / img_q),
            "img_to_image_mass": head_mean(acc["i2i"] / img_q),
            "txt_to_image_mass": head_mean(acc["t2i"] / txt_q),
            "txt_to_text_mass": head_mean(acc["t2t"] / txt_q),
            "residual_ratio": (float(attn_out_rms.mean().item()) / res_rms) if res_rms else None,
            "branch_proj_rms": branch_rms,
            "_head_sig": sig.cpu().numpy(),
            "_img_token_mass": acc["img_tok"].mean(0).cpu().numpy(),
            "_img_to_text": acc["i2t_tok"].mean(0).cpu().numpy(),
            "_txt_to_image": acc["t2i_tok"].mean(0).cpu().numpy(),
        }

    @staticmethod
    def _masked_rms(t, valid):
        """在有效 token 位置上算 RMS; t 为 [B,N,C], valid 为 [B,N] bool."""
        if t is None:
            return None
        if t.dim() == 4:
            t = t[:, :, 0]
        if t.shape[:2] != valid.shape:
            return None
        m = valid.unsqueeze(-1).to(t.dtype)
        num = (t.float().pow(2) * m).sum()
        den = m.expand_as(t).sum().clamp_min(1)
        return float((num / den).sqrt().item())

    # ------------------------------------------------------------------ #
    # 结果导出
    # ------------------------------------------------------------------ #
    def save(self, out_dir) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        npz: dict[str, np.ndarray] = {}
        summary = {"branch": self.branch,
                   "grid": list(self.grid) if self.grid else None,
                   "meta": self.meta,
                   "steps": []}
        for step in self._steps:
            s = step["step"]
            row = {"step": s, "sigma": step["sigma"], "layers": {}}
            sigs = []
            for lname, st in sorted(step["layers"].items(), key=lambda kv: int(kv[0])):
                row["layers"][lname] = {}
                head_sig = st.pop("_head_sig", None)
                itm = st.pop("_img_token_mass", None)
                i2t = st.pop("_img_to_text", None)
                t2i = st.pop("_txt_to_image", None)
                for k, val in st.items():
                    if val is None:
                        continue
                    arr = np.asarray(val)
                    npz[f"s{s}_L{lname}_{k}"] = arr
                    row["layers"][lname][k] = (float(arr) if arr.ndim == 0
                                              else [float(x) for x in arr.tolist()])
                if head_sig is not None:
                    npz[f"s{s}_L{lname}_head_sig"] = head_sig.astype(np.float32)
                    norm = head_sig / (np.linalg.norm(head_sig, axis=1, keepdims=True) + _EPS)
                    npz[f"s{s}_L{lname}_head_cos"] = (norm @ norm.T).astype(np.float32)
                    sigs.append(head_sig.mean(0))
                if itm is not None:
                    npz[f"s{s}_L{lname}_img_token_mass"] = itm.astype(np.float32)
                if i2t is not None:
                    npz[f"s{s}_L{lname}_img_to_text"] = i2t.astype(np.float32)
                if t2i is not None:
                    npz[f"s{s}_L{lname}_txt_to_image"] = t2i.astype(np.float32)
            if len(sigs) > 1:
                arr = np.stack(sigs)
                norm = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + _EPS)
                npz[f"s{s}_layer_cos"] = (norm @ norm.T).astype(np.float32)
            summary["steps"].append(row)
        np.savez_compressed(out_dir / "attn_profile.npz", **npz)
        (out_dir / "attn_profile.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            self.render_html(summary, npz, out_dir / "attn_profile.html")
        except Exception as exc:  # 图表失败不影响推理与原始数据落盘
            print(f"[attn_profile] HTML 渲染失败: {type(exc).__name__}: {exc}", flush=True)
        return summary

    # ------------------------------------------------------------------ #
    # 控制台摘要
    # ------------------------------------------------------------------ #
    @staticmethod
    def render_summary(summary: dict, max_steps: int | None = None) -> str:
        lines = ["", "── Attention Profiling ──"]
        steps = summary["steps"]
        if max_steps is not None and len(steps) > max_steps:
            idx = np.linspace(0, len(steps) - 1, max_steps).round().astype(int)
            steps = [steps[i] for i in dict.fromkeys(idx.tolist())]
        for step in steps:
            lines.append(f"step {step['step']:>3}  sigma={step['sigma']:.4f}")
            lines.append(f"{'layer':>5} {'entropy':>8} {'ent_norm':>9} {'eff_rank':>9} "
                         f"{'max_attn':>9} {'logit_std':>10} {'i2t':>7} {'t2i':>7} "
                         f"{'i2i':>7} {'t2t':>7} {'res_ratio':>10}")
            for lname, st in step["layers"].items():
                def m(key):
                    v = st.get(key)
                    return float(np.mean(v)) if v is not None else float("nan")
                lines.append(
                    f"{lname:>5} {m('entropy'):>8.4f} {m('entropy_norm'):>9.4f} "
                    f"{m('eff_rank'):>9.2f} {m('max_attention'):>9.4f} "
                    f"{m('logit_std'):>10.3f} {m('img_to_text_mass'):>7.4f} "
                    f"{m('txt_to_image_mass'):>7.4f} {m('img_to_image_mass'):>7.4f} "
                    f"{m('txt_to_text_mass'):>7.4f} {m('residual_ratio'):>10.4f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # HTML 图表 (纯 SVG 自包含单文件, 不依赖 matplotlib)
    # ------------------------------------------------------------------ #
    @staticmethod
    def render_html(summary: dict, npz_data: dict, out_path) -> None:
        steps = summary["steps"]
        layers = sorted({int(l) for st in steps for l in st["layers"]})
        colors = {f"L{l}": _hsl(i, len(layers)) for i, l in enumerate(layers)}
        sigmas = [float(st["sigma"]) for st in steps]
        x_vals = sigmas if all(s is not None for s in sigmas) else list(range(len(steps)))
        last = steps[-1]

        def layer_dict(st, l):
            # 内存中 layer 键为 int, 落盘 JSON 后为 str; 两者兼容
            d = st["layers"]
            return d.get(l) or d.get(str(l)) or {}

        def curve(key):
            series = {}
            for l in layers:
                ys = []
                for st in steps:
                    v = layer_dict(st, l).get(key)
                    ys.append(float(np.mean(v)) if isinstance(v, list)
                              else (None if v is None else float(v)))
                if any(y is not None for y in ys):
                    series[f"L{l}"] = ys
            return series

        sections: dict[str, list] = {}

        def add(section, svg, legend=""):
            if svg:
                sections.setdefault(section, []).append(
                    f'<div class="card">{svg}{legend}</div>')

        def card_lines(key, title):
            s = curve(key)
            if s:
                cols = {k: colors[k] for k in s}
                add("curves", _svg_lines(x_vals, s, cols, title), _legend(s, cols))

        # 1) 集中度 / 激活尺度随 σ 变化 (每层一条线)
        card_lines("entropy_norm", "归一化熵 (低=集中, 高=分散)")
        card_lines("eff_rank", "有效关注 token 数 exp(H)")
        card_lines("max_attention", "最大注意力 (过度集中检查)")
        card_lines("logit_std", "logit std (过尖锐检查)")
        card_lines("residual_ratio", "残差比 RMS(AV)/RMS(x)")
        # 2) 跨模态注意力质量
        card_lines("img_to_text_mass", "Img → Text (条件跟随)")
        card_lines("img_to_image_mass", "Img → Img (空间关系)")
        card_lines("txt_to_image_mass", "Text → Img")
        card_lines("txt_to_text_mass", "Text → Text (文本重组)")
        # 3) Layer×Head 熵热图, 按 query 的 token 类型区分 (首/中/尾 step)
        _ENT_KEYS = (("entropy", "全部 query"), ("entropy_img", "Image query"),
                     ("entropy_txt", "Text query"))
        for si in sorted({0, len(steps) // 2, len(steps) - 1}):
            st = steps[si]
            maps = []
            for key, lab in _ENT_KEYS:
                mat = [layer_dict(st, l).get(key) for l in layers]
                if all(m is not None for m in mat):
                    maps.append(_svg_heat(np.array(mat, dtype=np.float64),
                                f"{lab} 熵 · step {st['step']} (σ={st['sigma']:.3f})", cell=14))
            if maps:
                sections.setdefault("heat", []).append(
                    '<div class="card wide">' + "".join(maps) +
                    '<div class="note">行=层, 列=head; Image/Text query 分开统计 (低熵=该类 query 注意力集中)</div></div>')
        # 3b) per-head Img→Text mass 热图 (最后 step, 哪些 head 负责读条件)
        i2t_mat = [layer_dict(last, l).get("img_to_text_mass") for l in layers]
        if all(m is not None for m in i2t_mat):
            sections.setdefault("heat", []).append(
                '<div class="card wide">' +
                _svg_heat(np.array(i2t_mat, dtype=np.float64),
                          f"Img → Text mass per head · step {last['step']} (σ={last['sigma']:.3f})", cell=14) +
                '<div class="note">行=层, 列=head; 高亮 = 该 head 将 image query 的注意力投向 text 条件的比例高</div></div>')
        # 4) Head 冗余 (head_cos, 最后一 step)
        hcos = [ _svg_heat(a, f"L{l}", cell=15)
                 for l in layers
                 if (a := npz_data.get(f"s{last['step']}_L{l}_head_cos")) is not None ]
        if hcos:
            sections.setdefault("heat", []).append(
                '<div class="card wide">' + "".join(hcos) +
                '<div class="note">Head 余弦相似度 (最后 step): 对角线=1; 远离对角线的高亮值 = head 冗余</div></div>')
        # 5) Q/K/V RMS + logit std 柱状 (全 step 均值)
        per_layer = {l: {} for l in layers}
        for key in ("q_rms", "k_rms", "v_rms", "logit_std"):
            for l in layers:
                vals = [float(np.mean(v)) for st in steps
                        if isinstance((v := layer_dict(st, l).get(key)), list)]
                if vals:
                    per_layer[l][key] = float(np.mean(vals))
        if any(per_layer.values()):
            bars_svg, bars_legend = _svg_bars(per_layer, "Q/K/V RMS 与 logit std (全 step 均值, x 轴=层号)")
            add("bars", bars_svg, bars_legend)
        # 6) 空间注意力 (img_token_mass, 最后 step; 需要 grid)
        grid = summary.get("grid")
        if grid and grid[0] == 1:
            _, gh, gw = grid
            sp = []
            for l in layers:
                a = npz_data.get(f"s{last['step']}_L{l}_img_token_mass")
                if a is None:
                    continue
                if a.ndim == 2:  # [H, n_img] → head 均值
                    a = a.mean(0)
                if a.size == gh * gw:
                    sp.append(_svg_heat(a.reshape(gh, gw), f"L{l}", cell=12))
            if sp:
                sections.setdefault("spatial", []).append(
                    '<div class="card wide">' + "".join(sp) +
                    '<div class="note">每个 image token 被关注的总量 (head 均值, 最后 step) — 高亮区域即模型当前关注的主体</div></div>')

        meta = summary.get("meta") or {}
        meta_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in meta.items())
        grid_html = (f"<tr><td>grid (t,h,w)</td><td>{summary.get('grid')}</td></tr>"
                     if summary.get("grid") else "")
        sec_titles = {"curves": "1 · 集中度与激活尺度 vs σ (每层一条线)",
                      "heat": "2 · Layer × Head 熵热图 (区分 token 类型) 与 Head 冗余",
                      "bars": "3 · Q/K/V 激活尺度 (按层)",
                      "spatial": "4 · 空间注意力分布 (image token 被关注量)"}
        body = "".join(f"<h2>{sec_titles[k]}</h2>" + "".join(v)
                       for k, v in sections.items() if v)
        html = ("<!doctype html><html><head><meta charset='utf-8'>"
                "<title>Attention Profiling</title>"
                f"<style>{_HTML_STYLE}</style></head><body>"
                "<h1>DiT Attention Profiling</h1>"
                f'<table class="meta"><tr><td>branch</td><td>{summary.get("branch")}</td></tr>'
                f"{grid_html}{meta_rows}</table>" + body + "</body></html>")
        Path(out_path).write_text(html, encoding="utf-8")


_HTML_STYLE = """
body{background:#111418;color:#d8dee9;font-family:-apple-system,'Segoe UI',sans-serif;margin:24px;max-width:1480px}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:26px 0 8px;color:#88c0d0;border-bottom:1px solid #2a2f38;padding-bottom:6px}
.card{background:#171b21;border:1px solid #2a2f38;border-radius:10px;padding:12px 14px;margin:8px 8px 8px 0;display:inline-block;vertical-align:top}
.card.wide{display:block}
.legend{font-size:11px;color:#9aa5b1;margin-top:4px}
.legend span{margin-right:10px}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.note{font-size:11px;color:#7b8494;margin-top:6px}
table.meta{font-size:12px;color:#9aa5b1;margin:6px 0 2px;border-collapse:collapse}
table.meta td{padding:1px 16px 1px 0}
svg text{font-family:inherit}
"""


def _hsl(i, n, l=0.62):
    """层 → 颜色; 黄金角分散保证相邻层色差大."""
    hue = (i * 47) % 360
    return f"hsl({hue},72%,{int(l * 100)}%)"


_VIRIDIS = ((0.267, 0.005, 0.329), (0.283, 0.141, 0.458), (0.254, 0.265, 0.530),
            (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
            (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
            (0.741, 0.873, 0.150), (0.993, 0.906, 0.144))


def _viridis(t):
    t = min(max(t, 0.0), 1.0) * (len(_VIRIDIS) - 1)
    i = min(int(t), len(_VIRIDIS) - 2)
    a, b = _VIRIDIS[i], _VIRIDIS[i + 1]
    f = t - i
    return "rgb(%d,%d,%d)" % tuple(int(255 * (a[k] + (b[k] - a[k]) * f)) for k in range(3))


def _svg_lines(x_vals, series, colors, title, w=640, h=250):
    """多序列折线图; series: {name: [y per x]}, y 可含 None (断线)."""
    ml, mr, mt, mb = 56, 16, 30, 36
    pw, ph = w - ml - mr, h - mt - mb
    pts = [(y) for ys in series.values() for y in ys if y is not None]
    if not pts:
        return ""
    lo, hi = min(pts), max(pts)
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    n = max(len(x_vals) - 1, 1)

    def X(i):
        return ml + pw * i / n

    def Y(v):
        return mt + ph * (1 - (v - lo) / (hi - lo))

    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
           f'<text x="{ml}" y="18" fill="#88c0d0" font-size="13">{title}</text>']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = Y(v)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{w - mr}" y2="{y:.1f}" stroke="#242a33"/>')
        out.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" fill="#7b8494" font-size="10" '
                   f'text-anchor="end">{v:.3g}</text>')
    for i in sorted({0, len(x_vals) // 2, len(x_vals) - 1}):
        out.append(f'<text x="{X(i):.1f}" y="{h - 12}" fill="#7b8494" font-size="10" '
                   f'text-anchor="middle">σ={x_vals[i]:.3f}</text>')
    for name, ys in series.items():
        segs, cur = [], []
        for i, y in enumerate(ys):
            if y is None:
                if cur:
                    segs.append(cur)
                cur = []
            else:
                cur.append(f"{X(i):.1f},{Y(y):.1f}")
        if cur:
            segs.append(cur)
        for seg in segs:
            if len(seg) > 1:
                out.append(f'<polyline points="{" ".join(seg)}" fill="none" '
                           f'stroke="{colors.get(name, "#b48ead")}" stroke-width="1.8"/>')
    out.append("</svg>")
    return "".join(out)


def _legend(series, colors):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{colors[k]}"></i>{k}</span>' for k in series) + "</div>"


def _svg_heat(mat, title, cell=16):
    """热力图; mat: [rows, cols], hover 显示数值."""
    m = np.asarray(mat, dtype=np.float64)
    r, c = m.shape
    lo, hi = float(m.min()), float(m.max())
    rng = (hi - lo) or 1.0
    w, h = c * cell + 8, r * cell + 40
    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
           f'<text x="4" y="14" fill="#88c0d0" font-size="12">{title}</text>']
    for i in range(r):
        for j in range(c):
            out.append(f'<rect x="{4 + j * cell}" y="{20 + i * cell}" width="{cell - 1}" '
                       f'height="{cell - 1}" rx="1.5" fill="{_viridis((float(m[i, j]) - lo) / rng)}">'
                       f'<title>[{i},{j}] {m[i, j]:.4g}</title></rect>')
    out.append(f'<text x="4" y="{h - 6}" fill="#7b8494" font-size="10">[{lo:.3g}, {hi:.3g}]</text></svg>')
    return "".join(out)


_BAR_PALETTE = {"q_rms": "#5e81ac", "k_rms": "#88c0d0", "v_rms": "#a3be8c",
                "logit_std": "#d08770", "q_raw_rms": "#7b88a1", "k_raw_rms": "#6f9aa8"}


def _svg_bars(per_layer, title, w=640, h=250):
    """分组柱状图; per_layer: {layer: {series: value}}, x 轴 = 层号."""
    layers = sorted(l for l in per_layer if per_layer[l])
    names = [n for n in _BAR_PALETTE if any(n in per_layer[l] for l in layers)]
    if not layers or not names:
        return "", ""
    vals = [per_layer[l][n] for l in layers for n in names if n in per_layer[l]]
    hi = max(vals) or 1.0
    ml, mr, mt, mb = 44, 10, 28, 30
    pw, ph = w - ml - mr, h - mt - mb
    gw = pw / len(layers)
    bw = gw * 0.72 / len(names)
    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
           f'<text x="{ml}" y="16" fill="#88c0d0" font-size="13">{title}</text>']
    for k in range(5):
        v = hi * k / 4
        y = mt + ph * (1 - k / 4)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{w - mr}" y2="{y:.1f}" stroke="#242a33"/>')
        out.append(f'<text x="{ml - 5}" y="{y + 4:.1f}" fill="#7b8494" font-size="10" '
                   f'text-anchor="end">{v:.3g}</text>')
    step = max(1, len(layers) // 16)
    for gi, l in enumerate(layers):
        for si, name in enumerate(names):
            if name not in per_layer[l]:
                continue
            v = per_layer[l][name]
            bh = ph * v / hi
            x = ml + gi * gw + gw * 0.14 + si * bw
            out.append(f'<rect x="{x:.1f}" y="{mt + ph - bh:.1f}" width="{bw - 1.2:.1f}" '
                       f'height="{bh:.1f}" fill="{_BAR_PALETTE[name]}">'
                       f'<title>L{l} {name}={v:.4g}</title></rect>')
        if gi % step == 0:
            out.append(f'<text x="{ml + gi * gw + gw / 2:.1f}" y="{h - 10}" fill="#7b8494" '
                       f'font-size="9" text-anchor="middle">{l}</text>')
    out.append("</svg>")
    legend = _legend(names, _BAR_PALETTE)
    return "".join(out), legend
