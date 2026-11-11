"""Giga-Exp FFN MoE: top-2 路由 MoE FFN, 对齐 moe-h3 SparseMoE/Router 行为。

关键设计 (源自 moe-h3, 按 giga DiT 特性适配):
- router 全程 fp32: gate 权重常驻 fp32 (cast_router_fp32 在模型 .to(bf16) 后回浮),
  输入 cast float32 计算 logits, softmax/topk/aux loss 全 fp32。
- 强制均衡调度 RoutingSchedule 三阶段:
    A 期 (< phase_a_steps)   : 路由完全强制均衡 —— balanced random top-2 (每 expert 均匀命中),
                               与 gate logits 无关。dense→MoE 热启动时所有 expert 是同一份 FFN
                               的拷贝, 强制均衡保证各 expert 拿到均匀 token 流量后开始分化。
    B 期 (< phase_b_end_step): 按 learned_probability 线性递增混入学习路由, jitter 线性衰减;
                               warmup_ce_weight>0 时对 router logits 施加指向均衡目标的 CE。
    C 期 (之后)              : 纯学习路由 + 小 jitter (0.03), 推理恒用学习路由。
- aux loss 四件套 (权重在 MoEFFN 内乘好, 训练侧直接把各层 aux 加进总 loss):
    load_balance (Switch, 1e-2) + router_z_loss (1e-3) + entropy (默认 0) + warmup_ce (默认 0)。
- dispatch 按专家 argsort 分组计算, 布尔掩码选择 token, 空 expert 走空张量前向
  (无 host 同步, 该 expert 参数梯度自动为零, ZeRO allreduce 归零处理)。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn


class RoutingPhase(str, Enum):
    A = "A"   # 强制均衡 (balanced random top-2)
    B = "B"   # 学习路由渐进混入
    C = "C"   # 纯学习路由


@dataclass(frozen=True)
class RoutingSchedule:
    """与 moe-h3 RoutingSchedule 一致: A 期强制均衡 → B 期线性过渡 → C 期学习。"""

    phase_a_steps: int = 500
    phase_b_end_step: int = 2000
    phase_a_jitter: float = 0.5
    phase_b_jitter_start: float = 0.5
    phase_b_jitter_end: float = 0.1
    phase_c_jitter: float = 0.03

    def at(self, step: int) -> tuple[RoutingPhase, float, float]:
        if step < self.phase_a_steps:
            return RoutingPhase.A, 0.0, self.phase_a_jitter
        if step < self.phase_b_end_step:
            length = max(self.phase_b_end_step - self.phase_a_steps, 1)
            progress = min(max((step - self.phase_a_steps) / length, 0.0), 1.0)
            jitter = self.phase_b_jitter_start + progress * (self.phase_b_jitter_end - self.phase_b_jitter_start)
            return RoutingPhase.B, progress, jitter
        return RoutingPhase.C, 1.0, self.phase_c_jitter


def load_balancing_loss(logits, indices, num_experts, valid_mask=None):
    """Switch 风格负载均衡: E · Σ_i P_i · f_i (P=router 概率占比, f=token 分配占比)。"""
    probabilities = logits.float().softmax(dim=-1)
    assignments = F.one_hot(indices, num_classes=num_experts).float().mean(dim=-2)
    if valid_mask is not None:
        mask = valid_mask.reshape(-1).to(probabilities.dtype).unsqueeze(-1)
        denominator = mask.sum().clamp_min(1.0)
        probability_fraction = (probabilities * mask).sum(dim=0) / denominator
        assignment_fraction = (assignments * mask).sum(dim=0) / denominator
    else:
        probability_fraction = probabilities.mean(dim=0)
        assignment_fraction = assignments.mean(dim=0)
    return num_experts * torch.sum(probability_fraction * assignment_fraction)


def router_z_loss(logits, valid_mask=None):
    logits = logits.reshape(-1, logits.shape[-1]).float()
    if valid_mask is not None:
        mask = valid_mask.reshape(-1).bool()
        logits = logits[mask]
    if logits.numel() == 0:
        return logits.sum() * 0.0
    return torch.logsumexp(logits, dim=-1).square().mean()


def router_entropy_loss(logits, valid_mask=None):
    logits = logits.reshape(-1, logits.shape[-1]).float()
    if valid_mask is not None:
        mask = valid_mask.reshape(-1).bool()
        logits = logits[mask]
    if logits.numel() == 0:
        return logits.sum() * 0.0
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(-(log_probs.exp() * log_probs).sum(dim=-1)).mean()


def routing_cross_entropy(logits, targets, valid_mask=None):
    logits = logits.reshape(-1, logits.shape[-1]).float()
    targets = targets.reshape_as(logits).float()
    if valid_mask is not None:
        mask = valid_mask.reshape(-1).bool()
        logits, targets = logits[mask], targets[mask]
    if targets.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits, targets)


class MoEFFN(nn.Module):
    """替换 GigaBlock.ff 的 top-2 MoE FFN。每 expert 与原 dense FFN 同构:
    Linear(dim→ffn_dim) + SiLU + Linear(ffn_dim→dim), 无 bias。

    forward 返回 (output [B,S,dim], auxiliary_loss 标量); AdaLN 残差 gate 由 GigaBlock 侧乘,
    与 moe-h3 的 beta gate 位置一致 (collector 统计为 pre-AdaLN RMS 比)。
    """

    def __init__(self, dim, ffn_dim, num_experts=4, top_k=2,
                 load_balancing_weight=1e-2, router_z_loss_weight=1e-3,
                 router_entropy_weight=0.0, warmup_ce_weight=0.0,
                 phase_a_steps=500, phase_b_end_step=2000,
                 phase_a_jitter=0.5, phase_b_jitter_start=0.5,
                 phase_b_jitter_end=0.1, phase_c_jitter=0.03):
        super().__init__()
        if num_experts < 2:
            raise ValueError(f"num_experts 必须 >= 2, got {num_experts}")
        if top_k != 2:
            raise ValueError("当前实现要求 top_k=2 (与 moe-h3 一致)")
        self.dim, self.ffn_dim = int(dim), int(ffn_dim)
        self.num_experts, self.top_k = int(num_experts), int(top_k)
        self.load_balancing_weight = float(load_balancing_weight)
        self.router_z_loss_weight = float(router_z_loss_weight)
        self.router_entropy_weight = float(router_entropy_weight)
        self.warmup_ce_weight = float(warmup_ce_weight)
        self.schedule = RoutingSchedule(int(phase_a_steps), int(phase_b_end_step),
                                        float(phase_a_jitter), float(phase_b_jitter_start),
                                        float(phase_b_jitter_end), float(phase_c_jitter))
        # router 常驻 fp32 (cast_router_fp32 保证 .to(bf16) 后仍是 fp32)
        self.router = nn.Linear(dim, num_experts, bias=False, dtype=torch.float32)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, ffn_dim, bias=False), nn.SiLU(),
                          nn.Linear(ffn_dim, dim, bias=False))
            for _ in range(num_experts)
        ])
        self.routing_step = 0
        self.stats_collector = None
        self._stats_layer = "0"
        self.last_auxiliary_components = None
        self.last_phase = RoutingPhase.C

    def set_routing_step(self, step: int):
        self.routing_step = int(step)

    def set_stats_collector(self, collector):
        self.stats_collector = collector

    def _balanced_random_top2(self, num_tokens, device):
        """强制均衡随机 top-2: 每 expert 命中次数严格均匀 (对齐 moe-h3 Router._balanced_random_top2)。"""
        if num_tokens == 0:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        first = torch.arange(num_tokens, device=device) % self.num_experts
        first = first.index_select(0, torch.randperm(num_tokens, device=device))
        offset = torch.randint(1, self.num_experts, (), device=device)
        second = (first + offset) % self.num_experts
        indices = torch.stack((first, second), dim=-1)
        return torch.randperm(self.num_experts, device=device)[indices]

    def forward(self, x, valid_mask=None, modality=None, denoising_step=None, residual_gate=None):
        b, n, d = x.shape
        flat = x.reshape(-1, d)
        num_tokens = flat.shape[0]
        valid = valid_mask.reshape(-1).bool() if valid_mask is not None else torch.ones(num_tokens, dtype=torch.bool, device=x.device)

        # ---- router: 全 fp32 (输入 cast, gate 权重 fp32, logits/topk 均在 fp32 域) ----
        logits = self.router(flat.to(torch.float32))  # [T, E]
        phase, learned_probability, jitter = self.schedule.at(self.routing_step)
        self.last_phase = phase
        route_logits = logits
        if self.training and jitter > 0:
            route_logits = logits + torch.randn_like(logits) * jitter
        learned_indices = route_logits.topk(self.top_k, dim=-1).indices
        random_indices = None
        if self.training and phase is not RoutingPhase.C:
            random_indices = self._balanced_random_top2(num_tokens, logits.device)
            if phase is RoutingPhase.A:
                indices = random_indices  # A 期: 完全强制均衡, 与 logits 无关
            else:
                use_learned = torch.rand(num_tokens, device=logits.device) < learned_probability
                indices = torch.where(use_learned[:, None], learned_indices, random_indices)
        else:
            indices = learned_indices  # eval 恒用学习路由 (无 jitter)

        weights = route_logits.gather(-1, indices).float().softmax(dim=-1).to(flat.dtype)  # [T, k]

        # ---- aux loss (fp32, 权重乘好) ----
        load_balance = self.load_balancing_weight * load_balancing_loss(logits, indices, self.num_experts, valid)
        z_loss = self.router_z_loss_weight * router_z_loss(logits, valid)
        entropy = logits.new_zeros(())
        if self.router_entropy_weight:
            entropy = self.router_entropy_weight * router_entropy_loss(logits, valid)
        warmup_ce = logits.new_zeros(())
        if self.training and self.warmup_ce_weight and random_indices is not None:
            targets = torch.zeros_like(logits).scatter_(1, random_indices, 0.5)
            ce_weight = self.warmup_ce_weight
            if phase is RoutingPhase.B:
                ce_weight *= 1.0 - learned_probability
            warmup_ce = ce_weight * routing_cross_entropy(logits, targets, valid)
        auxiliary_loss = load_balance + z_loss + entropy + warmup_ce
        self.last_auxiliary_components = {
            "load_balance": load_balance.detach(), "z_loss": z_loss.detach(),
            "entropy": entropy.detach(), "warmup_ce": warmup_ce.detach(),
        }

        # ---- dispatch: 按 expert 排序分组, 布尔掩码选 token (无 host 同步) ----
        flat_indices = indices.reshape(-1, self.top_k)          # [T, k]
        flat_weights = weights.reshape(-1, self.top_k)
        dispatch_tokens = torch.arange(num_tokens, device=x.device)[:, None].expand(-1, self.top_k).reshape(-1)
        dispatch_slots = torch.arange(self.top_k, device=x.device).expand(num_tokens, -1).reshape(-1)
        dispatch_experts = flat_indices.reshape(-1)
        keep = valid.index_select(0, dispatch_tokens)
        dispatch_tokens, dispatch_slots, dispatch_experts = (dispatch_tokens[keep], dispatch_slots[keep], dispatch_experts[keep])
        order = dispatch_experts.argsort()
        dispatch_tokens = dispatch_tokens.index_select(0, order)
        dispatch_slots = dispatch_slots.index_select(0, order)
        dispatch_experts = dispatch_experts.index_select(0, order)

        output = torch.zeros_like(flat)
        rms_sum = torch.zeros(self.num_experts, dtype=torch.float32, device=x.device)
        for expert_id in range(self.num_experts):
            sel = dispatch_experts == expert_id
            ids = dispatch_tokens[sel]
            slots = dispatch_slots[sel]
            expert_output = self.experts[expert_id](flat.index_select(0, ids))
            with torch.no_grad():
                # 空 expert: sum(空)=0, 不产生 host 同步
                rms_sum[expert_id] = expert_output.detach().float().square().mean(-1).sqrt().sum()
            output.index_add_(0, ids, expert_output * flat_weights[ids, slots].unsqueeze(-1))

        output = output.view_as(x)
        if valid_mask is not None:
            output = output * valid_mask.to(output.dtype).reshape(b, n, 1)

        if self.stats_collector is not None:
            identity = torch.arange(num_tokens, device=x.device)  # 位置恒等: ODE 各步 token 布局不变
            self.stats_collector.update(
                self._stats_layer, logits, valid_mask=valid, topk_indices=indices,
                gate=residual_gate, expert_output_rms_sum=rms_sum,
                expert_output_rms_count=torch.bincount(dispatch_experts, minlength=self.num_experts).to(torch.float32),
                moe_output=output, modality=modality, denoising_step=denoising_step,
                token_ids=identity,
            )
        return output, auxiliary_loss


def apply_dense_to_moe_migration(current, state):
    """dense → MoE 热启动迁移 (强制均衡初始化): 旧 dense ckpt 的 blocks.{i}.ff.{0,2}.weight
    拷贝到该层全部 experts (每 expert 同一 dense FFN), 移除旧 ff.* 键; router gate 无对应
    权重保持随机初始化 (fp32)。

    - current: MoE 模型期望的 state_dict (只读, 判断模型是否 MoE);
    - state:   checkpoint state_dict (就地修改);
    - 返回 {"upcycled": [层号], "already_moe": bool, "model_moe": bool}
    """
    result = {"upcycled": [], "already_moe": False, "model_moe": False}
    layers = sorted({int(key.split(".")[1]) for key in current if ".ff.experts." in key})
    if not layers:
        return result
    result["model_moe"] = True
    first = layers[0]
    if f"blocks.{first}.ff.experts.0.0.weight" in state:
        result["already_moe"] = True
        return result
    # "blocks.{i}.ff.experts.{e}.{sub}.weight" → split(".") 索引 4 是 expert 编号
    num_experts = len({key.split(".")[4] for key in current if key.startswith(f"blocks.{first}.ff.experts.")})
    for i in layers:
        w0, w2 = state.get(f"blocks.{i}.ff.0.weight"), state.get(f"blocks.{i}.ff.2.weight")
        if w0 is None or w2 is None:
            continue  # 无 dense FFN 可拷贝: experts 保持随机初始化
        for e in range(num_experts):
            state[f"blocks.{i}.ff.experts.{e}.0.weight"] = w0
            state[f"blocks.{i}.ff.experts.{e}.2.weight"] = w2
        state.pop(f"blocks.{i}.ff.0.weight", None)
        state.pop(f"blocks.{i}.ff.2.weight", None)
        result["upcycled"].append(i)
    return result
