"""Giga-Exp MoE 统计收集器: 适配自 moe-h3 models/moe/statistics.py + distributed_stats.py。

- GPU 常驻流式累加, update 只收 router logits / topk / gate / expert RMS, 不保留 token 级路由;
- modality 简化为 image(0)/text(1) (giga DiT 双模态);
- lightweight (训练日志) / full (validation 出图) 两档;
- reduce_collector 在 ZeRO-1 数据并行下对 WORLD 组 all_reduce SUM (各 rank 看到不同 token)。
"""
import math
from typing import Any

import torch
import torch.distributed as dist

_MODALITIES = {0: "image", 1: "text"}


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().square().mean().sqrt()


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return _json_value(value.item() if value.numel() == 1 else value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class MoEStatsCollector:
    MODES = {"disabled", "lightweight", "full"}

    def __init__(self, mode: str = "lightweight", top_k: int = 2, dead_expert_threshold: float = 0.0):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}, got {mode!r}")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.mode = mode
        self.top_k = top_k
        self.dead_expert_threshold = dead_expert_threshold
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    def reset(self) -> None:
        self._layers: dict[str, dict[str, Any]] = {}
        self._previous_routes: dict[str, tuple[Any, tuple[int, ...], torch.Tensor, Any]] = {}

    def _layer(self, name: Any, experts: int, device: torch.device) -> dict[str, Any]:
        key = str(name)
        state = self._layers.get(key)
        if state is None:
            zero = lambda: torch.zeros((), dtype=torch.float64, device=device)
            state = {
                "assignment_count": torch.zeros(experts, dtype=torch.float64, device=device),
                "valid_token_count": zero(),
                "entropy_sum": zero(),
                "top1_prob_sum": zero(),
                "top2_mass_sum": zero(),
                "gap_sum": zero(),
                "gate_sum": zero(),
                "gate_count": zero(),
                "expert_output_rms_sum": torch.zeros(experts, dtype=torch.float64, device=device),
                "expert_output_rms_count": torch.zeros(experts, dtype=torch.float64, device=device),
                "modality_assignment_count": torch.zeros(2, experts, dtype=torch.float64, device=device),
                "denoising_step_assignment_count": {},
                "moe_rms_sum": zero(),
                "ffn_rms_sum": zero(),
                "rms_count": torch.zeros(4, dtype=torch.float64, device=device),
                "routing_stability_sum": zero(),
                "routing_stability_count": zero(),
                "routing_stability_na": zero(),
            }
            self._layers[key] = state
        elif state["assignment_count"].numel() != experts:
            raise ValueError(f"layer {key!r} changed expert count")
        return state

    @torch.no_grad()
    def update(
        self,
        layer: Any,
        router_logits: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        expert_outputs: torch.Tensor | None = None,
        expert_output_rms_sum: torch.Tensor | None = None,
        expert_output_rms_count: torch.Tensor | None = None,
        modality: torch.Tensor | None = None,
        denoising_step: Any = None,
        moe_output: torch.Tensor | None = None,
        ffn_output: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
    ) -> None:
        if not self.enabled:
            return
        if router_logits.ndim < 2:
            raise ValueError("router_logits must have token dimensions and an expert dimension")
        experts = router_logits.shape[-1]
        k = min(self.top_k, experts)
        flat_logits = router_logits.detach().reshape(-1, experts).float()
        if valid_mask is None:
            mask = torch.ones(flat_logits.shape[0], dtype=torch.bool, device=flat_logits.device)
        else:
            mask = valid_mask.detach().reshape(-1).to(device=flat_logits.device, dtype=torch.bool)
            if mask.numel() != flat_logits.shape[0]:
                raise ValueError("valid_mask must match router_logits token dimensions")
        state = self._layer(layer, experts, flat_logits.device)
        valid_count = mask.sum()
        if not bool(valid_count):
            return

        logits = flat_logits[mask]
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        if topk_indices is None:
            top_values, routes = probs.topk(k, dim=-1)
        else:
            routes = topk_indices.detach().reshape(-1, topk_indices.shape[-1]).to(flat_logits.device)[mask, :k]
            top_values = probs.gather(-1, routes)
        assignments = torch.bincount(routes.reshape(-1), minlength=experts).to(torch.float64)
        state["assignment_count"].add_(assignments)
        state["valid_token_count"].add_(valid_count)
        state["entropy_sum"].add_((-(probs * log_probs).sum(-1)).sum())
        state["top1_prob_sum"].add_(top_values[:, 0].sum())
        state["top2_mass_sum"].add_(top_values[:, : min(2, k)].sum())
        if experts > 1:
            best_two = probs.topk(2, dim=-1).values
            state["gap_sum"].add_((best_two[:, 0] - best_two[:, 1]).sum())

        if gate is not None:
            flat_gate = gate.detach().float()
            if flat_gate.ndim and flat_gate.shape[:-1] == router_logits.shape[:-1]:
                flat_gate = flat_gate.reshape(-1, flat_gate.shape[-1])[mask]
            elif flat_gate.numel() == flat_logits.shape[0]:
                flat_gate = flat_gate.reshape(-1)[mask]
            state["gate_sum"].add_(flat_gate.sum())
            state["gate_count"].add_(flat_gate.numel())

        if self.mode != "full":
            return
        if modality is not None:
            tags = modality.detach().reshape(-1).to(flat_logits.device)[mask]
            for tag in range(2):
                selected = routes[tags == tag]
                if selected.numel():
                    state["modality_assignment_count"][tag].add_(
                        torch.bincount(selected.reshape(-1), minlength=experts).to(torch.float64)
                    )
        if denoising_step is not None:
            step_key = str(denoising_step.item() if isinstance(denoising_step, torch.Tensor) and denoising_step.numel() == 1 else denoising_step)
            step_counts = state["denoising_step_assignment_count"].setdefault(
                step_key, torch.zeros(experts, dtype=torch.float64, device=flat_logits.device)
            )
            step_counts.add_(assignments)
        if expert_output_rms_sum is not None and expert_output_rms_count is not None:
            state["expert_output_rms_sum"].add_(expert_output_rms_sum.detach().to(torch.float64))
            state["expert_output_rms_count"].add_(expert_output_rms_count.detach().to(torch.float64))
        else:
            self._update_expert_rms(state, expert_outputs, routes, mask, experts)
        if moe_output is not None and ffn_output is not None:
            moe_values = moe_output.detach().reshape(-1, moe_output.shape[-1])[mask].float()
            ffn_values = ffn_output.detach().reshape(-1, ffn_output.shape[-1])[mask].float()
            state["moe_rms_sum"].add_(_rms(moe_values))
            state["ffn_rms_sum"].add_(_rms(ffn_values))
            state["rms_count"][0].add_(1)
            if gate is not None:
                gain = gate.detach().float()
                if gain.ndim and gain.shape == moe_output.shape[:-1]:
                    gain = gain.unsqueeze(-1)
                gated = (moe_output.detach().float() * gain).reshape(-1, moe_output.shape[-1])[mask]
                state["rms_count"][1].add_(_rms(gated))
                state["rms_count"][2].add_(_rms(ffn_values))
                state["rms_count"][3].add_(1)
        self._update_stability(str(layer), state, routes, mask, denoising_step, token_ids)

    collect = update
    __call__ = update

    def _update_expert_rms(self, state, outputs, routes, mask, experts):
        if outputs is None:
            return
        values = outputs.detach().float()
        if values.ndim >= 3 and values.shape[-2] == experts:
            values = values.reshape(-1, experts, values.shape[-1])[mask]
            rms = values.square().mean(-1).sqrt().sum(0).to(torch.float64)
            state["expert_output_rms_sum"].add_(rms)
            state["expert_output_rms_count"].add_(mask.sum())
        elif values.ndim >= 3 and values.shape[-2] == routes.shape[-1]:
            values = values.reshape(-1, routes.shape[-1], values.shape[-1])[mask]
            rms = values.square().mean(-1).sqrt().reshape(-1)
            ids = routes.reshape(-1)
            state["expert_output_rms_sum"].scatter_add_(0, ids, rms.to(torch.float64))
            state["expert_output_rms_count"].add_(torch.bincount(ids, minlength=experts))

    def _update_stability(self, layer, state, routes, mask, step, token_ids):
        if step is None:
            return
        identity = None
        if token_ids is not None:
            identity = token_ids.detach().reshape(-1).to(routes.device)[mask]
        previous = self._previous_routes.get(layer)
        shape = tuple(routes.shape)
        reliable = identity is not None and previous is not None and previous[0] is not None
        if previous is not None and reliable:
            previous_ids, previous_shape, previous_routes, previous_step = previous
            try:
                adjacent = float(step) == float(previous_step) + 1
            except (TypeError, ValueError):
                adjacent = False
            reliable = (adjacent and previous_ids.shape == identity.shape
                        and torch.equal(previous_ids, identity))
            if reliable and previous_shape == shape:
                overlap = (routes.unsqueeze(-1) == previous_routes.unsqueeze(-2)).any(-1).float().mean(-1)
                state["routing_stability_sum"].add_(overlap.sum())
                state["routing_stability_count"].add_(overlap.numel())
            else:
                state["routing_stability_na"].add_(1)
        elif previous is not None:
            state["routing_stability_na"].add_(1)
        self._previous_routes[layer] = (identity.clone() if identity is not None else None, shape, routes.clone(), step)

    def tensor_state(self) -> dict[str, dict[str, Any]]:
        return self._layers

    def summary(self) -> dict[str, Any]:
        if not self.enabled:
            return {"mode": "disabled", "layers": {}}
        layers = {}
        for name, state in self._layers.items():
            assignments = state["assignment_count"]
            valid = state["valid_token_count"]
            total_assignments = assignments.sum()
            coverage = assignments / valid.clamp_min(1)
            share = assignments / total_assignments.clamp_min(1)
            mean_load = assignments.mean()
            load_cv = assignments.std(unbiased=False) / mean_load.clamp_min(torch.finfo(torch.float64).eps)
            dead = assignments <= self.dead_expert_threshold
            denom = valid.clamp_min(1)
            item = {
                "assignment_count": assignments,
                "valid_token_count": valid,
                "assignment_coverage": coverage,
                "assignment_share": share,
                "load_cv": load_cv,
                "router_entropy": state["entropy_sum"] / denom,
                "top1_prob": state["top1_prob_sum"] / denom,
                "top2_mass": state["top2_mass_sum"] / denom,
                "top1_top2_gap": state["gap_sum"] / denom,
                "dead_expert": dead,
                "dead_expert_count": dead.sum(),
                "gate": state["gate_sum"] / state["gate_count"].clamp_min(1),
            }
            if self.mode == "full":
                counts = state["expert_output_rms_count"]
                item.update({
                    "expert_output_rms": torch.where(counts > 0, state["expert_output_rms_sum"] / counts.clamp_min(1), torch.nan),
                    "modality_usage": {
                        _MODALITIES[tag]: state["modality_assignment_count"][tag] / state["modality_assignment_count"][tag].sum().clamp_min(1)
                        for tag in range(2)
                    },
                    "denoising_step_usage": {
                        step: values / values.sum().clamp_min(1)
                        for step, values in state["denoising_step_assignment_count"].items()
                    },
                    "moe_ffn_rms_ratio": (state["moe_rms_sum"] / state["ffn_rms_sum"]
                                          if bool(state["ffn_rms_sum"] > 0) else None),
                    "gated_moe_ffn_rms_ratio": (state["rms_count"][1] / state["rms_count"][2]
                                                if bool(state["rms_count"][2] > 0) else None),
                    "routing_stability": (
                        state["routing_stability_sum"] / state["routing_stability_count"]
                        if bool(state["routing_stability_count"]) else None
                    ),
                    "routing_stability_na_count": state["routing_stability_na"],
                })
            layers[name] = _json_value(item)
        return {"mode": self.mode, "top_k": self.top_k, "layers": layers,
                "rms_comparison": "sum(update RMS(beta * moe_output)) / sum(update RMS(ffn_output)); raw omits beta; pre-AdaLN residual gate, not exact final residual contribution",
                "routing_stability_identity": "position_ids",
                "routing_stability_definition": "per-layer token-weighted mean adjacent observed-step Top-k overlap"}


@torch.no_grad()
def reduce_collector(collector: MoEStatsCollector, group=None) -> MoEStatsCollector:
    """ZeRO-1 数据并行: 对 WORLD 组 all_reduce SUM 所有累加张量 (各 rank 看到不同 token)。"""
    active = group is not None or (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)
    if not active:
        return collector
    expert_counts = {name: state["assignment_count"].numel() for name, state in collector.tensor_state().items()}
    for name, state in collector.tensor_state().items():
        experts = expert_counts[name]
        for value in state.values():
            tensors = []
            if isinstance(value, torch.Tensor):
                tensors = [value]
            elif isinstance(value, dict):
                tensors = [t for t in value.values() if isinstance(t, torch.Tensor)]
            for tensor in tensors:
                # 标量/4 元素计数向量和 [E] 向量都直接求和; 模态矩阵 [2,E] 也求和
                if tensor.ndim == 0 or tensor.shape[-1] == experts or tensor.ndim <= 1:
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return collector
