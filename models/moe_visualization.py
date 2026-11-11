"""Giga-Exp MoE validation 图表: 适配自 moe-h3 models/moe/visualization.py。

输入是 MoEStatsCollector.summary() 产生的 JSON; 输出 per-layer 用量热图、负载 CV、
router 熵、dead expert、routing stability、gate、expert RMS、模态用量与 3x3 dashboard。
"""
import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISCLAIMER = ("Statistics / weight changes do not imply task-quality improvement; a single checkpoint cannot establish a trend.\n"
              "Independent fixed-input, fixed-seed MoE on/off quality validation required (not run here).")
RATIO_NOTE = "Pre-AdaLN residual gate: MoE / dense-FFN RMS ratio unavailable (dense branch removed in giga MoE)"
MODALITIES = ("image", "text")


def _layers(summary):
    layers = summary.get("layers", {})
    if not layers:
        raise ValueError("summary contains no layer statistics")
    names = sorted(layers, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))
    return names, [layers[name] for name in names]


def _save(fig, output, formats=("png", "svg")):
    base = Path(output)
    base.parent.mkdir(parents=True, exist_ok=True)
    destinations = [base] if base.suffix.lower() in {".png", ".svg"} else [base.with_suffix(f".{fmt}") for fmt in formats]
    for destination in destinations:
        fig.savefig(destination, bbox_inches="tight")
    return destinations


def _finish(fig, output):
    fig.text(.5, .01, DISCLAIMER, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .96))
    try:
        return _save(fig, output)
    finally:
        plt.close(fig)


def _number(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else float("nan")


def _metric(summary, layers, metric):
    if metric == "routing_stability" and summary.get("routing_stability_identity") != "position_ids":
        return [float("nan")] * len(layers)
    return [_number(layer.get(metric)) for layer in layers]


def _ticks(ax, names, axis="x"):
    stride = max(1, len(names) // 12)
    indices = list(range(0, len(names), stride))
    if axis == "x":
        ax.set_xticks(indices, [names[i] for i in indices], rotation=60)
    else:
        ax.set_yticks(indices, [names[i] for i in indices])


def _line(ax, summary, metric, title, ylabel):
    names, layers = _layers(summary)
    values = _metric(summary, layers, metric)
    ax.plot(range(len(names)), values, marker=".")
    ax.set(title=title, xlabel="Layer", ylabel=ylabel)
    _ticks(ax, names)
    if metric == "routing_stability":
        ax.set_ylim(0, 1)
    if not any(math.isfinite(v) for v in values):
        ax.text(.5, .5, "N/A: unavailable / identity unverified", ha="center", transform=ax.transAxes, fontsize=8)


def _heatmap(ax, summary, metric, title, unit):
    names, layers = _layers(summary)
    experts = len(layers[0]["assignment_share"])
    matrix = [[_number(v) for v in layer.get(metric, [None] * experts)] for layer in layers]
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set(title=title, xlabel="Expert index", ylabel="Layer")
    _ticks(ax, names, "y")
    if metric == "assignment_share":
        from matplotlib.ticker import PercentFormatter
        ax.figure.colorbar(image, ax=ax, label=unit, format=PercentFormatter(1))
    else:
        ax.figure.colorbar(image, ax=ax, label=unit)


def draw_usage(ax, summary):
    _, layers = _layers(summary)
    count = len(layers[0]["assignment_share"])
    _heatmap(ax, summary, "assignment_share", f"Assignment share (uniform = 1/{count} = {1/count:.2%})",
             "Assignments / all assignments (%; layer sum = 100%)")


def draw_contribution(ax, summary):
    ax.text(.5, .5, "N/A: dense FFN branch removed (MoE replaces FFN)", ha="center", transform=ax.transAxes, fontsize=9)
    ax.set(title="Pre-AdaLN branch RMS comparison", xlabel="Layer", ylabel="RMS ratio (dimensionless)")


def _line_plot(summary, metric, output, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    _line(ax, summary, metric, title, ylabel)
    return _finish(fig, output)


def plot_usage_heatmap(summary, output):
    fig, ax = plt.subplots(figsize=(10, 9))
    draw_usage(ax, summary)
    return _finish(fig, output)


def plot_load_cv(summary, output):
    return _line_plot(summary, "load_cv", output, "Expert load CV (balanced = 0)", "CV (dimensionless)")


def plot_router_entropy(summary, output):
    return _line_plot(summary, "router_entropy", output, "Router entropy", "Entropy (nats)")


def plot_dead_experts(summary, output):
    return _line_plot(summary, "dead_expert_count", output, "Dead experts", "Expert count")


def plot_moe_ffn_ratio(summary, output):
    fig, ax = plt.subplots(figsize=(12, 6))
    draw_contribution(ax, summary)
    fig.suptitle(RATIO_NOTE, fontsize=10)
    return _finish(fig, output)


def plot_expert_rms(summary, output):
    fig, ax = plt.subplots(figsize=(10, 9))
    _heatmap(ax, summary, "expert_output_rms", "Raw expert output RMS (before gate)", "RMS (activation units)")
    return _finish(fig, output)


def plot_modality_usage(summary, output):
    names, layers = _layers(summary)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    count = len(layers[0]["assignment_share"])
    for ax, modality in zip(axes.flat, MODALITIES):
        from matplotlib.ticker import PercentFormatter
        matrix = []
        for layer in layers:
            values = layer.get("modality_usage", {}).get(modality, [])
            matrix.append(values if values and sum(_number(v) for v in values) > 0 else [float("nan")] * count)
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        ax.set(title=f"{modality}: assignment share (uniform = {1/count:.2%})", xlabel="Expert index", ylabel="Layer")
        _ticks(ax, names, "y")
        fig.colorbar(image, ax=ax, label="Within-modality assignments (%; blank = unavailable)", format=PercentFormatter(1))
    return _finish(fig, output)


def plot_routing_stability(summary, output):
    k = summary.get("top_k", 2)
    return _line_plot(summary, "routing_stability", output,
                      f"By layer: mean adjacent observed-step Top{k} overlap", "Overlap fraction [0, 1]; verified token IDs only")


def plot_gate(summary, output):
    return _line_plot(summary, "gate", output, "Mean signed MoE gate (AdaLN ff gate)", "Gate (dimensionless)")


def plot_denoising_step_usage(summary, output):
    """各去噪步 (validation ODE step 0..N-1) 的专家分配份额热图。"""
    names, layers = _layers(summary)
    steps = sorted({str(s) for layer in layers for s in layer.get("denoising_step_usage", {})}, key=lambda s: float(s))
    if not steps:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(.5, .5, "N/A: no denoising-step data", ha="center", transform=ax.transAxes)
        return _finish(fig, output)
    experts = len(layers[0]["assignment_share"])
    matrix = [[[_number(v) for v in layer.get("denoising_step_usage", {}).get(s, [None] * experts)]
               for s in steps]
              for layer in layers]
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set(title="Denoising-step expert usage (per ODE step share)", xlabel="Expert index", ylabel="Layer")
    _ticks(ax, names, "y")
    ax.set_xticks(range(len(steps)), steps)
    ax.figure.colorbar(image, ax=ax, label="Within-step assignment share")
    return _finish(fig, output)


def plot_dashboard(summary, output):
    fig, axes = plt.subplots(3, 3, figsize=(22, 17))
    draw_usage(axes[0, 0], summary)
    for ax, metric, title, unit in (
        (axes[0, 1], "load_cv", "Load CV (balanced = 0)", "CV (dimensionless)"),
        (axes[0, 2], "router_entropy", "Router entropy", "Entropy (nats)"),
        (axes[1, 0], "dead_expert_count", "Dead experts", "Expert count"),
        (axes[1, 2], "routing_stability", f"Mean adjacent observed-step Top{summary.get('top_k', 2)} overlap", "Overlap fraction; verified IDs only"),
        (axes[2, 0], "gate", "Mean signed MoE gate", "Gate (dimensionless)")):
        _line(ax, summary, metric, title, unit)
    draw_contribution(axes[1, 1], summary)
    _heatmap(axes[2, 1], summary, "expert_output_rms", "Raw expert RMS (before gate)", "RMS (activation units)")
    names, layers = _layers(summary)
    matrix = [[int(sum(layer.get("modality_usage", {}).get(name, [])) > 0) for name in MODALITIES] for layer in layers]
    image = axes[2, 2].imshow(matrix, aspect="auto", vmin=0, vmax=1)
    axes[2, 2].set(title="Modality assignment availability (not token share)", xlabel="Modality", ylabel="Layer",
                   xticks=range(len(MODALITIES)), xticklabels=MODALITIES)
    _ticks(axes[2, 2], names, "y")
    fig.colorbar(image, ax=axes[2, 2], label="Observed assignments: 0=no / 1=yes", ticks=[0, 1])
    fig.suptitle("MoE statistics | " + RATIO_NOTE, fontsize=13)
    return _finish(fig, output)


def checkpoint_trends(validation_dir):
    rows = []
    for path in Path(validation_dir).glob("step-*/moe_metrics/summary.json"):
        step = path.parents[1].name.removeprefix("step-")
        if not step.isdigit():
            continue
        try:
            summary = json.loads(path.read_text())
            layers = list(summary.get("layers", {}).values())
            row = {"step": int(step)}
            for metric in ("load_cv", "router_entropy", "dead_expert_count", "gate", "top1_prob"):
                values = [_number(layer.get(metric)) for layer in layers]
                values = [value for value in values if math.isfinite(value)]
                row[metric] = sum(values) / len(values) if values else None
                row[metric + "_layers"] = len(values)
            rows.append(row)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return sorted(rows, key=lambda row: row["step"])


def plot_checkpoint_trends(validation_dir, output):
    rows = checkpoint_trends(validation_dir)
    if not rows:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    steps = [row["step"] for row in rows]
    for ax, metrics, title, unit in (
        (axes[0], [("load_cv", "Load CV")], "Mean layer load CV (balanced = 0)", "CV (dimensionless)"),
        (axes[1], [("router_entropy", "Router entropy"), ("top1_prob", "Top-1 prob")], "Mean layer router statistics", "Value"),
        (axes[2], [("dead_expert_count", "Dead experts")], "Mean dead experts per layer", "Expert count")):
        for metric, label in metrics:
            values = [_number(row[metric]) for row in rows]
            ax.plot(steps, values, marker="o", label=label)
        ax.set(title=title, xlabel="Training checkpoint step", ylabel=unit, xticks=steps)
        ax.legend(fontsize=8)
    fig.suptitle("Available-layer means; missing data not imputed", fontsize=10)
    return _finish(fig, output)


PLOTS: dict[str, Callable] = {
    "layer_expert_usage_heatmap": plot_usage_heatmap,
    "load_cv_by_layer": plot_load_cv,
    "router_entropy_by_layer": plot_router_entropy,
    "dead_experts_by_layer": plot_dead_experts,
    "moe_ffn_rms_ratio": plot_moe_ffn_ratio,
    "expert_output_rms_heatmap": plot_expert_rms,
    "modality_expert_usage": plot_modality_usage,
    "routing_stability_by_layer": plot_routing_stability,
    "moe_gate_by_layer": plot_gate,
    "denoising_step_expert_usage": plot_denoising_step_usage,
    "moe_dashboard": plot_dashboard,
}


def plot_all(summary: dict[str, Any], output_dir: str | Path) -> dict[str, list[Path]]:
    import logging
    destination = Path(output_dir)
    results = {}
    for name, function in PLOTS.items():
        existing = set(plt.get_fignums())
        try:
            results[name] = function(summary, destination / name)
        except Exception:
            logging.getLogger(__name__).exception("MoE plot failed: %s at %s", name, destination)
        finally:
            for number in set(plt.get_fignums()) - existing:
                plt.close(number)
    return results
