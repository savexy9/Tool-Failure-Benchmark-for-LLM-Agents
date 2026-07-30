"""
Publication-quality charts for the Tool-Failure Reliability experiment.

Generates four figures:
  1. Detection rate by fault type (grouped bar: verify vs no-verify)
  2. Accuracy by prompt condition (grouped bar: clean vs faulted)
  3. Fault × Prompt condition heatmap (detection rate)
  4. Summary metrics table
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from evaluation.metrics import compute_all_metrics, Metrics

# ── Style constants ──────────────────────────────────────────────────────────

# Publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette (colorblind-friendly)
COLORS = {
    "no_verify": "#4C72B0",
    "verify": "#DD8452",
    "clean": "#55A868",
    "faulted": "#C44E52",
    "accent": "#8172B3",
}

FAULT_ORDER = ["none", "silent_wrong", "error", "malformed", "empty"]
FAULT_DISPLAY = {
    "none": "Clean",
    "silent_wrong": "Silent\nWrong",
    "error": "Exception",
    "malformed": "Malformed",
    "empty": "Empty",
}


# ── Chart 1: Detection rate by fault type ────────────────────────────────────


def plot_detection_by_fault(results: list[dict], save_path: Path) -> None:
    """Grouped bar chart: detection rate by fault type, split by prompt condition."""
    fault_types = [f for f in FAULT_ORDER if f != "none"]

    no_verify_rates = []
    verify_rates = []

    for ft in fault_types:
        subset_nv = [r for r in results if r["fault_type"] == ft and r["prompt_condition"] == "no_verify"]
        subset_v = [r for r in results if r["fault_type"] == ft and r["prompt_condition"] == "verify"]

        nv_rate = sum(1 for r in subset_nv if r["agent_flagged"]) / len(subset_nv) if subset_nv else 0
        v_rate = sum(1 for r in subset_v if r["agent_flagged"]) / len(subset_v) if subset_v else 0

        no_verify_rates.append(nv_rate * 100)
        verify_rates.append(v_rate * 100)

    x = np.arange(len(fault_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, no_verify_rates, width, label="No Verify Prompt", color=COLORS["no_verify"])
    bars2 = ax.bar(x + width / 2, verify_rates, width, label="With Verify Prompt", color=COLORS["verify"])

    ax.set_xlabel("Fault Type")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Agent Fault Detection Rate by Fault Type")
    ax.set_xticks(x)
    ax.set_xticklabels([FAULT_DISPLAY[f] for f in fault_types])
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Chart 2: Accuracy by prompt condition ────────────────────────────────────


def plot_accuracy_by_condition(results: list[dict], save_path: Path) -> None:
    """Grouped bar chart: accuracy split by clean/faulted and prompt condition."""
    conditions = ["no_verify", "verify"]
    categories = ["clean", "faulted"]

    data = {}
    for cond in conditions:
        for cat in categories:
            if cat == "clean":
                subset = [r for r in results if r["fault_type"] == "none" and r["prompt_condition"] == cond]
            else:
                subset = [r for r in results if r["fault_type"] != "none" and r["prompt_condition"] == cond]
            correct = sum(1 for r in subset if r["is_correct"])
            data[(cond, cat)] = (correct / len(subset) * 100) if subset else 0

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, [data[("no_verify", c)] for c in categories], width,
                   label="No Verify Prompt", color=COLORS["no_verify"])
    bars2 = ax.bar(x + width / 2, [data[("verify", c)] for c in categories], width,
                   label="With Verify Prompt", color=COLORS["verify"])

    ax.set_xlabel("Trial Category")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Agent Accuracy by Prompt Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(["Clean Trials", "Faulted Trials"])
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Chart 3: Fault × Prompt heatmap ─────────────────────────────────────────


def plot_fault_prompt_heatmap(results: list[dict], save_path: Path) -> None:
    """Heatmap: detection rate for each fault type × prompt condition."""
    fault_types = [f for f in FAULT_ORDER if f != "none"]
    conditions = ["no_verify", "verify"]
    cond_labels = ["No Verify", "With Verify"]

    matrix = np.zeros((len(fault_types), len(conditions)))

    for i, ft in enumerate(fault_types):
        for j, cond in enumerate(conditions):
            subset = [r for r in results if r["fault_type"] == ft and r["prompt_condition"] == cond]
            if subset:
                matrix[i, j] = sum(1 for r in subset if r["agent_flagged"]) / len(subset) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(cond_labels)
    ax.set_yticks(np.arange(len(fault_types)))
    ax.set_yticklabels([FAULT_DISPLAY[f] for f in fault_types])

    # Add text annotations
    for i in range(len(fault_types)):
        for j in range(len(conditions)):
            val = matrix[i, j]
            color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    ax.set_title("Detection Rate: Fault Type × Prompt Condition")
    fig.colorbar(im, ax=ax, label="Detection Rate (%)", shrink=0.8)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Chart 4: Summary metrics table ──────────────────────────────────────────


def plot_metrics_table(metrics: Metrics, save_path: Path) -> None:
    """Publication-quality summary metrics table as a figure."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis("off")

    table_data = [
        ["Metric", "Value"],
        ["Total Trials", f"{metrics.total_trials}"],
        ["Clean Trials", f"{metrics.clean_trials}"],
        ["Faulted Trials", f"{metrics.faulted_trials}"],
        ["Accuracy (Overall)", f"{metrics.accuracy_overall:.1%}"],
        ["Accuracy (Clean)", f"{metrics.accuracy_clean:.1%}"],
        ["Accuracy (Faulted)", f"{metrics.accuracy_faulted:.1%}"],
        ["Detection Rate", f"{metrics.detection_rate:.1%}"],
        ["Silent Failure Rate", f"{metrics.silent_failure_rate:.1%}"],
        ["Recovery Rate", f"{metrics.recovery_rate:.1%}"],
    ]

    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    # Style header row
    for j in range(2):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(2):
            if i % 2 == 0:
                table[i, j].set_facecolor("#E8E8E8")

    ax.set_title("Experiment Summary Metrics", fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Generate all charts ─────────────────────────────────────────────────────


def generate_all_charts(results: list[dict], figures_dir: Path) -> Metrics:
    """Generate all four publication-quality charts.

    Args:
        results: List of result dicts from the experiment.
        figures_dir: Directory to save figures.

    Returns:
        The computed Metrics object.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_all_metrics(results)

    print("\nGenerating charts...")

    plot_detection_by_fault(results, figures_dir / "detection_by_fault.png")
    plot_accuracy_by_condition(results, figures_dir / "accuracy_by_condition.png")
    plot_fault_prompt_heatmap(results, figures_dir / "fault_prompt_heatmap.png")
    plot_metrics_table(metrics, figures_dir / "summary_metrics.png")

    print(f"\nAll charts saved to {figures_dir}/")
    return metrics
