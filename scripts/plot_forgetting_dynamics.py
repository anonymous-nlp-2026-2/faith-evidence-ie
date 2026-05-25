"""GRPO Training Dynamics: Catastrophic Forgetting in QLoRA vs bf16.

Generates a 3-panel figure showing:
  (a) Training reward trajectory
  (b) Rel-F1 at evaluation checkpoints (forgetting)
  (c) Evi-F1 at evaluation checkpoints (inverse rise)
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

C_BF16 = "#2171B5"
C_QLORA_CED = "#E6550D"
C_QLORA_LR5E6 = "#756BB1"

STYLE_BF16 = dict(color=C_BF16, linestyle="-", marker="o", markersize=4.5, zorder=5)
STYLE_QLORA_CED = dict(color=C_QLORA_CED, linestyle="--", marker="s", markersize=4.5, zorder=4)
STYLE_QLORA_LR5E6 = dict(color=C_QLORA_LR5E6, linestyle=":", marker="^", markersize=4.5, zorder=3)

TRAINING_LOG_PATH = "/workspace/freige/training_logs_grpo.json"

SFT = {"rel_f1": 0.4057, "evi_f1": 0.7979, "edcr": 0.7118}

# Eval data with step=0 as SFT baseline anchor
EVAL_BF16 = {
    "steps": [0, 80, 90, 100],
    "rel_f1": [SFT["rel_f1"], 0.4456, 0.4523, 0.4549],
    "evi_f1": [SFT["evi_f1"], 0.8034, 0.8052, 0.8086],
}

EVAL_QLORA_CED = {
    "steps": [0, 10, 20, 30, 50, 60, 90, 100],
    "rel_f1": [SFT["rel_f1"], 0.003, 0.0088, 0.0338, 0.1563, 0.1467, 0.2108, 0.2167],
    "evi_f1": [SFT["evi_f1"], 0.72, 0.423, 0.5889, 0.5643, 0.5763, 0.5666, 0.5746],
}

EVAL_QLORA_LR5E6 = {
    "steps": [0, 20, 40, 60, 80, 100],
    "rel_f1": [SFT["rel_f1"], 0.0077, 0.0071, 0.0093, 0.0116, 0.0138],
    "evi_f1": [SFT["evi_f1"], 0.628, 0.758, 0.810, 0.828, 0.850],
}


def load_training_logs():
    with open(TRAINING_LOG_PATH) as f:
        return json.load(f)


def plot_panel_reward(ax, logs):
    for key, label, style in [
        ("bf16_ced_g8", "bf16 (CED, G=8, lr=5e-5)", STYLE_BF16),
        ("qlora_ced_kl001", "QLoRA (CED, G=8, lr=5e-5)", STYLE_QLORA_CED),
        ("qlora_ced_lr5e6", "QLoRA (CED, G=8, lr=5e-6)", STYLE_QLORA_LR5E6),
    ]:
        data = logs[key]
        steps = [d["step"] for d in data]
        rewards = [d["reward"] for d in data]
        ax.plot(steps, rewards, label=label, **style)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Total Reward")
    ax.set_title("(a) Training Reward", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.92, edgecolor="0.85",
              handlelength=2.2, borderpad=0.4, labelspacing=0.3)
    ax.set_xlim(-2, 107)
    ax.set_ylim(-0.05, 3.2)
    ax.grid(True, alpha=0.15, linewidth=0.4)


def plot_panel_rel_f1(ax, logs):
    ax.axhline(SFT["rel_f1"], color="0.5", linestyle="-.",
               linewidth=0.9, label="SFT baseline", zorder=1, alpha=0.7)

    ax.plot(EVAL_BF16["steps"], EVAL_BF16["rel_f1"],
            label="bf16 (CED, G=8, lr=5e-5)", **STYLE_BF16)
    ax.plot(EVAL_QLORA_CED["steps"], EVAL_QLORA_CED["rel_f1"],
            label="QLoRA (CED, G=8, lr=5e-5)", **STYLE_QLORA_CED)
    ax.plot(EVAL_QLORA_LR5E6["steps"], EVAL_QLORA_LR5E6["rel_f1"],
            label="QLoRA (CED, G=8, lr=5e-6)", **STYLE_QLORA_LR5E6)

    # Shade the forgetting region
    ax.fill_between([5, 105], 0, SFT["rel_f1"] * 0.5,
                    color="red", alpha=0.04, zorder=0)

    ax.annotate("catastrophic\nforgetting",
                xy=(80, 0.012), xytext=(42, 0.10),
                fontsize=7.5, color=C_QLORA_LR5E6, fontstyle="italic",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_QLORA_LR5E6,
                                lw=0.9, connectionstyle="arc3,rad=0.15"))

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Rel-F1")
    ax.set_title("(b) Relation Extraction F1", fontweight="bold")
    ax.legend(loc="center left", framealpha=0.92, edgecolor="0.85",
              handlelength=2.2, borderpad=0.4, labelspacing=0.3,
              bbox_to_anchor=(0.0, 0.55))
    ax.set_xlim(-2, 107)
    ax.set_ylim(-0.02, 0.52)
    ax.grid(True, alpha=0.15, linewidth=0.4)


def plot_panel_evi_f1(ax, logs):
    ax.axhline(SFT["evi_f1"], color="0.5", linestyle="-.",
               linewidth=0.9, label="SFT baseline", zorder=1, alpha=0.7)

    ax.plot(EVAL_BF16["steps"], EVAL_BF16["evi_f1"],
            label="bf16 (CED, G=8, lr=5e-5)", **STYLE_BF16)
    ax.plot(EVAL_QLORA_CED["steps"], EVAL_QLORA_CED["evi_f1"],
            label="QLoRA (CED, G=8, lr=5e-5)", **STYLE_QLORA_CED)
    ax.plot(EVAL_QLORA_LR5E6["steps"], EVAL_QLORA_LR5E6["evi_f1"],
            label="QLoRA (CED, G=8, lr=5e-6)", **STYLE_QLORA_LR5E6)

    # Shade region above SFT
    ax.fill_between([5, 105], SFT["evi_f1"], 0.92,
                    color="green", alpha=0.04, zorder=0)

    ax.annotate("Evi-F1 rises\nabove SFT",
                xy=(100, 0.850), xytext=(50, 0.90),
                fontsize=7.5, color=C_QLORA_LR5E6, fontstyle="italic",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_QLORA_LR5E6,
                                lw=0.9, connectionstyle="arc3,rad=-0.15"))

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Evi-F1")
    ax.set_title("(c) Evidence F1", fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.92, edgecolor="0.85",
              handlelength=2.2, borderpad=0.4, labelspacing=0.3)
    ax.set_xlim(-2, 107)
    ax.set_ylim(0.35, 0.95)
    ax.grid(True, alpha=0.15, linewidth=0.4)


def main():
    logs = load_training_logs()

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.8))
    fig.subplots_adjust(wspace=0.30, left=0.04, right=0.98, top=0.88, bottom=0.15)

    plot_panel_reward(axes[0], logs)
    plot_panel_rel_f1(axes[1], logs)
    plot_panel_evi_f1(axes[2], logs)

    out_dir = "/workspace/freige/figures"
    os.makedirs(out_dir, exist_ok=True)

    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"grpo_forgetting_dynamics.{ext}")
        fig.savefig(path)
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
