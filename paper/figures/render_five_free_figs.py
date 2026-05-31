#!/usr/bin/env python3
"""Render figures for the five free analyses (Apps. G.11–G.14 + extended G.4).

Outputs PNGs to paper/figures/. Run from the repo root.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("paper/figures")
RESULTS = Path("results/monitor_real")
DPI = 150


def fig_threshold_sweep_deployment():
    blob = json.loads((RESULTS / "threshold_sweep_deployment.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Pareto curve: spurious-edit reduction (x) vs useful-edit loss (y), per model
    colors = {"qwen": "#1f77b4", "codegemma": "#ff7f0e", "deepseek": "#2ca02c"}
    for label, data in blob["by_model"].items():
        rows = data["sweep"]
        red = np.asarray([r["spurious_edit_reduction"] for r in rows])
        loss = np.asarray([r["useful_edit_loss"] for r in rows])
        # Keep only rows with valid red, loss
        mask = np.isfinite(red) & np.isfinite(loss)
        red, loss = red[mask], loss[mask]
        axes[0].plot(red * 100, loss * 100, "-", color=colors[label],
                      alpha=0.6, label=label)
        # Highlight the knee point
        # Maximises (reduction - loss)
        knee_idx = int(np.argmax(red - loss))
        axes[0].scatter([red[knee_idx] * 100], [loss[knee_idx] * 100],
                        color=colors[label], s=80, zorder=3,
                        edgecolor="black", linewidth=0.8)
        axes[0].annotate(f"knee", (red[knee_idx] * 100, loss[knee_idx] * 100),
                         textcoords="offset points", xytext=(6, 4), fontsize=8,
                         color=colors[label])
    axes[0].set_xlabel("Spurious-edit reduction (%)")
    axes[0].set_ylabel("Useful-edit loss (%)")
    axes[0].set_title("(A) Deployment Pareto frontier (single-turn agent-loop)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-5, 105)

    # Sweep curve: threshold (x) vs final spurious rate (y)
    for label, data in blob["by_model"].items():
        rows = data["sweep"]
        # Normalise threshold to a percentile of its model-specific score range
        thr = np.asarray([r["threshold"] for r in rows])
        final_spur = np.asarray([r["final_spurious_rate"] for r in rows])
        useful_loss = np.asarray([r["useful_edit_loss"] for r in rows])
        # Plot final_spurious vs useful_loss (related to ROC)
        axes[1].plot(useful_loss * 100, final_spur * 100, "-",
                      color=colors[label], alpha=0.7, label=label)
    axes[1].set_xlabel("Useful-edit loss (%)")
    axes[1].set_ylabel("Final spurious-edit rate (%)")
    axes[1].set_title("(B) Final spurious commit rate vs useful-edit loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "threshold_sweep_deployment.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_layer_sweep_auc():
    blob = json.loads((RESULTS / "layer_sweep_auc.json").read_text())
    rows = blob["rows"]
    layers = [r["layer"] for r in rows]
    fresh = [r["auc_fresh_v_noop_L"] for r in rows]
    frozen = [r["auc_frozen_v_noop_L24"] for r in rows]
    norms = [r["v_noop_L_norm"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(layers, fresh, "o-", color="#1f77b4",
             label="Fresh v_noop_L (per-layer contrastive)", linewidth=2, markersize=5)
    ax1.plot(layers, frozen, "s-", color="#ff7f0e",
             label="Frozen L24-trained v_noop (transfer)", linewidth=2, markersize=5)
    ax1.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.6)
    ax1.text(0.5, 0.51, "chance", color="grey", fontsize=8)
    ax1.axvline(24, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax1.text(24.3, 0.55, "L24\n(causal patching peak)",
             color="red", fontsize=8, alpha=0.7)

    ax1.set_xlabel("Layer index (Qwen2.5-Coder-1.5B-Instruct, 28 layers)")
    ax1.set_ylabel("ROC-AUC on 499 SWE-bench-Verified-derived paired prompts")
    ax1.set_title("Layer-sweep AUC at pos −1: signal is broadly distributed")
    ax1.set_ylim(0.0, 1.05)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "layer_sweep_auc.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_sample_efficiency_cross_model():
    files = {
        "Qwen-1.5B":      RESULTS / "sample_efficiency.json",
        "CodeGemma-7B":   RESULTS / "sample_efficiency_codegemma.json",
        "DeepSeek-1.3B":  RESULTS / "sample_efficiency_deepseek.json",
    }
    colors = {"Qwen-1.5B": "#1f77b4", "CodeGemma-7B": "#ff7f0e", "DeepSeek-1.3B": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, p in files.items():
        blob = json.loads(p.read_text())["summary"]
        ns = sorted(int(k) for k in blob.keys())
        means = [blob[str(n)]["auc_mean"] for n in ns]
        stds = [blob[str(n)]["auc_std"] for n in ns]
        ax.errorbar(ns, means, yerr=stds, fmt="o-", color=colors[label],
                    label=label, capsize=4, linewidth=2, markersize=6)
        # Annotate the full-49 endpoint
        ax.annotate(f"{means[-1]:.3f}", (ns[-1], means[-1]),
                    textcoords="offset points", xytext=(6, 0),
                    fontsize=9, color=colors[label])
    ax.set_xscale("log")
    ax.set_xlabel("Number of paired toy tasks used to derive v_noop (N)")
    ax.set_ylabel("ROC-AUC on 499 SWE-bench-Verified-derived paired prompts")
    ax.set_title("Sample-efficiency curve: saturates by N≈5–10 on all three models")
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "sample_efficiency_cross_model.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_per_repo_g10():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = {"qwen": "#1f77b4", "codegemma": "#ff7f0e", "deepseek": "#2ca02c"}
    for ax, label in zip(axes, ("qwen", "codegemma", "deepseek")):
        p = RESULTS / f"agent_loop_per_repo_{label}.json"
        blob = json.loads(p.read_text())
        rows = [r for r in blob["rows"]
                if r["held_out"]["spurious_edit_reduction"] ==
                   r["held_out"]["spurious_edit_reduction"]]  # filter NaN
        # Show only repos with at least 5 tasks
        rows = [r for r in rows if r["n_tasks"] >= 5]
        rows.sort(key=lambda r: -r["held_out"]["spurious_edit_reduction"])
        names = [r["repo"].replace("_", "/") for r in rows]
        red = [r["held_out"]["spurious_edit_reduction"] * 100 for r in rows]
        loss = [r["held_out"]["useful_edit_loss"] * 100 if r["held_out"]["useful_edit_loss"] == r["held_out"]["useful_edit_loss"] else 0 for r in rows]
        y = np.arange(len(names))
        ax.barh(y - 0.2, red, height=0.4, color=colors[label],
                alpha=0.85, label="spurious-edit reduction")
        ax.barh(y + 0.2, loss, height=0.4, color="grey",
                alpha=0.7, label="useful-edit loss")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("% (held-out threshold)")
        ax.set_title(label, fontsize=11)
        ax.set_xlim(0, 105)
        ax.grid(True, alpha=0.3, axis="x")
        ax.legend(loc="lower right", fontsize=7)
        ax.invert_yaxis()

    fig.suptitle("Per-repo G.10 deployment metrics (repos with ≥5 paired tasks)",
                 fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "per_repo_g10.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    fig_threshold_sweep_deployment()
    fig_layer_sweep_auc()
    fig_sample_efficiency_cross_model()
    fig_per_repo_g10()


if __name__ == "__main__":
    main()
