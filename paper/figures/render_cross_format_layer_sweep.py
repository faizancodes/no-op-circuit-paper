#!/usr/bin/env python3
"""Heatmap figure: layer × direction × format AUC for CodeGemma + DeepSeek.

Source data: results/monitor_real/cross_format_layer_sweep_{model}.json
Output:      paper/figures/cross_format_layer_sweep.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    for ax, model in zip(axes, ("codegemma", "deepseek")):
        path = Path(f"results/monitor_real/cross_format_layer_sweep_{model}.json")
        blob = json.loads(path.read_text())
        rows = blob["rows"]
        layers = [r["layer"] for r in rows]

        # Three lines for v_toy: pytest, paraphrase_minimal, paraphrase_realistic
        v_toy_pytest = [r["v_toy__pytest_auc"] for r in rows]
        v_toy_paramin = [r["v_toy__paraphrase_minimal_auc"] for r in rows]
        v_toy_parareal = [r["v_toy__paraphrase_realistic_auc"] for r in rows]
        # v_para (in-sample on paraphrase-realistic) on pytest: tests
        # whether a paraphrase-derived direction transfers back to pytest.
        v_para_pytest = [r["v_para__pytest_auc"] for r in rows]

        ax.plot(layers, v_toy_pytest, "o-", color="#1f77b4",
                label="$v_{\\rm toy}$ on pytest format", linewidth=2)
        ax.plot(layers, v_toy_parareal, "s-", color="#d62728",
                label="$v_{\\rm toy}$ on realistic paraphrase", linewidth=2)
        ax.plot(layers, v_toy_paramin, "^-", color="#9467bd",
                label="$v_{\\rm toy}$ on minimal paraphrase",
                linewidth=1.2, alpha=0.7)
        ax.plot(layers, v_para_pytest, "d--", color="#2ca02c",
                label="$v_{\\rm para}$ on pytest format (in-sample on para)",
                linewidth=1.4, alpha=0.7)

        # Reference lines
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.6)
        ax.axhline(0.85, color="grey", linestyle="--", linewidth=1, alpha=0.4)
        ax.text(layers[-1] - 0.3, 0.51, "chance", fontsize=8, color="grey")

        # Mark the §4.3 canonical patching peak for this model
        peak = {"codegemma": 26, "deepseek": 22}[model]
        ax.axvline(peak, color="black", linestyle=":", linewidth=1, alpha=0.4)
        ax.text(peak + 0.2, 0.05, f"§4.3 peak\n(L{peak})", fontsize=8, alpha=0.7)

        ax.set_xlabel("Layer index")
        ax.set_ylabel("ROC-AUC")
        ax.set_title({
            "codegemma": "CodeGemma-7B (28 layers, pos $-1$)",
            "deepseek":  "DeepSeek-Coder-1.3B (24 layers, pos $-1$)",
        }[model])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    out = Path("paper/figures/cross_format_layer_sweep.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
