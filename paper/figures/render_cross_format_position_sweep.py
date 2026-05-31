#!/usr/bin/env python3
"""2x2 heatmap: paraphrase-realistic AUC over (layer × pos)
for v_toy and v_real on CodeGemma + DeepSeek.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _grid(rows, n_layers, positions, direction, fmt):
    """Return [n_layers, n_pos] AUC grid for (direction, format)."""
    key = f"{direction}__{fmt}_auc"
    g = np.full((n_layers, len(positions)), np.nan)
    pos_index = {p: i for i, p in enumerate(positions)}
    for r in rows:
        if r["position"] in pos_index:
            g[r["layer"], pos_index[r["position"]]] = r[key]
    return g


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, model in enumerate(("codegemma", "deepseek")):
        path = Path(f"results/monitor_real/cross_format_position_sweep_{model}.json")
        blob = json.loads(path.read_text())
        cfg = blob["config"]
        rows = blob["rows"]
        positions = cfg["positions"]
        n_layers = cfg["n_layers"]
        canonical = {"codegemma": (26, -1), "deepseek": (22, -1)}[model]
        for col, direction in enumerate(("v_toy", "v_real")):
            ax = axes[row, col]
            grid = _grid(rows, n_layers, positions, direction,
                          "paraphrase_realistic")
            im = ax.imshow(grid, aspect="auto", origin="lower",
                            vmin=0.0, vmax=1.0, cmap="RdYlGn",
                            interpolation="nearest")
            # Mark the canonical (layer, pos)
            cL, cP = canonical
            if cP in positions:
                px = positions.index(cP)
                ax.scatter([px], [cL], marker="x", s=80, color="black",
                            linewidths=2, label=f"§4.3 peak (L{cL}, pos {cP})")

            # Annotate the best v_toy cell
            ix = np.unravel_index(np.nanargmax(grid), grid.shape)
            ax.scatter([ix[1]], [ix[0]], marker="o", s=140,
                        facecolors="none", edgecolors="black", linewidths=2,
                        label=f"best ({direction}): L{ix[0]}, pos {positions[ix[1]]} "
                              f"AUC={grid[ix]:.3f}")
            ax.set_xticks(range(len(positions)))
            ax.set_xticklabels([f"{p:+d}" for p in positions])
            ax.set_xlabel("token position (relative to action token)")
            ax.set_ylabel("layer")
            title = {
                "codegemma": "CodeGemma-7B",
                "deepseek":  "DeepSeek-Coder-1.3B",
            }[model]
            ax.set_title(f"{title} — {direction} → paraphrase-realistic AUC",
                          fontsize=10)
            ax.legend(loc="lower right", fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = Path("paper/figures/cross_format_position_sweep.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
