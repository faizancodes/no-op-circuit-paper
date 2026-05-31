#!/usr/bin/env python3
"""Render the 3-panel hero figure for the represented-evidence-vs-action dissociation.

Light theme: white background, strict grayscale punctuated by a single teal accent
used sparingly on the focal element of each step, light heading, bold-uppercase small
labels, monospace for data, generous spacing.

Panels: (A) available  ->  (B) causally read out  ->  (C) not acted on.

Data (all real, no fabricated points):
  - Panel A: paper §5.1 headline monitor AUCs 0.989/0.950/0.888 (Qwen also in auc_ci.json).
  - Panels B & C: results/swe_peak_patching/qwen_swe_peak_patch_summary.json
                  (Qwen, L24/pos -1, 200 SWE-bench-Verified-derived paired prompts).

Output: paper/figures/main_dissociation.png (300 dpi, white background).
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
S = json.loads((REPO / "results/swe_peak_patching/qwen_swe_peak_patch_summary.json").read_text())

# ---- real numbers -----------------------------------------------------------
AUC = [("Qwen-1.5B", 0.989), ("CodeGemma-7B", 0.950), ("DeepSeek-1.3B", 0.888)]  # §5.1 headline
clean = S["clean"]
gap = clean["mean_buggy_minus_fixed_gap"]                       # 0.332
n_pairs = S["n_pairs"]                                          # 200
cellsB = ["L24_pos-1", "L12_pos-1", "L24_pos-8"]
sB = [S["by_cell"][c]["f2b"] for c in cellsB]
shifts = [s["mean_shift"] for s in sB]                          # +0.314, +0.029, -0.001
err = [[s["mean_shift"] - s["ci95_lo"] for s in sB], [s["ci95_hi"] - s["mean_shift"] for s in sB]]
recover_pct = 100.0 * shifts[0] / gap                          # ~95%
ACTIONS = ["view", "grep", "test", "edit", "noop"]
buggy = [100.0 * clean["argmax_buggy_counts"].get(a, 0) / n_pairs for a in ACTIONS]
fixed = [100.0 * clean["argmax_fixed_counts"].get(a, 0) / n_pairs for a in ACTIONS]

# ---- design tokens (light theme) --------------------------------------------
BASE = "#ffffff"
ACCENT = "#06c7a6"        # teal fill (the single accent)
ACCENT_TXT = "#048a72"    # darker teal for legible accent text on white
INK, SEC, TER, MUTE = "#14171a", "#565c63", "#80868b", "#9aa0a6"
GRID, EDGE = "#ededed", "#d9dce0"
GRAY_MID, GRAY_LIGHT, GRAY_CTRL = "#8e949a", "#c6cacd", "#b4b9be"
SANS = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
MONO = ["Geist Mono", "Menlo", "DejaVu Sans Mono", "monospace"]

plt.rcParams.update({
    "figure.facecolor": BASE, "savefig.facecolor": BASE, "axes.facecolor": BASE,
    "font.family": SANS, "font.size": 10,
    "text.color": INK, "axes.labelcolor": SEC,
    "xtick.color": TER, "ytick.color": TER, "xtick.labelsize": 9, "ytick.labelsize": 8.5,
    "axes.edgecolor": EDGE, "axes.linewidth": 0.8, "axes.axisbelow": True,
})

def style(ax):
    ax.set_facecolor(BASE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.xaxis.grid(False)

def title_block(ax, tag, desc):
    ax.text(0.5, 1.205, tag, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=INK, family=SANS)
    ax.text(0.5, 1.075, desc, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.8, color=MUTE, family=SANS)

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 5.4))
fig.subplots_adjust(left=0.065, right=0.985, top=0.70, bottom=0.185, wspace=0.42)
fig.suptitle("A represented-evidence-vs-action dissociation",
             fontsize=22, color=INK, fontweight="light", y=0.975)

# ----- Panel A: available -----
style(axA)
names, vals = [n for n, _ in AUC], [v for _, v in AUC]
axA.bar(names, vals, width=0.62, color=[ACCENT, GRAY_MID, GRAY_LIGHT], edgecolor=BASE, linewidth=0.6)
axA.set_ylim(0.5, 1.07)
axA.set_ylabel("ROC-AUC, failing vs passing  (0.5 = chance)", labelpad=10, fontsize=9.5)
for i, v in enumerate(vals):
    axA.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=10,
             fontweight="bold", color=(INK if i else ACCENT_TXT), family=MONO)
axA.set_xticks(range(3)); axA.set_xticklabels(names, family=MONO, fontsize=8.5, color=SEC)
title_block(axA, "(A)  AVAILABLE", "the pass/fail signal is linearly readable")

# ----- Panel B: causally read out -----
style(axB)
axB.bar(range(3), shifts, yerr=err, width=0.6, color=[ACCENT, GRAY_CTRL, GRAY_CTRL],
        edgecolor=BASE, linewidth=0.6, error_kw=dict(lw=1.2, capsize=4, ecolor=TER))
axB.axhline(0, color="#cbced2", lw=0.9)
axB.set_ylim(-0.06, 0.47)
axB.set_ylabel(r"$\Delta$(edit $-$ noop) under F$\to$B patching", labelpad=10, fontsize=9.5)
axB.text(0, shifts[0] + 0.028, f"+{shifts[0]:.2f}", ha="center", va="bottom",
         fontsize=11, fontweight="bold", color=ACCENT_TXT, family=MONO)
axB.text(1, shifts[1] + 0.022, f"+{shifts[1]:.2f}", ha="center", va="bottom", fontsize=8.5, color=TER, family=MONO)
axB.text(2, 0.018, f"{shifts[2]:+.2f}", ha="center", va="bottom", fontsize=8.5, color=TER, family=MONO)
axB.annotate(f"~{recover_pct:.0f}% of the buggy$-$fixed gap\n(p < 1e-28, n={n_pairs})",
             xy=(0.18, shifts[0]), xytext=(0.62, 0.40), fontsize=8.3, ha="left", va="center",
             color=SEC, arrowprops=dict(arrowstyle="-", lw=0.8, color="#c2c6ca"))
axB.set_xticks(range(3))
axB.set_xticklabels(["L24 / pos −1\nthe site", "L12 / pos −1\nwrong layer", "L24 / pos −8\nwrong position"],
                    family=MONO, fontsize=8)
for lbl, foc in zip(axB.get_xticklabels(), [True, False, False]):
    lbl.set_color(INK if foc else TER)
title_block(axB, "(B)  CAUSALLY READ OUT", "patching the site moves the edit$-$noop margin")

# ----- Panel C: not acted on -----
style(axC)
x = np.arange(len(ACTIONS)); w = 0.38
axC.bar(x - w/2, buggy, w, color=GRAY_MID, edgecolor=BASE, linewidth=0.6, label="buggy (tests fail)")
axC.bar(x + w/2, fixed, w, color=GRAY_LIGHT, edgecolor=BASE, linewidth=0.6, label="fixed (tests pass)")
axC.set_ylim(0, 108); axC.set_ylabel("first-token argmax rate (%)", labelpad=10, fontsize=9.5)
axC.set_xticks(x); axC.set_xticklabels(ACTIONS, family=MONO, fontsize=9, color=SEC)
axC.legend(fontsize=8.3, frameon=False, loc="upper center", labelcolor=SEC,
           handlelength=1.1, bbox_to_anchor=(0.5, 1.0))
ni = ACTIONS.index("noop")
axC.annotate("NOOP NEVER CHOSEN\n0% in both conditions", xy=(ni, 2.5), xytext=(ni - 0.15, 46),
             fontsize=8.8, ha="center", va="center", color=ACCENT_TXT, fontweight="bold", family=MONO,
             arrowprops=dict(arrowstyle="-|>", lw=1.6, color=ACCENT_TXT))
title_block(axC, "(C)  NOT ACTED ON", "yet the chosen action never changes")

out = REPO / "paper/figures/main_dissociation.png"
fig.savefig(out, dpi=300, facecolor=BASE, bbox_inches="tight", pad_inches=0.35)
print(f"wrote {out}")
print(f"  A AUC={vals}  B shifts={[round(s,3) for s in shifts]} (~{recover_pct:.0f}% gap)  "
      f"C noop={buggy[ni]}%/{fixed[ni]}%")
