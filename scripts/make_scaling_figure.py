#!/usr/bin/env python3
"""The 'money figure' — the four-act scaling spine, from results/paper_stats.json.
A: mechanism location (relative-depth law). B: mechanism strength (effect vs size).
C: prior-gated behavior (noop full-menu vs binary). D: deployable edit-veto.
Writes paper/figures/scaling_synthesis.{png,pdf}."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S = json.load(open("results/paper_stats.json"))
SIZES = ["1.5B", "3B", "7B", "14B", "32B"]
X = [1.5, 3, 7, 14, 32]  # params (B), log x-axis

# reference values for 1.5B from the paper; 7B uses the FINE-sweep peak (L27/0.964)
REL = {"1.5B": 0.857, "3B": S["F3"]["3B"]["rel_depth"], "7B": 0.964,
       "14B": S["F3"]["14B"]["rel_depth"], "32B": S["F3"]["32B"]["rel_depth"]}
# 1.5B = paper F→B +0.65; 7B = FINE-sweep peak (L27/+2.77), matching the §4.4 table
# (the coarse L26 grid undershot at +2.24 and is the only 7B value bootstrapped, so
# the fine 7B point carries no CI bar — see EFF_CI below).
EFF = {"1.5B": 0.65, "3B": S["F3"]["3B"]["f2b_mean"], "7B": 2.77,
       "14B": S["F3"]["14B"]["f2b_mean"], "32B": S["F3"]["32B"]["f2b_mean"]}
EFF_CI = {sz: S["F3"][sz]["ci"] for sz in ["3B", "14B", "32B"]}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 2, figsize=(9.2, 7.0))

# --- A: relative-depth law (mechanism location) ---
a = ax[0, 0]
ya = [REL[s] for s in SIZES]
a.plot(X, ya, "o-", color="#1f77b4", lw=2, ms=7)
a.axhspan(0.86, 0.96, color="#1f77b4", alpha=0.08, label="law band 0.86–0.96")
a.scatter([7], [REL["7B"]], s=160, facecolors="none", edgecolors="crimson", lw=2, zorder=5)
a.annotate("7B (=28 layers as 1.5B)\nwidth pushes site deeper", (7, REL["7B"]),
           xytext=(8, 0.80), fontsize=8, color="crimson",
           arrowprops=dict(arrowstyle="->", color="crimson"))
for s, x, y in zip(SIZES, X, ya):
    a.annotate(s, (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
a.set_xscale("log"); a.set_xticks(X); a.set_xticklabels(SIZES)
a.set_ylim(0.78, 0.97); a.set_xlabel("model size (params)")
a.set_ylabel("causal peak relative depth")
a.set_title("A · Mechanism location: relative-depth law\n(universal late-layer site, drifts deeper)", fontsize=9)
a.legend(fontsize=7, loc="lower right")

# --- B: mechanism strength (effect magnitude vs size) ---
b = ax[0, 1]
yb = [EFF[s] for s in SIZES]
ci_sizes = ["3B", "14B", "32B"]  # bootstrapped CIs only (1.5B paper value, 7B fine peak: no CI)
ci_x = [3, 14, 32]
ci_y = [EFF[s] for s in ci_sizes]
err = [[EFF[s] - EFF_CI[s][0] for s in ci_sizes],
       [EFF_CI[s][1] - EFF[s] for s in ci_sizes]]
b.plot(X, yb, "o-", color="#2ca02c", lw=2, ms=7)
b.errorbar(ci_x, ci_y, yerr=err, fmt="none", ecolor="#2ca02c", capsize=3)
for s, x, y in zip(SIZES, X, yb):
    b.annotate(s, (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
b.set_xscale("log"); b.set_yscale("log"); b.set_xticks(X); b.set_xticklabels(SIZES)
b.set_xlabel("model size (params)"); b.set_ylabel("peak F→B effect (logits, 95% CI)")
b.set_title("B · Mechanism strength: ~18× growth\n(causal edit−noop shift vs size)", fontsize=9)

# --- C: prior-gated, non-monotonic behavior ---
c = ax[1, 0]
full = {"1.5B": 0.0, "3B": S["F1"]["3B"]["rate"], "7B": 0.0, "14B": 0.0, "32B": S["F1"]["32B"]["rate"]}
binr = {s: S["binary"][s]["rate"] for s in SIZES}
c.plot(X, [100 * full[s] for s in SIZES], "o-", color="#d62728", lw=2, ms=7, label="full 5-action menu")
c.plot(X, [100 * binr[s] for s in SIZES], "s--", color="#ff7f0e", lw=2, ms=6, label="binary {edit, noop}")
c.set_xscale("log"); c.set_xticks(X); c.set_xticklabels(SIZES)
c.set_xlabel("model size (params)"); c.set_ylabel("do-nothing rate on passing (%)")
c.set_title("C · Behavior: prior-gated & non-monotonic\n(binary unmasks 7B; 14B/32B stay edit-locked)", fontsize=9)
c.legend(fontsize=8, loc="upper right")

# --- D: deployment — over-editing vs size, cut by the held-out veto ---
d = ax[1, 1]
LS = S["ladder_summary"]
oe_ev = [100 * LS[s]["ev"]["over_edit"] for s in SIZES]
veto = [100 * LS[s]["ev"]["over_edit_after"] for s in SIZES]
d.plot(X, oe_ev, "o-", color="#d62728", lw=2, ms=7, label="over-editing (evidence in context)")
d.plot(X, veto, "s--", color="#2ca02c", lw=2, ms=6, label="+ monitor veto (held-out)")
d.fill_between(X, veto, oe_ev, color="#2ca02c", alpha=0.10)
for s, x, y in zip(SIZES, X, oe_ev):
    d.annotate(s, (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
d.set_xscale("log"); d.set_xticks(X); d.set_xticklabels(SIZES)
d.set_xlabel("model size (params)"); d.set_ylabel("over-editing on passing tasks (%)")
d.set_ylim(0, 110)
d.set_title("D · Deployment: severe over-editing across scale,\ncut by the held-out edit-veto", fontsize=9)
d.legend(fontsize=7, loc="center right")

fig.suptitle("Pass/fail evidence in Qwen2.5-Coder: universal causal mechanism, "
             "prior-gated behavior, deployable veto", fontsize=11, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig("paper/figures/scaling_synthesis.png", dpi=170, bbox_inches="tight")
fig.savefig("paper/figures/scaling_synthesis.pdf", bbox_inches="tight")
print("wrote paper/figures/scaling_synthesis.png + .pdf")
