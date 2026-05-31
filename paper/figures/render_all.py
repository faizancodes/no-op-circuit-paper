#!/usr/bin/env python3
"""Render all paper figures from disk artifacts.

Outputs (all 300dpi PNGs) under paper/figures/:
  paired_task_diagram.png
  behavioral_delta.png
  patching_heatmap.png
  steering_dose_response.png
  failure_table.png
  negative_control.png
  monitor_roc.png
  monitor_real_roc.png
  sae_decomposition.png   (Figure 9: SAE feature decomposition of v_noop)

Captions for each figure are written to <name>.txt alongside.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
FIG_DIR = Path(__file__).resolve().parent
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def save_caption(path: Path, text: str) -> None:
    path.with_suffix(".txt").write_text(text.strip() + "\n")


# ------------------------------------------------------------------------------
# Figure 1: Paired task / methodology diagram
# ------------------------------------------------------------------------------

def fig_paired_task_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    # Issue box (shared)
    ax.add_patch(mpatches.FancyBboxPatch((0.3, 2.4), 1.9, 1.2, boxstyle="round,pad=0.04",
                                          ec="#444", fc="#fff7e6", lw=1.4))
    ax.text(1.25, 3.0, "ISSUE TEXT\n(identical)", ha="center", va="center", fontsize=10,
            fontweight="bold")

    # Buggy version
    ax.add_patch(mpatches.FancyBboxPatch((2.8, 4.0), 1.8, 0.9, boxstyle="round,pad=0.04",
                                          ec="#aa3030", fc="#ffecec", lw=1.4))
    ax.text(3.7, 4.45, "BUGGY\nparser.py\n+ FAILING tests", ha="center", va="center",
            fontsize=9, color="#aa3030", fontweight="bold")

    # Fixed version
    ax.add_patch(mpatches.FancyBboxPatch((2.8, 1.1), 1.8, 0.9, boxstyle="round,pad=0.04",
                                          ec="#308030", fc="#ecffec", lw=1.4))
    ax.text(3.7, 1.55, "FIXED\nparser.py\n+ PASSING tests", ha="center", va="center",
            fontsize=9, color="#308030", fontweight="bold")

    # Arrows from issue to both versions
    ax.annotate("", xy=(2.7, 4.4), xytext=(2.3, 3.2),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#666"))
    ax.annotate("", xy=(2.7, 1.6), xytext=(2.3, 2.8),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#666"))

    # Agent box (shared)
    ax.add_patch(mpatches.FancyBboxPatch((5.2, 2.4), 2.0, 1.2, boxstyle="round,pad=0.04",
                                          ec="#222", fc="#f0f0ff", lw=1.6))
    ax.text(6.2, 3.0, "Qwen2.5-Coder-1.5B\n[L24, pos −1]\nresid_pre",
            ha="center", va="center", fontsize=9, fontweight="bold")

    # Arrows from versions into agent
    ax.annotate("", xy=(5.1, 3.4), xytext=(4.7, 4.3),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#aa3030"))
    ax.annotate("", xy=(5.1, 2.6), xytext=(4.7, 1.7),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#308030"))

    # Action vocab on the right
    actions = ["view", "grep", "test", "edit", "noop"]
    colors = ["#666", "#666", "#666", "#aa3030", "#308030"]
    ax.text(8.4, 5.0, "Action vocabulary\n(next-token logits)", ha="center", fontsize=10,
            fontweight="bold")
    for i, (a, c) in enumerate(zip(actions, colors)):
        ax.add_patch(mpatches.FancyBboxPatch((7.9, 4.0 - i*0.6), 1.0, 0.45,
                                              boxstyle="round,pad=0.02",
                                              ec=c, fc="#ffffff", lw=1.1))
        ax.text(8.4, 4.22 - i*0.6, a, ha="center", va="center", fontsize=10,
                color=c, fontweight="bold")
    # Margin label
    ax.text(9.4, 1.5, "Margin =\nlogit(edit)\n− logit(noop)",
            ha="center", va="center", fontsize=9, style="italic", color="#444")

    # Arrow from agent → actions
    ax.annotate("", xy=(7.9, 3.0), xytext=(7.25, 3.0),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#222"))

    fig.suptitle("Paired buggy/fixed substrate · single-token action vocab · "
                 "L24 pos −1 hook site", fontsize=11)
    out = FIG_DIR / "paired_task_diagram.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    save_caption(out,
        "Figure 1. Paired-task substrate and intervention site. For each task, the "
        "agent is prompted with an identical issue text and one of two task "
        "snapshots: a BUGGY version where pytest reports a failure, or a FIXED "
        "version where it passes. The action token following 'Action: ' is read "
        "out at position −1 of the rendered chat template. The `edit − noop` "
        "logit margin is our scalar behavioral signal; we apply patching and "
        "steering interventions to the residual stream at layer 24, position −1 "
        "of Qwen2.5-Coder-1.5B-Instruct (28 layers total)."
    )
    print(f"  wrote {out.name}")


# ------------------------------------------------------------------------------
# Figure 2: Behavioral Δ-margin strip plot
# ------------------------------------------------------------------------------

def _load_deltas(cache_dir: Path, variant: str) -> list[float]:
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    pairs: dict[str, dict[str, float]] = {}
    for e in manifest["entries"]:
        if e["variant"] != variant:
            continue
        m = e["action_logits"]["edit"] - e["action_logits"]["noop"]
        pairs.setdefault(e["task_id"], {})[e["condition"]] = m
    return [s["buggy"] - s["fixed"] for s in pairs.values() if "buggy" in s and "fixed" in s]


def fig_behavioral_delta() -> None:
    qwen_dir = RESULTS / "cache-20260515T221105Z"
    cg_dir = RESULTS / "cache-codegemma_7b_it-20260516T031036Z"
    variants = ["issue_only", "code", "code_tests"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    rng = np.random.default_rng(0)
    labels = []
    positions = []
    width = 0.35
    for i, v in enumerate(variants):
        qwen_vals = _load_deltas(qwen_dir, v)
        cg_vals = _load_deltas(cg_dir, v)
        # strip + box plot per variant per model
        x_q = i*1.0 + 0.0
        x_c = i*1.0 + width + 0.05
        positions.extend([x_q, x_c])
        labels.extend([f"Qwen\n{v}", f"CodeGemma\n{v}"])
        # jittered scatter
        ax.scatter(x_q + rng.uniform(-0.08, 0.08, len(qwen_vals)), qwen_vals,
                   s=14, alpha=0.55, color="#1f77b4", label="Qwen-1.5B" if i==0 else None)
        ax.scatter(x_c + rng.uniform(-0.08, 0.08, len(cg_vals)), cg_vals,
                   s=14, alpha=0.55, color="#ff7f0e", label="CodeGemma-7B" if i==0 else None)
        # mean line
        ax.hlines(np.mean(qwen_vals), x_q-0.12, x_q+0.12, color="#1f77b4", linewidth=2)
        ax.hlines(np.mean(cg_vals), x_c-0.12, x_c+0.12, color="#ff7f0e", linewidth=2)

    ax.axhline(0, color="#999", lw=0.8, linestyle=":")
    ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ-margin = buggy(edit−noop) − fixed(edit−noop)")
    ax.set_title("Test evidence is required for the action shift")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    out = FIG_DIR / "behavioral_delta.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    save_caption(out,
        "Figure 2. Behavioral Δ-margin per task across three evidence levels and "
        "two models. Each point is one of 49 paired tasks; horizontal bars are "
        "per-condition means. Without test output (issue_only, code) the action "
        "distribution is essentially identical between buggy and fixed conditions. "
        "Adding the test transcript (code_tests) drives a positive shift toward "
        "noop in nearly every task on Qwen (mean Δ = +0.659) and a bimodal "
        "shift on CodeGemma (mean Δ = +1.347). Test evidence is the sole driver "
        "of the behavioral effect we localize mechanistically."
    )
    print(f"  wrote {out.name}")


# ------------------------------------------------------------------------------
# Figure 3: Patching heatmap (Qwen + CodeGemma side by side)
# ------------------------------------------------------------------------------

def fig_patching_heatmap() -> None:
    qwen = np.load(RESULTS / "patch-20260516T005329Z" / "aggregated.npz", allow_pickle=True)
    cg = np.load(RESULTS / "patch-codegemma_7b_it-20260516T031403Z" / "aggregated.npz", allow_pickle=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=300)

    for ax, (data, title, n) in zip(axes, [
        (qwen, "Qwen2.5-Coder-1.5B  (28 layers)", "N=43 pairs"),
        (cg,   "CodeGemma-7B-it  (28 layers)",   "N=49 pairs"),
    ]):
        mean_shift = data["mean_shift_f2b"]
        layers = data["layer_indices"]
        positions = data["position_offsets"]
        im = ax.imshow(mean_shift, aspect="auto", cmap="RdBu_r",
                       vmin=-1.5, vmax=1.5, origin="lower")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{int(l)}" for l in layers], fontsize=8)
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels([f"pos {int(p)}" for p in positions], fontsize=9)
        ax.set_title(f"{title}\nF→B mean shift  ({n})", fontsize=10)
        # Mark peak cell
        flat_idx = int(np.argmax(mean_shift))
        py, px = divmod(flat_idx, mean_shift.shape[1])
        ax.add_patch(mpatches.Rectangle((px-0.5, py-0.5), 1, 1,
                                          fill=False, ec="red", lw=2.2))
        ax.text(px, py, f"{mean_shift[py, px]:+.2f}", ha="center", va="center",
                fontsize=9, color="red", fontweight="bold")

    fig.colorbar(im, ax=axes, shrink=0.7, label="clean_buggy − patched margin",
                 location="right", pad=0.03)
    fig.suptitle("Causal site of the no-op direction (single-cell patches)", fontsize=12)

    out = FIG_DIR / "patching_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    save_caption(out,
        "Figure 3. Layer × position heatmap of mean shift in `edit − noop` "
        "margin when the FIXED residual is substituted into the BUGGY forward "
        "at the indicated (layer, position) cell. Positive values (red) mean "
        "patching pulled the action toward NOOP — the predicted direction. "
        "Both models show a clean late-layer concentration at position −1 (the "
        "action token). Qwen's peak at L24/pos −1 recovers ~98% of the "
        "behavioral gap with a single substitution. CodeGemma's peak (L26/pos "
        "−1, relative depth 0.93) is at the same relative depth class but "
        "carries 2× the mean magnitude in a bimodal subset of tasks."
    )
    print(f"  wrote {out.name}")


# ------------------------------------------------------------------------------
# Figure 4: Steering dose-response
# ------------------------------------------------------------------------------

def fig_steering_dose_response() -> None:
    qwen = np.load(RESULTS / "steer-20260516T021522Z" / "curves.npz", allow_pickle=True)
    alphas_q = qwen["alphas"]
    conds_q = [c if isinstance(c, str) else c.decode() for c in qwen["conditions"]]
    m_q = qwen["margin_mean"]

    cg_candidates = sorted(RESULTS.glob("steer-codegemma_7b_it-*"),
                            key=lambda p: p.name, reverse=True)
    have_cg = bool(cg_candidates) and (cg_candidates[0] / "curves.npz").is_file()

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    style_q = {"buggy": ("#aa3030", "o", "Qwen buggy (N=49)"),
               "fixed": ("#308030", "s", "Qwen fixed (N=49)")}
    for i, c in enumerate(conds_q):
        color, marker, label = style_q[c]
        ax.plot(alphas_q, m_q[i], color=color, marker=marker, lw=2,
                label=label, markersize=7)

    if have_cg:
        cg = np.load(cg_candidates[0] / "curves.npz", allow_pickle=True)
        alphas_c = cg["alphas"]
        conds_c = [c if isinstance(c, str) else c.decode() for c in cg["conditions"]]
        m_c = cg["margin_mean"]
        style_c = {"buggy": ("#cc6660", "o", "CodeGemma buggy (responsive N=20)"),
                   "fixed": ("#66aa66", "s", "CodeGemma fixed (responsive N=20)")}
        ax2 = ax.twinx()
        ax2.set_ylabel("CodeGemma mean (edit − noop) margin",
                       color="#666")
        for i, c in enumerate(conds_c):
            color, marker, label = style_c[c]
            ax2.plot(alphas_c, m_c[i], color=color, marker=marker, lw=1.6,
                     linestyle="--", label=label, markersize=6, alpha=0.85)
        ax2.tick_params(axis="y", labelcolor="#666")
        # Combine legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, frameon=False)
    else:
        ax.legend(loc="upper right", fontsize=9, frameon=False)

    ax.axvline(0, color="#999", lw=0.8, linestyle=":")
    ax.set_xlabel("steering coefficient  α  (in units of ‖v_noop‖)")
    ax.set_ylabel("Qwen mean (edit − noop) margin across 49 tasks")
    ax.set_title("Single-direction steering reproduces the behavioral gap")
    ax.text(0.02, 0.05,
            f"‖v_noop‖ Qwen = {float(qwen['direction_norm']):.2f}"
            + (f"   |   CodeGemma = {float(np.load(cg_candidates[0] / 'curves.npz', allow_pickle=True)['direction_norm']):.2f}" if have_cg else ""),
            transform=ax.transAxes, fontsize=9, color="#444")
    ax.spines["top"].set_visible(False)

    out = FIG_DIR / "steering_dose_response.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    save_caption(out,
        "Figure 4. Dose-response curves for additive steering at the patching "
        "peak: L24/pos −1 for Qwen, L26/pos −1 for CodeGemma. v_noop = "
        "mean(fixed) − mean(buggy) is added with coefficient α (in units of "
        "‖v_noop‖). For Qwen (solid, left y-axis), the mean `edit − noop` "
        "margin moves smoothly and monotonically across both conditions; α ≈ "
        "+1 drops buggy margin by ~0.66 logits, recovering the full clean "
        "buggy↔fixed gap with a single rank-1 intervention. For CodeGemma "
        "(dashed, right y-axis; responsive subset N=20), the curves are "
        "similarly monotonic but the absolute margins are much larger (~+15) "
        "so the same Δ does not flip the argmax action — confirming that the "
        "direction governs the margin continuously in both models but "
        "CodeGemma's saturated baseline requires a much larger α to alter "
        "discrete action choice."
    )
    print(f"  wrote {out.name}")


# ------------------------------------------------------------------------------
# Figure 5: Failure table
# ------------------------------------------------------------------------------

def fig_failure_table() -> None:
    # Most recent stale cache
    candidates = sorted(RESULTS.glob("cache-qwen25_coder_15b_instruc-*"),
                        key=lambda p: p.name, reverse=True)
    if not candidates:
        print("  SKIP failure_table.png (no stale cache present)")
        return
    stale_dir = candidates[0]
    ft_path = stale_dir / "failure_table.json"
    if not ft_path.is_file():
        print(f"  SKIP failure_table.png (no failure_table.json at {ft_path})")
        return
    ft = json.loads(ft_path.read_text())

    # Tighter aspect (10x3.6 vs 12x5) — two bars + a histogram don't need
    # 12in of width. Dropped the figure-level suptitle: the paper caption
    # already provides the contextual sentence.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=300,
                              gridspec_kw={"width_ratios": [1.0, 1.4]})

    # Left: action distribution stacked bars per variant.
    # Distinct hues per action so view / test don't both read as "grey":
    # view=teal-grey, grep=blue, test=mustard, edit=red, noop=green.
    variants = list(ft.keys())
    action_names = ["view", "grep", "test", "edit", "noop"]
    colors_map = {"view": "#7fb3a0", "grep": "#5b8def", "test": "#d4a017",
                  "edit": "#aa3030", "noop": "#308030"}
    ax = axes[0]
    bottom = np.zeros(len(variants))
    xpos = np.arange(len(variants))
    for act in action_names:
        vals = []
        for v in variants:
            d = ft[v]["argmax_distribution"]
            total = ft[v]["N"]
            vals.append(100 * d.get(act, 0) / total)
        ax.bar(xpos, vals, bottom=bottom, color=colors_map[act], label=act,
               edgecolor="white", linewidth=0.5, width=0.6)
        bottom += np.array(vals)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{v}\n(N={ft[v]['N']})" for v in variants], fontsize=9)
    ax.set_xlim(-0.6, len(variants) - 0.4)
    ax.set_ylim(0, 100); ax.set_ylabel("% of tasks (argmax)")
    ax.set_title("Argmax action under stale evidence", fontsize=10, pad=6)
    # Move bar-chart legend below the plot so it doesn't sit on top of bars.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=5, fontsize=8, frameon=False, columnspacing=1.0,
              handlelength=1.4, handleheight=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Right: projection distributions per variant (overlapping histograms)
    ax = axes[1]
    qwen_clean_b = float(list(ft.values())[0]["baseline_buggy_mean_proj"])
    qwen_clean_f = float(list(ft.values())[0]["baseline_fixed_mean_proj"])
    bins = np.linspace(-9, 3, 24)
    for v, color in zip(variants, ["#9966cc", "#dd9933"]):
        projs = [r["proj"] for r in ft[v]["rows"]]
        ax.hist(projs, bins=bins, alpha=0.55, color=color, label=f"{v}")
    ax.axvline(qwen_clean_b, color="#aa3030", lw=1.5, linestyle="--",
               label=f"clean buggy mean ({qwen_clean_b:+.2f})")
    ax.axvline(qwen_clean_f, color="#308030", lw=1.5, linestyle="--",
               label=f"clean fixed mean ({qwen_clean_f:+.2f})")
    ax.set_xlabel("projection onto v_noop  (at L24, pos −1)")
    ax.set_ylabel("# tasks")
    ax.set_title("v_noop projection sits between clean-buggy and clean-fixed",
                 fontsize=10, pad=6)
    ax.legend(loc="upper left", fontsize=8,
              frameon=True, facecolor="white", edgecolor="#cccccc",
              framealpha=0.95)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    out = FIG_DIR / "failure_table.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    save_caption(out,
        "Figure 5. Failure analysis under stale evidence (Qwen, N=49 per "
        "variant). Left: argmax-action distribution. Under both "
        "`stale_misleading` (fixed code + passing tests + misleading issue text) "
        "and `stale_flaky` (same + one unrelated synthetic flaky failure), the "
        "model overwhelmingly defers to `grep` (>90%) and explicitly chooses "
        "`noop` 0% of the time. The over-edit failure mode is mild (6–8%). "
        "Right: distribution of residual projections onto v_noop. The stale-"
        "variant distributions sit *between* the clean-buggy (mean −5.53) and "
        "clean-fixed (mean +0.36) baselines: the model internally registers "
        "the test-passing signal, but routes uncertainty to information-"
        "gathering rather than abstention."
    )
    print(f"  wrote {out.name}")


# ------------------------------------------------------------------------------
# Figure 6: Negative-control heatmap (`code` variant patching)
# ------------------------------------------------------------------------------

def fig_negative_control() -> None:
    candidates = sorted(RESULTS.glob("patch-qwen25_coder_15b_instruc-*"),
                        key=lambda p: p.name, reverse=True)
    if not candidates:
        print("  SKIP negative_control.png (no `code` patching run present)")
        return
    run_dir = candidates[0]
    agg_path = run_dir / "aggregated.npz"
    if not agg_path.is_file():
        print(f"  SKIP negative_control.png (run aggregated.npz absent; re-run analyze)")
        return
    data = np.load(agg_path, allow_pickle=True)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    mean_shift = data["mean_shift_f2b"]
    layers = data["layer_indices"]
    positions = data["position_offsets"]
    im = ax.imshow(mean_shift, aspect="auto", cmap="RdBu_r",
                   vmin=-1.5, vmax=1.5, origin="lower")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{int(l)}" for l in layers], fontsize=8)
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"pos {int(p)}" for p in positions], fontsize=9)
    ax.set_title("Negative control: patching on the `code` variant (no tests)\n"
                 f"max |mean shift| = {float(np.max(np.abs(mean_shift))):.3f}", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label="clean_buggy − patched margin")

    out = FIG_DIR / "negative_control.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    save_caption(out,
        "Figure 6 (supplement). Negative control: same patching protocol as "
        "Figure 3, but on the `code` variant (no test transcript shown). The "
        "L24/pos −1 cell that dominated under `code_tests` shows no peak here, "
        "confirming that the direction we identified is specifically triggered "
        "by test evidence, not by a generic code-difference signal."
    )
    print(f"  wrote {out.name}")


def fig_monitor_roc() -> None:
    loo = RESULTS / "monitor" / "loo_curves.npz"
    if not loo.is_file():
        print(f"  SKIP monitor_roc.png (no {loo})")
        return
    d = np.load(loo, allow_pickle=True)
    fpr = d["roc_fpr"]; tpr = d["roc_tpr"]
    pr_p = d["pr_precision"]; pr_r = d["pr_recall"]
    roc_auc = float(d["roc_auc"])
    pr_auc = float(d["pr_auc"])
    op_thresh = float(d["op_threshold"])
    roc_thresh = d["roc_thresh"]
    op_idx = int(np.argmin(np.abs(roc_thresh - op_thresh)))
    op_fpr = float(fpr[op_idx]); op_tpr = float(tpr[op_idx])
    op_precision = float(d["op_precision"])
    op_recall = float(d["op_recall"])
    op_accuracy = float(d["op_accuracy"])
    op_fer = float(d["op_false_edit_rate"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    # ROC
    ax = axes[0]
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"LOOCV  AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#bbb", lw=0.8, linestyle=":")
    ax.scatter([op_fpr], [op_tpr], color="#cc1111", s=70, zorder=5,
               label=f"operating point\nFPR={op_fpr:.2f}, TPR={op_tpr:.2f}")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("false-positive rate  (fixed → predicted-edit)")
    ax.set_ylabel("true-positive rate  (buggy → predicted-edit)")
    ax.set_title("ROC — pre-edit monitor")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Precision–recall
    ax = axes[1]
    ax.plot(pr_r, pr_p, color="#308030", lw=2, label=f"LOOCV  AP = {pr_auc:.3f}")
    ax.scatter([op_recall], [op_precision], color="#cc1111", s=70, zorder=5,
               label=f"operating point\nP={op_precision:.2f}, R={op_recall:.2f}")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision–recall — pre-edit monitor")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Pre-edit monitor: project resid_pre[24, pos −1] onto v_noop  "
        "(leave-one-out CV, N=49 tasks → 98 rows)", fontsize=11)
    out = FIG_DIR / "monitor_roc.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    save_caption(
        out,
        "Figure 7. Pre-edit monitor. For each of the 49 tasks, we hold the task "
        "out, compute v_noop on the OTHER 48 tasks' mean-difference at L24 / "
        "pos −1, and project the held-out task's BUGGY and FIXED residuals onto "
        f"that unit vector. Negated projection is the classifier score (lower "
        f"raw projection = more buggy-like = predicted-edit). Under LOOCV the "
        f"projection is fully out-of-sample. ROC-AUC = {roc_auc:.3f}; AP = "
        f"{pr_auc:.3f}. At the operating point (red dot, max balanced "
        f"accuracy), precision = {op_precision:.2f}, recall = {op_recall:.2f}, "
        f"accuracy = {op_accuracy:.2f}, false-edit rate (fixed prompts "
        f"misclassified as buggy) = {op_fer:.2f}. The monitor labels are the "
        f"GROUND TRUTH `buggy vs fixed` condition (i.e. the action a correct "
        f"agent should take), not the model's overt action choice — which is "
        f"the relevant target for a pre-edit safety system."
    )
    print(f"  wrote {out.name}")


def _load_real_curves(path):
    if not path.is_file():
        return None
    d = np.load(path, allow_pickle=True)
    op_thresh = float(d["op_threshold"])
    op_idx = int(np.argmin(np.abs(d["roc_thresh"] - op_thresh)))
    raw = d["raw_projections"]; labels = d["labels"]
    return {
        "fpr": d["roc_fpr"], "tpr": d["roc_tpr"],
        "pr_p": d["pr_precision"], "pr_r": d["pr_recall"],
        "roc_auc": float(d["roc_auc"]),
        "pr_auc": float(d["pr_auc"]),
        "op_fpr": float(d["roc_fpr"][op_idx]),
        "op_tpr": float(d["roc_tpr"][op_idx]),
        "op_precision": float(d["op_precision"]),
        "op_recall": float(d["op_recall"]),
        "op_accuracy": float(d["op_accuracy"]),
        "proj_b_mean": float(d["proj_buggy_mean"]),
        "proj_f_mean": float(d["proj_fixed_mean"]),
        "gap": float(d["gap"]),
        "proj_b": raw[labels == 1],
        "proj_f": raw[labels == 0],
        "n_pairs": int((labels == 1).sum()),
    }


def fig_monitor_real_roc() -> None:
    """Three-model overlay. Prefers the N=500 SWE-bench Verified runs when
    present; falls back to the original N=100 then the legacy single-curve
    N=29 .npz otherwise. DeepSeek added in Phase B."""
    base = RESULTS / "monitor_real"
    qwen_p = base / "real_curves_qwen_n500.npz"
    cg_p = base / "real_curves_codegemma_n500.npz"
    ds_p = base / "real_curves_deepseek_n500.npz"
    if not qwen_p.exists(): qwen_p = base / "real_curves_qwen_n100.npz"
    if not cg_p.exists(): cg_p = base / "real_curves_codegemma_n100.npz"
    legacy_p = base / "real_curves.npz"

    qwen = _load_real_curves(qwen_p)
    cg = _load_real_curves(cg_p)
    ds = _load_real_curves(ds_p)
    legacy = _load_real_curves(legacy_p) if (qwen is None and cg is None and ds is None) else None
    if qwen is None and cg is None and ds is None and legacy is None:
        print(f"  SKIP monitor_real_roc.png (no real-curve npz under {base})")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5), dpi=300)
    series = []
    if qwen is not None: series.append(("Qwen-1.5B", qwen, "#1f77b4", "o"))
    if cg is not None:   series.append(("CodeGemma-7B", cg, "#ff7f0e", "s"))
    if ds is not None:   series.append(("DeepSeek-1.3B", ds, "#2ca02c", "^"))
    if not series and legacy is not None:
        series.append(("Qwen (N=29 legacy)", legacy, "#1f77b4", "o"))

    # ---- Left: ROC ----
    ax = axes[0]
    for label, d, color, marker in series:
        ax.plot(d["fpr"], d["tpr"], color=color, lw=2,
                label=f"{label} (N={d['n_pairs']})  AUC = {d['roc_auc']:.3f}")
        ax.scatter([d["op_fpr"]], [d["op_tpr"]], color=color, s=60,
                   edgecolor="black", linewidth=0.8, zorder=5, marker=marker)
    ax.plot([0, 1], [0, 1], color="#bbb", lw=0.8, linestyle=":")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title("ROC — real SWE-bench tasks")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ---- Middle: PR ----
    ax = axes[1]
    for label, d, color, marker in series:
        ax.plot(d["pr_r"], d["pr_p"], color=color, lw=2,
                label=f"{label}  AP = {d['pr_auc']:.3f}")
        ax.scatter([d["op_recall"]], [d["op_precision"]], color=color, s=60,
                   edgecolor="black", linewidth=0.8, zorder=5, marker=marker)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision–recall — real tasks")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ---- Right: projection histograms (one row per model) ----
    ax = axes[2]
    all_raw = np.concatenate([np.concatenate([d["proj_b"], d["proj_f"]]) for _, d, _, _ in series])
    bins = np.linspace(min(all_raw.min(), -7), max(all_raw.max(), 3), 22)
    for label, d, color, _ in series:
        ax.hist(d["proj_b"], bins=bins, alpha=0.45, color=color,
                edgecolor="black", linewidth=0.4,
                label=f"{label} buggy (mean {d['proj_b_mean']:+.2f})")
        ax.hist(d["proj_f"], bins=bins, alpha=0.45, color=color, hatch="///",
                edgecolor="black", linewidth=0.4,
                label=f"{label} fixed (mean {d['proj_f_mean']:+.2f})")
    if any(s[0].startswith("Qwen") for s in series):
        ax.axvline(-5.53, color="#1f77b4", lw=1, linestyle="--", alpha=0.5,
                   label="Qwen toy buggy ref")
        ax.axvline(0.36, color="#1f77b4", lw=1, linestyle=":", alpha=0.5,
                   label="Qwen toy fixed ref")
    ax.set_xlabel("projection onto frozen v_noop")
    ax.set_ylabel("# real tasks")
    ax.set_title("Projection distributions (real tasks)")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    title_parts = []
    for label, d, _, _ in series:
        title_parts.append(f"{label}: AUC={d['roc_auc']:.3f}, N={d['n_pairs']}, gap={d['gap']:+.2f}")
    fig.suptitle(
        "Real-task transfer of frozen v_noop (no retraining) — "
        + "  ·  ".join(title_parts),
        fontsize=10,
    )
    out = FIG_DIR / "monitor_real_roc.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    caption_parts = []
    for label, d, _, _ in series:
        caption_parts.append(
            f"{label}: ROC-AUC = {d['roc_auc']:.3f}, AP = {d['pr_auc']:.3f}, "
            f"precision = {d['op_precision']:.2f}, recall = {d['op_recall']:.2f}, "
            f"projection gap = {d['gap']:+.2f} over N = {d['n_pairs']} paired tasks"
        )
    save_caption(out,
        "Figure 8. Real-task generalisation of the FROZEN, toy-trained v_noop "
        "(no retraining) evaluated on paired prompts derived from "
        "SWE-bench_Verified. Same protocol as Figure 7; the residual at the "
        "intervention site (L24/pos −1 for Qwen, L26/pos −1 for CodeGemma) is "
        "projected onto its model's toy-trained unit vector and the resulting "
        "scalar is the classifier score. "
        + " ".join(caption_parts) + ". "
        "Both models exhibit clean linear separability on real GitHub-issue "
        "prompts, confirming that the no-op direction is not a toy-distribution "
        "artefact."
    )
    print(f"  wrote {out.name}")


def main() -> int:
    print("=== render_all.py ===")
    fig_paired_task_diagram()
    fig_behavioral_delta()
    fig_patching_heatmap()
    fig_steering_dose_response()
    fig_failure_table()
    fig_negative_control()
    fig_monitor_roc()
    fig_monitor_real_roc()
    fig_sae_decomposition()
    fig_sae_decomposition_codegemma()
    fig_sae_topk_curve()
    fig_incontext_attribution()
    print("done.")
    return 0


# ------------------------------------------------------------------------------
# Figure 9: SAE feature decomposition of v_noop
# ------------------------------------------------------------------------------

def _omp_decompose(W_dec, v_centred, k):
    """Greedy OMP: pick k decoder rows that minimise ||v - W_dec[chosen].T @ c||."""
    residual = v_centred.clone()
    chosen: list[int] = []
    sub = W_dec[:0]
    sol = torch.zeros(0)
    for _ in range(k):
        scores = (W_dec @ residual).abs()
        if chosen:
            scores[torch.tensor(chosen)] = -1.0
        best = int(scores.argmax().item())
        chosen.append(best)
        sub = W_dec[chosen]
        sol = torch.linalg.lstsq(sub.T, v_centred.unsqueeze(1)).solution.squeeze(1)
        residual = v_centred - sub.T @ sol
    return chosen, sol, sub


def _cos(a, b):
    a = a.float(); b = b.float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def fig_sae_decomposition() -> None:
    """Four-panel Figure 9: v_noop SAE decomposition.

      A: cosine vs k (OMP and encoder-TopK, both SAEs) with the 0.85 brief gate
      B: per-feature signed contribution along v_noop for the OMP top-20, top-5
         annotated with semantic labels from the characterisation run
      C: ablation Δ(buggy-fixed gap) for OMP top-{8,32,128} + sae_recon sanity
      D: textual table summarising the top-5 OMP feature semantics
    """
    sys.path.insert(0, str(REPO / "src"))
    from no_op_circuit.interp.sae import SAEConfig, TopKSAE  # type: ignore

    sae_task_path = RESULTS / "sae" / "qwen_l24_resid_pre_TASK_d4096_k16.pt"
    sae_generic_path = RESULTS / "sae" / "qwen_l24_resid_pre_d24576_k32.pt"
    v_noop_path = RESULTS / "steer-20260516T021522Z" / "v_noop.pt"
    distributed_path = RESULTS / "sae" / "v_noop_features_DISTRIBUTED.json"
    characterise_path = RESULTS / "sae" / "feature_characterisations_DISTRIBUTED.json"
    ablate_path = RESULTS / "sae" / "ablate-distributed" / "ablation_results.json"

    # Bail with a stub if any file is missing (paper builds in CI without SAE artefacts).
    missing = [p for p in (sae_task_path, sae_generic_path, v_noop_path,
                           distributed_path, characterise_path, ablate_path)
               if not p.exists()]
    if missing:
        print(f"[fig9] missing artefacts, skipping: {[p.name for p in missing]}")
        return

    v_blob = torch.load(v_noop_path, map_location="cpu", weights_only=False)
    v = v_blob["direction"].float()

    # Compute OMP and TopK-enc curves for both SAEs
    ks = [8, 16, 32, 64, 128, 256, 512]
    curves: dict[str, dict[str, list[float]]] = {}
    for label, path in [("task SAE (d=4096)", sae_task_path),
                        ("generic SAE (d=24576)", sae_generic_path)]:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = SAEConfig(**blob["config"])
        sae = TopKSAE(cfg); sae.load_state_dict(blob["state_dict"]); sae.eval()
        W_dec = sae.W_dec.detach().float()
        b_dec = sae.b_dec.detach().float()
        v_centred = v - b_dec
        with torch.no_grad():
            pre = sae.encode_pre(v.unsqueeze(0))[0]
        cs_enc, cs_omp = [], []
        for k in ks:
            if k > cfg.d_sae:
                cs_enc.append(float("nan")); cs_omp.append(float("nan"))
                continue
            idx = pre.abs().topk(k).indices
            a = torch.zeros_like(pre); a[idx] = pre[idx]
            v_hat_enc = (a @ W_dec) + b_dec
            cs_enc.append(_cos(v_hat_enc, v))
            _, sol_k, sub = _omp_decompose(W_dec, v_centred, k)
            v_hat_omp = (sub.T @ sol_k) + b_dec
            cs_omp.append(_cos(v_hat_omp, v))
        curves[label] = {"enc": cs_enc, "omp": cs_omp}

    # Panel B / D inputs
    dist_blob = json.loads(distributed_path.read_text())
    top_feats = dist_blob["top_features"]
    top_n_b = min(20, len(top_feats))
    contribs = [f["v_contribution"] for f in top_feats[:top_n_b]]
    feat_ids = [int(f["feature_idx"]) for f in top_feats[:top_n_b]]

    # Hand-mapped semantic labels for the top-5 OMP features (verified against
    # results/sae/feature_characterisations_DISTRIBUTED.json). Short labels
    # suitable for the panel; longer narrative goes in §5.2.
    semantic_labels = {
        2669: "error/Error promoter",
        1954: "comment-context (#)",
        2950: "suppresses edit/view/git tokens",
        3129: "traceback promoter",
        3171: "'already'/'Already' promoter",
    }

    # Panel C inputs
    abl_blob = json.loads(ablate_path.read_text())
    pairs: dict[str, dict] = {}
    for r in abl_blob["rows"]:
        pairs.setdefault(r["task_id"], {})[r["condition"]] = r
    complete = [p for p in pairs.values() if "buggy" in p and "fixed" in p]

    def gap_stats(lab):
        gs = [p["buggy"]["margins"][lab] - p["fixed"]["margins"][lab] for p in complete]
        mu = float(np.mean(gs)); se = float(np.std(gs, ddof=1) / np.sqrt(len(gs)))
        return mu, se

    abl_labels_order = ["clean", "sae_recon", "ablate_omp_top8",
                        "ablate_omp_top32", "ablate_omp_top128"]
    abl_means, abl_sems = [], []
    for lab in abl_labels_order:
        mu, se = gap_stats(lab); abl_means.append(mu); abl_sems.append(se)

    # Random-8 baselines (two): "random-any" (Tier 1, drawn from full d_sae)
    # and "random-firing" (Tier 2, drawn from features with firing-count >= 5
    # excluding OMP top-128). Compute pooled mean gap across all seeds for each.
    def _pooled_random_gap(d_path: Path) -> tuple[float, float] | tuple[None, None]:
        if not d_path.exists():
            return None, None
        per_pair_gaps: list[float] = []
        for d in sorted(d_path.glob("seed_*")):
            blob = json.loads((d / "ablation_results.json").read_text())
            p2: dict[str, dict] = {}
            for r in blob["rows"]:
                p2.setdefault(r["task_id"], {})[r["condition"]] = r
            for pair in p2.values():
                if "buggy" in pair and "fixed" in pair:
                    per_pair_gaps.append(
                        pair["buggy"]["margins"]["ablate_random8"]
                        - pair["fixed"]["margins"]["ablate_random8"]
                    )
        if not per_pair_gaps:
            return None, None
        arr = np.asarray(per_pair_gaps)
        return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))

    random_any_mean, random_any_sem = _pooled_random_gap(RESULTS / "sae" / "ablate-random")
    random_fire_mean, random_fire_sem = _pooled_random_gap(RESULTS / "sae" / "ablate-random-firing")

    # ----- Plot -----
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    axA, axB, axC, axD = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # Panel A: cosine vs k
    colors = {"task SAE (d=4096)": "#1f77b4", "generic SAE (d=24576)": "#ff7f0e"}
    for label, c in curves.items():
        axA.plot(ks, c["omp"], "-o", color=colors[label], linewidth=2,
                 label=f"{label} — OMP (best k-sparse)")
        axA.plot(ks, c["enc"], "--s", color=colors[label], linewidth=1.5,
                 alpha=0.7, label=f"{label} — encoder TopK")
    axA.axhline(0.85, color="gray", linestyle=":", linewidth=1)
    axA.text(ks[0] * 1.1, 0.86, "brief threshold (cos = 0.85)", color="gray", fontsize=8)
    axA.set_xscale("log")
    axA.set_xlabel("number of active features k")
    axA.set_ylabel(r"cosine($\hat v$, $v_{\mathrm{noop}}$)")
    axA.set_title("A. v_noop reconstruction is geometrically dense\n(OMP: k≈128 → cos 0.80; k≈256 → cos 0.85)",
                  fontsize=11, loc="left")
    axA.set_ylim(-0.05, 1.05)
    axA.legend(loc="lower right", fontsize=8)
    axA.grid(True, alpha=0.3)

    # Panel B: per-feature signed contribution along v_noop.
    # Highlight the top-5 with a thicker black edge — Panel D names them
    # — instead of inline arrow-labels which used to overlap each other.
    x = np.arange(top_n_b)
    bar_colors = ["#2ca02c" if c >= 0 else "#d62728" for c in contribs]
    edge_widths = [1.6 if i < 5 else 0.5 for i in range(top_n_b)]
    axB.bar(x, contribs, color=bar_colors, edgecolor="black", linewidth=edge_widths)
    axB.set_xticks(x)
    axB.set_xticklabels([str(i) for i in feat_ids], rotation=70, fontsize=8)
    axB.set_xlabel("feature_idx (rank-ordered by |contribution|)")
    axB.set_ylabel(r"signed contribution to $v_{\mathrm{noop}}$")
    axB.set_title("B. Top-20 OMP features: contributions are small and balanced\n"
                  "(max single contribution = 0.24 of ‖v‖ = 5.89; bold-edge bars = top-5, see Panel D)",
                  fontsize=10, loc="left")
    axB.axhline(0, color="black", linewidth=0.5)
    axB.grid(True, axis="y", alpha=0.3)

    # Panel C: ablation Δ(buggy − fixed), with up to two random baselines.
    # Order: clean, SAE recon, random-any, random-firing, OMP top-8, top-32, top-128.
    c_means = list(abl_means[:2]); c_sems = list(abl_sems[:2])
    c_labels = ["clean", "SAE\nrecon"]; c_colors = ["#888", "#888"]
    if random_any_mean is not None:
        c_means.append(random_any_mean); c_sems.append(random_any_sem)
        c_labels.append("random-8\n(any, 10 seeds)"); c_colors.append("#e67e22")
    if random_fire_mean is not None:
        c_means.append(random_fire_mean); c_sems.append(random_fire_sem)
        c_labels.append("random-8\n(firing≥5)"); c_colors.append("#f5b041")
    c_means.extend(abl_means[2:]); c_sems.extend(abl_sems[2:])
    c_labels.extend(["OMP\ntop-8", "OMP\ntop-32", "OMP\ntop-128"])
    c_colors.extend(["#1f77b4"] * 3)

    pos = np.arange(len(c_means))
    clean_mu = c_means[0]
    axC.bar(pos, c_means, yerr=c_sems, color=c_colors,
            capsize=4, edgecolor="black", linewidth=0.5)
    axC.axhline(clean_mu, color="black", linestyle="--", linewidth=0.8,
                label=f"clean baseline ({clean_mu:+.2f})")
    axC.set_xticks(pos)
    axC.set_xticklabels(c_labels, fontsize=8.5)
    axC.set_ylabel(r"mean (margin$_\mathrm{buggy}$ − margin$_\mathrm{fixed}$)  [logits]")
    axC.set_title("C. Ablation: behavioral signal concentrated in 8 features;\n"
                  "random-8 (orange = any; gold = firing≥5) ≈ clean.",
                  fontsize=10.5, loc="left")
    axC.legend(loc="upper right", fontsize=8)
    axC.grid(True, axis="y", alpha=0.3)
    # Annotate % reduction only on the OMP bars — the random baselines
    # are visibly at the clean line so labeling them "+0%/-0%" was just
    # noise. Place labels INSIDE each OMP bar (white text near the top)
    # so adjacent labels can't collide horizontally.
    for i, lbl in enumerate(c_labels):
        if not lbl.startswith("OMP"):
            continue
        pct = (c_means[i] - clean_mu) / clean_mu * 100
        axC.text(pos[i], c_means[i] - 0.04, f"{pct:+.0f}%",
                 ha="center", va="top", fontsize=8.5, color="white",
                 fontweight="bold")

    # Panel D: semantic summary for top-5 features
    axD.axis("off")
    axD.set_title("D. Top-5 OMP features make mechanistic sense for 'no-op'",
                  fontsize=11, loc="left")
    header = [
        ("rank", "feat", "coef", "contrib", "interpretation"),
    ]
    rows_d = []
    # Tightened to fit the interp column without overflow. Dropped the
    # redundant "v_noop" prefix (the entire table decomposes v_noop)
    # and shortened "action-suppression" to fit.
    descriptions = {
        2669: "promotes 'error/Error' → SUBTRACTS error-attention",
        1954: "fires on '#' (Python comment marker) → ADDS",
        2950: "suppresses '\\tedit'/'\\tview'/'\\tgit' → ADDS action-suppr.",
        3129: "promotes 'traceback' → SUBTRACTS",
        3171: "promotes 'already', suppresses 'corrected' → ADDS",
    }
    for r, f in enumerate(top_feats[:5]):
        fid = int(f["feature_idx"])
        coef = f["omp_coef"]; contrib = f["v_contribution"]
        rows_d.append((str(r), str(fid), f"{coef:+.2f}", f"{contrib:+.3f}",
                       descriptions.get(fid, "(see characterisation JSON)")))
    cell_text = header + rows_d
    # colWidths are fractions of axes width, must SUM TO ≤ 1.0. Previous
    # iterations had two failure modes: (1) summing > 1.0 caused
    # matplotlib to squeeze and truncate "contrib" → "contri"; (2)
    # giving "rank" only 0.05 truncated it to "ran". Tightened the
    # numeric columns (they hold ≤ 6-char values) and gave the slack to
    # `interp` so the longest description fits without right-edge
    # overflow.
    table = axD.table(cellText=cell_text, loc="center", colLoc="left",
                      cellLoc="left",
                      colWidths=[0.07, 0.07, 0.08, 0.10, 0.68])
    table.auto_set_font_size(False); table.set_fontsize(8)
    table.scale(1.0, 1.5)
    # Make header row bold
    for j in range(5):
        cell = table[(0, j)]
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#eeeeee")

    plt.tight_layout()
    out = FIG_DIR / "sae_decomposition.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Figure 9. SAE feature decomposition of v_noop (Qwen2.5-Coder-1.5B-Instruct, "
        "L24 resid_pre, TopK SAE with d_sae=4096, k=16, EV=0.976 on the 1.23M-position "
        "task-distribution corpus). "
        "(A) Reconstruction cosine vs the number of active features for two SAEs and "
        "two selection rules. Even with optimal greedy selection (OMP) one needs "
        "k≈128 features to reach cos≥0.80 and k≈256 to clear 0.85: v_noop is "
        "geometrically dense in the SAE basis. The trained encoder is also suboptimal "
        "for OOD steering directions like v_noop (encoder-TopK lags OMP by 0.2–0.4 "
        "cosine at every k). "
        "(B) The OMP top-20 features' signed contributions to v_noop: the largest "
        "single contribution is 0.24 of ||v|| = 5.89; the distribution is small and "
        "balanced. (C) Per-feature-set ablation on 296 paired buggy/fixed prompts, "
        "measured as the gap in (edit−noop) logit margin between buggy and fixed "
        "conditions. The OMP top-8 alone removes 33% of the v_noop behavioural "
        "signal; OMP top-128 removes only 36%. Two matched random-8 baselines (10 "
        "seeds each) anchor specificity: random-any (orange; drawn from {0,…,4095}\\"
        "OMP_top128) gives ~0% reduction (Mann-Whitney p ≈ 3×10⁻⁹⁶); random-firing "
        "(gold; drawn from the 45 firing≥5 features outside OMP top-128) gives "
        "−0.85% mean across seeds (95% CI [−9.3%, +7.3%], per-seed range [−21.7%, "
        "+19.6%]; Mann-Whitney p ≈ 4×10⁻¹⁷). Only OMP's signed-coefficient-aware "
        "selection produces a consistent net reduction. (D) The top-5 OMP features "
        "form a coherent 'no-op' circuit: subtract error- and traceback-attention, "
        "add 'already' semantics, add action-token suppression."
    )
    save_caption(out, caption)
    print(f"[fig9] wrote {out.relative_to(REPO)}")


# ------------------------------------------------------------------------------
# Figure 10: SAE feature decomposition of v_noop on CodeGemma-7B
# ------------------------------------------------------------------------------

def fig_sae_decomposition_codegemma() -> None:
    """Same 4-panel layout as Figure 9, computed on CodeGemma's L26 SAE.

    Panel C has no random baselines (Phase 6 firing-baseline skipped per
    decision-gate spec after OMP top-8 ablation failed to clear p<0.05);
    the bar set is just clean / SAE-recon / OMP top-{8,32,128}.
    """
    sys.path.insert(0, str(REPO / "src"))
    from no_op_circuit.interp.sae import SAEConfig, TopKSAE  # type: ignore

    sae_path = RESULTS / "sae" / "codegemma_l26_resid_pre_TASK_d8192_k48.pt"
    v_noop_path = RESULTS / "steer-codegemma_7b_it-20260516T051943Z" / "v_noop.pt"
    distributed_path = RESULTS / "sae" / "v_noop_features_codegemma_DISTRIBUTED.json"
    characterise_path = RESULTS / "sae" / "feature_characterisations_codegemma.json"
    ablate_path = RESULTS / "sae" / "ablate-codegemma-distributed" / "ablation_results.json"

    missing = [p for p in (sae_path, v_noop_path, distributed_path,
                            characterise_path, ablate_path)
               if not p.exists()]
    if missing:
        print(f"[fig10] missing artefacts, skipping: {[p.name for p in missing]}")
        return

    v_blob = torch.load(v_noop_path, map_location="cpu", weights_only=False)
    v = v_blob["direction"].float()

    blob = torch.load(sae_path, map_location="cpu", weights_only=False)
    cfg = SAEConfig(**blob["config"])
    sae = TopKSAE(cfg); sae.load_state_dict(blob["state_dict"]); sae.eval()
    W_dec = sae.W_dec.detach().float()
    b_dec = sae.b_dec.detach().float()
    v_centred = v - b_dec
    with torch.no_grad():
        pre = sae.encode_pre(v.unsqueeze(0))[0]
    ks = [8, 16, 32, 64, 128, 256, 512, 1024]
    cs_enc, cs_omp = [], []
    for k in ks:
        if k > cfg.d_sae:
            cs_enc.append(float("nan")); cs_omp.append(float("nan"))
            continue
        idx = pre.abs().topk(k).indices
        a = torch.zeros_like(pre); a[idx] = pre[idx]
        v_hat_enc = (a @ W_dec) + b_dec
        cs_enc.append(_cos(v_hat_enc, v))
        _, sol_k, sub = _omp_decompose(W_dec, v_centred, k)
        v_hat_omp = (sub.T @ sol_k) + b_dec
        cs_omp.append(_cos(v_hat_omp, v))

    # Panel B / D inputs
    dist_blob = json.loads(distributed_path.read_text())
    top_feats = dist_blob["top_features"]
    top_n_b = min(20, len(top_feats))
    contribs = [f["v_contribution"] for f in top_feats[:top_n_b]]
    feat_ids = [int(f["feature_idx"]) for f in top_feats[:top_n_b]]

    # Hand-mapped semantic labels for the cleanly-interpretable CodeGemma top
    # features (others were polysemantic — see characterisation JSON).
    semantic_labels_cg = {
        6974: "promotes ' passed'/' OK'/' pass'  (test-pass)",
    }

    # Panel C inputs
    abl_blob = json.loads(ablate_path.read_text())
    pairs: dict[str, dict] = {}
    for r in abl_blob["rows"]:
        pairs.setdefault(r["task_id"], {})[r["condition"]] = r
    complete = [p for p in pairs.values() if "buggy" in p and "fixed" in p]

    def gap_stats(lab):
        gs = [p["buggy"]["margins"][lab] - p["fixed"]["margins"][lab] for p in complete]
        mu = float(np.mean(gs)); se = float(np.std(gs, ddof=1) / np.sqrt(len(gs)))
        return mu, se

    abl_labels_order = ["clean", "sae_recon",
                        "ablate_omp_top8", "ablate_omp_top32", "ablate_omp_top128"]
    abl_means, abl_sems = [], []
    for lab in abl_labels_order:
        mu, se = gap_stats(lab); abl_means.append(mu); abl_sems.append(se)

    # ----- Plot -----
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    axA, axB, axC, axD = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # Panel A: cosine vs k (single SAE, two selection rules)
    axA.plot(ks, cs_omp, "-o", color="#e74c3c", linewidth=2,
             label="CodeGemma SAE — OMP (best k-sparse)")
    axA.plot(ks, cs_enc, "--s", color="#e74c3c", linewidth=1.5, alpha=0.7,
             label="CodeGemma SAE — encoder TopK")
    axA.axhline(0.85, color="gray", linestyle=":", linewidth=1)
    axA.text(ks[0] * 1.1, 0.86, "cos = 0.85", color="gray", fontsize=8)
    axA.set_xscale("log")
    axA.set_xlabel("number of active features k")
    axA.set_ylabel(r"cosine($\hat v$, $v_{\mathrm{noop,cg}}$)")
    axA.set_title("A. v_noop_cg reconstruction is dense (replicates Qwen)\n"
                  "(OMP: k≈128 → cos 0.68; k≈256 → cos 0.78)",
                  fontsize=10.5, loc="left")
    axA.set_ylim(-0.1, 1.05)
    axA.legend(loc="lower right", fontsize=8)
    axA.grid(True, alpha=0.3)

    # Panel B: per-feature signed contribution.
    # Only F6974 is cleanly interpretable; mark it with a thicker black
    # edge instead of an inline arrow-label (the leader line overlapped
    # the title at every render scale).
    x = np.arange(top_n_b)
    bar_colors = ["#2ca02c" if c >= 0 else "#d62728" for c in contribs]
    interp_bar_idx = [i for i in range(top_n_b) if feat_ids[i] in semantic_labels_cg]
    edge_widths = [1.6 if i in interp_bar_idx else 0.5 for i in range(top_n_b)]
    axB.bar(x, contribs, color=bar_colors, edgecolor="black", linewidth=edge_widths)
    axB.set_xticks(x)
    axB.set_xticklabels([str(i) for i in feat_ids], rotation=70, fontsize=8)
    axB.set_xlabel("feature_idx (rank-ordered by |contribution|)")
    axB.set_ylabel(r"signed contribution to $v_{\mathrm{noop,cg}}$")
    axB.set_title("B. OMP top-20 contributions (max 0.21 of ‖v‖ = 6.68;\n"
                  "bold-edge bar = the one cleanly-interpretable feature, see Panel D)",
                  fontsize=10, loc="left")
    axB.axhline(0, color="black", linewidth=0.5)
    axB.grid(True, axis="y", alpha=0.3)

    # Panel C: ablation Δ(buggy − fixed)
    pos = np.arange(len(abl_labels_order))
    clean_mu = abl_means[0]
    bar_colors_c = ["#888"] * 2 + ["#1f77b4"] * 3
    axC.bar(pos, abl_means, yerr=abl_sems, color=bar_colors_c,
            capsize=4, edgecolor="black", linewidth=0.5)
    axC.axhline(clean_mu, color="black", linestyle="--", linewidth=0.8,
                label=f"clean baseline ({clean_mu:+.2f})")
    axC.set_xticks(pos)
    axC.set_xticklabels(["clean", "SAE\nrecon",
                         "OMP\ntop-8", "OMP\ntop-32", "OMP\ntop-128"],
                        fontsize=9)
    axC.set_ylabel(r"mean (margin$_\mathrm{buggy}$ − margin$_\mathrm{fixed}$)  [logits]")
    # Shorter 2-line title — the previous version's second line was 87
    # chars at fontsize 10 and ran rightward into Panel D's title area.
    axC.set_title("C. Ablation: top-8 not significant (+6%, p=0.10);\n"
                  "top-32/128 INCREASE gap (Qwen replication fails)",
                  fontsize=10, loc="left")
    axC.legend(loc="upper right", fontsize=8)
    axC.grid(True, axis="y", alpha=0.3)
    # Place pct labels INSIDE the bar (white, bold) so they can't collide
    # with the title or with neighbouring labels. For CodeGemma the
    # top-32 / top-128 bars sit ABOVE the clean baseline; the inside-top
    # rule keeps the label readable in both directions.
    for i, lab in enumerate(abl_labels_order):
        if lab in ("clean", "sae_recon"):
            continue
        pct = (abl_means[i] - clean_mu) / clean_mu * 100
        axC.text(pos[i], abl_means[i] - 0.10, f"{pct:+.0f}%",
                 ha="center", va="top", fontsize=8.5, color="white",
                 fontweight="bold")

    # Panel D: feature summary (CodeGemma's mixed-interpretability finding)
    axD.axis("off")
    axD.set_title("D. Only F6974 is cleanly interpretable; others polysemantic",
                  fontsize=10.5, loc="left")
    header = [("rank", "feat", "coef", "contrib", "interpretation")]
    rows_d = []
    # Tightened to fit the interp column width without right-edge overflow.
    descriptions_cg = {
        450: "polysemantic; lens: rare non-English tokens",
        5718: "polysemantic; fires on 'RGB' / whitespace",
        6974: "promotes 'passed', 'OK', 'pass' → test-pass (CLEAN)",
        8017: "polysemantic; fires on 'kernel' / 'size'",
        3309: "suppresses long newline runs; lens noisy",
    }
    for r, f in enumerate(top_feats[:5]):
        fid = int(f["feature_idx"])
        rows_d.append((str(r), str(fid), f"{f['omp_coef']:+.2f}",
                       f"{f['v_contribution']:+.3f}",
                       descriptions_cg.get(fid, "(see characterisation JSON)")))
    cell_text = header + rows_d
    # colWidths must SUM TO ≤ 1.0; old [0.07,0.08,0.09,0.10,1.20] = 1.54
    # caused matplotlib to squeeze and truncate "contrib" → "contri".
    table = axD.table(cellText=cell_text, loc="center", colLoc="left",
                      cellLoc="left",
                      colWidths=[0.07, 0.07, 0.08, 0.10, 0.68])
    table.auto_set_font_size(False); table.set_fontsize(8)
    table.scale(1.0, 1.5)
    for j in range(5):
        cell = table[(0, j)]
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#eeeeee")

    plt.tight_layout()
    out = FIG_DIR / "sae_decomposition_codegemma.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Figure 10. SAE feature decomposition of v_noop on CodeGemma-7B-it "
        "(L26 resid_pre, task-distribution TopK SAE with d_sae=8192, k=48, "
        "EV=0.830 on 1.20M positions; same 4-panel layout as Figure 9). "
        "(A) v_noop_cg reconstruction cosine vs number of active features — "
        "the dense-in-basis pattern replicates Qwen qualitatively (OMP k=128 "
        "→ cos 0.68, k=256 → cos 0.78). (B) Top-20 OMP signed contributions "
        "to v_noop_cg, max 0.21 of ‖v‖=6.68. (C) Per-feature-set ablation on "
        "146 paired buggy/fixed prompts (2 of 148 tasks dropped for exceeding "
        "the 2400-token cap needed to avoid A10G OOM on the 7B model): OMP "
        "top-8 reduction +6.4% (CI [−3.2%, +16.4%], Wilcoxon p=0.103 — NOT "
        "significant), top-32 and top-128 INCREASE the gap. The behavioural "
        "specificity claim that holds on Qwen (+33.1%, p<10⁻¹²) does NOT "
        "replicate on CodeGemma in our data. (D) Only feature 6974 (rank 2) "
        "is cleanly interpretable — a test-pass promoter, the analogue of "
        "Qwen's 'already-done' circuit element; the other top features have "
        "polysemantic logit-lens output."
    )
    save_caption(out, caption)
    print(f"[fig10] wrote {out.relative_to(REPO)}")


# ------------------------------------------------------------------------------
# Figure 11: cumulative OMP top-K ablation curve (Qwen)
# ------------------------------------------------------------------------------

def fig_sae_topk_curve() -> None:
    """One-panel curve: per-task gap-reduction % vs k for the OMP cumulative
    top-k ablation on Qwen. Shaded 95% bootstrap CI."""
    ablation_path = RESULTS / "sae" / "ablate-qwen-topk-interp" / "ablation_results.json"
    if not ablation_path.exists():
        print(f"[fig11] {ablation_path.name} not present, skipping")
        return

    blob = json.loads(ablation_path.read_text())
    pairs: dict[str, dict] = {}
    for r in blob["rows"]:
        pairs.setdefault(r["task_id"], {})[r["condition"]] = r
    complete = [p for p in pairs.values() if "buggy" in p and "fixed" in p]
    n = len(complete)

    def gap(label: str) -> "np.ndarray":
        return np.array([p["buggy"]["margins"][label] - p["fixed"]["margins"][label]
                          for p in complete])

    clean = gap("clean")
    clean_mu = float(clean.mean())

    # All ablate_omp_topK labels, sorted by K
    import re
    labels = sorted(
        [lab for lab in blob["rows"][0]["margins"].keys()
         if lab.startswith("ablate_omp_top")],
        key=lambda s: int(re.findall(r"\d+", s)[0]),
    )
    ks = [int(re.findall(r"\d+", s)[0]) for s in labels]
    rng = np.random.default_rng(0)
    B = 10_000
    mean_pct = []; ci_lo = []; ci_hi = []
    for lab in labels:
        ab = gap(lab); diffs = clean - ab
        mean_pct.append(diffs.mean() / clean_mu * 100)
        boot = np.array([diffs[rng.integers(0, n, size=n)].mean() for _ in range(B)])
        boot_pct = boot / clean_mu * 100
        ci_lo.append(float(np.quantile(boot_pct, 0.025)))
        ci_hi.append(float(np.quantile(boot_pct, 0.975)))
    mean_pct = np.array(mean_pct); ci_lo = np.array(ci_lo); ci_hi = np.array(ci_hi)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.fill_between(ks, ci_lo, ci_hi, color="#1f77b4", alpha=0.2, label="95% bootstrap CI")
    ax.plot(ks, mean_pct, "-o", color="#1f77b4", linewidth=2, markersize=6,
            label="OMP cumulative top-k reduction")
    # Mark the original top-8 anchor
    if 8 in ks:
        i8 = ks.index(8)
        ax.scatter([8], [mean_pct[i8]], s=120, color="#d62728", zorder=5,
                   label=f"§5.2 top-8 anchor ({mean_pct[i8]:+.1f}%)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks], fontsize=8)
    ax.set_xlabel("OMP cumulative top-k features ablated")
    ax.set_ylabel("mean (clean − ablated) buggy-fixed gap reduction  [%]")
    ax.set_title("Figure 11. Cumulative OMP top-k ablation curve (Qwen, N=148 paired tasks)",
                 fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "sae_topk_curve.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Quick verdict for the caption (sharpen vs keep)
    if 1 in ks and 8 in ks:
        ratio_1_over_8 = mean_pct[ks.index(1)] / mean_pct[ks.index(8)]
    else:
        ratio_1_over_8 = float("nan")
    caption = (
        f"Figure 11. Cumulative OMP top-k ablation on Qwen (N = {n} paired buggy/fixed "
        f"tasks). x-axis: number of cumulative OMP features ablated (top-1 through "
        f"top-{ks[-1]}, log scale). y-axis: mean per-task reduction in the buggy-fixed "
        f"margin gap (positive = ablation reduced the v_noop signal). Shaded band = 95% "
        f"bootstrap CI (B=10,000). Red dot marks the existing §5.2 top-8 anchor. The "
        f"curve starts at {mean_pct[0]:+.1f}% at k=1, climbs to {mean_pct[ks.index(8) if 8 in ks else -1]:+.1f}% "
        f"at k=8 (top-1/top-8 ratio = {ratio_1_over_8:.2f}), and plateaus near "
        f"{mean_pct[-1]:+.1f}% by k={ks[-1]}."
    )
    save_caption(out, caption)
    print(f"[fig11] wrote {out.relative_to(REPO)}")


# ------------------------------------------------------------------------------
# Figure 12: in-context attribution of v_noop projection
# ------------------------------------------------------------------------------

def fig_incontext_attribution() -> None:
    """Two-panel: (left) right-aligned mean projection at each offset from the
    Action position, buggy vs fixed; (right) bar chart of per-token mean
    projection per section."""
    npz_path = RESULTS / "attribution" / "incontext_projections.npz"
    sec_path = RESULTS / "attribution" / "section_projections.json"
    if not npz_path.exists() or not sec_path.exists():
        print(f"[fig12] attribution artefacts not present, skipping")
        return

    data = np.load(str(npz_path), allow_pickle=True)
    offsets = data["offsets"]
    mean_b = data["mean_buggy"]; sem_b = data["sem_buggy"]
    mean_f = data["mean_fixed"]; sem_f = data["sem_fixed"]
    token_strs = list(data["token_strs"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300,
                                    gridspec_kw={"width_ratios": [2, 1]})

    # Left: per-offset projection with buggy/fixed bands
    axL.fill_between(offsets, mean_b - sem_b, mean_b + sem_b,
                     color="#d62728", alpha=0.2)
    axL.fill_between(offsets, mean_f - sem_f, mean_f + sem_f,
                     color="#2ca02c", alpha=0.2)
    axL.plot(offsets, mean_b, "-o", color="#d62728", linewidth=1.5, markersize=4,
             label="buggy (mean ± sem)")
    axL.plot(offsets, mean_f, "-o", color="#2ca02c", linewidth=1.5, markersize=4,
             label="fixed (mean ± sem)")
    axL.axhline(0, color="black", linewidth=0.4)
    axL.set_xticks(offsets[::2])
    axL.set_xticklabels(
        [f"{int(offsets[i])}\n{token_strs[i]!r}" for i in range(0, len(offsets), 2)],
        rotation=70, fontsize=6, ha="right",
    )
    axL.set_xlabel("offset from Action position (token)")
    axL.set_ylabel(r"projection $\langle\mathrm{resid\_pre}, \hat v\rangle$")
    axL.set_title(
        "A. Per-offset v_noop projection on cached last 32 tokens (N=49 toy tasks)\n"
        "Token IDs identical between buggy/fixed; differential = attentional signal",
        fontsize=10, loc="left",
    )
    axL.legend(loc="lower left", fontsize=9)
    axL.grid(True, alpha=0.3)

    # Right: per-section mean per-token projection (loaded from JSON)
    secs = json.loads(sec_path.read_text())["sections"]
    section_names = list(secs.keys())
    nice_names = {"question_text": "question text\n(offsets −31…−19)",
                  "chat_template": "chat-template end\n(offsets −18…−7)",
                  "action_suffix": "Action: suffix\n(offsets −6…0)"}
    labels_pretty = [nice_names.get(n, n) for n in section_names]
    means_b = [secs[n]["mean_buggy"] for n in section_names]
    means_f = [secs[n]["mean_fixed"] for n in section_names]
    diffs = [secs[n]["differential"] for n in section_names]
    x = np.arange(len(section_names))
    w = 0.35
    axR.bar(x - w/2, means_b, w, color="#d62728", alpha=0.8, label="buggy")
    axR.bar(x + w/2, means_f, w, color="#2ca02c", alpha=0.8, label="fixed")
    axR.set_xticks(x); axR.set_xticklabels(labels_pretty, fontsize=8)
    axR.axhline(0, color="black", linewidth=0.4)
    axR.set_ylabel("mean per-token projection")
    axR.set_title("B. Section breakdown — differential concentrates in Action: suffix",
                  fontsize=10, loc="left")
    axR.legend(loc="upper left", fontsize=9)
    axR.grid(True, axis="y", alpha=0.3)
    # Annotate differential above each pair
    for i, d in enumerate(diffs):
        ymax = max(means_b[i], means_f[i]) + 0.4
        axR.text(i, ymax, f"Δ = +{d:.2f}", ha="center", fontsize=9, color="#444")

    plt.tight_layout()
    out = FIG_DIR / "incontext_attribution.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Figure 12. In-context attribution of v_noop. Toy substrate (N=49 paired tasks). "
        "The cached residual stream covers only the last 32 token positions of each "
        "prompt (an artefact of the existing cache); within this window the token IDs "
        "are identical across all tasks and between buggy/fixed conditions (chat-"
        "template close + canonical question text + 'Action: ' suffix), so any "
        "buggy-vs-fixed projection differential at a given offset is purely attentional "
        "information flow from the earlier (varying) test/code content. (A) Mean L24 "
        "resid_pre projection onto v_noop at each offset from the Action position. The "
        "differential grows monotonically as we approach the Action position, from "
        "≈+1.0 logit-units at offset −31 to +5.9 at offset 0 (the Action token "
        "itself). (B) Three-section split of (A) by token role: the differential is 4.5× "
        "stronger in the 6-token 'Action:' suffix (+3.34) than in the 13-token question "
        "text (+0.99). v_noop information is present throughout the prompt tail but "
        "crystallises sharply at the action-suffix positions."
    )
    save_caption(out, caption)
    print(f"[fig12] wrote {out.relative_to(REPO)}")


# ==============================================================================
# Composite main-body figures for the 9-page NeurIPS submission
# Each composes a tight subset of the data shown in standalone Figs 1-12.
# Standalone figures are retained for appendix use.
# ==============================================================================

def fig_main_mechanism() -> None:
    """Main Figure 1 — three-panel composite:
      (A) Paired-task substrate diagram (compact redo of fig_paired_task_diagram).
      (B) Qwen layer × position patching heatmap (Qwen panel only of fig_patching).
      (C) Steering dose-response, Qwen + CodeGemma overlay.
    """
    qwen_patch = np.load(RESULTS / "patch-20260516T005329Z" / "aggregated.npz",
                          allow_pickle=True)
    qwen_steer = np.load(RESULTS / "steer-20260516T021522Z" / "curves.npz",
                          allow_pickle=True)
    cg_candidates = sorted(RESULTS.glob("steer-codegemma_7b_it-*"),
                            key=lambda p: p.name, reverse=True)
    have_cg = bool(cg_candidates) and (cg_candidates[0] / "curves.npz").is_file()

    fig = plt.figure(figsize=(13, 3.6), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.6], wspace=0.32)

    # Panel A: substrate schematic (compact)
    axA = fig.add_subplot(gs[0, 0])
    axA.set_xlim(0, 10); axA.set_ylim(0, 6); axA.axis("off")
    axA.set_title("A. Paired buggy/fixed substrate", fontsize=11, loc="left")
    axA.add_patch(mpatches.Rectangle((0.2, 3.2), 4.4, 2.4, fc="#fbe9e7", ec="#c62828", lw=1.2))
    axA.text(2.4, 4.85, "BUGGY", ha="center", fontsize=10, fontweight="bold", color="#c62828")
    axA.text(2.4, 4.3, "parser.py", ha="center", fontsize=8.5, family="monospace", color="#444")
    axA.text(2.4, 3.7, "tests: FAILED", ha="center", fontsize=8, color="#c62828")
    axA.add_patch(mpatches.Rectangle((0.2, 0.4), 4.4, 2.4, fc="#e8f5e9", ec="#2e7d32", lw=1.2))
    axA.text(2.4, 2.1, "FIXED", ha="center", fontsize=10, fontweight="bold", color="#2e7d32")
    axA.text(2.4, 1.55, "parser.py", ha="center", fontsize=8.5, family="monospace", color="#444")
    axA.text(2.4, 0.95, "tests: PASSED", ha="center", fontsize=8, color="#2e7d32")
    # Model box: widened to (7.0, w=3.0). Action vocab reformatted from
    # 2 long lines ("edit | noop | grep" was 18 chars at fontsize 7.5 and
    # overflowed even after widening) into 3 short lines of ≤11 chars
    # each at fontsize 7.5, so the text fits cleanly inside the box at
    # every render resolution.
    axA.annotate("", xy=(7.0, 3.0), xytext=(4.8, 3.0),
                  arrowprops=dict(arrowstyle="->", lw=1.2, color="#555"))
    axA.text(5.9, 3.3, "agent\nprompt", ha="center", fontsize=8.5, color="#555")
    axA.add_patch(mpatches.Rectangle((7.0, 1.6), 3.0, 2.8, fc="#e3f2fd", ec="#1565c0", lw=1.2))
    axA.text(8.5, 3.85, "model", ha="center", fontsize=10, fontweight="bold", color="#1565c0")
    axA.text(8.5, 3.25, "Action:", ha="center", fontsize=9, family="monospace", color="#1565c0")
    axA.text(8.5, 2.45, "edit | noop\ngrep | view\ntest",
              ha="center", va="center", fontsize=7.5, family="monospace",
              color="#1565c0", linespacing=1.15)
    axA.text(8.5, 1.0, "scalar = edit − noop\nlogit margin",
              ha="center", fontsize=7.0, color="#444", style="italic")

    # Panel B: Qwen patching heatmap
    axB = fig.add_subplot(gs[0, 1])
    mean_shift = qwen_patch["mean_shift_f2b"]
    layers = qwen_patch["layer_indices"]
    positions = qwen_patch["position_offsets"]
    im = axB.imshow(mean_shift, aspect="auto", cmap="RdBu_r",
                     vmin=-1.5, vmax=1.5, origin="lower")
    axB.set_yticks(range(len(layers)))
    axB.set_yticklabels([f"L{int(l)}" for l in layers], fontsize=7)
    axB.set_xticks(range(len(positions)))
    axB.set_xticklabels([f"pos {int(p)}" for p in positions], fontsize=8)
    axB.set_title("B. Qwen patching (F→B mean Δ)", fontsize=11, loc="left")
    flat_idx = int(np.argmax(mean_shift))
    py, px = divmod(flat_idx, mean_shift.shape[1])
    axB.add_patch(mpatches.Rectangle((px - 0.5, py - 0.5), 1, 1,
                                       fill=False, ec="black", lw=2))
    axB.text(px, py, f"{mean_shift[py, px]:+.2f}", ha="center", va="center",
              fontsize=8.5, color="black", fontweight="bold")
    fig.colorbar(im, ax=axB, shrink=0.85, pad=0.04, label="Δ margin")

    # Panel C: steering dose-response
    axC = fig.add_subplot(gs[0, 2])
    alphas_q = qwen_steer["alphas"]
    conds_q = [c if isinstance(c, str) else c.decode() for c in qwen_steer["conditions"]]
    m_q = qwen_steer["margin_mean"]
    style_q = {"buggy": ("#c62828", "o", "Qwen buggy"),
                "fixed": ("#2e7d32", "s", "Qwen fixed")}
    for i, c in enumerate(conds_q):
        color, marker, label = style_q[c]
        axC.plot(alphas_q, m_q[i], color=color, marker=marker, lw=1.8,
                  label=label, markersize=5)
    if have_cg:
        cg = np.load(cg_candidates[0] / "curves.npz", allow_pickle=True)
        ac = cg["alphas"]
        cc = [c if isinstance(c, str) else c.decode() for c in cg["conditions"]]
        mc = cg["margin_mean"]
        axC2 = axC.twinx()
        style_c = {"buggy": ("#ef9a9a", "o", "CG buggy (N=20)"),
                    "fixed": ("#a5d6a7", "s", "CG fixed (N=20)")}
        for i, c in enumerate(cc):
            color, marker, label = style_c[c]
            axC2.plot(ac, mc[i], color=color, marker=marker, lw=1.2,
                       linestyle="--", label=label, markersize=4, alpha=0.85)
        axC2.tick_params(axis="y", labelcolor="#888")
        h1, l1 = axC.get_legend_handles_labels()
        h2, l2 = axC2.get_legend_handles_labels()
        axC.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7, frameon=False)
    else:
        axC.legend(loc="upper right", fontsize=8, frameon=False)
    axC.axvline(0, color="#999", lw=0.6, linestyle=":")
    axC.set_xlabel("steering α  (units of ‖v_noop‖)", fontsize=9)
    axC.set_ylabel("Qwen mean (edit−noop) margin", fontsize=9)
    axC.set_title("C. Rank-1 steering dose-response", fontsize=11, loc="left")
    axC.spines["top"].set_visible(False)

    out = FIG_DIR / "main_mechanism.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    save_caption(out,
        "Figure 1. (A) Paired buggy/fixed substrate: each task carries identical "
        "issue text but a buggy vs fixed source file and FAIL vs PASS test "
        "transcript; the model's first action-token logits give the scalar "
        "edit−noop margin we localize. (B) Qwen layer × position F→B patching "
        "heatmap: a single residual substitution at L24/pos −1 recovers ~98% of "
        "the buggy/fixed margin gap on 48/49 tasks. (C) Single-direction "
        "additive steering at the patching peak reproduces the full behavioural "
        "gap as a smooth monotonic dose-response on Qwen (solid) and "
        "CodeGemma's responsive subset (dashed)."
    )
    print(f"  wrote {out.name}")


def fig_main_monitor() -> None:
    """Main Figure 2 — three-model ROC + PR overlay (drop the
    projection-distribution histogram panel of the standalone Fig 8)."""
    base = RESULTS / "monitor_real"
    paths = {
        "Qwen-1.5B":     ("real_curves_qwen_n500.npz",     "#1f77b4", "o"),
        "CodeGemma-7B":  ("real_curves_codegemma_n500.npz", "#ff7f0e", "s"),
        "DeepSeek-1.3B": ("real_curves_deepseek_n500.npz",  "#2ca02c", "^"),
    }
    series = []
    for label, (fname, color, marker) in paths.items():
        d = _load_real_curves(base / fname)
        if d is not None:
            series.append((label, d, color, marker))
    if not series:
        print("  SKIP main_monitor.png (no real-curve npz under results/monitor_real)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300,
                              gridspec_kw={"wspace": 0.28})

    axL = axes[0]
    for label, d, color, marker in series:
        axL.plot(d["fpr"], d["tpr"], color=color, lw=2,
                  label=f"{label} (N={d['n_pairs']})  AUC = {d['roc_auc']:.3f}")
        axL.scatter([d["op_fpr"]], [d["op_tpr"]], color=color, s=55,
                     edgecolor="black", linewidth=0.7, zorder=5, marker=marker)
    axL.plot([0, 1], [0, 1], color="#bbb", lw=0.7, linestyle=":")
    axL.set_xlim(-0.02, 1.02); axL.set_ylim(-0.02, 1.02)
    axL.set_xlabel("false-positive rate")
    axL.set_ylabel("true-positive rate")
    axL.set_title("ROC — real SWE-bench Verified", fontsize=11, loc="left")
    axL.legend(loc="lower right", fontsize=8.5, frameon=False)
    axL.spines["top"].set_visible(False); axL.spines["right"].set_visible(False)

    axR = axes[1]
    for label, d, color, marker in series:
        axR.plot(d["pr_r"], d["pr_p"], color=color, lw=2,
                  label=f"{label}  AP = {d['pr_auc']:.3f}")
        axR.scatter([d["op_recall"]], [d["op_precision"]], color=color, s=55,
                     edgecolor="black", linewidth=0.7, zorder=5, marker=marker)
    axR.set_xlim(-0.02, 1.02); axR.set_ylim(-0.02, 1.02)
    axR.set_xlabel("recall"); axR.set_ylabel("precision")
    axR.set_title("Precision–recall — real tasks", fontsize=11, loc="left")
    axR.legend(loc="lower left", fontsize=8.5, frameon=False)
    axR.spines["top"].set_visible(False); axR.spines["right"].set_visible(False)

    out = FIG_DIR / "main_monitor.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    save_caption(out,
        "Figure 2. Pre-edit monitor on the full SWE-bench Verified benchmark, "
        "three-model overlay. Each model uses its own frozen v_noop derived "
        "from the 49 toy tasks (Qwen at L24/pos −1, CodeGemma at L26/pos −1, "
        "DeepSeek at L22/pos −1) and never retrained. Filled markers mark the "
        "balanced-accuracy operating point. The monitor transfers across all "
        "three families with descending strength; false-edit rates at the "
        "operating point are 2.6% / 15.5% / 19.6%, indicating model-class-"
        "specific threshold calibration is needed for deployment (Appendix J)."
    )
    print(f"  wrote {out.name}")


def fig_main_sae() -> None:
    """Main Figure 3 — two-panel composite:
      (A) Cumulative top-k OMP ablation curve on Qwen (the 3-feature plateau).
      (B) Argmax-action distribution under clean vs OMP top-8 ablation,
          buggy and fixed side by side (the 80% grep→edit flip)."""
    ablation_path = RESULTS / "sae" / "ablate-qwen-topk-interp" / "ablation_results.json"
    action_path = RESULTS / "sae" / "ablate-qwen-topk-interp" / "action_distribution.json"
    if not ablation_path.exists() or not action_path.exists():
        print(f"  SKIP main_sae.png (missing {ablation_path.name} or {action_path.name})")
        return

    import re
    blob = json.loads(ablation_path.read_text())
    pairs: dict[str, dict] = {}
    for r in blob["rows"]:
        pairs.setdefault(r["task_id"], {})[r["condition"]] = r
    complete = [p for p in pairs.values() if "buggy" in p and "fixed" in p]
    n = len(complete)
    def gap(label: str):
        return np.array([p["buggy"]["margins"][label] - p["fixed"]["margins"][label]
                          for p in complete])
    clean = gap("clean"); clean_mu = float(clean.mean())
    labels_topk = sorted(
        [lab for lab in blob["rows"][0]["margins"].keys() if lab.startswith("ablate_omp_top")],
        key=lambda s: int(re.findall(r"\d+", s)[0]),
    )
    ks = [int(re.findall(r"\d+", s)[0]) for s in labels_topk]
    rng = np.random.default_rng(0); B = 10_000
    mean_pct = []; ci_lo = []; ci_hi = []
    for lab in labels_topk:
        ab = gap(lab); diffs = clean - ab
        mean_pct.append(diffs.mean() / clean_mu * 100)
        boot = np.array([diffs[rng.integers(0, n, size=n)].mean() for _ in range(B)])
        boot_pct = boot / clean_mu * 100
        ci_lo.append(float(np.quantile(boot_pct, 0.025)))
        ci_hi.append(float(np.quantile(boot_pct, 0.975)))
    mean_pct = np.array(mean_pct); ci_lo = np.array(ci_lo); ci_hi = np.array(ci_hi)

    actdist = json.loads(action_path.read_text())
    action_names = actdist["action_names"]
    by_cond = actdist["by_condition_argmax_dist"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=300,
                                     gridspec_kw={"width_ratios": [1.2, 1.4], "wspace": 0.28})

    axL.fill_between(ks, ci_lo, ci_hi, color="#1f77b4", alpha=0.2, label="95% bootstrap CI")
    axL.plot(ks, mean_pct, "-o", color="#1f77b4", lw=2, markersize=6,
              label="cumulative OMP top-k")
    if 3 in ks:
        i3 = ks.index(3)
        axL.scatter([3], [mean_pct[i3]], s=140, color="#d62728", zorder=6,
                     marker="*", label=f"min sufficient (k=3, {mean_pct[i3]:+.1f}%)")
    axL.axhline(0, color="black", lw=0.4)
    axL.set_xscale("log")
    axL.set_xticks(ks); axL.set_xticklabels([str(k) for k in ks], fontsize=8)
    axL.set_xlabel("OMP cumulative top-k features ablated")
    axL.set_ylabel("buggy−fixed gap reduction  [%]")
    axL.set_title(f"A. Behavioural-sparse: k=3 sufficient (Qwen, N={n})",
                   fontsize=11, loc="left")
    axL.legend(loc="lower right", fontsize=8.5)
    axL.grid(True, alpha=0.25)

    cmp_label = "ablate_omp_top8"
    bug_cl = by_cond["buggy"]["clean"]; bug_ab = by_cond["buggy"][cmp_label]
    fix_cl = by_cond["fixed"]["clean"]; fix_ab = by_cond["fixed"][cmp_label]
    def pct(d, name):
        tot = sum(d.values()) or 1
        return 100.0 * d.get(name, 0) / tot
    x = np.arange(len(action_names))
    w = 0.20
    axR.bar(x - 1.5 * w, [pct(bug_cl, a) for a in action_names], w,
             color="#c62828", alpha=0.85, label="buggy / clean")
    axR.bar(x - 0.5 * w, [pct(bug_ab, a) for a in action_names], w,
             color="#c62828", alpha=0.5, hatch="//", label="buggy / OMP top-8 abl.")
    axR.bar(x + 0.5 * w, [pct(fix_cl, a) for a in action_names], w,
             color="#2e7d32", alpha=0.85, label="fixed / clean")
    axR.bar(x + 1.5 * w, [pct(fix_ab, a) for a in action_names], w,
             color="#2e7d32", alpha=0.5, hatch="//", label="fixed / OMP top-8 abl.")
    axR.set_xticks(x); axR.set_xticklabels(action_names, fontsize=9)
    axR.set_ylabel("argmax-action share  [%]")
    axR.set_ylim(0, 100)
    flip_pct = float(actdist["flip_rate_overall"]) * 100
    axR.set_title(f"B. Argmax flip: 80% grep→edit under ablation\n"
                   f"(flip rate {flip_pct:.1f}%; clean grep ≈84% → ablated edit ≥94%)",
                   fontsize=10.5, loc="left")
    axR.legend(loc="upper right", fontsize=7.5, frameon=False)
    axR.spines["top"].set_visible(False)
    axR.grid(True, axis="y", alpha=0.25)

    out = FIG_DIR / "main_sae.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    save_caption(out,
        f"Figure 3. SAE decomposition of v_noop on Qwen "
        f"(d_sae=4096, k=16, EV=0.976). (A) Cumulative OMP top-k ablation: "
        f"top-1 alone is ineffective, top-2 reaches +26.4%, top-3 hits the "
        f"+34% plateau; the minimum sufficient subset is 3 features, not 8. "
        f"(B) Argmax-action distribution under clean vs OMP top-8 ablation. "
        f"The model's clean default is `grep` (≈84%); ablating the 8 features "
        f"flips the argmax on {flip_pct:.1f}% of prompts and every flip is "
        f"`grep→edit`, collapsing the action distribution onto `edit` on both "
        f"buggy and fixed prompts. This is behavioural override, not "
        f"calibration. The same procedure on CodeGemma yields +6.4% reduction "
        f"(p=0.10, not significant) — see Appendix H."
    )
    print(f"  wrote {out.name}")


def fig_sample_efficiency() -> None:
    """Sample-efficiency curve: AUC vs number of toy training tasks.

    Reads results/monitor_real/sample_efficiency.json (per-N per-seed
    AUC) and renders a single panel with mean ± std error bars, plus
    the deterministic full-49-task baseline as a horizontal reference.
    """
    src = REPO / "results" / "monitor_real" / "sample_efficiency.json"
    if not src.is_file():
        print(f"  SKIP sample_efficiency.png ({src.relative_to(REPO)} missing)")
        return
    data = json.loads(src.read_text())
    summary = data["summary"]
    # Order numerically; pull mean ± std for each N
    Ns = sorted(int(k) for k in summary.keys())
    means = np.array([summary[str(n)]["auc_mean"] for n in Ns])
    stds = np.array([summary[str(n)]["auc_std"] for n in Ns])
    mins = np.array([summary[str(n)]["auc_min"] for n in Ns])
    maxs = np.array([summary[str(n)]["auc_max"] for n in Ns])
    full_n = max(Ns)
    full_auc = summary[str(full_n)]["auc_mean"]

    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=300)
    ax.errorbar(Ns, means, yerr=stds, fmt="o-", color="#1f77b4",
                linewidth=1.8, markersize=6, capsize=4,
                ecolor="#1f77b4", label="mean ± 1 std (10 random subsamples per N)")
    # Min-max range as light band
    ax.fill_between(Ns, mins, maxs, color="#1f77b4", alpha=0.12,
                    label=f"min–max across the {data['config']['n_seeds']} subsamples")
    # Full-set baseline
    ax.axhline(full_auc, color="#888", linestyle=":", linewidth=1.0,
               label=f"full-{full_n}-task baseline (AUC {full_auc:.4f})")

    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("number of toy training tasks  (paired buggy/fixed)")
    ax.set_ylabel("SWE-bench Verified ROC-AUC")
    ax.set_ylim(0.83, 1.005)
    ax.set_title("Sample efficiency of v_noop: full saturation by N ≈ 10",
                 fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = FIG_DIR / "sample_efficiency.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    save_caption(out,
        "Sample efficiency of v_noop on Qwen2.5-Coder-1.5B. AUC on the "
        "full 500-instance SWE-bench Verified evaluation set, as a "
        "function of the number of paired buggy/fixed toy tasks used to "
        "derive v_noop. Mean ± 1 std across 10 random subsamples per N; "
        "the full N=49 set is deterministic. The curve saturates at "
        "N ≈ 10 (mean AUC 0.988 vs full-set 0.989); even N=1 mean AUC "
        "is 0.954.")
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    sys.exit(main())
