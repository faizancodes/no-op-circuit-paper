#!/usr/bin/env python3
"""Phase 3: five-action logit decomposition of the u_tx direction (Qwen).

Reuses the existing steering sweep (results/steer-*/), which already logs all
five action logits per (task, condition, alpha) at Qwen L24/pos-1 on the
code_tests contrast. No GPU. Answers: as we move along u_tx, does noop ever
become competitive, or does grep stay dominant and only the edit-noop submargin
move?

Usage:
    python scripts/analyze_five_action_decomp.py --steer results/steer-20260516T021522Z
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("results/five_action_decomp")
FIG = Path("figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer", default="results/steer-20260516T021522Z")
    args = ap.parse_args()
    steer = Path(args.steer)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)

    z = np.load(steer / "curves.npz", allow_pickle=True)
    man = json.load(open(steer / "manifest.json"))
    alphas = [float(a) for a in z["alphas"]]
    conds = [str(c) for c in z["conditions"]]
    actions = [str(a) for a in z["actions"]]
    rates = z["action_rates"]   # (cond, alpha, action) argmax fraction
    logits = z["logit_means"]   # (cond, alpha, action) mean logit
    margin = z["margin_mean"]   # (cond, alpha) edit-noop
    ai = {a: i for i, a in enumerate(actions)}
    i0 = int(np.argmin(np.abs(np.array(alphas))))  # alpha = 0

    # Per-alpha argmax action = action with max mean logit (and rate).
    def margins_at(ci, k):
        L = logits[ci, k]
        return {
            "edit_minus_noop": float(L[ai["edit"]] - L[ai["noop"]]),
            "grep_minus_noop": float(L[ai["grep"]] - L[ai["noop"]]),
            "view_minus_noop": float(L[ai["view"]] - L[ai["noop"]]),
            "test_minus_noop": float(L[ai["test"]] - L[ai["noop"]]),
            "edit_minus_grep": float(L[ai["edit"]] - L[ai["grep"]]),
        }

    summary = {
        "source": str(steer),
        "model": man.get("model_name"),
        "layer": man.get("layer"),
        "position": man.get("position"),
        "n_prompts": man.get("n_prompts"),
        "alphas": alphas,
        "clean_alpha0": {
            c: {
                "logits": {a: float(logits[ci, i0, ai[a]]) for a in actions},
                "argmax_rates": {a: float(rates[ci, i0, ai[a]]) for a in actions},
                "margins": margins_at(ci, i0),
                "noop_logit_rank": int(
                    1 + sum(logits[ci, i0, ai[a]] > logits[ci, i0, ai["noop"]] for a in actions)
                ),
            }
            for ci, c in enumerate(conds)
        },
        # Across the WHOLE sweep:
        "max_noop_argmax_rate_any_alpha": float(rates[:, :, ai["noop"]].max()),
        "grep_argmax_rate_by_alpha": {
            c: [float(rates[ci, k, ai["grep"]]) for k in range(len(alphas))]
            for ci, c in enumerate(conds)
        },
        "noop_argmax_rate_by_alpha": {
            c: [float(rates[ci, k, ai["noop"]]) for k in range(len(alphas))]
            for ci, c in enumerate(conds)
        },
        "edit_minus_noop_margin_by_alpha": {
            c: [float(margin[ci, k]) for k in range(len(alphas))] for ci, c in enumerate(conds)
        },
        "dominant_action_by_alpha": {
            c: [actions[int(np.argmax(logits[ci, k]))] for k in range(len(alphas))]
            for ci, c in enumerate(conds)
        },
    }
    (OUT / "qwen_five_action_summary.json").write_text(json.dumps(summary, indent=2))
    # Also dump the per-alpha logit/rate tables as the "steering_logits" artifact.
    (OUT / "qwen_steering_logits.json").write_text(json.dumps({
        "alphas": alphas, "conditions": conds, "actions": actions,
        "logit_means": logits.tolist(), "action_rates": rates.tolist(),
        "edit_minus_noop_margin": margin.tolist(),
    }, indent=2))

    # Figure: (left) 5-action argmax rates vs alpha (buggy); (right) mean logits vs alpha (buggy).
    ci_b = conds.index("buggy")
    colors = {"view": "#888", "grep": "#1f77b4", "test": "#2ca02c", "edit": "#d62728", "noop": "#9467bd"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a in actions:
        ax[0].plot(alphas, rates[ci_b, :, ai[a]], "o-", color=colors[a], label=a)
    ax[0].set_xlabel(r"steering $\alpha$ along $u_{tx}$"); ax[0].set_ylabel("argmax rate (buggy)")
    ax[0].set_ylim(-0.02, 1.02); ax[0].set_title("Action argmax rate vs steering"); ax[0].legend(fontsize=8)
    for a in actions:
        ax[1].plot(alphas, logits[ci_b, :, ai[a]], "o-", color=colors[a], label=a)
    ax[1].set_xlabel(r"steering $\alpha$ along $u_{tx}$"); ax[1].set_ylabel("mean logit (buggy)")
    ax[1].set_title("Five-action logits vs steering"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "five_action_decomp_qwen.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2)[:1600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
