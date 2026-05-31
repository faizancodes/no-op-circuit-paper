#!/usr/bin/env python3
"""Analyze the Phase-2 action-order / binary / abstract-label control runs.

Reads results/action_order_control/<tag>_<experiment>_scores.json (produced by
modal_app/action_order_control.py), writes *_summary.json and a figure per
experiment. Pure local (numpy + matplotlib).

Usage:
    python scripts/analyze_action_order_control.py --tag qwen
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path("results/action_order_control")
FIG = Path("figures")


def _boot_ci(xs, n=10000, seed=0):
    if not xs:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    xs = np.asarray(xs, float)
    bs = rng.choice(xs, size=(n, len(xs)), replace=True).mean(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def analyze_action_order(tag: str):
    rows = json.load(open(RES / f"{tag}_action_order_scores.json"))["rows"]
    by_pos = defaultdict(list)          # noop_pos -> [is_noop_argmax]
    by_pos_cond = defaultdict(list)     # (pos,cond) -> [is_noop]
    margin_pos_cond = defaultdict(list)  # (pos,cond) -> [edit-noop]
    rank_pos = defaultdict(list)
    argmax_by_pos = defaultdict(Counter)
    for r in rows:
        p, c = r["noop_pos"], r["condition"]
        is_noop = int(r["argmax"] == "noop")
        by_pos[p].append(is_noop)
        by_pos_cond[(p, c)].append(is_noop)
        margin_pos_cond[(p, c)].append(r["action_logits"]["edit"] - r["action_logits"]["noop"])
        rank_pos[p].append(r["ranks"]["noop"])
        argmax_by_pos[p][r["argmax"]] += 1
    summary = {
        "n_rows": len(rows),
        "all_single_token": all(r["all_single_token"] for r in rows),
        "overall_noop_argmax_rate": sum(r["argmax"] == "noop" for r in rows) / len(rows),
        "noop_argmax_rate_by_position": {p: float(np.mean(by_pos[p])) for p in sorted(by_pos)},
        "noop_argmax_rate_by_position_ci": {p: _boot_ci(by_pos[p]) for p in sorted(by_pos)},
        "noop_rate_fixed_by_position": {
            p: float(np.mean(by_pos_cond[(p, "fixed")])) for p in sorted(by_pos)
        },
        "edit_minus_noop_margin_by_position_condition": {
            f"{p}_{c}": float(np.mean(margin_pos_cond[(p, c)]))
            for p in sorted(by_pos) for c in ("buggy", "fixed")
        },
        "noop_mean_rank_by_position": {p: float(np.mean(rank_pos[p])) for p in sorted(by_pos)},
        "argmax_distribution_by_position": {p: dict(argmax_by_pos[p]) for p in sorted(by_pos)},
        "max_noop_rate_any_position": max(float(np.mean(by_pos[p])) for p in by_pos),
    }
    (RES / f"{tag}_action_order_summary.json").write_text(json.dumps(summary, indent=2))

    # Figure: noop argmax rate by position (+ edit-noop margin by position/cond).
    positions = sorted(by_pos)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].bar(positions, [summary["noop_argmax_rate_by_position"][p] for p in positions], color="#444")
    ax[0].set_xlabel("noop position in action menu (0 = first)")
    ax[0].set_ylabel("noop argmax rate")
    ax[0].set_ylim(0, 1)
    ax[0].set_title(f"{tag}: noop never selected at any position")
    for c, col in (("buggy", "#c44"), ("fixed", "#48c")):
        ax[1].plot(positions, [np.mean(margin_pos_cond[(p, c)]) for p in positions], "o-", color=col, label=c)
    ax[1].set_xlabel("noop position in action menu (0 = first)")
    ax[1].set_ylabel("mean (edit − noop) logit")
    ax[1].set_title("edit − noop margin by position")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / f"action_order_control_{tag}.png", dpi=150)
    plt.close(fig)
    return summary


def analyze_binary(tag: str):
    rows = json.load(open(RES / f"{tag}_binary_scores.json"))["rows"]
    order_name = {0: "{edit,noop}", 1: "{noop,edit}"}
    rate, margin = defaultdict(list), defaultdict(list)
    for r in rows:
        k = (order_name[r["order_id"]], r["condition"])
        rate[k].append(int(r["argmax"] == "noop"))
        margin[k].append(r["action_logits"]["edit"] - r["action_logits"]["noop"])
    summary = {
        "n_rows": len(rows),
        "all_single_token": all(r["all_single_token"] for r in rows),
        "noop_argmax_rate": {f"{o}|{c}": float(np.mean(v)) for (o, c), v in rate.items()},
        "noop_rate_ci": {f"{o}|{c}": _boot_ci(v) for (o, c), v in rate.items()},
        "edit_minus_noop_margin": {f"{o}|{c}": float(np.mean(v)) for (o, c), v in margin.items()},
        "overall_noop_argmax_rate": sum(r["argmax"] == "noop" for r in rows) / len(rows),
    }
    (RES / f"{tag}_binary_summary.json").write_text(json.dumps(summary, indent=2))
    keys = sorted(rate)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar([f"{o}\n{c}" for o, c in keys], [np.mean(rate[k]) for k in keys], color="#444")
    ax.set_ylabel("noop argmax rate")
    ax.set_ylim(0, 1)
    ax.set_title(f"{tag}: binary edit/noop — abstention rate")
    fig.tight_layout()
    fig.savefig(FIG / f"binary_edit_noop_control_{tag}.png", dpi=150)
    plt.close(fig)
    return summary


def analyze_abstract_label(tag: str):
    rows = json.load(open(RES / f"{tag}_abstract_label_scores.json"))["rows"]
    by_pos = defaultdict(list)
    by_pos_cond = defaultdict(list)
    for r in rows:
        decoded = r["mapping"][r["argmax"]] if r.get("mapping") else None
        is_abstain = int(decoded == "noop")
        by_pos[r["noop_pos"]].append(is_abstain)
        by_pos_cond[(r["noop_pos"], r["condition"])].append(is_abstain)
    summary = {
        "n_rows": len(rows),
        "all_single_token": all(r["all_single_token"] for r in rows),
        "overall_abstain_rate": float(np.mean([x for v in by_pos.values() for x in v])),
        "abstain_rate_by_noop_label_position": {p: float(np.mean(by_pos[p])) for p in sorted(by_pos)},
        "abstain_rate_ci": {p: _boot_ci(by_pos[p]) for p in sorted(by_pos)},
        "max_abstain_rate_any_position": max(float(np.mean(by_pos[p])) for p in by_pos),
    }
    (RES / f"{tag}_abstract_label_summary.json").write_text(json.dumps(summary, indent=2))
    positions = sorted(by_pos)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(positions, [summary["abstain_rate_by_noop_label_position"][p] for p in positions], color="#444")
    ax.set_xlabel("noop's label position (A=0 ... E=4)")
    ax.set_ylabel("abstention rate (argmax letter maps to noop)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{tag}: abstract-label control")
    fig.tight_layout()
    fig.savefig(FIG / f"abstract_label_control_{tag}.png", dpi=150)
    plt.close(fig)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen")
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    done = {}
    for name, fn in (
        ("action_order", analyze_action_order),
        ("binary", analyze_binary),
        ("abstract_label", analyze_abstract_label),
    ):
        path = RES / f"{args.tag}_{name}_scores.json"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        done[name] = fn(args.tag)
        print(f"[ok] {name}: {json.dumps(done[name], indent=2)[:600]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
