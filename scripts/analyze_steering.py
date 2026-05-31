#!/usr/bin/env python3
"""Aggregate a steering run into per-condition dose-response curves.

Inputs:
  results/<steer-run>/manifest.json
  results/<steer-run>/v_noop.pt     (the direction we steered along)

Outputs (printed):
  - Per-condition × alpha: mean `edit - noop` margin and argmax-action distribution
  - The implied edit-rate (% argmax=edit) as a function of alpha
  - "Crossover alpha" — where buggy and fixed curves meet (the steering coef
    that makes the two prompts indistinguishable in their action distribution)
  - All-five-action mean logits per (condition, alpha) for stacked-area plots

Writes results/<steer-run>/curves.npz containing:
  alphas, margin_mean[(condition, alpha)], action_rates[(condition, alpha, action)]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    args = p.parse_args(argv)

    import numpy as np

    manifest_path = args.run_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"missing manifest: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["rows"]
    alphas = sorted({float(r["alpha"]) for r in rows})
    conditions = sorted({r["condition"] for r in rows})
    actions = list(rows[0]["action_logits"].keys())

    # Per-(condition, alpha) collect: margin, action_logits, argmax counts
    margin: dict[tuple[str, float], list[float]] = defaultdict(list)
    logit_sums: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    argmax_counter: dict[tuple[str, float], Counter] = defaultdict(Counter)

    for r in rows:
        key = (r["condition"], float(r["alpha"]))
        margin[key].append(r["edit_minus_noop"])
        argmax_counter[key][r["argmax_action"]] += 1
        for a in actions:
            logit_sums[(r["condition"], float(r["alpha"]), a)].append(
                r["action_logits"][a]
            )

    n_per_cond = {c: len({r["task_id"] for r in rows if r["condition"] == c}) for c in conditions}

    print(f"layer={manifest['layer']} pos={manifest['position']} hook={manifest['hook_point']}")
    print(f"||v_noop|| = {manifest['direction_norm']:.3f}")
    print(f"alphas = {alphas}")
    print(f"per-condition N (tasks) = {n_per_cond}")
    print()

    # ---- Dose-response table: mean margin ----
    print("=== mean (edit − noop) margin vs alpha ===")
    print(f"  {'alpha':>6}  " + "  ".join(f"{c:>10}" for c in conditions) + "  " + f"{'gap(b−f)':>10}")
    for a in alphas:
        per_cond_means = {}
        for c in conditions:
            v = margin[(c, a)]
            per_cond_means[c] = sum(v) / len(v) if v else float("nan")
        gap = per_cond_means.get("buggy", float("nan")) - per_cond_means.get("fixed", float("nan"))
        print(
            f"  {a:+6.2f}  "
            + "  ".join(f"{per_cond_means[c]:+10.3f}" for c in conditions)
            + f"  {gap:+10.3f}"
        )

    # ---- Argmax-action rates ----
    print("\n=== argmax-action distribution (%) per (condition, alpha) ===")
    for c in conditions:
        print(f"\n  condition: {c}")
        header = "  ".join(f"{a:>6}" for a in actions)
        print(f"    alpha     {header}")
        for a in alphas:
            counter = argmax_counter[(c, a)]
            total = sum(counter.values()) or 1
            cells = "  ".join(f"{100*counter.get(act,0)/total:5.0f}%" for act in actions)
            print(f"    {a:+6.2f}    {cells}")

    # ---- Mean per-action logits ----
    print("\n=== mean per-action logits per (condition, alpha) ===")
    for c in conditions:
        print(f"\n  condition: {c}")
        header = "  ".join(f"{a:>6}" for a in actions)
        print(f"    alpha     {header}")
        for a in alphas:
            cells = []
            for act in actions:
                vals = logit_sums[(c, a, act)]
                m = sum(vals) / len(vals) if vals else float("nan")
                cells.append(f"{m:+5.2f}")
            print(f"    {a:+6.2f}    " + "  ".join(cells))

    # ---- Crossover alpha (where buggy & fixed margin curves intersect) ----
    if "buggy" in conditions and "fixed" in conditions:
        print("\n=== crossover analysis ===")
        b_means = np.array([sum(margin[("buggy", a)]) / len(margin[("buggy", a)]) for a in alphas])
        f_means = np.array([sum(margin[("fixed", a)]) / len(margin[("fixed", a)]) for a in alphas])
        gap = b_means - f_means
        clean_b = b_means[alphas.index(0.0)]
        clean_f = f_means[alphas.index(0.0)]
        print(f"  clean (alpha=0): buggy={clean_b:+.3f}  fixed={clean_f:+.3f}  gap={clean_b - clean_f:+.3f}")
        # First alpha where buggy margin drops below clean_fixed margin
        crossed = [a for a, m in zip(alphas, b_means) if m <= clean_f]
        if crossed:
            print(f"  buggy margin drops below clean fixed at alpha = {min(crossed):+.2f}")
        crossed = [a for a, m in zip(alphas, f_means) if m >= clean_b]
        if crossed:
            print(f"  fixed margin rises above clean buggy at alpha = {min(crossed):+.2f}")

    # ---- Save aggregated arrays for plotting ----
    margin_mean = np.zeros((len(conditions), len(alphas)))
    action_rates = np.zeros((len(conditions), len(alphas), len(actions)))
    logit_means = np.zeros((len(conditions), len(alphas), len(actions)))
    for i, c in enumerate(conditions):
        for j, a in enumerate(alphas):
            vals = margin[(c, a)]
            margin_mean[i, j] = sum(vals) / len(vals) if vals else float("nan")
            counter = argmax_counter[(c, a)]
            total = sum(counter.values()) or 1
            for k, act in enumerate(actions):
                action_rates[i, j, k] = counter.get(act, 0) / total
                lvals = logit_sums[(c, a, act)]
                logit_means[i, j, k] = sum(lvals) / len(lvals) if lvals else float("nan")
    out = args.run_dir / "curves.npz"
    np.savez(
        str(out),
        alphas=np.asarray(alphas),
        conditions=np.asarray(conditions),
        actions=np.asarray(actions),
        margin_mean=margin_mean,
        action_rates=action_rates,
        logit_means=logit_means,
        direction_norm=np.asarray(manifest["direction_norm"]),
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
