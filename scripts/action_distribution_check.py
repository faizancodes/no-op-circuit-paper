#!/usr/bin/env python3
"""Argmax-action shift under OMP top-8 SAE ablation (Qwen).

Reads results/sae/ablate-qwen-topk-interp/ablation_results.json (Tier-3
free-wins Task 5 output — includes per-action logits per condition) and
computes:

  1. Per-condition argmax-action distribution under (a) clean and (b)
     OMP top-8 ablation. 5 actions × {buggy, fixed}.
  2. Per-task argmax-flip rate: % of (task, condition) pairs whose
     argmax(clean) differs from argmax(ablate_omp_top8). Broken down
     by directed flip (e.g. grep → edit, edit → noop).

Verdict branch:
  - flip rate < 10% AND |margin shift| > 30%  → "calibration-level effect"
  - flip rate > 25% AND flips are in the right direction → "behavioural override"
  - other → report as-is, no strong claim

Output: results/sae/ablate-qwen-topk-interp/action_distribution.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ablation",
                   type=Path,
                   default=Path("results/sae/ablate-qwen-topk-interp/ablation_results.json"))
    p.add_argument("--cmp-label", type=str, default="ablate_omp_top8",
                   help="Condition label to compare against 'clean'.")
    args = p.parse_args(argv)

    blob = json.loads(args.ablation.read_text())
    rows = blob["rows"]
    if "action_logits" not in rows[0]:
        print(f"ERROR: {args.ablation} has no action_logits column. "
              f"Re-run the ablation with the extended module (Tasks 5A/5B).",
              file=sys.stderr)
        return 2

    action_names = sorted(rows[0]["action_logits"]["clean"].keys())
    print(f"Action vocabulary: {action_names}")
    print(f"Comparing 'clean' vs '{args.cmp_label}'  ({len(rows)} rows)")

    # Per-condition argmax distributions
    by_cond: dict[str, dict[str, Counter[str]]] = {
        "buggy": {"clean": Counter(), args.cmp_label: Counter()},
        "fixed": {"clean": Counter(), args.cmp_label: Counter()},
    }
    flip_counter: Counter[tuple[str, str, str]] = Counter()  # (cond, from_action, to_action)
    flip_total = {"buggy": 0, "fixed": 0}
    n_total = {"buggy": 0, "fixed": 0}

    for r in rows:
        cond = r["condition"]
        if cond not in ("buggy", "fixed"):
            continue
        if args.cmp_label not in r["action_logits"]:
            print(f"WARN: row {r['task_id']} missing {args.cmp_label!r}; skipping")
            continue
        clean_logits = r["action_logits"]["clean"]
        cmp_logits = r["action_logits"][args.cmp_label]
        clean_argmax = max(clean_logits, key=lambda n: clean_logits[n])
        cmp_argmax = max(cmp_logits, key=lambda n: cmp_logits[n])
        by_cond[cond]["clean"][clean_argmax] += 1
        by_cond[cond][args.cmp_label][cmp_argmax] += 1
        n_total[cond] += 1
        if clean_argmax != cmp_argmax:
            flip_counter[(cond, clean_argmax, cmp_argmax)] += 1
            flip_total[cond] += 1

    # Per-condition flip rate
    flip_rate = {c: (flip_total[c] / max(n_total[c], 1)) for c in n_total}
    overall_n = sum(n_total.values())
    overall_flips = sum(flip_total.values())
    overall_rate = overall_flips / max(overall_n, 1)

    # Print
    print()
    print(f"=== Per-condition argmax-action distributions ===")
    print(f"{'condition':<8} {'label':<22}  " + "  ".join(f"{a:>6}" for a in action_names))
    print("-" * 80)
    for cond in ("buggy", "fixed"):
        for lab in ("clean", args.cmp_label):
            cnt = by_cond[cond][lab]; n = sum(cnt.values()) or 1
            cells = "  ".join(f"{100 * cnt.get(a, 0) / n:5.1f}%" for a in action_names)
            print(f"{cond:<8} {lab:<22}  {cells}")
    print()
    print(f"=== Argmax-flip rate ({args.cmp_label} vs clean) ===")
    print(f"  buggy: {flip_total['buggy']}/{n_total['buggy']} = {flip_rate['buggy']*100:.1f}%")
    print(f"  fixed: {flip_total['fixed']}/{n_total['fixed']} = {flip_rate['fixed']*100:.1f}%")
    print(f"  overall: {overall_flips}/{overall_n} = {overall_rate*100:.1f}%")
    print()
    if flip_counter:
        print(f"=== Top flip directions ===")
        for (cond, frm, to), c in sorted(flip_counter.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {cond:<6}  {frm:>6} → {to:<6}  n={c}")
    else:
        print("(no flips observed)")

    # Verdict
    avg_margin_shift_pct = float("nan")
    if "margins" in rows[0]:
        clean_gaps = [
            r1["buggy"]["margins"]["clean"] - r1["fixed"]["margins"]["clean"]
            for r1 in [_pair_by_task(rows, t) for t in {r["task_id"] for r in rows}]
            if r1 is not None
        ]
        ab_gaps = [
            r1["buggy"]["margins"][args.cmp_label] - r1["fixed"]["margins"][args.cmp_label]
            for r1 in [_pair_by_task(rows, t) for t in {r["task_id"] for r in rows}]
            if r1 is not None
        ]
        if clean_gaps and ab_gaps:
            import statistics
            mean_clean = statistics.mean(clean_gaps); mean_ab = statistics.mean(ab_gaps)
            avg_margin_shift_pct = (mean_clean - mean_ab) / mean_clean * 100

    if overall_rate < 0.10 and not (avg_margin_shift_pct != avg_margin_shift_pct) \
            and abs(avg_margin_shift_pct) > 30:
        verdict = ("Calibration-level effect: ablation moves the buggy-fixed margin gap "
                   f"by {avg_margin_shift_pct:+.1f}% but the model's chosen action "
                   f"flips on only {overall_rate*100:.1f}% of prompts. The eight features "
                   "shift the model's logit-level confidence in edit-vs-noop without "
                   "rewriting its overt argmax behaviour on this substrate.")
    elif overall_rate > 0.25:
        verdict = (f"Behavioural override: {overall_rate*100:.1f}% of prompts flip "
                   "argmax under OMP top-8 ablation — the eight features causally "
                   "control the model's chosen action on this substrate.")
    else:
        verdict = (f"Intermediate effect: {overall_rate*100:.1f}% argmax-flip rate, "
                   f"{avg_margin_shift_pct:+.1f}% gap shift. Neither pure calibration nor "
                   "behavioural override.")
    print()
    print(f"VERDICT: {verdict}")

    out = {
        "n_rows": len(rows),
        "n_per_condition": n_total,
        "action_names": action_names,
        "by_condition_argmax_dist": {
            c: {lab: dict(cnt) for lab, cnt in conds.items()}
            for c, conds in by_cond.items()
        },
        "flip_rate_per_condition": flip_rate,
        "flip_rate_overall": overall_rate,
        "flip_directions": [
            {"condition": c, "from": f, "to": t, "count": n}
            for (c, f, t), n in sorted(flip_counter.items(), key=lambda kv: -kv[1])
        ],
        "avg_buggy_fixed_gap_reduction_pct": avg_margin_shift_pct,
        "verdict": verdict,
    }
    out_path = args.ablation.parent / "action_distribution.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return 0


def _pair_by_task(rows, task_id):
    pair = {r["condition"]: r for r in rows if r["task_id"] == task_id}
    if "buggy" in pair and "fixed" in pair:
        return pair
    return None


if __name__ == "__main__":
    sys.exit(main())
