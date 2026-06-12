#!/usr/bin/env python3
"""Analyze agent-loop pilots: over-editing, recall, evidence-gathering, and the
monitor edit-veto tradeoff. Reads results/agentloop/*_loops*.json."""

from __future__ import annotations

import glob
import json

import numpy as np


def auc(pos, neg) -> float:
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def main() -> int:
    for f in sorted(glob.glob("results/agentloop/*_loops*.json")):
        d = json.load(open(f))
        R = d["results"]
        model = d["model"].split("/")[-1]
        mode = "evidence-present" if f.endswith("_ev.json") else "from-issue"
        fixed = [r for r in R if r["condition"] == "fixed"]
        buggy = [r for r in R if r["condition"] == "buggy"]
        er = lambda rs: 100 * np.mean([r["terminal_action"] == "edit" for r in rs]) if rs else float("nan")
        tr = lambda rs: 100 * np.mean([r["tested"] for r in rs]) if rs else float("nan")

        print(f"\n### {model} · {mode}  (N={len(fixed)} passing / {len(buggy)} buggy)")
        print(f"  over-editing (passing→edit): {er(fixed):5.1f}%    "
              f"correct-edit (buggy→edit): {er(buggy):5.1f}%")
        print(f"  tested-before-deciding: passing {tr(fixed):.0f}%  buggy {tr(buggy):.0f}%")

        # veto: over-edits = passing-condition edits (block these); correct = buggy edits (keep these)
        fe = [r["decision_proj"] for r in fixed if r["terminal_action"] == "edit" and r["decision_proj"] is not None]
        be = [r["decision_proj"] for r in buggy if r["terminal_action"] == "edit" and r["decision_proj"] is not None]
        if not fe or not be:
            print("  (veto: not enough edits with a decision projection)")
            continue
        a = auc(fe, be)  # passing-edit proj should be HIGH (v=fixed-buggy), buggy-edit LOW
        print(f"  veto signal: decision-proj separates passing-edit vs buggy-edit "
              f"AUC={a:.3f}  (passing μ={np.mean(fe):+.2f}, buggy μ={np.mean(be):+.2f})")
        fe_a, be_a = np.asarray(fe), np.asarray(be)
        best = (-2.0, 0.0, 0.0, 0.0)
        for thr in np.unique(np.concatenate([fe_a, be_a])):
            blocked = float((fe_a > thr).mean())      # over-edits correctly vetoed
            preserved = float((be_a <= thr).mean())   # correct edits not vetoed
            j = blocked + preserved - 1
            if j > best[0]:
                best = (j, float(thr), blocked, preserved)
        _, thr, blk, pre = best
        oe0, rec0 = er(fixed), er(buggy)
        print(f"  veto @Youden thr={thr:+.2f}: over-edits BLOCKED {100*blk:.0f}% · "
              f"correct-edits PRESERVED {100*pre:.0f}%")
        print(f"    => over-editing {oe0:.0f}% → {oe0*(1-blk):.0f}%   |   "
              f"correct-edit recall {rec0:.0f}% → {rec0*pre:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
