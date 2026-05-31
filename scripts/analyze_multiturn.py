#!/usr/bin/env python3
"""Analyse the temporally-separated transcript experiment (App. G.16).

Reads the JSON produced by `modal_app/multiturn_experiment.py` and reports,
per multi-turn variant:
  - v_noop projection ROC-AUC (positive class = buggy; score = -projection,
    matching the §5.1 sign convention)
  - turn-local regex AUC      (predict buggy iff "FAILED" in decision-local context)
  - full-scrollback regex AUC (predict buggy iff "FAILED" anywhere in prompt)
  - mean projection (buggy vs fixed) and argmax-action distribution

The headline comparison (with existing reference points):
  transcript ADJACENT  (§5.1)            : projection AUC 0.989
  transcript DISTANT   (stale multiturn) : projection AUC = ?   <- this experiment
  transcript ABSENT    (§G.9 1-turn)     : projection AUC ≤ 0.52
  multiturn no-transcript (format control): projection AUC = ?  <- should be chance

Usage:
    .venv/bin/python scripts/analyze_multiturn.py results/monitor_real/multiturn_experiment_*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _auc(labels, scores) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    p = argparse.ArgumentParser()
    p.add_argument("json", nargs="?", default="",
                   help="path to multiturn_experiment_*.json (default: latest)")
    args = p.parse_args(argv)

    path = args.json
    if not path:
        cands = sorted(glob.glob("results/monitor_real/multiturn_experiment_*.json"))
        if not cands:
            print("no multiturn_experiment_*.json found", file=sys.stderr)
            return 1
        path = cands[-1]
    blob = json.loads(Path(path).read_text())
    rows = blob["rows"]
    print(f"loaded {len(rows)} rows from {path}")
    print(f"config: {blob['config']}")
    print()

    by_variant = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)

    for variant in sorted(by_variant):
        rs = by_variant[variant]
        # pair by task_id
        by_task = defaultdict(dict)
        for r in rs:
            by_task[r["task_id"]][r["condition"]] = r
        paired = [t for t, s in by_task.items() if "buggy" in s and "fixed" in s]
        n = len(paired)

        proj_b = np.array([by_task[t]["buggy"]["projection"] for t in paired])
        proj_f = np.array([by_task[t]["fixed"]["projection"] for t in paired])
        # positive class = buggy; score = -projection (higher = more buggy-like)
        labels = np.r_[np.ones(n, int), np.zeros(n, int)]
        proj_auc = _auc(labels, np.r_[-proj_b, -proj_f])

        # regex baselines: predict buggy iff FAILED present
        tl_b = np.array([int(by_task[t]["buggy"]["failed_in_last_user"]) for t in paired])
        tl_f = np.array([int(by_task[t]["fixed"]["failed_in_last_user"]) for t in paired])
        full_b = np.array([int(by_task[t]["buggy"]["failed_in_full"]) for t in paired])
        full_f = np.array([int(by_task[t]["fixed"]["failed_in_full"]) for t in paired])
        tl_auc = _auc(labels, np.r_[tl_b, tl_f])
        full_auc = _auc(labels, np.r_[full_b, full_f])

        argmax_b = Counter(by_task[t]["buggy"]["argmax_action"] for t in paired)
        argmax_f = Counter(by_task[t]["fixed"]["argmax_action"] for t in paired)

        print(f"=== {variant}  (N={n} paired) ===")
        print(f"  v_noop projection ROC-AUC : {proj_auc:.4f}")
        print(f"    mean proj  buggy={proj_b.mean():+.3f}  fixed={proj_f.mean():+.3f}  "
              f"gap(fixed-buggy)={proj_f.mean()-proj_b.mean():+.3f}")
        print(f"  turn-local regex AUC      : {tl_auc:.4f}   "
              f"(FAILED in decision-local: buggy {tl_b.sum()}/{n}, fixed {tl_f.sum()}/{n})")
        print(f"  full-scrollback regex AUC : {full_auc:.4f}   "
              f"(FAILED anywhere: buggy {full_b.sum()}/{n}, fixed {full_f.sum()}/{n})")
        print(f"  argmax buggy : {dict(argmax_b)}")
        print(f"  argmax fixed : {dict(argmax_f)}")
        print()

    print("Reference points (existing):")
    print("  transcript ADJACENT (§5.1)         : projection AUC 0.989")
    print("  transcript ABSENT   (§G.9 1-turn)  : projection AUC ≤ 0.52")
    return 0


if __name__ == "__main__":
    sys.exit(main())
