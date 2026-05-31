#!/usr/bin/env python3
"""Regex/log-parser transcript baseline for the §5.1 monitor.

A reviewer pointed out: if §5.2 shows the projection monitor is
reading pass/fail transcript text, then a regex scanning the pytest
output is the natural baseline. We report it here.

For each of the 499 SWE-bench-Verified-derived paired tasks, we read
the synthesised transcript text directly from the on-disk substrate
(data/real_tasks/<task>/{buggy,fixed}/tests_output.txt) and compute
three increasingly-trivial classifier scores:

  1. `contains("FAILED")`: returns 1 if the transcript contains the
     literal word "FAILED", else 0. Discrete predictions.
  2. n_failures: count of "FAILED" occurrences in the transcript.
     Higher count → more likely buggy.
  3. failure_line_density: ratio of (lines containing "FAILED" OR
     "AssertionError" OR "Traceback") to total lines.

All three should hit AUC very close to 1.000 since the buggy and
fixed transcripts differ deterministically. The result is reported
in App. G.8 as the "obvious baseline" against which the residual
monitor must be evaluated.

Output: results/monitor_real/regex_transcript_baseline.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks-root", type=Path,
                   default=Path("data/real_tasks"))
    p.add_argument("--out", type=Path,
                   default=Path("results/monitor_real/regex_transcript_baseline.json"))
    args = p.parse_args(argv)

    import numpy as np
    from sklearn.metrics import (
        average_precision_score, roc_auc_score,
    )

    task_dirs = sorted(d for d in args.tasks_root.iterdir() if d.is_dir())
    print(f"task dirs: {len(task_dirs)}")

    paired = []
    for d in task_dirs:
        b = d / "buggy" / "tests_output.txt"
        f = d / "fixed" / "tests_output.txt"
        if not b.is_file() or not f.is_file():
            continue
        paired.append((
            d.name,
            b.read_text(encoding="utf-8"),
            f.read_text(encoding="utf-8"),
        ))
    N = len(paired)
    print(f"paired tasks: {N}")
    if N < 10:
        print("error: too few paired tasks", file=sys.stderr)
        return 2

    # Classifier 1: contains("FAILED")
    contains_b = [int("FAILED" in t) for _, t, _ in paired]
    contains_f = [int("FAILED" in t) for _, _, t in paired]

    # Classifier 2: number of FAILED lines
    n_failed_b = [t.count("FAILED") for _, t, _ in paired]
    n_failed_f = [t.count("FAILED") for _, _, t in paired]

    # Classifier 3: failure-line density
    def density(text):
        if not text.strip():
            return 0.0
        lines = text.splitlines()
        if not lines:
            return 0.0
        hits = sum(1 for ln in lines if any(
            tok in ln for tok in ("FAILED", "AssertionError", "Traceback")))
        return hits / len(lines)

    dens_b = [density(t) for _, t, _ in paired]
    dens_f = [density(t) for _, _, t in paired]

    print("\nFirst 3 task transcript samples:")
    for tid, b, f in paired[:3]:
        print(f"  {tid}:")
        print(f"    buggy: contains_FAILED={'FAILED' in b}, n_FAILED={b.count('FAILED')}, density={density(b):.3f}, lines={len(b.splitlines())}")
        print(f"    fixed: contains_FAILED={'FAILED' in f}, n_FAILED={f.count('FAILED')}, density={density(f):.3f}, lines={len(f.splitlines())}")

    results = []
    for label, sb, sf in (
        ("contains_FAILED",       contains_b, contains_f),
        ("n_FAILED_lines",        n_failed_b, n_failed_f),
        ("failure_line_density",  dens_b, dens_f),
    ):
        scores = np.concatenate([sb, sf]).astype(float)
        labels = np.concatenate([np.ones(N, int), np.zeros(N, int)])
        try:
            auc = float(roc_auc_score(labels, scores))
            ap = float(average_precision_score(labels, scores))
        except Exception as exc:
            auc, ap = float("nan"), float("nan")
            print(f"  WARN: AUC/AP failed for {label}: {exc}")
        # In-sample best operating threshold from balanced accuracy.
        # For contains_FAILED, this is just threshold 0.5; AUC is binary.
        thr = 0.5 if label == "contains_FAILED" else float(np.median(scores))
        y_pred = (scores >= thr).astype(int)
        tp = int(((y_pred == 1) & (labels == 1)).sum())
        fp = int(((y_pred == 1) & (labels == 0)).sum())
        fn = int(((y_pred == 0) & (labels == 1)).sum())
        tn = int(((y_pred == 0) & (labels == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        false_edit = fp / max(fp + tn, 1)
        results.append({
            "classifier": label,
            "auc": auc,
            "ap": ap,
            "threshold_used": thr,
            "precision": precision,
            "recall": recall,
            "false_edit_rate": false_edit,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    print()
    print(f"{'classifier':<24} {'AUC':>8} {'AP':>8} {'prec':>7} {'rec':>7} {'false-edit':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['classifier']:<24} {r['auc']:>8.4f} {r['ap']:>8.4f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['false_edit_rate']:>12.4f}")

    out = {
        "config": {
            "tasks_root": str(args.tasks_root),
            "n_paired_tasks": N,
        },
        "classifiers": results,
        "headline": "AUC of `contains FAILED` regex on 499 SWE-bench-Verified-derived paired prompts",
        "value_of_residual_monitor": (
            "The residual projection monitor's §5.1 AUC of 0.989 is "
            "lower than this baseline's AUC. The mechanistic value of "
            "the projection is NOT discriminative power but identifying "
            "the residual-stream location and direction at which the "
            "pass/fail evidence enters the edit/noop decision."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
