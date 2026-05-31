#!/usr/bin/env python3
"""Pre-edit monitor: project residual onto v_noop at L24/pos −1 and score.

For each of the 49 paired (buggy, fixed) tasks we evaluate a leave-one-out
monitor: v_noop is computed from the OTHER 48 task pairs' mean-difference at
L24/pos −1, the held-out task's residual is projected onto that unit vector,
and the signed scalar is treated as a classifier score.

Labels are buggy=1 (should edit), fixed=0 (should noop). Lower projection
corresponds to buggy-like, so the canonical "predict buggy" rule is
`-projection > threshold` (we negate the score before passing to sklearn).

Outputs:
  results/monitor/loo_curves.npz
  prints a 5-row summary table (ROC-AUC, PR-AUC, operating-point metrics)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/Users/faizanahmed/no-op-circuit/results/cache-20260515T221105Z"),
    )
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/Users/faizanahmed/no-op-circuit/results/monitor/loo_curves.npz"),
    )
    args = p.parse_args(argv)

    import numpy as np
    import torch
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    # Collect the 49 paired (buggy, fixed) resid_pre activations at the cell.
    by_task: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for pt in sorted(args.cache_dir.rglob("*__code_tests.pt")):
        name = pt.stem  # "<cond>__code_tests"
        cond = name.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        K = int(payload["resid_pre"].shape[2])
        pos_abs = args.position if args.position >= 0 else K + args.position
        vec = payload["resid_pre"][args.layer, 0, pos_abs, :].float().numpy()
        by_task[payload["task_id"]][cond] = vec

    pairs = [
        (tid, sides["buggy"], sides["fixed"])
        for tid, sides in by_task.items()
        if "buggy" in sides and "fixed" in sides
    ]
    print(f"paired tasks loaded: {len(pairs)}  (expected 49)")
    if len(pairs) < 4:
        print("error: too few pairs for LOOCV", file=sys.stderr)
        return 2

    # --- LOOCV: for each task, compute v_noop from the OTHER 48 task pairs. ---
    task_ids: list[str] = []
    conditions: list[str] = []
    scores: list[float] = []   # we store `-projection` so higher = more buggy-like
    raw_projs: list[float] = []
    labels: list[int] = []

    pairs_arr = pairs  # list of (tid, buggy_vec, fixed_vec)
    for i, (tid, b_i, f_i) in enumerate(pairs_arr):
        # v from the other 48 pairs
        others = [(b, f) for j, (_, b, f) in enumerate(pairs_arr) if j != i]
        buggy_mean = np.mean([b for b, _ in others], axis=0)
        fixed_mean = np.mean([f for _, f in others], axis=0)
        v = fixed_mean - buggy_mean
        v_norm = float(np.linalg.norm(v))
        v_unit = v / max(v_norm, 1e-12)
        proj_b = float(b_i @ v_unit)
        proj_f = float(f_i @ v_unit)
        # Two scored rows per task — one buggy, one fixed.
        task_ids.extend([tid, tid])
        conditions.extend(["buggy", "fixed"])
        raw_projs.extend([proj_b, proj_f])
        scores.extend([-proj_b, -proj_f])  # higher score => more buggy-like
        labels.extend([1, 0])

    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    raw_projs_arr = np.asarray(raw_projs, dtype=np.float64)

    # --- ROC + PR ---
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    fpr, tpr, roc_thresh = roc_curve(y_true, y_score)
    pr_p, pr_r, pr_thresh = precision_recall_curve(y_true, y_score)

    # --- Operating point: maximize balanced accuracy = (TPR + (1-FPR))/2 ---
    bal_acc = 0.5 * (tpr + (1.0 - fpr))
    best = int(np.argmax(bal_acc))
    op_threshold = float(roc_thresh[best])
    y_pred = (y_score >= op_threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    op_precision = tp / max(tp + fp, 1)
    op_recall = tp / max(tp + fn, 1)
    op_accuracy = (tp + tn) / len(y_true)
    op_false_edit_rate = fp / max(fp + tn, 1)  # fraction of FIXED tasks predicted buggy

    print()
    print(f"=== LOO monitor at L{args.layer}, pos {args.position} ===")
    print(f"  N rows         : {len(y_true)}  (49 buggy + 49 fixed)")
    print(f"  raw projection : mean(buggy)={raw_projs_arr[::2].mean():+.3f}  "
          f"mean(fixed)={raw_projs_arr[1::2].mean():+.3f}")
    print(f"  ROC-AUC        : {roc_auc:.4f}")
    print(f"  PR-AUC         : {pr_auc:.4f}")
    print(f"  Operating point (max balanced accuracy):")
    print(f"    threshold     = {op_threshold:+.4f}  (on -projection)")
    print(f"    precision     = {op_precision:.4f}  ({tp}/{tp+fp})")
    print(f"    recall        = {op_recall:.4f}  ({tp}/{tp+fn})")
    print(f"    accuracy      = {op_accuracy:.4f}  ({tp+tn}/{len(y_true)})")
    print(f"    false-edit    = {op_false_edit_rate:.4f}  ({fp}/{fp+tn} fixed misclassified as buggy)")

    if roc_auc >= 0.80:
        print(f"\nGATE: STRONG (ROC-AUC ≥ 0.80) — paper-strength monitor result.")
    elif roc_auc < 0.65:
        print(f"\nGATE: WEAK (ROC-AUC < 0.65) — report negative result.")
    else:
        print(f"\nGATE: BETWEEN (0.65 ≤ ROC-AUC < 0.80) — report as moderate.")

    # --- Save curves ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(args.out),
        task_ids=np.asarray(task_ids),
        conditions=np.asarray(conditions),
        scores=y_score,
        raw_projections=raw_projs_arr,
        labels=y_true,
        roc_fpr=fpr,
        roc_tpr=tpr,
        roc_thresh=roc_thresh,
        pr_precision=pr_p,
        pr_recall=pr_r,
        pr_thresh=pr_thresh,
        roc_auc=np.asarray(roc_auc),
        pr_auc=np.asarray(pr_auc),
        op_threshold=np.asarray(op_threshold),
        op_precision=np.asarray(op_precision),
        op_recall=np.asarray(op_recall),
        op_accuracy=np.asarray(op_accuracy),
        op_false_edit_rate=np.asarray(op_false_edit_rate),
        op_tp=np.asarray(tp),
        op_fp=np.asarray(fp),
        op_fn=np.asarray(fn),
        op_tn=np.asarray(tn),
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
