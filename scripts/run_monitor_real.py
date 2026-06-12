#!/usr/bin/env python3
"""Evaluate the pre-edit monitor on real SWE-bench tasks using a FROZEN v_noop.

The whole point is to test transfer: v_noop was computed once on the 49 toy
tasks (training distribution). We do NOT recompute it. We project each real
task's L24/pos-1 residual onto the unit v_noop and check whether the same
threshold rule that worked on toys (low projection ⇒ predict buggy) still
discriminates.

Inputs:
  --cache-dir    : real-task cache run (default: most recent results/cache-real-*)
  --v-noop       : path to v_noop.pt (default: the toy-trained one)
  --layer, --position : intervention site (default: L24, pos −1)

Outputs (printed):
  - mean projection for real-buggy vs real-fixed
  - ROC-AUC, AP, op-point precision/recall/accuracy
  - per-task argmax-action distribution
Writes results/monitor_real/real_curves.npz with the same fields as
loo_curves.npz so the figure-renderer can read either.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _resolve_default_cache_dir() -> Path:
    root = Path("results")
    cands = sorted(root.glob("cache-real-*"), key=lambda p: p.name, reverse=True)
    if not cands:
        raise SystemExit("no cache-real-* under results/; pass --cache-dir explicitly")
    return cands[0]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="real-task cache run dir (default: most recent results/cache-real-*)")
    p.add_argument("--v-noop", type=Path,
                   default=Path("results/steer-20260516T021522Z/v_noop.pt"))
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--out", type=Path,
                   default=Path("results/monitor_real/real_curves.npz"))
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for numpy/torch/random (monitor eval is deterministic, "
                        "but downstream callers may resample/permute these arrays).")
    args = p.parse_args(argv)

    import random
    import numpy as np
    import torch
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    cache_dir = args.cache_dir or _resolve_default_cache_dir()
    print(f"cache_dir : {cache_dir}")

    v_blob = torch.load(args.v_noop, map_location="cpu", weights_only=False)
    v = v_blob["direction"].float()
    v_unit = (v / v.norm()).numpy()
    print(f"v_noop    : layer={v_blob['layer']}, pos={v_blob['position']}, "
          f"||v||={v_blob['norm']:.3f}, source_N={v_blob['n_pairs']}")
    assert v_blob["layer"] == args.layer and v_blob["position"] == args.position, (
        "v_noop's intervention site does not match --layer/--position"
    )

    # Collect per-(task, condition) projections + argmax action
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for pt in sorted(cache_dir.rglob("*__code_tests.pt")):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        K = payload["resid_pre"].shape[2]
        pos_abs = args.position if args.position >= 0 else K + args.position
        vec = payload["resid_pre"][args.layer, 0, pos_abs, :].float().numpy()
        proj = float(vec @ v_unit)
        action_logits = payload["action_logits"]
        argmax_name = max(action_logits.items(), key=lambda kv: kv[1]["logit"])[0]
        by_task[payload["task_id"]][cond] = {"proj": proj, "argmax": argmax_name}

    pairs = [(t, s["buggy"], s["fixed"]) for t, s in by_task.items()
             if "buggy" in s and "fixed" in s]
    print(f"paired real tasks: {len(pairs)}")
    if len(pairs) < 10:
        print("error: too few paired tasks", file=sys.stderr)
        return 2

    proj_buggy = np.asarray([b["proj"] for _, b, _ in pairs])
    proj_fixed = np.asarray([f["proj"] for _, _, f in pairs])
    print(f"\nmean projection (frozen v_noop, real distribution):")
    print(f"  buggy : mean={proj_buggy.mean():+.3f}  median={float(np.median(proj_buggy)):+.3f}  N={len(proj_buggy)}")
    print(f"  fixed : mean={proj_fixed.mean():+.3f}  median={float(np.median(proj_fixed)):+.3f}  N={len(proj_fixed)}")
    print(f"  gap   : mean(fixed)-mean(buggy) = {proj_fixed.mean()-proj_buggy.mean():+.3f}")
    print(f"  (toy baselines for comparison: clean-buggy −5.53 / clean-fixed +0.36)")

    # Score = −projection (higher = more buggy-like)
    scores = np.concatenate([-proj_buggy, -proj_fixed])
    labels = np.concatenate([np.ones_like(proj_buggy, dtype=int),
                              np.zeros_like(proj_fixed, dtype=int)])

    roc_auc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))
    fpr, tpr, roc_thresh = roc_curve(labels, scores)
    pr_p, pr_r, pr_thresh = precision_recall_curve(labels, scores)
    bal_acc = 0.5 * (tpr + (1.0 - fpr))
    best = int(np.argmax(bal_acc))
    op_threshold = float(roc_thresh[best])
    y_pred = (scores >= op_threshold).astype(int)
    tp = int(((y_pred == 1) & (labels == 1)).sum())
    fp = int(((y_pred == 1) & (labels == 0)).sum())
    fn = int(((y_pred == 0) & (labels == 1)).sum())
    tn = int(((y_pred == 0) & (labels == 0)).sum())
    op_precision = tp / max(tp + fp, 1)
    op_recall = tp / max(tp + fn, 1)
    op_accuracy = (tp + tn) / len(labels)
    op_false_edit_rate = fp / max(fp + tn, 1)

    print(f"\n=== monitor metrics on REAL tasks ===")
    print(f"  ROC-AUC        : {roc_auc:.4f}")
    print(f"  PR-AUC (AP)    : {pr_auc:.4f}")
    print(f"  operating pt   : -projection ≥ {op_threshold:+.4f}")
    print(f"    precision    : {op_precision:.4f}  ({tp}/{tp+fp})")
    print(f"    recall       : {op_recall:.4f}  ({tp}/{tp+fn})")
    print(f"    accuracy     : {op_accuracy:.4f}  ({tp+tn}/{len(labels)})")
    print(f"    false-edit   : {op_false_edit_rate:.4f}  ({fp}/{fp+tn})")

    # argmax-action distribution
    argmax_b = Counter([b["argmax"] for _, b, _ in pairs])
    argmax_f = Counter([f["argmax"] for _, _, f in pairs])
    print(f"\nargmax-action distribution (real tasks):")
    actions = ["view", "grep", "test", "edit", "noop"]
    print(f"  cond   " + "  ".join(f"{a:>6}" for a in actions))
    for label_, cnt in (("buggy", argmax_b), ("fixed", argmax_f)):
        n = sum(cnt.values()) or 1
        cells = "  ".join(f"{100*cnt.get(a,0)/n:5.0f}%" for a in actions)
        print(f"  {label_:<6} {cells}")

    # Verdict gate
    print()
    gap = float(proj_fixed.mean() - proj_buggy.mean())
    if roc_auc >= 0.80 and gap > 1.5:
        verdict = "STRONG"
    elif roc_auc >= 0.65:
        verdict = "MIXED"
    else:
        verdict = "WEAK"
    print(f"DECISION GATE: {verdict}  (ROC-AUC={roc_auc:.3f}, projection gap={gap:+.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(args.out),
        task_ids=np.asarray([t for t, _, _ in pairs] * 2),
        conditions=np.asarray(["buggy"]*len(pairs) + ["fixed"]*len(pairs)),
        scores=scores,
        raw_projections=np.concatenate([proj_buggy, proj_fixed]),
        labels=labels,
        roc_fpr=fpr, roc_tpr=tpr, roc_thresh=roc_thresh,
        pr_precision=pr_p, pr_recall=pr_r, pr_thresh=pr_thresh,
        roc_auc=np.asarray(roc_auc),
        pr_auc=np.asarray(pr_auc),
        op_threshold=np.asarray(op_threshold),
        op_precision=np.asarray(op_precision),
        op_recall=np.asarray(op_recall),
        op_accuracy=np.asarray(op_accuracy),
        op_false_edit_rate=np.asarray(op_false_edit_rate),
        op_tp=np.asarray(tp), op_fp=np.asarray(fp),
        op_fn=np.asarray(fn), op_tn=np.asarray(tn),
        proj_buggy_mean=np.asarray(float(proj_buggy.mean())),
        proj_fixed_mean=np.asarray(float(proj_fixed.mean())),
        gap=np.asarray(gap),
        verdict=np.asarray(verdict),
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
