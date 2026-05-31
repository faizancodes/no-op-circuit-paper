#!/usr/bin/env python3
"""Held-out threshold calibration for the §5.1 monitor.

The §5.1 operating-point precision/recall/false-edit rows are
in-sample: the balanced-accuracy threshold is fit on the same
499-instance evaluation set it is then evaluated on. Reviewers
correctly flag this as not deployment-meaningful.

This script computes out-of-sample operating-point metrics two ways:

(A) Random 50/50 split, K seeds. For each seed:
      - Split tasks into calibration (50%) and evaluation (50%).
      - Fit threshold on calibration (balanced-accuracy).
      - Apply threshold on the eval split.
    Report mean ± bootstrap CI across seeds.

(B) Leave-one-repo-out. For each unique repo:
      - Fit threshold on all OTHER repos pooled.
      - Apply on the held-out repo's tasks.
    Pool the held-out predictions; report metrics.

Inputs: existing cached projections in `results/cache-real-qwen-n500-*`.
No new compute required.

Output: `results/monitor_real/held_out_thresholds.json`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _collect_projections(cache_dir: Path, v_unit, layer: int, pos: int):
    """Return list of (task_id, condition, score) where score = -projection."""
    import torch
    rows: list[dict] = []
    for pt in sorted(cache_dir.rglob("*__code_tests.pt")):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        T = int(payload["resid_pre"].shape[2])
        abs_pos = pos if pos >= 0 else T + pos
        vec = payload["resid_pre"][layer, 0, abs_pos, :].float().numpy()
        proj = float(vec @ v_unit)
        rows.append({
            "task_id": payload["task_id"],
            "condition": cond,
            "score": -proj,
        })
    return rows


def _fit_balanced_threshold(scores, labels):
    """Fit threshold maximizing balanced accuracy. Returns threshold."""
    import numpy as np
    from sklearn.metrics import roc_curve
    fpr, tpr, thresh = roc_curve(labels, scores)
    bal_acc = 0.5 * (tpr + (1 - fpr))
    best = int(np.argmax(bal_acc))
    return float(thresh[best])


def _opmetrics(scores, labels, threshold):
    """Compute operating-point precision/recall/false-edit at threshold."""
    import numpy as np
    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_pred == 1) & (labels == 1)).sum())
    fp = int(((y_pred == 1) & (labels == 0)).sum())
    fn = int(((y_pred == 0) & (labels == 1)).sum())
    tn = int(((y_pred == 0) & (labels == 0)).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall":    tp / max(tp + fn, 1),
        "accuracy":  (tp + tn) / max(len(labels), 1),
        "false_edit_rate": fp / max(fp + tn, 1),
    }


def _repo_of(task_id: str) -> str:
    """SWE-bench task_ids look like 'astropy_astropy_12907'. The first two
    underscore-separated tokens identify the repo."""
    parts = task_id.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return task_id


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path,
                   default=Path("results/cache-real-qwen-n500-20260516T235301Z"))
    p.add_argument("--v-noop", type=Path,
                   default=Path("results/steer-20260516T021522Z/v_noop.pt"))
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--pos", type=int, default=-1)
    p.add_argument("--n-splits", type=int, default=200,
                   help="Random 50/50 splits to average over.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=Path("results/monitor_real/held_out_thresholds.json"))
    args = p.parse_args(argv)

    import numpy as np
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    v_blob = torch.load(args.v_noop, map_location="cpu", weights_only=False)
    v_unit = (v_blob["direction"].float() / v_blob["direction"].float().norm()).numpy()
    print(f"v_noop: L{v_blob['layer']}/pos {v_blob['position']:+d}")

    rows = _collect_projections(args.cache_dir, v_unit, args.layer, args.pos)
    # Group into paired tasks
    by_task: dict[str, dict[str, float]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], {})[r["condition"]] = r["score"]
    paired = sorted(
        [(t, s["buggy"], s["fixed"]) for t, s in by_task.items()
         if "buggy" in s and "fixed" in s])
    N = len(paired)
    print(f"paired tasks: {N}")

    task_ids = np.asarray([t for t, _, _ in paired])
    sb = np.asarray([b for _, b, _ in paired])  # buggy scores
    sf = np.asarray([f for _, _, f in paired])  # fixed scores
    repos = np.asarray([_repo_of(t) for t, _, _ in paired])

    # In-sample baseline for reference
    all_scores = np.concatenate([sb, sf])
    all_labels = np.concatenate([np.ones(N, int), np.zeros(N, int)])
    in_sample_thr = _fit_balanced_threshold(all_scores, all_labels)
    in_sample_metrics = _opmetrics(all_scores, all_labels, in_sample_thr)
    print(f"\nin-sample (§5.1 numbers, for reference):")
    print(f"  threshold:        {in_sample_thr:+.4f}")
    print(f"  precision:        {in_sample_metrics['precision']:.4f}")
    print(f"  recall:           {in_sample_metrics['recall']:.4f}")
    print(f"  false_edit_rate:  {in_sample_metrics['false_edit_rate']:.4f}")

    # (A) Random 50/50 splits
    print(f"\n(A) Random 50/50 split × {args.n_splits} seeds")
    rng = np.random.default_rng(args.seed)
    metrics_per_split = []
    thresholds_per_split = []
    for s in range(args.n_splits):
        idx = np.arange(N)
        rng.shuffle(idx)
        n_cal = N // 2
        cal_idx, ev_idx = idx[:n_cal], idx[n_cal:]
        cal_scores = np.concatenate([sb[cal_idx], sf[cal_idx]])
        cal_labels = np.concatenate([
            np.ones(len(cal_idx), int), np.zeros(len(cal_idx), int)
        ])
        ev_scores = np.concatenate([sb[ev_idx], sf[ev_idx]])
        ev_labels = np.concatenate([
            np.ones(len(ev_idx), int), np.zeros(len(ev_idx), int)
        ])
        thr = _fit_balanced_threshold(cal_scores, cal_labels)
        m = _opmetrics(ev_scores, ev_labels, thr)
        m["threshold"] = thr
        metrics_per_split.append(m)
        thresholds_per_split.append(thr)

    def _ci(name):
        arr = np.asarray([m[name] for m in metrics_per_split])
        return {
            "mean":    float(arr.mean()),
            "std":     float(arr.std(ddof=1)),
            "ci_2_5":  float(np.percentile(arr, 2.5)),
            "ci_97_5": float(np.percentile(arr, 97.5)),
        }

    split_stats = {
        "threshold":       _ci("threshold"),
        "precision":       _ci("precision"),
        "recall":          _ci("recall"),
        "accuracy":        _ci("accuracy"),
        "false_edit_rate": _ci("false_edit_rate"),
    }
    print(f"  threshold (mean ± std):        {split_stats['threshold']['mean']:+.4f} ± {split_stats['threshold']['std']:.4f}")
    print(f"  precision  [95% range]:        {split_stats['precision']['mean']:.4f}  "
          f"[{split_stats['precision']['ci_2_5']:.4f}, {split_stats['precision']['ci_97_5']:.4f}]")
    print(f"  recall     [95% range]:        {split_stats['recall']['mean']:.4f}  "
          f"[{split_stats['recall']['ci_2_5']:.4f}, {split_stats['recall']['ci_97_5']:.4f}]")
    print(f"  false_edit [95% range]:        {split_stats['false_edit_rate']['mean']:.4f}  "
          f"[{split_stats['false_edit_rate']['ci_2_5']:.4f}, {split_stats['false_edit_rate']['ci_97_5']:.4f}]")

    # (B) Leave-one-repo-out
    print(f"\n(B) Leave-one-repo-out")
    unique_repos = sorted(set(repos.tolist()))
    print(f"  unique repos: {len(unique_repos)}: {unique_repos}")

    pooled_scores: list[float] = []
    pooled_labels: list[int] = []
    per_repo: list[dict] = []
    for repo in unique_repos:
        in_repo = repos == repo
        if not any(in_repo):
            continue
        out_repo = ~in_repo
        cal_scores = np.concatenate([sb[out_repo], sf[out_repo]])
        cal_labels = np.concatenate([
            np.ones(int(out_repo.sum()), int),
            np.zeros(int(out_repo.sum()), int)
        ])
        ev_scores = np.concatenate([sb[in_repo], sf[in_repo]])
        ev_labels = np.concatenate([
            np.ones(int(in_repo.sum()), int),
            np.zeros(int(in_repo.sum()), int)
        ])
        thr = _fit_balanced_threshold(cal_scores, cal_labels)
        m = _opmetrics(ev_scores, ev_labels, thr)
        m["threshold"] = thr
        m["repo"] = repo
        m["n_fixed_in_repo"] = int(in_repo.sum())
        per_repo.append(m)
        pooled_scores.extend(ev_scores.tolist())
        pooled_labels.extend(ev_labels.tolist())

    # Pooled held-out metrics (this is the headline LOO number)
    # Compute on pooled (score, prediction) where each prediction used a
    # repo-specific threshold fit on out-of-repo data.
    pooled_y_pred = []
    for m_repo in per_repo:
        thr = m_repo["threshold"]
        in_repo_mask = repos == m_repo["repo"]
        ev_scores = np.concatenate([sb[in_repo_mask], sf[in_repo_mask]])
        pooled_y_pred.extend((ev_scores >= thr).astype(int).tolist())
    pooled_scores_arr = np.asarray(pooled_scores)
    pooled_labels_arr = np.asarray(pooled_labels)
    pooled_y_pred_arr = np.asarray(pooled_y_pred)
    tp = int(((pooled_y_pred_arr == 1) & (pooled_labels_arr == 1)).sum())
    fp = int(((pooled_y_pred_arr == 1) & (pooled_labels_arr == 0)).sum())
    fn = int(((pooled_y_pred_arr == 0) & (pooled_labels_arr == 1)).sum())
    tn = int(((pooled_y_pred_arr == 0) & (pooled_labels_arr == 0)).sum())
    pooled = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall":    tp / max(tp + fn, 1),
        "accuracy":  (tp + tn) / max(len(pooled_labels_arr), 1),
        "false_edit_rate": fp / max(fp + tn, 1),
    }
    print(f"  pooled LOO precision : {pooled['precision']:.4f}")
    print(f"  pooled LOO recall    : {pooled['recall']:.4f}")
    print(f"  pooled LOO accuracy  : {pooled['accuracy']:.4f}")
    print(f"  pooled LOO false-edit: {pooled['false_edit_rate']:.4f}")

    # Persist
    out = {
        "config": {
            "cache_dir": str(args.cache_dir),
            "v_noop": str(args.v_noop),
            "layer": args.layer, "pos": args.pos,
            "n_splits": args.n_splits, "seed": args.seed,
            "n_paired_tasks": N,
        },
        "in_sample_for_reference": {
            "threshold": in_sample_thr, **in_sample_metrics
        },
        "random_50_50_split": split_stats,
        "leave_one_repo_out": {
            "per_repo": per_repo,
            "pooled": pooled,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
