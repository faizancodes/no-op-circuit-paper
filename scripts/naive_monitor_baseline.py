#!/usr/bin/env python3
"""Naive-monitor baselines for §5.1.1 of the paper.

Two specificity controls on the v_noop direction's monitoring power
(ROC-AUC 0.993 on the 99 real SWE-bench paired tasks):

  (a) Random unit-direction baseline. For N random unit vectors in the
      1536-D residual space at (L24, pos −1), compute the AUC of the linear
      probe (signed projection) over the same 99 paired prompts. Report the
      distribution: mean, 95th and 99th percentiles, max. If v_noop's 0.993
      sits in the upper tail, the direction is privileged.

  (b) Full-last-layer / mid-layer probe. Train a logistic-regression probe
      on the full 1536-D residual at the chosen site (L24 default; also
      L27 as an "even later" comparator) using LOOCV over 99 paired tasks.
      Report ROC-AUC. If a 1536-parameter probe doesn't beat v_noop's 0.993
      by much, the 1-D v_noop signal is essentially as good as the full
      residual — a strong direction-specificity claim.

Input cache: results/cache-real-qwen-n100-<TS>/<TS>/<task_id>/{buggy,fixed}__code_tests.pt
(see scripts/run_monitor_real.py for the loading idioms).

Outputs: results/monitor_real/baselines.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _resolve_default_cache_dir() -> Path:
    root = Path("/Users/faizanahmed/no-op-circuit/results")
    cands = sorted(root.glob("cache-real-qwen*"), key=lambda p: p.name, reverse=True)
    if not cands:
        raise SystemExit("no cache-real-qwen-* under results/; pass --cache-dir")
    return cands[0]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="real-task cache run dir (default: most recent results/cache-real-qwen-*)")
    p.add_argument("--v-noop", type=Path,
                   default=Path("/Users/faizanahmed/no-op-circuit/results/steer-20260516T021522Z/v_noop.pt"))
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--probe-layers", type=int, nargs="+", default=[24, 27],
                   help="layers at which to train the full-residual LR probe")
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--n-random", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=Path("/Users/faizanahmed/no-op-circuit/results/monitor_real/baselines.json"))
    args = p.parse_args(argv)

    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import roc_auc_score

    cache_dir = args.cache_dir or _resolve_default_cache_dir()
    print(f"cache_dir : {cache_dir}")

    v_blob = torch.load(args.v_noop, map_location="cpu", weights_only=False)
    v = v_blob["direction"].float()
    v_unit = (v / v.norm()).numpy()
    print(f"v_noop    : layer={v_blob['layer']}, pos={v_blob['position']}, "
          f"||v||={v_blob['norm']:.3f}")

    # ---- Load per-task residuals at the chosen layer(s) and (pos=-1) ----
    # Cache structure: <cache_dir>/<cache_dir.name>/<task_id>/{buggy,fixed}__code_tests.pt
    per_task: dict[str, dict[str, dict[int, "np.ndarray"]]] = defaultdict(lambda: defaultdict(dict))
    files = list(cache_dir.rglob("*__code_tests.pt"))
    print(f"found {len(files)} cache files")
    for pt in sorted(files):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        K = payload["resid_pre"].shape[2]
        pos_abs = args.position if args.position >= 0 else K + args.position
        tid = payload["task_id"]
        for L in set([args.layer] + args.probe_layers):
            per_task[tid][cond][L] = payload["resid_pre"][L, 0, pos_abs, :].float().numpy()

    pairs = [(t, s) for t, s in per_task.items() if "buggy" in s and "fixed" in s]
    print(f"paired real tasks: {len(pairs)}")
    if len(pairs) < 10:
        print("error: too few paired tasks", file=sys.stderr)
        return 2

    n = len(pairs)
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])  # buggy=1

    # ---- v_noop baseline (sanity, for direct comparison) ----
    buggy_vec_layer = np.stack([s["buggy"][args.layer] for _, s in pairs])
    fixed_vec_layer = np.stack([s["fixed"][args.layer] for _, s in pairs])
    proj_buggy = buggy_vec_layer @ v_unit
    proj_fixed = fixed_vec_layer @ v_unit
    scores_vnoop = np.concatenate([-proj_buggy, -proj_fixed])
    auc_vnoop = float(roc_auc_score(labels, scores_vnoop))
    print(f"\nv_noop monitor (sanity reproduction at L{args.layer}): ROC-AUC = {auc_vnoop:.4f}")

    # ---- (a) Random unit-direction distribution ----
    rng = np.random.default_rng(args.seed)
    aucs = np.empty(args.n_random)
    print(f"\n(a) drawing N={args.n_random} random unit vectors in d={v_unit.shape[0]}…")
    for i in range(args.n_random):
        u = rng.standard_normal(v_unit.shape[0]).astype(np.float64)
        u /= max(np.linalg.norm(u), 1e-12)
        pb = buggy_vec_layer @ u
        pf = fixed_vec_layer @ u
        # The "predict buggy" rule on v_noop is `-projection > thresh`.
        # For random u we don't know the sign; use abs-AUC = max(AUC, 1-AUC).
        s = np.concatenate([-pb, -pf])
        a = roc_auc_score(labels, s)
        aucs[i] = max(a, 1.0 - a)
    aucs.sort()
    pct = lambda q: float(np.quantile(aucs, q))
    rank_of_vnoop = int(np.sum(aucs < auc_vnoop)) / args.n_random * 100
    print(f"  random AUC distribution:")
    print(f"    mean   = {aucs.mean():.4f}")
    print(f"    median = {pct(0.50):.4f}")
    print(f"    p95    = {pct(0.95):.4f}")
    print(f"    p99    = {pct(0.99):.4f}")
    print(f"    max    = {aucs.max():.4f}  (across N={args.n_random})")
    print(f"  v_noop AUC = {auc_vnoop:.4f}  → percentile rank among random: {rank_of_vnoop:.2f}%")

    # ---- (b) Full-residual LR probe with LOOCV per requested layer ----
    print(f"\n(b) Full-residual LR probe, LOOCV over {n} pairs…")
    probe_results: dict[int, dict] = {}
    loo = LeaveOneOut()
    for L in args.probe_layers:
        X = np.concatenate([
            np.stack([s["buggy"][L] for _, s in pairs]),
            np.stack([s["fixed"][L] for _, s in pairs]),
        ])
        y = labels.copy()
        # Build per-pair index so LOOCV holds out both (buggy, fixed) for a task.
        pair_idx = np.concatenate([np.arange(n), np.arange(n)])
        scores_loo = np.empty(2 * n)
        for held_out_task in range(n):
            mask = (pair_idx != held_out_task)
            X_tr = X[mask]; y_tr = y[mask]
            clf = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
            clf.fit(X_tr, y_tr)
            X_te = X[~mask]
            scores_loo[~mask] = clf.decision_function(X_te)
        auc_full = float(roc_auc_score(y, scores_loo))
        probe_results[L] = {"auc": auc_full}
        print(f"  layer L{L}: ROC-AUC = {auc_full:.4f}  (d=1536 parameters, LOOCV by task)")

    # ---- Save ----
    out = {
        "v_noop_layer": int(v_blob["layer"]),
        "v_noop_position": int(v_blob["position"]),
        "n_pairs": n,
        "v_noop_auc": auc_vnoop,
        "random_n": args.n_random,
        "random_auc_mean": float(aucs.mean()),
        "random_auc_p50": pct(0.50),
        "random_auc_p95": pct(0.95),
        "random_auc_p99": pct(0.99),
        "random_auc_max": float(aucs.max()),
        "v_noop_percentile_among_random": rank_of_vnoop,
        "full_residual_probe": {str(L): r for L, r in probe_results.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
