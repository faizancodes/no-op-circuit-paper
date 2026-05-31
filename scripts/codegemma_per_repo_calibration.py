#!/usr/bin/env python3
"""Per-repo ROC-AUC + threshold calibration counterfactual for CodeGemma.

The §6 deployment paragraph claimed per-repo threshold calibration "recovers most
of the headline AUC's discrimination." This script replaces that hand-wave with
the actual number: for each repo with N_fixed >= 10, compute per-repo AUC and
the false-edit rate under each repo's own balanced-accuracy threshold. Then
report the pooled false-edit rate that a deployer would see if they applied each
repo's calibrated threshold instead of the global one.

Reads:
  results/monitor_real/real_curves_codegemma_n500.npz   (scores + labels per task)
  data/real_tasks/<task_id>/meta.yaml                    (swe_bench_repo)

Writes:
  results/monitor_real/codegemma_per_repo_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _safe_load_yaml(p: Path):
    try:
        import yaml
        return yaml.safe_load(p.read_text())
    except Exception:
        return None


def _per_repo_metrics(scores, labels, n_min_fixed: int = 10):
    """For a single repo's score/label arrays, compute AUC + balanced-T +
    FE<=5%-T trade. Returns dict or None if too few samples."""
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_fixed = int((labels == 0).sum())
    n_buggy = int((labels == 1).sum())
    if n_fixed < n_min_fixed or n_buggy < 1:
        return None
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        return None

    fpr, tpr, thresh = roc_curve(labels, scores)
    bal = 0.5 * (tpr + (1.0 - fpr))
    bi = int(np.argmax(bal))
    bal_T = float(thresh[bi])
    pred = scores >= bal_T
    tp = int(((labels == 1) & pred).sum())
    fp = int(((labels == 0) & pred).sum())
    fn = int(((labels == 1) & ~pred).sum())
    tn = int(((labels == 0) & ~pred).sum())
    bal_metrics = {
        "threshold": bal_T,
        "false_edit": fp / max(n_fixed, 1),
        "recall": tp / max(n_buggy, 1),
        "precision": tp / max(tp + fp, 1) if (tp + fp) > 0 else 1.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

    fe5_T = None; fe5_metrics = None
    for t in sorted(thresh):
        pred5 = scores >= t
        fp5 = int(((labels == 0) & pred5).sum())
        if fp5 / max(n_fixed, 1) <= 0.05:
            tp5 = int(((labels == 1) & pred5).sum())
            fe5_T = float(t)
            fe5_metrics = {
                "threshold": float(t),
                "false_edit": fp5 / max(n_fixed, 1),
                "recall": tp5 / max(n_buggy, 1),
                "precision": tp5 / max(tp5 + fp5, 1) if (tp5 + fp5) > 0 else 1.0,
                "tp": tp5, "fp": fp5,
            }
            break

    return {
        "n_fixed": n_fixed, "n_buggy": n_buggy, "auc": auc,
        "balanced": bal_metrics, "fe5": fe5_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz",
                   type=Path,
                   default=Path("results/monitor_real/real_curves_codegemma_n500.npz"))
    p.add_argument("--meta-dir", type=Path, default=Path("data/real_tasks"))
    p.add_argument("--n-min-fixed", type=int, default=10)
    p.add_argument("--out",
                   type=Path,
                   default=Path("results/monitor_real/codegemma_per_repo_calibration.json"))
    args = p.parse_args(argv)

    import numpy as np

    blob = np.load(args.npz, allow_pickle=True)
    global_T = float(blob["op_threshold"])
    global_auc = float(blob["roc_auc"])
    print(f"Loaded {args.npz.name}: global AUC = {global_auc:.4f}, "
          f"global balanced-T = {global_T:+.4f}")

    # Per-task repo lookup
    repo_of: dict[str, str | None] = {}
    for tid in set(map(str, blob["task_ids"])):
        m = _safe_load_yaml(args.meta_dir / tid / "meta.yaml")
        repo_of[tid] = (m or {}).get("swe_bench_repo") if isinstance(m, dict) else None

    # Group scores by repo
    by_repo_scores: dict[str, list[float]] = {}
    by_repo_labels: dict[str, list[int]] = {}
    for i in range(len(blob["task_ids"])):
        tid = str(blob["task_ids"][i])
        r = repo_of.get(tid) or "(unknown)"
        by_repo_scores.setdefault(r, []).append(float(blob["scores"][i]))
        by_repo_labels.setdefault(r, []).append(int(blob["labels"][i]))

    # Per-repo metrics
    rows: list[dict] = []
    for repo in sorted(by_repo_scores, key=lambda r: -len(by_repo_scores[r])):
        m = _per_repo_metrics(by_repo_scores[repo], by_repo_labels[repo],
                              n_min_fixed=args.n_min_fixed)
        if m is None:
            continue
        rows.append({"repo": repo, **m})

    print(f"\n{'repo':<32} {'N_fix':>6} {'AUC':>8} "
          f"{'bal-T':>9} {'bal-FE':>8} {'bal-rec':>9}  "
          f"{'fe5-T':>9} {'fe5-FE':>8} {'fe5-rec':>9}")
    print("-" * 110)
    for r in rows:
        bal = r["balanced"]
        fe5 = r["fe5"]
        fe5_str = (f"{fe5['threshold']:+9.3f} {fe5['false_edit']*100:>7.2f}% "
                   f"{fe5['recall']*100:>8.1f}%") if fe5 else f"{'-':>9} {'-':>8} {'-':>9}"
        print(f"{r['repo']:<32} {r['n_fixed']:>6} {r['auc']:>8.4f} "
              f"{bal['threshold']:+9.3f} {bal['false_edit']*100:>7.2f}% "
              f"{bal['recall']*100:>8.1f}%  {fe5_str}")

    # ---- Pooled counterfactual: each repo at its OWN balanced-T ----
    repo_calibrated_T: dict[str, float] = {r["repo"]: r["balanced"]["threshold"]
                                            for r in rows}
    pooled_tp = pooled_fp = pooled_tn = pooled_fn = 0
    pooled_pairs = 0
    pooled_fixed = pooled_buggy = 0
    for repo, scores in by_repo_scores.items():
        labels = by_repo_labels[repo]
        T = repo_calibrated_T.get(repo, global_T)  # fall back to global for tiny repos
        for s, lab in zip(scores, labels):
            pred = s >= T
            if lab == 1 and pred: pooled_tp += 1
            elif lab == 1 and not pred: pooled_fn += 1
            elif lab == 0 and pred: pooled_fp += 1
            else: pooled_tn += 1
            pooled_fixed += int(lab == 0); pooled_buggy += int(lab == 1)
            pooled_pairs += 1

    pooled_fe_calib = pooled_fp / max(pooled_fixed, 1)
    pooled_rec_calib = pooled_tp / max(pooled_buggy, 1)
    pooled_prec_calib = pooled_tp / max(pooled_tp + pooled_fp, 1) if (pooled_tp + pooled_fp) > 0 else 1.0
    pooled_acc_calib = (pooled_tp + pooled_tn) / max(pooled_pairs, 1)

    # ---- Pooled at global T (for comparison) ----
    g_tp = g_fp = g_tn = g_fn = 0
    for repo, scores in by_repo_scores.items():
        labels = by_repo_labels[repo]
        for s, lab in zip(scores, labels):
            pred = s >= global_T
            if lab == 1 and pred: g_tp += 1
            elif lab == 1 and not pred: g_fn += 1
            elif lab == 0 and pred: g_fp += 1
            else: g_tn += 1
    pooled_fe_global = g_fp / max(pooled_fixed, 1)
    pooled_rec_global = g_tp / max(pooled_buggy, 1)
    pooled_prec_global = g_tp / max(g_tp + g_fp, 1) if (g_tp + g_fp) > 0 else 1.0

    print(f"\n=== POOLED COUNTERFACTUAL (across {pooled_pairs} obs from "
          f"{len(rows)} eligible repos; tiny repos fall back to global-T) ===")
    print(f"  global balanced-T (single threshold for all):")
    print(f"    pooled FE = {pooled_fe_global*100:.2f}%  recall = {pooled_rec_global*100:.2f}%  "
          f"precision = {pooled_prec_global*100:.2f}%")
    print(f"  per-repo balanced-T (each repo uses its own):")
    print(f"    pooled FE = {pooled_fe_calib*100:.2f}%  recall = {pooled_rec_calib*100:.2f}%  "
          f"precision = {pooled_prec_calib*100:.2f}%")
    print(f"  FE reduction: {(pooled_fe_global - pooled_fe_calib)*100:+.2f}%-points "
          f"({pooled_fe_calib/max(pooled_fe_global,1e-9)*100:.1f}% of global)")

    # Decision-gate check on django/sphinx
    gate_pass = True
    for repo in ("django/django", "sphinx-doc/sphinx"):
        match = next((r for r in rows if r["repo"] == repo), None)
        if match and match["auc"] < 0.85:
            print(f"  DECISION GATE: {repo} per-repo AUC {match['auc']:.4f} < 0.85")
            gate_pass = False
    if gate_pass:
        print(f"  DECISION GATE: PASSES (django + sphinx per-repo AUCs both >= 0.85)")

    out = {
        "global": {
            "auc": global_auc,
            "balanced_threshold": global_T,
            "pooled_fe": pooled_fe_global,
            "pooled_recall": pooled_rec_global,
            "pooled_precision": pooled_prec_global,
        },
        "per_repo": rows,
        "pooled_calibrated": {
            "n_obs": pooled_pairs,
            "pooled_fe": pooled_fe_calib,
            "pooled_recall": pooled_rec_calib,
            "pooled_precision": pooled_prec_calib,
            "pooled_accuracy": pooled_acc_calib,
            "tp": pooled_tp, "fp": pooled_fp, "fn": pooled_fn, "tn": pooled_tn,
        },
        "gate_pass_django_sphinx_auc_ge_0p85": gate_pass,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
