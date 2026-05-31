#!/usr/bin/env python3
"""Bootstrap AUC + AP confidence intervals + LOOCV permutation null.

Addresses Blocker B4 of the pre-submission audit. Computes:

  1. Paired-bootstrap 95% CI for ROC-AUC and AP on each model's
     full SWE-bench Verified curve (Qwen / CodeGemma / DeepSeek).
  2. Per-repo bootstrap CIs for AUC on CodeGemma (the model with the
     weakest false-edit rate). Repos with N_fixed < 30 are flagged
     `underpowered`.
  3. Permutation null for the toy-substrate LOOCV monitor AUC
     (Appendix F's `1.000` claim): shuffle labels 1000× and recompute
     LOOCV AUC; report the observed AUC's percentile.

Outputs are written to results/monitor_real/auc_ci.json with structure:

  {
    "global": {
      "qwen":      {"auc": ..., "auc_ci_lo": ..., "auc_ci_hi": ...,
                    "ap": ..., "ap_ci_lo": ..., "ap_ci_hi": ..., "n": ...},
      "codegemma": {...},
      "deepseek":  {...}
    },
    "per_repo_codegemma": [
      {"repo": ..., "n_fixed": ..., "auc": ..., "auc_ci_lo": ...,
       "auc_ci_hi": ..., "underpowered": bool}, ...
    ],
    "loocv_toy_permutation": {
      "observed_auc": 1.0,
      "null_mean": ...,  "null_std": ...,
      "null_q975": ...,  "null_max": ...,
      "p_value": ...,    "n_perm": 1000
    }
  }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_metric_ci(scores, labels, metric_fn, n_boot: int, rng):
    import numpy as np

    n = len(scores)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Skip resamples where one class is missing — undefined AUC
        if len(set(labels[idx].tolist())) < 2:
            boot[b] = np.nan
            continue
        boot[b] = metric_fn(labels[idx], scores[idx])
    boot = boot[~np.isnan(boot)]
    if len(boot) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _global_cis(npz_path: Path, n_boot: int, rng) -> dict:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    z = np.load(npz_path, allow_pickle=True)
    scores = np.asarray(z["scores"], dtype=float)
    labels = np.asarray(z["labels"], dtype=int)
    auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    auc_lo, auc_hi = _bootstrap_metric_ci(scores, labels, roc_auc_score, n_boot, rng)
    ap_lo, ap_hi = _bootstrap_metric_ci(scores, labels, average_precision_score, n_boot, rng)
    return {
        "n": int(len(scores)),
        "auc": auc, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
        "ap": ap, "ap_ci_lo": ap_lo, "ap_ci_hi": ap_hi,
    }


def _per_repo_cis(npz_path: Path, meta_dir: Path, n_boot: int,
                  underpowered_thresh: int, min_n_fixed: int, rng) -> list[dict]:
    import numpy as np
    import yaml
    from sklearn.metrics import roc_auc_score

    z = np.load(npz_path, allow_pickle=True)
    task_ids = [str(t) for t in z["task_ids"]]
    scores = np.asarray(z["scores"], dtype=float)
    labels = np.asarray(z["labels"], dtype=int)

    def _repo_of(tid: str) -> str | None:
        p = meta_dir / tid / "meta.yaml"
        if not p.is_file():
            return None
        try:
            m = yaml.safe_load(p.read_text())
            return (m or {}).get("swe_bench_repo")
        except Exception:
            return None

    by_repo: dict[str, dict] = {}
    for i, tid in enumerate(task_ids):
        r = _repo_of(tid) or "(unknown)"
        by_repo.setdefault(r, {"scores": [], "labels": []})
        by_repo[r]["scores"].append(scores[i])
        by_repo[r]["labels"].append(labels[i])

    rows: list[dict] = []
    for repo, d in sorted(by_repo.items()):
        s = np.asarray(d["scores"]); lab = np.asarray(d["labels"], dtype=int)
        n_fixed = int((lab == 0).sum())
        n_buggy = int((lab == 1).sum())
        if n_fixed < min_n_fixed or n_buggy < 1:
            continue
        try:
            auc = float(roc_auc_score(lab, s))
        except ValueError:
            continue
        lo, hi = _bootstrap_metric_ci(s, lab, roc_auc_score, n_boot, rng)
        rows.append({
            "repo": repo, "n_fixed": n_fixed, "n_buggy": n_buggy,
            "auc": auc, "auc_ci_lo": lo, "auc_ci_hi": hi,
            "underpowered": n_fixed < underpowered_thresh,
        })
    rows.sort(key=lambda r: -r["n_fixed"])
    return rows


def _loocv_permutation_null(npz_path: Path, n_perm: int, rng) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    z = np.load(npz_path, allow_pickle=True)
    scores = np.asarray(z["scores"], dtype=float)
    labels = np.asarray(z["labels"], dtype=int)
    observed_auc = float(roc_auc_score(labels, scores))
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(labels)
        null[i] = roc_auc_score(perm, scores)
    p_value = float(((null >= observed_auc).sum() + 1) / (n_perm + 1))
    return {
        "observed_auc": observed_auc,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "null_q975": float(np.quantile(null, 0.975)),
        "null_max": float(null.max()),
        "p_value": p_value,
        "n_perm": n_perm,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--monitor-real-dir", type=Path,
                   default=Path("results/monitor_real"))
    p.add_argument("--loocv-toy-npz", type=Path,
                   default=Path("results/monitor/loo_curves.npz"))
    p.add_argument("--meta-dir", type=Path, default=Path("data/real_tasks"))
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--underpowered-thresh", type=int, default=30)
    p.add_argument("--min-n-fixed-per-repo", type=int, default=5)
    p.add_argument("--out", type=Path,
                   default=Path("results/monitor_real/auc_ci.json"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import random

    import numpy as np
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    models = {
        "qwen":      args.monitor_real_dir / "real_curves_qwen_n500.npz",
        "codegemma": args.monitor_real_dir / "real_curves_codegemma_n500.npz",
        "deepseek":  args.monitor_real_dir / "real_curves_deepseek_n500.npz",
    }
    missing = [p for p in models.values() if not p.is_file()]
    if missing:
        print(f"missing curves files: {missing}", file=sys.stderr)
        return 2

    out = {"global": {}, "per_repo_codegemma": [], "loocv_toy_permutation": {},
           "config": {"n_boot": args.n_boot, "n_perm": args.n_perm,
                      "seed": args.seed,
                      "underpowered_thresh": args.underpowered_thresh}}

    print(f"=== Global AUC + AP CIs (B={args.n_boot}, seed={args.seed}) ===")
    print(f"{'model':<12} {'N':>5} {'AUC':>7} {'95% CI':>20} {'AP':>7} {'95% CI':>20}")
    for name, path in models.items():
        rng_for_model = np.random.default_rng(args.seed + hash(name) % 10**6)
        out["global"][name] = _global_cis(path, args.n_boot, rng_for_model)
        g = out["global"][name]
        print(f"{name:<12} {g['n']:>5} {g['auc']:>7.4f} "
              f"[{g['auc_ci_lo']:.4f}, {g['auc_ci_hi']:.4f}] "
              f"{g['ap']:>7.4f} [{g['ap_ci_lo']:.4f}, {g['ap_ci_hi']:.4f}]")

    print(f"\n=== Per-repo CIs (CodeGemma, N_fixed >= {args.min_n_fixed_per_repo}) ===")
    rng_repo = np.random.default_rng(args.seed + 9999)
    out["per_repo_codegemma"] = _per_repo_cis(
        models["codegemma"], args.meta_dir, args.n_boot,
        args.underpowered_thresh, args.min_n_fixed_per_repo, rng_repo,
    )
    print(f"{'repo':<32} {'N_fix':>6} {'AUC':>7} {'95% CI':>20} {'underpowered':>13}")
    for r in out["per_repo_codegemma"]:
        flag = "*" if r["underpowered"] else " "
        print(f"{r['repo']:<32} {r['n_fixed']:>6} {r['auc']:>7.4f} "
              f"[{r['auc_ci_lo']:.4f}, {r['auc_ci_hi']:.4f}]  {flag}")

    print(f"\n=== Toy LOOCV permutation null (B={args.n_perm}) ===")
    if args.loocv_toy_npz.is_file():
        rng_perm = np.random.default_rng(args.seed + 12345)
        out["loocv_toy_permutation"] = _loocv_permutation_null(
            args.loocv_toy_npz, args.n_perm, rng_perm,
        )
        n = out["loocv_toy_permutation"]
        print(f"  observed AUC : {n['observed_auc']:.4f}")
        print(f"  null mean    : {n['null_mean']:.4f}")
        print(f"  null 97.5%   : {n['null_q975']:.4f}")
        print(f"  null max     : {n['null_max']:.4f}")
        print(f"  p-value      : {n['p_value']:.4f}  (n_perm={n['n_perm']})")
    else:
        print(f"  loocv npz not found at {args.loocv_toy_npz} — skipping null", file=sys.stderr)
        out["loocv_toy_permutation"] = {"error": f"missing: {args.loocv_toy_npz}"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
