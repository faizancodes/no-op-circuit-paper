#!/usr/bin/env python3
"""Investigate the CodeGemma N=99 → N=499 monitor scaling regression.

Phase A revealed that on the full SWE-bench Verified, the CodeGemma monitor's
ROC-AUC dropped from 0.958 (first 99 instances) to 0.933, and the false-edit
rate at the operating point rose from 9.1% → 15.5% (77/497 fixed-condition
tasks misclassified as 'predict edit'). This script characterises that
regression — what kinds of tasks drive it — using only the existing N=499/497
caches (no Modal compute).

Outputs `results/monitor_real/codegemma_regression.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _safe_load_yaml(p: Path):
    try:
        import yaml
        return yaml.safe_load(p.read_text())
    except Exception:
        return None


def _repo_of(task_id: str, meta_dir: Path) -> str | None:
    m = _safe_load_yaml(meta_dir / task_id / "meta.yaml")
    return (m or {}).get("swe_bench_repo") if isinstance(m, dict) else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qwen",
                   type=Path,
                   default=Path("results/monitor_real/real_curves_qwen_n500.npz"))
    p.add_argument("--cg",
                   type=Path,
                   default=Path("results/monitor_real/real_curves_codegemma_n500.npz"))
    p.add_argument("--meta-dir", type=Path, default=Path("data/real_tasks"))
    p.add_argument("--cg-cache",
                   type=Path,
                   default=None,
                   help="dir with codegemma fixed__code_tests.pt cache (for token-length analysis)")
    p.add_argument("--out",
                   type=Path,
                   default=Path("results/monitor_real/codegemma_regression.json"))
    args = p.parse_args(argv)

    import numpy as np
    from scipy.stats import mannwhitneyu

    if args.cg_cache is None:
        cands = sorted(Path("results").glob("cache-real-codegemma-n500-*"),
                       key=lambda p: p.name, reverse=True)
        if cands:
            inner = cands[0] / cands[0].name
            args.cg_cache = inner if inner.exists() else cands[0]

    # Load monitor outputs
    qwen = np.load(args.qwen, allow_pickle=True)
    cg = np.load(args.cg, allow_pickle=True)
    q_thr = float(qwen["op_threshold"])
    c_thr = float(cg["op_threshold"])
    print(f"Op-thresholds  Qwen: {q_thr:+.4f}  CodeGemma: {c_thr:+.4f}")
    print(f"AUC            Qwen: {float(qwen['roc_auc']):.4f}  "
          f"CodeGemma: {float(cg['roc_auc']):.4f}")

    # ---- Build per-(task_id, condition) lookup ----
    def by_key(blob):
        return {(str(blob["task_ids"][i]), str(blob["conditions"][i])):
                {"score": float(blob["scores"][i]), "label": int(blob["labels"][i]),
                 "raw_proj": float(blob["raw_projections"][i])}
                for i in range(len(blob["task_ids"]))}

    q_by = by_key(qwen); c_by = by_key(cg)
    common_keys = set(q_by) & set(c_by)
    fixed_keys = [k for k in common_keys if k[1] == "fixed"]
    print(f"Paired fixed-condition tasks across both monitors: {len(fixed_keys)}")

    # ---- 1. Misclassified fixed-condition tasks per model ----
    # label=0 (fixed); predicted_buggy = score >= op_threshold (the "predict buggy" rule).
    q_misclass = [k for k in fixed_keys if q_by[k]["score"] >= q_thr]
    c_misclass = [k for k in fixed_keys if c_by[k]["score"] >= c_thr]
    print(f"\nFalse-edit rate at op-point:")
    print(f"  Qwen:      {len(q_misclass):>3}/{len(fixed_keys)} = "
          f"{len(q_misclass)/len(fixed_keys)*100:.2f}%")
    print(f"  CodeGemma: {len(c_misclass):>3}/{len(fixed_keys)} = "
          f"{len(c_misclass)/len(fixed_keys)*100:.2f}%")

    # ---- 2. Per-repo breakdown ----
    repo_of_task: dict[str, str | None] = {}
    for tid in {k[0] for k in fixed_keys}:
        repo_of_task[tid] = _repo_of(tid, args.meta_dir)
    repo_total: Counter[str] = Counter()
    repo_cg_misclass: Counter[str] = Counter()
    repo_q_misclass: Counter[str] = Counter()
    for k in fixed_keys:
        r = repo_of_task.get(k[0]) or "(unknown)"
        repo_total[r] += 1
        if k in set(c_misclass): repo_cg_misclass[r] += 1
        if k in set(q_misclass): repo_q_misclass[r] += 1

    print(f"\n=== CodeGemma false-edit rate per repo (sorted by absolute count) ===")
    print(f"{'repo':<32} {'fixed':>6} {'CG miss':>8} {'CG rate':>8} {'Qwen miss':>10} {'Qwen rate':>10}")
    print("-" * 82)
    rep_rows = []
    for r in sorted(repo_total, key=lambda r: -repo_cg_misclass.get(r, 0)):
        if repo_total[r] < 2:
            continue
        cg_n = repo_cg_misclass.get(r, 0)
        q_n = repo_q_misclass.get(r, 0)
        cg_rate = cg_n / repo_total[r]
        q_rate = q_n / repo_total[r]
        rep_rows.append({"repo": r, "n_fixed": repo_total[r],
                          "cg_misclass": cg_n, "cg_rate": cg_rate,
                          "q_misclass": q_n, "q_rate": q_rate})
        print(f"{r:<32} {repo_total[r]:>6} {cg_n:>8} {cg_rate*100:>7.1f}% "
              f"{q_n:>10} {q_rate*100:>9.1f}%")

    # ---- 3. Qwen-vs-CodeGemma overlap on misclassifications ----
    overlap = set(q_misclass) & set(c_misclass)
    cg_only = set(c_misclass) - set(q_misclass)
    q_only  = set(q_misclass) - set(c_misclass)
    print(f"\n=== Misclassification overlap (fixed-condition false-edits) ===")
    print(f"  both models misclassify : {len(overlap):>3} of {len(c_misclass)} CG cases "
          f"({len(overlap)/max(len(c_misclass),1)*100:.1f}% of CG misses, "
          f"{len(overlap)/max(len(q_misclass),1)*100:.1f}% of Qwen misses)")
    print(f"  CodeGemma-only miss     : {len(cg_only):>3}")
    print(f"  Qwen-only miss          : {len(q_only):>3}")

    # ---- 4. Score-distribution histogram of CG misclassifications ----
    cg_miss_scores = np.array([c_by[k]["score"] for k in c_misclass])
    cg_correct_fixed_scores = np.array([c_by[k]["score"] for k in fixed_keys
                                         if k not in set(c_misclass)])
    print(f"\n=== CodeGemma score distribution (op-thr = {c_thr:+.3f}) ===")
    print(f"  misclassified (above thr): mean={cg_miss_scores.mean():+.3f} "
          f"std={cg_miss_scores.std():.3f} min={cg_miss_scores.min():+.3f} "
          f"max={cg_miss_scores.max():+.3f}")
    print(f"  correctly classified     : mean={cg_correct_fixed_scores.mean():+.3f} "
          f"std={cg_correct_fixed_scores.std():.3f}")
    print(f"  margin above threshold among the misclassified:")
    bins = [(0, 1), (1, 2), (2, 5), (5, 10), (10, float("inf"))]
    for lo, hi in bins:
        above = cg_miss_scores - c_thr
        n = int(((above >= lo) & (above < hi)).sum())
        label = f"  +[{lo:>4.1f}, {hi:>5})" if hi != float("inf") else f"  +[{lo:>4.1f},  inf)"
        print(f"{label}: {n:>3} ({n/max(len(cg_miss_scores),1)*100:.1f}%)")

    # ---- 5. Token-length correlation ----
    tok_misclass: list[int] = []
    tok_correct: list[int] = []
    if args.cg_cache and args.cg_cache.exists():
        import torch
        for tid in {k[0] for k in fixed_keys}:
            pt = args.cg_cache / tid / "fixed__code_tests.pt"
            if not pt.exists():
                continue
            try:
                pl = torch.load(pt, map_location="cpu", weights_only=False)
            except Exception:
                continue
            n_tok = int(pl["seq_len"])
            if (tid, "fixed") in set(c_misclass):
                tok_misclass.append(n_tok)
            else:
                tok_correct.append(n_tok)
        if tok_misclass and tok_correct:
            tm = np.array(tok_misclass); tc = np.array(tok_correct)
            stat, p_mwu = mannwhitneyu(tm, tc, alternative="two-sided")
            print(f"\n=== Token-length comparison (CG fixed-condition) ===")
            print(f"  misclassified : mean={tm.mean():.0f}  median={float(np.median(tm)):.0f}  n={len(tm)}")
            print(f"  correctly     : mean={tc.mean():.0f}  median={float(np.median(tc)):.0f}  n={len(tc)}")
            print(f"  Mann-Whitney U two-sided p = {p_mwu:.3e}")

    # ---- Save ----
    out = {
        "n_fixed_paired": len(fixed_keys),
        "op_threshold_qwen": q_thr,
        "op_threshold_codegemma": c_thr,
        "n_misclass_qwen": len(q_misclass),
        "n_misclass_codegemma": len(c_misclass),
        "overlap_both_miss": len(overlap),
        "overlap_pct_of_cg_miss": len(overlap)/max(len(c_misclass),1)*100,
        "cg_only_miss": len(cg_only),
        "qwen_only_miss": len(q_only),
        "per_repo": rep_rows,
        "cg_misclass_score_stats": {
            "mean": float(cg_miss_scores.mean()),
            "std": float(cg_miss_scores.std()),
            "min": float(cg_miss_scores.min()),
            "max": float(cg_miss_scores.max()),
        },
        "cg_correct_score_stats": {
            "mean": float(cg_correct_fixed_scores.mean()),
            "std": float(cg_correct_fixed_scores.std()),
        },
        "token_length": {
            "misclass_mean": float(np.mean(tok_misclass)) if tok_misclass else None,
            "correct_mean": float(np.mean(tok_correct)) if tok_correct else None,
            "n_misclass": len(tok_misclass),
            "n_correct": len(tok_correct),
        },
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
