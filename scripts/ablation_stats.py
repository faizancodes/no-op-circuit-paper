#!/usr/bin/env python3
"""Bootstrap CIs + Wilcoxon for the OMP ablation reductions in §5.2.

Reads results/sae/ablate-distributed/ablation_results.json, pairs by task_id,
computes the (buggy − fixed) margin gap per task per condition, then for each
ablation set reports:
  - mean reduction vs clean (logits and %)
  - 95% bootstrap CI on the reduction % (B=10000, seed=0)
  - one-sided Wilcoxon signed-rank p (clean > ablated, per-task)

Numbers must match the §5.2 ablation table; this script is the canonical
recomputation referenced from there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ablation",
                   type=Path,
                   default=Path("results/sae/ablate-distributed/ablation_results.json"))
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import numpy as np
    from scipy.stats import wilcoxon

    blob = json.loads(args.ablation.read_text())
    pairs: dict[str, dict] = {}
    for r in blob["rows"]:
        pairs.setdefault(r["task_id"], {})[r["condition"]] = r
    complete = [p_ for p_ in pairs.values() if "buggy" in p_ and "fixed" in p_]
    n = len(complete)
    print(f"Paired tasks: {n}")

    def gaps(lab: str) -> "np.ndarray":
        return np.array([p_["buggy"]["margins"][lab] - p_["fixed"]["margins"][lab]
                         for p_ in complete])

    clean = gaps("clean")
    clean_mean = float(clean.mean())
    print(f"\nClean gap: mean={clean_mean:+.4f}  sem={clean.std(ddof=1)/np.sqrt(n):.4f}\n")

    rng = np.random.default_rng(args.seed)
    rows_out = []
    labels = [lab for lab in blob["rows"][0]["margins"].keys()
              if lab not in ("clean",)]
    print(f"{'set':<22} {'red %':>8} {'95% CI low':>10} {'95% CI hi':>10} {'Wilcoxon p':>13}")
    print("-" * 70)
    for lab in labels:
        ab = gaps(lab)
        diffs = clean - ab  # positive ⇒ ablation reduced the gap
        diff_mean_pct = float(diffs.mean() / clean_mean * 100)

        boot_means = np.empty(args.n_boot)
        for b in range(args.n_boot):
            idx = rng.integers(0, n, size=n)
            boot_means[b] = diffs[idx].mean()
        boot_pct = boot_means / clean_mean * 100
        ci_lo = float(np.quantile(boot_pct, 0.025))
        ci_hi = float(np.quantile(boot_pct, 0.975))

        # One-sided Wilcoxon: alternative='greater' tests clean > ab per task
        stat, pval = wilcoxon(diffs, alternative="greater")
        rows_out.append({
            "label": lab,
            "reduction_logits": float(diffs.mean()),
            "reduction_pct": diff_mean_pct,
            "ci95_pct": [ci_lo, ci_hi],
            "wilcoxon_stat": float(stat),
            "wilcoxon_p": float(pval),
            "n_paired": n,
            "n_boot": args.n_boot,
        })
        print(f"{lab:<22} {diff_mean_pct:>+7.2f}% {ci_lo:>+9.2f}% {ci_hi:>+9.2f}% {pval:>13.2e}")

    out = args.ablation.parent / "ablation_stats.json"
    out.write_text(json.dumps({"clean_gap_mean": clean_mean, "rows": rows_out}, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
