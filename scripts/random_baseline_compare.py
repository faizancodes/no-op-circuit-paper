#!/usr/bin/env python3
"""Compare OMP top-8 ablation effect vs random-8 baseline distributions.

Loads OMP top-8 from results/sae/ablate-distributed/ablation_results.json
and per-seed random ablation results from one or more
results/sae/<baseline_dir>/seed_*/ablation_results.json hierarchies. Each
random seed JSON contains the same 296 paired buggy/fixed prompts with the
`ablate_random8` margin column.

For each random baseline directory, reports pooled per-task gap-reduction
statistics + one-sided Mann-Whitney U vs OMP top-8 paired-task reductions.

Default behaviour: compare BOTH `ablate-random` (random-any, original Tier 1
baseline) and `ablate-random-firing` (Tier 2 firing-only baseline). Writes
a summary JSON next to each baseline directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def paired_gaps(rows, lab: str):
    import numpy as np
    pairs: dict[str, dict] = {}
    for r in rows:
        pairs.setdefault(r["task_id"], {})[r["condition"]] = r
    complete = [p for p in pairs.values() if "buggy" in p and "fixed" in p]
    arr = np.array([p["buggy"]["margins"][lab] - p["fixed"]["margins"][lab]
                    for p in complete])
    return arr


def analyse_random_dir(d: Path, omp_red, omp_clean_mean: float, *, rng, n_boot: int):
    """Compute per-seed and pooled stats for one random-baseline directory."""
    import numpy as np
    from scipy.stats import mannwhitneyu

    seed_dirs = sorted(d.glob("seed_*"))
    if not seed_dirs:
        return None
    per_seed: list[dict] = []
    all_red: list[float] = []
    print(f"\n=== {d.name} ===")
    print(f"Found {len(seed_dirs)} seeds.")
    print(f"{'seed':<6} {'features':<35} {'red %':>8} {'95% CI':>20}")
    print("-" * 75)
    for sd in seed_dirs:
        blob = json.loads((sd / "ablation_results.json").read_text())
        seed = int(blob["seed"])
        feats = blob["feature_indices"]
        clean = paired_gaps(blob["rows"], "clean")
        ab = paired_gaps(blob["rows"], "ablate_random8")
        red = clean - ab
        red_pct = float(red.mean() / clean.mean() * 100)
        n = len(red)
        boot = np.array([red[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
        boot_pct = boot / clean.mean() * 100
        ci_lo = float(np.quantile(boot_pct, 0.025))
        ci_hi = float(np.quantile(boot_pct, 0.975))
        per_seed.append({"seed": seed, "features": [int(x) for x in feats],
                         "reduction_pct": red_pct, "ci95_pct": [ci_lo, ci_hi]})
        all_red.extend(red.tolist())
        feat_str = "[" + ", ".join(str(f) for f in feats[:6])
        feat_str += ", ...]" if len(feats) > 6 else "]"
        print(f"{seed:<6} {feat_str:<35} {red_pct:>+7.2f}% "
              f"[{ci_lo:>+6.2f}%, {ci_hi:>+6.2f}%]")

    arr = np.asarray(all_red)
    pooled_pct = float(arr.mean() / omp_clean_mean * 100)
    pooled_sd_pct = float(arr.std(ddof=1) / omp_clean_mean * 100)
    seed_means = np.array([s["reduction_pct"] for s in per_seed])
    boot_seed = np.array([
        seed_means[rng.integers(0, len(seed_means), size=len(seed_means))].mean()
        for _ in range(n_boot)
    ])
    seed_ci = (float(np.quantile(boot_seed, 0.025)),
               float(np.quantile(boot_seed, 0.975)))

    stat, mwu_p = mannwhitneyu(np.asarray(omp_red), arr, alternative="greater")
    if mwu_p < 0.01:
        verdict = ("OMP top-8 effect is significantly larger than this random baseline "
                   "→ specificity established at this control level.")
    elif mwu_p < 0.05:
        verdict = "OMP top-8 effect is marginally larger than this random baseline."
    else:
        verdict = ("OMP top-8 effect is NOT significantly larger than this random baseline "
                   "— specificity claim does NOT hold at this control level; reframe honestly.")

    print()
    print(f"Pooled across {len(seed_dirs)} seeds × {len(omp_red)} tasks:")
    print(f"  random mean reduction = {arr.mean():+.4f} logits ({pooled_pct:+.2f}%)")
    print(f"  paired-task sd        = {pooled_sd_pct:.2f}% of OMP clean gap")
    print(f"  across-seed mean      = {float(seed_means.mean()):+.2f}% "
          f"(95% CI [{seed_ci[0]:+.2f}%, {seed_ci[1]:+.2f}%])")
    print(f"  Mann-Whitney U vs OMP: U={stat:.0f}, p={mwu_p:.3e}")
    print(f"  → {verdict}")

    return {
        "dir": d.name,
        "n_seeds": len(seed_dirs),
        "pooled_reduction_pct": pooled_pct,
        "across_seed_mean_pct": float(seed_means.mean()),
        "across_seed_ci95_pct": list(seed_ci),
        "paired_sd_pct": pooled_sd_pct,
        "mannwhitney_U": float(stat),
        "mannwhitney_p_one_sided": float(mwu_p),
        "verdict": verdict,
        "per_seed": per_seed,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--omp",
                   type=Path,
                   default=Path("results/sae/ablate-distributed/ablation_results.json"))
    p.add_argument("--random-dirs",
                   type=Path, nargs="+",
                   default=[Path("results/sae/ablate-random"),
                            Path("results/sae/ablate-random-firing")])
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import numpy as np

    omp_blob = json.loads(args.omp.read_text())
    omp_clean = paired_gaps(omp_blob["rows"], "clean")
    omp_ab = paired_gaps(omp_blob["rows"], "ablate_omp_top8")
    omp_clean_mean = float(omp_clean.mean())
    omp_red = omp_clean - omp_ab
    print(f"OMP top-8: clean gap = {omp_clean_mean:+.4f}, "
          f"reduction = {omp_red.mean():+.4f} logits "
          f"({omp_red.mean()/omp_clean_mean*100:+.2f}%)")

    rng = np.random.default_rng(args.seed)
    summaries: dict[str, dict] = {}
    for d in args.random_dirs:
        if not d.exists():
            print(f"\nNOTE: {d} does not exist; skipping.")
            continue
        result = analyse_random_dir(d, omp_red, omp_clean_mean, rng=rng,
                                    n_boot=args.n_boot)
        if result is None:
            continue
        summaries[d.name] = result
        out = d / "summary.json"
        out.write_text(json.dumps({
            "omp_top8_reduction_pct": float(omp_red.mean()/omp_clean_mean*100),
            "omp_clean_gap_mean": omp_clean_mean,
            **result,
        }, indent=2))
        print(f"  wrote {out}")

    if len(summaries) >= 2:
        print("\n=== 3-row specificity table ===")
        print(f"{'set':<30} {'reduction':>12} {'across-seed CI':>22}"
              f" {'Mann-Whitney p':>16}")
        print("-" * 85)
        omp_pct = float(omp_red.mean() / omp_clean_mean * 100)
        print(f"{'OMP top-8 (Tier 1, distributed)':<30} {omp_pct:>+11.2f}% "
              f"{'—':>22} {'—':>16}")
        for name, r in summaries.items():
            ci = r["across_seed_ci95_pct"]
            print(f"{name:<30} {r['across_seed_mean_pct']:>+11.2f}% "
                  f"[{ci[0]:>+6.2f}%, {ci[1]:>+6.2f}%]   {r['mannwhitney_p_one_sided']:>15.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
