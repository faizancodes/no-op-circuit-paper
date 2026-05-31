#!/usr/bin/env python3
"""Sample-efficiency curve for v_noop on Qwen2.5-Coder-1.5B.

The full v_noop direction used in §5.1 is derived from all 49 toy
paired tasks. A natural reviewer question: "Would 5 tasks have
sufficed? 10? Where does the curve saturate?"

Protocol
--------
For each N in {1, 5, 10, 25, 49}:
  For each seed s in 0..K-1 (K=10, except K=1 at N=49 since the
  deterministic full set is the only sample):
    1. Draw N paired-task indices WITHOUT replacement from the 49 toys.
    2. Compute v_noop_sub = mean(fixed_residuals[indices]) -
                            mean(buggy_residuals[indices])  at L24/pos -1.
    3. Unit-normalise v_noop_sub.
    4. Project every SWE-bench Verified residual at the same cell onto
       v_noop_sub_unit; score = -projection (matches existing convention
       so higher score = more buggy-like).
    5. Compute ROC-AUC and AP against the (buggy=1, fixed=0) labels.

The evaluation set is the full 499-instance SWE-bench Verified cache
the headline AUC 0.989 is reported on. The ONLY source of stochasticity
is the toy subsample — projection + AUC are deterministic.

Output
------
results/monitor_real/sample_efficiency.json
  { "config": ..., "rows": [ {"n": N, "seed": s, "auc": ..., "ap": ...} ] }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_paired_at_cell(cache_dir: Path, layer: int, pos: int):
    """Return (task_ids, buggy_resid, fixed_resid) — paired tasks only."""
    import numpy as np
    import torch

    by_task: dict[str, dict[str, "np.ndarray"]] = {}
    for pt in sorted(cache_dir.rglob("*__code_tests.pt")):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        T = int(blob["resid_pre"].shape[2])
        abs_pos = pos if pos >= 0 else T + pos
        vec = blob["resid_pre"][layer, 0, abs_pos, :].float().numpy()
        by_task.setdefault(blob["task_id"], {})[cond] = vec
    paired = [(tid, d["buggy"], d["fixed"])
              for tid, d in by_task.items()
              if "buggy" in d and "fixed" in d]
    paired.sort(key=lambda r: r[0])
    task_ids = [r[0] for r in paired]
    buggy = np.stack([r[1] for r in paired])
    fixed = np.stack([r[2] for r in paired])
    return task_ids, buggy, fixed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--toy-cache-dir",
        type=Path,
        default=Path("results/cache-20260515T221105Z/cache-20260515T221105Z"),
        help="Qwen toy substrate cache root (per-task subdirs).",
    )
    p.add_argument(
        "--real-cache-dir",
        type=Path,
        default=None,
        help="SWE-bench Verified cache root (default: most recent cache-real-qwen-n500-*).",
    )
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--pos", type=int, default=-1)
    p.add_argument("--ns", type=str, default="1,5,10,25,49",
                   help="Comma-separated subsample sizes.")
    p.add_argument("--n-seeds", type=int, default=10,
                   help="Random seeds per N (except N=full-set, which is K=1).")
    p.add_argument("--seed", type=int, default=0,
                   help="Master seed; per-(N,replicate) seeds are derived deterministically.")
    p.add_argument("--out", type=Path,
                   default=Path("results/monitor_real/sample_efficiency.json"))
    args = p.parse_args(argv)

    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    # Pin RNG state
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.real_cache_dir is None:
        cands = sorted(Path("results").glob("cache-real-qwen-n500-*"),
                       key=lambda p_: p_.name, reverse=True)
        if not cands:
            print("no cache-real-qwen-n500-* under results/", file=sys.stderr)
            return 2
        inner = cands[0] / cands[0].name
        args.real_cache_dir = inner if inner.exists() else cands[0]

    print(f"toy_cache_dir : {args.toy_cache_dir}")
    print(f"real_cache_dir: {args.real_cache_dir}")
    print(f"layer/pos     : L{args.layer} / pos {args.pos:+d}")
    print(f"sample sizes  : {args.ns}")
    print(f"seeds per N   : {args.n_seeds} (K=1 at N=full-set)")
    print()

    # --- Load toy residuals (paired) ---
    print("loading toy residuals…")
    toy_ids, toy_buggy, toy_fixed = _load_paired_at_cell(
        args.toy_cache_dir, args.layer, args.pos)
    n_toy = len(toy_ids)
    print(f"  loaded {n_toy} paired toy tasks; D = {toy_buggy.shape[1]}")

    # --- Load SWE-bench residuals (paired) + build labels ---
    print("loading SWE-bench residuals…")
    real_ids, real_buggy, real_fixed = _load_paired_at_cell(
        args.real_cache_dir, args.layer, args.pos)
    n_real = len(real_ids)
    print(f"  loaded {n_real} paired SWE-bench tasks")

    # Score = -projection so HIGHER = more buggy-like (matches monitor)
    real_X = np.concatenate([real_buggy, real_fixed], axis=0)
    real_y = np.concatenate(
        [np.ones(n_real, dtype=int), np.zeros(n_real, dtype=int)]
    )

    # --- Sample sizes ---
    Ns = sorted({int(x.strip()) for x in args.ns.split(",") if x.strip()})
    Ns = [n for n in Ns if 1 <= n <= n_toy]
    if not Ns:
        print(f"no valid Ns in {args.ns} given {n_toy} toys", file=sys.stderr)
        return 2

    rows: list[dict] = []
    summary: dict[int, dict] = {}

    print()
    print(f"{'N':>4} {'seed':>5} {'auc':>8} {'ap':>8}")
    print("-" * 30)
    for N in Ns:
        if N >= n_toy:
            k_seeds = 1
        else:
            k_seeds = args.n_seeds
        per_n_aucs: list[float] = []
        per_n_aps: list[float] = []
        for replicate in range(k_seeds):
            # Derive a per-(N, replicate) RNG via SeedSequence so output is
            # fully reproducible AND type-clean across numpy versions.
            ss = np.random.SeedSequence([args.seed, N, replicate])
            rng = np.random.default_rng(ss)
            if N >= n_toy:
                idx = np.arange(n_toy)
            else:
                idx = rng.choice(n_toy, size=N, replace=False)
            v = toy_fixed[idx].mean(axis=0) - toy_buggy[idx].mean(axis=0)
            v_unit = v / max(float(np.linalg.norm(v)), 1e-12)
            scores = -(real_X @ v_unit)
            auc = float(roc_auc_score(real_y, scores))
            ap = float(average_precision_score(real_y, scores))
            rows.append({"n": N, "seed": replicate,
                         "task_ids": [toy_ids[i] for i in idx.tolist()],
                         "v_norm": float(np.linalg.norm(v)),
                         "auc": auc, "ap": ap})
            per_n_aucs.append(auc); per_n_aps.append(ap)
            print(f"{N:>4} {replicate:>5} {auc:>8.4f} {ap:>8.4f}")
        aucs = np.asarray(per_n_aucs)
        aps = np.asarray(per_n_aps)
        summary[N] = {
            "n_replicates": int(k_seeds),
            "auc_mean": float(aucs.mean()),
            "auc_std": float(aucs.std(ddof=1)) if k_seeds > 1 else 0.0,
            "auc_min": float(aucs.min()),
            "auc_max": float(aucs.max()),
            "ap_mean": float(aps.mean()),
            "ap_std": float(aps.std(ddof=1)) if k_seeds > 1 else 0.0,
        }
        print(f"  → mean AUC {aucs.mean():.4f} ± {aucs.std(ddof=1) if k_seeds>1 else 0:.4f}"
              f"  (min {aucs.min():.4f}, max {aucs.max():.4f})")

    out = {
        "config": {
            "toy_cache_dir": str(args.toy_cache_dir),
            "real_cache_dir": str(args.real_cache_dir),
            "layer": args.layer, "pos": args.pos,
            "n_toy_tasks": n_toy,
            "n_real_tasks": n_real,
            "sample_sizes": Ns,
            "n_seeds": args.n_seeds,
            "master_seed": args.seed,
        },
        "summary": {str(k): v for k, v in summary.items()},
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
