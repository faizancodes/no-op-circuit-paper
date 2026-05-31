#!/usr/bin/env python3
"""Multiple-testing-corrected permutation null for the Qwen patching grid.

Closes Important-I1 from the pre-submission audit. App. D selects the
peak F→B mean shift over a (layer × position) grid. A reviewer's
natural pre-empt: "How do you know L24/pos −1 isn't just the
multiple-testing winner of ~28 cells of sampling noise?"

We answer with the canonical max-statistic permutation test. Under H0
("the buggy/fixed labels carry no systematic information at any cell"),
the F→B per-task shift at each cell is symmetric around 0, so
multiplying each task's shift vector by a random ±1 sign vector is
exchangeable. The null distribution of the grid maximum, sampled by
repeating this many times, naturally accounts for the multiple-
testing burden because we always take max across all 28 cells.

p-value: fraction of permutations whose null max ≥ observed max
(plus the +1/+1 small-sample correction).

Inputs
------
results/patch-20260516T005329Z/aggregated.npz
  Per-task F→B shifts: `shift_per_pair_f2b`, shape (N_tasks, L, P)
  Cell coords:         `layer_indices` (L,), `position_offsets` (P,)

Output
------
results/patch_grid_permutation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz",
        type=Path,
        default=Path("results/patch-20260516T005329Z/aggregated.npz"),
    )
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/patch_grid_permutation.json"),
    )
    args = p.parse_args(argv)

    import numpy as np

    rng = np.random.default_rng(args.seed)

    blob = np.load(args.npz, allow_pickle=True)
    shifts = np.asarray(blob["shift_per_pair_f2b"], dtype=float)
    layers = list(blob["layer_indices"])
    positions = list(blob["position_offsets"])
    n_tasks, n_layers, n_positions = shifts.shape
    n_cells = n_layers * n_positions

    observed_mean = shifts.mean(axis=0)
    observed_max = float(observed_mean.max())
    peak_idx = np.unravel_index(observed_mean.argmax(), observed_mean.shape)
    peak_layer = int(layers[peak_idx[0]])
    peak_pos = int(positions[peak_idx[1]])

    print(f"loaded:     {args.npz}")
    print(f"grid:       {n_layers} layers × {n_positions} positions = {n_cells} cells")
    print(f"n_tasks:    {n_tasks}")
    print(f"observed max = {observed_max:+.4f}  at L{peak_layer}/pos {peak_pos:+d}")
    print(f"\nrunning {args.n_perm} sign-flip permutations (seed={args.seed})…")

    null_max = np.empty(args.n_perm, dtype=float)
    for b in range(args.n_perm):
        # ±1 per task, independent across tasks
        signs = rng.choice([-1.0, 1.0], size=n_tasks)
        permuted = shifts * signs[:, None, None]
        null_max[b] = permuted.mean(axis=0).max()

    # +1 / +1 corrected empirical p-value (Phipson & Smyth 2010).
    n_ge = int((null_max >= observed_max).sum())
    p_value = (n_ge + 1) / (args.n_perm + 1)
    null_q975 = float(np.quantile(null_max, 0.975))
    null_q99 = float(np.quantile(null_max, 0.99))
    null_max_observed = float(null_max.max())
    null_mean = float(null_max.mean())
    null_std = float(null_max.std(ddof=1))

    out = {
        "config": {
            "npz": str(args.npz),
            "n_perm": args.n_perm,
            "seed": args.seed,
            "test_statistic": "max over grid of mean F→B shift",
            "permutation": "independent ±1 sign-flip per task",
        },
        "grid": {
            "n_layers": n_layers,
            "n_positions": n_positions,
            "n_cells": n_cells,
            "layers": [int(l) for l in layers],
            "positions": [int(p_) for p_ in positions],
            "n_tasks": n_tasks,
        },
        "observed": {
            "max_mean_shift": observed_max,
            "peak_layer": peak_layer,
            "peak_position": peak_pos,
        },
        "null": {
            "mean": null_mean,
            "std": null_std,
            "q975": null_q975,
            "q99": null_q99,
            "max": null_max_observed,
            "n_ge_observed": n_ge,
        },
        "p_value": p_value,
    }

    print("\n=== max-statistic permutation null ===")
    print(f"  null mean :  {null_mean:.4f}")
    print(f"  null std  :  {null_std:.4f}")
    print(f"  null 97.5%:  {null_q975:.4f}")
    print(f"  null 99%  :  {null_q99:.4f}")
    print(f"  null max  :  {null_max_observed:.4f}")
    print(f"  n ≥ obs   :  {n_ge}/{args.n_perm}")
    print(f"  p-value   :  {p_value:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
