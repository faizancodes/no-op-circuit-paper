#!/usr/bin/env python3
"""Train layerwise probes on a downloaded activation-cache run.

Pre-req: download the run from the activations Volume, e.g.

    modal volume get noop-activations <run_id>/  results/<run_id>/

Then:

    python scripts/train_probes.py results/<run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from no_op_circuit.interp.probes import load_run, results_to_json, train_layerwise


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--variant", default=None, help="Restrict to one variant.")
    p.add_argument("--positions", default="-1",
                   help="Comma-separated positions to probe (negative = from end). Default -1 (action pos).")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional JSON file to write per-(variant, layer, position) AUCs.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for numpy/torch/random and scikit-learn CV splits.")
    args = p.parse_args(argv)

    import random
    import numpy as np
    try:
        import torch
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    np.random.seed(args.seed)
    random.seed(args.seed)

    positions = [int(x) for x in args.positions.split(",") if x.strip()]

    datasets = load_run(args.run_dir, variant_filter=args.variant)
    if not datasets:
        print(f"no activation files found under {args.run_dir}", file=sys.stderr)
        return 2

    all_results = []
    for variant, ds in datasets.items():
        print(f"\n=== variant: {variant} (N={len(ds.task_ids)} paired tasks, "
              f"L={ds.n_layers}, K={ds.n_positions}, D={ds.hidden}) ===")
        if len(ds.task_ids) < 4:
            print(f"  too few paired tasks (need >=4 for 2-fold split); skipping.")
            continue
        results = train_layerwise(ds, positions=positions)
        all_results.extend(results)
        # Compact print: best (layer, position) by AUC
        results.sort(key=lambda r: -r.auc)
        print(f"  top 5 (layer, position, AUC):")
        for r in results[:5]:
            print(f"    L{r.layer:>2}  pos={r.position:>3}  AUC={r.auc:.3f}")
        # Mean AUC at the action position, by layer.
        for pos in positions:
            pos_results = [r for r in results if r.position == pos]
            if pos_results:
                pos_results.sort(key=lambda r: r.layer)
                print(f"  per-layer AUC at pos {pos}:")
                # show every layer with a small bar chart
                for r in pos_results:
                    bar = "█" * int((r.auc - 0.5) * 60) if r.auc > 0.5 else ""
                    print(f"    L{r.layer:>2}  {r.auc:.3f}  {bar}")

    if args.out is not None:
        args.out.write_text(results_to_json(all_results))
        print(f"\nwrote {len(all_results)} probe results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
