#!/usr/bin/env python3
"""Layer-sweep AUC curve on Qwen.

The App. G.2 adversarial-baselines table reports wrong-layer AUC at
L12/pos −1 (0.998). This script extends that to the full 28-layer
curve, two ways:

  (A) FROZEN-direction transfer. Project the real-task residuals at
      EVERY layer onto the *same* L24-trained v_noop unit vector.
      AUC vs layer for the toy direction.

  (B) PER-LAYER derived direction. At each layer L, derive
        v_noop_L = mean(fixed_toy_resid[L]) - mean(buggy_toy_resid[L])
      from the 49 paired toy tasks, then project the real-task
      residuals at L onto v_noop_L. AUC vs layer for a freshly-
      contrastive direction.

This visualises App. G.2's claim that the discriminative signal is
broadly distributed across layers and not specifically a feature of
L24 — which §4.1's patching peak identifies separately.

Output: results/monitor_real/layer_sweep_auc.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_paired_at_layer(cache_dir: Path, layer: int, pos: int, variant: str = "code_tests"):
    """Return (task_ids, buggy_NxD, fixed_NxD)."""
    import numpy as np
    import torch
    by_task: dict[str, dict] = {}
    pattern = f"*__{variant}.pt"
    for pt in sorted(cache_dir.rglob(pattern)):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        T = int(payload["resid_pre"].shape[2])
        abs_pos = pos if pos >= 0 else T + pos
        vec = payload["resid_pre"][layer, 0, abs_pos, :].float().numpy()
        by_task.setdefault(payload["task_id"], {})[cond] = vec
    paired = sorted(t for t, s in by_task.items() if "buggy" in s and "fixed" in s)
    buggy = np.stack([by_task[t]["buggy"] for t in paired])
    fixed = np.stack([by_task[t]["fixed"] for t in paired])
    return paired, buggy, fixed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--toy-cache", type=Path,
                   default=Path("results/cache-20260515T221105Z"))
    p.add_argument("--real-cache", type=Path,
                   default=Path("results/cache-real-qwen-n500-20260516T235301Z"))
    p.add_argument("--v-noop", type=Path,
                   default=Path("results/steer-20260516T021522Z/v_noop.pt"))
    p.add_argument("--n-layers", type=int, default=28)
    p.add_argument("--pos", type=int, default=-1)
    p.add_argument("--out", type=Path,
                   default=Path("results/monitor_real/layer_sweep_auc.json"))
    args = p.parse_args(argv)

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    # Resolve nested-subdir caches
    toy_inner = args.toy_cache / args.toy_cache.name
    toy_root = toy_inner if toy_inner.exists() else args.toy_cache
    real_inner = args.real_cache / args.real_cache.name
    real_root = real_inner if real_inner.exists() else args.real_cache

    # Load the canonical (L24) v_noop
    v_blob = torch.load(args.v_noop, map_location="cpu", weights_only=False)
    v_canon = v_blob["direction"].float().numpy()
    v_canon_unit = v_canon / np.linalg.norm(v_canon)
    print(f"v_noop (canonical): L{v_blob['layer']}/pos {v_blob['position']}")

    rows = []
    print(f"\n{'layer':>5} {'fresh AUC':>11} {'frozen AUC':>11} {'|v_L|':>8}")
    print("-" * 50)
    for L in range(args.n_layers):
        # Toy residuals at this layer: derive a fresh v_noop_L
        _, toy_b, toy_f = _load_paired_at_layer(toy_root, L, args.pos, "code_tests")
        v_L = toy_f.mean(axis=0) - toy_b.mean(axis=0)
        v_L_norm = float(np.linalg.norm(v_L))
        v_L_unit = v_L / max(v_L_norm, 1e-12)

        # Real residuals at this layer
        _, real_b, real_f = _load_paired_at_layer(real_root, L, args.pos, "code_tests")
        N = real_b.shape[0]
        scores_fresh = np.concatenate([-(real_b @ v_L_unit), -(real_f @ v_L_unit)])
        scores_frozen = np.concatenate([-(real_b @ v_canon_unit), -(real_f @ v_canon_unit)])
        labels = np.concatenate([np.ones(N, int), np.zeros(N, int)])

        auc_fresh = float(roc_auc_score(labels, scores_fresh))
        auc_frozen = float(roc_auc_score(labels, scores_frozen))
        rows.append({
            "layer": L,
            "v_noop_L_norm": v_L_norm,
            "auc_fresh_v_noop_L": auc_fresh,
            "auc_frozen_v_noop_L24": auc_frozen,
            "n_real_tasks": int(N),
        })
        marker = "  ← L24 (causal peak)" if L == v_blob["layer"] else ""
        print(f"{L:>5d} {auc_fresh:>11.4f} {auc_frozen:>11.4f} {v_L_norm:>8.3f}{marker}")

    # Summary
    fresh = [r["auc_fresh_v_noop_L"] for r in rows]
    frozen = [r["auc_frozen_v_noop_L24"] for r in rows]
    summary = {
        "fresh_max_auc": max(fresh),
        "fresh_max_layer": int(np.argmax(fresh)),
        "frozen_max_auc": max(frozen),
        "frozen_max_layer": int(np.argmax(frozen)),
        "fresh_at_L24": fresh[v_blob["layer"]],
        "frozen_at_L24": frozen[v_blob["layer"]],
    }
    print()
    print(f"Fresh v_noop_L peak: AUC {summary['fresh_max_auc']:.4f} at L{summary['fresh_max_layer']}")
    print(f"Frozen L24-trained v_noop peak: AUC {summary['frozen_max_auc']:.4f} at L{summary['frozen_max_layer']}")
    print(f"At causal-patching site L24: fresh {summary['fresh_at_L24']:.4f}, frozen {summary['frozen_at_L24']:.4f}")

    out = {
        "config": {
            "toy_cache": str(args.toy_cache),
            "real_cache": str(args.real_cache),
            "v_noop": str(args.v_noop),
            "n_layers": args.n_layers,
            "pos": args.pos,
            "n_real_tasks": rows[0]["n_real_tasks"],
        },
        "rows": rows,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
