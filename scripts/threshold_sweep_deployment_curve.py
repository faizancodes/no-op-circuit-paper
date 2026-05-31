#!/usr/bin/env python3
"""§G.10 threshold-sweep deployment curve.

The §G.10 single-turn agent-loop simulation reported two operating
points per model (in-sample and held-out 50/50 mean). This script
sweeps the monitor threshold across the full range and reports the
full deployment trade-off curve: spurious-edit-reduction vs
useful-edit-loss as a function of threshold.

For deployment, this is the Pareto frontier readers will want to
see. Different deployment regimes (interactive vs autonomous;
human-review-cheap vs human-review-expensive) sit at different
points on this curve.

Output:
  results/monitor_real/threshold_sweep_deployment.json (curves per model)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_pairs(cache_root: Path, layer: int, pos: int, v_unit):
    import torch
    by_task: dict[str, dict] = {}
    for pt in sorted(cache_root.rglob("*__code_tests.pt")):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        T = int(payload["resid_pre"].shape[2])
        abs_pos = pos if pos >= 0 else T + pos
        vec = payload["resid_pre"][layer, 0, abs_pos, :].float().numpy()
        proj = float(vec @ v_unit)
        logits = payload["action_logits"]
        argmax = max(logits.items(), key=lambda kv: kv[1]["logit"])[0]
        by_task.setdefault(payload["task_id"], {})[cond] = {
            "score": -proj,
            "argmax": argmax,
        }
    return [(t, s) for t, s in by_task.items() if "buggy" in s and "fixed" in s]


def _sweep(pairs):
    """Return a sorted list of (threshold, reduction, loss, final_spurious) tuples."""
    import numpy as np
    scores_b = np.asarray([s["buggy"]["score"] for _, s in pairs])
    scores_f = np.asarray([s["fixed"]["score"] for _, s in pairs])
    proposes_b = np.asarray([s["buggy"]["argmax"] == "edit" for _, s in pairs])
    proposes_f = np.asarray([s["fixed"]["argmax"] == "edit" for _, s in pairs])

    n_useful_proposed = int(proposes_b.sum())
    n_spurious_proposed = int(proposes_f.sum())
    N = len(pairs)

    # Candidate thresholds: sample 200 uniformly in the score range.
    all_scores = np.concatenate([scores_b, scores_f])
    thr_lo, thr_hi = float(all_scores.min()) - 0.5, float(all_scores.max()) + 0.5
    grid = list(np.linspace(thr_lo, thr_hi, 200))
    # Plus the actual operating points
    out_rows = []
    for thr in grid:
        # Veto when score < thr
        b_veto = scores_b < thr
        f_veto = scores_f < thr
        # Useful edits killed = proposed AND vetoed
        c = int((proposes_b & b_veto).sum())
        # Useful edits committed = proposed AND not vetoed
        a = int((proposes_b & ~b_veto).sum())
        # Spurious edits killed = proposed AND vetoed
        f_ = int((proposes_f & f_veto).sum())
        d = int((proposes_f & ~f_veto).sum())
        reduction = f_ / max(n_spurious_proposed, 1) if n_spurious_proposed > 0 else float("nan")
        loss = c / max(n_useful_proposed, 1) if n_useful_proposed > 0 else float("nan")
        out_rows.append({
            "threshold": float(thr),
            "spurious_blocked": f_,
            "spurious_committed": d,
            "useful_killed": c,
            "useful_committed": a,
            "spurious_edit_reduction": reduction,
            "useful_edit_loss": loss,
            "final_spurious_rate": d / max(N, 1),
        })
    return out_rows, {
        "n_pairs": N,
        "n_useful_proposed": n_useful_proposed,
        "n_spurious_proposed": n_spurious_proposed,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="qwen,codegemma,deepseek")
    p.add_argument("--out", type=Path,
                   default=Path("results/monitor_real/threshold_sweep_deployment.json"))
    args = p.parse_args(argv)

    import torch

    model_configs = {
        "qwen": {
            "cache_dir": Path("results/cache-real-qwen-n500-20260516T235301Z"),
            "v_noop":   Path("results/steer-20260516T021522Z/v_noop.pt"),
            "layer": 24, "pos": -1,
        },
        "codegemma": {
            "cache_dir": Path("results/cache-real-codegemma-n500-20260516T235731Z"),
            "v_noop":   Path("results/steer-codegemma_7b_it-20260516T051943Z/v_noop.pt"),
            "layer": 26, "pos": -1,
        },
        "deepseek": {
            "cache_dir": Path("results/cache-real-deepseek-n500-20260517T013041Z"),
            "v_noop":   Path("results/steer-deepseek-coder-13b-instruct-20260517T012848Z/v_noop.pt"),
            "layer": 22, "pos": -1,
        },
    }
    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    by_model: dict[str, dict] = {}
    for label in selected:
        cfg = model_configs[label]
        v_blob = torch.load(cfg["v_noop"], map_location="cpu", weights_only=False)
        v_unit = (v_blob["direction"].float() / v_blob["direction"].float().norm()).numpy()
        inner = cfg["cache_dir"] / cfg["cache_dir"].name
        cache_root = inner if inner.exists() else cfg["cache_dir"]
        pairs = _load_pairs(cache_root, cfg["layer"], cfg["pos"], v_unit)
        rows, meta = _sweep(pairs)
        # Find "knee" points worth highlighting: 100% reduction smallest threshold,
        # 0% loss largest threshold, and balanced.
        # (We just save the full grid; figure code can highlight.)
        by_model[label] = {
            "config": {
                "label": label,
                **{k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
                **meta,
            },
            "sweep": rows,
        }
        # Headline summary at a few thresholds
        print(f"[{label}] N={meta['n_pairs']}  useful_proposed={meta['n_useful_proposed']}  spurious_proposed={meta['n_spurious_proposed']}")
        # Print the row at maximal reduction with minimal loss (knee point)
        valid = [r for r in rows if r["spurious_edit_reduction"] == r["spurious_edit_reduction"]
                 and r["useful_edit_loss"] == r["useful_edit_loss"]]
        if valid:
            # Pick the threshold maximising spurious_reduction - useful_loss
            best = max(valid, key=lambda r: r["spurious_edit_reduction"] - r["useful_edit_loss"])
            print(f"  knee (maximises red - loss): thr={best['threshold']:+.3f}  "
                  f"reduction={best['spurious_edit_reduction']*100:.1f}%  "
                  f"loss={best['useful_edit_loss']*100:.1f}%  "
                  f"final_spurious={best['final_spurious_rate']*100:.2f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"by_model": by_model}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
