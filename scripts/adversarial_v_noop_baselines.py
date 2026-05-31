#!/usr/bin/env python3
"""Adversarial baselines for v_noop's privileged-direction claim (Blocker B6).

The §G.2 1000-random-unit-vector baseline is degenerate — uniform unit
vectors on the 1535-sphere are near-orthogonal to any discriminative
direction. This script adds harder controls:

  A. PCA1 at the same site (L24, pos -1) over fixed + buggy `code_tests`
     residuals from the toy substrate. A privileged direction should
     dominate the first variance-explaining axis at its own cell.

  B. v_noop at the wrong (layer, position): compute mean(fixed) -
     mean(buggy) using toy `code_tests` residuals at (L12, pos -1) and
     (L24, pos -8), then project SWE-bench scores onto each. The
     mechanism is at L24/pos -1; wrong cells should generalise poorly.

  C. Cross-model transfer: Qwen v_noop is 1536-dim, CodeGemma L26 is
     3072-dim. Direct application is dimensionally invalid; we document
     this explicitly rather than fabricating a comparison.

Outputs results/monitor_real/adversarial_baselines.json with one row per
baseline (AUC + bootstrap 95% CI on SWE-bench Verified).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _seed_all(seed: int) -> None:
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _load_toy_residuals_at(toy_cache_dir: Path, layer: int, pos: int):
    """Return (fixed_resid, buggy_resid) stacked (N, D) at (layer, pos)."""
    import numpy as np
    import torch

    fix_rows: list[np.ndarray] = []
    bug_rows: list[np.ndarray] = []
    for task_dir in sorted(toy_cache_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for cond in ("fixed", "buggy"):
            p = task_dir / f"{cond}__code_tests.pt"
            if not p.is_file():
                continue
            blob = torch.load(p, map_location="cpu", weights_only=False)
            r = blob["resid_pre"]                 # (L, 1, T, D)
            T = int(r.shape[2])
            abs_pos = pos if pos >= 0 else T + pos
            vec = r[layer, 0, abs_pos, :].float().numpy()
            (fix_rows if cond == "fixed" else bug_rows).append(vec)
    if not fix_rows or not bug_rows:
        raise RuntimeError(f"no residuals at (L{layer}, pos {pos}) in {toy_cache_dir}")
    return np.stack(fix_rows), np.stack(bug_rows)


def _project_swebench(real_cache_dir: Path, v: "object", layer: int, pos: int):
    """Project SWE-bench resid_pre[layer, pos] onto unit-v, return
    (scores, labels). scores = -projection so higher = more buggy."""
    import numpy as np
    import torch
    from collections import defaultdict

    v = np.asarray(v, dtype=float)
    v_unit = v / max(float(np.linalg.norm(v)), 1e-12)
    by_task: dict[str, dict[str, float]] = defaultdict(dict)
    for pt in sorted(real_cache_dir.rglob("*__code_tests.pt")):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        r = blob["resid_pre"]
        T = int(r.shape[2])
        abs_pos = pos if pos >= 0 else T + pos
        x = r[layer, 0, abs_pos, :].float().numpy()
        by_task[blob["task_id"]][cond] = float(x @ v_unit)
    paired = [(t, d["buggy"], d["fixed"]) for t, d in by_task.items()
              if "buggy" in d and "fixed" in d]
    if not paired:
        raise RuntimeError(f"no paired SWE-bench observations in {real_cache_dir}")
    proj_buggy = np.array([b for _, b, _ in paired])
    proj_fixed = np.array([f for _, _, f in paired])
    scores = np.concatenate([-proj_buggy, -proj_fixed])
    labels = np.concatenate([
        np.ones_like(proj_buggy, dtype=int),
        np.zeros_like(proj_fixed, dtype=int),
    ])
    return scores, labels, len(paired)


def _auc_and_ci(scores, labels, n_boot: int, rng):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(labels, scores))
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, len(scores), size=len(scores))
        if len(set(labels[idx].tolist())) < 2:
            boot[b] = np.nan
            continue
        boot[b] = roc_auc_score(labels[idx], scores[idx])
    boot = boot[~np.isnan(boot)]
    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))
    return auc, lo, hi


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--toy-cache-dir", type=Path,
                   default=Path("results/cache-20260515T221105Z/cache-20260515T221105Z"),
                   help="Qwen toy substrate cache root (with per-task subdirs).")
    p.add_argument("--real-cache-dir", type=Path,
                   default=None,
                   help="SWE-bench Verified cache root (default: most-recent cache-real-qwen-*).")
    p.add_argument("--v-noop", type=Path,
                   default=Path("results/steer-20260516T021522Z/v_noop.pt"),
                   help="Frozen toy-trained v_noop for the (correct-site) reference row.")
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--pos", type=int, default=-1)
    p.add_argument("--wrong-layer", type=int, default=12,
                   help="Baseline B: wrong layer for v_noop comparison.")
    p.add_argument("--wrong-pos", type=int, default=-8,
                   help="Baseline B: wrong position for v_noop comparison.")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=Path("results/monitor_real/adversarial_baselines.json"))
    args = p.parse_args(argv)

    import numpy as np
    import torch
    _seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.real_cache_dir is None:
        cands = sorted(Path("results").glob("cache-real-qwen-n500-*"),
                       key=lambda p_: p_.name, reverse=True)
        if not cands:
            print("no cache-real-qwen-n500-* under results/", file=sys.stderr)
            return 2
        inner = cands[0] / cands[0].name
        args.real_cache_dir = inner if inner.exists() else cands[0]
    print(f"real_cache_dir : {args.real_cache_dir}")
    print(f"toy_cache_dir  : {args.toy_cache_dir}")

    # ---- Reference: v_noop at the correct site ----
    v_blob = torch.load(args.v_noop, map_location="cpu", weights_only=False)
    v_noop = v_blob["direction"].float().numpy()
    print(f"v_noop site    : (L{v_blob['layer']}, pos {v_blob['position']}); ||v||={float(v_blob['norm']):.3f}")
    assert v_blob["layer"] == args.layer and v_blob["position"] == args.pos

    rows: list[dict] = []

    print("\n--- baselines ---")
    s, l, n = _project_swebench(args.real_cache_dir, v_noop, args.layer, args.pos)
    auc, lo, hi = _auc_and_ci(s, l, args.n_boot, rng)
    print(f"  v_noop (L{args.layer}, pos {args.pos:+d})           AUC {auc:.4f} [{lo:.4f}, {hi:.4f}]  N={n} pairs")
    rows.append({"name": f"v_noop (L{args.layer}, pos {args.pos:+d})",
                 "site": [args.layer, args.pos],
                 "auc": auc, "ci_lo": lo, "ci_hi": hi, "n_pairs": n,
                 "kind": "reference"})

    # ---- A. PCA1 of resid_pre at the same site ----
    # PC1's sign is arbitrary; orient it so that the toy fixed/buggy mean
    # difference is positive (matching v_noop's convention).
    fix_resid, bug_resid = _load_toy_residuals_at(args.toy_cache_dir,
                                                  args.layer, args.pos)
    X = np.concatenate([fix_resid, bug_resid], axis=0)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca1 = Vt[0]
    toy_signed_gap = float((fix_resid.mean(axis=0) - bug_resid.mean(axis=0)) @ pca1)
    if toy_signed_gap < 0:
        pca1 = -pca1
        print(f"  (PCA1 sign flipped to align with toy fixed - buggy mean)")
    s, l, n = _project_swebench(args.real_cache_dir, pca1, args.layer, args.pos)
    auc, lo, hi = _auc_and_ci(s, l, args.n_boot, rng)
    print(f"  PCA1 of resid_pre[L{args.layer}, pos {args.pos:+d}]  AUC {auc:.4f} [{lo:.4f}, {hi:.4f}]")
    rows.append({"name": f"PCA1 of resid_pre (L{args.layer}, pos {args.pos:+d})",
                 "site": [args.layer, args.pos],
                 "auc": auc, "ci_lo": lo, "ci_hi": hi, "n_pairs": n,
                 "kind": "adversarial-A-pca1",
                 "note": "sign chosen so that toy fixed - buggy mean is positive"})

    # ---- B1. v_noop at wrong layer ----
    try:
        fix_w, bug_w = _load_toy_residuals_at(args.toy_cache_dir,
                                              args.wrong_layer, args.pos)
        v_wrong_layer = fix_w.mean(axis=0) - bug_w.mean(axis=0)
        s, l, n = _project_swebench(args.real_cache_dir, v_wrong_layer,
                                    args.wrong_layer, args.pos)
        auc, lo, hi = _auc_and_ci(s, l, args.n_boot, rng)
        print(f"  v_noop at WRONG layer (L{args.wrong_layer}, pos {args.pos:+d})  AUC {auc:.4f} [{lo:.4f}, {hi:.4f}]")
        rows.append({"name": f"v_noop at WRONG layer (L{args.wrong_layer}, pos {args.pos:+d})",
                     "site": [args.wrong_layer, args.pos],
                     "auc": auc, "ci_lo": lo, "ci_hi": hi, "n_pairs": n,
                     "kind": "adversarial-B-wrong-layer"})
    except Exception as e:
        print(f"  wrong-layer baseline failed: {e}", file=sys.stderr)

    # ---- B2. v_noop at wrong position ----
    try:
        fix_w, bug_w = _load_toy_residuals_at(args.toy_cache_dir,
                                              args.layer, args.wrong_pos)
        v_wrong_pos = fix_w.mean(axis=0) - bug_w.mean(axis=0)
        s, l, n = _project_swebench(args.real_cache_dir, v_wrong_pos,
                                    args.layer, args.wrong_pos)
        auc, lo, hi = _auc_and_ci(s, l, args.n_boot, rng)
        print(f"  v_noop at WRONG pos (L{args.layer}, pos {args.wrong_pos:+d})    AUC {auc:.4f} [{lo:.4f}, {hi:.4f}]")
        rows.append({"name": f"v_noop at WRONG pos (L{args.layer}, pos {args.wrong_pos:+d})",
                     "site": [args.layer, args.wrong_pos],
                     "auc": auc, "ci_lo": lo, "ci_hi": hi, "n_pairs": n,
                     "kind": "adversarial-B-wrong-pos"})
    except Exception as e:
        print(f"  wrong-position baseline failed: {e}", file=sys.stderr)

    # ---- C. Cross-model dim-mismatch ----
    cg_v_path = Path("results/steer-codegemma_7b_it-20260516T051943Z/v_noop.pt")
    cg_blob = (torch.load(cg_v_path, map_location="cpu", weights_only=False)
               if cg_v_path.is_file() else None)
    cg_dim = int(cg_blob["direction"].shape[0]) if cg_blob is not None else None
    qwen_dim = int(v_noop.shape[0])
    print(f"  Cross-model transfer SKIPPED — dim(Qwen v_noop)={qwen_dim} "
          f"vs dim(CodeGemma v_noop)={cg_dim} are incompatible without "
          f"a learned projection (would conflate baseline weakness with "
          f"projection-choice artefact).")
    rows.append({
        "name": "Cross-model (Qwen v_noop on CodeGemma site)",
        "site": None,
        "auc": None, "ci_lo": None, "ci_hi": None, "n_pairs": None,
        "kind": "skipped-dimension-mismatch",
        "note": f"Qwen v_noop is {qwen_dim}-dim; CodeGemma L26 cell is "
                f"{cg_dim}-dim. Direct projection is undefined; a learned "
                f"linear map would entangle baseline-direction quality with "
                f"projection-fit quality. We leave this open.",
    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {"seed": args.seed, "n_boot": args.n_boot,
                   "layer": args.layer, "pos": args.pos,
                   "wrong_layer": args.wrong_layer, "wrong_pos": args.wrong_pos},
        "baselines": rows,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
