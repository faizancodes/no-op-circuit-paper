#!/usr/bin/env python3
"""No-transcript real-task evaluation: does ANY residual direction
discriminate buggy from fixed code without a pytest transcript?

For each of the 499 SWE-bench-Verified-derived paired prompts, we
compute three classifiers at Qwen L24/pos -1, using ONLY the no-
transcript `code` variant cache:

  1. FROZEN v_noop transfer: project onto the toy-trained v_noop
     (the §5.1 direction). Already shown to collapse (AUC 0.499)
     in `score_alt_variant_with_v_noop.py`.

  2. FRESH v_noop_no_tx: derive a new contrastive direction from
     the no-transcript paired residuals themselves —
       v_no_tx = mean(fixed_code_resid) - mean(buggy_code_resid)
     and project. This is in-sample to the test data, so it's an
     UPPER BOUND on what a 1-d direction could achieve.

  3. FULL-residual probe: train a logistic-regression probe on the
     1536-D residual at L24/pos-1 to classify buggy/fixed, under
     leave-one-out CV over the 499 paired tasks. Caveat: with
     d=1536 ≫ n=499 LOO has overfit risk, so this is an upper
     bound on what's linearly separable.

If even the LOO-CV probe and the in-sample fresh direction fail
to lift AUC meaningfully above chance, then code semantics is
genuinely not encoded along an accessible linear direction at
L24/pos-1 in the no-transcript setting — and the §5.2 transcript-
text reading of the §5.1 monitor is fully confirmed at real-task
scale.

Output: results/monitor_real/no_transcript_full_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_paired_residuals(cache_dir: Path, variant: str, layer: int, pos: int):
    """Return (task_ids, buggy_resid_NxD, fixed_resid_NxD)."""
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
    paired_tids = sorted(
        t for t, s in by_task.items() if "buggy" in s and "fixed" in s)
    buggy = np.stack([by_task[t]["buggy"] for t in paired_tids])
    fixed = np.stack([by_task[t]["fixed"] for t in paired_tids])
    return paired_tids, buggy, fixed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--variant", default="code")
    p.add_argument("--v-noop", type=Path,
                   default=Path("results/steer-20260516T021522Z/v_noop.pt"))
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--pos", type=int, default=-1)
    p.add_argument("--out", type=Path,
                   default=Path("results/monitor_real/no_transcript_full_analysis.json"))
    args = p.parse_args(argv)

    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    inner = args.cache_dir / args.cache_dir.name
    cache_root = inner if inner.exists() else args.cache_dir
    print(f"cache_dir : {cache_root}")
    print(f"variant   : {args.variant}")
    print(f"site      : L{args.layer}/pos {args.pos:+d}")

    # Load no-transcript paired residuals
    tids, B, F = _load_paired_residuals(cache_root, args.variant, args.layer, args.pos)
    N, D = B.shape
    print(f"paired tasks: N={N}  D={D}")

    X = np.concatenate([B, F], axis=0)
    y = np.concatenate([np.ones(N, int), np.zeros(N, int)])

    # ===== 1. Frozen v_noop transfer =====
    v_blob = torch.load(args.v_noop, map_location="cpu", weights_only=False)
    v_unit = (v_blob["direction"].float() / v_blob["direction"].float().norm()).numpy()
    proj_b_frozen = B @ v_unit
    proj_f_frozen = F @ v_unit
    scores_frozen = np.concatenate([-proj_b_frozen, -proj_f_frozen])
    auc_frozen = float(roc_auc_score(y, scores_frozen))
    print(f"\n(1) FROZEN toy-trained v_noop projection AUC: {auc_frozen:.4f}")

    # ===== 2. Fresh v_no_tx (in-sample) =====
    v_no_tx = F.mean(axis=0) - B.mean(axis=0)
    v_no_tx_norm = float(np.linalg.norm(v_no_tx))
    v_no_tx_unit = v_no_tx / max(v_no_tx_norm, 1e-12)
    cos_with_frozen = float(np.dot(v_no_tx_unit, v_unit))
    scores_fresh = np.concatenate([-(B @ v_no_tx_unit), -(F @ v_no_tx_unit)])
    auc_fresh_in_sample = float(roc_auc_score(y, scores_fresh))
    print(f"\n(2) FRESH v_no_tx (in-sample contrastive at code-variant residuals):")
    print(f"    ||v_no_tx|| = {v_no_tx_norm:.4f}  cos(v_no_tx, frozen v_noop) = {cos_with_frozen:+.4f}")
    print(f"    in-sample AUC: {auc_fresh_in_sample:.4f}")

    # Same direction under LOO over PAIRS (paired-task LOO)
    aucs_loo: list[float] = []
    for i in range(N):
        mask = np.ones(N, dtype=bool); mask[i] = False
        v = F[mask].mean(0) - B[mask].mean(0)
        vu = v / max(np.linalg.norm(v), 1e-12)
        # score the held-out pair
        s_b, s_f = -(B[i] @ vu), -(F[i] @ vu)
        # pair-level AUC isn't well-defined on 2 points; instead store
        # the held-out scores and labels and aggregate.
        aucs_loo.append((s_b, s_f))
    s_b_all = np.asarray([x[0] for x in aucs_loo])
    s_f_all = np.asarray([x[1] for x in aucs_loo])
    auc_fresh_loo = float(roc_auc_score(
        np.concatenate([np.ones(N, int), np.zeros(N, int)]),
        np.concatenate([s_b_all, s_f_all]),
    ))
    print(f"    LOO-pair AUC (proper out-of-sample): {auc_fresh_loo:.4f}")

    # ===== 3. Full-residual probe (LOO-CV) =====
    print(f"\n(3) FULL-residual logistic-regression probe (1536-D, LOO over paired tasks)...")
    loo_scores = np.empty(2 * N)
    for i in range(N):
        train_mask = np.ones(N, dtype=bool); train_mask[i] = False
        X_train = np.concatenate([B[train_mask], F[train_mask]], axis=0)
        y_train = np.concatenate([np.ones(train_mask.sum(), int),
                                   np.zeros(train_mask.sum(), int)])
        # Use a low C (high regularization) given d >> n
        clf = LogisticRegression(C=0.1, max_iter=2000)
        clf.fit(X_train, y_train)
        # Predict on held-out pair
        ho = np.stack([B[i], F[i]], axis=0)
        prob = clf.predict_proba(ho)[:, 1]
        loo_scores[i] = prob[0]      # buggy held-out → P(buggy)
        loo_scores[N + i] = prob[1]  # fixed held-out → P(buggy)
        if (i + 1) % 50 == 0:
            print(f"    LOO {i+1}/{N}")
    auc_probe_loo = float(roc_auc_score(y, loo_scores))
    print(f"    LOO probe AUC: {auc_probe_loo:.4f}")

    # ===== Summary =====
    print()
    print("=" * 60)
    print(f"NO-TRANSCRIPT REAL-TASK EVALUATION SUMMARY (Qwen L24/pos -1, N={N})")
    print("=" * 60)
    print(f"  Frozen toy v_noop (§5.1 transfer):         AUC = {auc_frozen:.4f}")
    print(f"  Fresh v_no_tx, in-sample:                  AUC = {auc_fresh_in_sample:.4f}")
    print(f"  Fresh v_no_tx, paired LOO out-of-sample:   AUC = {auc_fresh_loo:.4f}")
    print(f"  1536-D LR probe, LOO out-of-sample:        AUC = {auc_probe_loo:.4f}")
    print()
    if max(auc_fresh_loo, auc_probe_loo) < 0.6:
        verdict = "DECISIVE: no linear direction at L24/pos-1 discriminates code semantics in the absence of a transcript."
    elif max(auc_fresh_loo, auc_probe_loo) < 0.75:
        verdict = "WEAK: some residual signal present but far below the with-transcript baseline."
    else:
        verdict = "STRONG: a residual direction DOES discriminate without a transcript — the paper's narrative needs another adjustment."
    print(f"VERDICT: {verdict}")

    out = {
        "config": {
            "cache_dir": str(cache_root),
            "variant": args.variant,
            "v_noop": str(args.v_noop),
            "layer": args.layer, "pos": args.pos,
            "n_paired_tasks": N,
        },
        "frozen_v_noop_transfer_auc": auc_frozen,
        "fresh_v_no_tx": {
            "v_norm": v_no_tx_norm,
            "cos_with_frozen_v_noop": cos_with_frozen,
            "in_sample_auc": auc_fresh_in_sample,
            "loo_pair_auc": auc_fresh_loo,
        },
        "full_residual_loo_probe_auc": auc_probe_loo,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
