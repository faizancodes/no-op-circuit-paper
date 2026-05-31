#!/usr/bin/env python3
"""Score arbitrary cache_dataset runs with the frozen toy-trained v_noop.

Used by the lexical-redaction and action-vocab-swap control experiments
to project the new cache's residuals onto the SAME v_noop the §5.1 monitor
uses, and compare AUC / projection-gap / argmax distribution.

Per-cache outputs (printed + written):
  - paired-task projections (buggy vs fixed)
  - ROC-AUC, AP under the §5.1 (-projection) scoring convention
  - in-sample balanced-accuracy operating-point metrics (for comparison)
  - argmax-action distribution (which action wins under the new condition?)

Usage:
  python scripts/score_alt_variant_with_v_noop.py \
    --cache-dir results/cache-real-qwen-lex-redact-20260518T081303Z \
    --variant-name code_tests_lex_redacted \
    --label "lex_redacted" \
    --out results/monitor_real/lex_redacted_scores.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _resolve_cache_subdir(cache_dir: Path) -> Path:
    """If cache_dir contains a nested same-name subdir, descend into it."""
    inner = cache_dir / cache_dir.name
    return inner if inner.exists() else cache_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--variant-name", required=True,
                   help="The variant filename suffix (e.g. 'code_tests', 'code_tests_lex_redacted').")
    p.add_argument("--label", default="alt",
                   help="Short label for this run (used in output filenames + summary).")
    p.add_argument("--v-noop", type=Path,
                   default=Path("results/steer-20260516T021522Z/v_noop.pt"))
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--pos", type=int, default=-1)
    p.add_argument("--out", type=Path,
                   default=None,
                   help="JSON output path. Default: results/monitor_real/{label}_scores.json")
    args = p.parse_args(argv)

    import numpy as np
    import torch
    from sklearn.metrics import (
        average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
    )

    cache_dir = _resolve_cache_subdir(args.cache_dir)
    print(f"cache_dir : {cache_dir}")
    if args.out is None:
        args.out = Path(f"results/monitor_real/{args.label}_scores.json")

    v_blob = torch.load(args.v_noop, map_location="cpu", weights_only=False)
    v = v_blob["direction"].float()
    v_unit = (v / v.norm()).numpy()
    print(f"v_noop    : L{v_blob['layer']}/pos {v_blob['position']:+d}  "
          f"|v|={v_blob['norm']:.3f}  source_N={v_blob['n_pairs']}")
    assert v_blob["layer"] == args.layer and v_blob["position"] == args.pos

    pattern = f"*__{args.variant_name}.pt"
    by_task: dict[str, dict] = {}
    for pt in sorted(cache_dir.rglob(pattern)):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        T = int(payload["resid_pre"].shape[2])
        abs_pos = args.pos if args.pos >= 0 else T + args.pos
        vec = payload["resid_pre"][args.layer, 0, abs_pos, :].float().numpy()
        proj = float(vec @ v_unit)
        action_logits = payload["action_logits"]
        # Logits is a dict {name: {"logit": ..., ...}}. Find argmax by logit.
        argmax_name = max(action_logits.items(),
                          key=lambda kv: kv[1]["logit"])[0]
        by_task.setdefault(payload["task_id"], {})[cond] = {
            "projection": proj,
            "score": -proj,
            "argmax": argmax_name,
            "logits": {k: float(v_["logit"]) for k, v_ in action_logits.items()},
        }

    paired = sorted([(t, s["buggy"], s["fixed"]) for t, s in by_task.items()
                     if "buggy" in s and "fixed" in s])
    N = len(paired)
    print(f"paired tasks: {N}")
    if N < 10:
        print("error: too few paired tasks", file=sys.stderr)
        return 2

    proj_b = np.asarray([b["projection"] for _, b, _ in paired])
    proj_f = np.asarray([f["projection"] for _, _, f in paired])
    score_b = -proj_b
    score_f = -proj_f
    print(f"\nmean projection on v_noop:")
    print(f"  buggy : mean={proj_b.mean():+.3f}  median={float(np.median(proj_b)):+.3f}")
    print(f"  fixed : mean={proj_f.mean():+.3f}  median={float(np.median(proj_f)):+.3f}")
    print(f"  gap   : mean(fixed)-mean(buggy) = {proj_f.mean()-proj_b.mean():+.3f}")

    scores = np.concatenate([score_b, score_f])
    labels = np.concatenate([np.ones_like(score_b, dtype=int),
                              np.zeros_like(score_f, dtype=int)])
    roc_auc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))
    fpr, tpr, roc_thr = roc_curve(labels, scores)
    bal_acc = 0.5 * (tpr + (1 - fpr))
    best = int(np.argmax(bal_acc))
    op_thr = float(roc_thr[best])
    y_pred = (scores >= op_thr).astype(int)
    tp = int(((y_pred == 1) & (labels == 1)).sum())
    fp = int(((y_pred == 1) & (labels == 0)).sum())
    fn = int(((y_pred == 0) & (labels == 1)).sum())
    tn = int(((y_pred == 0) & (labels == 0)).sum())
    op = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": op_thr,
        "precision": tp / max(tp + fp, 1),
        "recall":    tp / max(tp + fn, 1),
        "accuracy":  (tp + tn) / max(len(labels), 1),
        "false_edit_rate": fp / max(fp + tn, 1),
    }

    print(f"\n=== monitor metrics on {args.label} cache (frozen v_noop) ===")
    print(f"  ROC-AUC        : {roc_auc:.4f}  (§5.1 baseline: 0.989)")
    print(f"  PR-AUC (AP)    : {pr_auc:.4f}")
    print(f"  op-threshold   : -projection ≥ {op_thr:+.4f} (in-sample)")
    print(f"    precision    : {op['precision']:.4f}  ({tp}/{tp+fp})")
    print(f"    recall       : {op['recall']:.4f}  ({tp}/{tp+fn})")
    print(f"    false-edit   : {op['false_edit_rate']:.4f}  ({fp}/{fp+tn})")

    # Argmax distribution per condition
    argmax_b = Counter(b["argmax"] for _, b, _ in paired)
    argmax_f = Counter(f["argmax"] for _, _, f in paired)
    # Sample one task's action vocab for the print header (assume uniform per cache).
    vocab = list(paired[0][1]["logits"].keys())
    print(f"\nargmax-action distribution (vocab: {vocab}):")
    print("  cond   " + "  ".join(f"{a:>8}" for a in vocab))
    for label_, cnt in (("buggy", argmax_b), ("fixed", argmax_f)):
        n = sum(cnt.values()) or 1
        print(f"  {label_:<6} " + "  ".join(
            f"{100*cnt.get(a, 0)/n:7.1f}%" for a in vocab))

    out = {
        "config": {
            "cache_dir": str(cache_dir),
            "variant_name": args.variant_name,
            "label": args.label,
            "v_noop": str(args.v_noop),
            "layer": args.layer, "pos": args.pos,
            "n_paired_tasks": N,
            "action_vocab": vocab,
        },
        "mean_projection": {
            "buggy": float(proj_b.mean()),
            "fixed": float(proj_f.mean()),
            "gap": float(proj_f.mean() - proj_b.mean()),
        },
        "metrics": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "op_in_sample": op,
        },
        "argmax_distribution": {
            "buggy": dict(argmax_b),
            "fixed": dict(argmax_f),
        },
        "per_task": [
            {
                "task_id": t,
                "buggy_projection": float(b["projection"]),
                "fixed_projection": float(f["projection"]),
                "buggy_argmax": b["argmax"],
                "fixed_argmax": f["argmax"],
            }
            for (t, b, f) in paired
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
