#!/usr/bin/env python3
"""Cross-format layer sweep on CodeGemma / DeepSeek.

The G.12 cross-model paraphrase result showed CodeGemma's L26
and DeepSeek's L22 v_noop directions don't transfer to paraphrased
prompts. This script asks: does some OTHER (layer, position) cell
on these models host a paraphrase-robust direction?

For each layer L (at pos = -1, the action token), evaluate three
candidate directions × three formats:

  Directions (per layer):
    v_toy_L:     mean(toy_fixed[L])    - mean(toy_buggy[L])
                 — the canonical toy-derived direction at layer L.
                 Mirrors the paper's frozen-transfer methodology.
    v_real_L:    mean(real_pytest_fixed[L]) - mean(real_pytest_buggy[L])
                 — uses 499 paired real-pytest samples to derive
                 the direction. In-sample for pytest evaluation,
                 cross-format for the paraphrase evaluations.
    v_para_L:    mean(real_para_real_fixed[L]) - mean(real_para_real_buggy[L])
                 — in-sample on realistic-paraphrase. Tests
                 whether ANY direction at L can discriminate
                 paraphrased prompts at all (a "noise ceiling").

  Evaluation formats:
    pytest (the §5.1 paired prompts)
    paraphrase_minimal (G.12 minimal-NL variant)
    paraphrase_realistic (G.12 prose-NL variant)

A "paraphrase-robust direction" is a (layer, direction) pair where
realistic-paraphrase AUC > ~0.85 AND pytest AUC > 0.85.

Outputs:
  results/monitor_real/cross_format_layer_sweep_{model}.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _load_paired_at_pos(cache_root: Path, variant: str, pos: int):
    """Return (task_ids, buggy[N,L,D], fixed[N,L,D]) for all layers at `pos`."""
    import numpy as np
    import torch

    by_task: dict[str, dict[str, "np.ndarray"]] = {}
    pattern = f"*__{variant}.pt"
    for pt in sorted(cache_root.rglob(pattern)):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        resid = blob["resid_pre"]  # [L, B, T, D]
        T = int(resid.shape[2])
        abs_pos = pos if pos >= 0 else T + pos
        # [L, D]
        vec_per_layer = resid[:, 0, abs_pos, :].float().numpy()
        by_task.setdefault(blob["task_id"], {})[cond] = vec_per_layer
    paired = sorted(t for t, s in by_task.items() if "buggy" in s and "fixed" in s)
    if not paired:
        return [], None, None
    buggy = np.stack([by_task[t]["buggy"] for t in paired])  # [N, L, D]
    fixed = np.stack([by_task[t]["fixed"] for t in paired])
    return paired, buggy, fixed


def _auc_at_layer(scores_buggy_L: "np.ndarray", scores_fixed_L: "np.ndarray") -> float:
    """Compute AUC at a single layer given per-task projection scores.

    `scores_*_L` shape: [N_real]. Positive class = buggy (per §5.1 convention,
    higher score = more buggy-like). score = -projection.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    N = scores_buggy_L.shape[0]
    labels = np.concatenate([np.ones(N, int), np.zeros(N, int)])
    scores = np.concatenate([-scores_buggy_L, -scores_fixed_L])
    return float(roc_auc_score(labels, scores))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=("codegemma", "deepseek"))
    p.add_argument("--toy-cache", type=Path, required=True)
    p.add_argument("--real-pytest-cache", type=Path, required=True)
    p.add_argument("--real-para-min-cache", type=Path, required=True)
    p.add_argument("--real-para-real-cache", type=Path, required=True)
    p.add_argument("--pos", type=int, default=-1)
    p.add_argument("--out", type=Path,
                   default=None)
    args = p.parse_args(argv)

    import numpy as np

    if args.out is None:
        args.out = Path(f"results/monitor_real/cross_format_layer_sweep_{args.model}.json")

    def _resolve(p: Path) -> Path:
        inner = p / p.name
        return inner if inner.exists() else p

    # ===== Load all caches at all layers (pos = -1) =====
    print(f"[{args.model}] loading toy pytest cache...", file=sys.stderr)
    _, toy_b, toy_f = _load_paired_at_pos(
        _resolve(args.toy_cache), "code_tests", args.pos)
    n_toy, n_layers, D = toy_b.shape  # type: ignore
    print(f"  toy:  {n_toy} pairs, {n_layers} layers, {D} hidden",
          file=sys.stderr)

    print(f"[{args.model}] loading real pytest cache...", file=sys.stderr)
    _, real_pyt_b, real_pyt_f = _load_paired_at_pos(
        _resolve(args.real_pytest_cache), "code_tests", args.pos)
    n_real_pyt = real_pyt_b.shape[0]  # type: ignore
    print(f"  real pytest: {n_real_pyt} pairs", file=sys.stderr)

    print(f"[{args.model}] loading paraphrase-minimal cache...", file=sys.stderr)
    _, real_min_b, real_min_f = _load_paired_at_pos(
        _resolve(args.real_para_min_cache),
        "code_tests_paraphrased_minimal", args.pos)
    n_real_min = real_min_b.shape[0]  # type: ignore
    print(f"  real paraphrase-min: {n_real_min} pairs", file=sys.stderr)

    print(f"[{args.model}] loading paraphrase-realistic cache...", file=sys.stderr)
    _, real_real_b, real_real_f = _load_paired_at_pos(
        _resolve(args.real_para_real_cache),
        "code_tests_paraphrased_realistic", args.pos)
    n_real_real = real_real_b.shape[0]  # type: ignore
    print(f"  real paraphrase-real: {n_real_real} pairs", file=sys.stderr)

    # ===== Sweep layers =====
    rows = []
    print()
    print(f"{'L':>3} | {'v_toy':>7} {'v_real':>7} {'v_para':>7} | "
          f"{'v_toy':>7} {'v_real':>7} {'v_para':>7} | "
          f"{'v_toy':>7} {'v_real':>7} {'v_para':>7}")
    print(f"{'  ':>3} | {'pytest':>7} {'pytest':>7} {'pytest':>7} | "
          f"{'minimal':>7} {'minimal':>7} {'minimal':>7} | "
          f"{'realist':>7} {'realist':>7} {'realist':>7}")
    print("-" * 90)
    for L in range(n_layers):
        # Three candidate directions at this layer
        v_toy = toy_f[:, L, :].mean(0) - toy_b[:, L, :].mean(0)
        v_toy = v_toy / max(np.linalg.norm(v_toy), 1e-12)
        v_real = real_pyt_f[:, L, :].mean(0) - real_pyt_b[:, L, :].mean(0)
        v_real = v_real / max(np.linalg.norm(v_real), 1e-12)
        v_para = real_real_f[:, L, :].mean(0) - real_real_b[:, L, :].mean(0)
        v_para = v_para / max(np.linalg.norm(v_para), 1e-12)

        # Evaluate on three formats
        # AUC convention: positive class = buggy, score = -projection.
        out_row = {"layer": L}
        for fmt_label, fmt_b, fmt_f in (
            ("pytest", real_pyt_b, real_pyt_f),
            ("paraphrase_minimal", real_min_b, real_min_f),
            ("paraphrase_realistic", real_real_b, real_real_f),
        ):
            scores_b = fmt_b[:, L, :]  # [N_real, D]
            scores_f = fmt_f[:, L, :]
            for dir_label, v in (("v_toy", v_toy), ("v_real", v_real),
                                  ("v_para", v_para)):
                proj_b = scores_b @ v
                proj_f = scores_f @ v
                auc = _auc_at_layer(proj_b, proj_f)
                out_row[f"{dir_label}__{fmt_label}_auc"] = auc

        rows.append(out_row)
        # Print a compact summary: AUC of each direction × each format
        cells = []
        for fmt_label in ("pytest", "paraphrase_minimal", "paraphrase_realistic"):
            for dir_label in ("v_toy", "v_real", "v_para"):
                key = f"{dir_label}__{fmt_label}_auc"
                cells.append(f"{out_row[key]:>7.3f}")
        print(f"{L:>3} | {cells[0]} {cells[1]} {cells[2]} | "
              f"{cells[3]} {cells[4]} {cells[5]} | "
              f"{cells[6]} {cells[7]} {cells[8]}")

    # ===== Find the "paraphrase-robust" cells =====
    # Criterion: pytest AUC > 0.85 AND paraphrase-realistic AUC > 0.85
    print()
    print("=== Paraphrase-robust candidates (pytest AUC>0.85 AND real-paraphrase AUC>0.85) ===")
    candidates = []
    for r in rows:
        for dir_label in ("v_toy", "v_real", "v_para"):
            pytest_auc = r[f"{dir_label}__pytest_auc"]
            para_real_auc = r[f"{dir_label}__paraphrase_realistic_auc"]
            if pytest_auc > 0.85 and para_real_auc > 0.85:
                candidates.append({
                    "layer": r["layer"],
                    "direction": dir_label,
                    "pytest_auc": pytest_auc,
                    "para_real_auc": para_real_auc,
                    "para_min_auc": r[f"{dir_label}__paraphrase_minimal_auc"],
                })
    if not candidates:
        print("  (none found)")
    else:
        for c in candidates:
            print(f"  L{c['layer']:>2} {c['direction']:>7}: "
                  f"pytest={c['pytest_auc']:.3f} "
                  f"para_real={c['para_real_auc']:.3f} "
                  f"para_min={c['para_min_auc']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {
            "model": args.model,
            "pos": args.pos,
            "n_layers": int(n_layers),
            "n_toy": int(n_toy),
            "n_real": int(n_real_pyt),
        },
        "rows": rows,
        "paraphrase_robust_candidates": candidates,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
