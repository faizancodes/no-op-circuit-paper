#!/usr/bin/env python3
"""2D (layer × position) cross-format sweep on CodeGemma / DeepSeek.

App. G.13 swept layers at pos = −1 only. This extends to a 2D
(layer, position) grid — does *some* off-action-token position
host a paraphrase-robust direction that pos = −1 missed? The key
target is DeepSeek, where no (layer, pos=−1) cell had a v_toy
direction that survived format change.

For each (layer L, position p) in {−1, −2, ..., −8} × {0..L_max−1}:

  Compute three candidate directions (same as G.13):
    v_toy(L, p)  = mean(toy_fixed[L, p])  − mean(toy_buggy[L, p])
    v_real(L, p) = mean(real_pyt_fixed[L, p]) − mean(real_pyt_buggy[L, p])
    v_para(L, p) = mean(real_para_fixed[L, p]) − mean(real_para_buggy[L, p])

  Evaluate each on three formats (pytest, paraphrase_minimal,
  paraphrase_realistic) on the 499 paired real-task prompts.

  9 AUCs per cell × N_layers × N_pos.

A "paraphrase-robust" cell = pytest AUC > 0.85 AND
realistic-paraphrase AUC > 0.85.

Output: results/monitor_real/cross_format_position_sweep_{model}.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_paired_at_positions(cache_root: Path, variant: str, positions: list[int]):
    """Return (task_ids, buggy[N,L,P,D], fixed[N,L,P,D]) at the given positions.

    `positions` is a list of negative ints (e.g. [-1, -2, -3, ..., -8])
    indexing from the end of the sequence.
    """
    import numpy as np
    import torch

    by_task: dict[str, dict] = {}
    pattern = f"*__{variant}.pt"
    for pt in sorted(cache_root.rglob(pattern)):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        resid = blob["resid_pre"]  # [L, B, T, D]
        T = int(resid.shape[2])
        abs_positions = [p if p >= 0 else T + p for p in positions]
        # [L, len(positions), D]
        slab = resid[:, 0, abs_positions, :].float().numpy()
        by_task.setdefault(blob["task_id"], {})[cond] = slab
    paired = sorted(t for t, s in by_task.items() if "buggy" in s and "fixed" in s)
    if not paired:
        return [], None, None
    buggy = np.stack([by_task[t]["buggy"] for t in paired])  # [N, L, P, D]
    fixed = np.stack([by_task[t]["fixed"] for t in paired])
    return paired, buggy, fixed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=("codegemma", "deepseek"))
    p.add_argument("--toy-cache", type=Path, required=True)
    p.add_argument("--real-pytest-cache", type=Path, required=True)
    p.add_argument("--real-para-min-cache", type=Path, required=True)
    p.add_argument("--real-para-real-cache", type=Path, required=True)
    p.add_argument("--positions", type=str, default="-1,-2,-3,-4,-5,-6,-7,-8")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    import numpy as np
    from sklearn.metrics import roc_auc_score

    positions = [int(x.strip()) for x in args.positions.split(",")]
    print(f"[{args.model}] positions to sweep: {positions}", file=sys.stderr)

    if args.out is None:
        args.out = Path(f"results/monitor_real/cross_format_position_sweep_{args.model}.json")

    def _resolve(p: Path) -> Path:
        inner = p / p.name
        return inner if inner.exists() else p

    # ===== Load all caches at all positions =====
    print(f"[{args.model}] loading toy pytest cache...", file=sys.stderr)
    toy_ids, toy_b, toy_f = _load_paired_at_positions(
        _resolve(args.toy_cache), "code_tests", positions)
    n_toy, n_layers, n_pos, D = toy_b.shape  # type: ignore
    print(f"  toy: {n_toy} pairs, {n_layers} layers, {n_pos} positions, {D} hidden",
          file=sys.stderr)

    print(f"[{args.model}] loading real pytest cache...", file=sys.stderr)
    pyt_ids, real_pyt_b, real_pyt_f = _load_paired_at_positions(
        _resolve(args.real_pytest_cache), "code_tests", positions)
    print(f"  real pytest: {len(pyt_ids)} pairs", file=sys.stderr)

    print(f"[{args.model}] loading paraphrase-minimal cache...", file=sys.stderr)
    min_ids, real_min_b, real_min_f = _load_paired_at_positions(
        _resolve(args.real_para_min_cache),
        "code_tests_paraphrased_minimal", positions)
    print(f"  real para-min: {len(min_ids)} pairs", file=sys.stderr)

    print(f"[{args.model}] loading paraphrase-realistic cache...", file=sys.stderr)
    real_ids, real_real_b, real_real_f = _load_paired_at_positions(
        _resolve(args.real_para_real_cache),
        "code_tests_paraphrased_realistic", positions)
    print(f"  real para-real: {len(real_ids)} pairs", file=sys.stderr)

    # Intersect task IDs across all three real caches so the same set of
    # paired tasks is scored under every format.
    common_real = sorted(set(pyt_ids) & set(min_ids) & set(real_ids))
    print(f"[{args.model}] intersection across real caches: {len(common_real)} pairs",
          file=sys.stderr)

    def _filter_by_ids(arr, ids_present, common):
        idx = [ids_present.index(t) for t in common]
        return arr[idx]

    real_pyt_b = _filter_by_ids(real_pyt_b, pyt_ids, common_real)
    real_pyt_f = _filter_by_ids(real_pyt_f, pyt_ids, common_real)
    real_min_b = _filter_by_ids(real_min_b, min_ids, common_real)
    real_min_f = _filter_by_ids(real_min_f, min_ids, common_real)
    real_real_b = _filter_by_ids(real_real_b, real_ids, common_real)
    real_real_f = _filter_by_ids(real_real_f, real_ids, common_real)
    n_real = len(common_real)

    # ===== Sweep (layer × position) =====
    N_real = real_pyt_b.shape[0]  # type: ignore
    labels = np.concatenate([np.ones(N_real, int), np.zeros(N_real, int)])
    rows = []
    candidates = []

    def auc_for(scores_b, scores_f):
        return float(roc_auc_score(
            labels, np.concatenate([-scores_b, -scores_f])))

    print()
    print(f"  Computing {n_layers * n_pos} grid cells × 3 directions × 3 formats...",
          file=sys.stderr)
    for L in range(n_layers):
        for pi, pval in enumerate(positions):
            v_toy = toy_f[:, L, pi, :].mean(0) - toy_b[:, L, pi, :].mean(0)
            v_toy /= max(np.linalg.norm(v_toy), 1e-12)
            v_real = real_pyt_f[:, L, pi, :].mean(0) - real_pyt_b[:, L, pi, :].mean(0)
            v_real /= max(np.linalg.norm(v_real), 1e-12)
            v_para = real_real_f[:, L, pi, :].mean(0) - real_real_b[:, L, pi, :].mean(0)
            v_para /= max(np.linalg.norm(v_para), 1e-12)

            row = {"layer": L, "position": pval}
            for fmt_label, fmt_b, fmt_f in (
                ("pytest", real_pyt_b, real_pyt_f),
                ("paraphrase_minimal", real_min_b, real_min_f),
                ("paraphrase_realistic", real_real_b, real_real_f),
            ):
                b_slab = fmt_b[:, L, pi, :]
                f_slab = fmt_f[:, L, pi, :]
                for dir_label, v in (("v_toy", v_toy), ("v_real", v_real),
                                      ("v_para", v_para)):
                    proj_b = b_slab @ v
                    proj_f = f_slab @ v
                    row[f"{dir_label}__{fmt_label}_auc"] = auc_for(proj_b, proj_f)
            rows.append(row)

            for dir_label in ("v_toy", "v_real", "v_para"):
                pytest_auc = row[f"{dir_label}__pytest_auc"]
                para_real_auc = row[f"{dir_label}__paraphrase_realistic_auc"]
                if pytest_auc > 0.85 and para_real_auc > 0.85:
                    candidates.append({
                        "layer": L, "position": pval,
                        "direction": dir_label,
                        "pytest_auc": pytest_auc,
                        "para_real_auc": para_real_auc,
                        "para_min_auc": row[f"{dir_label}__paraphrase_minimal_auc"],
                    })

    # ===== Print summary =====
    print()
    print(f"=== {args.model}: paraphrase-robust (layer, pos) cells (pytest>0.85 AND para-real>0.85) ===")
    if not candidates:
        print("  (none found)")
    else:
        # Group by direction
        from collections import defaultdict
        by_dir = defaultdict(list)
        for c in candidates:
            by_dir[c["direction"]].append(c)
        for dir_label, cs in by_dir.items():
            print(f"\n  Direction = {dir_label}: {len(cs)} cells")
            # Sort by max(pytest + para_real)
            cs.sort(key=lambda c: -(c["pytest_auc"] + c["para_real_auc"]))
            for c in cs[:10]:
                print(f"    L{c['layer']:>2} pos {c['position']:>+3}: "
                      f"pytest={c['pytest_auc']:.3f} "
                      f"para_real={c['para_real_auc']:.3f} "
                      f"para_min={c['para_min_auc']:.3f}")

    # Best v_toy cell summary (separately, since that's the paper's
    # methodology and the key question for DeepSeek)
    print()
    print(f"=== {args.model}: best v_toy cell by paraphrase-realistic AUC ===")
    v_toy_rows = [(r["layer"], r["position"],
                    r["v_toy__pytest_auc"],
                    r["v_toy__paraphrase_realistic_auc"],
                    r["v_toy__paraphrase_minimal_auc"]) for r in rows]
    v_toy_rows.sort(key=lambda x: -x[3])
    print(f"  {'L':>3} {'pos':>4} {'pytest':>8} {'para_real':>10} {'para_min':>10}")
    for L, pp, py, pr, pm in v_toy_rows[:10]:
        print(f"  {L:>3} {pp:>+4} {py:>8.4f} {pr:>10.4f} {pm:>10.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {
            "model": args.model,
            "positions": positions,
            "n_layers": int(n_layers),
            "n_pos": int(n_pos),
            "n_toy": int(n_toy),
            "n_real": int(n_real),
        },
        "rows": rows,
        "paraphrase_robust_candidates": candidates,
    }, indent=2))
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
