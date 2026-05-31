#!/usr/bin/env python3
"""Contradictory-transcript control: distinguishes a CODE-driven monitor
from a TRANSCRIPT-TEXT detector.

This is the single load-bearing reviewer ask. The original 499 paired
SWE-bench-Verified-derived prompts give the MATCHED diagonal of a 2x2
(code x transcript) design:

  (B,B) buggy code + buggy transcript    ← in --orig-cache
  (F,F) fixed code + fixed transcript    ← in --orig-cache

The new `code_tests_swapped` variant gives the OFF-DIAGONAL cells:

  (B,F) buggy code + fixed (passing) transcript     ← in --swap-cache,
                                                      stored as buggy__code_tests_swapped.pt
  (F,B) fixed code + buggy (failing) transcript     ← in --swap-cache,
                                                      stored as fixed__code_tests_swapped.pt

For each prompt we extract resid_pre[L=24, pos=-1] and project onto the
FROZEN toy-trained v_noop. The §5.1 monitor scoring convention is
score = -projection (higher score = more buggy-like).

Outputs (in --out-dir):

  contradictory_transcript_2x2.json
    per-task scores at all 4 cells, plus aggregate statistics

  contradictory_transcript_summary.txt
    human-readable summary table

Verdict logic:
  If |ΔCode| >> |ΔTranscript| → v_noop is CODE-driven (paper claim survives)
  If |ΔTranscript| >> |ΔCode| → v_noop is TRANSCRIPT-driven (paper needs reframing)
  If they're comparable    → both contribute

We also report:
  - mean score per cell + 95% bootstrap CI
  - AUC on the swapped subset under CODE labels and under TRANSCRIPT labels
  - Argmax-action distribution per cell
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _load_resid_score(pt_path: Path, layer: int, pos: int, v_unit) -> dict:
    """Load one cache .pt and return projection-derived score + argmax."""
    import torch
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    T = int(payload["resid_pre"].shape[2])
    pos_abs = pos if pos >= 0 else T + pos
    vec = payload["resid_pre"][layer, 0, pos_abs, :].float().numpy()
    proj = float(vec @ v_unit)
    action_logits = payload["action_logits"]
    argmax_name = max(action_logits.items(),
                      key=lambda kv: kv[1]["logit"])[0]
    edit_minus_noop = (action_logits["edit"]["logit"]
                       - action_logits["noop"]["logit"])
    return {
        "task_id": payload["task_id"],
        "projection": proj,
        "score": -proj,  # §5.1 convention: higher = more buggy-like
        "argmax": argmax_name,
        "edit_noop_margin": float(edit_minus_noop),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig-cache", type=Path, required=True,
                   help="Directory holding *__code_tests.pt files for the 499 originals.")
    p.add_argument("--swap-cache", type=Path, required=True,
                   help="Directory holding *__code_tests_swapped.pt files.")
    p.add_argument("--v-noop", type=Path,
                   default=Path("results/steer-20260516T021522Z/v_noop.pt"))
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/monitor_real"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args(argv)

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Load v_noop ---
    v_blob = torch.load(args.v_noop, map_location="cpu", weights_only=False)
    v = v_blob["direction"].float()
    v_unit = (v / v.norm()).numpy()
    assert v_blob["layer"] == args.layer and v_blob["position"] == args.position
    print(f"v_noop: L{v_blob['layer']} / pos {v_blob['position']:+d}  "
          f"|v|={v_blob['norm']:.3f}  source_N={v_blob['n_pairs']}")

    # --- Collect scores from each cache ---
    # cells[ (code, transcript) ][ task_id ] = {projection, score, argmax, ...}
    cells: dict[tuple[str, str], dict[str, dict]] = {
        ("buggy", "buggy"):   {},  # (B,B): from orig buggy__code_tests
        ("fixed", "fixed"):   {},  # (F,F): from orig fixed__code_tests
        ("buggy", "fixed"):   {},  # (B,F): from swap buggy__code_tests_swapped
        ("fixed", "buggy"):   {},  # (F,B): from swap fixed__code_tests_swapped
    }

    print("\nLoading original (matched) cache...")
    for pt in sorted(args.orig_cache.rglob("*__code_tests.pt")):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        entry = _load_resid_score(pt, args.layer, args.position, v_unit)
        # cond == "buggy" → (B,B); cond == "fixed" → (F,F)
        cells[(cond, cond)][entry["task_id"]] = entry
    print(f"  (B,B): {len(cells[('buggy','buggy')])}  (F,F): {len(cells[('fixed','fixed')])}")

    print("\nLoading swapped (contradictory) cache...")
    for pt in sorted(args.swap_cache.rglob("*__code_tests_swapped.pt")):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        entry = _load_resid_score(pt, args.layer, args.position, v_unit)
        # swap variant: cond=="buggy" → buggy code + fixed transcript → (B,F)
        #               cond=="fixed" → fixed code + buggy transcript → (F,B)
        if cond == "buggy":
            cells[("buggy", "fixed")][entry["task_id"]] = entry
        else:
            cells[("fixed", "buggy")][entry["task_id"]] = entry
    print(f"  (B,F): {len(cells[('buggy','fixed')])}  (F,B): {len(cells[('fixed','buggy')])}")

    # --- Restrict to tasks present in ALL 4 cells ---
    common = set.intersection(*(set(d.keys()) for d in cells.values()))
    print(f"\nTasks present in all 4 cells: {len(common)}")
    if len(common) < 10:
        print("error: too few tasks present in all 4 cells; refusing to compute stats",
              file=sys.stderr)
        return 2

    common_ids = sorted(common)

    def cell_scores(code_c, tx_c):
        return np.asarray([cells[(code_c, tx_c)][t]["score"] for t in common_ids])

    bb = cell_scores("buggy", "buggy")
    ff = cell_scores("fixed", "fixed")
    bf = cell_scores("buggy", "fixed")
    fb = cell_scores("fixed", "buggy")

    # --- 2x2 means + main effects ---
    cell_mean = {
        "buggy_code_buggy_tx_BB": float(bb.mean()),
        "buggy_code_fixed_tx_BF": float(bf.mean()),
        "fixed_code_buggy_tx_FB": float(fb.mean()),
        "fixed_code_fixed_tx_FF": float(ff.mean()),
    }
    main_code = float(0.5 * ((bb.mean() + bf.mean()) - (fb.mean() + ff.mean())))
    main_tx = float(0.5 * ((bb.mean() + fb.mean()) - (bf.mean() + ff.mean())))
    interaction = float(0.25 * ((bb.mean() - bf.mean()) - (fb.mean() - ff.mean())))

    # --- Bootstrap CIs over paired tasks ---
    rng = np.random.default_rng(args.seed)
    N = len(common_ids)
    boots = []
    for _ in range(args.bootstrap):
        idx = rng.integers(0, N, size=N)
        b_bb, b_ff, b_bf, b_fb = bb[idx], ff[idx], bf[idx], fb[idx]
        b_main_code = 0.5 * ((b_bb.mean() + b_bf.mean())
                             - (b_fb.mean() + b_ff.mean()))
        b_main_tx = 0.5 * ((b_bb.mean() + b_fb.mean())
                           - (b_bf.mean() + b_ff.mean()))
        b_interaction = 0.25 * ((b_bb.mean() - b_bf.mean())
                                - (b_fb.mean() - b_ff.mean()))
        boots.append([b_main_code, b_main_tx, b_interaction])
    boots = np.asarray(boots)
    ci_code = (float(np.percentile(boots[:, 0], 2.5)),
               float(np.percentile(boots[:, 0], 97.5)))
    ci_tx = (float(np.percentile(boots[:, 1], 2.5)),
             float(np.percentile(boots[:, 1], 97.5)))
    ci_inter = (float(np.percentile(boots[:, 2], 2.5)),
                float(np.percentile(boots[:, 2], 97.5)))

    # --- AUCs ---
    # On ALL 4 cells (2N positives, 2N negatives), under each labelling.
    all_scores = np.concatenate([bb, bf, fb, ff])
    code_labels = np.concatenate([
        np.ones(N, int), np.ones(N, int),       # bb, bf → code is buggy
        np.zeros(N, int), np.zeros(N, int),     # fb, ff → code is fixed
    ])
    tx_labels = np.concatenate([
        np.ones(N, int), np.zeros(N, int),      # bb, bf → tx is failing/passing
        np.ones(N, int), np.zeros(N, int),      # fb, ff → tx is failing/passing
    ])
    auc_code_all4 = float(roc_auc_score(code_labels, all_scores))
    auc_tx_all4 = float(roc_auc_score(tx_labels, all_scores))

    # On SWAPPED-ONLY cells (cleanest test): if code and transcript disagree
    # within a single prompt, which one does v_noop follow?
    swap_scores = np.concatenate([bf, fb])
    swap_code_labels = np.concatenate([np.ones(N, int), np.zeros(N, int)])  # bf → buggy code, fb → fixed code
    swap_tx_labels = np.concatenate([np.zeros(N, int), np.ones(N, int)])    # bf → passing tx, fb → failing tx
    auc_code_swap = float(roc_auc_score(swap_code_labels, swap_scores))
    auc_tx_swap = float(roc_auc_score(swap_tx_labels, swap_scores))

    # --- Argmax distribution per cell ---
    argmax_dist = {}
    for (code_c, tx_c) in cells.keys():
        c = Counter(cells[(code_c, tx_c)][t]["argmax"] for t in common_ids)
        argmax_dist[f"{code_c}_code_{tx_c}_tx"] = dict(c)

    # --- Verdict ---
    delta_ratio = abs(main_code) / max(abs(main_tx), 1e-9)
    if delta_ratio > 3:
        verdict = "CODE-DRIVEN (paper claim survives)"
    elif delta_ratio < 1/3:
        verdict = "TRANSCRIPT-DRIVEN (paper needs reframing)"
    else:
        verdict = "MIXED (both code and transcript contribute)"

    # --- Print summary ---
    print(f"\n=== 2x2 cell means (score = -projection onto v_noop) ===")
    print(f"  (B,B) buggy code + buggy tx : {bb.mean():+.3f}")
    print(f"  (B,F) buggy code + fixed tx : {bf.mean():+.3f}")
    print(f"  (F,B) fixed code + buggy tx : {fb.mean():+.3f}")
    print(f"  (F,F) fixed code + fixed tx : {ff.mean():+.3f}")
    print(f"\n=== Main effects (positive = pushes toward higher score = more buggy-like) ===")
    print(f"  ΔCode       (buggy − fixed code, avg over tx)    : {main_code:+.3f}  CI {ci_code}")
    print(f"  ΔTranscript (failing − passing tx, avg over code): {main_tx:+.3f}  CI {ci_tx}")
    print(f"  Interaction                                      : {interaction:+.3f}  CI {ci_inter}")
    print(f"\n=== AUCs ===")
    print(f"  ALL 4 cells (2x2N labels)")
    print(f"    code-label  AUC: {auc_code_all4:.4f}  (1.0 = pure code-driven)")
    print(f"    tx-label    AUC: {auc_tx_all4:.4f}    (1.0 = pure tx-driven)")
    print(f"  SWAPPED ONLY (off-diagonal: B+F and F+B)")
    print(f"    code-label  AUC: {auc_code_swap:.4f}  (1.0 = pure code-driven)")
    print(f"    tx-label    AUC: {auc_tx_swap:.4f}    (1.0 = pure tx-driven)")
    print(f"\n=== Verdict ===")
    print(f"  |ΔCode|/|ΔTranscript| = {delta_ratio:.3f}")
    print(f"  {verdict}")

    # --- Persist ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "contradictory_transcript_2x2.json"
    out = {
        "config": {
            "orig_cache": str(args.orig_cache),
            "swap_cache": str(args.swap_cache),
            "v_noop": str(args.v_noop),
            "layer": args.layer,
            "position": args.position,
            "seed": args.seed,
            "bootstrap": args.bootstrap,
            "n_tasks_common": N,
        },
        "cell_means": cell_mean,
        "main_effects": {
            "delta_code": main_code,
            "delta_code_ci95": ci_code,
            "delta_transcript": main_tx,
            "delta_transcript_ci95": ci_tx,
            "interaction": interaction,
            "interaction_ci95": ci_inter,
            "ratio_code_over_tx": float(delta_ratio),
        },
        "aucs": {
            "code_label_all4": auc_code_all4,
            "tx_label_all4": auc_tx_all4,
            "code_label_swap_only": auc_code_swap,
            "tx_label_swap_only": auc_tx_swap,
        },
        "argmax_distribution_by_cell": argmax_dist,
        "per_task": {
            t: {
                "BB_score": float(cells[("buggy", "buggy")][t]["score"]),
                "FF_score": float(cells[("fixed", "fixed")][t]["score"]),
                "BF_score": float(cells[("buggy", "fixed")][t]["score"]),
                "FB_score": float(cells[("fixed", "buggy")][t]["score"]),
                "BB_argmax": cells[("buggy", "buggy")][t]["argmax"],
                "FF_argmax": cells[("fixed", "fixed")][t]["argmax"],
                "BF_argmax": cells[("buggy", "fixed")][t]["argmax"],
                "FB_argmax": cells[("fixed", "buggy")][t]["argmax"],
            }
            for t in common_ids
        },
        "verdict": verdict,
    }
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
