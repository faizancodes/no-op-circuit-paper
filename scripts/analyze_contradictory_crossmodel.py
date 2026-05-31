#!/usr/bin/env python3
"""Analyse the cross-model contradictory-transcript control (§5.2 replication).

Reads results/monitor_real/contradictory_crossmodel_*.json (from
modal_app/contradictory_crossmodel.py) and, for each model, computes the
identical statistics as scripts/contradictory_transcript_analysis.py:

  cell means BB/BF/FB/FF  (score = -projection; higher = more buggy-like)
  ΔCode       = 0.5 * ((BB+BF) - (FB+FF))   effect of buggy vs fixed CODE
  ΔTranscript = 0.5 * ((BB+FB) - (BF+FF))   effect of failing vs passing TRANSCRIPT
  interaction = 0.25 * ((BB-BF) - (FB-FF))
  bootstrap 95% CIs over paired tasks (B=10000, seed=0)
  AUCs under code-labels vs transcript-labels (all-4 cells and swapped-only)
  verdict: |ΔCode|/|ΔTranscript|

Reference (existing Qwen §5.2): ΔTranscript +4.73, ΔCode +0.002 (CI crosses 0).

Usage:
    .venv/bin/python scripts/analyze_contradictory_crossmodel.py [path.json]
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def analyse_model(rows: list[dict]) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    by_cell = defaultdict(dict)  # cell -> {task_id: score}
    argmax_by_cell = defaultdict(Counter)
    for r in rows:
        by_cell[r["cell"]][r["task_id"]] = r["score"]
        argmax_by_cell[r["cell"]][r["argmax_action"]] += 1

    common = set.intersection(*(set(by_cell[c].keys()) for c in ("BB", "FF", "BF", "FB")))
    ids = sorted(common)
    n = len(ids)

    def vec(cell):
        return np.asarray([by_cell[cell][t] for t in ids])

    bb, ff, bf, fb = vec("BB"), vec("FF"), vec("BF"), vec("FB")

    main_code = float(0.5 * ((bb.mean() + bf.mean()) - (fb.mean() + ff.mean())))
    main_tx = float(0.5 * ((bb.mean() + fb.mean()) - (bf.mean() + ff.mean())))
    interaction = float(0.25 * ((bb.mean() - bf.mean()) - (fb.mean() - ff.mean())))

    rng = np.random.default_rng(0)
    boots = []
    for _ in range(10000):
        idx = rng.integers(0, n, n)
        b_bb, b_ff, b_bf, b_fb = bb[idx], ff[idx], bf[idx], fb[idx]
        boots.append([
            0.5 * ((b_bb.mean() + b_bf.mean()) - (b_fb.mean() + b_ff.mean())),
            0.5 * ((b_bb.mean() + b_fb.mean()) - (b_bf.mean() + b_ff.mean())),
        ])
    boots = np.asarray(boots)
    ci_code = (float(np.percentile(boots[:, 0], 2.5)), float(np.percentile(boots[:, 0], 97.5)))
    ci_tx = (float(np.percentile(boots[:, 1], 2.5)), float(np.percentile(boots[:, 1], 97.5)))

    # swapped-only AUC: when code and transcript disagree, which does v_noop follow?
    swap_scores = np.concatenate([bf, fb])
    swap_code_labels = np.concatenate([np.ones(n, int), np.zeros(n, int)])   # bf buggy code, fb fixed code
    swap_tx_labels = np.concatenate([np.zeros(n, int), np.ones(n, int)])     # bf passing tx, fb failing tx
    auc_code_swap = float(roc_auc_score(swap_code_labels, swap_scores))
    auc_tx_swap = float(roc_auc_score(swap_tx_labels, swap_scores))

    ratio = abs(main_code) / max(abs(main_tx), 1e-9)
    verdict = ("TRANSCRIPT-DRIVEN" if ratio < 1 / 3 else
               "CODE-DRIVEN" if ratio > 3 else "MIXED")

    return {
        "n": n,
        "cell_means": {"BB": float(bb.mean()), "BF": float(bf.mean()),
                       "FB": float(fb.mean()), "FF": float(ff.mean())},
        "delta_code": main_code, "delta_code_ci95": ci_code,
        "delta_transcript": main_tx, "delta_transcript_ci95": ci_tx,
        "interaction": interaction,
        "auc_code_swap_only": auc_code_swap,
        "auc_transcript_swap_only": auc_tx_swap,
        "verdict": verdict,
        "argmax_by_cell": {c: dict(argmax_by_cell[c]) for c in ("BB", "BF", "FB", "FF")},
    }


def main(argv=None) -> int:
    path = (argv or sys.argv[1:])[0] if (argv or sys.argv[1:]) else ""
    if not path:
        cands = sorted(glob.glob("results/monitor_real/contradictory_crossmodel_*.json"))
        if not cands:
            print("no contradictory_crossmodel_*.json found", file=sys.stderr)
            return 1
        path = cands[-1]
    blob = json.loads(Path(path).read_text())
    print(f"loaded {path}\n")
    for model_name, payload in blob["results"].items():
        a = analyse_model(payload["rows"])
        print(f"=== {model_name}  (L{payload['layer']}/pos-1, N={a['n']} paired) ===")
        cm = a["cell_means"]
        print(f"  cell means (score = -proj; higher = buggy-like):")
        print(f"    (B,B) buggy code+buggy tx : {cm['BB']:+.3f}")
        print(f"    (B,F) buggy code+fixed tx : {cm['BF']:+.3f}")
        print(f"    (F,B) fixed code+buggy tx : {cm['FB']:+.3f}")
        print(f"    (F,F) fixed code+fixed tx : {cm['FF']:+.3f}")
        print(f"  ΔCode       (buggy-fixed code, avg over tx) : "
              f"{a['delta_code']:+.3f}  CI95 [{a['delta_code_ci95'][0]:+.3f}, {a['delta_code_ci95'][1]:+.3f}]")
        print(f"  ΔTranscript (failing-passing tx, avg / code): "
              f"{a['delta_transcript']:+.3f}  CI95 [{a['delta_transcript_ci95'][0]:+.3f}, {a['delta_transcript_ci95'][1]:+.3f}]")
        print(f"  interaction : {a['interaction']:+.3f}")
        print(f"  swapped-only AUC  code-labels: {a['auc_code_swap_only']:.3f}   "
              f"transcript-labels: {a['auc_transcript_swap_only']:.3f}")
        print(f"  |ΔCode|/|ΔTranscript| = {abs(a['delta_code'])/max(abs(a['delta_transcript']),1e-9):.3f}  -> {a['verdict']}")
        print()
    print("Reference (existing Qwen §5.2): ΔTranscript +4.73, ΔCode +0.002 (CI crosses 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
