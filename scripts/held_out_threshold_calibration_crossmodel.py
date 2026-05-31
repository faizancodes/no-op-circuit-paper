#!/usr/bin/env python3
"""Cross-model held-out operating-point calibration (CPU, no model forwards).

Step-3 companion to scripts/held_out_threshold_calibration.py (which covers
Qwen on the canonical-menu §5.1 set via cached projections). Those raw §5.1
per-task projection caches (results/cache-real-*) are not retained locally, so
a canonical-menu random-split/LOO classifier table for CodeGemma/DeepSeek
cannot be reproduced without re-running Modal forwards.

What we CAN do from existing artifacts: the G.17 held-out-paraphrase score
JSONs (results/heldout_paraphrase_robustness/{model}_scores.json) hold a genuine
disjoint held-out set per model -- train-template prompts and held-out-template
prompts at the §4.3 reported cell, with per-prompt residual_score and
label_failing. This script fits the balanced-accuracy threshold on the
TRAIN-template prompts and applies it, frozen, to the HELD-OUT-template prompts,
reporting out-of-sample precision/recall/accuracy/fixed-condition FPR for all
three models. This is a true held-out operating point (threshold never sees the
held-out templates) in the paraphrase setting -- distinct from, and not a
substitute for, the §5.1 canonical-menu calibration.

Inputs:  results/heldout_paraphrase_robustness/{qwen,codegemma,deepseek}_scores.json
Outputs: results/monitor_real/held_out_threshold_calibration_crossmodel.{json,md}
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results/heldout_paraphrase_robustness"
OUT_JSON = REPO / "results/monitor_real/held_out_threshold_calibration_crossmodel.json"
OUT_MD = REPO / "results/monitor_real/held_out_threshold_calibration_crossmodel.md"
MODELS = ("qwen", "codegemma", "deepseek")
REPORTED_HELDOUT_AUC = {"qwen": 0.943, "codegemma": 0.649, "deepseek": 0.619}


def _fit_balanced_threshold(scores, labels):
    import numpy as np
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores)
    bal = 0.5 * (tpr + (1.0 - fpr))
    return float(thr[int(np.argmax(bal))])


def _auc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def _opmetrics(scores, labels, threshold):
    import numpy as np
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "fixed_condition_fpr": fp / max(fp + tn, 1),
    }


def calibrate(model: str) -> dict:
    d = json.loads((SRC / f"{model}_scores.json").read_text())
    tr = d["train"]
    ho = d["heldout"]
    tr_s = [r["residual_score"] for r in tr]
    tr_y = [int(r["label_failing"]) for r in tr]
    ho_s = [r["residual_score"] for r in ho]
    ho_y = [int(r["label_failing"]) for r in ho]

    thr = _fit_balanced_threshold(tr_s, tr_y)
    train_m = _opmetrics(tr_s, tr_y, thr)
    held_m = _opmetrics(ho_s, ho_y, thr)
    return {
        "n_train_prompts": len(tr_s),
        "n_heldout_prompts": len(ho_s),
        "train_fit_threshold": thr,
        "train_auc": _auc(tr_s, tr_y),
        "heldout_auc": _auc(ho_s, ho_y),
        "heldout_auc_reported_g17": REPORTED_HELDOUT_AUC.get(model),
        "in_sample_train_operating_point": train_m,
        "heldout_operating_point_frozen_threshold": held_m,
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    res = {m: calibrate(m) for m in MODELS}
    OUT_JSON.write_text(json.dumps(res, indent=2))

    L = ["# Cross-model held-out operating-point calibration (paraphrase setting)", "",
         "Threshold fit on TRAIN-template prompts (balanced accuracy), applied frozen to ",
         "HELD-OUT-template prompts. Source: `results/heldout_paraphrase_robustness/"
         "{model}_scores.json`. CPU only, no model forwards.", "",
         "| model | train thr | held-out AUC | held-out precision | held-out recall | "
         "held-out fixed-cond FPR | held-out acc |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for m in MODELS:
        r = res[m]
        h = r["heldout_operating_point_frozen_threshold"]
        L.append(f"| {m} | {r['train_fit_threshold']:+.3f} | {r['heldout_auc']:.3f} | "
                 f"{h['precision']:.3f} | {h['recall']:.3f} | {h['fixed_condition_fpr']:.3f} | "
                 f"{h['accuracy']:.3f} |")
    L += ["", "Held-out AUC sanity-check vs G.17 reported: " +
          ", ".join(f"{m} {res[m]['heldout_auc']:.3f} (reported {REPORTED_HELDOUT_AUC[m]})"
                    for m in MODELS) + ".",
          "", "Note: paraphrase-setting held-out calibration; NOT the §5.1 canonical-menu",
          "calibration (raw §5.1 per-task caches are not retained locally)."]
    OUT_MD.write_text("\n".join(L) + "\n")

    print("=== cross-model held-out operating-point calibration (paraphrase) ===")
    for m in MODELS:
        r = res[m]; h = r["heldout_operating_point_frozen_threshold"]
        print(f"{m:10} train_thr={r['train_fit_threshold']:+.3f} "
              f"held_AUC={r['heldout_auc']:.3f} (rep {REPORTED_HELDOUT_AUC[m]}) "
              f"held: P={h['precision']:.3f} R={h['recall']:.3f} "
              f"FPR={h['fixed_condition_fpr']:.3f} acc={h['accuracy']:.3f}")
    print(f"\nwrote {OUT_JSON.relative_to(REPO)} and {OUT_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
