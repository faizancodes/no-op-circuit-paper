#!/usr/bin/env python3
"""Phase 4: compare residual monitor vs regex vs bag-of-words on noisy transcripts.

Reads results/noisy_monitor/<tag>_monitor_projections.json (monitor scores from
modal_app/noisy_monitor.py) and recomputes the regex classifiers (the paper's
App. G.8 baselines) and a bag-of-words logistic baseline on the SAME transformed
transcripts, locally. Reports ROC-AUC per method per variant.

Usage: python scripts/analyze_noisy_monitor.py --tag qwen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from no_op_circuit.config import DATA_DIR
from no_op_circuit.dataset import iter_tasks
from no_op_circuit.dataset.schema import apply_transcript_noise

VAR_TRANSFORM = {
    "code_tests": None,
    "code_tests_noisy_flaky": "flaky",
    "code_tests_many_passing": "many_passing",
    "code_tests_truncated": "truncated",
    "code_tests_summary_only": "summary_only",
}


def transcript_for(task, cond, kind):
    t = task.test_output(cond) or ""
    return apply_transcript_noise(t, kind) if kind else t


def density(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    hits = sum(any(tok in ln for tok in ("FAILED", "AssertionError", "Traceback")) for ln in lines)
    return hits / len(lines)


def auc(scores, labels):
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen")
    ap.add_argument("--tasks", default="real")
    args = ap.parse_args()

    proj = json.load(open(f"results/noisy_monitor/{args.tag}_monitor_projections.json"))
    # group monitor scores: variant -> task_id -> {cond: score}
    mon: dict[str, dict[str, dict[str, float]]] = {}
    for r in proj["rows"]:
        mon.setdefault(r["variant"], {}).setdefault(r["task_id"], {})[r["condition"]] = r["score"]

    tasks = {t.task_id: t for t in iter_tasks(tasks_dir=DATA_DIR / ("real_tasks" if args.tasks == "real" else "tasks"))}

    table = {}
    for variant, kind in VAR_TRANSFORM.items():
        if variant not in mon:
            continue
        paired = [(tid, d["buggy"], d["fixed"]) for tid, d in mon[variant].items() if "buggy" in d and "fixed" in d]
        paired.sort()
        tids = [t for t, _, _ in paired]
        # monitor
        mscore = [b for _, b, _ in paired] + [f for _, _, f in paired]
        labels = [1] * len(paired) + [0] * len(paired)
        # texts for regex / bow (same tasks/order)
        txt_b = [transcript_for(tasks[t], "buggy", kind) for t in tids]
        txt_f = [transcript_for(tasks[t], "fixed", kind) for t in tids]
        texts = txt_b + txt_f
        contains = [int("FAILED" in x) for x in texts]
        nfail = [x.count("FAILED") for x in texts]
        dens = [density(x) for x in texts]
        # bag-of-words logistic, 5-fold CV AUC
        X = CountVectorizer(min_df=1).fit_transform(texts)
        try:
            bow_proba = cross_val_predict(LogisticRegression(max_iter=1000), X, labels, cv=5, method="predict_proba")[:, 1]
            bow_auc = auc(bow_proba, labels)
        except Exception as e:  # noqa: BLE001
            bow_auc = float("nan")
        table[variant] = {
            "n_pairs": len(paired),
            "monitor_auc": auc(mscore, labels),
            "regex_contains_FAILED_auc": auc(contains, labels),
            "regex_n_FAILED_auc": auc(nfail, labels),
            "regex_density_auc": auc(dens, labels),
            "bow_auc": bow_auc,
        }

    out = Path("results/noisy_monitor")
    (out / f"{args.tag}_noisy_comparison.json").write_text(json.dumps(table, indent=2))
    # pretty print
    cols = ["monitor_auc", "regex_contains_FAILED_auc", "regex_n_FAILED_auc", "regex_density_auc", "bow_auc"]
    print(f"{'variant':28s} {'N':>4} " + " ".join(f"{c.replace('_auc',''):>22}" for c in cols))
    for v, m in table.items():
        print(f"{v:28s} {m['n_pairs']:>4} " + " ".join(f"{m[c]:>22.3f}" for c in cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
