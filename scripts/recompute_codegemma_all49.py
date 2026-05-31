#!/usr/bin/env python3
"""Recompute CodeGemma downstream analyses with the all-49 v_noop_cg.

Background: the §5.1 headline was updated from 0.933 → 0.950 by switching
the CodeGemma v_noop_cg derivation from a 20-task responsive subset to
all 49 toys. The downstream analyses (per-repo breakdown G.3,
edit-action veto G.10, threshold sweep G.11) still used the OLD
responsive-subset direction. This script recomputes them with the
all-49 direction at (L26, pos=-1) and emits a JSON with all the
numbers we need to propagate into the paper.

Outputs:
  results/monitor_real/codegemma_all49_recompute.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


TOY_DIR = Path("results/cache-codegemma_7b_it-20260516T031036Z/cache-codegemma_7b_it-20260516T031036Z")
REAL_DIR = Path("results/cache-real-codegemma-n500-20260516T235731Z/cache-real-codegemma-n500-20260516T235731Z")
META_DIR = Path("data/real_tasks")
OUT = Path("results/monitor_real/codegemma_all49_recompute.json")


def _repo_of(task_id: str) -> str:
    p = (META_DIR / task_id / "meta.yaml")
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line.startswith("swe_bench_repo:"):
                return line.split(":", 1)[1].strip().replace("/", "_")
    parts = task_id.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else task_id


def _argmax_action(action_logits: dict) -> str:
    return max(action_logits.items(), key=lambda kv: kv[1]["logit"])[0]


def main() -> int:
    print("[1/4] Derive v_noop_all49 at (L26, pos=-1) from 49 CodeGemma toys.",
          file=sys.stderr)
    by_task_t = {}
    for pt in sorted(TOY_DIR.rglob("*__code_tests.pt")):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        by_task_t.setdefault(blob["task_id"], {})[cond] = (
            blob["resid_pre"][26, 0, -1, :].float().numpy()
        )
    paired = sorted(t for t, s in by_task_t.items() if "buggy" in s and "fixed" in s)
    assert len(paired) == 49, f"expected 49 toy pairs, got {len(paired)}"
    buggy = np.stack([by_task_t[t]["buggy"] for t in paired])
    fixed = np.stack([by_task_t[t]["fixed"] for t in paired])
    v_noop_raw = fixed.mean(0) - buggy.mean(0)
    v_noop = v_noop_raw / np.linalg.norm(v_noop_raw)
    print(f"  v_noop: |v|_raw = {np.linalg.norm(v_noop_raw):.4f}", file=sys.stderr)

    print("[2/4] Score 497 real CodeGemma pairs + capture argmax actions.",
          file=sys.stderr)
    rows = []
    for pt in sorted(REAL_DIR.rglob("*__code_tests.pt")):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        resid = blob["resid_pre"][26, 0, -1, :].float().numpy()
        proj = float(resid @ v_noop)
        argmax = _argmax_action(blob["action_logits"])
        rows.append({
            "task_id": blob["task_id"],
            "condition": cond,
            "repo": _repo_of(blob["task_id"]),
            "projection": proj,
            "argmax_action": argmax,
        })

    # Aggregate paired tasks
    by_task = defaultdict(dict)
    for r in rows:
        by_task[r["task_id"]][r["condition"]] = r
    paired_r = sorted(t for t, s in by_task.items() if "buggy" in s and "fixed" in s)
    N = len(paired_r)
    print(f"  paired real tasks: {N}", file=sys.stderr)

    proj_buggy = np.array([by_task[t]["buggy"]["projection"] for t in paired_r])
    proj_fixed = np.array([by_task[t]["fixed"]["projection"] for t in paired_r])
    argmax_buggy = [by_task[t]["buggy"]["argmax_action"] for t in paired_r]
    argmax_fixed = [by_task[t]["fixed"]["argmax_action"] for t in paired_r]
    repos = [by_task[t]["buggy"]["repo"] for t in paired_r]

    # ===== Headline metrics =====
    print("[3/4] Headline AUC + balanced-accuracy threshold + 95% CIs.",
          file=sys.stderr)
    labels = np.r_[np.ones(N, int), np.zeros(N, int)]
    score = np.r_[-proj_buggy, -proj_fixed]  # convention: higher = more buggy
    auc = float(roc_auc_score(labels, score))
    ap = float(average_precision_score(labels, score))

    rng = np.random.default_rng(0)
    aucs, aps = [], []
    for _ in range(10000):
        idx = rng.integers(0, N, N)
        sc = np.r_[-proj_buggy[idx], -proj_fixed[idx]]
        aucs.append(float(roc_auc_score(labels, sc)))
        aps.append(float(average_precision_score(labels, sc)))
    auc_lo, auc_hi = float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
    ap_lo, ap_hi = float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))

    # Balanced-acc threshold
    thresholds = np.unique(score)
    best_t, best_bacc = None, -1.0
    for t in thresholds:
        pred = (score >= t).astype(int)
        tpr = ((pred == 1) & (labels == 1)).sum() / max(1, (labels == 1).sum())
        tnr = ((pred == 0) & (labels == 0)).sum() / max(1, (labels == 0).sum())
        bacc = 0.5 * (tpr + tnr)
        if bacc > best_bacc:
            best_bacc, best_t = bacc, float(t)
    pred = (score >= best_t).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    # In score-space the threshold is on -projection. Convert to projection-space:
    # vetoing means projection >= proj_T  ⇔  -projection <= -proj_T  ⇔  score < t
    proj_T = -best_t
    print(f"  balanced-T (score-space): {best_t:.4f}; "
          f"projection-T: {proj_T:.4f}", file=sys.stderr)
    print(f"  AUC = {auc:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]; "
          f"AP = {ap:.4f}; FER = {fp/(fp+tn):.4f}", file=sys.stderr)

    # ===== Per-repo metrics =====
    by_repo = defaultdict(lambda: {"buggy_proj": [], "fixed_proj": [],
                                    "argmax_buggy": [], "argmax_fixed": [],
                                    "tasks": []})
    for i, r in enumerate(repos):
        by_repo[r]["buggy_proj"].append(proj_buggy[i])
        by_repo[r]["fixed_proj"].append(proj_fixed[i])
        by_repo[r]["argmax_buggy"].append(argmax_buggy[i])
        by_repo[r]["argmax_fixed"].append(argmax_fixed[i])
        by_repo[r]["tasks"].append(paired_r[i])

    per_repo_rows = []
    pooled_fer_balanced = 0
    pooled_fer_per_repo_cal = 0
    pooled_buggy_correct_balanced = 0
    pooled_buggy_correct_per_repo_cal = 0
    pooled_n_fixed = 0
    pooled_n_buggy = 0

    for repo in sorted(by_repo.keys()):
        d = by_repo[repo]
        n = len(d["buggy_proj"])
        if n < 5:
            continue
        b = np.array(d["buggy_proj"]); f = np.array(d["fixed_proj"])
        # AUC (positive class = buggy)
        try:
            r_lab = np.r_[np.ones(n), np.zeros(n)]
            r_sc = np.r_[-b, -f]
            r_auc = float(roc_auc_score(r_lab, r_sc))
        except Exception:
            r_auc = None

        # FER at global balanced threshold
        # In projection-space: vetoed if projection >= proj_T
        # In score-space (score = -projection): vetoed if score < -proj_T
        veto_thr = proj_T
        n_fixed_correctly_vetoed = int(np.sum(f >= veto_thr))
        n_fixed_missed = n - n_fixed_correctly_vetoed
        fer_global = n_fixed_missed / n
        # Buggy correctly NOT vetoed: projection < proj_T
        n_buggy_kept = int(np.sum(b < veto_thr))
        recall_global = n_buggy_kept / n

        # Per-repo balanced threshold
        labs = np.r_[np.ones(n, int), np.zeros(n, int)]
        scs = np.r_[-b, -f]
        ts = np.unique(scs)
        best_r_t, best_r_bacc = None, -1.0
        for t in ts:
            pp = (scs >= t).astype(int)
            tpr = ((pp == 1) & (labs == 1)).sum() / max(1, n)
            tnr = ((pp == 0) & (labs == 0)).sum() / max(1, n)
            bacc = 0.5 * (tpr + tnr)
            if bacc > best_r_bacc:
                best_r_bacc, best_r_t = bacc, float(t)
        per_repo_proj_T = -best_r_t
        n_fixed_correctly_vetoed_pr = int(np.sum(f >= per_repo_proj_T))
        n_fixed_missed_pr = n - n_fixed_correctly_vetoed_pr
        fer_pr = n_fixed_missed_pr / n
        n_buggy_kept_pr = int(np.sum(b < per_repo_proj_T))

        per_repo_rows.append({
            "repo": repo,
            "n_pairs": int(n),
            "auc": r_auc,
            "global_threshold_fer": fer_global,
            "global_threshold_n_fixed_missed": n_fixed_missed,
            "per_repo_threshold": per_repo_proj_T,
            "per_repo_threshold_fer": fer_pr,
            "per_repo_threshold_n_fixed_missed": n_fixed_missed_pr,
        })

        pooled_n_fixed += n
        pooled_n_buggy += n
        pooled_fer_balanced += n_fixed_missed
        pooled_fer_per_repo_cal += n_fixed_missed_pr
        pooled_buggy_correct_balanced += n_buggy_kept
        pooled_buggy_correct_per_repo_cal += n_buggy_kept_pr

    pooled = {
        "n_pairs_in_repos_ge5": pooled_n_fixed,
        "global_threshold_pooled_fer": pooled_fer_balanced / pooled_n_fixed,
        "global_threshold_pooled_recall": pooled_buggy_correct_balanced / pooled_n_buggy,
        "per_repo_calibration_pooled_fer": pooled_fer_per_repo_cal / pooled_n_fixed,
        "per_repo_calibration_pooled_recall": pooled_buggy_correct_per_repo_cal / pooled_n_buggy,
    }

    # ===== Gate-veto simulation (G.10 equivalent) =====
    # For each task and condition, did the model argmax `edit`?
    # If yes, would the projection gate veto it (projection >= proj_T)?
    # buggy + argmax=edit + gate_vetoes = "useful edit blocked"  (bad)
    # fixed + argmax=edit + gate_vetoes = "spurious edit blocked" (good)
    print("[4/4] Gate-veto simulation at global balanced threshold.",
          file=sys.stderr)
    useful_proposed = sum(1 for i in range(N) if argmax_buggy[i] == "edit")
    spurious_proposed = sum(1 for i in range(N) if argmax_fixed[i] == "edit")
    useful_blocked = sum(
        1 for i in range(N)
        if argmax_buggy[i] == "edit" and proj_buggy[i] >= proj_T
    )
    spurious_blocked = sum(
        1 for i in range(N)
        if argmax_fixed[i] == "edit" and proj_fixed[i] >= proj_T
    )
    gate_summary = {
        "threshold_projection_space": proj_T,
        "useful_proposed": useful_proposed,
        "spurious_proposed": spurious_proposed,
        "useful_blocked": useful_blocked,
        "spurious_blocked": spurious_blocked,
        "useful_edit_loss":
            useful_blocked / useful_proposed if useful_proposed else 0.0,
        "spurious_edit_reduction":
            spurious_blocked / spurious_proposed if spurious_proposed else 0.0,
        "final_spurious_rate":
            (spurious_proposed - spurious_blocked) / N,
    }
    print(f"  spurious blocked: {spurious_blocked}/{spurious_proposed} "
          f"= {gate_summary['spurious_edit_reduction']:.4f}; "
          f"useful blocked: {useful_blocked}/{useful_proposed} "
          f"= {gate_summary['useful_edit_loss']:.4f}; "
          f"final spurious rate: {gate_summary['final_spurious_rate']:.4f}",
          file=sys.stderr)

    # Argmax distribution by condition
    from collections import Counter
    argmax_dist = {
        "buggy": dict(Counter(argmax_buggy)),
        "fixed": dict(Counter(argmax_fixed)),
    }
    print(f"  argmax buggy: {argmax_dist['buggy']}", file=sys.stderr)
    print(f"  argmax fixed: {argmax_dist['fixed']}", file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "config": {
            "model": "codegemma",
            "cell": "(L26, pos=-1)",
            "v_noop": "derived from all 49 toy pairs",
            "n_paired_real": N,
        },
        "headline": {
            "auc": auc,
            "auc_ci95": [auc_lo, auc_hi],
            "ap": ap,
            "ap_ci95": [ap_lo, ap_hi],
            "balanced_threshold_score_space": best_t,
            "balanced_threshold_projection_space": proj_T,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "false_edit_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        },
        "per_repo": per_repo_rows,
        "per_repo_pooled": pooled,
        "gate_simulation": gate_summary,
        "argmax_distribution": argmax_dist,
    }, indent=2))
    print(f"\nwrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
