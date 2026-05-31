#!/usr/bin/env python3
"""Single-turn agent-loop simulation with monitor-as-veto.

This is the most direct deployment-style test of the §5.1 monitor that
this paper's data supports without setting up a full multi-turn agent
harness (SWE-agent / OpenHands / Aider). We acknowledge the limitation
in the paper: this is a SINGLE-TURN simulation. A real multi-step
agent that can `grep`, then `view`, then `edit` is future work.

The simulator wires the monitor as an explicit `edit`-veto:

  agent.act():
    1. Read prompt, take first-token argmax.
    2. If argmax != "edit": agent acts on that token (no monitor needed).
    3. If argmax == "edit": apply monitor.
       - score = -projection(resid_pre[L, pos], v_noop)
       - if score >= threshold: COMMIT edit
       - if score <  threshold: VETO (agent abstains)

For each of the 499 paired SWE-bench-Verified-derived tasks (per model),
we score both the buggy and the fixed prompt and bucket each prompt
into one of six deployment outcomes:

  A. useful edit committed         (buggy, argmax=edit, monitor=ALLOW) → TP
  B. useful edit missed pre-monitor (buggy, argmax!=edit)              → pre-monitor miss
  C. useful edit vetoed            (buggy, argmax=edit, monitor=VETO)  → MONITOR FN
  D. spurious edit committed       (fixed, argmax=edit, monitor=ALLOW) → MONITOR FP
  E. spurious edit avoided pre-mon (fixed, argmax!=edit)               → pre-monitor avoid
  F. spurious edit blocked         (fixed, argmax=edit, monitor=VETO)  → MONITOR TN

Deployment metrics:
  spurious_edit_reduction = F / (D + F)      ← how many bad edits caught
  useful_edit_loss        = C / (A + C)      ← how many good edits killed
  edit_proposal_rate_b    = (A+C) / N        ← what fraction of buggy the model wanted to edit
  edit_proposal_rate_f    = (D+F) / N        ← what fraction of fixed the model wanted to edit

We run this for three models and two thresholds (the §5.1 in-sample and
the App. G.5 held-out-50/50-mean).

Output: results/monitor_real/agent_loop_simulation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _load_pairs(cache_dir: Path, layer: int, pos: int, v_unit):
    """Return list of (task_id, {cond: {projection, score, argmax, ...}})."""
    import torch
    by_task: dict[str, dict] = {}
    for pt in sorted(cache_dir.rglob("*__code_tests.pt")):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        T = int(payload["resid_pre"].shape[2])
        abs_pos = pos if pos >= 0 else T + pos
        vec = payload["resid_pre"][layer, 0, abs_pos, :].float().numpy()
        proj = float(vec @ v_unit)
        action_logits = payload["action_logits"]
        argmax_name = max(action_logits.items(),
                          key=lambda kv: kv[1]["logit"])[0]
        by_task.setdefault(payload["task_id"], {})[cond] = {
            "projection": proj,
            "score": -proj,
            "argmax": argmax_name,
        }
    return [(t, s) for t, s in by_task.items() if "buggy" in s and "fixed" in s]


def _fit_balanced_threshold(scores, labels):
    import numpy as np
    from sklearn.metrics import roc_curve
    fpr, tpr, thresh = roc_curve(labels, scores)
    bal_acc = 0.5 * (tpr + (1 - fpr))
    return float(thresh[int(np.argmax(bal_acc))])


def _simulate(pairs, threshold: float):
    """Score the deployment outcome buckets at a given threshold."""
    A = B = C = D = E = F = 0
    edits_b: list[bool] = []
    edits_f: list[bool] = []
    veto_b: list[bool] = []
    veto_f: list[bool] = []
    for tid, s in pairs:
        b, f = s["buggy"], s["fixed"]
        # Buggy side
        b_proposes_edit = (b["argmax"] == "edit")
        b_monitor_veto = (b["score"] < threshold)
        edits_b.append(b_proposes_edit)
        veto_b.append(b_proposes_edit and b_monitor_veto)
        if b_proposes_edit and not b_monitor_veto:
            A += 1
        elif not b_proposes_edit:
            B += 1
        elif b_proposes_edit and b_monitor_veto:
            C += 1

        # Fixed side
        f_proposes_edit = (f["argmax"] == "edit")
        f_monitor_veto = (f["score"] < threshold)
        edits_f.append(f_proposes_edit)
        veto_f.append(f_proposes_edit and f_monitor_veto)
        if f_proposes_edit and not f_monitor_veto:
            D += 1
        elif not f_proposes_edit:
            E += 1
        elif f_proposes_edit and f_monitor_veto:
            F += 1

    N = len(pairs)
    spurious_reduction = (F / (D + F)) if (D + F) > 0 else float("nan")
    useful_loss = (C / (A + C)) if (A + C) > 0 else float("nan")
    edit_b_rate = (A + C) / max(N, 1)
    edit_f_rate = (D + F) / max(N, 1)
    final_spurious_rate = D / max(N, 1)
    return {
        "threshold": threshold,
        "n_pairs": N,
        "buckets": {
            "A_useful_committed":      A,
            "B_useful_missed_premon":  B,
            "C_useful_vetoed":         C,
            "D_spurious_committed":    D,
            "E_spurious_avoided_premon": E,
            "F_spurious_blocked":      F,
        },
        "deployment_metrics": {
            "spurious_edit_reduction":   spurious_reduction,
            "useful_edit_loss":          useful_loss,
            "edit_proposal_rate_buggy":  edit_b_rate,
            "edit_proposal_rate_fixed":  edit_f_rate,
            "final_spurious_edit_rate":  final_spurious_rate,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--v-noop", type=Path, required=True)
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--pos", type=int, default=-1)
    p.add_argument("--label", required=True, help="Model label, e.g. 'qwen', 'codegemma', 'deepseek'")
    p.add_argument("--out", type=Path,
                   default=Path("results/monitor_real/agent_loop_simulation.json"))
    args = p.parse_args(argv)

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    v_blob = torch.load(args.v_noop, map_location="cpu", weights_only=False)
    v_unit = (v_blob["direction"].float() / v_blob["direction"].float().norm()).numpy()
    assert v_blob["layer"] == args.layer and v_blob["position"] == args.pos

    inner = args.cache_dir / args.cache_dir.name
    cache_root = inner if inner.exists() else args.cache_dir
    pairs = _load_pairs(cache_root, args.layer, args.pos, v_unit)
    N = len(pairs)
    print(f"[{args.label}] paired tasks: {N}")

    # Compute three threshold candidates
    scores = np.concatenate([
        np.asarray([s["buggy"]["score"] for _, s in pairs]),
        np.asarray([s["fixed"]["score"] for _, s in pairs]),
    ])
    labels = np.concatenate([np.ones(N, int), np.zeros(N, int)])
    in_sample_thr = _fit_balanced_threshold(scores, labels)
    auc = float(roc_auc_score(labels, scores))

    # Held-out: do 200 random 50/50 splits, fit threshold on each cal half,
    # report mean threshold + per-split deployment metrics.
    rng = np.random.default_rng(0)
    split_thresholds: list[float] = []
    for _ in range(200):
        idx = np.arange(N)
        rng.shuffle(idx)
        nc = N // 2
        cal_idx = idx[:nc]
        cal_scores = np.concatenate([
            np.asarray([pairs[i][1]["buggy"]["score"] for i in cal_idx]),
            np.asarray([pairs[i][1]["fixed"]["score"] for i in cal_idx]),
        ])
        cal_labels = np.concatenate([np.ones(len(cal_idx), int),
                                      np.zeros(len(cal_idx), int)])
        split_thresholds.append(_fit_balanced_threshold(cal_scores, cal_labels))
    held_out_thr = float(np.mean(split_thresholds))

    print(f"[{args.label}] AUC: {auc:.4f}")
    print(f"[{args.label}] in-sample threshold: {in_sample_thr:+.4f}")
    print(f"[{args.label}] held-out threshold (mean of 200 splits): {held_out_thr:+.4f}")

    # Argmax-action distribution baselines
    arg_b = Counter(s["buggy"]["argmax"] for _, s in pairs)
    arg_f = Counter(s["fixed"]["argmax"] for _, s in pairs)
    actions = sorted(set(list(arg_b.keys()) + list(arg_f.keys())))
    print(f"\n[{args.label}] argmax distribution:")
    print("  cond  " + "  ".join(f"{a:>8}" for a in actions))
    for lab, cnt in (("buggy", arg_b), ("fixed", arg_f)):
        n = sum(cnt.values()) or 1
        print(f"  {lab:<5} " + "  ".join(f"{100*cnt.get(a,0)/n:7.1f}%" for a in actions))

    # Simulate under both thresholds
    out_by_thr: dict[str, dict] = {}
    for label, thr in (("in_sample", in_sample_thr), ("held_out_50_50", held_out_thr)):
        sim = _simulate(pairs, thr)
        out_by_thr[label] = sim
        print(f"\n[{args.label} | threshold={label} @ {thr:+.4f}]")
        print(f"  Bucket A (useful edits committed):           {sim['buckets']['A_useful_committed']}")
        print(f"  Bucket B (useful edits missed pre-monitor):  {sim['buckets']['B_useful_missed_premon']}")
        print(f"  Bucket C (useful edits VETOED — bad):        {sim['buckets']['C_useful_vetoed']}")
        print(f"  Bucket D (SPURIOUS edits committed — bad):   {sim['buckets']['D_spurious_committed']}")
        print(f"  Bucket E (spurious avoided pre-monitor):     {sim['buckets']['E_spurious_avoided_premon']}")
        print(f"  Bucket F (SPURIOUS edits BLOCKED — good):    {sim['buckets']['F_spurious_blocked']}")
        m = sim["deployment_metrics"]
        if (sim["buckets"]["D_spurious_committed"] + sim["buckets"]["F_spurious_blocked"]) > 0:
            print(f"  spurious-edit reduction: {m['spurious_edit_reduction']*100:5.1f}% "
                  f"({sim['buckets']['F_spurious_blocked']} blocked of "
                  f"{sim['buckets']['D_spurious_committed']+sim['buckets']['F_spurious_blocked']} agent-proposed spurious)")
        if (sim["buckets"]["A_useful_committed"] + sim["buckets"]["C_useful_vetoed"]) > 0:
            print(f"  useful-edit loss        : {m['useful_edit_loss']*100:5.1f}% "
                  f"({sim['buckets']['C_useful_vetoed']} killed of "
                  f"{sim['buckets']['A_useful_committed']+sim['buckets']['C_useful_vetoed']} agent-proposed useful)")
        print(f"  edit-proposal rate buggy: {m['edit_proposal_rate_buggy']*100:5.1f}%")
        print(f"  edit-proposal rate fixed: {m['edit_proposal_rate_fixed']*100:5.1f}%")

    out = {
        "config": {
            "label": args.label,
            "cache_dir": str(args.cache_dir),
            "v_noop": str(args.v_noop),
            "layer": args.layer, "pos": args.pos,
            "n_paired_tasks": N,
        },
        "auc": auc,
        "in_sample_threshold": in_sample_thr,
        "held_out_threshold_mean": held_out_thr,
        "argmax_distribution": {
            "buggy": dict(arg_b),
            "fixed": dict(arg_f),
        },
        "simulation_by_threshold": out_by_thr,
    }

    # Append-or-create the multi-model output
    if args.out.is_file():
        existing = json.loads(args.out.read_text())
        if "by_model" not in existing:
            existing = {"by_model": {}}
    else:
        existing = {"by_model": {}}
    existing["by_model"][args.label] = out
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(existing, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
