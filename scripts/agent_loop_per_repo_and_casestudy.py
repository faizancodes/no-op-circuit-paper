#!/usr/bin/env python3
"""Two free post-hoc analyses of the §G.10 single-turn agent-loop simulation:

A. Per-repo breakdown. Group the 499 paired tasks by their
   SWE-bench repo (django, sympy, ...) and compute deployment
   metrics within each repo. Adds robustness analysis to G.10.

B. Useful-edit-veto case study. For the 14 Qwen prompts where
   argmax==edit on the buggy condition AND the monitor vetoed
   (Bucket C — "useful edit vetoed"), look at:
     - the projection score (just below threshold? way below?)
     - the task / repo / transcript shape
     - what would have happened under the held-out threshold
   Find any pattern that suggests when the monitor over-vetoes.

Inputs:
  results/cache-real-qwen-n500-*  (the §5.1 cache for residuals + logits)
  results/cache-real-codegemma-n500-* and results/cache-real-deepseek-n500-*
  results/steer-*/v_noop.pt (per-model v_noop)
  results/monitor_real/agent_loop_simulation.json (thresholds)
  data/real_tasks/<task>/buggy/tests_output.txt + meta.yaml

Outputs:
  results/monitor_real/agent_loop_per_repo.json
  results/monitor_real/agent_loop_casestudy.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _repo_of(task_id: str) -> str:
    parts = task_id.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return task_id


def _load_pairs(cache_root: Path, layer: int, pos: int, v_unit):
    import torch
    by_task: dict[str, dict] = {}
    for pt in sorted(cache_root.rglob("*__code_tests.pt")):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        T = int(payload["resid_pre"].shape[2])
        abs_pos = pos if pos >= 0 else T + pos
        vec = payload["resid_pre"][layer, 0, abs_pos, :].float().numpy()
        proj = float(vec @ v_unit)
        logits = payload["action_logits"]
        argmax = max(logits.items(), key=lambda kv: kv[1]["logit"])[0]
        by_task.setdefault(payload["task_id"], {})[cond] = {
            "projection": proj,
            "score": -proj,
            "argmax": argmax,
            "logits": {k: float(v["logit"]) for k, v in logits.items()},
        }
    return [(t, s) for t, s in by_task.items() if "buggy" in s and "fixed" in s]


def _per_repo_metrics(pairs, threshold):
    by_repo: dict[str, dict] = defaultdict(lambda: {"A": 0, "B": 0, "C": 0,
                                                     "D": 0, "E": 0, "F": 0, "n": 0})
    for tid, s in pairs:
        repo = _repo_of(tid)
        by_repo[repo]["n"] += 1
        b, f = s["buggy"], s["fixed"]
        b_edit = (b["argmax"] == "edit")
        b_veto = (b["score"] < threshold)
        f_edit = (f["argmax"] == "edit")
        f_veto = (f["score"] < threshold)
        if b_edit and not b_veto:
            by_repo[repo]["A"] += 1
        elif not b_edit:
            by_repo[repo]["B"] += 1
        elif b_edit and b_veto:
            by_repo[repo]["C"] += 1
        if f_edit and not f_veto:
            by_repo[repo]["D"] += 1
        elif not f_edit:
            by_repo[repo]["E"] += 1
        elif f_edit and f_veto:
            by_repo[repo]["F"] += 1
    # Compute deployment metrics per repo
    rows = []
    for repo, b in sorted(by_repo.items()):
        spurious_proposed = b["D"] + b["F"]
        useful_proposed = b["A"] + b["C"]
        rows.append({
            "repo": repo,
            "n_tasks": b["n"],
            "useful_proposed": useful_proposed,
            "spurious_proposed": spurious_proposed,
            "spurious_blocked_F": b["F"],
            "useful_killed_C": b["C"],
            "spurious_edit_reduction":
                (b["F"] / spurious_proposed) if spurious_proposed > 0 else float("nan"),
            "useful_edit_loss":
                (b["C"] / useful_proposed) if useful_proposed > 0 else float("nan"),
            "edit_propose_rate_buggy":
                useful_proposed / max(b["n"], 1),
            "edit_propose_rate_fixed":
                spurious_proposed / max(b["n"], 1),
            "final_spurious_rate": b["D"] / max(b["n"], 1),
        })
    return rows


def _case_study(pairs, threshold, threshold_held_out, tasks_root: Path):
    """Find Bucket-C cases (useful edit vetoed) and characterise them."""
    cases = []
    for tid, s in pairs:
        b = s["buggy"]
        b_edit = (b["argmax"] == "edit")
        b_veto_in_sample = (b["score"] < threshold)
        b_veto_held_out = (b["score"] < threshold_held_out)
        if not (b_edit and b_veto_in_sample):
            continue
        # This is a useful-edit-veto case.
        repo = _repo_of(tid)
        buggy_tx_path = tasks_root / tid / "buggy" / "tests_output.txt"
        buggy_tx = buggy_tx_path.read_text(encoding="utf-8") if buggy_tx_path.exists() else ""
        n_failed_lines = buggy_tx.count("FAILED")
        n_assert_lines = buggy_tx.count("AssertionError")
        n_lines = len(buggy_tx.splitlines())
        cases.append({
            "task_id": tid,
            "repo": repo,
            "buggy_projection": b["projection"],
            "buggy_score": b["score"],
            "threshold_in_sample": threshold,
            "threshold_held_out": threshold_held_out,
            "score_minus_thresh_in_sample": b["score"] - threshold,
            "score_minus_thresh_held_out": b["score"] - threshold_held_out,
            "still_vetoed_at_held_out": b_veto_held_out,
            "buggy_argmax": b["argmax"],
            "edit_logit": b["logits"].get("edit", float("nan")),
            "noop_logit": b["logits"].get("noop", float("nan")),
            "edit_minus_noop": b["logits"].get("edit", 0) - b["logits"].get("noop", 0),
            "buggy_transcript_n_lines": n_lines,
            "buggy_transcript_n_failed": n_failed_lines,
            "buggy_transcript_n_assertions": n_assert_lines,
            "buggy_transcript_first_60chars": buggy_tx[:60].replace("\n", "↵"),
        })
    return cases


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--v-noop", type=Path, required=True)
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--pos", type=int, default=-1)
    p.add_argument("--label", required=True)
    p.add_argument("--tasks-root", type=Path, default=Path("data/real_tasks"))
    p.add_argument("--out-prefix", type=Path,
                   default=Path("results/monitor_real"))
    args = p.parse_args(argv)

    import numpy as np
    import torch
    from sklearn.metrics import roc_curve

    v_blob = torch.load(args.v_noop, map_location="cpu", weights_only=False)
    v_unit = (v_blob["direction"].float() / v_blob["direction"].float().norm()).numpy()
    assert v_blob["layer"] == args.layer and v_blob["position"] == args.pos

    inner = args.cache_dir / args.cache_dir.name
    cache_root = inner if inner.exists() else args.cache_dir
    pairs = _load_pairs(cache_root, args.layer, args.pos, v_unit)
    N = len(pairs)
    print(f"[{args.label}] paired tasks: {N}")

    # Reconstruct the two thresholds from the agent_loop_simulation.json file
    sim_path = args.out_prefix / "agent_loop_simulation.json"
    if not sim_path.exists():
        print(f"missing {sim_path}", file=sys.stderr)
        return 2
    sim_blob = json.loads(sim_path.read_text())["by_model"]
    sim_model = sim_blob[args.label]
    thr_in = sim_model["in_sample_threshold"]
    thr_held = sim_model["held_out_threshold_mean"]
    print(f"[{args.label}] in-sample thr={thr_in:+.4f}  held-out thr={thr_held:+.4f}")

    # ===== Per-repo breakdown =====
    rows_in = _per_repo_metrics(pairs, thr_in)
    rows_held = _per_repo_metrics(pairs, thr_held)
    rows = []
    by_repo_in = {r["repo"]: r for r in rows_in}
    for r_h in rows_held:
        r_i = by_repo_in[r_h["repo"]]
        rows.append({
            "repo": r_h["repo"],
            "n_tasks": r_h["n_tasks"],
            "useful_proposed": r_h["useful_proposed"],
            "spurious_proposed": r_h["spurious_proposed"],
            "in_sample": {
                "spurious_edit_reduction": r_i["spurious_edit_reduction"],
                "useful_edit_loss": r_i["useful_edit_loss"],
                "spurious_blocked": r_i["spurious_blocked_F"],
                "useful_killed": r_i["useful_killed_C"],
                "final_spurious_rate": r_i["final_spurious_rate"],
            },
            "held_out": {
                "spurious_edit_reduction": r_h["spurious_edit_reduction"],
                "useful_edit_loss": r_h["useful_edit_loss"],
                "spurious_blocked": r_h["spurious_blocked_F"],
                "useful_killed": r_h["useful_killed_C"],
                "final_spurious_rate": r_h["final_spurious_rate"],
            },
        })
    rows.sort(key=lambda r: -r["n_tasks"])

    print(f"\n=== Per-repo G.10 metrics ({args.label}, held-out threshold) ===")
    print(f"{'repo':<28} {'N':>4} {'usef':>5} {'spur':>5} {'red%':>6} {'loss%':>6} {'final%':>7}")
    print("-" * 75)
    for r in rows:
        ho = r["held_out"]
        red_str = f"{ho['spurious_edit_reduction']*100:5.1f}" if not (
            ho['spurious_edit_reduction'] != ho['spurious_edit_reduction']) else "  n/a"
        loss_str = f"{ho['useful_edit_loss']*100:5.1f}" if not (
            ho['useful_edit_loss'] != ho['useful_edit_loss']) else "  n/a"
        print(f"{r['repo']:<28} {r['n_tasks']:>4d} {r['useful_proposed']:>5d} "
              f"{r['spurious_proposed']:>5d} {red_str:>5} {loss_str:>5} "
              f"{ho['final_spurious_rate']*100:7.2f}")

    per_repo_out = {
        "config": {
            "label": args.label,
            "in_sample_threshold": thr_in,
            "held_out_threshold": thr_held,
            "n_paired_tasks": N,
        },
        "rows": rows,
    }
    per_repo_path = args.out_prefix / f"agent_loop_per_repo_{args.label}.json"
    per_repo_path.write_text(json.dumps(per_repo_out, indent=2))
    print(f"\nwrote {per_repo_path}")

    # ===== Case study (only meaningful for Qwen — limited Bucket C on others) =====
    if args.label == "qwen":
        cases = _case_study(pairs, thr_in, thr_held, args.tasks_root)
        # Sort by absolute distance from threshold — closest first
        cases.sort(key=lambda c: c["score_minus_thresh_in_sample"])
        print(f"\n=== Case study: {len(cases)} useful-edit vetoes (in-sample threshold) ===")
        print(f"{'task_id':<40} {'repo':<20} {'score':>8} {'Δthr':>7} "
              f"{'still@HO':>9} {'e-n':>7} {'nFAIL':>6} {'nLn':>4}")
        print("-" * 110)
        for c in cases:
            print(f"{c['task_id']:<40} {c['repo']:<20} "
                  f"{c['buggy_score']:+8.3f} "
                  f"{c['score_minus_thresh_in_sample']:+7.3f} "
                  f"{str(c['still_vetoed_at_held_out']):>9} "
                  f"{c['edit_minus_noop']:+7.3f} "
                  f"{c['buggy_transcript_n_failed']:>6d} "
                  f"{c['buggy_transcript_n_lines']:>4d}")

        case_summary = {
            "n_useful_vetoes": len(cases),
            "by_repo": dict(Counter(c["repo"] for c in cases)),
            "n_still_vetoed_at_held_out": sum(1 for c in cases if c["still_vetoed_at_held_out"]),
            "median_score_below_threshold": float(
                sorted(c["score_minus_thresh_in_sample"] for c in cases)[len(cases) // 2]
            ) if cases else None,
            "cases": cases,
        }
        case_path = args.out_prefix / f"agent_loop_casestudy_{args.label}.json"
        case_path.write_text(json.dumps(case_summary, indent=2))
        print(f"\nwrote {case_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
