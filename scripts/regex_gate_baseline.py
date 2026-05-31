#!/usr/bin/env python3
"""Regex-gated edit-veto baseline for §G.10 (R7 reviewer ask).

The §G.10 single-turn edit-action veto simulation reports the projection
monitor as a gate. The reviewer correctly pointed out that the regex
baseline (App. G.8 — AUC 1.000 as a classifier) should also be evaluated
as an edit-gate in the SAME six-bucket deployment framework, so the
comparison is apples-to-apples.

The regex gate rule (matching App. G.8's simplest classifier):

  agent.act():
    1. Read prompt; take first-token argmax.
    2. If argmax != 'edit': agent acts on that token (no gate needed).
    3. If argmax == 'edit': inspect the transcript.
       - If transcript contains "FAILED": ALLOW the edit.
       - Otherwise: VETO.

We compute the same six-bucket outcome model as G.10:

  A. useful edit committed         (buggy, argmax=edit, ALLOW)
  B. useful edit missed pre-gate   (buggy, argmax!=edit)
  C. useful edit VETOED            (buggy, argmax=edit, VETO) ← bad
  D. SPURIOUS edit committed       (fixed, argmax=edit, ALLOW) ← bad
  E. spurious avoided pre-gate     (fixed, argmax!=edit)
  F. SPURIOUS edit BLOCKED         (fixed, argmax=edit, VETO) ← good

  spurious-edit reduction = F / (D + F)
  useful-edit loss        = C / (A + C)

Predicted outcome in this clean synthesized-transcript setting:
  Since buggy transcripts contain "FAILED" and fixed transcripts do
  not (by construction in build_real_tasks.py), the regex gate
  should give 100% spurious-edit reduction at 0% useful-edit loss
  across all three models — strictly dominating the projection
  monitor in this exact setting.

Output: results/monitor_real/regex_gate_simulation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_argmax_pairs(cache_root: Path):
    """Return list of (task_id, {cond: argmax_action})."""
    import torch
    by_task: dict[str, dict] = {}
    for pt in sorted(cache_root.rglob("*__code_tests.pt")):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        logits = payload["action_logits"]
        argmax = max(logits.items(), key=lambda kv: kv[1]["logit"])[0]
        by_task.setdefault(payload["task_id"], {})[cond] = argmax
    return [(t, s) for t, s in by_task.items() if "buggy" in s and "fixed" in s]


def _read_transcript(tasks_root: Path, task_id: str, cond: str) -> str:
    p = tasks_root / task_id / cond / "tests_output.txt"
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def _gate_simulation(pairs, tasks_root: Path, regex_token: str = "FAILED"):
    """Apply regex gate; return six-bucket counts + deployment metrics."""
    A = B = C = D = E = F = 0
    for tid, s in pairs:
        b_argmax = s["buggy"]
        f_argmax = s["fixed"]
        b_tx = _read_transcript(tasks_root, tid, "buggy")
        f_tx = _read_transcript(tasks_root, tid, "fixed")
        # Regex gate: ALLOW edits when transcript contains the failure token.
        b_gate_allow = (regex_token in b_tx)   # buggy transcript has FAILED → allow
        f_gate_allow = (regex_token in f_tx)   # fixed transcript shouldn't → veto
        # Buggy side
        if b_argmax == "edit" and b_gate_allow:
            A += 1
        elif b_argmax != "edit":
            B += 1
        elif b_argmax == "edit" and not b_gate_allow:
            C += 1
        # Fixed side
        if f_argmax == "edit" and f_gate_allow:
            D += 1
        elif f_argmax != "edit":
            E += 1
        elif f_argmax == "edit" and not f_gate_allow:
            F += 1

    N = len(pairs)
    spurious_proposed = D + F
    useful_proposed = A + C
    return {
        "n_pairs": N,
        "regex_token": regex_token,
        "buckets": {
            "A_useful_committed":       A,
            "B_useful_missed_pregate":  B,
            "C_useful_vetoed":          C,
            "D_spurious_committed":     D,
            "E_spurious_avoided_pregate": E,
            "F_spurious_blocked":       F,
        },
        "deployment_metrics": {
            "spurious_edit_reduction":
                (F / spurious_proposed) if spurious_proposed > 0 else float("nan"),
            "useful_edit_loss":
                (C / useful_proposed) if useful_proposed > 0 else float("nan"),
            "edit_proposal_rate_buggy":  useful_proposed / max(N, 1),
            "edit_proposal_rate_fixed":  spurious_proposed / max(N, 1),
            "final_spurious_rate":       D / max(N, 1),
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks-root", type=Path, default=Path("data/real_tasks"))
    p.add_argument("--out", type=Path,
                   default=Path("results/monitor_real/regex_gate_simulation.json"))
    args = p.parse_args(argv)

    model_configs = {
        "qwen": Path("results/cache-real-qwen-n500-20260516T235301Z"),
        "codegemma": Path("results/cache-real-codegemma-n500-20260516T235731Z"),
        "deepseek": Path("results/cache-real-deepseek-n500-20260517T013041Z"),
    }
    out: dict[str, dict] = {}
    print(f"{'model':<12} {'N':>4} {'usef':>5} {'spur':>5} {'red%':>7} {'loss%':>7} {'finsp%':>8}")
    print("-" * 60)
    for label, cache_dir in model_configs.items():
        inner = cache_dir / cache_dir.name
        cache_root = inner if inner.exists() else cache_dir
        pairs = _load_argmax_pairs(cache_root)
        sim = _gate_simulation(pairs, args.tasks_root)
        m = sim["deployment_metrics"]
        print(f"{label:<12} {sim['n_pairs']:>4d} "
              f"{sim['buckets']['A_useful_committed'] + sim['buckets']['C_useful_vetoed']:>5d} "
              f"{sim['buckets']['D_spurious_committed'] + sim['buckets']['F_spurious_blocked']:>5d} "
              f"{m['spurious_edit_reduction']*100:>6.1f}% "
              f"{m['useful_edit_loss']*100:>6.1f}% "
              f"{m['final_spurious_rate']*100:>7.2f}%")
        out[label] = sim

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"by_model": out}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
