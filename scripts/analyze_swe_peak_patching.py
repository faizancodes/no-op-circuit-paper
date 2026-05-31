#!/usr/bin/env python
"""Aggregate Qwen SWE-derived peak-cell patching scores into bootstrap summary.

Reads :  results/swe_peak_patching/<tag>_swe_peak_patch_scores[_toy].json
Writes:  results/swe_peak_patching/<tag>_swe_peak_patch_summary[_toy].json

Reports per cell:
  - F→B mean shift in ``edit - noop`` margin, 95% bootstrap CI, % positive
  - B→F mean shift in ``edit - noop`` margin, 95% bootstrap CI, % positive
  - argmax transition counts (5-action)
  - Wilcoxon signed-rank one-sided p (if scipy is available)

Sign convention (matches paper §4.1 / Experiment-A spec):
  F→B shift = m_buggy_clean - m_buggy_patched  (positive ⇒ pushed toward fixed)
  B→F shift = m_fixed_patched - m_fixed_clean  (positive ⇒ pushed toward buggy)
Both signs are reported as "toward closure of the buggy/fixed margin gap".
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


_ACTIONS_CANONICAL = ["view", "grep", "test", "edit", "noop"]


def bootstrap_ci(values, n_boot: int = 10_000, ci: float = 0.95, seed: int = 0):
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return float(arr.mean()), lo, hi


def _argmax_action(logit_dict: dict, actions: list[str]) -> str:
    return max(actions, key=lambda a: logit_dict[a])


def summarize(scores_path: Path, out_path: Path) -> dict:
    data = json.loads(scores_path.read_text())
    rows = data["rows"]
    cells: list[tuple[int, int]] = [tuple(c) for c in data["cells"]]
    # The script JSON optionally records the action vocabulary used (when a
    # custom single-token menu was passed via --action-words). Fall back to
    # the canonical 5-tuple for back-compat with older score files.
    actions: list[str] = list(data.get("action_names") or _ACTIONS_CANONICAL)
    abstain: str = data.get("abstain_word") or actions[-1]
    # The "edit" action keeps its canonical name across both menus.
    edit_name = "edit"

    m_buggy_clean: list[float] = []
    m_fixed_clean: list[float] = []
    argmax_buggy_clean: list[str] = []
    argmax_fixed_clean: list[str] = []
    for r in rows:
        cb = r["clean_buggy_logits"]
        cf = r["clean_fixed_logits"]
        m_buggy_clean.append(cb[edit_name] - cb[abstain])
        m_fixed_clean.append(cf[edit_name] - cf[abstain])
        argmax_buggy_clean.append(_argmax_action(cb, actions))
        argmax_fixed_clean.append(_argmax_action(cf, actions))

    gap = [b - f for b, f in zip(m_buggy_clean, m_fixed_clean)]
    summary_clean = {
        "n_pairs": len(rows),
        "mean_margin_buggy": statistics.mean(m_buggy_clean),
        "mean_margin_fixed": statistics.mean(m_fixed_clean),
        "mean_buggy_minus_fixed_gap": statistics.mean(gap),
        "median_buggy_minus_fixed_gap": statistics.median(gap),
        "argmax_buggy_counts": dict(Counter(argmax_buggy_clean)),
        "argmax_fixed_counts": dict(Counter(argmax_fixed_clean)),
    }

    cells_summary: dict[str, dict] = {}
    for (L, P) in cells:
        cell_id = f"L{L}_pos{P}"
        shift_f2b: list[float] = []
        shift_b2f: list[float] = []
        argmax_b_post: list[str] = []
        argmax_f_post: list[str] = []
        for r, mb, mf in zip(rows, m_buggy_clean, m_fixed_clean):
            f2b = r["patched"][cell_id]["f2b_logits"]
            b2f = r["patched"][cell_id]["b2f_logits"]
            m_b_post = f2b[edit_name] - f2b[abstain]
            m_f_post = b2f[edit_name] - b2f[abstain]
            shift_f2b.append(mb - m_b_post)
            shift_b2f.append(m_f_post - mf)
            argmax_b_post.append(_argmax_action(f2b, actions))
            argmax_f_post.append(_argmax_action(b2f, actions))

        mean_f2b, lo_f2b, hi_f2b = bootstrap_ci(shift_f2b)
        mean_b2f, lo_b2f, hi_b2f = bootstrap_ci(shift_b2f)
        pct_pos_f2b = sum(s > 0 for s in shift_f2b) / len(shift_f2b) if shift_f2b else float("nan")
        pct_pos_b2f = sum(s > 0 for s in shift_b2f) / len(shift_b2f) if shift_b2f else float("nan")

        try:
            from scipy.stats import wilcoxon  # type: ignore[import-not-found]
            p_f2b = float(getattr(wilcoxon(shift_f2b, alternative="greater"), "pvalue")) if shift_f2b else None
            p_b2f = float(getattr(wilcoxon(shift_b2f, alternative="greater"), "pvalue")) if shift_b2f else None
        except Exception:
            p_f2b = p_b2f = None

        flow_f2b = Counter(zip(argmax_buggy_clean, argmax_b_post))
        flow_b2f = Counter(zip(argmax_fixed_clean, argmax_f_post))

        cells_summary[cell_id] = {
            "f2b": {
                "mean_shift": mean_f2b,
                "ci95_lo": lo_f2b,
                "ci95_hi": hi_f2b,
                "pct_positive": pct_pos_f2b,
                "p_wilcoxon_one_sided_greater": p_f2b,
                "shifts": shift_f2b,
            },
            "b2f": {
                "mean_shift": mean_b2f,
                "ci95_lo": lo_b2f,
                "ci95_hi": hi_b2f,
                "pct_positive": pct_pos_b2f,
                "p_wilcoxon_one_sided_greater": p_b2f,
                "shifts": shift_b2f,
            },
            "argmax_flow_buggy_clean_to_f2b": {f"{a}->{b}": v for (a, b), v in flow_f2b.items()},
            "argmax_flow_fixed_clean_to_b2f": {f"{a}->{b}": v for (a, b), v in flow_b2f.items()},
            "argmax_buggy_postF2B_counts": dict(Counter(argmax_b_post)),
            "argmax_fixed_postB2F_counts": dict(Counter(argmax_f_post)),
        }

    summary = {
        "model": data["model"],
        "tasks": data["tasks"],
        "variant": data["variant"],
        "n_pairs": len(rows),
        "seed": data["seed"],
        "stratify_by_repo": data["stratify_by_repo"],
        "cells": data["cells"],
        "action_names": actions,
        "abstain_word": abstain,
        "margin_scalar": f"{edit_name} - {abstain}",
        "clean": summary_clean,
        "by_cell": cells_summary,
        "scores_path": str(scores_path),
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[analyze] wrote {out_path}")
    print("\n=== HEADLINE ===")
    print(f"N pairs                          : {len(rows)}")
    print(f"Clean buggy - fixed margin gap   : {summary_clean['mean_buggy_minus_fixed_gap']:+.3f}")
    print(f"Mean margin buggy / fixed        : {summary_clean['mean_margin_buggy']:+.3f} / "
          f"{summary_clean['mean_margin_fixed']:+.3f}")
    for cell_id, s in cells_summary.items():
        f2b = s["f2b"]
        b2f = s["b2f"]
        print(
            f"  {cell_id:<12}  F→B {f2b['mean_shift']:+.3f} [{f2b['ci95_lo']:+.3f}, "
            f"{f2b['ci95_hi']:+.3f}] ({f2b['pct_positive']*100:.0f}% pos)  "
            f"B→F {b2f['mean_shift']:+.3f} [{b2f['ci95_lo']:+.3f}, {b2f['ci95_hi']:+.3f}] "
            f"({b2f['pct_positive']*100:.0f}% pos)"
        )
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "scores",
        type=Path,
        nargs="?",
        default=Path("results/swe_peak_patching/qwen_swe_peak_patch_scores.json"),
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if not args.scores.exists():
        raise SystemExit(f"scores file not found: {args.scores}")
    out = args.out or args.scores.with_name(args.scores.name.replace("_scores", "_summary"))
    summarize(args.scores, out)


if __name__ == "__main__":
    main()
