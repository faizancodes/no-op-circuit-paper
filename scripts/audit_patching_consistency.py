#!/usr/bin/env python
"""Audit the CodeGemma §4.3 / §5.6 patching consistency.

Compares:
  OLD: results/patch-codegemma_7b_it-20260516T031403Z/ (§4.3 numbers)
  NEW: results/five_action_decomp/codegemma_swe_peak_patch_scores_toy.json (§5.6)

For each task present in both, records:
  - clean buggy / fixed margin (edit - noop) under each artifact
  - difference between old and new per-task
  - L26/pos -1 F→B and B→F shifts under each artifact (when the old grid
    contains that cell)
  - aggregate mean/median/% positive comparisons

Writes JSON to results/consistency_audit/patching_consistency_audit.json.
"""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
OLD_DIR = REPO / "results/patch-codegemma_7b_it-20260516T031403Z/patch-codegemma_7b_it-20260516T031403Z"
OLD_MANIFEST = REPO / "results/patch-codegemma_7b_it-20260516T031403Z/manifest.json"
NEW_FILE = REPO / "results/five_action_decomp/codegemma_swe_peak_patch_scores_toy.json"
OUT_FILE = REPO / "results/consistency_audit/patching_consistency_audit.json"


def _stat(xs):
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": float(statistics.mean(xs)),
        "median": float(statistics.median(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
        "pct_positive": float(sum(1 for x in xs if x > 0) / len(xs)),
    }


def load_old() -> dict:
    """Load the §4.3 CodeGemma patching artifact."""
    manifest = json.loads(OLD_MANIFEST.read_text())
    summaries = {s["task_id"]: s for s in manifest["summaries"]}
    # Find a sample .pt to inspect the grid schema
    sample_path = sorted(OLD_DIR.glob("*__code_tests__patch.pt"))[0]
    sample = torch.load(sample_path, weights_only=False)
    sample_meta = {
        "model_name": sample.get("model_name"),
        "hook_point": sample.get("hook_point"),
        "variant": sample.get("variant"),
        "layer_indices": sample.get("layer_indices"),
        "position_offsets": sample.get("position_offsets"),
        "action_ids": sample.get("action_ids"),
        "bidirectional": sample.get("bidirectional"),
    }
    layer_idx_to_row = {l: i for i, l in enumerate(sample_meta["layer_indices"])}
    pos_to_col = {p: i for i, p in enumerate(sample_meta["position_offsets"])}
    if 26 not in layer_idx_to_row or -1 not in pos_to_col:
        raise SystemExit(f"L26/pos -1 not in old grid: layers={sample_meta['layer_indices']}, pos={sample_meta['position_offsets']}")
    Lrow = layer_idx_to_row[26]
    Pcol = pos_to_col[-1]

    rows = {}
    for tid in summaries:
        pth = OLD_DIR / f"{tid}__code_tests__patch.pt"
        if not pth.exists():
            continue
        d = torch.load(pth, weights_only=False)
        cb = float(d["clean_buggy_margin"])
        cf = float(d["clean_fixed_margin"])
        # patched_margins_f2b is grid[L_idx][P_idx] = m_buggy_under_F2B_patch
        # F→B shift_at_cell = clean_buggy - patched (positive = pushed toward fixed)
        m_b_post = float(d["patched_margins_f2b"][Lrow][Pcol])
        f2b_shift = cb - m_b_post
        # B→F if bidirectional
        b2f_shift = None
        if d.get("bidirectional") and d.get("patched_margins_b2f"):
            m_f_post = float(d["patched_margins_b2f"][Lrow][Pcol])
            b2f_shift = m_f_post - cf
        rows[tid] = {
            "clean_buggy_margin": cb,
            "clean_fixed_margin": cf,
            "gap_buggy_minus_fixed": cb - cf,
            "L26_pos-1_f2b_shift": f2b_shift,
            "L26_pos-1_b2f_shift": b2f_shift,
        }
    return {"meta": sample_meta, "rows": rows}


def load_new() -> dict:
    data = json.loads(NEW_FILE.read_text())
    rows = {}
    for r in data["rows"]:
        tid = r["task_id"]
        cb_l = r["clean_buggy_logits"]
        cf_l = r["clean_fixed_logits"]
        cb = cb_l["edit"] - cb_l["noop"]
        cf = cf_l["edit"] - cf_l["noop"]
        patched = r["patched"].get("L26_pos-1") or {}
        f2b_l = patched.get("f2b_logits") or {}
        b2f_l = patched.get("b2f_logits") or {}
        m_b_post = (f2b_l["edit"] - f2b_l["noop"]) if f2b_l else None
        m_f_post = (b2f_l["edit"] - b2f_l["noop"]) if b2f_l else None
        rows[tid] = {
            "clean_buggy_margin": cb,
            "clean_fixed_margin": cf,
            "gap_buggy_minus_fixed": cb - cf,
            "L26_pos-1_f2b_shift": (cb - m_b_post) if m_b_post is not None else None,
            "L26_pos-1_b2f_shift": (m_f_post - cf) if m_f_post is not None else None,
            "action_ids": r["action_ids"],
        }
    meta = {
        "model_name": data["model"],
        "tasks": data["tasks"],
        "variant": data["variant"],
        "action_names": data.get("action_names"),
        "abstain_word": data.get("abstain_word"),
        "cells": data["cells"],
    }
    return {"meta": meta, "rows": rows}


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[audit] OLD: {OLD_DIR}")
    print(f"[audit] NEW: {NEW_FILE}")
    old = load_old()
    new = load_new()

    common_ids = sorted(set(old["rows"]) & set(new["rows"]))
    print(f"[audit] {len(common_ids)} common task ids")

    # Per-task diff
    diffs = []
    for tid in common_ids:
        o = old["rows"][tid]
        n = new["rows"][tid]
        diffs.append({
            "task_id": tid,
            "old_clean_buggy": o["clean_buggy_margin"],
            "new_clean_buggy": n["clean_buggy_margin"],
            "buggy_diff_new_minus_old": n["clean_buggy_margin"] - o["clean_buggy_margin"],
            "old_clean_fixed": o["clean_fixed_margin"],
            "new_clean_fixed": n["clean_fixed_margin"],
            "fixed_diff_new_minus_old": n["clean_fixed_margin"] - o["clean_fixed_margin"],
            "old_gap": o["gap_buggy_minus_fixed"],
            "new_gap": n["gap_buggy_minus_fixed"],
            "old_f2b_shift": o["L26_pos-1_f2b_shift"],
            "new_f2b_shift": n["L26_pos-1_f2b_shift"],
            "old_b2f_shift": o["L26_pos-1_b2f_shift"],
            "new_b2f_shift": n["L26_pos-1_b2f_shift"],
        })

    def stat(key):
        return _stat([d[key] for d in diffs if d[key] is not None])

    aggregate = {
        "common_n": len(common_ids),
        "old_clean_buggy_stat": stat("old_clean_buggy"),
        "new_clean_buggy_stat": stat("new_clean_buggy"),
        "buggy_diff_stat": stat("buggy_diff_new_minus_old"),
        "old_clean_fixed_stat": stat("old_clean_fixed"),
        "new_clean_fixed_stat": stat("new_clean_fixed"),
        "fixed_diff_stat": stat("fixed_diff_new_minus_old"),
        "old_gap_stat": stat("old_gap"),
        "new_gap_stat": stat("new_gap"),
        "old_f2b_shift_stat": stat("old_f2b_shift"),
        "new_f2b_shift_stat": stat("new_f2b_shift"),
        "old_b2f_shift_stat": stat("old_b2f_shift"),
        "new_b2f_shift_stat": stat("new_b2f_shift"),
        "n_buggy_changed": sum(1 for d in diffs if abs(d["buggy_diff_new_minus_old"]) > 0.5),
        "n_fixed_changed": sum(1 for d in diffs if abs(d["fixed_diff_new_minus_old"]) > 0.5),
    }

    # Per-task action id check (verify single-token edit+noop in BOTH)
    new_action_ids = next(iter(new["rows"].values())).get("action_ids")
    old_action_ids = old["meta"]["action_ids"]
    ids_match = old_action_ids == new_action_ids

    out = {
        "old_artifact": str(OLD_DIR.relative_to(REPO)),
        "new_artifact": str(NEW_FILE.relative_to(REPO)),
        "old_meta": old["meta"],
        "new_meta": new["meta"],
        "old_action_ids_match_new": ids_match,
        "old_action_ids": old_action_ids,
        "new_action_ids_sample": new_action_ids,
        "common_n": len(common_ids),
        "old_only_ids": sorted(set(old["rows"]) - set(new["rows"])),
        "new_only_ids": sorted(set(new["rows"]) - set(old["rows"])),
        "aggregate": aggregate,
        "per_task_diffs": diffs,
    }
    OUT_FILE.write_text(json.dumps(out, indent=2, default=str))
    print(f"[audit] wrote {OUT_FILE}")
    print()
    print("=== HEADLINE COMPARISON ===")
    print(f"common_n: {len(common_ids)}")
    print(f"old action ids: {old_action_ids}")
    print(f"new action ids (sample): {new_action_ids}")
    print(f"action ids match: {ids_match}")
    print()
    print(f"old clean_buggy mean/med/min/max: {aggregate['old_clean_buggy_stat']}")
    print(f"new clean_buggy mean/med/min/max: {aggregate['new_clean_buggy_stat']}")
    print(f"buggy diff (new - old):           {aggregate['buggy_diff_stat']}")
    print()
    print(f"old clean_fixed mean/med/min/max: {aggregate['old_clean_fixed_stat']}")
    print(f"new clean_fixed mean/med/min/max: {aggregate['new_clean_fixed_stat']}")
    print(f"fixed diff (new - old):           {aggregate['fixed_diff_stat']}")
    print()
    print(f"old gap (B-F)   mean/med:         {aggregate['old_gap_stat']}")
    print(f"new gap (B-F)   mean/med:         {aggregate['new_gap_stat']}")
    print()
    print(f"old F→B shift at L26/pos-1 stat:  {aggregate['old_f2b_shift_stat']}")
    print(f"new F→B shift at L26/pos-1 stat:  {aggregate['new_f2b_shift_stat']}")
    print(f"old B→F shift at L26/pos-1 stat:  {aggregate['old_b2f_shift_stat']}")
    print(f"new B→F shift at L26/pos-1 stat:  {aggregate['new_b2f_shift_stat']}")


if __name__ == "__main__":
    main()
