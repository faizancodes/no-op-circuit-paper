#!/usr/bin/env python
"""Audit the Qwen §4.1/§4.3 vs §5.6 five-action patching consistency.

Qwen is the core causal claim, so any numerical mismatch between the §4.1/§4.3
patching estimates and the §5.6 five-action discrete table must be explained,
not smoothed over.

Compares:
  OLD: results/patch-qwen25_coder_15b_instruc-20260516T051331Z/  (§4.1/§4.3 source;
       bidirectional grid, max_suffix=2, layer_step=2, BOTH `code` and `code_tests`
       variants present — we filter to `code_tests`)
  NEW: results/five_action_decomp/qwen_patching_five_action_scores.json  (§5.6 table;
       produced by modal_app/swe_peak_patching.py --tasks toy, cells incl. L24/pos -1)

For each task present in both (matched by task_id), records the clean
buggy/fixed `edit - noop` margins, the B-F gap, and the L24/pos -1 F->B / B->F
shifts under each artifact, using the paper's hypothesis-confirming sign:
  F->B shift = clean_buggy_margin - patched_buggy_margin   (positive = toward fixed)
  B->F shift = patched_fixed_margin - clean_fixed_margin   (positive = toward buggy)

Writes JSON to results/consistency_audit/qwen_five_action_consistency_audit.json.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
OLD_ROOT = REPO / "results/patch-qwen25_coder_15b_instruc-20260516T051331Z"
OLD_DIR = OLD_ROOT / "patch-qwen25_coder_15b_instruc-20260516T051331Z"
OLD_MANIFEST = OLD_ROOT / "manifest.json"
NEW_FILE = REPO / "results/five_action_decomp/qwen_patching_five_action_scores.json"
OUT_FILE = REPO / "results/consistency_audit/qwen_five_action_consistency_audit.json"

LAYER = 24
POS = -1


def _stat(xs):
    xs = [x for x in xs if x is not None]
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


# Established §4.1/§4.3 reference values from the ORIGINAL paper renderer
# (the code_tests Qwen patch artifact is archived to the HF dataset
# faizancodes/no-op-circuit-caches and is not present locally; these are
# quoted from the manuscript for context, NOT recomputed here).
PAPER_REF = {
    "source": "paper §4.1/§4.3 (original renderer; code_tests artifact HF-archived, not local)",
    "clean_BminusF_gap_all49": 0.659,
    "f2b_all49_one_way": 0.648,
    "f2b_43task_bidirectional": 0.69,
    "b2f_43task_bidirectional": 0.64,
}


def load_old() -> dict:
    """Load the §4.1/§4.3 Qwen patching artifact, code_tests variant, L24/pos -1.

    The local May-2026 Qwen patch run contains only the `code` negative-control
    variant; the `code_tests` artifact that produced the §4.1 +0.648/+0.659
    headline is archived to HF and not present locally. We detect that case and
    return a sentinel so the audit reports "reconciliation incomplete" rather
    than silently comparing against the wrong variant.
    """
    manifest = json.loads(OLD_MANIFEST.read_text())
    task_ids = sorted({s["task_id"] for s in manifest["summaries"]})
    variants_present = sorted({s["variant"] for s in manifest["summaries"]})
    code_tests_pts = sorted(OLD_DIR.glob("*__code_tests__patch.pt"))
    if not code_tests_pts:
        return {
            "meta": {
                "run_id": manifest.get("run_id"),
                "model_name": manifest.get("model_name"),
                "variants_present_locally": variants_present,
                "code_tests_available_locally": False,
                "note": ("local artifact is the `code` negative-control variant only; "
                         "code_tests source archived to HF faizancodes/no-op-circuit-caches"),
            },
            "rows": {},
            "code_tests_available": False,
        }
    sample_path = code_tests_pts[0]
    sample = torch.load(sample_path, weights_only=False)
    sample_meta = {
        "model_name": sample.get("model_name"),
        "hook_point": sample.get("hook_point"),
        "variant": sample.get("variant"),
        "layer_indices": sample.get("layer_indices"),
        "position_offsets": sample.get("position_offsets"),
        "action_ids": sample.get("action_ids"),
        "bidirectional": sample.get("bidirectional"),
        "max_suffix": manifest.get("max_suffix"),
        "layer_step": manifest.get("layer_step"),
        "run_id": manifest.get("run_id"),
    }
    layer_idx_to_row = {l: i for i, l in enumerate(sample_meta["layer_indices"])}
    pos_to_col = {p: i for i, p in enumerate(sample_meta["position_offsets"])}
    if LAYER not in layer_idx_to_row or POS not in pos_to_col:
        raise SystemExit(
            f"L{LAYER}/pos {POS} not in old grid: layers={sample_meta['layer_indices']}, "
            f"pos={sample_meta['position_offsets']}"
        )
    Lrow = layer_idx_to_row[LAYER]
    Pcol = pos_to_col[POS]

    rows = {}
    for tid in task_ids:
        pth = OLD_DIR / f"{tid}__code_tests__patch.pt"
        if not pth.exists():
            continue
        d = torch.load(pth, weights_only=False)
        cb = float(d["clean_buggy_margin"])
        cf = float(d["clean_fixed_margin"])
        m_b_post = float(d["patched_margins_f2b"][Lrow][Pcol])
        f2b_shift = cb - m_b_post
        b2f_shift = None
        if d.get("bidirectional") and d.get("patched_margins_b2f"):
            m_f_post = float(d["patched_margins_b2f"][Lrow][Pcol])
            b2f_shift = m_f_post - cf
        rows[tid] = {
            "clean_buggy_margin": cb,
            "clean_fixed_margin": cf,
            "gap_buggy_minus_fixed": cb - cf,
            "f2b_shift": f2b_shift,
            "b2f_shift": b2f_shift,
        }
    sample_meta["code_tests_available_locally"] = True
    return {"meta": sample_meta, "rows": rows, "code_tests_available": True}


def load_new() -> dict:
    data = json.loads(NEW_FILE.read_text())
    cell = f"L{LAYER}_pos{POS}"
    rows = {}
    for r in data["rows"]:
        tid = r["task_id"]
        cb_l = r["clean_buggy_logits"]
        cf_l = r["clean_fixed_logits"]
        cb = cb_l["edit"] - cb_l["noop"]
        cf = cf_l["edit"] - cf_l["noop"]
        patched = r["patched"].get(cell) or {}
        f2b_l = patched.get("f2b_logits") or {}
        b2f_l = patched.get("b2f_logits") or {}
        m_b_post = (f2b_l["edit"] - f2b_l["noop"]) if f2b_l else None
        m_f_post = (b2f_l["edit"] - b2f_l["noop"]) if b2f_l else None
        rows[tid] = {
            "clean_buggy_margin": cb,
            "clean_fixed_margin": cf,
            "gap_buggy_minus_fixed": cb - cf,
            "f2b_shift": (cb - m_b_post) if m_b_post is not None else None,
            "b2f_shift": (m_f_post - cf) if m_f_post is not None else None,
            "action_ids": r["action_ids"],
        }
    meta = {
        "model_name": data["model"],
        "tasks": data["tasks"],
        "variant": data["variant"],
        "cells": data["cells"],
        "action_names": data.get("action_names"),
        "abstain_word": data.get("abstain_word"),
        "seed": data.get("seed"),
    }
    return {"meta": meta, "rows": rows}


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[audit] OLD: {OLD_DIR}")
    print(f"[audit] NEW: {NEW_FILE}")
    old = load_old()
    new = load_new()

    # New-run aggregate stats (always computable).
    new_rows = list(new["rows"].values())
    new_summary = {
        "clean_buggy_stat": _stat([r["clean_buggy_margin"] for r in new_rows]),
        "clean_fixed_stat": _stat([r["clean_fixed_margin"] for r in new_rows]),
        "gap_stat": _stat([r["gap_buggy_minus_fixed"] for r in new_rows]),
        "f2b_shift_stat": _stat([r["f2b_shift"] for r in new_rows]),
        "b2f_shift_stat": _stat([r["b2f_shift"] for r in new_rows]),
    }

    if not old.get("code_tests_available"):
        # Original code_tests artifact unavailable locally → reconciliation
        # cannot be completed per-task. Report honestly; do NOT claim agreement.
        conclusion = (
            "RECONCILIATION INCOMPLETE: the original §4.1/§4.3 Qwen `code_tests` "
            "patch artifact is archived to HF (faizancodes/no-op-circuit-caches) and "
            "is not present locally (the local May-2026 Qwen patch run is the `code` "
            "negative-control variant only). The §5.6 five-action values are therefore "
            "from the CURRENT renderer and are NOT directly reconciled against the "
            "§4.1 estimates. The F->B shift (new +%.3f) is close to the §4.1 all-49 "
            "estimate (+%.3f); the clean B-F gap (new +%.3f) is lower than §4.1's "
            "+%.3f, consistent with the post-May-2026 renderer change also seen in the "
            "CodeGemma audit. Treat §5.6 Qwen numbers as a current-renderer discrete "
            "check, not a revision of §4.1."
        ) % (
            new_summary["f2b_shift_stat"]["mean"], PAPER_REF["f2b_all49_one_way"],
            new_summary["gap_stat"]["mean"], PAPER_REF["clean_BminusF_gap_all49"],
        )
        out = {
            "cell": f"L{LAYER}/pos{POS}",
            "old_artifact_local": str(OLD_DIR.relative_to(REPO)),
            "old_code_tests_available_locally": False,
            "old_meta": old["meta"],
            "paper_reference_values": PAPER_REF,
            "new_artifact": str(NEW_FILE.relative_to(REPO)),
            "new_meta": new["meta"],
            "new_run_summary": new_summary,
            "reconciliation_status": "incomplete_old_code_tests_archived_to_hf",
            "conclusion": conclusion,
        }
        OUT_FILE.write_text(json.dumps(out, indent=2, default=str))
        print(f"[audit] wrote {OUT_FILE}\n")
        print("=== QWEN RECONCILIATION (INCOMPLETE) ===")
        print(f"local old artifact: {old['meta'].get('run_id')} "
              f"(variants present locally: {old['meta'].get('variants_present_locally')})")
        print("original code_tests artifact: ARCHIVED TO HF, not local")
        print(f"paper §4.1 reference: clean gap +{PAPER_REF['clean_BminusF_gap_all49']}, "
              f"F->B all-49 +{PAPER_REF['f2b_all49_one_way']}, "
              f"43-task bidir +{PAPER_REF['f2b_43task_bidirectional']}/{PAPER_REF['b2f_43task_bidirectional']}")
        print(f"new (current renderer): clean gap {new_summary['gap_stat']}")
        print(f"new F->B shift: {new_summary['f2b_shift_stat']}")
        print(f"new B->F shift: {new_summary['b2f_shift_stat']}")
        print(f"\nCONCLUSION:\n{conclusion}")
        return

    common_ids = sorted(set(old["rows"]) & set(new["rows"]))
    print(f"[audit] {len(common_ids)} common task ids")

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
            "old_f2b_shift": o["f2b_shift"],
            "new_f2b_shift": n["f2b_shift"],
            "old_b2f_shift": o["b2f_shift"],
            "new_b2f_shift": n["b2f_shift"],
        })

    def stat(key):
        return _stat([d[key] for d in diffs])

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
        "max_abs_buggy_diff": max((abs(d["buggy_diff_new_minus_old"]) for d in diffs), default=None),
        "max_abs_fixed_diff": max((abs(d["fixed_diff_new_minus_old"]) for d in diffs), default=None),
        "n_buggy_changed_gt_0p5": sum(1 for d in diffs if abs(d["buggy_diff_new_minus_old"]) > 0.5),
        "n_fixed_changed_gt_0p5": sum(1 for d in diffs if abs(d["fixed_diff_new_minus_old"]) > 0.5),
    }

    new_action_ids = next(iter(new["rows"].values())).get("action_ids")
    old_action_ids = old["meta"]["action_ids"]
    ids_match = old_action_ids == new_action_ids

    out = {
        "old_artifact": str(OLD_DIR.relative_to(REPO)),
        "new_artifact": str(NEW_FILE.relative_to(REPO)),
        "cell": f"L{LAYER}/pos{POS}",
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
    print("=== HEADLINE COMPARISON (Qwen L24/pos -1, code_tests, N=49 toy) ===")
    print(f"common_n: {len(common_ids)}  | action ids match: {ids_match}")
    print(f"old action ids: {old_action_ids}")
    print(f"new action ids: {new_action_ids}")
    print(f"old grid: layers={old['meta']['layer_indices']} pos={old['meta']['position_offsets']} "
          f"max_suffix={old['meta']['max_suffix']} layer_step={old['meta']['layer_step']}")
    print()
    print(f"old clean_buggy: {aggregate['old_clean_buggy_stat']}")
    print(f"new clean_buggy: {aggregate['new_clean_buggy_stat']}")
    print(f"buggy diff (new-old): {aggregate['buggy_diff_stat']}")
    print()
    print(f"old clean_fixed: {aggregate['old_clean_fixed_stat']}")
    print(f"new clean_fixed: {aggregate['new_clean_fixed_stat']}")
    print(f"fixed diff (new-old): {aggregate['fixed_diff_stat']}")
    print()
    print(f"old gap (B-F): {aggregate['old_gap_stat']}")
    print(f"new gap (B-F): {aggregate['new_gap_stat']}")
    print()
    print(f"old F->B shift: {aggregate['old_f2b_shift_stat']}")
    print(f"new F->B shift: {aggregate['new_f2b_shift_stat']}")
    print(f"old B->F shift: {aggregate['old_b2f_shift_stat']}")
    print(f"new B->F shift: {aggregate['new_b2f_shift_stat']}")


if __name__ == "__main__":
    main()
