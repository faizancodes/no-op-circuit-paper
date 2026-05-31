#!/usr/bin/env python
"""Experiment A: reconcile §5.6 current-renderer Qwen five-action numbers vs §4.1.

Decision rule (from the task): if the archived §4.1 toy `code_tests` PATCH
artifact (the one carrying clean+patched margins behind +0.648/+0.659) is
unavailable, reconciliation cannot be completed and §5.6 is a renderer-specific
sanity check. This script records the availability finding, the current-renderer
five-action stats, and the quoted §4.1 reference values, and writes both a JSON
and a markdown report under results/consistency_audit/.

Availability finding (checked 2026-05-28):
  - LOCAL: only results/patch-qwen25_coder_15b_instruc-20260516T051331Z/ exists,
    and it is the `code` NEGATIVE-CONTROL variant (no code_tests patch grid).
  - HF dataset faizancodes/no-op-circuit-caches: contains only SWE-derived
    residual caches (cache-real-qwen-n500-*, cache-real-qwen-swap-n500-*) and
    `sae/`; NO toy code_tests PATCH artifact (0 qwen patch files).
  => the §4.1 toy patch artifact cannot be loaded for a per-task prompt-hash /
     margin reconciliation without a GPU rerun, and a rerun would reproduce the
     CURRENT renderer, not the original. Reconciliation: INCOMPLETE.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NEW_FILE = REPO / "results/five_action_decomp/qwen_patching_five_action_scores.json"
OUT_JSON = REPO / "results/consistency_audit/qwen_five_action_reconciliation.json"
OUT_MD = REPO / "results/consistency_audit/qwen_five_action_reconciliation.md"

PAPER_REF = {
    "source": "paper §4.1/§4.3 (original renderer; toy code_tests PATCH artifact archived/unavailable)",
    "clean_BminusF_gap_all49": 0.659,
    "f2b_all49_one_way": 0.648,
    "f2b_43task_bidirectional": 0.69,
    "b2f_43task_bidirectional": 0.64,
}

AVAILABILITY = {
    "local_qwen_patch_dirs": ["results/patch-qwen25_coder_15b_instruc-20260516T051331Z (code variant only)"],
    "local_toy_code_tests_patch_artifact": False,
    "hf_dataset": "faizancodes/no-op-circuit-caches",
    "hf_qwen_dirs": ["cache-real-qwen-n500-* (SWE-derived residual caches)",
                     "cache-real-qwen-swap-n500-* (contradictory-swap residual caches)",
                     "sae/"],
    "hf_toy_code_tests_patch_artifact": False,
    "prompt_hash_comparison_possible": False,
    "reason": ("the original §4.1 toy code_tests PATCH artifact (clean+patched margins) "
               "is neither local (only the `code` negative-control is) nor on HF (only "
               "SWE-derived residual caches are archived); a GPU rerun would reproduce "
               "the current renderer, not the original, so it cannot reconcile."),
}


def _stat(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {
        "n": len(xs), "mean": float(statistics.mean(xs)),
        "median": float(statistics.median(xs)), "min": float(min(xs)),
        "max": float(max(xs)),
        "pct_positive": float(sum(1 for x in xs if x > 0) / len(xs)),
    }


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(NEW_FILE.read_text())
    cell = "L24_pos-1"
    gap, f2b, b2f = [], [], []
    for r in data["rows"]:
        cb = r["clean_buggy_logits"]["edit"] - r["clean_buggy_logits"]["noop"]
        cf = r["clean_fixed_logits"]["edit"] - r["clean_fixed_logits"]["noop"]
        gap.append(cb - cf)
        p = r["patched"][cell]
        f2b.append(cb - (p["f2b_logits"]["edit"] - p["f2b_logits"]["noop"]))
        b2f.append((p["b2f_logits"]["edit"] - p["b2f_logits"]["noop"]) - cf)
    new_stats = {
        "clean_BminusF_gap": _stat(gap),
        "f2b_shift": _stat(f2b),
        "b2f_shift": _stat(b2f),
        "n_pairs": len(data["rows"]),
        "cell": "L24/pos-1",
        "artifact": str(NEW_FILE.relative_to(REPO)),
        "renderer": "current five-action decomposition renderer (modal_app/swe_peak_patching.py)",
    }
    decision = (
        "RECONCILIATION INCOMPLETE — current-renderer sanity check only. "
        "The §4.1 toy code_tests PATCH artifact is unavailable (see availability), "
        "so per-task prompt-hash / margin reconciliation could not be performed. "
        f"The current-renderer F->B shift (+{new_stats['f2b_shift']['mean']:.3f}) is close "
        f"to the §4.1 all-49 estimate (+{PAPER_REF['f2b_all49_one_way']}), but the "
        f"current-renderer clean B-F gap (+{new_stats['clean_BminusF_gap']['mean']:.3f}) "
        f"is lower than §4.1's +{PAPER_REF['clean_BminusF_gap_all49']}. Most likely cause: "
        "fixed-condition prompt-rendering drift after the May-2026 paper run (same "
        "direction observed in the CodeGemma audit). Per the decision rule, §5.6 is "
        "treated as a renderer-specific sanity check and its clean-gap column is removed; "
        "the §4.1/§4.3 causal-localization numbers are NOT revised."
    )
    out = {
        "experiment": "A: Qwen five-action current-renderer vs §4.1 reconciliation",
        "availability": AVAILABILITY,
        "paper_reference_values": PAPER_REF,
        "current_renderer_stats": new_stats,
        "reconciliation_status": "incomplete_artifact_unavailable",
        "decision": decision,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    md = f"""# Qwen five-action reconciliation (Experiment A)

**Status: RECONCILIATION INCOMPLETE — §5.6 is a current-renderer sanity check.**

## Availability
- Local Qwen patch artifact: `results/patch-qwen25_coder_15b_instruc-20260516T051331Z/`
  is the **`code` negative-control variant only** (no toy `code_tests` patch grid).
- HF dataset `faizancodes/no-op-circuit-caches`: holds only SWE-derived residual
  caches (`cache-real-qwen-n500-*`, `cache-real-qwen-swap-n500-*`) and `sae/`;
  **no toy `code_tests` patch artifact**.
- Therefore the §4.1 toy patch artifact (clean+patched margins behind +0.648/+0.659)
  could not be loaded; a prompt-hash / per-task margin reconciliation was not possible,
  and a GPU rerun would reproduce the current renderer rather than the original.

## Numbers (no merge performed)
| quantity | §4.1/§4.3 (archived renderer) | §5.6 current renderer (this run) |
|---|---|---|
| clean B−F gap (all 49) | +{PAPER_REF['clean_BminusF_gap_all49']} | +{new_stats['clean_BminusF_gap']['mean']:.3f} |
| F→B shift at L24/pos−1 | +{PAPER_REF['f2b_all49_one_way']} (all-49 one-way) | +{new_stats['f2b_shift']['mean']:.3f} |
| B→F shift at L24/pos−1 | +{PAPER_REF['b2f_43task_bidirectional']} (43-task bidir) | +{new_stats['b2f_shift']['mean']:.3f} |

The F→B shift is close (+{new_stats['f2b_shift']['mean']:.3f} vs +{PAPER_REF['f2b_all49_one_way']});
the clean gap differs (+{new_stats['clean_BminusF_gap']['mean']:.3f} vs +{PAPER_REF['clean_BminusF_gap_all49']}),
consistent with fixed-condition renderer drift after May-2026 (same direction as the CodeGemma audit).

## Decision
{decision}
"""
    OUT_MD.write_text(md)
    print(f"[reconcile] wrote {OUT_JSON}")
    print(f"[reconcile] wrote {OUT_MD}")
    print(f"\nstatus: incomplete (toy code_tests patch artifact unavailable)")
    print(f"current-renderer clean gap +{new_stats['clean_BminusF_gap']['mean']:.3f} vs §4.1 +{PAPER_REF['clean_BminusF_gap_all49']}")
    print(f"current-renderer F->B    +{new_stats['f2b_shift']['mean']:.3f} vs §4.1 +{PAPER_REF['f2b_all49_one_way']}")


if __name__ == "__main__":
    main()
