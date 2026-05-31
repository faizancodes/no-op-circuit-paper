#!/usr/bin/env python3
"""Failure-mode analysis: project stale-variant activations onto v_noop.

The hypothesis: under `stale_misleading` and `stale_flaky` evidence, the
model's L24 / pos −1 residual still encodes the no-op signal (high projection
onto v_noop) — but its argmax action may still be `edit`. That gap between
internal knowledge and overt action is the failure mode the paper claims to
identify.

Outputs:
  results/<stale-cache-run>/failure_table.json
  also prints summary tables + the knowing-vs-doing correlation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stale_cache_dir", type=Path)
    p.add_argument(
        "--v-noop",
        type=Path,
        default=Path("/Users/faizanahmed/no-op-circuit/results/steer-20260516T021522Z/v_noop.pt"),
    )
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--clean-cache-dir", type=Path,
                   default=Path("/Users/faizanahmed/no-op-circuit/results/cache-20260515T221105Z"),
                   help="Clean cache (Qwen, code_tests) to compute baseline projection.")
    args = p.parse_args(argv)

    import numpy as np
    import torch

    # 1. Load v_noop
    v_blob = torch.load(args.v_noop, map_location="cpu", weights_only=False)
    v = v_blob["direction"].float()
    v_unit = v / v.norm()
    print(f"v_noop: layer={v_blob['layer']}, pos={v_blob['position']}, ||v||={v_blob['norm']:.3f}, "
          f"hook_point={v_blob.get('hook_point','resid_pre')}")
    assert v_blob["layer"] == args.layer and v_blob["position"] == args.position, (
        "Mismatch between v_noop reference and --layer/--position args."
    )

    def proj(resid_pre):
        # resid_pre shape (L, B=1, K, D); position is signed (negative offset)
        K = resid_pre.shape[2]
        pos_abs = args.position if args.position >= 0 else K + args.position
        x = resid_pre[args.layer, 0, pos_abs, :].float()
        return float((x @ v_unit).item())

    # 2. Baselines: clean buggy and clean fixed `code_tests` projections.
    baseline_buggy: list[float] = []
    baseline_fixed: list[float] = []
    for pt in args.clean_cache_dir.rglob("*__code_tests.pt"):
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        p_val = proj(payload["resid_pre"])
        (baseline_buggy if payload["condition"] == "buggy" else baseline_fixed).append(p_val)
    print(
        f"clean baselines (code_tests):\n"
        f"  buggy proj: mean={np.mean(baseline_buggy):+.3f}  median={np.median(baseline_buggy):+.3f}  N={len(baseline_buggy)}\n"
        f"  fixed proj: mean={np.mean(baseline_fixed):+.3f}  median={np.median(baseline_fixed):+.3f}  N={len(baseline_fixed)}"
    )

    # 3. For each stale-variant prompt: compute proj + argmax action.
    by_variant: dict[str, list[dict]] = {}
    for pt in sorted(args.stale_cache_dir.rglob("*.pt")):
        name = pt.stem
        if "__" not in name:
            continue
        condition, variant = name.split("__", 1)
        if variant not in ("stale_misleading", "stale_flaky"):
            continue
        # stale_* variants pin to fixed regardless of leg; both legs render the
        # same prompt content. We dedupe by keeping just one leg per task to
        # avoid double counting.
        if condition != "buggy":
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        action_table = payload["action_logits"]
        argmax_name = max(action_table.items(), key=lambda kv: kv[1]["logit"])[0]
        p_val = proj(payload["resid_pre"])
        by_variant.setdefault(variant, []).append(
            {
                "task_id": payload["task_id"],
                "argmax": argmax_name,
                "proj": p_val,
                "edit_logit": action_table["edit"]["logit"],
                "noop_logit": action_table["noop"]["logit"],
                "margin": action_table["edit"]["logit"] - action_table["noop"]["logit"],
            }
        )

    tables: dict[str, dict] = {}
    print()
    for variant, rows in by_variant.items():
        if not rows:
            continue
        print(f"=== {variant} (N={len(rows)}) ===")
        action_rates: dict[str, int] = {}
        for r in rows:
            action_rates[r["argmax"]] = action_rates.get(r["argmax"], 0) + 1
        total = len(rows)
        print(f"  argmax distribution:")
        for act in ["view", "grep", "test", "edit", "noop"]:
            n = action_rates.get(act, 0)
            print(f"    {act:5s}: {n:>3} ({100*n/total:5.1f}%)")

        edit_rows = [r for r in rows if r["argmax"] == "edit"]
        noop_rows = [r for r in rows if r["argmax"] == "noop"]
        other_rows = [r for r in rows if r["argmax"] not in ("edit", "noop")]
        edit_rate = len(edit_rows) / total
        noop_rate = len(noop_rows) / total

        # Knowing-vs-doing correlation: proj vs P(argmax=edit)
        projs = np.asarray([r["proj"] for r in rows])
        edits = np.asarray([1 if r["argmax"] == "edit" else 0 for r in rows], dtype=float)
        if len(set(edits)) > 1:
            corr = float(np.corrcoef(projs, edits)[0, 1])
        else:
            corr = float("nan")
        print(f"  proj overall : mean={projs.mean():+.3f}  median={float(np.median(projs)):+.3f}")
        print(f"  proj | edit  : mean={np.mean([r['proj'] for r in edit_rows]):+.3f}  N={len(edit_rows)}" if edit_rows else "  proj | edit  : N=0")
        print(f"  proj | noop  : mean={np.mean([r['proj'] for r in noop_rows]):+.3f}  N={len(noop_rows)}" if noop_rows else "  proj | noop  : N=0")
        print(f"  proj | other : mean={np.mean([r['proj'] for r in other_rows]):+.3f}  N={len(other_rows)}" if other_rows else "  proj | other : N=0")
        print(f"  corr(proj, argmax==edit) = {corr:+.3f}")
        print()

        # Verdict per variant
        clean_buggy_mean = float(np.mean(baseline_buggy))
        clean_fixed_mean = float(np.mean(baseline_fixed))
        flag_strong = (edit_rate >= 0.20 and (
            edit_rows and np.mean([r["proj"] for r in edit_rows]) >= clean_buggy_mean
        ))
        flag_null = edit_rate < 0.05
        tables[variant] = {
            "N": total,
            "edit_rate": edit_rate,
            "noop_rate": noop_rate,
            "argmax_distribution": action_rates,
            "proj_mean_overall": float(projs.mean()),
            "proj_mean_edit": float(np.mean([r["proj"] for r in edit_rows])) if edit_rows else None,
            "proj_mean_noop": float(np.mean([r["proj"] for r in noop_rows])) if noop_rows else None,
            "corr_proj_vs_edit": corr,
            "baseline_buggy_mean_proj": clean_buggy_mean,
            "baseline_fixed_mean_proj": clean_fixed_mean,
            "gate_strong_result": bool(flag_strong),
            "gate_null_result": bool(flag_null),
            "rows": rows,
        }

    out_path = args.stale_cache_dir / "failure_table.json"
    out_path.write_text(json.dumps(tables, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
