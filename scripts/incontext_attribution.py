#!/usr/bin/env python3
"""Per-position v_noop projection across the cached last-32 token window.

For each toy task × {buggy, fixed} we project the L24 resid_pre at every
cached position onto unit v_noop. Across the 49 toys, the token IDs at
these 32 positions are IDENTICAL (chat-template machinery + question text
+ `Action: ` suffix); the differential between buggy and fixed projections
is therefore purely attentional information flow from the earlier
(varying) test/code content.

The toy cache stores only last_k=32 positions (a limitation of the
existing artefact — re-caching the full 800–1300 token prompts would
require fresh Modal compute and is out of scope for this free-wins
round). We report what the cache supports:

  - Right-aligned mean projection vs offset from the Action position
    (offsets ∈ [-31, 0]), separately for buggy and fixed prompts. Shows
    where in the prompt tail the buggy/fixed projection gap emerges.
  - A coarse three-section split within the 32-token window
    (question-text / chat-template-end / Action-suffix), aggregated by
    matching offset ranges to token roles using the (constant)
    last-32 token sequence.

Outputs:
  results/attribution/incontext_projections.npz
  results/attribution/section_projections.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir",
                   type=Path,
                   default=Path("results/cache-20260515T221105Z/cache-20260515T221105Z"))
    p.add_argument("--v-noop",
                   type=Path,
                   default=Path("results/steer-20260516T021522Z/v_noop.pt"))
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--out-dir",
                   type=Path,
                   default=Path("results/attribution"))
    args = p.parse_args(argv)

    import numpy as np
    import torch

    args.out_dir.mkdir(parents=True, exist_ok=True)

    v_blob = torch.load(args.v_noop, map_location="cpu", weights_only=False)
    v = v_blob["direction"].float()
    v_unit = (v / v.norm()).numpy()
    print(f"v_noop: layer={v_blob['layer']}, pos={v_blob['position']}, "
          f"||v||={v_blob['norm']:.3f}")

    by_task: dict[str, dict[str, "np.ndarray"]] = {}
    reference_token_ids: list[int] | None = None
    n_loaded = 0
    for d in sorted(args.cache_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            pb = torch.load(d / "buggy__code_tests.pt", map_location="cpu", weights_only=False)
            pf = torch.load(d / "fixed__code_tests.pt", map_location="cpu", weights_only=False)
        except FileNotFoundError:
            continue
        if reference_token_ids is None:
            reference_token_ids = pb["input_ids_last_k"].tolist()
        proj_b = (pb["resid_pre"][args.layer, 0, :, :].float().numpy() @ v_unit)
        proj_f = (pf["resid_pre"][args.layer, 0, :, :].float().numpy() @ v_unit)
        by_task[d.name] = {"buggy": proj_b, "fixed": proj_f}
        n_loaded += 1
    print(f"loaded {n_loaded} tasks")

    K = next(iter(by_task.values()))["buggy"].shape[0]
    print(f"cached positions per task: K = {K}")

    # Verify token sequence is constant across tasks (chat template + question
    # text + Action: suffix should be identical for the code_tests variant).
    n_mismatch = 0
    for d in sorted(args.cache_dir.iterdir())[: min(10, n_loaded)]:
        if not d.is_dir(): continue
        try:
            pf = torch.load(d / "fixed__code_tests.pt", map_location="cpu", weights_only=False)
        except FileNotFoundError: continue
        if pf["input_ids_last_k"].tolist() != reference_token_ids:
            n_mismatch += 1
    print(f"last-32 token sequence constant across first 10 tasks: "
          f"{'YES' if n_mismatch == 0 else f'NO ({n_mismatch} mismatches)'}")

    # Stack into (n_tasks, K) arrays per condition
    buggy_arr = np.stack([by_task[t]["buggy"] for t in sorted(by_task)])
    fixed_arr = np.stack([by_task[t]["fixed"] for t in sorted(by_task)])
    n_tasks = buggy_arr.shape[0]

    # Right-aligned: offsets [-K+1, 0]
    offsets = np.arange(-K + 1, 1)
    mean_b = buggy_arr.mean(axis=0); sem_b = buggy_arr.std(axis=0, ddof=1) / np.sqrt(n_tasks)
    mean_f = fixed_arr.mean(axis=0); sem_f = fixed_arr.std(axis=0, ddof=1) / np.sqrt(n_tasks)
    diff = mean_f - mean_b  # fixed - buggy

    # Per-offset paired Wilcoxon (does fixed > buggy at this offset?)
    from scipy.stats import wilcoxon
    per_offset_p = np.full(K, np.nan)
    for i in range(K):
        d = fixed_arr[:, i] - buggy_arr[:, i]
        if np.allclose(d, 0):
            per_offset_p[i] = 1.0
        else:
            try:
                _, p_ = wilcoxon(d, alternative="greater")
                per_offset_p[i] = p_
            except ValueError:
                per_offset_p[i] = 1.0

    # Decode tokens with the model tokenizer for labelling.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
        token_strs = [tok.decode([tid]).replace("\n", "\\n") for tid in reference_token_ids]
    except Exception as e:
        print(f"WARN: tokenizer load failed: {e}; using token ids only")
        token_strs = [str(tid) for tid in reference_token_ids]

    print()
    print(f"{'offset':>6} {'token':<14} {'mean(buggy)':>11} {'mean(fixed)':>11} "
          f"{'fixed-buggy':>11} {'Wilcoxon p':>11}")
    print("-" * 70)
    for i in range(K):
        print(f"{offsets[i]:>+6d} {token_strs[i]!r:<14} {mean_b[i]:>+11.3f} "
              f"{mean_f[i]:>+11.3f} {diff[i]:>+11.3f} {per_offset_p[i]:>11.2e}")

    # Coarse section grouping based on the (constant) token sequence.
    # For Qwen + code_tests variant the canonical last-32 layout is:
    #   offsets [-31, -19]  question text "...What is the next action? ..."
    #   offsets [-18, -7]   chat-template close (<|im_end|> + <|im_start|>assistant)
    #   offsets [-6, 0]     "Action: " suffix
    # We define section boundaries by index, not by token-string matching, since
    # the layout is constant across tasks (verified above).
    sections = {
        "question_text":  (0, K - 19),
        "chat_template":  (K - 18, K - 7),
        "action_suffix":  (K - 6, K),
    }
    section_out: dict[str, dict] = {}
    print(f"\n{'section':<18} {'positions':<18} {'mean(buggy)':>11} "
          f"{'mean(fixed)':>11} {'fixed-buggy':>11}")
    print("-" * 75)
    for name, (lo, hi) in sections.items():
        b_slice = buggy_arr[:, lo:hi]
        f_slice = fixed_arr[:, lo:hi]
        # Per-task per-token then averaged
        per_task_buggy = b_slice.mean(axis=1)
        per_task_fixed = f_slice.mean(axis=1)
        d = per_task_fixed - per_task_buggy
        try:
            _, sec_p = wilcoxon(d, alternative="greater") if not np.allclose(d, 0) else (None, 1.0)
        except ValueError:
            sec_p = 1.0
        section_out[name] = {
            "offset_range": [int(lo - K + 1), int(hi - 1 - K + 1)],
            "n_tokens": int(hi - lo),
            "mean_buggy": float(per_task_buggy.mean()),
            "mean_fixed": float(per_task_fixed.mean()),
            "differential": float(d.mean()),
            "sem_differential": float(d.std(ddof=1) / np.sqrt(len(d))),
            "wilcoxon_p_one_sided": float(sec_p),
        }
        print(f"{name:<18} [{lo - K + 1:+4d}, {hi - 1 - K + 1:+4d}] "
              f"({hi - lo:>2d}t)  {per_task_buggy.mean():>+11.3f} "
              f"{per_task_fixed.mean():>+11.3f} {d.mean():>+11.3f}  "
              f"Wilcoxon p={sec_p:.2e}")

    # Save
    out_npz = args.out_dir / "incontext_projections.npz"
    np.savez(
        str(out_npz),
        offsets=offsets,
        token_ids=np.asarray(reference_token_ids),
        token_strs=np.asarray(token_strs, dtype=object),
        buggy=buggy_arr,
        fixed=fixed_arr,
        mean_buggy=mean_b, sem_buggy=sem_b,
        mean_fixed=mean_f, sem_fixed=sem_f,
        differential=diff,
        per_offset_wilcoxon_p=per_offset_p,
    )
    print(f"\nwrote {out_npz}")
    out_json = args.out_dir / "section_projections.json"
    out_json.write_text(json.dumps({
        "n_tasks": n_tasks, "K": int(K), "layer": int(args.layer),
        "sections": section_out,
        "first_offset_p_below_005": int(offsets[np.where(per_offset_p < 0.05)[0][0]])
            if (per_offset_p < 0.05).any() else None,
    }, indent=2))
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
