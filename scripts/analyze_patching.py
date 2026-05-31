#!/usr/bin/env python3
"""Aggregate per-pair patching heatmaps into the canonical figure.

For each pair, the patching run saved:
  clean_buggy_margin                — `edit - noop` on clean buggy forward
  patched_margins[layer][position]  — `edit - noop` after substituting the
                                       FIXED residual at (layer, position)
                                       into the buggy forward

The hypothesis-confirming quantity is:
  shift = clean_buggy_margin - patched_margin   (positive ⇒ patch pulled
                                                 the action toward noop,
                                                 matching the fixed side)

We average shift across pairs and report the (layer, position) cells with
the strongest mean shift — those are the candidate sites for the causal
no-op direction.

    python scripts/analyze_patching.py results/patch-<run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _print_heatmap(label: str, mean_shift, layer_indices, position_offsets):
    P = mean_shift.shape[1]
    print(f"=== {label} ===")
    print(f"  {'layer':<5} " + "  ".join(f"p={p:>3}" for p in position_offsets))
    for i, layer in enumerate(layer_indices):
        row = "  ".join(f"{mean_shift[i, j]:+.3f}" for j in range(P))
        print(f"  L{layer:<3} {row}")


def _print_top(label, mean, median, frac, layer_indices, position_offsets, k=10):
    P = mean.shape[1]
    flat = mean.flatten()
    idx_sorted = flat.argsort()[::-1]
    print(f"\n=== {label}: top {k} (layer, position) ===")
    for rank in range(min(k, len(flat))):
        flat_idx = int(idx_sorted[rank])
        li, pj = divmod(flat_idx, P)
        print(
            f"  L{layer_indices[li]:<3} pos={position_offsets[pj]:>3}  "
            f"mean={mean[li, pj]:+.3f}  median={median[li, pj]:+.3f}  positive: {frac[li, pj]*100:.0f}%"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args(argv)

    import numpy as np
    import torch

    files = sorted(args.run_dir.rglob("*__patch.pt"))
    if not files:
        print(f"no patching heatmap files under {args.run_dir}", file=sys.stderr)
        return 2

    layer_indices = None
    position_offsets = None
    shifts_f2b: list[np.ndarray] = []
    shifts_b2f: list[np.ndarray] = []
    clean_buggy: list[float] = []
    clean_fixed: list[float] = []
    task_ids: list[str] = []
    bidirectional = False

    for f in files:
        payload = torch.load(f, map_location="cpu", weights_only=False)
        layer_indices = layer_indices or payload["layer_indices"]
        position_offsets = position_offsets or payload["position_offsets"]
        cb = float(payload["clean_buggy_margin"])
        clean_buggy.append(cb)
        # F → B: clean_buggy − patched (positive ⇒ moved toward fixed/noop)
        f2b = np.asarray(payload.get("patched_margins_f2b", payload["patched_margins"]), dtype=np.float32)
        shifts_f2b.append(cb - f2b)

        if payload.get("bidirectional") and "patched_margins_b2f" in payload:
            bidirectional = True
            cf = float(payload["clean_fixed_margin"])
            clean_fixed.append(cf)
            b2f = np.asarray(payload["patched_margins_b2f"], dtype=np.float32)
            # B → F: patched − clean_fixed (positive ⇒ moved toward buggy/edit)
            shifts_b2f.append(b2f - cf)
        task_ids.append(payload["task_id"])

    arr_f2b = np.stack(shifts_f2b, axis=0)
    mean_f2b = arr_f2b.mean(axis=0)
    median_f2b = np.median(arr_f2b, axis=0)
    frac_pos_f2b = (arr_f2b > 0).mean(axis=0)

    assert layer_indices is not None and position_offsets is not None
    print(f"pairs              : {len(files)}")
    print(f"layer_indices      : {layer_indices}")
    print(f"position_offsets   : {position_offsets}")
    print(f"bidirectional      : {bidirectional}")
    print(f"clean buggy margins: mean={np.mean(clean_buggy):+.3f}  median={np.median(clean_buggy):+.3f}")
    if bidirectional:
        print(f"clean fixed margins: mean={np.mean(clean_fixed):+.3f}  median={np.median(clean_fixed):+.3f}")
    print()

    _print_heatmap(
        "F→B mean shift (clean_buggy − patched)  positive ⇒ patch pulled toward NOOP",
        mean_f2b, layer_indices, position_offsets,
    )
    _print_top("F→B", mean_f2b, median_f2b, frac_pos_f2b, layer_indices, position_offsets, k=args.top_k)

    arr_b2f = mean_b2f = median_b2f = frac_pos_b2f = None
    if bidirectional:
        arr_b2f = np.stack(shifts_b2f, axis=0)
        mean_b2f = arr_b2f.mean(axis=0)
        median_b2f = np.median(arr_b2f, axis=0)
        frac_pos_b2f = (arr_b2f > 0).mean(axis=0)
        print()
        _print_heatmap(
            "B→F mean shift (patched − clean_fixed)  positive ⇒ patch pulled toward EDIT",
            mean_b2f, layer_indices, position_offsets,
        )
        _print_top("B→F", mean_b2f, median_b2f, frac_pos_b2f, layer_indices, position_offsets, k=args.top_k)

        # Symmetry check: cells where BOTH directions show strong positive shift.
        symmetric = np.minimum(mean_f2b, mean_b2f)
        print()
        print("=== symmetric strength = min(F→B, B→F) — top cells where BOTH directions work ===")
        flat = symmetric.flatten()
        P = mean_f2b.shape[1]
        idx_sorted = flat.argsort()[::-1]
        for rank in range(min(args.top_k, len(flat))):
            flat_idx = int(idx_sorted[rank])
            li, pj = divmod(flat_idx, P)
            f2b_v = mean_f2b[li, pj]
            b2f_v = mean_b2f[li, pj]
            sym = symmetric[li, pj]
            print(
                f"  L{layer_indices[li]:<3} pos={position_offsets[pj]:>3}  "
                f"F→B={f2b_v:+.3f}  B→F={b2f_v:+.3f}  min={sym:+.3f}"
            )

    out = args.run_dir / "aggregated.npz"
    save_kwargs: dict[str, np.ndarray] = dict(
        layer_indices=np.asarray(layer_indices),
        position_offsets=np.asarray(position_offsets),
        shift_per_pair_f2b=arr_f2b,
        mean_shift_f2b=mean_f2b,
        median_shift_f2b=median_f2b,
        frac_positive_f2b=frac_pos_f2b,
        task_ids=np.asarray(task_ids),
        clean_buggy=np.asarray(clean_buggy),
        # legacy aliases for prior callers
        shift_per_pair=arr_f2b,
        mean_shift=mean_f2b,
        median_shift=median_f2b,
        frac_positive=frac_pos_f2b,
        clean_margins=np.asarray(clean_buggy),
    )
    if bidirectional:
        save_kwargs["shift_per_pair_b2f"] = arr_b2f  # noqa: F821 (defined in the bidirectional branch above)
        save_kwargs["mean_shift_b2f"] = mean_b2f      # noqa: F821
        save_kwargs["median_shift_b2f"] = median_b2f  # noqa: F821
        save_kwargs["frac_positive_b2f"] = frac_pos_b2f  # noqa: F821
        save_kwargs["clean_fixed"] = np.asarray(clean_fixed)
    np.savez(str(out), **save_kwargs)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
