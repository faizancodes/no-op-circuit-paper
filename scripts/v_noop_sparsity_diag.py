#!/usr/bin/env python3
"""Diagnose whether v_noop is dense or sparse in the SAE basis.

For each k in a sweep we compute the cosine between v_noop and the rank-k
reconstruction using:
  (1) top-k by |encoder activation| on v_noop (what the SAE actually does),
  (2) greedy OMP (best k decoder columns by repeated residual projection),
  (3) least-squares with ALL decoder columns (the upper-bound dense recon).

If (1)/(2) at k=16 are both poor but (3) is near-perfect, v_noop is dense
in the basis: no k-sparse SAE recovers it, regardless of capacity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae", type=Path,
                   default=Path("results/sae/qwen_l24_resid_pre_TASK_d4096_k16.pt"))
    p.add_argument("--v-noop", type=Path,
                   default=Path("results/steer-20260516T021522Z/v_noop.pt"))
    p.add_argument("--ks", type=int, nargs="+",
                   default=[16, 32, 64, 128, 256, 512, 1024])
    p.add_argument("--label", type=str, default="")
    args = p.parse_args(argv)

    import torch
    import torch.nn.functional as F

    from no_op_circuit.interp.sae import SAEConfig, TopKSAE

    blob = torch.load(args.sae, map_location="cpu", weights_only=False)
    cfg = SAEConfig(**blob["config"])
    sae = TopKSAE(cfg)
    sae.load_state_dict(blob["state_dict"])
    sae.eval()
    print(f"SAE: d_in={cfg.d_in}, d_sae={cfg.d_sae}, k_train={cfg.k}")
    if args.label:
        print(f"label: {args.label}")

    v_blob = torch.load(args.v_noop, map_location="cpu", weights_only=False)
    v = v_blob["direction"].float()             # (d_in,)
    print(f"v_noop: ||v||={v.norm():.4f}")

    W_dec = sae.W_dec.detach().float()          # (d_sae, d_in), unit rows
    b_dec = sae.b_dec.detach().float()          # (d_in,)
    v_centred = v - b_dec                       # the thing we actually decode

    # (3) Dense least-squares recon: project v_centred onto span of W_dec rows.
    # Since d_sae > d_in and rows are unit-norm but not orthogonal, the rows
    # span the full input space (generically), so the upper bound is 1.0.
    # We still compute it via lstsq for completeness.
    sol = torch.linalg.lstsq(W_dec.T, v_centred.unsqueeze(1)).solution.squeeze(1)
    v_hat_dense = (W_dec.T @ sol) + b_dec
    cos_dense = float(F.cosine_similarity(v_hat_dense.unsqueeze(0), v.unsqueeze(0))[0])
    ratio_dense = float(v_hat_dense.norm() / v.norm())
    print(f"\nDense LSQ over all {cfg.d_sae} features:")
    print(f"  cosine = {cos_dense:+.4f}   ||v_hat||/||v|| = {ratio_dense:.4f}   (basis-coverage upper bound)")

    # (1) TopK by encoder activation
    print(f"\n{'k':<6} {'topk-enc cos':>14}  {'omp cos':>10}  {'ratio_enc':>10}  {'ratio_omp':>10}")
    with torch.no_grad():
        pre = sae.encode_pre(v.unsqueeze(0))[0]   # (d_sae,)
    for k in args.ks:
        if k > cfg.d_sae:
            continue
        # --- (1) top-k by encoder magnitude ---
        idx_enc = pre.abs().topk(k).indices
        a_enc = torch.zeros_like(pre); a_enc[idx_enc] = pre[idx_enc]
        v_hat_enc = (a_enc @ W_dec) + b_dec
        cos_enc = float(F.cosine_similarity(v_hat_enc.unsqueeze(0), v.unsqueeze(0))[0])
        ratio_enc = float(v_hat_enc.norm() / v.norm())

        # --- (2) Orthogonal matching pursuit ---
        # Greedy: pick decoder column with max |projection onto residual|, then
        # least-squares refit coefficients on selected set, repeat.
        residual = v_centred.clone()
        chosen: list[int] = []
        sub = W_dec[:0]
        sol_k = torch.zeros(0)
        for _ in range(k):
            # row-wise dot product of each unit-norm decoder row with residual
            scores = (W_dec @ residual).abs()
            # exclude already chosen
            if chosen:
                scores[torch.tensor(chosen)] = -1.0
            best = int(scores.argmax().item())
            chosen.append(best)
            # least-squares refit
            sub = W_dec[chosen]                   # (k', d_in)
            sol_k = torch.linalg.lstsq(sub.T, v_centred.unsqueeze(1)).solution.squeeze(1)
            residual = v_centred - sub.T @ sol_k
        v_hat_omp = (sub.T @ sol_k) + b_dec
        cos_omp = float(F.cosine_similarity(v_hat_omp.unsqueeze(0), v.unsqueeze(0))[0])
        ratio_omp = float(v_hat_omp.norm() / v.norm())

        print(f"{k:<6} {cos_enc:>+14.4f}  {cos_omp:>+10.4f}  {ratio_enc:>10.4f}  {ratio_omp:>10.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
