#!/usr/bin/env python3
"""Build the OMP-based distributed feature set for v_noop.

Phase 3 (encoder-based) shows v_noop is dense in the SAE basis. We instead
pick features via Orthogonal Matching Pursuit: greedily select decoder
columns that best explain the residual after re-fitting, up to k=128.

We also rank by signed contribution to v_noop's direction:
    contrib_i = (coef_i * W_dec[i]) . (v / ||v||)
The top-20 by |contrib| become the characterisation target.

Output schema mirrors `v_noop_features.json` so characterise_sae_features
and ablate_sae_features can consume it unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def omp(W_dec, v_centred, k):
    import torch
    residual = v_centred.clone()
    chosen: list[int] = []
    sub = W_dec[:0]
    sol = None
    for _ in range(k):
        scores = (W_dec @ residual).abs()
        if chosen:
            scores[torch.tensor(chosen)] = -1.0
        best = int(scores.argmax().item())
        chosen.append(best)
        sub = W_dec[chosen]
        sol = torch.linalg.lstsq(sub.T, v_centred.unsqueeze(1)).solution.squeeze(1)
        residual = v_centred - sub.T @ sol
    return chosen, sol  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae", type=Path,
                   default=Path("results/sae/qwen_l24_resid_pre_TASK_d4096_k16.pt"))
    p.add_argument("--v-noop", type=Path,
                   default=Path("results/steer-20260516T021522Z/v_noop.pt"))
    p.add_argument("--k", type=int, default=128, help="OMP top-K")
    p.add_argument("--top-n-characterise", type=int, default=20)
    p.add_argument("--out", type=Path,
                   default=Path("results/sae/v_noop_features_DISTRIBUTED.json"))
    args = p.parse_args(argv)

    import torch
    import torch.nn.functional as F

    from no_op_circuit.interp.sae import SAEConfig, TopKSAE

    blob = torch.load(args.sae, map_location="cpu", weights_only=False)
    cfg = SAEConfig(**blob["config"])
    sae = TopKSAE(cfg); sae.load_state_dict(blob["state_dict"]); sae.eval()
    print(f"SAE: d_in={cfg.d_in}, d_sae={cfg.d_sae}, k_train={cfg.k}")

    v_blob = torch.load(args.v_noop, map_location="cpu", weights_only=False)
    v = v_blob["direction"].float()
    v_unit = v / v.norm()
    print(f"v_noop: ||v||={v.norm():.4f}")

    W_dec = sae.W_dec.detach().float()
    b_dec = sae.b_dec.detach().float()
    v_centred = v - b_dec

    chosen, coefs = omp(W_dec, v_centred, args.k)
    v_hat = (W_dec[chosen].T @ coefs) + b_dec
    cos = float(F.cosine_similarity(v_hat.unsqueeze(0), v.unsqueeze(0))[0])
    norm_ratio = float(v_hat.norm() / v.norm())
    print(f"OMP-{args.k} reconstruction: cos={cos:+.4f}  ||v_hat||/||v||={norm_ratio:.4f}")

    # Per-feature signed contribution along v direction
    contribs: list[tuple[int, float, float]] = []
    for rank_, (idx, coef) in enumerate(zip(chosen, coefs.tolist())):
        dec_vec = W_dec[idx]
        contrib = float((coef * dec_vec) @ v_unit)
        dec_cos = float(F.cosine_similarity(dec_vec.unsqueeze(0), v.unsqueeze(0))[0])
        contribs.append((rank_, idx, coef, contrib, dec_cos))  # type: ignore[arg-type]

    contribs.sort(key=lambda r: -abs(r[3]))

    out_features = [
        {
            "feature_idx": int(idx),
            "omp_rank": int(rank_),
            "omp_coef": float(coef),
            "v_contribution": float(contrib),
            "dec_cosine": float(dec_cos),
            "dec_norm": float(W_dec[idx].norm()),
        }
        for (rank_, idx, coef, contrib, dec_cos) in contribs
    ]

    print(f"\nTop-{args.top_n_characterise} of {args.k} by |contribution to v|:")
    print(f"  {'rank':<5} {'feat_idx':<10} {'coef':>10} {'contrib':>10} {'dec_cos':>10}")
    for r, feat in enumerate(out_features[: args.top_n_characterise]):
        print(f"  {r:<5} {feat['feature_idx']:<10} "
              f"{feat['omp_coef']:>+10.4f} {feat['v_contribution']:>+10.4f} "
              f"{feat['dec_cosine']:>+10.4f}")

    summary = {
        "method": "OMP",
        "k": args.k,
        "sae_path": str(args.sae),
        "v_noop_path": str(args.v_noop),
        "v_noop_recon_cosine": cos,
        "v_noop_recon_norm_ratio": norm_ratio,
        "v_noop_norm": v_blob["norm"],
        "sae_config": blob["config"],
        "sae_training_stats": {
            "final_ev": blob["training_stats"]["final_ev"],
            "final_l0": blob["training_stats"]["final_l0"],
            "final_dead_frac": blob["training_stats"]["final_dead_frac"],
            "n_positions": blob["training_stats"]["n_positions"],
        },
        # Compatible with characterise/ablate scripts that key off `top_features`
        "top_features": out_features,
        "top_features_for_characterisation": out_features[: args.top_n_characterise],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
