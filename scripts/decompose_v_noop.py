#!/usr/bin/env python3
"""Decompose v_noop into sparse SAE features.

Loads:
  results/steer-20260516T021522Z/v_noop.pt
  results/sae/qwen_l24_resid_pre_d24576_k32.pt

Reports two complementary feature rankings:
  - by encoder activation: which features fire when v_noop is encoded
  - by decoder cosine: which decoder vectors best reconstruct v_noop

Saves results/sae/v_noop_features.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sae", type=Path,
                   default=Path("results/sae/qwen_l24_resid_pre_d24576_k32.pt"))
    p.add_argument("--v-noop", type=Path,
                   default=Path("results/steer-20260516T021522Z/v_noop.pt"))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", type=Path,
                   default=Path("results/sae/v_noop_features.json"))
    args = p.parse_args(argv)

    import torch
    import torch.nn.functional as F

    from no_op_circuit.interp.sae import SAEConfig, TopKSAE

    if not args.sae.is_file():
        print(f"missing SAE checkpoint: {args.sae}", file=sys.stderr)
        return 2

    blob = torch.load(args.sae, map_location="cpu", weights_only=False)
    cfg = SAEConfig(**blob["config"])
    sae = TopKSAE(cfg)
    sae.load_state_dict(blob["state_dict"])
    sae.eval()
    print(f"SAE: d_in={cfg.d_in}, d_sae={cfg.d_sae}, k={cfg.k}")
    print(f"  training stats: ev={blob['training_stats']['final_ev']:.4f}, "
          f"l0={blob['training_stats']['final_l0']:.1f}, "
          f"dead={blob['training_stats']['final_dead_frac']*100:.1f}%")

    v_blob = torch.load(args.v_noop, map_location="cpu", weights_only=False)
    v = v_blob["direction"].float()  # (d_in,)
    v_unit = v / v.norm()
    print(f"v_noop: layer={v_blob['layer']}, pos={v_blob['position']}, ||v||={v_blob['norm']:.3f}")

    # 1) Encoder activations on v_noop (single position, batch=1)
    with torch.no_grad():
        pre = sae.encode_pre(v.unsqueeze(0))[0]   # (d_sae,) full pre-activations
        topk_idx = pre.abs().topk(cfg.k).indices  # indices that survive TopK
        a = torch.zeros_like(pre); a[topk_idx] = pre[topk_idx]
        # Reconstruct
        x_hat = sae.decode(a.unsqueeze(0))[0]
        recon_cos = float(F.cosine_similarity(x_hat.unsqueeze(0), v.unsqueeze(0))[0])
        recon_norm_ratio = float(x_hat.norm() / v.norm())

    print(f"\nv_noop SAE reconstruction:")
    print(f"  cosine(v_hat, v) = {recon_cos:+.4f}")
    print(f"  ||v_hat|| / ||v|| = {recon_norm_ratio:.4f}")

    # Top-K by encoder activation (within the TopK-surviving set)
    enc_top = a.abs().topk(args.top_k)
    enc_indices = enc_top.indices.tolist()
    enc_values = a[enc_top.indices].tolist()

    # 2) Decoder cosine similarity with v
    W_dec = sae.W_dec  # (d_sae, d_in)
    cos_sim = F.cosine_similarity(W_dec, v.unsqueeze(0), dim=1)  # (d_sae,)
    dec_top = cos_sim.abs().topk(args.top_k)
    dec_indices = dec_top.indices.tolist()
    dec_values = cos_sim[dec_top.indices].tolist()

    print(f"\nTop-{args.top_k} features by encoder activation:")
    print(f"  {'rank':<5} {'feat_idx':<10} {'enc_act':>10}  {'dec_cos':>10}")
    for r, (idx, val) in enumerate(zip(enc_indices, enc_values)):
        print(f"  {r:<5} {idx:<10} {val:+10.4f}  {cos_sim[idx]:+10.4f}")

    print(f"\nTop-{args.top_k} features by decoder cosine with v_noop:")
    print(f"  {'rank':<5} {'feat_idx':<10} {'dec_cos':>10}  {'enc_act':>10}")
    for r, (idx, val) in enumerate(zip(dec_indices, dec_values)):
        print(f"  {r:<5} {idx:<10} {val:+10.4f}  {a[idx]:+10.4f}")

    # Rank-agreement check
    agree = len(set(enc_indices) & set(dec_indices))
    disagree = args.top_k - agree
    print(f"\nrank agreement: {agree}/{args.top_k} features in both top-K lists")
    if disagree > 3:
        print(f"WARNING: top-K by enc vs dec disagree on >{disagree} features (noisy SAE?)")

    # Combine into a unified top set (union) for downstream characterisation
    unified = list(dict.fromkeys(enc_indices + dec_indices))  # preserve order, dedupe

    summary = {
        "v_noop_recon_cosine": recon_cos,
        "v_noop_recon_norm_ratio": recon_norm_ratio,
        "v_noop_norm": v_blob["norm"],
        "sae_config": blob["config"],
        "sae_training_stats": blob["training_stats"],
        "top_features": [
            {
                "feature_idx": int(idx),
                "enc_activation": float(a[idx]),
                "dec_cosine": float(cos_sim[idx]),
                "dec_norm": float(W_dec[idx].norm()),
                "rank_by_enc": enc_indices.index(idx) if idx in enc_indices else None,
                "rank_by_dec": dec_indices.index(idx) if idx in dec_indices else None,
            }
            for idx in unified
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
