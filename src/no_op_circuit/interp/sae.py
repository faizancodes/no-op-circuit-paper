"""Minimal TopK Sparse Autoencoder.

Architecture (Makhzani 2013, Gao et al. 2024):
    encoded = encoder(x - b_dec) + b_enc       # pre-activations
    topk    = top-k by magnitude  (others zeroed)
    decoded = decoder(topk) + b_dec
    loss    = MSE(x, decoded)

Design choices:
    - TopK enforces sparsity via the top-k operator; no L1 penalty needed.
    - Decoder columns are unit-norm after every optimiser step (the "decoder
      constraint") so encoder magnitudes are meaningful and comparable.
    - We use a single shared `b_dec` (pre-encoder centring + decoder bias)
      following Anthropic's "Towards Monosemanticity" formulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_in: int
    d_sae: int
    k: int  # number of active features per position
    init_scale: float = 1.0


class TopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Encoder: d_in → d_sae
        self.W_enc = nn.Parameter(torch.empty(cfg.d_in, cfg.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        # Decoder: d_sae → d_in (no bias on decoder; b_dec is shared)
        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        # Initialise decoder as the transpose of the encoder, then unit-norm.
        with torch.no_grad():
            self.W_dec.data.copy_(self.W_enc.data.t())
            self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norm = self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.data.div_(norm)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-activation feature values (before TopK)."""
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return TopK-sparse features."""
        z = self.encode_pre(x)
        return self._topk_mask(z)

    def _topk_mask(self, z: torch.Tensor) -> torch.Tensor:
        # Keep top-k by magnitude per position; zero the rest.
        k = self.cfg.k
        if k >= z.size(-1):
            return z
        _, idx = z.abs().topk(k, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(-1, idx, 1.0)
        return z * mask

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return a @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        a = self.encode(x)
        x_hat = self.decode(a)
        mse = F.mse_loss(x_hat, x)
        with torch.no_grad():
            # explained variance: 1 - var(residual) / var(x)
            resid = x - x_hat
            ev = 1.0 - resid.var(dim=0).mean() / x.var(dim=0).mean().clamp_min(1e-8)
            l0 = (a != 0).float().sum(dim=-1).mean()
        stats = {"mse": float(mse.item()), "ev": float(ev.item()), "l0": float(l0.item())}
        return x_hat, mse, stats
