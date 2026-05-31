"""Modal function: train a TopK SAE on cached L24 resid_pre activations.

Reads chunk files from a Volume subdir (default `corpus_l24_resid_pre/`),
concatenates them lazily (mmap when possible), trains a TopK SAE per the
config below, saves the trained weights + training stats back to the Volume.

TODO(B5-modal): The headline Qwen SAE artifact
(`sae/qwen_l24_resid_pre_TASK_d4096_k16.pt`) was trained PRIOR to seed
pinning, so its feature indices F1954/F2669/F2950/F3129/F3171 referenced
in paper §5.2 / Appendix H.5 are non-reproducible from this script alone.
To produce a fully-reproducible artifact, re-run with `--seed 0`, then
re-run `scripts/decompose_v_noop.py` and
`modal_app/ablate_sae_features.py` to regenerate the OMP top-k results;
update Appendix H.5 F-indices/coefficients if they shift, and re-render
`paper/figures/sae_decomposition.png` via `paper/figures/render_all.py`.
This re-run is held back from the current submission to preserve the
existing numerical results pending user approval.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .common import (
    ACTIVATIONS_DIR,
    DEFAULT_GPU,
    DEFAULT_MODEL,
    HF_CACHE_DIR,
    activations_vol,
    app,
    hf_cache_vol,
)

_SAE_TIMEOUT_S = 60 * 60 * 2  # 2 hours


@app.function(
    gpu=DEFAULT_GPU,
    timeout=_SAE_TIMEOUT_S,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        ACTIVATIONS_DIR: activations_vol,
    },
)
def train_sae(
    *,
    corpus_subdir: str = "corpus_l24_resid_pre",
    out_path: str = "sae/qwen_l24_resid_pre_d24576_k32.pt",
    d_sae: int = 24576,
    k: int = 32,
    n_epochs: int = 5,
    batch_size: int = 4096,
    lr: float = 3e-4,
    warmup_steps: int = 500,
    log_every: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    import math
    import random

    import numpy as np
    import torch
    import torch.nn.functional as F  # noqa: F401  (left for parity)
    from torch.optim.lr_scheduler import LambdaLR

    from no_op_circuit.interp.sae import SAEConfig, TopKSAE

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[train_sae] seeded all RNGs with seed={seed}", flush=True)

    corpus_dir = Path(ACTIVATIONS_DIR) / corpus_subdir
    manifest_path = corpus_dir / "manifest.json"
    assert manifest_path.is_file(), f"missing {manifest_path}"
    manifest = json.loads(manifest_path.read_text())
    d_in = int(manifest["d_model"])
    n_positions = int(manifest["n_positions"])
    print(f"[train_sae] corpus: {n_positions:,} positions, d={d_in}", flush=True)

    # Load all chunks into a single (N, d) tensor on CPU (bf16 — ~30GB for 10M×1536)
    # We page them to GPU one batch at a time.
    chunk_files = sorted(corpus_dir.glob("chunk_*.pt"))
    assert chunk_files, f"no chunk_*.pt in {corpus_dir}"
    print(f"[train_sae] loading {len(chunk_files)} chunks into CPU…", flush=True)
    t0 = time.time()
    chunks = [torch.load(p, map_location="cpu", weights_only=True) for p in chunk_files]
    data = torch.cat(chunks, dim=0)
    del chunks
    N = data.shape[0]
    print(f"[train_sae] cat: shape={tuple(data.shape)} dtype={data.dtype}; load took {time.time()-t0:.1f}s", flush=True)

    # SAE training in fp32 for numerical stability; data stays bf16 on CPU,
    # cast to fp32 per batch on GPU.
    cfg = SAEConfig(d_in=d_in, d_sae=d_sae, k=k)
    sae = TopKSAE(cfg).to(device, dtype=torch.float32)
    optim = torch.optim.Adam(sae.parameters(), lr=lr)

    steps_per_epoch = N // batch_size
    total_steps = n_epochs * steps_per_epoch
    print(f"[train_sae] cfg: d_in={d_in} d_sae={d_sae} k={k} | "
          f"steps_per_epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        # cosine decay over remaining steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = LambdaLR(optim, lr_lambda=lr_lambda)

    feature_activation_count = torch.zeros(d_sae, device=device)
    log_history: list[dict[str, float]] = []
    global_step = 0
    t_train_start = time.time()

    for epoch in range(n_epochs):
        # New shuffle per epoch
        perm = torch.randperm(N)
        for batch_start in range(0, N - batch_size + 1, batch_size):
            indices = perm[batch_start : batch_start + batch_size]
            batch = data.index_select(0, indices).to(device, dtype=torch.float32, non_blocking=True)

            x_hat, mse, stats = sae(batch)
            optim.zero_grad(set_to_none=True)
            mse.backward()
            optim.step()
            sched.step()
            sae.normalize_decoder()

            # Track dead features: count which features fired this batch
            with torch.no_grad():
                a = sae.encode(batch)
                fired = (a.abs() > 0).any(dim=0)
                feature_activation_count += fired.float()

            global_step += 1
            if global_step % log_every == 0:
                with torch.no_grad():
                    dead_frac = float((feature_activation_count == 0).float().mean().item())
                msg = (
                    f"[train_sae] epoch {epoch} step {global_step}/{total_steps}  "
                    f"loss={stats['mse']:.4f}  ev={stats['ev']:.4f}  "
                    f"l0={stats['l0']:.1f}  dead={dead_frac*100:.1f}%  "
                    f"lr={sched.get_last_lr()[0]:.2e}"
                )
                print(msg, flush=True)
                log_history.append({
                    "step": global_step, "epoch": epoch,
                    "loss": stats["mse"], "ev": stats["ev"],
                    "l0": stats["l0"], "dead_frac": dead_frac,
                    "lr": sched.get_last_lr()[0],
                })
                if global_step % (log_every * 10) == 0:
                    activations_vol.commit()
        print(f"[train_sae] === end of epoch {epoch}: dead-feature rate "
              f"{float((feature_activation_count == 0).float().mean().item())*100:.1f}% ===", flush=True)

    train_seconds = time.time() - t_train_start

    # Held-out eval batch (random sample)
    with torch.no_grad():
        eval_idx = torch.randperm(N)[: min(batch_size * 4, N)]
        eval_batch = data.index_select(0, eval_idx).to(device, dtype=torch.float32)
        _, _, eval_stats = sae(eval_batch)
        dead_frac_final = float((feature_activation_count == 0).float().mean().item())

    print()
    print(f"[train_sae] FINAL eval: ev={eval_stats['ev']:.4f}  mse={eval_stats['mse']:.4f}  "
          f"l0={eval_stats['l0']:.1f}  dead={dead_frac_final*100:.1f}%", flush=True)

    out_full = Path(ACTIVATIONS_DIR) / out_path
    out_full.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k_: v.cpu() for k_, v in sae.state_dict().items()},
            "config": {"d_in": d_in, "d_sae": d_sae, "k": k},
            "training_stats": {
                "n_positions": N,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "total_steps": global_step,
                "final_ev": eval_stats["ev"],
                "final_l0": eval_stats["l0"],
                "final_dead_frac": dead_frac_final,
                "train_seconds": train_seconds,
                "log_history": log_history,
            },
            "model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "layer": 24,
            "hook_point": "resid_pre",
        },
        out_full,
    )
    activations_vol.commit()
    print(f"[train_sae] wrote {out_full}", flush=True)

    return {
        "out_path": out_path,
        "n_positions": N,
        "total_steps": global_step,
        "final_ev": eval_stats["ev"],
        "final_l0": eval_stats["l0"],
        "final_dead_frac": dead_frac_final,
        "train_seconds": train_seconds,
    }


@app.local_entrypoint()
def train_sae_entry(
    corpus_subdir: str = "corpus_l24_resid_pre",
    out_path: str = "sae/qwen_l24_resid_pre_d24576_k32.pt",
    n_epochs: int = 5,
    d_sae: int = 24576,
    k: int = 32,
    batch_size: int = 4096,
    seed: int = 0,
    model: str = DEFAULT_MODEL,
    layer: int = 24,
):
    # `model` and `layer` are metadata-only here (train_sae consumes cached
    # activations from `corpus_subdir`). Threading them through anyway for
    # interface parity with the rest of the pipeline.
    print(f"[entry] corpus_subdir={corpus_subdir}  out_path={out_path}  "
          f"model={model}  layer={layer}  d_sae={d_sae}  k={k}  seed={seed}")
    result = train_sae.remote(
        corpus_subdir=corpus_subdir,
        out_path=out_path,
        n_epochs=n_epochs,
        d_sae=d_sae,
        k=k,
        batch_size=batch_size,
        seed=seed,
    )
    print()
    print(f"[done] {result}")
