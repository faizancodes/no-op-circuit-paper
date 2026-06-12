"""Shared Modal app/image/volume/secret definitions.

Importing this module is the canonical way to get the App handle. Keep
function definitions in sibling modules that import from here.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)

APP_NAME = "no-op-circuit"

# ---------------------------------------------------------------------------
# Image: minimal CUDA-enabled stack for HF inference + interp hooks.
# We pin reasonably loose ranges so we pick up bug fixes; tighten for a paper.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.46",
        "accelerate>=1.0",
        "safetensors",
        "huggingface_hub",
        "hf_transfer",
        "numpy",
        "pyyaml",
        "tqdm",
        "einops",
        "python-dotenv",  # imported by no_op_circuit.config and modal_app.common
        "datasets",       # streaming corpora for SAE training
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Ship the project library so Modal functions can `from no_op_circuit ...`.
    .add_local_python_source("no_op_circuit", copy=False)
)

# ---------------------------------------------------------------------------
# Volumes: persistent storage for HF model cache and our activation outputs.
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("noop-hf-cache", create_if_missing=True)
activations_vol = modal.Volume.from_name("noop-activations", create_if_missing=True)

HF_CACHE_DIR = "/cache/hf"
ACTIVATIONS_DIR = "/cache/activations"

# ---------------------------------------------------------------------------
# Secrets: HF token for gated/private model downloads, OpenRouter for any
# LLM-assisted dataset generation. We pick them up from the local .env at
# import time and ship them as a single inline Secret.
# ---------------------------------------------------------------------------
_secret_env: dict[str, str | None] = {
    k: v
    for k, v in {
        "HF_TOKEN": os.environ.get("HF_TOKEN"),
        "HF_HUB_TOKEN": os.environ.get("HF_TOKEN"),  # newer HF naming
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY"),
    }.items()
    if v
}
runtime_secret = modal.Secret.from_dict(_secret_env)

# ---------------------------------------------------------------------------
# App. Functions are attached in sibling modules.
# ---------------------------------------------------------------------------
app = modal.App(name=APP_NAME, image=image, secrets=[runtime_secret])

DEFAULT_GPU = "A10G"  # temporarily for task-corpus cache + small SAE retrain
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_TIMEOUT_S = 60 * 30  # 30 min per call ceiling

# ---------------------------------------------------------------------------
# GPU tiering for bigger same-family models.
#
# Bigger models need bigger cards than the A10G default. modal 1.3.3 has no
# per-call GPU override for plain Functions (Function.with_options is Cls-only;
# clone() takes no overrides), so we PRE-REGISTER one Modal Function per GPU
# tier (distinct name=, same underlying impl) and dispatch the right one per
# model. Tiers are single-GPU so the residual-patch hooks stay on one device
# (no device_map="auto", which would split layers across devices).
# ---------------------------------------------------------------------------
GPU_TIERS = (
    "T4", "L4", "A10G", "L40S",
    "A100-40GB", "A100", "A100-80GB", "H100", "H200", "B200",
)

_GPU_BIG = ("70b", "65b", "34b", "33b", "32b", "30b", "27b")  # ~27-70B -> 80GB card
_GPU_MID = ("13b", "14b", "15b")                               # ~13-15B -> A100-80GB
_GPU_SMALLMID = ("6.7b", "6b", "7b", "8b", "9b")               # ~6-9B   -> A100-40GB


def gpu_for_model(model_name: str) -> str:
    """Heuristic single-GPU tier for a HF model slug (by param-count substring)."""
    s = model_name.lower()
    if any(k in s for k in _GPU_BIG):
        return "H100"
    if any(k in s for k in _GPU_MID):
        return "A100-80GB"
    if any(k in s for k in _GPU_SMALLMID):
        return "A100-40GB"
    return "A10G"  # <= ~3B


# Minimum per-call timeout by tier (big models run long patching grids); the
# registered timeout is max(this, the job's own base timeout).
_TIER_MIN_TIMEOUT = {"A100-40GB": 2 * 60 * 60, "A100-80GB": 4 * 60 * 60, "H100": 6 * 60 * 60}


def _tier_slug(tier: str) -> str:
    return tier.lower().replace("-", "").replace(":", "x")


def resolve_gpu(model_name: str, gpu: str = "auto") -> str:
    """Map a CLI --gpu value to a concrete tier ("" keeps the declared GPU).

    - "auto"             -> gpu_for_model(model_name)
    - "" | "default"     -> "" (do not override; use the Function's declared GPU)
    - explicit tier str  -> normalized against GPU_TIERS (case-insensitive);
                            unknown/`tier:count` strings pass through unchanged.
    """
    if gpu in ("", "default"):
        return ""
    if gpu == "auto":
        return gpu_for_model(model_name)
    for t in GPU_TIERS:
        if gpu.lower() == t.lower():
            return t
    return gpu  # future tiers / "H100:2" count syntax pass through


def register_tiers(raw_fn, base_name: str, *, volumes: dict, base_timeout: int,
                   extra_tiers: tuple = ("A100-40GB", "A100-80GB", "H100")) -> dict:
    """Register one Modal Function per GPU tier (besides the caller's A10G
    default) and return a {tier: Function} registry. A distinct name= per tier
    keeps them separate; the same raw impl backs all of them."""
    reg: dict = {}
    for tier in extra_tiers:
        reg[tier] = app.function(
            name=f"{base_name}__{_tier_slug(tier)}",
            gpu=tier,
            timeout=max(base_timeout, _TIER_MIN_TIMEOUT.get(tier, 0)),
            volumes=volumes,
        )(raw_fn)
    return reg


def record_spawn(kind: str, model: str, run_id: str, call_id: str) -> None:
    """Persist a spawned FunctionCall handle so results are retrievable after the
    laptop reconnects. The remote function commits its outputs to the Volume
    regardless, so this is a convenience index, not the source of truth."""
    import json
    from datetime import datetime, timezone

    from no_op_circuit.config import RESULTS_DIR

    rec = {
        "kind": kind, "model": model, "run_id": run_id, "call_id": call_id,
        "spawned_at": datetime.now(timezone.utc).isoformat(),
    }
    spawn_dir = RESULTS_DIR / "spawns"
    spawn_dir.mkdir(parents=True, exist_ok=True)
    with open(spawn_dir / "index.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spawn.json").write_text(json.dumps(rec, indent=2))
    print(
        f"[{kind}] SPAWNED detached call={call_id} run_id={run_id}\n"
        f"  → runs to completion on Modal even if this laptop disconnects;\n"
        f"  → outputs commit to volume 'noop-activations/{run_id}';\n"
        f"  → fetch later: modal volume get noop-activations {run_id} results/{run_id}",
        flush=True,
    )


def pick_tier(registry: dict, model_name: str, gpu: str = "auto",
              *, default_tier: str = "A10G"):
    """Return the pre-registered Modal Function for `model_name`'s GPU tier.

    `registry` must include `default_tier`. gpu="auto" picks by model size;
    "" / "default" -> default_tier; an explicit tier must be pre-registered.
    """
    tier = resolve_gpu(model_name, gpu) or default_tier
    fn = registry.get(tier)
    if fn is None:
        raise SystemExit(
            f"GPU tier {tier!r} not registered for this job; available: "
            f"{sorted(registry)}. Pre-register it or pass a different --gpu."
        )
    return fn
