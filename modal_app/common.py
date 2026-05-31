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
