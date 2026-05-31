"""Project-wide paths and constants. Env loading lives here too."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
TASKS_DIR = DATA_DIR / "tasks"
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

load_dotenv(REPO_ROOT / ".env", override=False)


def env(name: str, *, required: bool = True, default: str | None = None) -> str | None:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(
            f"Missing required env var {name!r}. Add it to {REPO_ROOT / '.env'}."
        )
    return val


# Modal names — keep in sync with modal_app/common.py
MODAL_APP_NAME = "no-op-circuit"
HF_CACHE_VOLUME = "noop-hf-cache"
ACTIVATIONS_VOLUME = "noop-activations"

# Default model for phase 1. We'll add replications later.
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_MODEL_REVISION: str | None = None  # pin once we've validated the pipeline
