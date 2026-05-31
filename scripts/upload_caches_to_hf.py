#!/usr/bin/env python3
"""Upload the large activation caches to HuggingFace Datasets.

Mirrors `results/` (minus the small JSON-only dirs that live in git) to a
public dataset repo on the Hub. Each cache directory becomes a top-level
folder in the repo. `upload_large_folder` is resumable — re-running this
script after a network interruption picks up where it left off
(state cached under `results/.cache/.huggingface/large_upload/`).

What gets uploaded:
- All `cache-*/` directories (toy + real n500 + paraphrase + lex-redact +
  swap + code-only across the three models). ~66 GB of .pt activation
  caches.
- `sae/` directory (~585 MB of TopK SAE weights and ablation runs).

What stays local (and in git via -f for the JSONs):
- `monitor_real/*.json`
- `patch-*/manifest.json`, `steer-*/manifest.json` — small metadata only.

After the upload finishes the local caches can be deleted; anyone reproducing
the paper can re-fetch with:
    huggingface-cli download faizancodes/no-op-circuit-caches \
        --repo-type dataset --local-dir results/

(Or use `snapshot_download` from `huggingface_hub`.)

Run:
    .venv/bin/python scripts/upload_caches_to_hf.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


REPO_ID = "faizancodes/no-op-circuit-caches"
RESULTS = Path("results")


def main() -> int:
    load_dotenv()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set in .env", file=sys.stderr)
        return 1

    api = HfApi(token=token)

    # Create (or reuse) the public dataset repo.
    print(f"[1/2] Ensure repo exists: {REPO_ID}", file=sys.stderr)
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )

    # Upload everything under `results/` EXCEPT the small dirs that already
    # live in git or are too small to be worth uploading. `upload_large_folder`
    # is resumable: state is cached under
    # `results/.cache/.huggingface/large_upload/` so re-running picks up where
    # an interruption left off.
    print(f"[2/2] Upload large folder {RESULTS} → datasets/{REPO_ID}",
          file=sys.stderr)
    print("       (this can run for hours; progress prints every 60 s)",
          file=sys.stderr)
    print("       Resumable: re-run this script if interrupted.",
          file=sys.stderr)

    t0 = time.time()
    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(RESULTS),
        ignore_patterns=[
            "monitor_real/**",
            "patch-*/**",
            "steer-*/**",
            ".cache/**",
            ".gitignore",
        ],
        num_workers=8,
    )
    dt = time.time() - t0
    print(f"\n[done] upload_large_folder returned after {dt:.0f} s",
          file=sys.stderr)
    print(f"Browse: https://huggingface.co/datasets/{REPO_ID}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
