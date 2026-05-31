#!/usr/bin/env python3
"""Verify every local cache file is mirrored on the HF dataset before deletion.

For each local directory under results/ that we uploaded, check that the HF
dataset repo holds a file at the SAME relative-to-results path with the SAME
byte size. Size match catches truncated/partial uploads and absence; combined
with the per-directory count match it is strong evidence the upload is complete.
A random sha256 spot-check per directory adds content-level confidence.

Exit 0 only if every checked file matches. Prints a per-directory verdict so a
human (or the caller) can decide what is safe to delete.

Usage:
    .venv/bin/python scripts/verify_hf_upload.py \
        --dirs cache-real-codegemma-paraphrase-20260519T015806Z ... \
        [--sha-spotcheck 3]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO_ID = "faizancodes/no-op-circuit-caches"
RESULTS = Path("results")


def _hf_size_map(token: str) -> dict[str, int]:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    sizes: dict[str, int] = {}
    for item in api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True):
        size = getattr(item, "size", None)
        path = getattr(item, "path", None)
        if size is not None and path is not None:
            sizes[path] = int(size)
    return sizes


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _hf_sha256(token: str, repo_path: str) -> str | None:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    for item in api.list_repo_tree(REPO_ID, repo_type="dataset",
                                   path_in_repo=str(Path(repo_path).parent),
                                   recursive=False):
        if getattr(item, "path", None) == repo_path:
            lfs = getattr(item, "lfs", None)
            if lfs is not None:
                return getattr(lfs, "sha256", None)
    return None


def main(argv=None) -> int:
    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        return 2

    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True,
                   help="top-level results/ subdir names to verify")
    p.add_argument("--sha-spotcheck", type=int, default=3,
                   help="random files per dir to sha256-verify (0 to skip)")
    args = p.parse_args(argv)

    print(f"Fetching HF file tree for {REPO_ID} …", file=sys.stderr)
    hf = _hf_size_map(token)
    print(f"  HF holds {len(hf)} files", file=sys.stderr)

    import random
    rng = random.Random(0)
    all_ok = True
    for d in args.dirs:
        local_dir = RESULTS / d
        if not local_dir.is_dir():
            print(f"[{d}] LOCAL MISSING — skip"); continue
        local_files = [f for f in local_dir.rglob("*") if f.is_file()]
        missing, sizemismatch, ok = [], [], []
        for f in local_files:
            rel = str(f.relative_to(RESULTS))
            if rel not in hf:
                missing.append(rel)
            elif hf[rel] != f.stat().st_size:
                sizemismatch.append((rel, f.stat().st_size, hf[rel]))
            else:
                ok.append(rel)
        # sha256 spot-check on a random sample of the size-OK files
        sha_ok, sha_bad = 0, []
        if args.sha_spotcheck and ok:
            sample = rng.sample(ok, min(args.sha_spotcheck, len(ok)))
            for rel in sample:
                local_sha = _sha256(RESULTS / rel)
                hf_sha = _hf_sha256(token, rel)
                if hf_sha is None:
                    sha_bad.append((rel, "no-hf-sha"))  # non-LFS small file; size match suffices
                elif hf_sha != local_sha:
                    sha_bad.append((rel, "MISMATCH"))
                else:
                    sha_ok += 1
        verdict = "SAFE TO DELETE" if (not missing and not sizemismatch
                                       and not any(x[1] == "MISMATCH" for x in sha_bad)) else "DO NOT DELETE"
        if verdict != "SAFE TO DELETE":
            all_ok = False
        print(f"[{d}] {len(local_files)} local files | "
              f"size-ok {len(ok)} | missing {len(missing)} | "
              f"size-mismatch {len(sizemismatch)} | "
              f"sha-spotcheck {sha_ok}/{args.sha_spotcheck} ok"
              f"{' (non-LFS: '+str(len([x for x in sha_bad if x[1]=='no-hf-sha']))+')' if sha_bad else ''}"
              f"  -> {verdict}")
        for rel in missing[:5]:
            print(f"      MISSING ON HF: {rel}")
        for rel, ls, hs in sizemismatch[:5]:
            print(f"      SIZE MISMATCH: {rel} local={ls} hf={hs}")
        for rel, why in sha_bad:
            if why == "MISMATCH":
                print(f"      SHA MISMATCH: {rel}")

    print()
    print("ALL VERIFIED — safe to delete the listed dirs." if all_ok
          else "VERIFICATION FAILED — do NOT delete.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
