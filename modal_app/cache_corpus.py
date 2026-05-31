"""Modal function: cache L24 resid_pre over a streamed Python corpus.

For SAE training we want ~10M positions of resid_pre at the intervention
layer. We stream `bigcode/the-stack-smol[data/python]` (fallback:
`codeparrot/codeparrot-clean`), tokenise each doc to max 2048 tokens, run
a forward pass, and grab `hidden_states[24]` (which equals the input to
layer 24 = resid_pre L24 under the HF convention).

Activations are saved in flat (N, D) chunks of ~500K positions in bf16
(~1.5 GB each) to the activations Volume. A manifest summarises totals.
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

_CORPUS_TIMEOUT_S = 60 * 60  # 1 hour


@app.function(
    gpu=DEFAULT_GPU,
    timeout=_CORPUS_TIMEOUT_S,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        ACTIVATIONS_DIR: activations_vol,
    },
)
def cache_corpus(
    *,
    out_subdir: str = "corpus_l24_resid_pre",
    model_name: str = DEFAULT_MODEL,
    layer: int = 24,
    max_positions: int = 5_000_000,
    chunk_positions: int = 500_000,
    max_seq_len: int = 1024,    # shorter to fit more in a batch
    batch_size: int = 8,         # forwards multiple docs at once
    dtype: str = "bfloat16",
    dataset: str = "bigcode/the-stack-smol",
    dataset_subdir: str = "data/python",
    dataset_split: str = "train",
    n_docs_cap: int = 100_000,
) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[cache_corpus] loading {model_name} ({dtype})…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    n_layers_total = len(model.model.layers)
    d_model = model.config.hidden_size
    print(f"[cache_corpus] loaded in {time.time()-t0:.1f}s; n_layers={n_layers_total}, d={d_model}", flush=True)
    assert 0 <= layer <= n_layers_total, f"layer {layer} out of range for {n_layers_total} layers"

    print(f"[cache_corpus] streaming {dataset} (subdir={dataset_subdir})…", flush=True)
    try:
        ds = load_dataset(dataset, data_dir=dataset_subdir, split=dataset_split, streaming=True)
    except Exception as exc:
        print(f"[cache_corpus] primary dataset failed: {exc!r}; trying codeparrot fallback…", flush=True)
        ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)

    out_dir = Path(ACTIVATIONS_DIR) / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any prior partial run so chunk indices are deterministic.
    for old in out_dir.glob("chunk_*.pt"):
        old.unlink()

    buffer: list[torch.Tensor] = []  # list of (n_i, D) tensors on CPU bf16
    buffer_pos = 0
    total_positions = 0
    n_docs_used = 0
    n_chunks = 0

    def flush_chunk():
        nonlocal n_chunks, buffer, buffer_pos
        if not buffer:
            return
        chunk = torch.cat(buffer, dim=0).contiguous()
        path = out_dir / f"chunk_{n_chunks:04d}.pt"
        torch.save(chunk.to(torch.bfloat16), path)
        n_chunks += 1
        print(
            f"[cache_corpus]   wrote {path.name} ({chunk.shape[0]:,} positions, "
            f"total {total_positions:,})",
            flush=True,
        )
        if n_chunks % 5 == 0:
            activations_vol.commit()
        buffer = []
        buffer_pos = 0

    # Batched forwards: collect `batch_size` documents, pad, forward once.
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _flush_doc_batch(docs_token_ids: list[list[int]]) -> int:
        nonlocal buffer_pos, total_positions
        if not docs_token_ids:
            return 0
        max_len = max(len(d) for d in docs_token_ids)
        padded = torch.full((len(docs_token_ids), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros_like(padded, dtype=torch.long)
        for i, d in enumerate(docs_token_ids):
            padded[i, : len(d)] = torch.tensor(d, dtype=torch.long)
            attn[i, : len(d)] = 1
        padded = padded.to(device); attn = attn.to(device)
        with torch.no_grad():
            out = model(input_ids=padded, attention_mask=attn,
                        output_hidden_states=True, use_cache=False)
        resid_batch = out.hidden_states[layer].to("cpu", dtype=torch.bfloat16)
        added = 0
        for i, d in enumerate(docs_token_ids):
            ln = len(d)
            buffer.append(resid_batch[i, :ln, :].contiguous())
            buffer_pos += ln
            added += ln
        total_positions += added
        return added

    pending: list[list[int]] = []
    for doc_idx, doc in enumerate(ds):
        if doc_idx >= n_docs_cap:
            print(f"[cache_corpus] hit n_docs_cap={n_docs_cap}; stopping", flush=True)
            break
        if total_positions >= max_positions:
            break
        content = doc.get("content") or doc.get("text") or ""
        if not content or not isinstance(content, str):
            continue
        ids = tok(content, truncation=True, max_length=max_seq_len, add_special_tokens=False)["input_ids"]
        if len(ids) < 8:
            continue
        pending.append(ids)
        n_docs_used += 1
        if len(pending) >= batch_size:
            _flush_doc_batch(pending)
            pending = []
            if buffer_pos >= chunk_positions:
                flush_chunk()
            if n_docs_used % 200 == 0:
                elapsed = time.time() - t0
                rate = total_positions / max(elapsed, 1)
                eta = (max_positions - total_positions) / max(rate, 1)
                print(
                    f"[cache_corpus] {n_docs_used} docs · {total_positions:,} positions · "
                    f"{rate:,.0f} pos/s · ETA {eta/60:.1f} min",
                    flush=True,
                )
    if pending:
        _flush_doc_batch(pending)
    flush_chunk()
    activations_vol.commit()

    manifest = {
        "model_name": model_name,
        "layer": layer,
        "hook_point": "resid_pre",
        "d_model": d_model,
        "dtype": dtype,
        "n_positions": total_positions,
        "n_docs": n_docs_used,
        "n_chunks": n_chunks,
        "max_seq_len": max_seq_len,
        "chunk_positions": chunk_positions,
        "dataset": dataset,
        "dataset_subdir": dataset_subdir,
        "elapsed_seconds": time.time() - t0,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    activations_vol.commit()
    return manifest


@app.local_entrypoint()
def cache_corpus_entry(
    out_subdir: str = "corpus_l24_resid_pre",
    max_positions: int = 10_000_000,
):
    manifest = cache_corpus.remote(out_subdir=out_subdir, max_positions=max_positions)
    print()
    print(f"[done] manifest: {manifest}")
    # Also write a local manifest copy for inspection
    from pathlib import Path
    import json
    from no_op_circuit.config import RESULTS_DIR
    out_dir = RESULTS_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote local copy: {out_dir / 'manifest.json'}")
