"""Modal function: cache L24 resid_pre over our task-distribution prompts.

The generic-Python SAE trained in Phase 2 reconstructs v_noop poorly (cos
0.21). v_noop lives at L24/pos -1 *after* our chat-template-rendered Action:
prompt, which is a different distribution from middle-of-Python-file
residuals. This caches resid_pre at *all positions* of our 49 toy + 99 real
tasks crossed with all 5 variants × buggy/fixed, giving an SAE training set
that by construction spans v_noop's neighbourhood.
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

_CACHE_TIMEOUT_S = 60 * 30


@app.function(
    gpu=DEFAULT_GPU,
    timeout=_CACHE_TIMEOUT_S,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        ACTIVATIONS_DIR: activations_vol,
    },
)
def cache_task_corpus(
    prompt_texts: list[str],
    *,
    out_subdir: str = "task_l24_resid_pre",
    model_name: str = DEFAULT_MODEL,
    layer: int = 24,
    max_seq_len: int = 4096,
    batch_size: int = 4,
    chunk_positions: int = 200_000,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

    t0 = time.time()
    print(f"[cache_task] loading {model_name} ({dtype})…", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=HF_CACHE_DIR, torch_dtype=torch_dtype, device_map=device,
    )
    model.eval()
    n_layers_total = len(model.model.layers)
    d_model = model.config.hidden_size
    print(f"[cache_task] loaded in {time.time()-t0:.1f}s; n_layers={n_layers_total}, d={d_model}", flush=True)
    assert 0 <= layer <= n_layers_total
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    out_dir = Path(ACTIVATIONS_DIR) / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("chunk_*.pt"):
        old.unlink()

    buffer: list[torch.Tensor] = []
    buffer_pos = 0
    total_positions = 0
    n_chunks = 0
    n_prompts_used = 0
    n_truncated = 0

    def flush_chunk():
        nonlocal n_chunks, buffer, buffer_pos
        if not buffer:
            return
        chunk = torch.cat(buffer, dim=0).contiguous().to(torch.bfloat16)
        path = out_dir / f"chunk_{n_chunks:04d}.pt"
        torch.save(chunk, path)
        n_chunks += 1
        print(
            f"[cache_task]   wrote {path.name} ({chunk.shape[0]:,} positions, "
            f"total {total_positions:,})",
            flush=True,
        )
        if n_chunks % 3 == 0:
            activations_vol.commit()
        buffer = []
        buffer_pos = 0

    # Tokenise everything up front so we can sort by length and pack batches
    # tightly (less padding waste).
    print(f"[cache_task] tokenising {len(prompt_texts)} prompts…", flush=True)
    tokenised: list[list[int]] = []
    for txt in prompt_texts:
        ids = tok(txt, truncation=True, max_length=max_seq_len, add_special_tokens=False)["input_ids"]
        if len(ids) > max_seq_len * 0.95:
            n_truncated += 1
        if len(ids) >= 8:
            tokenised.append(ids)
    # Sort by length so batches have minimal padding waste
    tokenised.sort(key=len)
    print(f"[cache_task] kept {len(tokenised)} prompts (truncated {n_truncated} at {max_seq_len})", flush=True)

    for batch_start in range(0, len(tokenised), batch_size):
        docs = tokenised[batch_start : batch_start + batch_size]
        max_len = max(len(d) for d in docs)
        padded = torch.full((len(docs), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros_like(padded, dtype=torch.long)
        for i, d in enumerate(docs):
            padded[i, : len(d)] = torch.tensor(d, dtype=torch.long)
            attn[i, : len(d)] = 1
        padded = padded.to(device); attn = attn.to(device)
        with torch.no_grad():
            out = model(input_ids=padded, attention_mask=attn,
                        output_hidden_states=True, use_cache=False)
        resid_batch = out.hidden_states[layer].to("cpu", dtype=torch.bfloat16)
        added = 0
        for i, d in enumerate(docs):
            ln = len(d)
            buffer.append(resid_batch[i, :ln, :].contiguous())
            buffer_pos += ln
            added += ln
        total_positions += added
        n_prompts_used += len(docs)
        if buffer_pos >= chunk_positions:
            flush_chunk()
        if n_prompts_used % 50 == 0 or n_prompts_used == len(tokenised):
            elapsed = time.time() - t0
            rate = total_positions / max(elapsed, 1)
            print(
                f"[cache_task] {n_prompts_used}/{len(tokenised)} prompts · "
                f"{total_positions:,} positions · {rate:,.0f} pos/s",
                flush=True,
            )

    flush_chunk()
    activations_vol.commit()

    manifest = {
        "model_name": model_name,
        "layer": layer,
        "hook_point": "resid_pre",
        "d_model": d_model,
        "dtype": dtype,
        "n_positions": total_positions,
        "n_prompts": n_prompts_used,
        "n_chunks": n_chunks,
        "max_seq_len": max_seq_len,
        "chunk_positions": chunk_positions,
        "n_truncated_at_max_len": n_truncated,
        "elapsed_seconds": time.time() - t0,
        "source": "task_corpus (toy + real × all variants × {buggy,fixed})",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    activations_vol.commit()
    return manifest


@app.local_entrypoint()
def cache_task_corpus_entry(
    out_subdir: str = "task_l24_resid_pre",
    max_seq_len: int = 4096,
    model: str = DEFAULT_MODEL,
    layer: int = 24,
    batch_size: int = 4,
):
    """Build all (task × condition × variant) prompts locally and ship them."""
    from pathlib import Path
    from transformers import AutoTokenizer

    from no_op_circuit.agent import build_prompt
    from no_op_circuit.agent.prompt import render_chat_template_safe
    from no_op_circuit.dataset import iter_tasks, VARIANTS
    from no_op_circuit.config import TASKS_DIR

    # We use a local tokenizer just to render the chat template; the Modal side
    # will re-tokenise (cheap) so we ship plain strings.
    tok = AutoTokenizer.from_pretrained(model)
    prompts: list[str] = []
    skipped: dict[str, int] = {"stale_flaky_no_transcript": 0, "render_error": 0}

    for tasks_dir in (TASKS_DIR, Path("data/real_tasks")):
        for t in iter_tasks(tasks_dir=tasks_dir):
            for cond in ("buggy", "fixed"):
                for variant in VARIANTS.values():
                    # stale_flaky requires fixed_flaky_test_output; skip if missing.
                    if variant.pin_tests_to == "fixed_flaky" and t.fixed_flaky_test_output is None:
                        skipped["stale_flaky_no_transcript"] += 1
                        continue
                    try:
                        b = build_prompt(t, cond, variant)
                    except Exception:
                        skipped["render_error"] += 1
                        continue
                    text = render_chat_template_safe(
                        tok, b.messages, tokenize=False, add_generation_prompt=True
                    ) + b.action_suffix
                    prompts.append(text)

    print(f"[entry] built {len(prompts)} prompts; skipped {skipped}")
    print(f"[entry] model={model}  layer={layer}  out_subdir={out_subdir}  batch_size={batch_size}")
    manifest = cache_task_corpus.remote(
        prompts, out_subdir=out_subdir, max_seq_len=max_seq_len,
        model_name=model, layer=layer, batch_size=batch_size,
    )
    print()
    print(f"[done] manifest: {manifest}")

    # Local copy for inspection
    import json
    from no_op_circuit.config import RESULTS_DIR
    out_dir = RESULTS_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote local copy: {out_dir / 'manifest.json'}")
