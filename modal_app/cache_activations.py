"""Modal function: cache residual-stream activations for a batch of prompts.

Local callers pass a list of PromptJob dicts. The function:
  1. Loads Qwen2.5-Coder-Instruct (cached on a Modal Volume).
  2. For each prompt, renders the chat template, appends an action suffix,
     tokenizes, runs a single forward pass with residual-stream hooks.
  3. Persists per-prompt activations + action logits + tokens to the
     activations Volume.
  4. Returns a manifest summarizing each saved entry (paths + shapes +
     action-logit table) so the local script can verify the run.

Each PromptJob:
    task_id          : str
    condition        : "buggy" | "fixed"
    variant_name     : str
    messages         : list[{"role": str, "content": str}]
    action_suffix    : str (e.g. "Action: ")
    action_names     : list[str] (the candidate next tokens we score)
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
    DEFAULT_TIMEOUT_S,
    HF_CACHE_DIR,
    activations_vol,
    app,
    hf_cache_vol,
)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_S,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        ACTIVATIONS_DIR: activations_vol,
    },
)
def cache_activations(
    prompt_jobs: list[dict[str, Any]],
    *,
    run_id: str,
    model_name: str = DEFAULT_MODEL,
    last_k_positions: int = 32,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Run forward + residual-stream cache on each prompt; persist outputs.

    Returns a manifest dict.
    """
    # Imports inside the function so they only happen on the Modal worker.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from no_op_circuit.agent.prompt import render_chat_template_safe
    from no_op_circuit.interp.hooks import cache_forward

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    print(f"[cache_activations] loading tokenizer + model ({model_name}, {dtype})…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    t_load = time.time() - t0
    print(f"[cache_activations] model loaded in {t_load:.1f}s on {device}", flush=True)

    def action_first_token_id(prefix: str, name: str) -> tuple[int, int]:
        """Id of the token the model emits FIRST when continuing `prefix` with `name`.

        BPE merges across the prefix/name boundary in some tokenizers (e.g.
        "Action: edit" may tokenize as ["Action", ":", " edit"] rather than
        ["Action", ":", " ", "edit"]). We resolve this by diffing the
        tokenization of `prefix` against `prefix + name`.
        """
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        full_ids = tokenizer.encode(prefix + name, add_special_tokens=False)
        # Walk back through any shared prefix; the first divergent token in
        # full_ids is what we want.
        i = 0
        while i < len(prefix_ids) and i < len(full_ids) and prefix_ids[i] == full_ids[i]:
            i += 1
        if i >= len(full_ids):
            raise ValueError(f"No new token introduced by appending {name!r} to {prefix!r}")
        new_tail = full_ids[i:]
        return new_tail[0], len(new_tail)

    run_dir = Path(ACTIVATIONS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for job in prompt_jobs:
        task_id = job["task_id"]
        condition = job["condition"]
        variant = job["variant_name"]
        messages = job["messages"]
        action_suffix = job.get("action_suffix", "Action: ")
        action_names: list[str] = job["action_names"]

        # Render chat template (with Gemma-safe system→user fold), then
        # append the action-suffix so the next predicted token is the action.
        rendered = render_chat_template_safe(
            tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = rendered + action_suffix

        # Build per-action token ids relative to the action suffix's end. We
        # also probe both leading-space and no-leading-space variants because
        # tokenization is context-sensitive in BPE.
        # Tokenize each action name the way it actually appears right after
        # the rendered prompt — diffing prefix vs prefix+name resolves any
        # BPE merges across the boundary.
        action_token_ids: dict[str, dict[str, Any]] = {}
        for name in action_names:
            tok_id, n_toks = action_first_token_id(full_text, name)
            action_token_ids[name] = {
                "first_token_id": int(tok_id),
                "first_token_str": tokenizer.decode([tok_id]),
                "n_tokens": int(n_toks),
            }

        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        seq_len = int(input_ids.shape[1])

        # We only consume the LAST-position logits; ask the LM to compute
        # only those (saves an O(T × V) buffer that OOMs A10G on 7B models
        # with long prompts). `num_logits_to_keep` / `logits_to_keep` are
        # the API name across recent transformers versions; try both.
        kw_with_slicing = dict(inputs)
        forward_kwargs = {}
        try:
            sig = model.forward.__doc__ or ""
        except Exception:
            sig = ""
        # Most transformers≥4.40 accept num_logits_to_keep; 5.x renamed to logits_to_keep.
        if "logits_to_keep" in sig:
            forward_kwargs["logits_to_keep"] = 1
        else:
            forward_kwargs["num_logits_to_keep"] = 1

        with torch.no_grad(), cache_forward(model, last_k=last_k_positions) as cache:
            try:
                out = model(**kw_with_slicing, **forward_kwargs)
            except TypeError:
                # Fallback: no slicing kwarg supported — pay the full memory cost.
                out = model(**kw_with_slicing)
        logits = out.logits  # (1, T, V) or (1, 1, V) when slicing took effect

        # Next-token distribution at the LAST position predicts the action.
        next_token_logits = logits[0, -1, :].float().cpu()
        next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)
        action_table = {
            name: {
                **info,
                "logit": float(next_token_logits[info["first_token_id"]].item()),
                "logprob": float(next_token_logprobs[info["first_token_id"]].item()),
            }
            for name, info in action_token_ids.items()
        }

        # Save activations to the Volume as a single torch payload per job.
        out_dir = run_dir / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{condition}__{variant}.pt"
        payload = {
            "task_id": task_id,
            "condition": condition,
            "variant": variant,
            "model_name": model_name,
            "dtype": dtype,
            "seq_len": seq_len,
            "last_k": last_k_positions,
            "input_ids_last_k": input_ids[0, -last_k_positions:].cpu(),
            "last_token_text": [
                tokenizer.decode([int(t)]) for t in input_ids[0, -last_k_positions:].tolist()
            ],
            "resid_pre": cache.resid_pre.cpu() if cache.resid_pre is not None else None,
            "resid_post": cache.resid_post.cpu() if cache.resid_post is not None else None,
            "resid_final": cache.resid_final.cpu() if cache.resid_final is not None else None,
            "action_logits": action_table,
            "top_k_next_tokens": _topk_table(next_token_logits, tokenizer, k=10),
        }
        torch.save(payload, out_path)

        entries.append(
            {
                "task_id": task_id,
                "condition": condition,
                "variant": variant,
                "path": str(out_path.relative_to(ACTIVATIONS_DIR)),
                "seq_len": seq_len,
                "resid_post_shape": list(cache.resid_post.shape) if cache.resid_post is not None else None,
                "action_logits": {name: action_table[name]["logit"] for name in action_names},
                "action_argmax": max(action_table.items(), key=lambda kv: kv[1]["logit"])[0],
            }
        )
        print(
            f"[cache_activations] {task_id}/{condition}/{variant}  seq_len={seq_len}  "
            f"argmax={entries[-1]['action_argmax']}",
            flush=True,
        )

    manifest = {
        "run_id": run_id,
        "model_name": model_name,
        "dtype": dtype,
        "last_k_positions": last_k_positions,
        "entries": entries,
        "load_seconds": t_load,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    activations_vol.commit()
    return manifest


def _topk_table(logits, tokenizer, k: int = 10) -> list[dict[str, Any]]:
    import torch

    vals, idxs = torch.topk(logits, k=k)
    return [
        {
            "token_id": int(idx),
            "token": tokenizer.decode([int(idx)]),
            "logit": float(val),
        }
        for val, idx in zip(vals.tolist(), idxs.tolist())
    ]
