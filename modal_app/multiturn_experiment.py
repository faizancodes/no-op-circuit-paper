"""Temporally-separated transcript experiment (App. G.15).

Tests whether the Qwen L24/pos-1 pass/fail direction carries the verdict
*forward* to a decision point where the transcript is positionally distant —
separated from the decision by condition-neutral intervening agent turns.

For each real task and each of two multi-turn variants
(`code_tests_stale_multiturn`, `code_multiturn_notranscript`) we run a single
forward pass, capture `resid_pre[24, pos-1]`, project onto the frozen toy-
trained `v_noop`, and read the action logits. We also record two regex flags
per prompt (computed locally from the prompt text):
  - failed_in_last_user : "FAILED" in the decision-point local context only
  - failed_in_full      : "FAILED" anywhere in the full scrollback

Only a small per-prompt JSON is returned (projection + logits + flags), so
no large activation tensors are downloaded — this avoids competing for local
bandwidth with the HF cache upload.

Run:
    modal run -m modal_app.multiturn_experiment
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .common import (
    DEFAULT_GPU,
    DEFAULT_TIMEOUT_S,
    HF_CACHE_DIR,
    activations_vol,
    ACTIVATIONS_DIR,
    app,
    hf_cache_vol,
)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_S,
    volumes={HF_CACHE_DIR: hf_cache_vol, ACTIVATIONS_DIR: activations_vol},
)
def score_multiturn(
    prompt_jobs: list[dict[str, Any]],
    *,
    v_noop: list[float],
    layer: int,
    model_name: str,
    dtype: str = "bfloat16",
) -> list[dict[str, Any]]:
    """Forward each prompt, project resid_pre[layer, -1] onto v_noop."""
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from no_op_circuit.agent.prompt import render_chat_template_safe
    from no_op_circuit.interp.hooks import cache_forward

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[multiturn] loading {model_name} ({dtype})…", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=HF_CACHE_DIR, torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()

    v = torch.tensor(v_noop, dtype=torch.float32)
    v = v / v.norm()  # unit-normalise (AUC is scale-invariant; keeps scores tidy)

    def action_first_token_id(prefix: str, name: str) -> int:
        prefix_ids = tok.encode(prefix, add_special_tokens=False)
        full_ids = tok.encode(prefix + name, add_special_tokens=False)
        i = 0
        while i < len(prefix_ids) and i < len(full_ids) and prefix_ids[i] == full_ids[i]:
            i += 1
        return full_ids[i]

    out_rows: list[dict[str, Any]] = []
    for k, job in enumerate(prompt_jobs):
        messages = job["messages"]
        action_names = job["action_names"]
        rendered = render_chat_template_safe(
            tok, messages, tokenize=False, add_generation_prompt=True
        )
        full_text = rendered + job.get("action_suffix", "Action: ")
        action_ids = {n: action_first_token_id(full_text, n) for n in action_names}

        inputs = tok(full_text, return_tensors="pt").to(device)
        with torch.no_grad(), cache_forward(model, last_k=1,
                                            hook_points=("resid_pre",)) as cache:
            out = model(**inputs)
        logits = out.logits[0, -1, :].float().cpu()
        # resid_pre shape (L, B, T=1, D) → layer, batch0, last pos
        resid = cache.resid_pre[layer, 0, -1, :].float().cpu()
        proj = float(torch.dot(resid, v))

        action_logits = {n: float(logits[i].item()) for n, i in action_ids.items()}
        argmax = max(action_logits.items(), key=lambda kv: kv[1])[0]

        out_rows.append({
            "task_id": job["task_id"],
            "condition": job["condition"],
            "variant": job["variant_name"],
            "projection": proj,
            "action_logits": action_logits,
            "argmax_action": argmax,
            "failed_in_last_user": job["failed_in_last_user"],
            "failed_in_full": job["failed_in_full"],
        })
        if (k + 1) % 200 == 0:
            print(f"[multiturn] {k+1}/{len(prompt_jobs)}", flush=True)

    return out_rows


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    v_noop_path: str = "results/steer-20260516T021522Z/v_noop.pt",
    tasks_root: str = "data/real_tasks",
    variants: str = "code_tests_stale_multiturn,code_multiturn_notranscript",
    out_path: str = "",
):
    from pathlib import Path
    import torch

    from no_op_circuit.agent import ACTION_NAMES, build_prompt
    from no_op_circuit.dataset import VARIANTS, iter_tasks

    blob = torch.load(v_noop_path, map_location="cpu", weights_only=False)
    direction = blob["direction"].tolist()
    layer = int(blob["layer"])
    pos = int(blob["position"])
    assert pos == -1, f"expected pos -1, got {pos}"
    print(f"[multiturn] v_noop: L{layer}/pos{pos} |v|={blob['norm']:.3f}")

    tasks = list(iter_tasks(Path(tasks_root)))
    variant_names = [v.strip() for v in variants.split(",") if v.strip()]
    print(f"[multiturn] {len(tasks)} tasks × {len(variant_names)} variants")

    jobs: list[dict[str, Any]] = []
    for vname in variant_names:
        variant = VARIANTS[vname]
        for task in tasks:
            for cond in ("buggy", "fixed"):
                b = build_prompt(task, cond, variant, list(ACTION_NAMES))
                last_user = b.messages[-1]["content"]
                full = "\n".join(m["content"] for m in b.messages)
                jobs.append({
                    "task_id": b.task_id,
                    "condition": cond,
                    "variant_name": b.variant_name,
                    "messages": b.messages,
                    "action_suffix": b.action_suffix,
                    "action_names": list(ACTION_NAMES),
                    "failed_in_last_user": "FAILED" in last_user,
                    "failed_in_full": "FAILED" in full,
                })
    print(f"[multiturn] {len(jobs)} prompts → GPU")

    rows = score_multiturn.remote(
        jobs, v_noop=direction, layer=layer, model_name=model,
    )

    if not out_path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = f"results/monitor_real/multiturn_experiment_{ts}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({
        "config": {"model": model, "v_noop_path": v_noop_path, "layer": layer,
                   "pos": pos, "tasks_root": tasks_root, "variants": variant_names},
        "rows": rows,
    }, indent=2))
    print(f"[multiturn] wrote {out_path}  ({len(rows)} rows)")
