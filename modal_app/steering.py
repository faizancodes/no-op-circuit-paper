"""Modal function: dose-response steering at a single (layer, position) site.

Given a direction vector `v` and a sweep of coefficients `alphas`, run the
model on each prompt with `alpha * v` added to the residual stream at
(layer, position) and record the resulting action-logit table. The output is
the canonical dose-response curve for the no-op direction.
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

_STEER_TIMEOUT_S = 60 * 60  # 1 hour; sweeps are cheap, but plenty of headroom


@app.function(
    gpu=DEFAULT_GPU,
    timeout=_STEER_TIMEOUT_S,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        ACTIVATIONS_DIR: activations_vol,
    },
)
def run_steering(
    prompts: list[dict[str, Any]],
    *,
    run_id: str,
    direction: list[float],     # the v_steer vector (length D); buggy → fixed
    alphas: list[float],         # sweep of steering coefficients
    layer: int,
    position: int = -1,           # absolute or negative; we resolve relative to seq len
    hook_point: str = "resid_pre",
    model_name: str = DEFAULT_MODEL,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    """For each prompt × alpha, return the action-logit table.

    Each prompt dict:
        task_id        : str
        condition      : "buggy" | "fixed"
        variant_name   : str
        messages       : list[chat-message-dict]
        action_suffix  : str
        action_names   : list[str]
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from no_op_circuit.agent.prompt import render_chat_template_safe
    from no_op_circuit.interp.hooks import SteeringInjection, steered_forward

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    print(f"[steering] loading {model_name} ({dtype})…", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    print(f"[steering] loaded in {time.time()-t0:.1f}s on {device}", flush=True)

    v = torch.tensor(direction, dtype=torch.float32)
    v_norm = float(v.norm().item())
    print(f"[steering] direction ||v|| = {v_norm:.3f}", flush=True)

    def tokenize(messages, suffix):
        rendered = render_chat_template_safe(tok, messages, tokenize=False, add_generation_prompt=True)
        full = rendered + suffix
        return tok(full, return_tensors="pt").to(device), full

    def action_first_id(prefix: str, name: str) -> int:
        a = tok.encode(prefix, add_special_tokens=False)
        b = tok.encode(prefix + name, add_special_tokens=False)
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return b[i] if i < len(b) else b[-1]

    run_dir = Path(ACTIVATIONS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        action_suffix = prompt.get("action_suffix", "Action: ")
        action_names: list[str] = prompt["action_names"]

        inputs, full_text = tokenize(prompt["messages"], action_suffix)
        seq_len = int(inputs["input_ids"].shape[1])
        abs_position = position if position >= 0 else seq_len + position

        action_ids = {n: action_first_id(full_text, n) for n in action_names}

        for alpha in alphas:
            injection = SteeringInjection(
                layer_idx=layer,
                hook_point=hook_point,
                position=abs_position,
                direction=v,
                alpha=float(alpha),
            )
            with torch.no_grad(), steered_forward(model, [injection]):
                next_logits = model(**inputs).logits[0, -1, :].float().cpu()
            logit_table = {n: float(next_logits[action_ids[n]].item()) for n in action_names}
            margin = logit_table["edit"] - logit_table["noop"]
            argmax_name = max(logit_table.items(), key=lambda kv: kv[1])[0]
            rows.append(
                {
                    "task_id": prompt["task_id"],
                    "condition": prompt["condition"],
                    "variant": prompt["variant_name"],
                    "alpha": float(alpha),
                    "seq_len": seq_len,
                    "abs_position": abs_position,
                    "action_logits": logit_table,
                    "edit_minus_noop": margin,
                    "argmax_action": argmax_name,
                }
            )
        print(
            f"[steering] {prompt['task_id']}/{prompt['condition']}/{prompt['variant_name']}  "
            f"swept {len(alphas)} alphas",
            flush=True,
        )

    manifest = {
        "run_id": run_id,
        "model_name": model_name,
        "layer": layer,
        "position": position,
        "hook_point": hook_point,
        "alphas": list(alphas),
        "direction_norm": v_norm,
        "n_prompts": len(prompts),
        "rows": rows,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    activations_vol.commit()
    return manifest
