"""Per-model pre-flight health check — run before any large GPU sweep.

Validates, for ONE model on its resolved GPU tier, in a single short GPU call:
  * the model loads and the chat template renders;
  * the five action names are single-token in the scored context (or reports
    which are not — e.g. DeepSeek; pass ``--action-vocab`` to test a clean set);
  * a forward produces sane action logits (argmax + edit-minus-abstain margin);
  * the residual-stream **cache / patch / steer hooks all run on this
    architecture** — the important check for Gemma-2, whose architecture differs
    from the Qwen/Llama/DeepSeek models the hooks were written against.

It runs load + a handful of forwards on a single toy pair, so it is cheap
(~one cold start). Use it to de-risk every new model before Phase 1.

    modal run -m modal_app.smoke_model --model Qwen/Qwen2.5-Coder-32B-Instruct
    modal run -m modal_app.smoke_model --model google/gemma-2-27b-it
    modal run -m modal_app.smoke_model --model deepseek-ai/deepseek-coder-6.7b-instruct \
        --action-vocab view,find,test,edit,done
"""

from __future__ import annotations

import json
from typing import Any

from .common import (
    DEFAULT_GPU,
    DEFAULT_MODEL,
    HF_CACHE_DIR,
    app,
    hf_cache_vol,
    pick_tier,
    register_tiers,
    resolve_gpu,
)

from no_op_circuit.agent import ACTION_NAMES, build_prompt
from no_op_circuit.dataset import VARIANTS, load_task


@app.function(gpu=DEFAULT_GPU, timeout=20 * 60, volumes={HF_CACHE_DIR: hf_cache_vol})
def model_health(
    buggy_messages: list[dict],
    fixed_messages: list[dict],
    action_suffix: str,
    action_names: list[str],
    *,
    model_name: str = DEFAULT_MODEL,
    peak_frac: float = 0.88,
    edit: str = "edit",
    abstain: str = "noop",
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    import os
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from no_op_circuit.agent.prompt import render_chat_template_safe
    from no_op_circuit.interp.hooks import (
        ResidualPatch,
        SteeringInjection,
        cache_forward,
        patched_forward,
        steered_forward,
    )

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    health: dict[str, Any] = {"model": model_name, "ok": False}

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=HF_CACHE_DIR, torch_dtype=td, device_map=device
    )
    model.eval()
    health["load_seconds"] = round(time.time() - t0, 1)

    layers = model.model.layers if hasattr(model, "model") else model.transformer.h
    n_layers = len(layers)
    peak_layer = max(0, min(n_layers - 1, round(peak_frac * n_layers)))
    health.update(
        n_layers=n_layers,
        hidden_size=int(getattr(model.config, "hidden_size", -1)),
        peak_layer_guess=peak_layer,
        peak_rel_depth=round(peak_layer / n_layers, 3),
    )

    def first_new(prefix: str, name: str):
        p = tok.encode(prefix, add_special_tokens=False)
        f = tok.encode(prefix + name, add_special_tokens=False)
        i = 0
        while i < len(p) and i < len(f) and p[i] == f[i]:
            i += 1
        return f[i:], len(f) - i

    def tokenize(messages):
        rendered = render_chat_template_safe(
            tok, messages, tokenize=False, add_generation_prompt=True
        )
        full = rendered + action_suffix
        return tok(full, return_tensors="pt").to(device), full

    def fwd_last(inputs):
        with torch.no_grad():
            try:
                return model(**inputs, num_logits_to_keep=1).logits[0, -1, :].float()
            except TypeError:
                return model(**inputs).logits[0, -1, :].float()

    buggy_inputs, buggy_text = tokenize(buggy_messages)
    fixed_inputs, _ = tokenize(fixed_messages)
    health["seq_len_buggy"] = int(buggy_inputs["input_ids"].shape[1])

    # Single-token action audit in the exact scored context.
    tok_audit = {}
    for nm in action_names:
        tail, n = first_new(buggy_text, nm)
        tok_audit[nm] = {
            "n_tokens": n,
            "single_token": n == 1,
            "decoded": "".join(tok.decode([t]) for t in tail),
        }
    health["action_tokens"] = tok_audit
    health["all_single_token"] = all(v["single_token"] for v in tok_audit.values())
    action_ids = {nm: first_new(buggy_text, nm)[0][0] for nm in action_names}
    edit_id, abst_id = action_ids[edit], action_ids[abstain]

    # (1) plain forward + action logits
    blog = fwd_last(buggy_inputs)
    logit_tab = {nm: float(blog[action_ids[nm]]) for nm in action_names}
    clean_margin = float(blog[edit_id] - blog[abst_id])
    health.update(
        buggy_action_logits={k: round(v, 3) for k, v in logit_tab.items()},
        buggy_argmax=max(logit_tab, key=lambda k: logit_tab[k]),
        buggy_edit_minus_abstain=round(clean_margin, 3),
    )

    # (2) cache hook — capture fixed residuals at the last K positions
    K = 8
    with torch.no_grad(), cache_forward(model, last_k=K) as fcache:
        _ = model(**fixed_inputs)
    fresid = fcache.resid_pre
    health["cache_ok"] = fresid is not None
    health["resid_pre_shape"] = list(fresid.shape) if fresid is not None else None
    assert fresid is not None, "cache_forward returned no resid_pre — hook incompatible with this arch"

    # (3) patch hook — substitute fixed resid at (peak_layer, pos -1) into buggy fwd
    buggy_pos = int(buggy_inputs["input_ids"].shape[1]) - 1
    patch = [ResidualPatch(
        layer_idx=peak_layer, hook_point="resid_pre",
        position=buggy_pos, value=fresid[peak_layer, 0, -1, :],
    )]
    with torch.no_grad(), patched_forward(model, patch):
        plog = model(**buggy_inputs).logits[0, -1, :].float()
    health["patch_ok"] = True
    health["patched_edit_minus_abstain"] = round(float(plog[edit_id] - plog[abst_id]), 3)

    # (4) steer hook — add the (fixed - buggy) contrast once at alpha=1
    with torch.no_grad(), cache_forward(model, last_k=K) as bcache:
        _ = model(**buggy_inputs)
    bresid = bcache.resid_pre
    assert bresid is not None
    direction = (fresid[peak_layer, 0, -1, :].float()
                 - bresid[peak_layer, 0, -1, :].float())
    inj = [SteeringInjection(
        layer_idx=peak_layer, hook_point="resid_pre",
        position=buggy_pos, direction=direction, alpha=1.0,
    )]
    with torch.no_grad(), steered_forward(model, inj):
        slog = model(**buggy_inputs).logits[0, -1, :].float()
    health["steer_ok"] = True
    health["steered_edit_minus_abstain"] = round(float(slog[edit_id] - slog[abst_id]), 3)

    if torch.cuda.is_available():
        health["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    health["ok"] = bool(health["cache_ok"] and health["patch_ok"] and health["steer_ok"])
    return health


# Pre-registered GPU-tier variants; main() dispatches by model size.
HEALTH_TIERS = {
    "A10G": model_health,
    **register_tiers(
        model_health.get_raw_f(), "model_health",
        volumes={HF_CACHE_DIR: hf_cache_vol}, base_timeout=20 * 60,
    ),
}


@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    task_id: str = "parser_empty_input",
    variant: str = "code_tests",
    action_vocab: str = "",     # e.g. "view,find,test,edit,done" for DeepSeek
    gpu: str = "auto",
    peak_frac: float = 0.88,    # relative-depth guess for the patch/steer cell
):
    names = (
        [a.strip() for a in action_vocab.split(",") if a.strip()]
        if action_vocab else list(ACTION_NAMES)
    )
    if len(names) < 4:
        raise SystemExit("need >=4 action names (… edit … abstain-last)")
    edit_name = names[3] if len(names) >= 4 else "edit"
    abstain_name = names[-1]

    task = load_task(task_id)
    var = VARIANTS[variant]
    b = build_prompt(task, "buggy", var, action_names=names)
    f = build_prompt(task, "fixed", var, action_names=names)

    fn = pick_tier(HEALTH_TIERS, model, gpu)
    tier = resolve_gpu(model, gpu) or "A10G"
    print(f"[smoke_model] {model}  gpu tier={tier}  actions={names}", flush=True)

    health = fn.remote(
        b.messages, f.messages, b.action_suffix, names,
        model_name=model, peak_frac=peak_frac,
        edit=edit_name, abstain=abstain_name,
    )
    print(json.dumps(health, indent=2))

    bad_tok = [n for n, v in health.get("action_tokens", {}).items() if not v["single_token"]]
    failed = [k for k in ("cache_ok", "patch_ok", "steer_ok") if not health.get(k)]
    print(
        f"\n[smoke_model] OK={health.get('ok')}  load={health.get('load_seconds')}s  "
        f"n_layers={health.get('n_layers')}  peak~L{health.get('peak_layer_guess')}"
        f"(rel {health.get('peak_rel_depth')})  gpu_mem={health.get('gpu_mem_gb')}GB  "
        f"argmax={health.get('buggy_argmax')}",
        flush=True,
    )
    if bad_tok:
        print(f"[smoke_model] multi-token actions: {bad_tok} -> pass --action-vocab with a single-token set")
    if failed:
        print(f"[smoke_model] FAILED hooks: {failed} -> inspect hooks.py for this architecture")
    return health
