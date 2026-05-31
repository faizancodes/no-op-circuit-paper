"""Modal function: count how often each SAE feature fires at (L24, pos -1)
across the 296 paired buggy/fixed prompts.

Purpose: build a tighter random-feature ablation baseline for §5.2. The
existing random-8 baseline drew from the full 4096-feature pool, but TopK
sparsity (k=16) means random draws almost never overlap the firing set on
any given prompt, producing ~0 effect by construction. This module computes
the empirical firing-count distribution so we can resample baselines from
{features that fire on >=5 prompts} \\ OMP_top128.

Output: results/sae/firing_features.json
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
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

_TIMEOUT_S = 60 * 30


@app.function(
    gpu=DEFAULT_GPU,
    timeout=_TIMEOUT_S,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        ACTIVATIONS_DIR: activations_vol,
    },
)
def compute_firing_features(
    prompts: list[dict[str, Any]],
    *,
    sae_path: str = "sae/qwen_l24_resid_pre_TASK_d4096_k16.pt",
    model_name: str = DEFAULT_MODEL,
    layer: int = 24,
    position: int = -1,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from no_op_circuit.agent.prompt import render_chat_template_safe
    from no_op_circuit.interp.sae import SAEConfig, TopKSAE

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]

    t0 = time.time()
    print(f"[firing] loading {model_name}…", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=HF_CACHE_DIR, torch_dtype=torch_dtype, device_map=device,
    )
    model.eval()
    blob = torch.load(Path(ACTIVATIONS_DIR) / sae_path, map_location="cpu", weights_only=False)
    cfg = SAEConfig(**blob["config"])
    sae = TopKSAE(cfg).to(device, dtype=torch.float32)
    sae.load_state_dict(blob["state_dict"])
    sae.eval()
    print(f"[firing] model+SAE ready in {time.time()-t0:.1f}s (d_sae={cfg.d_sae}, k={cfg.k})",
          flush=True)

    layers = model.model.layers
    assert 0 <= layer < len(layers)

    captured: dict[str, Any] = {"clean_resid": None}
    def capture_hook(_mod, args, kwargs=None):
        if args and isinstance(args[0], torch.Tensor):
            captured["clean_resid"] = args[0][:, position, :].detach().clone()
        return None

    firing_counts: Counter[int] = Counter()
    n_prompts = 0
    for p_idx, prompt in enumerate(prompts):
        rendered = render_chat_template_safe(
            tok, prompt["messages"], tokenize=False, add_generation_prompt=True
        )
        full_text = rendered + prompt.get("action_suffix", "Action: ")
        inputs = tok(full_text, return_tensors="pt").to(device)

        handle = layers[layer].register_forward_pre_hook(capture_hook, with_kwargs=True)
        try:
            with torch.no_grad():
                _ = model(**inputs).logits
        finally:
            handle.remove()
        clean_resid = captured["clean_resid"]
        assert clean_resid is not None

        x = clean_resid.to(dtype=torch.float32)
        with torch.no_grad():
            a = sae.encode(x)              # (B, d_sae), TopK-sparse
        fired = (a[0].abs() > 0).nonzero(as_tuple=False).squeeze(-1).tolist()
        for f in fired:
            firing_counts[int(f)] += 1
        n_prompts += 1
        if (p_idx + 1) % 50 == 0:
            print(f"[firing]   {p_idx+1}/{len(prompts)} prompts; "
                  f"{len(firing_counts)} distinct features have fired", flush=True)

    n_ever = len(firing_counts)
    n_5plus = sum(1 for c in firing_counts.values() if c >= 5)
    print(f"[firing] done: {n_prompts} prompts, "
          f"{n_ever} ever-fired features, {n_5plus} with count >=5", flush=True)

    out = {
        "n_prompts": n_prompts,
        "d_sae": cfg.d_sae,
        "k": cfg.k,
        "firing_counts": {str(k): int(v) for k, v in firing_counts.items()},
        "n_ever_fired": n_ever,
        "n_fired_5plus": n_5plus,
        "sae_path": sae_path,
        "model_name": model_name,
        "layer": layer,
        "position": position,
        "elapsed_seconds": time.time() - t0,
    }
    out_path = Path(ACTIVATIONS_DIR) / "sae" / "firing_features.json"
    out_path.write_text(json.dumps(out, indent=2))
    activations_vol.commit()
    print(f"[firing] wrote {out_path}", flush=True)
    return out


@app.local_entrypoint()
def compute_firing_entry(
    sae_path: str = "sae/qwen_l24_resid_pre_TASK_d4096_k16.pt",
    model: str = DEFAULT_MODEL,
    layer: int = 24,
    out: str = "results/sae/firing_features.json",
):
    from no_op_circuit.dataset import iter_tasks, VARIANTS
    from no_op_circuit.agent import ACTION_NAMES, build_prompt
    from no_op_circuit.config import TASKS_DIR

    prompts: list[dict] = []
    for tasks_dir in (TASKS_DIR, Path("data/real_tasks")):
        for t in iter_tasks(tasks_dir=tasks_dir):
            for cond in ("buggy", "fixed"):
                b = build_prompt(t, cond, VARIANTS["code_tests"])
                prompts.append({
                    "task_id": t.task_id,
                    "condition": cond,
                    "variant_name": "code_tests",
                    "messages": b.messages,
                    "action_suffix": b.action_suffix,
                    "action_names": ACTION_NAMES,
                })
    print(f"[entry] {len(prompts)} prompts  model={model}  layer={layer}  sae={sae_path}")
    result = compute_firing_features.remote(
        prompts, sae_path=sae_path, model_name=model, layer=layer,
    )
    print(f"[done] n_ever_fired={result['n_ever_fired']}, "
          f"n_fired_5plus={result['n_fired_5plus']}")

    # Pull file locally
    import subprocess
    local = Path(out)
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["modal", "volume", "get", "noop-activations",
             "sae/firing_features.json", str(local), "--force"],
            check=True,
        )
        print(f"[entry] downloaded to {local}")
    except Exception as e:
        print(f"[entry] WARN download failed: {e}")
