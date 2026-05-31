"""Modal function: per-feature ablation study at L24/pos −1.

For each tracked SAE feature (and the union of all), we modify the
residual at the action position before layer 24's input by:

    a       = sae.encode(x)
    error   = x − sae.decode(a)             # SAE reconstruction residual
    a'      = a; a'[feature_idx] = 0
    x'      = sae.decode(a') + error
    (then replace the residual at (L24, pos -1) with x')

We re-run all 49 toy `code_tests` prompts × 2 conditions (98) and all 99
real-task prompts × 2 conditions (198), giving 296 prompts per ablation
condition. We also evaluate a `sae_recon` sanity baseline where we
encode-then-decode WITHOUT zeroing any feature — this should match
the clean forward to fp precision.
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

_ABLATE_TIMEOUT_S = 60 * 60 * 2


@app.function(
    gpu=DEFAULT_GPU,
    timeout=_ABLATE_TIMEOUT_S,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        ACTIVATIONS_DIR: activations_vol,
    },
)
def ablate_sae_features(
    prompts: list[dict[str, Any]],
    feature_sets: list[list[int]],
    *,
    sae_path: str = "sae/qwen_l24_resid_pre_d24576_k32.pt",
    model_name: str = DEFAULT_MODEL,
    layer: int = 24,
    position: int = -1,
    dtype: str = "bfloat16",
    run_id: str = "ablate-default",
    feature_set_labels: list[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    import random

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from no_op_circuit.agent.prompt import render_chat_template_safe
    from no_op_circuit.interp.hooks import ResidualPatch, patched_forward
    from no_op_circuit.interp.sae import SAEConfig, TopKSAE

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    # Mitigate CUDA fragmentation for tight 7B-model fits on A10G — the
    # allocator otherwise reserves chunks it can't reuse for the next varying
    # prompt length and OOMs midway through. Must be set BEFORE first CUDA op.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[ablate] seeded all RNGs with seed={seed}", flush=True)

    t0 = time.time()
    print(f"[ablate] loading {model_name}…", flush=True)
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
    print(f"[ablate] model+SAE ready in {time.time()-t0:.1f}s", flush=True)

    layers = model.model.layers
    n_layers = len(layers)
    assert 0 <= layer < n_layers

    # Hook helper: capture clean resid at (L24, pos -1) on a normal forward
    captured = {"clean_resid": None}
    def capture_hook(_mod, args, kwargs=None):
        if args and isinstance(args[0], torch.Tensor):
            captured["clean_resid"] = args[0][:, -1, :].detach().clone()  # (B, D)
        return None

    def action_first_token_id(prefix: str, name: str) -> int:
        a = tok.encode(prefix, add_special_tokens=False)
        b = tok.encode(prefix + name, add_special_tokens=False)
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return b[i] if i < len(b) else b[-1]

    def build_replacement(clean_resid: torch.Tensor, feature_set: list[int] | None) -> torch.Tensor:
        """Return the position vector to use after SAE reconstruction (and
        optional feature ablation). `feature_set=None` ⇒ sae_recon (no
        ablation); else zero those feature indices in `a`."""
        x = clean_resid.to(dtype=torch.float32)
        with torch.no_grad():
            a = sae.encode(x)
            error = x - sae.decode(a)
            if feature_set is not None and len(feature_set) > 0:
                a = a.clone()
                idx = torch.tensor(feature_set, device=a.device, dtype=torch.long)
                a[:, idx] = 0.0
            x_new = sae.decode(a) + error
        return x_new.to(dtype=clean_resid.dtype)

    results: list[dict[str, Any]] = []
    total_prompts = len(prompts)
    n_conditions = 1 + 1 + len(feature_sets)  # clean + sae_recon + each set
    print(f"[ablate] {total_prompts} prompts × {n_conditions} conditions = {total_prompts*n_conditions} forwards", flush=True)

    for p_idx, prompt in enumerate(prompts):
        rendered = render_chat_template_safe(
            tok, prompt["messages"], tokenize=False, add_generation_prompt=True
        )
        full_text = rendered + prompt.get("action_suffix", "Action: ")
        inputs = tok(full_text, return_tensors="pt").to(device)
        action_names = prompt["action_names"]
        # resolve action ids
        action_ids = {
            n: action_first_token_id(full_text, n) for n in action_names
        }
        edit_id = action_ids["edit"]; noop_id = action_ids["noop"]

        # --- 1. clean forward, with capture hook ---
        handle = layers[layer].register_forward_pre_hook(capture_hook, with_kwargs=True)
        try:
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
        finally:
            handle.remove()
        clean_margin = float((logits[edit_id] - logits[noop_id]).item())
        clean_action_logits = {n: float(logits[action_ids[n]].item()) for n in action_names}
        clean_resid = captured["clean_resid"]
        assert clean_resid is not None

        # --- 2. sae_recon (sanity) and per-feature-set ablations ---
        per_condition_margin: dict[str, float] = {"clean": clean_margin}
        per_condition_action_logits: dict[str, dict[str, float]] = {"clean": clean_action_logits}
        abs_pos = inputs["input_ids"].shape[1] + position  # position is signed
        if feature_set_labels is not None:
            assert len(feature_set_labels) == len(feature_sets), \
                f"label count {len(feature_set_labels)} != feature_sets {len(feature_sets)}"
            labeled = list(zip(feature_set_labels, feature_sets))
        else:
            labeled = [
                (f"ablate_{'_'.join(str(i) for i in fs)}" if len(fs) <= 5 else f"ablate_set{i}",
                 fs)
                for i, fs in enumerate(feature_sets)
            ]
        for label, fs in [("sae_recon", None)] + labeled:
            replacement = build_replacement(clean_resid, fs)
            patch = ResidualPatch(layer_idx=layer, hook_point="resid_pre",
                                  position=abs_pos, value=replacement[0])
            with torch.no_grad(), patched_forward(model, [patch]):
                out = model(**inputs).logits[0, -1, :]
            margin = float((out[edit_id] - out[noop_id]).item())
            per_condition_margin[label] = margin
            per_condition_action_logits[label] = {
                n: float(out[action_ids[n]].item()) for n in action_names
            }

        results.append({
            "task_id": prompt["task_id"],
            "condition": prompt["condition"],
            "variant": prompt["variant_name"],
            "margins": per_condition_margin,
            "action_logits": per_condition_action_logits,
        })
        # Free intermediates between prompts so cached fragments don't accumulate.
        del inputs, logits, clean_resid, replacement, patch, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (p_idx + 1) % 25 == 0:
            print(f"[ablate]   {p_idx+1}/{total_prompts} prompts", flush=True)
            activations_vol.commit()

    out_dir = Path(ACTIVATIONS_DIR) / "sae" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "sae_path": sae_path,
        "layer": layer,
        "position": position,
        "feature_sets": feature_sets,
        "n_prompts": total_prompts,
        "rows": results,
    }
    out_path = out_dir / "ablation_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    activations_vol.commit()
    print(f"[ablate] wrote {out_path}", flush=True)
    return payload


@app.local_entrypoint()
def ablate_entry(
    features_json: str = "results/sae/v_noop_features_DISTRIBUTED.json",
    sae_path: str = "sae/qwen_l24_resid_pre_TASK_d4096_k16.pt",
    run_id: str = "ablate-distributed",
    model: str = DEFAULT_MODEL,
    layer: int = 24,
    max_prompt_tokens: int = 4096,
    cuts: str = "8,32,128",
    seed: int = 0,
):
    """Build the prompt list from toy + real caches and ship to Modal.

    With the DISTRIBUTED feature file, we ablate ranked-subset cumulants of
    the OMP top-128: top-8, top-32, top-128 (full set). This tests whether
    the distributed signal survives partial ablation.
    """
    from no_op_circuit.dataset import iter_tasks, VARIANTS
    from no_op_circuit.agent import ACTION_NAMES, build_prompt
    from no_op_circuit.config import TASKS_DIR

    feats_blob = json.loads(Path(features_json).read_text())
    top_features = feats_blob["top_features"]
    # Already sorted by |v_contribution| descending in the DISTRIBUTED builder.
    feature_indices = [int(f["feature_idx"]) for f in top_features]

    is_distributed = feats_blob.get("method") == "OMP"
    if is_distributed:
        # Parse cumulative cut points: e.g. "1,2,3,4,5,6,7,8,16,32,128".
        try:
            cut_list = sorted(set(int(c.strip()) for c in cuts.split(",") if c.strip()))
        except ValueError as e:
            raise SystemExit(f"--cuts must be comma-separated integers; got {cuts!r}: {e}")
        # Clamp to available features; warn if any cut exceeds the OMP set size.
        cut_list = [c for c in cut_list if 1 <= c <= len(feature_indices)]
        if not cut_list:
            raise SystemExit(f"no valid cuts in {cuts!r} (have {len(feature_indices)} OMP features)")
        feature_sets = [feature_indices[:c] for c in cut_list]
        feature_set_labels = [f"ablate_omp_top{c}" for c in cut_list]
        print(f"[entry] cumulative cuts: {cut_list}")
    else:
        # Legacy: each individual feature + the union
        feature_sets = [[i] for i in feature_indices] + [feature_indices]
        feature_set_labels = None  # function will auto-label

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

    # Pre-filter by token length. CodeGemma-7B on A10G can OOM on full-4096
    # prompts when lm_head allocates 1.7 GB on top of model + activations.
    # We drop BOTH sides of a task if either exceeds the cap, so paired
    # buggy/fixed analysis remains valid on the kept tasks.
    if max_prompt_tokens < 4096:
        from transformers import AutoTokenizer
        from no_op_circuit.agent.prompt import render_chat_template_safe
        tok = AutoTokenizer.from_pretrained(model)
        long_task_ids: set[str] = set()
        for p in prompts:
            text = render_chat_template_safe(
                tok, p["messages"], tokenize=False, add_generation_prompt=True
            ) + p.get("action_suffix", "Action: ")
            n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
            if n_tok > max_prompt_tokens:
                long_task_ids.add(p["task_id"])
        before = len(prompts)
        prompts = [p for p in prompts if p["task_id"] not in long_task_ids]
        print(f"[entry] dropped {len(long_task_ids)} tasks "
              f"({before - len(prompts)} prompts) for exceeding "
              f"max_prompt_tokens={max_prompt_tokens}")

    print(f"[entry] {len(prompts)} prompts, {len(feature_sets)} feature sets, "
          f"model={model}, layer={layer}, SAE={sae_path}")
    result = ablate_sae_features.remote(
        prompts, feature_sets,
        sae_path=sae_path, run_id=run_id,
        model_name=model, layer=layer,
        feature_set_labels=feature_set_labels,
        seed=seed,
    )
    print(f"[done] wrote: {result['n_prompts']} rows; out: {result.get('out_path','?')}")

    # Pull the file local
    import subprocess
    local = Path("results/sae") / run_id
    local.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["modal", "volume", "get", "noop-activations",
             f"sae/{run_id}/ablation_results.json", str(local / "ablation_results.json"),
             "--force"],
            check=True,
        )
        print(f"[entry] downloaded to {local / 'ablation_results.json'}")
    except Exception as e:
        print(f"[entry] WARN download failed: {e}")
