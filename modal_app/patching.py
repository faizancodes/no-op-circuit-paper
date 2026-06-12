"""Modal function: paired activation patching for the edit-vs-noop circuit.

For a list of (task_id, variant) pairs, this:

  1. Loads the model (cached on the Volume).
  2. Runs the BUGGY prompt with residual-stream cache.
  3. For each (layer, suffix_position) in a sweep grid, patches the BUGGY
     forward by substituting the captured FIXED residual at that site and
     measures the resulting `edit - noop` logit margin.
  4. Saves a per-pair heatmap (layers × positions) of margin shift.

Two key design decisions:

  * Position alignment is by **suffix offset** — both prompts may have
    different lengths, so we anchor position -1 (next-token) and walk
    backwards up to `max_suffix`.
  * Patching uses `resid_pre` (the input to layer L) by default, matching the
    standard activation-patching convention. resid_post is configurable.
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
    register_tiers,
)

_PATCH_TIMEOUT_S = 60 * 60 * 2  # 2 hours — bidirectional patching is ~2x cost


@app.function(
    gpu=DEFAULT_GPU,
    timeout=_PATCH_TIMEOUT_S,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        ACTIVATIONS_DIR: activations_vol,
    },
)
def run_patching(
    pairs: list[dict[str, Any]],
    *,
    run_id: str,
    model_name: str = DEFAULT_MODEL,
    hook_point: str = "resid_pre",
    max_suffix: int = 8,
    layer_step: int = 1,
    layer_min: int = 0,
    layer_max: int = -1,
    dtype: str = "bfloat16",
    bidirectional: bool = False,
) -> dict[str, Any]:
    """Paired activation patching for every (task, variant) pair in `pairs`.

    When `bidirectional` is True, also patches BUGGY residuals into the FIXED
    forward at every (layer, position), so each saved payload contains both
    `patched_margins_f2b` (existing) and `patched_margins_b2f` arrays plus
    `clean_fixed_margin`. The asymmetry test for the no-op direction needs
    both legs.

    Each pair dict:
        task_id        : str
        variant_name   : str
        buggy_messages : list[chat-message-dict]
        fixed_messages : list[chat-message-dict]
        action_suffix  : str
        action_names   : list[str]
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from no_op_circuit.agent.prompt import render_chat_template_safe
    from no_op_circuit.interp.hooks import (
        ResidualPatch,
        cache_forward,
        patched_forward,
    )

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[patching] loading {model_name} ({dtype})…", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    print(f"[patching] loaded in {time.time()-t0:.1f}s", flush=True)

    layers = (model.model.layers if hasattr(model, "model") else model.transformer.h)
    n_layers = len(layers)
    hi = layer_max if layer_max >= 0 else n_layers - 1
    layer_indices = [l for l in range(0, n_layers, layer_step) if layer_min <= l <= hi]

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
    summaries: list[dict[str, Any]] = []

    for pair in pairs:
        task_id = pair["task_id"]
        variant = pair["variant_name"]
        action_names = pair["action_names"]
        action_suffix = pair.get("action_suffix", "Action: ")

        buggy_inputs, buggy_text = tokenize(pair["buggy_messages"], action_suffix)
        fixed_inputs, _ = tokenize(pair["fixed_messages"], action_suffix)

        # Resolve action token ids on the buggy-side prompt; the suffix is
        # identical between buggy and fixed so the ids agree either side.
        action_ids = {n: action_first_id(buggy_text, n) for n in action_names}
        edit_id, noop_id = action_ids["edit"], action_ids["noop"]

        # Capture FIXED residuals at every layer × last-K position.
        with torch.no_grad(), cache_forward(
            model, last_k=max_suffix, hook_points=(hook_point,)
        ) as fixed_cache:
            fixed_clean = model(**fixed_inputs).logits[0, -1, :]
        fixed_resid = fixed_cache.resid_pre if hook_point == "resid_pre" else fixed_cache.resid_post
        assert fixed_resid is not None, f"fixed_cache has no {hook_point} tensor"
        _, _, K, _ = fixed_resid.shape  # (L, B=1, K, D)
        clean_fixed_margin = float((fixed_clean[edit_id] - fixed_clean[noop_id]).item())

        # Capture BUGGY residuals + clean margin in one forward.
        if bidirectional:
            with torch.no_grad(), cache_forward(
                model, last_k=max_suffix, hook_points=(hook_point,)
            ) as buggy_cache:
                buggy_clean = model(**buggy_inputs).logits[0, -1, :]
            buggy_resid = buggy_cache.resid_pre if hook_point == "resid_pre" else buggy_cache.resid_post
            assert buggy_resid is not None
        else:
            with torch.no_grad():
                buggy_clean = model(**buggy_inputs).logits[0, -1, :]
            buggy_resid = None
        clean_buggy_margin = float((buggy_clean[edit_id] - buggy_clean[noop_id]).item())

        buggy_seq = int(buggy_inputs["input_ids"].shape[1])
        fixed_seq = int(fixed_inputs["input_ids"].shape[1])
        position_offsets = list(range(-K, 0))  # e.g. [-K..-1]

        heat_f2b: list[list[float]] = []   # patch FIXED's residual into BUGGY forward
        heat_b2f: list[list[float]] = []   # patch BUGGY's residual into FIXED forward
        for layer in layer_indices:
            row_f2b: list[float] = []
            row_b2f: list[float] = []
            for pos_signed in position_offsets:
                cache_idx = K + pos_signed
                buggy_pos = buggy_seq + pos_signed
                fixed_pos = fixed_seq + pos_signed

                # ---- F → B: substitute fixed residual into buggy forward ----
                f2b_patches = [
                    ResidualPatch(
                        layer_idx=layer,
                        hook_point=hook_point,
                        position=buggy_pos,
                        value=fixed_resid[layer, 0, cache_idx, :],
                    )
                ]
                with torch.no_grad(), patched_forward(model, f2b_patches):
                    out = model(**buggy_inputs).logits[0, -1, :]
                row_f2b.append(float((out[edit_id] - out[noop_id]).item()))

                # ---- B → F: substitute buggy residual into fixed forward ----
                if bidirectional:
                    assert buggy_resid is not None
                    b2f_patches = [
                        ResidualPatch(
                            layer_idx=layer,
                            hook_point=hook_point,
                            position=fixed_pos,
                            value=buggy_resid[layer, 0, cache_idx, :],
                        )
                    ]
                    with torch.no_grad(), patched_forward(model, b2f_patches):
                        out = model(**fixed_inputs).logits[0, -1, :]
                    row_b2f.append(float((out[edit_id] - out[noop_id]).item()))

            heat_f2b.append(row_f2b)
            if bidirectional:
                heat_b2f.append(row_b2f)

        # Per-pair payload — keep the legacy keys, add the bidirectional fields.
        payload = {
            "task_id": task_id,
            "variant": variant,
            "model_name": model_name,
            "hook_point": hook_point,
            "layer_indices": layer_indices,
            "position_offsets": position_offsets,
            "clean_buggy_margin": clean_buggy_margin,
            "clean_fixed_margin": clean_fixed_margin,
            "patched_margins": heat_f2b,        # legacy alias
            "patched_margins_f2b": heat_f2b,
            "bidirectional": bidirectional,
            "action_ids": action_ids,
        }
        if bidirectional:
            payload["patched_margins_b2f"] = heat_b2f
        out_path = run_dir / f"{task_id}__{variant}__patch.pt"
        torch.save(payload, out_path)

        summary = {
            "task_id": task_id,
            "variant": variant,
            "clean_buggy_margin": clean_buggy_margin,
            "clean_fixed_margin": clean_fixed_margin,
            "min_patched_f2b": min(min(r) for r in heat_f2b) if heat_f2b else None,
            "max_patched_f2b": max(max(r) for r in heat_f2b) if heat_f2b else None,
            "path": str(out_path.relative_to(ACTIVATIONS_DIR)),
        }
        if bidirectional:
            summary["min_patched_b2f"] = min(min(r) for r in heat_b2f) if heat_b2f else None
            summary["max_patched_b2f"] = max(max(r) for r in heat_b2f) if heat_b2f else None
        summaries.append(summary)
        msg = (
            f"[patching] {task_id}/{variant}  "
            f"clean(b/f)={clean_buggy_margin:+.2f}/{clean_fixed_margin:+.2f}  "
            f"f2b∈[{summary['min_patched_f2b']:+.2f},{summary['max_patched_f2b']:+.2f}]"
        )
        if bidirectional:
            msg += f"  b2f∈[{summary['min_patched_b2f']:+.2f},{summary['max_patched_b2f']:+.2f}]"
        print(msg,
        )
        # Periodic commit so partial work survives a function timeout.
        if len(summaries) % 5 == 0:
            activations_vol.commit()

    manifest = {
        "run_id": run_id,
        "model_name": model_name,
        "hook_point": hook_point,
        "max_suffix": max_suffix,
        "layer_step": layer_step,
        "bidirectional": bidirectional,
        "n_pairs": len(pairs),
        "summaries": summaries,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    activations_vol.commit()
    return manifest


# Pre-registered GPU-tier variants; patch_dataset dispatches by model size.
PATCH_TIERS = {
    "A10G": run_patching,
    **register_tiers(
        run_patching.get_raw_f(), "run_patching",
        volumes={HF_CACHE_DIR: hf_cache_vol, ACTIVATIONS_DIR: activations_vol},
        base_timeout=_PATCH_TIMEOUT_S,
    ),
}
