"""Modal function: random-feature ablation baseline for §5.2.

Tier-1 specificity control for the OMP top-8 result. For each seed we draw
8 SAE features uniformly without replacement from {0,…,d_sae-1} \\
OMP-top-128, then run the same per-prompt ablation pipeline as
`ablate_sae_features.py`. If the random-8 reduction distribution sits
clearly below the OMP top-8 effect (33.1%), the OMP picks are
direction-specific, not just "any 8 features perturb the residual".

Model + SAE are loaded once; the 296 paired prompts are re-tokenised
per-seed loop iteration (cheap). Each seed writes its own
`results/sae/ablate-random/seed_<S>/ablation_results.json` so they can be
analysed independently by `scripts/random_baseline_compare.py`.
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

_ABLATE_TIMEOUT_S = 60 * 60 * 3  # generous: 10 seeds × ~10 min


@app.function(
    gpu=DEFAULT_GPU,
    timeout=_ABLATE_TIMEOUT_S,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        ACTIVATIONS_DIR: activations_vol,
    },
)
def ablate_random_features(
    prompts: list[dict[str, Any]],
    excluded_feature_indices: list[int],
    *,
    sae_path: str = "sae/qwen_l24_resid_pre_TASK_d4096_k16.pt",
    model_name: str = DEFAULT_MODEL,
    layer: int = 24,
    position: int = -1,
    dtype: str = "bfloat16",
    n_seeds: int = 10,
    n_features_per_seed: int = 8,
    base_seed: int = 0,
    run_id_prefix: str = "ablate-random",
    eligible_pool: list[int] | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from no_op_circuit.agent.prompt import render_chat_template_safe
    from no_op_circuit.interp.hooks import ResidualPatch, patched_forward
    from no_op_circuit.interp.sae import SAEConfig, TopKSAE

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]

    t0 = time.time()
    print(f"[random] loading {model_name}…", flush=True)
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
    d_sae = cfg.d_sae
    print(f"[random] model+SAE ready in {time.time()-t0:.1f}s "
          f"(d_sae={d_sae}, excluded={len(excluded_feature_indices)} OMP features)",
          flush=True)

    layers = model.model.layers
    n_layers = len(layers)
    assert 0 <= layer < n_layers

    captured: dict[str, Any] = {"clean_resid": None}
    def capture_hook(_mod, args, kwargs=None):
        if args and isinstance(args[0], torch.Tensor):
            captured["clean_resid"] = args[0][:, -1, :].detach().clone()
        return None

    def action_first_token_id(prefix: str, name: str) -> int:
        a = tok.encode(prefix, add_special_tokens=False)
        b = tok.encode(prefix + name, add_special_tokens=False)
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return b[i] if i < len(b) else b[-1]

    def build_replacement(clean_resid: torch.Tensor, feature_set: list[int]) -> torch.Tensor:
        x = clean_resid.to(dtype=torch.float32)
        with torch.no_grad():
            a = sae.encode(x)
            error = x - sae.decode(a)
            if feature_set:
                a = a.clone()
                idx = torch.tensor(feature_set, device=a.device, dtype=torch.long)
                a[:, idx] = 0.0
            x_new = sae.decode(a) + error
        return x_new.to(dtype=clean_resid.dtype)

    excluded_set = set(int(i) for i in excluded_feature_indices)
    if eligible_pool is not None:
        # Caller-supplied pool (e.g. firing-≥5 features); still apply exclusion.
        eligible = [int(i) for i in eligible_pool if int(i) not in excluded_set]
        print(f"[random] eligible pool size: {len(eligible)} "
              f"(caller-supplied, after excluding {len(excluded_set)})", flush=True)
    else:
        eligible = [i for i in range(d_sae) if i not in excluded_set]
        print(f"[random] eligible pool size: {len(eligible)} (full d_sae minus excluded)",
              flush=True)
    assert len(eligible) >= n_features_per_seed, (
        f"eligible pool ({len(eligible)}) smaller than n_features_per_seed "
        f"({n_features_per_seed})"
    )

    summaries: list[dict[str, Any]] = []
    for s in range(n_seeds):
        seed = base_seed + s
        # Deterministic draw via torch.Generator
        g = torch.Generator(); g.manual_seed(seed)
        perm = torch.randperm(len(eligible), generator=g)[:n_features_per_seed].tolist()
        rand_features = sorted(int(eligible[i]) for i in perm)
        print(f"\n[random] === seed {seed}: features {rand_features} ===", flush=True)

        run_id = f"{run_id_prefix}/seed_{seed:02d}"
        results: list[dict[str, Any]] = []
        for p_idx, prompt in enumerate(prompts):
            rendered = render_chat_template_safe(
                tok, prompt["messages"], tokenize=False, add_generation_prompt=True
            )
            full_text = rendered + prompt.get("action_suffix", "Action: ")
            inputs = tok(full_text, return_tensors="pt").to(device)
            action_names = prompt["action_names"]
            action_ids = {n: action_first_token_id(full_text, n) for n in action_names}
            edit_id = action_ids["edit"]; noop_id = action_ids["noop"]

            handle = layers[layer].register_forward_pre_hook(capture_hook, with_kwargs=True)
            try:
                with torch.no_grad():
                    logits = model(**inputs).logits[0, -1, :]
            finally:
                handle.remove()
            clean_margin = float((logits[edit_id] - logits[noop_id]).item())
            clean_resid = captured["clean_resid"]
            assert clean_resid is not None

            abs_pos = inputs["input_ids"].shape[1] + position
            replacement = build_replacement(clean_resid, rand_features)
            patch = ResidualPatch(layer_idx=layer, hook_point="resid_pre",
                                  position=abs_pos, value=replacement[0])
            with torch.no_grad(), patched_forward(model, [patch]):
                out = model(**inputs).logits[0, -1, :]
            ab_margin = float((out[edit_id] - out[noop_id]).item())

            results.append({
                "task_id": prompt["task_id"],
                "condition": prompt["condition"],
                "variant": prompt["variant_name"],
                "margins": {"clean": clean_margin, "ablate_random8": ab_margin},
            })
            if (p_idx + 1) % 50 == 0:
                print(f"[random]   seed {seed} prompt {p_idx+1}/{len(prompts)}", flush=True)

        out_dir = Path(ACTIVATIONS_DIR) / "sae" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "seed": seed,
            "sae_path": sae_path,
            "layer": layer,
            "position": position,
            "feature_indices": rand_features,
            "excluded_set_size": len(excluded_set),
            "n_prompts": len(prompts),
            "rows": results,
        }
        (out_dir / "ablation_results.json").write_text(json.dumps(payload, indent=2))
        activations_vol.commit()
        summaries.append({"seed": seed, "features": rand_features,
                          "out_path": str(out_dir.relative_to(ACTIVATIONS_DIR) / "ablation_results.json")})
        print(f"[random]   wrote seed_{seed:02d}/ablation_results.json", flush=True)

    print(f"\n[random] all {n_seeds} seeds done in {time.time()-t0:.1f}s", flush=True)
    return {"n_seeds": n_seeds, "n_features_per_seed": n_features_per_seed,
            "n_prompts": len(prompts), "summaries": summaries}


@app.local_entrypoint()
def ablate_random_entry(
    features_json: str = "results/sae/v_noop_features_DISTRIBUTED.json",
    sae_path: str = "sae/qwen_l24_resid_pre_TASK_d4096_k16.pt",
    n_seeds: int = 10,
    n_features_per_seed: int = 8,
    base_seed: int = 0,
    eligible_pool_json: str | None = None,
    firing_min: int = 5,
    run_id_prefix: str | None = None,
    model: str = DEFAULT_MODEL,
    layer: int = 24,
):
    """Run random-feature ablation as a specificity control for OMP top-8.

    eligible_pool_json: optional path to a firing-features JSON (output of
        compute_firing_features). When supplied, the random sample is drawn
        from features with firing count >= firing_min, restricted to
        non-OMP-top-128. The default run_id_prefix flips to
        "ablate-random-firing" so the outputs don't collide with the
        original full-d_sae random run.
    """
    from no_op_circuit.dataset import iter_tasks, VARIANTS
    from no_op_circuit.agent import ACTION_NAMES, build_prompt
    from no_op_circuit.config import TASKS_DIR

    feats_blob = json.loads(Path(features_json).read_text())
    excluded = [int(f["feature_idx"]) for f in feats_blob["top_features"]]
    print(f"[entry] excluding {len(excluded)} OMP features (top-128) from eligible pool")

    eligible_pool: list[int] | None = None
    if eligible_pool_json is not None:
        fblob = json.loads(Path(eligible_pool_json).read_text())
        eligible_pool = [int(k) for k, v in fblob["firing_counts"].items()
                         if int(v) >= firing_min]
        print(f"[entry] firing-only pool: {len(eligible_pool)} features "
              f"fire on >= {firing_min} prompts (before OMP-128 exclusion)")
        if run_id_prefix is None:
            run_id_prefix = "ablate-random-firing"
    if run_id_prefix is None:
        run_id_prefix = "ablate-random"

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
    print(f"[entry] {len(prompts)} prompts × {n_seeds} seeds = "
          f"{len(prompts) * n_seeds * 2} forwards (run_id_prefix={run_id_prefix})")

    result = ablate_random_features.remote(
        prompts, excluded,
        sae_path=sae_path,
        n_seeds=n_seeds, n_features_per_seed=n_features_per_seed,
        base_seed=base_seed,
        run_id_prefix=run_id_prefix,
        eligible_pool=eligible_pool,
        model_name=model, layer=layer,
    )
    print(f"[done] {result['n_seeds']} seeds, {result['n_prompts']} prompts each")

    # Pull each seed file
    import subprocess
    local_root = Path(f"results/sae/{run_id_prefix}")
    local_root.mkdir(parents=True, exist_ok=True)
    for s in result["summaries"]:
        seed = s["seed"]
        local_dir = local_root / f"seed_{seed:02d}"
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["modal", "volume", "get", "noop-activations", s["out_path"],
                 str(local_dir / "ablation_results.json"), "--force"],
                check=True,
            )
        except Exception as e:
            print(f"[entry] WARN download seed_{seed:02d} failed: {e}")
    print(f"[entry] downloaded {len(result['summaries'])} seeds to {local_root}")
