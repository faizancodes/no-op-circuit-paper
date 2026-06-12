"""Local entrypoint: compute v_noop from cached activations, sweep steering on Modal.

Pipeline:
  1. Load cached resid_pre tensors at (layer, position=-1) for buggy & fixed,
     from a prior cache_dataset run on disk.
  2. Compute v_noop = mean(fixed) - mean(buggy). Pointing buggy → fixed.
  3. Build prompts for every (task, condition) in the chosen variant.
  4. Ship prompts + v_noop to the Modal `run_steering` function with the alpha
     sweep.
  5. Pull the manifest local, write it to results/<run_id>/.

    modal run -m modal_app.steer_dataset \\
        --cache-dir results/cache-20260515T221105Z \\
        --variant code_tests \\
        --layer 24 --position -1

By default we sweep alphas at the canonical \"natural unit\" of ||v_noop|| —
expressed in units of v_noop's own norm:
    -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3.
With unit_alpha_in_v_norm=True (default), alpha=1 means \"add v_noop\" exactly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import app, pick_tier, record_spawn, resolve_gpu
from .steering import STEER_TIERS

from no_op_circuit.agent import ACTION_NAMES, build_prompt
from no_op_circuit.config import RESULTS_DIR
from no_op_circuit.dataset import VARIANTS, iter_tasks


DEFAULT_ALPHAS = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def _compute_v_noop(
    cache_dir: Path,
    *,
    variant: str,
    layer: int,
    position: int,
    hook_point: str = "resid_pre",
    restrict_task_ids: set[str] | None = None,
):
    """Compute v_noop = mean(fixed) - mean(buggy) at (layer, position).

    Returns (v_tensor_float32, n_pairs, ||v||).
    """
    import torch

    buggy_acts: list = []
    fixed_acts: list = []
    pair_task_ids: list[str] = []

    # cache_dir may be flat or contain a nested <run_id>/ dir (modal volume get
    # behaviour). rglob handles both.
    by_task: dict[str, dict[str, Any]] = {}
    for pt in sorted(cache_dir.rglob("*.pt")):
        name = pt.stem  # e.g. "<condition>__<variant>"
        if "__" not in name:
            continue
        condition, var_name = name.split("__", 1)
        if var_name != variant or condition not in ("buggy", "fixed"):
            continue
        try:
            payload = torch.load(pt, map_location="cpu", weights_only=False)
        except (RuntimeError, EOFError, OSError) as exc:
            # A flaky `modal volume get` can corrupt a file; skip it (its pair
            # drops) rather than crash the whole v_noop computation.
            print(f"[steer_dataset] WARN: skipping unreadable {pt.name}: {exc}", flush=True)
            continue
        task_id = payload["task_id"]
        by_task.setdefault(task_id, {})[condition] = payload

    for task_id in sorted(by_task):
        sides = by_task[task_id]
        if "buggy" not in sides or "fixed" not in sides:
            continue
        if restrict_task_ids is not None and task_id not in restrict_task_ids:
            continue
        # resid_pre shape (L, B=1, K, D). Index by layer + position (negative
        # offset resolved against K).
        b_t = sides["buggy"]["resid_pre"] if hook_point == "resid_pre" else sides["buggy"]["resid_post"]
        f_t = sides["fixed"]["resid_pre"] if hook_point == "resid_pre" else sides["fixed"]["resid_post"]
        if b_t is None or f_t is None:
            continue
        K = int(b_t.shape[2])
        pos_abs = position if position >= 0 else K + position
        buggy_acts.append(b_t[layer, 0, pos_abs, :].float())
        fixed_acts.append(f_t[layer, 0, pos_abs, :].float())
        pair_task_ids.append(task_id)

    if not buggy_acts:
        raise SystemExit(
            f"No paired activations found in {cache_dir} for variant={variant!r}"
        )
    buggy_mean = torch.stack(buggy_acts).mean(0)
    fixed_mean = torch.stack(fixed_acts).mean(0)
    v = fixed_mean - buggy_mean
    return v, len(pair_task_ids), float(v.norm().item()), pair_task_ids


@app.local_entrypoint()
def steer_dataset(
    cache_dir: str = "results/cache-20260515T221105Z",
    variant: str = "code_tests",
    layer: int = 24,
    position: int = -1,
    hook_point: str = "resid_pre",
    alphas: str = "",         # comma-separated; defaults to DEFAULT_ALPHAS
    task_glob: str = "*",
    task_ids: str = "",       # comma-separated; restricts to these tasks AND
                              # restricts v_noop computation to them too
    run_id: str = "",
    model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    gpu: str = "auto",          # "auto" picks a tier by model size; "default" keeps A10G
    spawn: bool = False,        # with `modal run --detach`: fire-and-forget, survives shutdown
):
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise SystemExit(f"cache_dir not found: {cache_path}")

    var = VARIANTS[variant]
    if var.pin_files_to is not None or var.pin_tests_to is not None:
        raise SystemExit(f"variant {variant!r} pins a side; steering needs paired buggy/fixed")

    alpha_list = (
        [float(x) for x in alphas.split(",") if x.strip()]
        if alphas
        else list(DEFAULT_ALPHAS)
    )

    explicit_ids = {t.strip() for t in task_ids.split(",") if t.strip()} or None
    print(
        f"[steer_dataset] computing v_noop from {cache_path}"
        + (f" (restricted to {len(explicit_ids)} tasks)" if explicit_ids else ""),
        flush=True,
    )
    v, n_pairs, v_norm, source_task_ids = _compute_v_noop(
        cache_path,
        variant=variant,
        layer=layer,
        position=position,
        hook_point=hook_point,
        restrict_task_ids=explicit_ids,
    )
    print(
        f"[steer_dataset] v_noop computed over N={n_pairs} pairs · "
        f"layer={layer} pos={position} hook_point={hook_point} · ||v|| = {v_norm:.3f}",
        flush=True,
    )

    # Build prompts for every (task, condition) — match the same set used to
    # compute v_noop so the result is in-distribution.
    import fnmatch
    selected_set = set(source_task_ids)
    prompts: list[dict] = []
    for task in iter_tasks():
        if task.task_id not in selected_set or not fnmatch.fnmatch(task.task_id, task_glob):
            continue
        for condition in ("buggy", "fixed"):
            try:
                build = build_prompt(task, condition, var)
            except ValueError:
                continue
            prompts.append(
                {
                    "task_id": build.task_id,
                    "condition": build.condition,
                    "variant_name": build.variant_name,
                    "messages": build.messages,
                    "action_suffix": build.action_suffix,
                    "action_names": ACTION_NAMES,
                }
            )

    if not prompts:
        raise SystemExit("No prompts to steer — check --task-glob and --variant.")

    model_slug = model.split("/")[-1].replace(".", "").replace("-", "_").lower()[:24]
    if not run_id:
        run_id = f"steer-{model_slug}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    print(
        f"[steer_dataset] model={model} · {len(prompts)} prompts × {len(alpha_list)} alphas "
        f"= {len(prompts)*len(alpha_list)} forwards · run_id={run_id}",
        flush=True,
    )

    fn = pick_tier(STEER_TIERS, model, gpu)
    print(f"[steer_dataset] gpu tier={resolve_gpu(model, gpu) or 'A10G'}", flush=True)
    if spawn:
        call = fn.spawn(
            prompts, run_id=run_id, model_name=model, direction=v.tolist(),
            alphas=alpha_list, layer=layer, position=position, hook_point=hook_point,
        )
        record_spawn("steer_dataset", model, run_id, call.object_id)
        # v_noop.pt is computed locally from the cache; recompute on demand if needed.
        return
    manifest = fn.remote(
        prompts,
        run_id=run_id,
        model_name=model,
        direction=v.tolist(),
        alphas=alpha_list,
        layer=layer,
        position=position,
        hook_point=hook_point,
    )

    # Compact print: per (condition, alpha) mean margin across tasks.
    rows = manifest["rows"]
    by_cond_alpha: dict[tuple[str, float], list[float]] = {}
    for r in rows:
        by_cond_alpha.setdefault((r["condition"], r["alpha"]), []).append(r["edit_minus_noop"])

    print("\n[steer_dataset] dose-response: mean (edit−noop) margin vs alpha")
    print(f"  {'alpha':>6}  {'buggy':>10}  {'fixed':>10}  {'gap':>10}")
    for alpha in alpha_list:
        b = by_cond_alpha.get(("buggy", alpha), [])
        f = by_cond_alpha.get(("fixed", alpha), [])
        b_m = sum(b) / len(b) if b else float("nan")
        f_m = sum(f) / len(f) if f else float("nan")
        gap = b_m - f_m if (b and f) else float("nan")
        print(f"  {alpha:+6.2f}  {b_m:+10.3f}  {f_m:+10.3f}  {gap:+10.3f}")

    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Also persist the steering vector and metadata for plotting/SAE work.
    import torch as _torch
    _torch.save(
        {
            "direction": v,
            "norm": v_norm,
            "n_pairs": n_pairs,
            "layer": layer,
            "position": position,
            "hook_point": hook_point,
            "variant": variant,
            "source_task_ids": source_task_ids,
        },
        out_dir / "v_noop.pt",
    )
    print(f"\n[steer_dataset] wrote manifest + v_noop.pt to {out_dir}/")
