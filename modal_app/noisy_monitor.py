"""Phase 4: lean residual-monitor scorer for noisy-transcript variants.

Projects resid_pre[layer, pos=-1] onto the frozen unit u_tx (legacy artifact
`v_noop`) and returns score = -projection (the Sec 5.1 convention), per prompt.
Caches only the final position (last_k=1) -- no full-residual storage. Validate
against the code_tests headline (AUC ~0.989) before trusting on noisy variants.

Run (Qwen; validation + 4 noisy variants in one job):
    modal run -m modal_app.noisy_monitor --variants \
      code_tests,code_tests_noisy_flaky,code_tests_many_passing,code_tests_truncated,code_tests_summary_only
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import DEFAULT_GPU, DEFAULT_MODEL, HF_CACHE_DIR, app, hf_cache_vol


def _run_score_monitor(
    jobs: list[dict[str, Any]],
    *,
    model_name: str = DEFAULT_MODEL,
    layer: int = 24,
    u_tx: list[float] | None = None,
    dtype: str = "bfloat16",
) -> list[dict[str, Any]]:
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from no_op_circuit.agent.prompt import render_chat_template_safe
    from no_op_circuit.interp.hooks import cache_forward

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=HF_CACHE_DIR, torch_dtype=td, device_map=device
    )
    model.eval()
    u = torch.tensor(u_tx, dtype=torch.float32)
    u = u / u.norm()

    rows = []
    for n, job in enumerate(jobs):
        rendered = render_chat_template_safe(
            tok, job["messages"], tokenize=False, add_generation_prompt=True
        )
        full = rendered + job.get("action_suffix", "Action: ")
        inputs = tok(full, return_tensors="pt").to(device)
        with torch.no_grad(), cache_forward(model, last_k=1, hook_points=("resid_pre",)) as cache:
            try:
                model(**inputs, num_logits_to_keep=1)  # avoid full-vocab logits on 7B
            except TypeError:
                model(**inputs)
        vec = cache.resid_pre[layer, 0, 0, :].float().cpu()
        proj = float(vec @ u)
        rows.append({
            "task_id": job.get("task_id"), "condition": job.get("condition"),
            "variant": job.get("variant_name"), "projection": proj, "score": -proj,
        })
        if n % 200 == 0:
            print(f"[score_monitor] {n}/{len(jobs)}", flush=True)
    return rows


@app.function(gpu=DEFAULT_GPU, timeout=60 * 60, volumes={HF_CACHE_DIR: hf_cache_vol})
def score_monitor(jobs, *, model_name=DEFAULT_MODEL, layer=24, u_tx=None, dtype="bfloat16"):
    return _run_score_monitor(jobs, model_name=model_name, layer=layer, u_tx=u_tx, dtype=dtype)


@app.function(gpu="A100", timeout=60 * 60, volumes={HF_CACHE_DIR: hf_cache_vol})
def score_monitor_a100(jobs, *, model_name=DEFAULT_MODEL, layer=24, u_tx=None, dtype="bfloat16"):
    return _run_score_monitor(jobs, model_name=model_name, layer=layer, u_tx=u_tx, dtype=dtype)


@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    variants: str = "code_tests",
    tasks: str = "real",
    layer: int = 24,
    v_noop: str = "results/steer-20260516T021522Z/v_noop.pt",
    limit: int = 0,
):
    import torch

    from no_op_circuit.agent.prompt import build_prompt
    from no_op_circuit.config import DATA_DIR
    from no_op_circuit.dataset import VARIANTS, iter_tasks

    blob = torch.load(v_noop, map_location="cpu", weights_only=False)
    assert int(blob["layer"]) == layer, (blob["layer"], layer)
    u = blob["direction"].float()
    u = (u / u.norm()).tolist()

    tasks_dir = DATA_DIR / ("real_tasks" if tasks == "real" else "tasks")
    all_tasks = list(iter_tasks(tasks_dir=tasks_dir))
    if limit:
        all_tasks = all_tasks[:limit]
    var_names = [v.strip() for v in variants.split(",") if v.strip()]

    jobs = []
    for vn in var_names:
        var = VARIANTS[vn]
        for task in all_tasks:
            for cond in ("buggy", "fixed"):
                pb = build_prompt(task, cond, var)
                jobs.append({"task_id": task.task_id, "condition": cond, "variant_name": vn,
                             "messages": pb.messages, "action_suffix": pb.action_suffix})

    print(f"[noisy_monitor] {len(jobs)} jobs over {len(all_tasks)} tasks, variants={var_names}")
    rows = score_monitor.remote(jobs, model_name=model, layer=layer, u_tx=u)
    out = Path("results/noisy_monitor")
    out.mkdir(parents=True, exist_ok=True)
    tag = "qwen" if "qwen" in model.lower() else model.split("/")[-1]
    (out / f"{tag}_monitor_projections.json").write_text(
        json.dumps({"model": model, "layer": layer, "variants": var_names, "rows": rows}, indent=2)
    )
    print(f"[noisy_monitor] wrote results/noisy_monitor/{tag}_monitor_projections.json ({len(rows)} rows)")
