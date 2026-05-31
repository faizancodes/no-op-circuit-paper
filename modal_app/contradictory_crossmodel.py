"""Contradictory-transcript control on CodeGemma + DeepSeek (cross-model §5.2).

Replicates the §5.2 2x2 (code x transcript) control — currently Qwen-only —
on the other two models, using each model's frozen toy-trained v_noop at its
§4.3 patching-peak cell. Four cells per task:

  (B,B) buggy code + buggy transcript    <- code_tests / buggy
  (F,F) fixed code + fixed transcript    <- code_tests / fixed
  (B,F) buggy code + fixed transcript    <- code_tests_swapped / buggy
  (F,B) fixed code + buggy transcript    <- code_tests_swapped / fixed

For each cell we forward once, capture resid_pre[peak_layer, pos -1], project
onto the unit v_noop, and record score = -projection (the §5.1 convention:
higher = more buggy-like) plus the first-token argmax action. Only a small
per-cell JSON is returned (no large tensors) so nothing competes with the HF
cache upload for local bandwidth.

The local analysis (scripts/analyze_contradictory_crossmodel.py) computes the
identical statistics as scripts/contradictory_transcript_analysis.py:
  ΔCode       = 0.5 * ((BB+BF) - (FB+FF))
  ΔTranscript = 0.5 * ((BB+FB) - (BF+FF))
  + interaction, bootstrap CIs, code-vs-transcript AUCs.

Run:
    modal run -m modal_app.contradictory_crossmodel
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .common import (
    DEFAULT_GPU,
    DEFAULT_TIMEOUT_S,
    HF_CACHE_DIR,
    ACTIVATIONS_DIR,
    activations_vol,
    app,
    hf_cache_vol,
)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT_S,
    volumes={HF_CACHE_DIR: hf_cache_vol, ACTIVATIONS_DIR: activations_vol},
)
def score_cells(
    prompt_jobs: list[dict[str, Any]],
    *,
    v_noop: list[float],
    layer: int,
    model_name: str,
    max_tokens: int = 4096,
    dtype: str = "bfloat16",
) -> list[dict[str, Any]]:
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from no_op_circuit.agent.prompt import render_chat_template_safe
    from no_op_circuit.interp.hooks import cache_forward

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)
    # Reduce fragmentation OOMs on the 7B model (per the CUDA error hint).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}[dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[contra] loading {model_name} ({dtype})…", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=HF_CACHE_DIR, torch_dtype=td, device_map=device,
    )
    model.eval()

    v = torch.tensor(v_noop, dtype=torch.float32)
    v = v / v.norm()

    def first_tok(prefix: str, name: str) -> int:
        pi = tok.encode(prefix, add_special_tokens=False)
        fi = tok.encode(prefix + name, add_special_tokens=False)
        i = 0
        while i < len(pi) and i < len(fi) and pi[i] == fi[i]:
            i += 1
        return fi[i]

    # Only the last-position logits are consumed. Ask the LM to materialise just
    # those — the full (1, T, V) buffer OOMs A10G on CodeGemma-7B (vocab ~256k).
    # `logits_to_keep` (transformers>=5) / `num_logits_to_keep` (4.40+).
    sig = (model.forward.__doc__ or "")
    slice_kw = {"logits_to_keep": 1} if "logits_to_keep" in sig else {"num_logits_to_keep": 1}

    rows: list[dict[str, Any]] = []
    n_skip = 0
    for k, job in enumerate(prompt_jobs):
        rendered = render_chat_template_safe(
            tok, job["messages"], tokenize=False, add_generation_prompt=True)
        full = rendered + job.get("action_suffix", "Action: ")
        ids = tok(full, return_tensors="pt")
        if int(ids["input_ids"].shape[1]) > max_tokens:
            n_skip += 1
            continue  # over-long → drops from the common-across-4-cells set
        anames = job["action_names"]
        aids = {n: first_tok(full, n) for n in anames}
        inputs = {kk: vv.to(device) for kk, vv in ids.items()}
        with torch.no_grad(), cache_forward(model, last_k=1,
                                            hook_points=("resid_pre",)) as cache:
            try:
                out = model(**inputs, **slice_kw)
            except TypeError:
                out = model(**inputs)
        logits = out.logits[0, -1, :].float().cpu()
        resid = cache.resid_pre[layer, 0, -1, :].float().cpu()
        proj = float(torch.dot(resid, v))
        al = {n: float(logits[i].item()) for n, i in aids.items()}
        argmax = max(al.items(), key=lambda x: x[1])[0]
        rows.append({
            "task_id": job["task_id"],
            "cell": job["cell"],          # "BB" | "FF" | "BF" | "FB"
            "score": -proj,               # §5.1 convention: higher = buggy-like
            "argmax_action": argmax,
        })
        if (k + 1) % 400 == 0:
            print(f"[contra] {k+1}/{len(prompt_jobs)} (skipped {n_skip})", flush=True)
    print(f"[contra] done: {len(rows)} scored, {n_skip} over-{max_tokens}-token skips",
          flush=True)
    return rows


@app.local_entrypoint()
def main(tasks_root: str = "data/real_tasks", out_path: str = ""):
    from pathlib import Path
    import torch

    from no_op_circuit.agent import ACTION_NAMES, build_prompt
    from no_op_circuit.dataset import VARIANTS, iter_tasks

    # (model_name, v_noop_path, layer, max_tokens)
    targets = [
        ("google/codegemma-7b-it",
         "results/v_noop_codegemma_all49.pt", 26, 2400),
        ("deepseek-ai/deepseek-coder-1.3b-instruct",
         "results/steer-deepseek-coder-13b-instruct-20260517T012848Z/v_noop.pt",
         22, 4096),
    ]

    tasks = list(iter_tasks(Path(tasks_root)))
    print(f"[contra] {len(tasks)} tasks")

    # Build the 4 cells per task (model-agnostic; tokenisation happens on GPU).
    cell_specs = [
        ("code_tests", "buggy", "BB"),
        ("code_tests", "fixed", "FF"),
        ("code_tests_swapped", "buggy", "BF"),
        ("code_tests_swapped", "fixed", "FB"),
    ]
    jobs: list[dict[str, Any]] = []
    for vname, cond, cell in cell_specs:
        variant = VARIANTS[vname]
        for task in tasks:
            b = build_prompt(task, cond, variant, list(ACTION_NAMES))
            jobs.append({
                "task_id": b.task_id,
                "cell": cell,
                "messages": b.messages,
                "action_suffix": b.action_suffix,
                "action_names": list(ACTION_NAMES),
            })
    print(f"[contra] {len(jobs)} cell-prompts (4 cells x {len(tasks)} tasks)")

    results = {}
    for model_name, v_path, layer, max_tokens in targets:
        blob = torch.load(v_path, map_location="cpu", weights_only=False)
        direction = blob["direction"].tolist()
        assert int(blob["layer"]) == layer and int(blob["position"]) == -1
        print(f"[contra] scoring {model_name}  L{layer}/pos-1  |v|={blob['norm']:.3f}  "
              f"cap={max_tokens}")
        rows = score_cells.remote(
            jobs, v_noop=direction, layer=layer, model_name=model_name,
            max_tokens=max_tokens,
        )
        results[model_name] = {
            "v_noop_path": v_path, "layer": layer, "max_tokens": max_tokens,
            "rows": rows,
        }

    if not out_path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = f"results/monitor_real/contradictory_crossmodel_{ts}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"tasks_root": tasks_root,
                                          "results": results}, indent=2))
    print(f"[contra] wrote {out_path}")
