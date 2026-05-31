"""Modal job: action-order / abstract-label / binary action-menu controls.

Phase 2 of the critical-review revision. Tests whether the 0% explicit-`noop`
rate is a list-position artifact (noop always last) rather than a property of the
abstention content. Lean scorer: forward pass + action-token logits only (no
residual caching).

Run (Qwen, smoke then full):
    modal run -m modal_app.action_order_control --experiment action_order --limit 10
    modal run -m modal_app.action_order_control --experiment action_order
    modal run -m modal_app.action_order_control --experiment binary
    modal run -m modal_app.action_order_control --experiment abstract_label

Writes results/action_order_control/<tag>_<experiment>_scores.json locally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_GPU,
    DEFAULT_MODEL,
    HF_CACHE_DIR,
    app,
    hf_cache_vol,
)


def _run_scoring(
    jobs: list[dict[str, Any]],
    *,
    model_name: str = DEFAULT_MODEL,
    dtype: str = "bfloat16",
) -> list[dict[str, Any]]:
    """Score each job's candidate action tokens at the final position."""
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from no_op_circuit.agent.prompt import render_chat_template_safe

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=HF_CACHE_DIR, torch_dtype=td, device_map=device
    )
    model.eval()

    def first_tok(prefix: str, name: str) -> tuple[int, int]:
        p = tok.encode(prefix, add_special_tokens=False)
        f = tok.encode(prefix + name, add_special_tokens=False)
        i = 0
        while i < len(p) and i < len(f) and p[i] == f[i]:
            i += 1
        if i >= len(f):
            raise ValueError(f"no new token from {name!r}")
        return f[i], len(f) - i

    rows: list[dict[str, Any]] = []
    for n, job in enumerate(jobs):
        rendered = render_chat_template_safe(
            tok, job["messages"], tokenize=False, add_generation_prompt=True
        )
        full = rendered + job.get("action_suffix", "Action: ")
        names = job["action_names"]
        ids, ntoks = {}, {}
        for nm in names:
            tid, nt = first_tok(full, nm)
            ids[nm], ntoks[nm] = tid, nt
        inputs = tok(full, return_tensors="pt").to(device)
        with torch.no_grad():
            try:
                out = model(**inputs, num_logits_to_keep=1)
            except TypeError:
                out = model(**inputs)
        nl = out.logits[0, -1, :].float().cpu()
        logits = {nm: float(nl[ids[nm]]) for nm in names}
        # rank of each candidate among the candidate set (1 = highest logit)
        ordered = sorted(names, key=lambda m: logits[m], reverse=True)
        ranks = {nm: ordered.index(nm) + 1 for nm in names}
        argmax = ordered[0]
        rows.append(
            {
                "task_id": job.get("task_id"),
                "condition": job.get("condition"),
                "order_id": job.get("order_id"),
                "noop_pos": job.get("noop_pos"),
                "names": names,
                "mapping": job.get("mapping"),
                "action_logits": logits,
                "ranks": ranks,
                "argmax": argmax,
                "n_tokens": ntoks,
                "all_single_token": all(v == 1 for v in ntoks.values()),
                "seq_len": int(inputs["input_ids"].shape[1]),
            }
        )
        if n % 200 == 0:
            print(f"[score_actions] {n}/{len(jobs)}  argmax={argmax}", flush=True)
    return rows


@app.function(gpu=DEFAULT_GPU, timeout=60 * 60, volumes={HF_CACHE_DIR: hf_cache_vol})
def score_actions(jobs, *, model_name: str = DEFAULT_MODEL, dtype: str = "bfloat16"):
    return _run_scoring(jobs, model_name=model_name, dtype=dtype)


@app.function(gpu="A100", timeout=60 * 60, volumes={HF_CACHE_DIR: hf_cache_vol})
def score_actions_a100(jobs, *, model_name: str = DEFAULT_MODEL, dtype: str = "bfloat16"):
    return _run_scoring(jobs, model_name=model_name, dtype=dtype)


def _tag(model_name: str) -> str:
    s = model_name.lower()
    if "qwen" in s:
        return "qwen"
    if "codegemma" in s or "code-gemma" in s:
        return "codegemma"
    if "deepseek" in s:
        return "deepseek"
    return s.split("/")[-1]


@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    experiment: str = "action_order",
    variant: str = "code_tests",
    tasks: str = "real",
    limit: int = 0,
    gpu: str = "",
    action_words: str = "",
    abstain_word: str = "done",
):
    from no_op_circuit.agent.action_order import (
        build_abstract_label_prompts,
        build_action_order_prompts,
        build_binary_prompts,
        build_custom_order_prompts,
        build_letter_only_prompts,
        to_job,
    )
    from no_op_circuit.config import DATA_DIR
    from no_op_circuit.dataset import VARIANTS, iter_tasks

    tasks_dir = DATA_DIR / ("real_tasks" if tasks == "real" else "tasks")
    var = VARIANTS[variant]
    all_tasks = list(iter_tasks(tasks_dir=tasks_dir))
    if limit:
        all_tasks = all_tasks[:limit]

    jobs: list[dict[str, Any]] = []
    for task in all_tasks:
        for condition in ("buggy", "fixed"):
            if experiment == "action_order":
                for k, order, noop_pos, pb in build_action_order_prompts(task, condition, var):
                    jobs.append(to_job(pb, order, order_id=k, noop_pos=noop_pos))
            elif experiment == "binary":
                for k, order, noop_pos, pb in build_binary_prompts(task, condition, var):
                    jobs.append(to_job(pb, order, order_id=k, noop_pos=noop_pos))
            elif experiment == "abstract_label":
                for k, mapping, noop_idx, pb in build_abstract_label_prompts(task, condition, var):
                    jobs.append(to_job(pb, list("ABCDE"), order_id=k, noop_pos=noop_idx, mapping=mapping))
            elif experiment == "letter_only":
                for k, _m, _n, pb in build_letter_only_prompts(task, condition, var):
                    jobs.append(to_job(pb, list("ABCDE"), order_id=k))
            elif experiment == "action_order_custom":
                words = [w.strip() for w in action_words.split(",") if w.strip()]
                for k, order, ap, pb in build_custom_order_prompts(task, condition, var, words, abstain_word):
                    jobs.append(to_job(pb, order, order_id=k, noop_pos=ap))
            else:
                raise SystemExit(f"unknown experiment {experiment!r}")

    print(f"[action_order] {experiment}: {len(jobs)} jobs over {len(all_tasks)} tasks ({tasks})")
    fn = score_actions_a100 if gpu else score_actions
    rows = fn.remote(jobs, model_name=model)

    out_dir = Path("results/action_order_control")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_tag(model)}_{experiment}_scores.json"
    out_path.write_text(json.dumps({"model": model, "experiment": experiment,
                                    "variant": variant, "tasks": tasks,
                                    "n_tasks": len(all_tasks), "rows": rows}, indent=2))
    print(f"[action_order] wrote {out_path} ({len(rows)} rows)")
