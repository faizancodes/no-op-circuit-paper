"""Qwen SWE-derived peak-cell causal patching with all-five action logits.

Experiment A in the reviewer-driven revision: tests whether the same
Qwen L24/pos -1 causal readout that moves the `edit - noop` margin on the
49 toy tasks also moves it on a deterministic subset of SWE-bench-Verified-
derived `code_tests` paired prompts. Wrong-layer (L12/pos -1) and
wrong-position (L24/pos -8) controls are computed side-by-side. All five
action logits are logged so argmax transitions can be reported.

Run (smoke then full):
    modal run -m modal_app.swe_peak_patching --n-pairs 10
    modal run -m modal_app.swe_peak_patching --n-pairs 200

Use --tasks toy --n-pairs 49 to reproduce the same five-action measurement
on the toy substrate (Experiment B in the prompt).

Writes:
    results/swe_peak_patching/<tag>_swe_peak_patch_scores.json    (per-pair)
Run `scripts/analyze_swe_peak_patching.py` locally for bootstrap summaries.
"""

from __future__ import annotations

import json
import random
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

# (layer index into model.model.layers, position offset relative to next-token
# slot). Cell 1 is the claimed causal site; cell 2 wrong-layer same-position;
# cell 3 wrong-position same-layer.
DEFAULT_CELLS: list[tuple[int, int]] = [(24, -1), (12, -1), (24, -8)]
LAST_K = 8  # must cover the most-negative offset (pos -8)


def _cell_id(layer: int, pos: int) -> str:
    return f"L{layer}_pos{pos}"


def _repo_of(task_id: str) -> str:
    """SWE-bench task IDs are ``<owner>_<repo>_<issue>``; return ``owner/repo``."""
    head = task_id.rsplit("_", 1)[0]
    parts = head.split("_", 1)
    return f"{parts[0]}/{parts[1]}" if len(parts) == 2 else head


def _tag(model_name: str) -> str:
    s = model_name.lower()
    if "qwen" in s:
        return "qwen"
    if "deepseek" in s:
        return "deepseek"
    if "codegemma" in s or "code-gemma" in s:
        return "codegemma"
    return s.split("/")[-1]


def _select_pairs(all_ids: list[str], n: int, seed: int, stratify_by_repo: bool) -> list[str]:
    """Deterministic subset selection. Stratify by repo when requested."""
    rng = random.Random(seed)
    sorted_ids = sorted(all_ids)
    if not stratify_by_repo:
        ids = sorted_ids[:]
        rng.shuffle(ids)
        return ids[:n]
    by_repo: dict[str, list[str]] = {}
    for tid in sorted_ids:
        by_repo.setdefault(_repo_of(tid), []).append(tid)
    repos = sorted(by_repo)
    total = sum(len(v) for v in by_repo.values())
    alloc: dict[str, int] = {r: max(1, round(n * len(by_repo[r]) / total)) for r in repos}
    diff = n - sum(alloc.values())
    by_size = sorted(repos, key=lambda r: -len(by_repo[r]))
    i = 0
    while diff != 0 and i < 100_000:
        r = by_size[i % len(by_size)]
        if diff > 0 and alloc[r] < len(by_repo[r]):
            alloc[r] += 1
            diff -= 1
        elif diff < 0 and alloc[r] > 1:
            alloc[r] -= 1
            diff += 1
        i += 1
    selected: list[str] = []
    for r in repos:
        chunk = by_repo[r][:]
        rng.shuffle(chunk)
        selected.extend(chunk[: alloc[r]])
    return selected


def _run_swe_peak_patching(
    pairs: list[dict[str, Any]],
    *,
    cells: list[list[int]] | None = None,
    model_name: str = DEFAULT_MODEL,
    dtype: str = "bfloat16",
) -> list[dict[str, Any]]:
    """Per pair: 2 clean + (#cells × 2) patched forwards; log all 5 action logits."""
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from no_op_circuit.agent.prompt import render_chat_template_safe
    from no_op_circuit.interp.hooks import ResidualPatch, cache_forward, patched_forward

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)

    cells_t: list[tuple[int, int]] = [
        (int(c[0]), int(c[1])) for c in (cells if cells is not None else [list(c) for c in DEFAULT_CELLS])
    ]
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[swe_peak] loading {model_name} ({dtype}); cells={cells_t}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=HF_CACHE_DIR, torch_dtype=td, device_map=device
    )
    model.eval()
    print(f"[swe_peak] loaded in {time.time()-t0:.1f}s", flush=True)

    def action_first_id(prefix: str, name: str) -> int:
        a = tok.encode(prefix, add_special_tokens=False)
        b = tok.encode(prefix + name, add_special_tokens=False)
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return b[i] if i < len(b) else b[-1]

    def tokenize(messages, suffix):
        rendered = render_chat_template_safe(tok, messages, tokenize=False, add_generation_prompt=True)
        full = rendered + suffix
        return tok(full, return_tensors="pt").to(device), full

    rows: list[dict[str, Any]] = []
    for n, pair in enumerate(pairs):
        task_id = pair["task_id"]
        action_names = pair["action_names"]
        action_suffix = pair.get("action_suffix", "Action: ")

        buggy_inputs, buggy_text = tokenize(pair["buggy_messages"], action_suffix)
        fixed_inputs, _ = tokenize(pair["fixed_messages"], action_suffix)
        action_ids = {nm: action_first_id(buggy_text, nm) for nm in action_names}

        # Clean BUGGY + cache resid_pre last_k
        with torch.no_grad(), cache_forward(
            model, last_k=LAST_K, hook_points=("resid_pre",)
        ) as buggy_cache:
            buggy_clean = model(**buggy_inputs).logits[0, -1, :].float().cpu()
        buggy_resid = buggy_cache.resid_pre
        assert buggy_resid is not None

        # Clean FIXED + cache resid_pre last_k
        with torch.no_grad(), cache_forward(
            model, last_k=LAST_K, hook_points=("resid_pre",)
        ) as fixed_cache:
            fixed_clean = model(**fixed_inputs).logits[0, -1, :].float().cpu()
        fixed_resid = fixed_cache.resid_pre
        assert fixed_resid is not None

        def logits_dict(L) -> dict[str, float]:
            return {nm: float(L[action_ids[nm]].item()) for nm in action_names}

        clean_buggy_l = logits_dict(buggy_clean)
        clean_fixed_l = logits_dict(fixed_clean)
        buggy_seq = int(buggy_inputs["input_ids"].shape[1])
        fixed_seq = int(fixed_inputs["input_ids"].shape[1])

        patched: dict[str, dict[str, dict[str, float]]] = {}
        for (layer, off) in cells_t:
            cell = _cell_id(layer, off)
            cache_idx = LAST_K + off
            buggy_pos = buggy_seq + off
            fixed_pos = fixed_seq + off

            f2b_patch = [ResidualPatch(
                layer_idx=layer, hook_point="resid_pre",
                position=buggy_pos, value=fixed_resid[layer, 0, cache_idx, :],
            )]
            with torch.no_grad(), patched_forward(model, f2b_patch):
                out_f2b = model(**buggy_inputs).logits[0, -1, :].float().cpu()

            b2f_patch = [ResidualPatch(
                layer_idx=layer, hook_point="resid_pre",
                position=fixed_pos, value=buggy_resid[layer, 0, cache_idx, :],
            )]
            with torch.no_grad(), patched_forward(model, b2f_patch):
                out_b2f = model(**fixed_inputs).logits[0, -1, :].float().cpu()

            patched[cell] = {
                "f2b_logits": logits_dict(out_f2b),
                "b2f_logits": logits_dict(out_b2f),
            }

        rows.append({
            "task_id": task_id,
            "repo": pair.get("repo", ""),
            "buggy_seq": buggy_seq,
            "fixed_seq": fixed_seq,
            "action_ids": action_ids,
            "clean_buggy_logits": clean_buggy_l,
            "clean_fixed_logits": clean_fixed_l,
            "patched": patched,
        })

        if n % 20 == 0 or n == len(pairs) - 1:
            ab = pair["action_names"][3]  # "edit"
            an = pair["action_names"][4]  # abstain word (noop/done)
            mb = clean_buggy_l[ab] - clean_buggy_l[an]
            mf = clean_fixed_l[ab] - clean_fixed_l[an]
            print(f"[swe_peak] {n+1}/{len(pairs)} {task_id} clean m_b/m_f={mb:+.2f}/{mf:+.2f}", flush=True)

    return rows


@app.function(
    gpu=DEFAULT_GPU,
    timeout=60 * 60,
    volumes={HF_CACHE_DIR: hf_cache_vol, ACTIVATIONS_DIR: activations_vol},
)
def run_swe_peak_patching(pairs, *, cells=None, model_name=DEFAULT_MODEL, dtype="bfloat16"):
    return _run_swe_peak_patching(pairs, cells=cells, model_name=model_name, dtype=dtype)


@app.function(
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={HF_CACHE_DIR: hf_cache_vol, ACTIVATIONS_DIR: activations_vol},
)
def run_swe_peak_patching_a100(pairs, *, cells=None, model_name=DEFAULT_MODEL, dtype="bfloat16"):
    return _run_swe_peak_patching(pairs, cells=cells, model_name=model_name, dtype=dtype)


@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    n_pairs: int = 200,
    seed: int = 0,
    tasks: str = "real",
    variant: str = "code_tests",
    cells: str = "",  # override: "L,pos;L,pos;L,pos"
    stratify_by_repo: bool = True,
    output_dir: str = "results/swe_peak_patching",
    gpu: str = "",  # "" = A10G default; "a100" = A100
    action_words: str = "",  # override canonical menu; e.g. "view,find,test,edit,done"
    abstain_word: str = "noop",  # last word in action_words is the abstention label
):
    from no_op_circuit.agent.actions import ACTION_NAMES
    from no_op_circuit.agent.prompt import build_prompt
    from no_op_circuit.config import DATA_DIR
    from no_op_circuit.dataset import VARIANTS, iter_tasks

    tasks_dir = DATA_DIR / ("real_tasks" if tasks == "real" else "tasks")
    var = VARIANTS[variant]

    custom_words: list[str] | None = None
    if action_words:
        custom_words = [w.strip() for w in action_words.split(",") if w.strip()]
        if abstain_word and abstain_word not in custom_words:
            raise SystemExit(f"abstain_word {abstain_word!r} must appear in action_words")

    all_tasks = list(iter_tasks(tasks_dir=tasks_dir))
    all_ids = [t.task_id for t in all_tasks]
    if n_pairs and 0 < n_pairs < len(all_ids):
        chosen_ids = _select_pairs(all_ids, n_pairs, seed, stratify_by_repo)
    else:
        chosen_ids = sorted(all_ids)
    id_to_task = {t.task_id: t for t in all_tasks}
    ordered = [id_to_task[i] for i in chosen_ids]

    pairs: list[dict[str, Any]] = []
    for task in ordered:
        if custom_words is not None:
            # Re-render the prompt with a custom single-token action vocabulary
            # (e.g. DeepSeek-safe `{view, find, test, edit, done}`). Reuse the
            # canonical menu order helper so the system prompt matches our
            # other custom-vocab controls (§5.5).
            from no_op_circuit.agent.action_order import system_prompt_for_order
            buggy = build_prompt(task, "buggy", var, action_names=custom_words)
            fixed = build_prompt(task, "fixed", var, action_names=custom_words)
            sys_msg = {"role": "system", "content": system_prompt_for_order(custom_words)}
            buggy.messages[0] = sys_msg
            fixed.messages[0] = sys_msg
            action_names = list(custom_words)
        else:
            buggy = build_prompt(task, "buggy", var)
            fixed = build_prompt(task, "fixed", var)
            action_names = list(ACTION_NAMES)
        pairs.append({
            "task_id": task.task_id,
            "repo": _repo_of(task.task_id),
            "variant_name": variant,
            "buggy_messages": buggy.messages,
            "fixed_messages": fixed.messages,
            "action_suffix": buggy.action_suffix,
            "action_names": action_names,
        })

    if cells:
        parsed = [[int(x) for x in chunk.split(",")] for chunk in cells.split(";") if chunk.strip()]
    else:
        parsed = [list(c) for c in DEFAULT_CELLS]

    print(f"[swe_peak] {len(pairs)} pairs (tasks={tasks}, n_pairs={n_pairs}, "
          f"seed={seed}, stratify={stratify_by_repo}), cells={parsed}, "
          f"gpu={gpu or 'A10G'}, vocab={pairs[0]['action_names'] if pairs else []}")

    fn = run_swe_peak_patching_a100 if gpu.lower() == "a100" else run_swe_peak_patching
    rows = fn.remote(pairs, cells=parsed, model_name=model)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if tasks == "real" else "_toy"
    if custom_words is not None:
        suffix = f"{suffix}_single_token"
    out_path = out_dir / f"{_tag(model)}_swe_peak_patch_scores{suffix}.json"
    out_path.write_text(json.dumps({
        "model": model,
        "tasks": tasks,
        "variant": variant,
        "n_pairs": len(pairs),
        "seed": seed,
        "stratify_by_repo": stratify_by_repo,
        "cells": parsed,
        "action_names": pairs[0]["action_names"] if pairs else [],
        "abstain_word": abstain_word if custom_words is not None else "noop",
        "task_ids": [p["task_id"] for p in pairs],
        "rows": rows,
    }, indent=2))
    print(f"[swe_peak] wrote {out_path} ({len(rows)} rows)")
