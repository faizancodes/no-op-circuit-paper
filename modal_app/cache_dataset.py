"""Bulk activation-caching entrypoint.

Iterates the full TASKS_DIR, builds prompts for each (task × variant × condition)
triple, ships them to the GPU function, and writes the manifest to the
activations Volume. Designed for phase-2 probe / patching workflows where we
need the same activation tensors over a non-trivial corpus.

    modal run -m modal_app.cache_dataset
    modal run -m modal_app.cache_dataset --variants issue_only,code,code_tests
    modal run -m modal_app.cache_dataset --task-glob "*off_by_one*"
"""

from __future__ import annotations

import fnmatch
import json
import subprocess
from datetime import datetime, timezone

from .cache_activations import cache_activations
from .common import app

from no_op_circuit.agent import ACTION_NAMES, build_prompt
from no_op_circuit.config import RESULTS_DIR
from no_op_circuit.dataset import VARIANTS, iter_tasks


@app.local_entrypoint()
def cache_dataset(
    variants: str = "issue_only,code,code_tests",
    task_glob: str = "*",
    run_id: str = "",
    model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    tasks_root: str = "",
    download_manifest: bool = True,
    download_activations: bool = True,
    max_prompt_tokens: int = 4096,
    action_vocab: str = "",
):
    from pathlib import Path
    from no_op_circuit.config import TASKS_DIR
    variant_names = [v.strip() for v in variants.split(",") if v.strip()]
    chosen_variants = [VARIANTS[v] for v in variant_names]

    if action_vocab:
        action_names_override = [a.strip() for a in action_vocab.split(",") if a.strip()]
        if len(action_names_override) != len(ACTION_NAMES):
            raise SystemExit(
                f"--action-vocab must have exactly {len(ACTION_NAMES)} comma-separated "
                f"actions; got {len(action_names_override)}: {action_names_override}"
            )
        print(f"[cache_dataset] action_vocab override: {action_names_override}")
    else:
        action_names_override = list(ACTION_NAMES)

    tasks_dir = Path(tasks_root) if tasks_root else TASKS_DIR
    if not tasks_dir.is_dir():
        raise SystemExit(f"--tasks-root not a directory: {tasks_dir}")

    jobs: list[dict] = []
    selected_task_ids: list[str] = []
    for task in iter_tasks(tasks_dir=tasks_dir):
        if not fnmatch.fnmatch(task.task_id, task_glob):
            continue
        selected_task_ids.append(task.task_id)
        for variant in chosen_variants:
            for condition in ("buggy", "fixed"):
                # stale_* variants pin a side; skip when the pinned transcript
                # is missing (e.g., stale_flaky without a flaky transcript).
                try:
                    build = build_prompt(task, condition, variant,
                                         action_names=action_names_override)
                except ValueError:
                    continue
                jobs.append(
                    {
                        "task_id": build.task_id,
                        "condition": build.condition,
                        "variant_name": build.variant_name,
                        "messages": build.messages,
                        "action_suffix": build.action_suffix,
                        "action_names": action_names_override,
                    }
                )

    if not jobs:
        print("[cache_dataset] no jobs to run — check --task-glob and --variants")
        return

    # Optional length pre-filter (needed for CodeGemma-7B on A10G to avoid OOM
    # on long real-task prompts; lm_head allocation tips over near 4K tokens).
    # We tokenise each prompt locally and drop BOTH conditions of a (task, variant)
    # pair if either side exceeds the cap, so paired analysis stays valid.
    if max_prompt_tokens < 4096:
        from transformers import AutoTokenizer
        from no_op_circuit.agent.prompt import render_chat_template_safe
        tok = AutoTokenizer.from_pretrained(model)
        # group by (task_id, variant); max-len across the 2 conditions
        from collections import defaultdict
        lens: dict[tuple[str, str], int] = defaultdict(int)
        for j in jobs:
            text = render_chat_template_safe(
                tok, j["messages"], tokenize=False, add_generation_prompt=True
            ) + j.get("action_suffix", "Action: ")
            n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
            key = (j["task_id"], j["variant_name"])
            lens[key] = max(lens[key], n_tok)
        drop_keys = {k for k, n in lens.items() if n > max_prompt_tokens}
        if drop_keys:
            before = len(jobs)
            jobs = [j for j in jobs
                    if (j["task_id"], j["variant_name"]) not in drop_keys]
            dropped_tasks = {k[0] for k in drop_keys}
            print(f"[cache_dataset] dropped {len(drop_keys)} (task, variant) pairs "
                  f"({before - len(jobs)} prompts; {len(dropped_tasks)} distinct tasks) "
                  f"for exceeding max_prompt_tokens={max_prompt_tokens}")
        else:
            print(f"[cache_dataset] all prompts within max_prompt_tokens={max_prompt_tokens}")

    model_slug = model.split("/")[-1].replace(".", "").replace("-", "_").lower()[:24]
    if not run_id:
        run_id = f"cache-{model_slug}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    print(
        f"[cache_dataset] model={model} · {len(selected_task_ids)} tasks · "
        f"{len(chosen_variants)} variants · {len(jobs)} prompts · run_id={run_id}",
        flush=True,
    )
    manifest = cache_activations.remote(jobs, run_id=run_id, model_name=model)

    # Per-row summary (compact).
    print("\n[cache_dataset] action argmax / (edit - noop) margin per row:")
    by_task: dict[str, list[dict]] = {}
    for e in manifest["entries"]:
        by_task.setdefault(e["task_id"], []).append(e)
    for tid in sorted(by_task):
        for e in by_task[tid]:
            margin = e["action_logits"]["edit"] - e["action_logits"]["noop"]
            print(
                f"  {tid:<40} {e['condition']:<5} {e['variant']:<14} "
                f"argmax={e['action_argmax']:<6} edit-noop={margin:+.2f}"
            )

    # Paired contrast (where buggy and fixed exist for a given variant).
    print("\n[cache_dataset] paired Δ margin (buggy − fixed) — positive ⇒ test evidence pushes toward noop:")
    for tid in sorted(by_task):
        pairs: dict[str, dict[str, float]] = {}
        for e in by_task[tid]:
            v = e["variant"]
            margin = e["action_logits"]["edit"] - e["action_logits"]["noop"]
            pairs.setdefault(v, {})[e["condition"]] = margin
        for v, sides in pairs.items():
            if "buggy" in sides and "fixed" in sides:
                d = sides["buggy"] - sides["fixed"]
                print(f"  {tid:<40} {v:<14} Δ={d:+.2f}")

    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if download_manifest:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\n[cache_dataset] wrote manifest to {out_dir / 'manifest.json'}")

    if download_activations:
        print(f"[cache_dataset] downloading {len(manifest['entries'])} .pt files via modal CLI…")
        try:
            subprocess.run(
                [
                    "modal", "volume", "get",
                    "noop-activations", run_id, str(out_dir),
                    "--force",
                ],
                check=True,
            )
            print(f"[cache_dataset] downloaded all .pt files to {out_dir}/")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(
                f"[cache_dataset] WARN: download failed ({exc}). Fetch manually with:\n"
                f"  modal volume get noop-activations {run_id}/ {out_dir}/"
            )

    print("\n[cache_dataset] done.", flush=True)
