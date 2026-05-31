"""Run paired activation patching across the dataset.

For each (task, variant) where buggy+fixed both render, this ships a job to
the `run_patching` GPU function. The function patches FIXED residuals into the
BUGGY forward at each (layer, suffix-position) and reports the shift in the
`edit - noop` logit margin.

    modal run -m modal_app.patch_dataset
    modal run -m modal_app.patch_dataset --variant code_tests
    modal run -m modal_app.patch_dataset --task-glob "*off_by_one*"
"""

from __future__ import annotations

import fnmatch
import json
import subprocess
from datetime import datetime, timezone

from .common import app
from .patching import run_patching

from no_op_circuit.agent import ACTION_NAMES, build_prompt
from no_op_circuit.config import RESULTS_DIR
from no_op_circuit.dataset import VARIANTS, iter_tasks


@app.local_entrypoint()
def patch_dataset(
    variant: str = "code_tests",
    task_glob: str = "*",
    task_ids: str = "",        # comma-separated list; overrides task_glob if set
    run_id: str = "",          # explicit run_id (e.g. to extend a prior run)
    max_pairs: int = 0,        # 0 = no cap
    max_suffix: int = 8,
    layer_step: int = 1,
    bidirectional: bool = False,
    model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    download_heatmaps: bool = True,
):
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}. Known: {list(VARIANTS)}")
    var = VARIANTS[variant]
    if var.pin_files_to is not None or var.pin_tests_to is not None:
        raise SystemExit(f"variant {variant!r} pins a side; patching needs paired buggy/fixed")

    explicit_ids = {t.strip() for t in task_ids.split(",") if t.strip()}

    pairs: list[dict] = []
    selected: list[str] = []
    for task in iter_tasks():
        if explicit_ids:
            if task.task_id not in explicit_ids:
                continue
        elif not fnmatch.fnmatch(task.task_id, task_glob):
            continue
        try:
            buggy = build_prompt(task, "buggy", var)
            fixed = build_prompt(task, "fixed", var)
        except ValueError:
            continue
        selected.append(task.task_id)
        pairs.append(
            {
                "task_id": task.task_id,
                "variant_name": variant,
                "buggy_messages": buggy.messages,
                "fixed_messages": fixed.messages,
                "action_suffix": buggy.action_suffix,
                "action_names": ACTION_NAMES,
            }
        )
        if max_pairs and len(pairs) >= max_pairs:
            break

    if explicit_ids:
        missing = explicit_ids - set(selected)
        if missing:
            print(f"[patch_dataset] WARN: requested task_ids not found on disk: {sorted(missing)}")

    if not pairs:
        print("[patch_dataset] no pairs to patch — check --task-glob, --task-ids, --variant")
        return

    model_slug = model.split("/")[-1].replace(".", "").replace("-", "_").lower()[:24]
    if not run_id:
        run_id = f"patch-{model_slug}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    print(
        f"[patch_dataset] model={model} · {len(pairs)} pairs · variant={variant} · "
        f"max_suffix={max_suffix} · layer_step={layer_step} · run_id={run_id}",
        flush=True,
    )
    manifest = run_patching.remote(
        pairs,
        run_id=run_id,
        model_name=model,
        max_suffix=max_suffix,
        layer_step=layer_step,
        bidirectional=bidirectional,
    )

    is_bidir = bool(manifest.get("bidirectional"))
    print("\n[patch_dataset] per-pair clean margins and patched ranges:")
    if is_bidir:
        print(f"  {'task_id':<40} clean(b/f)   f2b∈[min,max]      b2f∈[min,max]")
    else:
        print(f"  {'task_id':<40} clean_b   f2b∈[min,max]")
    for s in manifest["summaries"]:
        cb = s["clean_buggy_margin"]
        f2b = (s.get("min_patched_f2b") or s.get("min_patched_margin"),
               s.get("max_patched_f2b") or s.get("max_patched_margin"))
        if is_bidir:
            cf = s["clean_fixed_margin"]
            b2f = (s.get("min_patched_b2f"), s.get("max_patched_b2f"))
            print(
                f"  {s['task_id']:<40} {cb:+.2f}/{cf:+.2f}   "
                f"[{f2b[0]:+.2f},{f2b[1]:+.2f}]    [{b2f[0]:+.2f},{b2f[1]:+.2f}]"
            )
        else:
            print(f"  {s['task_id']:<40} {cb:+.2f}   [{f2b[0]:+.2f},{f2b[1]:+.2f}]")

    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[patch_dataset] wrote manifest to {out_dir / 'manifest.json'}")

    if download_heatmaps:
        print(f"[patch_dataset] downloading {len(manifest['summaries'])} heatmap .pt files…")
        try:
            subprocess.run(
                [
                    "modal", "volume", "get",
                    "noop-activations", run_id, str(out_dir),
                    "--force",
                ],
                check=True,
            )
            print(f"[patch_dataset] downloaded heatmaps to {out_dir}/")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(
                f"[patch_dataset] WARN: download failed ({exc}). Fetch manually with:\n"
                f"  modal volume get noop-activations {run_id}/ {out_dir}/"
            )
