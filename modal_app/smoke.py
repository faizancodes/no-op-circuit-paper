"""End-to-end smoke entrypoint.

Run with:

    modal run modal_app/smoke.py

It loads the parser_empty_input task, renders prompts for the three paired
variants (issue_only, code, code_tests) under both buggy and fixed conditions,
ships them to the GPU function, and prints the resulting action logits and
shapes so we can sanity-check the pipeline.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .cache_activations import cache_activations
from .common import app

from no_op_circuit.agent import ACTION_NAMES, build_prompt
from no_op_circuit.config import RESULTS_DIR
from no_op_circuit.dataset import VARIANTS, load_task


@app.local_entrypoint()
def smoke(
    task_id: str = "parser_empty_input",
    variants: str = "issue_only,code,code_tests",
    download: bool = True,
):
    task = load_task(task_id)
    variant_names = [v.strip() for v in variants.split(",") if v.strip()]

    jobs = []
    for variant_name in variant_names:
        variant = VARIANTS[variant_name]
        for condition in ("buggy", "fixed"):
            build = build_prompt(task, condition, variant)
            jobs.append(
                {
                    "task_id": build.task_id,
                    "condition": build.condition,
                    "variant_name": build.variant_name,
                    "messages": build.messages,
                    "action_suffix": build.action_suffix,
                    "action_names": ACTION_NAMES,
                }
            )

    run_id = f"smoke-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    print(f"[smoke] launching {len(jobs)} prompts, run_id={run_id}", flush=True)
    manifest = cache_activations.remote(jobs, run_id=run_id)

    # Pretty-print the headline numbers.
    print("\n[smoke] action logits per (condition, variant):", flush=True)
    print(f"  {'condition':<8} {'variant':<18} {'argmax':<8} " + "  ".join(f"{n:>8}" for n in ACTION_NAMES))
    for e in manifest["entries"]:
        logits = e["action_logits"]
        print(
            f"  {e['condition']:<8} {e['variant']:<18} {e['action_argmax']:<8} "
            + "  ".join(f"{logits[n]:>8.3f}" for n in ACTION_NAMES)
        )

    # Edit-vs-noop margin: positive => model wants to edit, negative => abstain.
    print("\n[smoke] edit - noop logit margin (positive ⇒ edit):", flush=True)
    for e in manifest["entries"]:
        margin = e["action_logits"]["edit"] - e["action_logits"]["noop"]
        print(f"  {e['condition']:<8} {e['variant']:<18} margin={margin:+.3f}")

    if download:
        out_dir = RESULTS_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\n[smoke] wrote manifest to {manifest_path}", flush=True)

        # Download one sample .pt to validate the round-trip.
        if manifest["entries"]:
            sample = manifest["entries"][0]
            remote = sample["path"]  # relative to volume root: {run_id}/{task}/{cond__var}.pt
            local_target = out_dir / Path(remote).name
            try:
                subprocess.run(
                    [
                        "modal", "volume", "get",
                        "noop-activations", remote, str(local_target),
                        "--force",
                    ],
                    check=True,
                )
                print(f"[smoke] downloaded {remote} → {local_target}", flush=True)
                size = local_target.stat().st_size
                print(f"[smoke] file size: {size/1024:.1f} KB", flush=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                print(f"[smoke] WARN: could not download sample ({exc})", flush=True)

    print("\n[smoke] done.", flush=True)
    return manifest
