#!/usr/bin/env python
"""Audit the G.17 held-out paraphrase counts from artifacts (CPU, no forwards).

Resolves the ambiguous paper sentence about held-out-template counts by reading
results/heldout_paraphrase_robustness/{qwen,codegemma,deepseek}_scores.json and
reporting, per model: unique task IDs, per-template task/condition counts,
token-cap drops, and how the pooled held-out AUC is composed.

Writes:
  results/heldout_paraphrase_robustness/count_audit.json
  results/heldout_paraphrase_robustness/count_audit.md
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIR = REPO / "results/heldout_paraphrase_robustness"
MODELS = ("qwen", "codegemma", "deepseek")
N_BASE_TASKS = {"qwen": 499, "codegemma": 497, "deepseek": 499}  # §5.1 SWE-derived base


def audit_model(model: str) -> dict:
    scores_path = DIR / f"{model}_scores.json"
    summ_path = DIR / f"{model}_summary.json"
    d = json.loads(scores_path.read_text())
    summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    out: dict = {
        "scores_file": str(scores_path.relative_to(REPO)),
        "summary_file": str(summ_path.relative_to(REPO)) if summ_path.exists() else None,
        "site": summ.get("site"),
        "direction_artifact": summ.get("direction_artifact"),
    }
    for fam in ("train", "heldout"):
        rows = d[fam]
        uniq = sorted({r["task_id"] for r in rows})
        per_tmpl = {}
        for ti in sorted({r.get("template_idx") for r in rows}):
            sub = [r for r in rows if r.get("template_idx") == ti]
            per_tmpl[f"template_{ti}"] = {
                "unique_task_ids": len(set(r["task_id"] for r in sub)),
                "prompts_by_condition": dict(Counter(r["condition"] for r in sub)),
            }
        out[fam] = {
            "total_prompts": len(rows),
            "unique_task_ids": len(uniq),
            "prompts_by_condition": dict(Counter(r["condition"] for r in rows)),
            "per_template": per_tmpl,
        }
    # token-cap / missing-task check: held-out unique tasks vs §5.1 base
    held_uniq = out["heldout"]["unique_task_ids"]
    out["base_tasks_5p1"] = N_BASE_TASKS.get(model)
    out["token_cap_or_missing_drops"] = (N_BASE_TASKS.get(model, held_uniq) - held_uniq)
    # how pooled held-out AUC is composed
    n_tmpl = len(out["heldout"]["per_template"])
    tmpl_task_sum = sum(t["unique_task_ids"] for t in out["heldout"]["per_template"].values())
    out["pooled_composition"] = (
        f"{held_uniq} paired tasks split across {n_tmpl} held-out templates "
        f"({', '.join(str(t['unique_task_ids']) for t in out['heldout']['per_template'].values())} tasks); "
        f"{out['heldout']['total_prompts']} total prompts; pooled AUC pools all templates"
    )
    out["pooled_is_split_not_per_template_499"] = (tmpl_task_sum == held_uniq and n_tmpl > 1)
    return out


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    audit = {m: audit_model(m) for m in MODELS}
    (DIR / "count_audit.json").write_text(json.dumps(audit, indent=2))

    lines = ["# G.17 held-out paraphrase count audit", "",
             "Per-model held-out counts from `*_scores.json` (CPU, no model forwards).", ""]
    lines.append("| model | held-out unique tasks | template A tasks | template B tasks | total prompts | token-cap/missing drops |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in MODELS:
        a = audit[m]; h = a["heldout"]; pt = h["per_template"]
        ta = list(pt.values())
        lines.append(f"| {m} | {h['unique_task_ids']} | "
                     f"{ta[0]['unique_task_ids'] if len(ta)>0 else '—'} | "
                     f"{ta[1]['unique_task_ids'] if len(ta)>1 else '—'} | "
                     f"{h['total_prompts']} | {a['token_cap_or_missing_drops']} |")
    lines += ["", "## Pooled composition", ""]
    for m in MODELS:
        lines.append(f"- **{m}**: {audit[m]['pooled_composition']}")
    (DIR / "count_audit.md").write_text("\n".join(lines) + "\n")

    print("=== G.17 count audit ===")
    for m in MODELS:
        a = audit[m]; h = a["heldout"]
        print(f"{m:10} held-out unique tasks={h['unique_task_ids']} "
              f"(A={list(h['per_template'].values())[0]['unique_task_ids']}, "
              f"B={list(h['per_template'].values())[1]['unique_task_ids']}) "
              f"prompts={h['total_prompts']} drops={a['token_cap_or_missing_drops']} "
              f"| split-not-per-template-499={a['pooled_is_split_not_per_template_499']}")
    print(f"\nwrote {DIR/'count_audit.json'} and count_audit.md")


if __name__ == "__main__":
    main()
