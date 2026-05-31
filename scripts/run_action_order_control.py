#!/usr/bin/env python3
"""Action-order and binary action-menu controls (paper: Action-order control).

Tests whether the 0% explicit-`noop` rate is a list-position artifact (`noop` is
always listed last in the canonical menu) rather than a property of the
abstention content. Runs LOCALLY on Qwen2.5-Coder-1.5B-Instruct (MPS/CPU),
reusing the paper's action-token scoring methodology
(`modal_app/cache_activations.py`): render the chat template + "Action: ", then
read the last-position logit of each action's first token.

Experiment A (position-balanced cyclic orders): `noop` occupies each of the 5
menu slots once. Experiment C (binary): `{edit, noop}` and `{noop, edit}`.

Usage:
    python scripts/run_action_order_control.py --build-only          # no model
    python scripts/run_action_order_control.py --run-local           # full toy run
    python scripts/run_action_order_control.py --run-local --limit 2 # smoke test

Artifacts:
    results/action_order_control/qwen_action_order_scores.json
    results/action_order_control/qwen_action_order_summary.json
    results/action_order_control/qwen_binary_edit_noop_scores.json
    results/action_order_control/qwen_binary_edit_noop_summary.json
    figures/action_order_control_qwen.png
    figures/binary_edit_noop_control_qwen.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from no_op_circuit.agent.action_order import binary_orders, cyclic_action_orders
from no_op_circuit.agent.prompt import build_prompt, render_chat_template_safe
from no_op_circuit.agent.action_order import system_prompt_for_order
from no_op_circuit.config import DEFAULT_MODEL
from no_op_circuit.dataset import VARIANTS, iter_tasks

OUT = Path("results/action_order_control")
FIGS = Path("figures")
ACTIONS5 = ["view", "grep", "test", "edit", "noop"]


def load_model(slug: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(slug)
    model = AutoModelForCausalLM.from_pretrained(slug, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok, device


def action_first_token_id(tok, prefix: str, name: str) -> int:
    """First token the model emits when continuing `prefix` with `name`
    (diff-tokenize across the BPE boundary; matches cache_activations.py)."""
    pre = tok.encode(prefix, add_special_tokens=False)
    full = tok.encode(prefix + name, add_special_tokens=False)
    i = 0
    while i < len(pre) and i < len(full) and pre[i] == full[i]:
        i += 1
    if i >= len(full):
        raise ValueError(f"no new token for {name!r}")
    return int(full[i])


def verify_single_token(tok, action_names: list[str]) -> dict[str, int]:
    """Each action's first token id, asserting single-token after 'Action: '."""
    prefix = "Action: "
    ids = {}
    for n in action_names:
        pre = tok.encode(prefix, add_special_tokens=False)
        full = tok.encode(prefix + n, add_special_tokens=False)
        i = 0
        while i < len(pre) and i < len(full) and pre[i] == full[i]:
            i += 1
        n_new = len(full) - i
        ids[n] = (int(full[i]), n_new)
    return ids


def score(model, tok, messages, action_names, device):
    import torch

    rendered = render_chat_template_safe(
        tok, messages, tokenize=False, add_generation_prompt=True
    )
    full = rendered + "Action: "
    ids = {n: action_first_token_id(tok, full, n) for n in action_names}
    inp = tok(full, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    nl = out.logits[0, -1, :].float().cpu()
    logits = {n: float(nl[ids[n]]) for n in action_names}
    argmax = max(logits, key=logits.get)
    ranked = sorted(action_names, key=lambda n: logits[n], reverse=True)
    noop_rank = ranked.index("noop") + 1 if "noop" in action_names else None
    return logits, argmax, noop_rank, int(inp["input_ids"].shape[1])


def run(limit: int | None, variant_name: str = "code_tests"):
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    print(f"[aoc] loading {DEFAULT_MODEL} ...", flush=True)
    t0 = time.time()
    model, tok, device = load_model(DEFAULT_MODEL)
    print(f"[aoc] loaded on {device} in {time.time()-t0:.0f}s", flush=True)

    # Single-token check for all 5 actions (Phase 2A.3).
    st = verify_single_token(tok, ACTIONS5)
    single = {n: (nt == 1) for n, (_id, nt) in st.items()}
    print(f"[aoc] single-token after 'Action: ': {single}", flush=True)

    variant = VARIANTS[variant_name]
    tasks = list(iter_tasks())
    if limit:
        tasks = tasks[:limit]
    orders = cyclic_action_orders(ACTIONS5)

    # ---- Experiment A: position-balanced cyclic orders ----
    a_rows = []
    t0 = time.time()
    for ti, task in enumerate(tasks):
        for condition in ("buggy", "fixed"):
            for oid, order in enumerate(orders):
                msgs = build_prompt(task, condition, variant, action_names=order).messages
                msgs = [{"role": "system", "content": system_prompt_for_order(order)}] + msgs[1:]
                logits, argmax, noop_rank, plen = score(model, tok, msgs, order, device)
                a_rows.append({
                    "task_id": task.task_id, "condition": condition,
                    "order_id": oid, "order": order, "noop_pos": order.index("noop"),
                    "argmax": argmax, "is_noop": argmax == "noop",
                    "edit_minus_noop": logits["edit"] - logits["noop"],
                    "noop_logit": logits["noop"], "noop_rank": noop_rank,
                    "logits": logits, "prompt_len": plen,
                })
        if (ti + 1) % 5 == 0:
            print(f"[aoc:A] {ti+1}/{len(tasks)} tasks ({time.time()-t0:.0f}s)", flush=True)
    (OUT / "qwen_action_order_scores.json").write_text(json.dumps(a_rows, indent=2))

    # ---- Experiment C: binary edit/noop ----
    c_rows = []
    for task in tasks:
        for condition in ("buggy", "fixed"):
            for order in binary_orders():
                msgs = build_prompt(task, condition, variant, action_names=order).messages
                logits, argmax, _r, plen = score(model, tok, msgs, order, device)
                c_rows.append({
                    "task_id": task.task_id, "condition": condition,
                    "order": order, "noop_pos": order.index("noop"),
                    "argmax": argmax, "is_noop": argmax == "noop",
                    "edit_minus_noop": logits["edit"] - logits["noop"],
                    "logits": logits, "prompt_len": plen,
                })
    (OUT / "qwen_binary_edit_noop_scores.json").write_text(json.dumps(c_rows, indent=2))

    summarize(a_rows, c_rows, single, st, len(tasks), variant_name)
    make_figures(a_rows, c_rows)
    print("[aoc] done.", flush=True)


def _ci95(vals):
    import statistics
    if len(vals) < 2:
        return [None, None]
    m = statistics.mean(vals)
    sd = statistics.pstdev(vals)
    se = sd / (len(vals) ** 0.5)
    return [m - 1.96 * se, m + 1.96 * se]


def summarize(a_rows, c_rows, single, st, n_tasks, variant_name):
    import statistics
    by_pos = defaultdict(list)
    for r in a_rows:
        by_pos[r["noop_pos"]].append(r)
    A = {}
    for pos in sorted(by_pos):
        rows = by_pos[pos]
        for cond in ("buggy", "fixed", "all"):
            sub = rows if cond == "all" else [r for r in rows if r["condition"] == cond]
            if not sub:
                continue
            ed = [r["edit_minus_noop"] for r in sub]
            A[f"pos{pos}_{cond}"] = {
                "n": len(sub),
                "noop_argmax_rate": sum(r["is_noop"] for r in sub) / len(sub),
                "mean_edit_minus_noop": statistics.mean(ed),
                "edit_minus_noop_ci95": _ci95(ed),
                "mean_noop_rank": statistics.mean(r["noop_rank"] for r in sub),
                "argmax_dist": dict(sorted(
                    {a: sum(r["argmax"] == a for r in sub) for a in ACTIONS5}.items())),
            }
    Csum = {}
    for order in (["edit", "noop"], ["noop", "edit"]):
        key = "_".join(order)
        for cond in ("buggy", "fixed", "all"):
            sub = [r for r in c_rows if r["order"] == order
                   and (cond == "all" or r["condition"] == cond)]
            if not sub:
                continue
            ed = [r["edit_minus_noop"] for r in sub]
            Csum[f"{key}_{cond}"] = {
                "n": len(sub),
                "noop_argmax_rate": sum(r["is_noop"] for r in sub) / len(sub),
                "mean_edit_minus_noop": statistics.mean(ed),
            }
    overall_noop = sum(r["is_noop"] for r in a_rows) / len(a_rows)
    summary = {
        "model": DEFAULT_MODEL, "dataset": f"{n_tasks} toy tasks (data/tasks)",
        "variant": variant_name,
        "single_token_after_Action": {n: {"id": st[n][0], "n_tokens": st[n][1],
                                          "single": single[n]} for n in ACTIONS5},
        "experiment_A_by_noop_position": A,
        "experiment_A_overall_noop_argmax_rate": overall_noop,
        "experiment_C_binary": Csum,
        "headline": (
            f"noop argmax rate across all positions = {overall_noop:.3f}; "
            "see per-position breakdown."
        ),
    }
    (OUT / "qwen_action_order_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "qwen_binary_edit_noop_summary.json").write_text(json.dumps(
        {"model": DEFAULT_MODEL, "dataset": f"{n_tasks} toy tasks", "binary": Csum},
        indent=2))
    print("[aoc] summary:", json.dumps({
        "overall_noop_rate": overall_noop,
        "noop_rate_by_pos": {p: A[f"pos{p}_all"]["noop_argmax_rate"]
                             for p in range(5) if f"pos{p}_all" in A},
        "binary": {k: v["noop_argmax_rate"] for k, v in Csum.items() if k.endswith("_all")},
    }, indent=2), flush=True)


def make_figures(a_rows, c_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[aoc] matplotlib unavailable, skipping figures: {e}")
        return
    from collections import defaultdict
    import statistics
    pos = list(range(5))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for cond, color in (("buggy", "tab:red"), ("fixed", "tab:green")):
        rate = []
        margin = []
        for p in pos:
            sub = [r for r in a_rows if r["noop_pos"] == p and r["condition"] == cond]
            rate.append(sum(r["is_noop"] for r in sub) / len(sub) if sub else 0)
            margin.append(statistics.mean(r["edit_minus_noop"] for r in sub) if sub else 0)
        ax[0].plot(pos, rate, "o-", label=cond, color=color)
        ax[1].plot(pos, margin, "o-", label=cond, color=color)
    ax[0].set(xlabel="noop list position (0=first)", ylabel="noop argmax rate",
              title="Action-order control: noop rate by position", ylim=(-0.02, 1.02))
    ax[1].set(xlabel="noop list position (0=first)", ylabel="mean (edit - noop) logit",
              title="edit - noop margin by position")
    for a in ax:
        a.axhline(0, color="gray", lw=0.5)
        a.legend()
        a.set_xticks(pos)
    fig.tight_layout()
    fig.savefig(FIGS / "action_order_control_qwen.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["edit,noop", "noop,edit"]
    for cond, color in (("buggy", "tab:red"), ("fixed", "tab:green")):
        rates = []
        for order in (["edit", "noop"], ["noop", "edit"]):
            sub = [r for r in c_rows if r["order"] == order and r["condition"] == cond]
            rates.append(sum(r["is_noop"] for r in sub) / len(sub) if sub else 0)
        ax.plot(labels, rates, "o-", label=cond, color=color)
    ax.set(ylabel="noop argmax rate", title="Binary edit/noop control", ylim=(-0.02, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "binary_edit_noop_control_qwen.png", dpi=150)
    plt.close(fig)
    print(f"[aoc] wrote figures to {FIGS}/", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--run-local", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variant", default="code_tests")
    args = ap.parse_args()
    if args.run_local:
        run(args.limit, args.variant)
        return 0
    # build-only: emit the prompt manifest, no model
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for task in iter_tasks():
        n += 1
    print(f"[aoc] build-only: {n} tasks x 2 conditions x "
          f"{len(cyclic_action_orders(ACTIONS5))} orders available. "
          "Use --run-local to score on Qwen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
