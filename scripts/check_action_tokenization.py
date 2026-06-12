#!/usr/bin/env python3
"""Action-tokenization audit (paper Phase 1).

Resolves whether the five action names are single-token under each model's
tokenizer *in the exact scored context* (the first token emitted after the
final ``Action: `` of the prompt, resolved by diffing prefix vs prefix+name --
the same rule modal_app/cache_activations.py uses to read action logits).

Also (Phase 2) searches single-token synonym candidates for a DeepSeek
five-action vocabulary.

Outputs:
  results/tokenization/action_tokenization.json
  results/tokenization/action_tokenization.md
  results/tokenization/deepseek_single_token_action_candidates.json
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Qwen2.5-Coder size ladder. The family shares one tokenizer across sizes, so a
# single row is sufficient in principle; we list each size for an explicit
# per-size confirmation (and to fail loudly if any slug is wrong/gated).
MODELS = {
    "qwen_0.5b": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "qwen_1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",  # paper baseline
    "qwen_3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen_7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen_14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "qwen_32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
}
# Not in the Qwen-only roster; the Phase-2 synonym-search block below skips
# cleanly when this key is absent from `audit`.
DEEPSEEK_AUDIT_KEY = "deepseek_1.3b"
ACTIONS = ["view", "grep", "test", "edit", "noop"]
EXTRA = ["done", "skip"]
LABELS = ["A", "B", "C", "D", "E"]
SCORE_PREFIX = "Action: "  # the exact suffix the renderer ends with

# Phase 2 candidate synonyms per action meaning (single-token search for DeepSeek)
CANDIDATES = {
    "view": ["view", "look", "open", "read", "show"],
    "search": ["grep", "find", "search", "scan", "seek"],
    "test": ["test", "run", "check"],
    "edit": ["edit", "fix", "patch", "change"],
    "noop": ["noop", "done", "skip", "stop", "finish", "pass"],
}


def first_new(tok, prefix: str, name: str):
    p = tok.encode(prefix, add_special_tokens=False)
    f = tok.encode(prefix + name, add_special_tokens=False)
    i = 0
    while i < len(p) and i < len(f) and p[i] == f[i]:
        i += 1
    tail = f[i:]
    return tail, len(tail)


def main():
    from transformers import AutoTokenizer

    out = Path("results/tokenization")
    out.mkdir(parents=True, exist_ok=True)
    audit: dict = {}
    for tag, slug in MODELS.items():
        try:
            tok = AutoTokenizer.from_pretrained(slug)
        except Exception as e:  # noqa: BLE001
            audit[tag] = {"error": repr(e)[:200]}
            print(f"[{tag}] tokenizer load FAILED: {repr(e)[:120]}")
            continue
        rec = {}
        for name in ACTIONS + EXTRA + LABELS:
            tail, n = first_new(tok, SCORE_PREFIX, name)
            rec[name] = {
                "scored_ids": tail,
                "scored_decoded": [tok.decode([t]) for t in tail],
                "n_tokens": n,
                "single_token": n == 1,
                "raw_ids": tok.encode(name, add_special_tokens=False),
            }
        audit[tag] = {"model": slug, "tokens": rec,
                      "actions_all_single_token": all(rec[a]["single_token"] for a in ACTIONS)}
        miss = [a for a in ACTIONS if not rec[a]["single_token"]]
        print(f"[{tag}] actions single-token: {audit[tag]['actions_all_single_token']}"
              + (f"  multi-token: {miss}" if miss else ""))

    (out / "action_tokenization.json").write_text(json.dumps(audit, indent=2))

    # Markdown table
    lines = ["# Action-tokenization audit", "",
             "Exact scored form = first token(s) after the final `Action: ` "
             "(diff prefix vs prefix+name), matching the action-logit readout.", "",
             "| model | action | n_tokens | single-token? | scored decoded |",
             "|---|---|---:|---|---|"]
    for tag in MODELS:
        if "tokens" not in audit.get(tag, {}):
            lines.append(f"| {tag} | (tokenizer load failed) | | | |")
            continue
        for a in ACTIONS:
            r = audit[tag]["tokens"][a]
            lines.append(f"| {tag} | {a} | {r['n_tokens']} | "
                         f"{'yes' if r['single_token'] else '**no**'} | "
                         f"`{''.join(r['scored_decoded'])}` |")
    (out / "action_tokenization.md").write_text("\n".join(lines) + "\n")

    # Phase 2: DeepSeek single-token vocabulary search
    cand_out = {}
    if "tokens" in audit.get(DEEPSEEK_AUDIT_KEY, {}):
        from transformers import AutoTokenizer as AT
        dtok = AT.from_pretrained(MODELS[DEEPSEEK_AUDIT_KEY])
        chosen = {}
        for meaning, opts in CANDIDATES.items():
            single = []
            for w in opts:
                _, n = first_new(dtok, SCORE_PREFIX, w)
                if n == 1:
                    single.append(w)
            cand_out[meaning] = {"options": opts, "single_token_options": single}
            chosen[meaning] = single[0] if single else None
        cand_out["chosen_vocab"] = chosen
        cand_out["valid_vocab_exists"] = all(v is not None for v in chosen.values())
        (out / "deepseek_single_token_action_candidates.json").write_text(json.dumps(cand_out, indent=2))
        print(f"[deepseek] single-token vocab exists: {cand_out['valid_vocab_exists']}  chosen={chosen}")

    print(f"wrote {out}/action_tokenization.json + .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
