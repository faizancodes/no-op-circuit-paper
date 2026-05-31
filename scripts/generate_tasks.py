#!/usr/bin/env python3
"""Generate paired buggy/fixed tasks via LLM + execution-based validation.

Usage examples:

  # Pilot run: 1 archetype, 2 candidates (cheap smoke for the gen pipeline)
  python scripts/generate_tasks.py --pilot

  # Generate N candidates for a specific archetype
  python scripts/generate_tasks.py --archetype off_by_one_loop --n 5

  # Sweep all archetypes with N candidates each
  python scripts/generate_tasks.py --sweep --per-archetype 5

  # Limit total wall-clock attempts (defends against API/budget runaways)
  python scripts/generate_tasks.py --sweep --per-archetype 5 --max-attempts 80
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

import yaml

from no_op_circuit.config import DATA_DIR, TASKS_DIR
from no_op_circuit.dataset.generator import generate_candidate, load_archetypes
from no_op_circuit.dataset.materialize import materialize
from no_op_circuit.dataset.validator import short_summary, validate_candidate


_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)


def _collect_seen_function_names() -> dict[str, list[str]]:
    """Per archetype, list of primary function names already on disk."""
    out: dict[str, list[str]] = {}
    if not TASKS_DIR.exists():
        return out
    for child in sorted(TASKS_DIR.iterdir()):
        meta_path = child / "meta.yaml"
        if not meta_path.is_file():
            continue
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        archetype = meta.get("source_archetype") or "_unknown"
        primary = meta.get("primary_file")
        if not primary:
            continue
        src_path = child / "buggy" / primary
        if not src_path.is_file():
            continue
        src = src_path.read_text(encoding="utf-8")
        m = _DEF_RE.search(src)
        if m:
            out.setdefault(archetype, []).append(m.group(1))
    return out


GEN_LOG = DATA_DIR / "_generation_log.jsonl"


def _log_line(entry: dict) -> None:
    GEN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with GEN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def _primary_function_name(candidate: dict) -> str | None:
    primary = candidate.get("primary_file")
    if not primary:
        return None
    for f in candidate.get("buggy_files", []) + candidate.get("fixed_files", []):
        if f.get("path") == primary:
            m = _DEF_RE.search(f.get("content", ""))
            if m:
                return m.group(1)
    return None


def _run_one(
    arch_id: str,
    generator_kwargs: dict,
    *,
    seen_function_names: list[str],
    domain_hint: str | None,
) -> dict:
    """Generate + validate + materialize one candidate. Returns log entry dict."""
    archs = {a.id: a for a in load_archetypes()}
    arch = archs[arch_id]
    instance_id = f"variant-{uuid.uuid4().hex[:6]}"

    entry: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "archetype": arch_id,
        "instance_id": instance_id,
        "domain_hint": domain_hint,
        "status": "pending",
    }
    t0 = time.time()
    try:
        candidate = generate_candidate(
            arch,
            instance_id=instance_id,
            seen_function_names=seen_function_names,
            domain_hint=domain_hint,
            **generator_kwargs,
        )
        candidate["_archetype"] = arch_id
        entry["proposed_task_id"] = candidate.get("task_id")
        entry["function_name"] = _primary_function_name(candidate)
        # C6: pin the snapshot the OpenRouter alias actually routed to so
        # future runs are reproducible from the log alone.
        if "_resolved_model" in candidate:
            entry["resolved_model"] = candidate["_resolved_model"]
        if "_response_id" in candidate:
            entry["response_id"] = candidate["_response_id"]
        gen_seconds = time.time() - t0
        entry["gen_seconds"] = round(gen_seconds, 2)

        result = validate_candidate(candidate)
        entry["validation"] = short_summary(result)
        if not result.valid:
            entry["status"] = "rejected"
            entry["reject_reason"] = result.reason
            return entry

        out_dir = materialize(candidate, result)
        entry["status"] = "accepted"
        entry["task_dir"] = str(out_dir.relative_to(TASKS_DIR.parent))
        return entry
    except Exception as exc:  # noqa: BLE001
        entry["status"] = "error"
        entry["error_type"] = type(exc).__name__
        entry["error"] = str(exc)[:400]
        entry["traceback"] = traceback.format_exc().splitlines()[-6:]
        return entry


def _summarize(entries: list[dict]) -> None:
    by_status: dict[str, int] = {}
    by_arch_accept: dict[str, int] = {}
    by_arch_total: dict[str, int] = {}
    reject_reasons: dict[str, int] = {}
    total_gen_s = 0.0
    for e in entries:
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
        a = e.get("archetype", "?")
        by_arch_total[a] = by_arch_total.get(a, 0) + 1
        if e["status"] == "accepted":
            by_arch_accept[a] = by_arch_accept.get(a, 0) + 1
        if e["status"] == "rejected":
            r = e.get("reject_reason", "unknown")[:60]
            reject_reasons[r] = reject_reasons.get(r, 0) + 1
        total_gen_s += float(e.get("gen_seconds", 0) or 0)

    print("\n=== generation summary ===")
    print(f"  total attempts : {len(entries)}")
    for k, v in sorted(by_status.items()):
        print(f"    {k:<10} {v}")
    print(f"  accept rate    : {by_status.get('accepted', 0)/max(len(entries),1):.0%}")
    print(f"  LLM seconds    : {total_gen_s:.1f}s")
    print("  per archetype  :")
    for a in sorted(by_arch_total):
        ok = by_arch_accept.get(a, 0)
        total = by_arch_total[a]
        print(f"    {a:<28} {ok}/{total}")
    if reject_reasons:
        print("  top reject reasons:")
        for r, n in sorted(reject_reasons.items(), key=lambda kv: -kv[1])[:6]:
            print(f"    {n:>3}  {r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archetype", help="Single archetype to generate.")
    p.add_argument("--sweep", action="store_true", help="Sweep all archetypes.")
    p.add_argument("--pilot", action="store_true",
                   help="Cheap pipeline smoke: one archetype, two candidates.")
    p.add_argument("--n", type=int, default=3,
                   help="With --archetype: number of candidates to attempt.")
    p.add_argument("--per-archetype", type=int, default=5,
                   help="With --sweep: candidates per archetype.")
    p.add_argument("--max-attempts", type=int, default=None,
                   help="Hard cap on total attempts across the run.")
    p.add_argument("--model", default=None,
                   help="OpenRouter model slug; defaults to module setting.")
    p.add_argument("--temperature", type=float, default=0.85)
    args = p.parse_args(argv)

    gen_kwargs = {"temperature": args.temperature}
    if args.model:
        gen_kwargs["model"] = args.model

    archetypes_full = load_archetypes()
    archetypes = [a.id for a in archetypes_full]
    domains_by_arch = {a.id: list(a.domains) for a in archetypes_full}

    plan: list[str] = []
    if args.pilot:
        plan = [archetypes[0]] * 2
    elif args.archetype:
        if args.archetype not in archetypes:
            print(f"unknown archetype: {args.archetype}. Known: {archetypes}", file=sys.stderr)
            return 2
        plan = [args.archetype] * args.n
    elif args.sweep:
        for a in archetypes:
            plan.extend([a] * args.per_archetype)
    else:
        p.print_help()
        return 2

    if args.max_attempts is not None:
        plan = plan[: args.max_attempts]

    seen_by_arch: dict[str, list[str]] = _collect_seen_function_names()
    domain_cursor: dict[str, int] = {}

    print(f"plan: {len(plan)} attempts across {len(set(plan))} archetypes", flush=True)
    print(f"gen log: {GEN_LOG}", flush=True)
    entries: list[dict] = []
    for i, arch_id in enumerate(plan, 1):
        domains = domains_by_arch.get(arch_id, [])
        domain_hint = None
        if domains:
            cursor = domain_cursor.get(arch_id, 0)
            domain_hint = domains[cursor % len(domains)]
            domain_cursor[arch_id] = cursor + 1
        seen = list(seen_by_arch.get(arch_id, []))

        entry = _run_one(
            arch_id,
            gen_kwargs,
            seen_function_names=seen,
            domain_hint=domain_hint,
        )
        if entry["status"] == "accepted" and entry.get("function_name"):
            seen_by_arch.setdefault(arch_id, []).append(entry["function_name"])
        _log_line(entry)
        entries.append(entry)
        marker = {"accepted": "✓", "rejected": "✗", "error": "!"}.get(entry["status"], "?")
        line = (
            f"  [{i:>3}/{len(plan)}] {marker} {arch_id:<28} "
            f"{entry.get('proposed_task_id') or entry.get('error_type') or '-'}"
        )
        if domain_hint:
            line += f"  [{domain_hint}]"
        if entry.get("reject_reason"):
            line += f"  ← {entry['reject_reason'][:60]}"
        print(line, flush=True)

    _summarize(entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
