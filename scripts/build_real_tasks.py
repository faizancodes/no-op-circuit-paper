#!/usr/bin/env python3
"""Ingest the first 30 SWE-bench_Verified tasks into our paired-task on-disk
schema so the existing build_prompt / cache_dataset / monitor pipeline can
process them without modification.

Per task we synthesise the SAME shape we use for toy tasks:

    data/real_tasks/<instance_id>/
      meta.yaml
      buggy/
        <basename>.py        ← file at base_commit, 80-line window around the hunk
        tests_output.txt     ← synthesised FAILED-line transcript from FAIL_TO_PASS
      fixed/
        <basename>.py        ← same window with the gold patch applied in-memory
        tests_output.txt     ← synthesised all-pass transcript

We do NOT run pytest / Docker. The transcripts are formatted to look real
but their content is derived directly from SWE-bench's FAIL_TO_PASS /
PASS_TO_PASS lists. That's intentional: the monitor reads the residual
stream at a position WHERE THE PROMPT IS THE INPUT, so all that matters is
that the prompt presents (real source) + (a syntactically real-looking test
transcript carrying the right FAIL/PASS signal).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "data" / "real_tasks"


_HUNK_HEADER = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


def _parse_patch(patch: str) -> list[tuple[str, list[dict]]]:
    files: list[tuple[str, list[dict]]] = []
    current_path: str | None = None
    hunks: list[dict] = []
    cur_hunk: dict | None = None
    in_hunk = False
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            if current_path is not None:
                files.append((current_path, hunks))
            m = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            current_path = m.group(2) if m else None
            hunks = []
            cur_hunk = None
            in_hunk = False
            continue
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        m = _HUNK_HEADER.match(line)
        if m and current_path is not None:
            cur_hunk = {
                "old_start": int(m.group(1)),
                "old_count": int(m.group(2)) if m.group(2) else 1,
                "new_start": int(m.group(3)),
                "new_count": int(m.group(4)) if m.group(4) else 1,
                "body": [],
            }
            hunks.append(cur_hunk)
            in_hunk = True
            continue
        if in_hunk and cur_hunk is not None:
            if line.startswith((" ", "+", "-", "\\")):
                cur_hunk["body"].append(line)
            else:
                in_hunk = False
    if current_path is not None:
        files.append((current_path, hunks))
    return files


def _biggest_py_hunk(files: list[tuple[str, list[dict]]]) -> tuple[str, dict] | None:
    best: tuple[str, dict] | None = None
    best_size = 0
    for path, hunks in files:
        if not path.endswith(".py"):
            continue
        if "/test" in path or path.startswith("test"):
            continue
        for h in hunks:
            n = sum(1 for ln in h["body"] if ln.startswith(("+", "-")))
            if n > best_size:
                best_size = n
                best = (path, h)
    if best is None:
        for path, hunks in files:
            if not path.endswith(".py"):
                continue
            for h in hunks:
                n = sum(1 for ln in h["body"] if ln.startswith(("+", "-")))
                if n > best_size:
                    best_size = n
                    best = (path, h)
    return best


def _fetch_file_at_commit(repo: str, commit: str, path: str, timeout: int = 20) -> str:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    req = Request(url, headers={"User-Agent": "no-op-circuit/0.1"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _window(lines: list[str], hunk: dict, n_context: int = 40) -> tuple[list[str], int]:
    start = max(0, hunk["old_start"] - 1 - n_context)
    end = min(len(lines), hunk["old_start"] - 1 + hunk["old_count"] + n_context)
    offset = (hunk["old_start"] - 1) - start
    return lines[start:end], offset


def _apply_hunk_in_window(window_lines: list[str], offset: int, hunk: dict) -> list[str]:
    new_region: list[str] = []
    for ln in hunk["body"]:
        if ln.startswith("+"):
            new_region.append(ln[1:])
        elif ln.startswith(" "):
            new_region.append(ln[1:])
        elif ln.startswith("-"):
            continue
    return window_lines[:offset] + new_region + window_lines[offset + hunk["old_count"]:]


def _synth_buggy_transcript(fail_to_pass: list[str], pass_to_pass: list[str]) -> str:
    n_fail = len(fail_to_pass)
    n_pass = len(pass_to_pass)
    failures = []
    for name in fail_to_pass[:3]:
        failures.append(
            f"FAILED {name} - AssertionError: behaviour does not match expected output"
        )
    if n_fail > 3:
        failures.append(f"... and {n_fail - 3} more failures (truncated)")
    return (
        "=================================== FAILURES ===================================\n"
        + "\n".join(failures)
        + "\n=========================== short test summary info ============================\n"
        + "\n".join(failures)
        + f"\n========================= {n_fail} failed, {n_pass} passed in 0.04s =========================\n"
    )


def _synth_fixed_transcript(fail_to_pass: list[str], pass_to_pass: list[str]) -> str:
    n_pass = len(fail_to_pass) + len(pass_to_pass)
    return f"========================= {n_pass} passed in 0.04s =========================\n"


def _safe_id(instance_id: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", instance_id.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s or not s[0].isalpha():
        s = "task_" + s
    return s


def build_one(task: dict, *, n_context: int = 40) -> tuple[str | None, str | None]:
    instance_id = task["instance_id"]
    repo = task["repo"]
    base_commit = task["base_commit"]
    issue = task.get("problem_statement", "") or ""
    if len(issue) > 1500:
        issue = issue[:1500].rstrip() + "…"

    parsed = _parse_patch(task.get("patch", "") or "")
    best = _biggest_py_hunk(parsed)
    if best is None:
        return None, "no python hunk in patch"
    path, hunk = best

    try:
        full = _fetch_file_at_commit(repo, base_commit, path)
    except HTTPError as e:
        return None, f"file fetch {e.code} {path}"
    except (URLError, TimeoutError) as e:
        return None, f"file fetch error {type(e).__name__} {path}"

    lines = full.split("\n")
    window, offset = _window(lines, hunk, n_context=n_context)
    if offset < 0 or offset + hunk["old_count"] > len(window):
        return None, "hunk offset out of bounds after windowing"

    buggy_content = "\n".join(window).rstrip() + "\n"
    fixed_window = _apply_hunk_in_window(window, offset, hunk)
    fixed_content = "\n".join(fixed_window).rstrip() + "\n"
    if buggy_content == fixed_content:
        return None, "buggy/fixed identical after applying hunk (degenerate)"

    fail_to_pass = task.get("FAIL_TO_PASS") or []
    pass_to_pass = task.get("PASS_TO_PASS") or []
    if isinstance(fail_to_pass, str):
        import json as _json
        try: fail_to_pass = _json.loads(fail_to_pass)
        except Exception: fail_to_pass = [fail_to_pass]
    if isinstance(pass_to_pass, str):
        import json as _json
        try: pass_to_pass = _json.loads(pass_to_pass)
        except Exception: pass_to_pass = [pass_to_pass]
    if not fail_to_pass:
        return None, "FAIL_TO_PASS empty"

    buggy_transcript = _synth_buggy_transcript(fail_to_pass, pass_to_pass)
    fixed_transcript = _synth_fixed_transcript(fail_to_pass, pass_to_pass)

    task_id = _safe_id(instance_id)
    out = OUT_ROOT / task_id
    if out.exists():
        import shutil
        shutil.rmtree(out)
    (out / "buggy").mkdir(parents=True)
    (out / "fixed").mkdir(parents=True)

    basename = Path(path).name
    (out / "buggy" / basename).write_text(buggy_content, encoding="utf-8")
    (out / "fixed" / basename).write_text(fixed_content, encoding="utf-8")
    (out / "buggy" / "tests_output.txt").write_text(buggy_transcript, encoding="utf-8")
    (out / "fixed" / "tests_output.txt").write_text(fixed_transcript, encoding="utf-8")

    test_cmd_target = f"pytest -q {fail_to_pass[0].split('::')[0]}" if fail_to_pass else "pytest -q"
    meta = {
        "task_id": task_id,
        "description": f"SWE-bench_Verified instance {instance_id} (repo: {repo})",
        "issue_text": issue + "\n",
        "primary_file": basename,
        "test_command": test_cmd_target,
        "expected_action": {"buggy": "edit", "fixed": "noop"},
        "source": "swe_bench_verified",
        "swe_bench_instance_id": instance_id,
        "swe_bench_repo": repo,
        "swe_bench_base_commit": base_commit,
        "swe_bench_modified_path": path,
    }
    (out / "meta.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return task_id, None


_REASON_BUCKETS = [
    ("no python hunk",          "no python hunk in patch"),
    ("file fetch HTTP",         "file fetch "),
    ("file fetch error",        "file fetch error"),
    ("hunk out of bounds",      "hunk offset out of bounds"),
    ("degenerate (buggy==fixed)", "buggy/fixed identical"),
    ("FAIL_TO_PASS empty",      "FAIL_TO_PASS empty"),
    ("unhandled exception",     "unhandled "),
]


def _bucketise(err: str) -> str:
    for label, prefix in _REASON_BUCKETS:
        if err.startswith(prefix):
            return label
    return "other"


def main(argv: list[str] | None = None) -> int:
    global OUT_ROOT
    p = argparse.ArgumentParser(description=__doc__)
    # --n-tasks is the new name; --n kept as alias for backward compat.
    p.add_argument("--n-tasks", "--n", type=int, default=500,
                   dest="n_tasks",
                   help="how many of the 500 SWE-bench Verified instances to ingest")
    p.add_argument("--n-context", type=int, default=40)
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT,
                   help="where to write data/real_tasks/<task_id>/ directories")
    args = p.parse_args(argv)

    OUT_ROOT = args.out_dir

    from collections import Counter
    from datasets import load_dataset
    print(f"loading SWE-bench_Verified test split…", flush=True)
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    n_total = len(ds)
    n_target = min(args.n_tasks, n_total)
    print(f"  loaded {n_total} total tasks; targeting first {n_target}", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    written: list[str] = []          # this run (newly created)
    skipped_existing: list[str] = [] # already present (idempotent skip)
    failures: list[tuple[str, str]] = []
    skip_reasons: Counter[str] = Counter()
    for i in range(n_target):
        task = ds[i]
        iid = task["instance_id"]
        candidate_id = _safe_id(iid)
        meta_path = OUT_ROOT / candidate_id / "meta.yaml"
        if meta_path.exists():
            skipped_existing.append(candidate_id)
            print(f"  [{i+1:>3}/{n_target}] ⊙ {iid} → already exists, skip", flush=True)
            continue
        try:
            task_id, err = build_one(task, n_context=args.n_context)
        except Exception as exc:  # noqa: BLE001
            task_id, err = None, f"unhandled {type(exc).__name__}: {exc}"
        if err:
            failures.append((iid, err))
            skip_reasons[_bucketise(err)] += 1
            print(f"  [{i+1:>3}/{n_target}] ✗ {iid} → {err}", flush=True)
        else:
            written.append(task_id)
            print(f"  [{i+1:>3}/{n_target}] ✓ {iid} → {task_id}", flush=True)
        time.sleep(0.05)

    yield_total = len(written) + len(skipped_existing)
    print()
    print(f"=== summary ===")
    print(f"  targeted          : {n_target}")
    print(f"  newly written     : {len(written)}")
    print(f"  already existed   : {len(skipped_existing)}")
    print(f"  failed            : {len(failures)}")
    print(f"  total yield       : {yield_total} / {n_target} "
          f"({100*yield_total/max(n_target,1):.1f}%)")

    if skip_reasons:
        print(f"\n=== skip-reason histogram (this run) ===")
        for reason, n in skip_reasons.most_common():
            print(f"  {n:>3}  {reason}")
        if failures and len(failures) > 0:
            print(f"\nfirst 10 failures (for inspection):")
            for iid, err in failures[:10]:
                print(f"  - {iid}: {err}")

    total_dirs = sum(1 for d in OUT_ROOT.iterdir() if (d / "meta.yaml").exists())
    print(f"\ntotal task dirs in {OUT_ROOT}: {total_dirs}")
    return 0 if yield_total >= 15 else 1


if __name__ == "__main__":
    sys.exit(main())
