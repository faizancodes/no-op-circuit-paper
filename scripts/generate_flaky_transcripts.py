#!/usr/bin/env python3
"""Generate `data/tasks/<task>/fixed/tests_output_flaky.txt` for every task.

The flaky transcript is the existing passing `fixed/tests_output.txt` PLUS one
synthetic, unrelated, generic test failure spliced in. The fake failure must
NOT correspond to the task's bug archetype — it should look like the kind of
flaky CI failure you'd get from network timeouts, clock skew, or disk quotas.

Validation:
  - Resulting transcript contains at least one `FAILED ` line.
  - The fake failed test name does NOT appear in the task's existing
    fixed/test_*.py file (otherwise the flaky variant would be a duplicate of
    the buggy variant).
  - The fake failed test name does NOT reference any token from the task's
    meta.yaml::issue_text or primary_file (otherwise it would leak the bug).

We use OpenRouter's claude-sonnet-4.5 (same model used for task generation).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from no_op_circuit.config import TASKS_DIR
from no_op_circuit.llm import chat, parse_json


_GENERIC_FAILURE_FLAVORS = [
    "network timeout / DNS",
    "disk quota / temp file cleanup",
    "system clock skew",
    "race condition in a thread-pool stub",
    "an environment-variable lookup",
    "a stale Docker socket",
    "a flaky randomized seed",
    "a flaky timezone conversion",
    "Unicode normalization on a CI runner",
]


_PROMPT = """Insert ONE synthetic, unrelated, flaky CI test failure into the \
existing pytest output below. The flaky failure must look like the kind of \
non-deterministic CI noise that has nothing to do with the function under \
test.

Hard constraints:
- The flaky test name MUST be one of:
    test_network_timeout, test_disk_quota, test_clock_skew, test_temp_cleanup,
    test_thread_pool_race, test_env_var_lookup, test_docker_socket,
    test_random_seed, test_tz_conversion, test_unicode_normalization
- The flaky test name MUST NOT appear in the user-provided existing test file.
- The flaky failure MUST NOT mention any names from the user-provided issue \
text or primary file name.
- The output must keep the original passing tests and append ONE flaky \
failure, then update the pytest summary line to reflect 1 failure + N passed.
- Pick a failure FLAVOR from this list (rotate variety): {flavors}

Existing passing transcript:
```
{passing_transcript}
```

Existing tests in the task (do not collide):
```
{test_file_content}
```

Issue text (do not echo names from here):
```
{issue_text}
```

Primary file: `{primary_file}`

Return STRICT JSON with one key:
{{
  "flaky_transcript": "<the full new transcript including the flaky failure and a corrected summary line>"
}}"""


_ALLOWED_FAILURE_NAMES = {
    "test_network_timeout",
    "test_disk_quota",
    "test_clock_skew",
    "test_temp_cleanup",
    "test_thread_pool_race",
    "test_env_var_lookup",
    "test_docker_socket",
    "test_random_seed",
    "test_tz_conversion",
    "test_unicode_normalization",
}


def _find_failed_names(transcript: str) -> list[str]:
    """Return all 'FAILED <path>::<name>' (or 'FAILED <name>') test names."""
    out: list[str] = []
    for line in transcript.split("\n"):
        s = line.strip()
        # patterns: "FAILED test_x.py::test_foo - ..." or "FAILED test_foo ..."
        m = re.search(r"FAILED\s+(?:[\w./]+::)?(test_[A-Za-z0-9_]+)", s)
        if m:
            out.append(m.group(1))
    return out


def _existing_test_names(test_src: str) -> set[str]:
    return set(re.findall(r"def\s+(test_[A-Za-z0-9_]+)\s*\(", test_src))


def _scrub(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text.lower()))


def _validate(
    transcript: str,
    *,
    existing_tests: set[str],
    issue_tokens: set[str],
    primary_file_tokens: set[str],
    passing_transcript: str,
) -> str | None:
    if not transcript.strip():
        return "empty transcript"
    failed = _find_failed_names(transcript)
    if not failed:
        return "no FAILED line in output"
    new_fails = [n for n in failed if n not in existing_tests]
    if not new_fails:
        return f"flaky failure shares name with existing test: {failed}"
    for n in new_fails:
        if n not in _ALLOWED_FAILURE_NAMES:
            return f"flaky failure name {n!r} not in allow-list"
    # Only consider tokens introduced by the LLM (not those already in the
    # passing transcript, which carries issue-related test names by design).
    existing_tokens = _scrub(passing_transcript)
    new_tokens = _scrub(transcript) - existing_tokens
    forbidden = (issue_tokens | primary_file_tokens) - _scrub(
        " ".join(existing_tests) + " " + " ".join(_ALLOWED_FAILURE_NAMES)
    )
    leaks = new_tokens & forbidden
    if leaks:
        return f"flaky-only tokens leak issue/primary names: {sorted(leaks)[:5]}"
    return None


def _load_task_inputs(task_dir: Path) -> dict[str, Any]:
    meta = yaml.safe_load((task_dir / "meta.yaml").read_text(encoding="utf-8"))
    passing = (task_dir / "fixed" / "tests_output.txt").read_text(encoding="utf-8")
    test_files = list((task_dir / "fixed").glob("test_*.py"))
    if not test_files:
        raise RuntimeError(f"no test_*.py in {task_dir / 'fixed'}")
    test_src = test_files[0].read_text(encoding="utf-8")
    return {
        "task_id": meta["task_id"],
        "issue_text": meta["issue_text"],
        "primary_file": meta["primary_file"],
        "passing_transcript": passing,
        "test_file_content": test_src,
        "existing_tests": _existing_test_names(test_src),
    }


_TEMPLATE_FAILURES = {
    "test_network_timeout": (
        "FAILED test_ci_runner.py::test_network_timeout - "
        "ConnectionError: HTTPSConnectionPool(host='cdn.example.com', port=443): "
        "Read timed out. (read timeout=2)"
    ),
    "test_disk_quota": (
        "FAILED test_ci_runner.py::test_disk_quota - "
        "OSError: [Errno 28] No space left on device"
    ),
    "test_clock_skew": (
        "FAILED test_ci_runner.py::test_clock_skew - "
        "AssertionError: NTP drift exceeded threshold (3.7s > 1.0s)"
    ),
    "test_temp_cleanup": (
        "FAILED test_ci_runner.py::test_temp_cleanup - "
        "PermissionError: [Errno 13] Permission denied: '/tmp/pytest-xyz'"
    ),
    "test_thread_pool_race": (
        "FAILED test_ci_runner.py::test_thread_pool_race - "
        "RuntimeError: ThreadPoolExecutor queue closed unexpectedly"
    ),
    "test_env_var_lookup": (
        "FAILED test_ci_runner.py::test_env_var_lookup - "
        "KeyError: 'CI_RUNNER_TAG'"
    ),
    "test_docker_socket": (
        "FAILED test_ci_runner.py::test_docker_socket - "
        "FileNotFoundError: [Errno 2] No such file or directory: '/var/run/docker.sock'"
    ),
    "test_random_seed": (
        "FAILED test_ci_runner.py::test_random_seed - "
        "AssertionError: random.seed unset across worker boundary"
    ),
    "test_tz_conversion": (
        "FAILED test_ci_runner.py::test_tz_conversion - "
        "ValueError: zoneinfo backend unavailable on CI runner"
    ),
    "test_unicode_normalization": (
        "FAILED test_ci_runner.py::test_unicode_normalization - "
        "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0"
    ),
}


def _deterministic_flaky(inputs: dict[str, Any]) -> str:
    """Deterministic fallback: pick an allowed flaky name not in existing tests
    and append a single FAILED block + corrected summary line to the passing
    transcript."""
    name = next(
        n for n in sorted(_ALLOWED_FAILURE_NAMES) if n not in inputs["existing_tests"]
    )
    passing = inputs["passing_transcript"].rstrip()
    # Extract original pass count from summary line if present.
    m = re.search(r"(\d+)\s+passed", passing)
    n_passed = int(m.group(1)) if m else 0
    failure_line = _TEMPLATE_FAILURES[name]
    body = (
        passing.rsplit("\n", 1)[0]
        if "passed" in passing.splitlines()[-1]
        else passing
    )
    # If the last line was a summary, drop it; replace with our own.
    new_summary = f"=========== 1 failed, {n_passed} passed in 0.04s ==========="
    return f"{body}\n=================================== FAILURES ===================================\n{failure_line}\n=========================== short test summary info ============================\n{failure_line}\n{new_summary}\n"


def _generate_one(inputs: dict[str, Any], *, flavor_idx: int) -> str:
    flavors = ", ".join(_GENERIC_FAILURE_FLAVORS)
    prompt = _PROMPT.format(
        flavors=flavors,
        passing_transcript=inputs["passing_transcript"],
        test_file_content=inputs["test_file_content"],
        issue_text=inputs["issue_text"],
        primary_file=inputs["primary_file"],
    )
    result = chat(
        messages=[
            {
                "role": "system",
                "content": "You author synthetic pytest output fragments for "
                "mechanistic-interpretability research. Output STRICT JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    payload = parse_json(result.text)
    return payload["flaky_transcript"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retries", type=int, default=2, help="LLM retries per task on validation failure.")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate flaky transcripts even when files already exist.")
    p.add_argument("--limit", type=int, default=0, help="Cap tasks (0 = all).")
    args = p.parse_args(argv)

    tasks = sorted(p for p in TASKS_DIR.iterdir() if (p / "meta.yaml").is_file())
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"tasks: {len(tasks)}", flush=True)
    written = skipped = failed = 0
    samples: list[str] = []

    for i, task_dir in enumerate(tasks, 1):
        out_path = task_dir / "fixed" / "tests_output_flaky.txt"
        if out_path.is_file() and not args.overwrite:
            skipped += 1
            print(f"[{i:>3}/{len(tasks)}] · {task_dir.name} → skip (exists)", flush=True)
            continue

        inputs = _load_task_inputs(task_dir)
        issue_tokens = _scrub(inputs["issue_text"])
        primary_file_tokens = _scrub(inputs["primary_file"].replace(".py", ""))

        attempt_results: list[tuple[str, str | None]] = []
        for attempt in range(args.retries + 1):
            try:
                transcript = _generate_one(inputs, flavor_idx=i + attempt)
            except Exception as exc:  # noqa: BLE001
                attempt_results.append(("", f"LLM error: {exc!r}"))
                continue
            err = _validate(
                transcript,
                existing_tests=inputs["existing_tests"],
                issue_tokens=issue_tokens,
                primary_file_tokens=primary_file_tokens,
                passing_transcript=inputs["passing_transcript"],
            )
            attempt_results.append((transcript, err))
            if err is None:
                break

        good = next((t for t, e in attempt_results if e is None), None)
        if good is None:
            # Deterministic fallback — guarantees a valid transcript.
            try:
                good = _deterministic_flaky(inputs)
                err = _validate(
                    good,
                    existing_tests=inputs["existing_tests"],
                    issue_tokens=issue_tokens,
                    primary_file_tokens=primary_file_tokens,
                    passing_transcript=inputs["passing_transcript"],
                )
                if err is not None:
                    raise RuntimeError(err)
                print(f"[{i:>3}/{len(tasks)}] ⚙ {task_dir.name} → DETERMINISTIC FALLBACK", flush=True)
            except Exception as exc:
                failed += 1
                err_msg = attempt_results[-1][1] if attempt_results else "no attempts"
                print(f"[{i:>3}/{len(tasks)}] ✗ {task_dir.name} → FAIL: {err_msg} | fallback: {exc}", flush=True)
                continue

        out_path.write_text(good if good.endswith("\n") else good + "\n", encoding="utf-8")
        written += 1
        if len(samples) < 3:
            samples.append(f"--- {task_dir.name} ---\n{good}\n")
        print(f"[{i:>3}/{len(tasks)}] ✓ {task_dir.name}", flush=True)

    print()
    print(f"summary: written={written}  skipped={skipped}  failed={failed}  / {len(tasks)}")
    if samples:
        print("\n=== samples (first 3 written) ===")
        for s in samples:
            print(s)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
