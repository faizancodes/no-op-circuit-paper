#!/usr/bin/env python3
"""Summary statistics for the materialized task corpus.

Prints:
  - task count
  - archetype distribution
  - per-task: source LoC, test count, buggy/fixed exit
  - flags any task whose meta.yaml looks malformed
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

import yaml

from no_op_circuit.config import TASKS_DIR


_FAILED_RE = re.compile(r"^(\d+)\s+failed\b", re.MULTILINE)
_PASSED_RE = re.compile(r"(\d+)\s+passed\s+in\s+\d", re.IGNORECASE)
_TEST_DEF_RE = re.compile(r"^\s*def\s+test_\w+", re.MULTILINE)


def _count_loc(content: str) -> int:
    return sum(1 for line in content.splitlines() if line.strip() and not line.strip().startswith("#"))


def _summarize_test_output(out: str) -> tuple[int, int]:
    f_match = _FAILED_RE.search(out)
    p_match = _PASSED_RE.search(out)
    return (
        int(f_match.group(1)) if f_match else 0,
        int(p_match.group(1)) if p_match else 0,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--show-each", action="store_true", help="Print one row per task.")
    args = p.parse_args(argv)

    rows: list[dict] = []
    arch_counter: Counter[str] = Counter()
    issues: list[str] = []

    for child in sorted(TASKS_DIR.iterdir()):
        meta_path = child / "meta.yaml"
        if not meta_path.is_file():
            continue
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"{child.name}: meta.yaml parse error: {exc}")
            continue

        primary = meta.get("primary_file", "")
        buggy_src = (child / "buggy" / primary).read_text(encoding="utf-8") if primary and (child / "buggy" / primary).is_file() else ""
        fixed_src = (child / "fixed" / primary).read_text(encoding="utf-8") if primary and (child / "fixed" / primary).is_file() else ""
        buggy_out = (child / "buggy" / "tests_output.txt").read_text(encoding="utf-8") if (child / "buggy" / "tests_output.txt").is_file() else ""
        fixed_out = (child / "fixed" / "tests_output.txt").read_text(encoding="utf-8") if (child / "fixed" / "tests_output.txt").is_file() else ""

        # Try to locate the test file under either condition.
        test_files = [f for f in (child / "buggy").iterdir() if f.name.startswith("test_") and f.suffix == ".py"]
        test_src = test_files[0].read_text(encoding="utf-8") if test_files else ""
        n_tests = len(_TEST_DEF_RE.findall(test_src))

        b_failed, b_passed = _summarize_test_output(buggy_out)
        f_failed, f_passed = _summarize_test_output(fixed_out)

        archetype = meta.get("source_archetype", "(hand-authored)")
        arch_counter[archetype] += 1

        if b_failed == 0:
            issues.append(f"{child.name}: buggy_test_output reports 0 failures — wrong label?")
        if f_failed > 0:
            issues.append(f"{child.name}: fixed_test_output reports {f_failed} failures — wrong label?")

        rows.append({
            "task_id": child.name,
            "archetype": archetype,
            "buggy_loc": _count_loc(buggy_src),
            "fixed_loc": _count_loc(fixed_src),
            "n_tests": n_tests,
            "buggy_summary": f"{b_failed}f/{b_passed}p",
            "fixed_summary": f"{f_failed}f/{f_passed}p",
        })

    print(f"tasks: {len(rows)}")
    print(f"archetype distribution:")
    for arch, n in arch_counter.most_common():
        print(f"  {arch:<30} {n}")
    if rows:
        avg_buggy = sum(r["buggy_loc"] for r in rows) / len(rows)
        avg_tests = sum(r["n_tests"] for r in rows) / len(rows)
        print(f"\naverages: buggy_loc={avg_buggy:.1f}  n_tests={avg_tests:.1f}")
    if issues:
        print("\nissues:")
        for s in issues:
            print(f"  - {s}")
    if args.show_each:
        print("\nper-task:")
        print(f"  {'task_id':<40} {'archetype':<25} {'loc':<6} {'tests':<6} {'buggy':<8} {'fixed':<8}")
        for r in rows:
            print(
                f"  {r['task_id']:<40} {r['archetype']:<25} "
                f"{r['buggy_loc']:<6} {r['n_tests']:<6} "
                f"{r['buggy_summary']:<8} {r['fixed_summary']:<8}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
