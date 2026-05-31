"""Validate a generated TaskCandidate by actually running its tests.

Acceptance criteria (the labels MUST be ground-truth):

  * fixed:  pytest exits 0 (all tests pass)
  * buggy:  pytest exits 1 (at least one test failed)
  * no collection errors (exit 2/3/4) and no "no tests" (exit 5)
  * source files contain no obviously dangerous calls
  * only standard-library imports

Anything else is rejected with a reason string for the gen log.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Python 3.10+ stdlib top-level module names — anything else is rejected as an
# external dependency. Pulled from the official stdlib list; covers Python
# 3.10 - 3.13. (We allow `pytest` because the test file imports it.)
_STDLIB_MODULES = set(sys.stdlib_module_names) | {"pytest", "__future__"}

_DANGEROUS_PATTERNS = [
    re.compile(r"\bsubprocess\.\w+"),
    re.compile(r"\bos\.system\b"),
    re.compile(r"\bos\.popen\b"),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"\b__import__\s*\("),
    re.compile(r"\bopen\s*\([^)]*[\"\']w"),  # writing to disk
    re.compile(r"\bsocket\b"),
    re.compile(r"\burllib\."),
    re.compile(r"\brequests\b"),
    re.compile(r"\bhttpx\b"),
]

_FROM_RE = re.compile(r"^\s*from\s+([\w\.]+)\s+import\b")
_IMPORT_RE = re.compile(r"^\s*import\s+(.+?)(?:\s*#.*)?$")

_PYTEST_TIMEOUT_S = 20


@dataclass
class ValidationResult:
    valid: bool
    reason: str | None = None
    buggy_returncode: int | None = None
    fixed_returncode: int | None = None
    buggy_test_output: str = ""
    fixed_test_output: str = ""


def _safety_scan(source: str) -> str | None:
    for pat in _DANGEROUS_PATTERNS:
        m = pat.search(source)
        if m:
            return f"dangerous pattern: {m.group(0)!r}"
    return None


def _imports_in(source: str) -> list[str]:
    """Return top-level module names imported anywhere in `source`."""
    out: list[str] = []
    for line in source.split("\n"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _FROM_RE.match(s)
        if m:
            out.append(m.group(1).split(".", 1)[0])
            continue
        m = _IMPORT_RE.match(s)
        if m:
            for tok in m.group(1).split(","):
                tok = tok.strip()
                if not tok:
                    continue
                # handle `X as Y`
                tok = tok.split()[0]
                out.append(tok.split(".", 1)[0])
    return out


def _import_scan(source: str, own_modules: set[str]) -> str | None:
    allowed = _STDLIB_MODULES | own_modules
    for mod in _imports_in(source):
        if mod and mod not in allowed:
            return f"non-stdlib import: {mod!r}"
    return None


def _required_fields(candidate: dict[str, Any]) -> str | None:
    required = ["task_id", "issue_text", "primary_file", "test_command",
                "buggy_files", "fixed_files", "test_file"]
    for k in required:
        if k not in candidate:
            return f"missing field: {k!r}"
    if not isinstance(candidate["buggy_files"], list) or not candidate["buggy_files"]:
        return "buggy_files must be a non-empty list"
    if not isinstance(candidate["fixed_files"], list) or not candidate["fixed_files"]:
        return "fixed_files must be a non-empty list"
    if not isinstance(candidate["test_file"], dict) or "path" not in candidate["test_file"]:
        return "test_file must be an object with `path` and `content`"
    if not re.fullmatch(r"[a-z][a-z0-9_]*", candidate["task_id"] or ""):
        return f"task_id must be snake_case: {candidate.get('task_id')!r}"
    return None


def _write_dir(target: Path, files: list[dict[str, str]], test_file: dict[str, str]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for fdef in files:
        p = target / fdef["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(fdef["content"], encoding="utf-8")
    (target / test_file["path"]).write_text(test_file["content"], encoding="utf-8")


def _run_pytest(workdir: Path, test_filename: str) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", test_filename]
    try:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=_PYTEST_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT — pytest exceeded the time budget."
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def validate_candidate(candidate: dict[str, Any]) -> ValidationResult:
    err = _required_fields(candidate)
    if err:
        return ValidationResult(False, err)

    # The candidate's own module names (basenames of any .py file the
    # candidate defines) — the test file is permitted to import these.
    own_modules: set[str] = set()
    for fdef in candidate["buggy_files"] + candidate["fixed_files"]:
        name = fdef.get("path", "")
        if name.endswith(".py") and "/" not in name:
            own_modules.add(name[:-3])

    # Safety + import scan on every source file (buggy + fixed + test).
    all_sources = (
        [f["content"] for f in candidate["buggy_files"]]
        + [f["content"] for f in candidate["fixed_files"]]
        + [candidate["test_file"]["content"]]
    )
    for src in all_sources:
        if (s := _safety_scan(src)) is not None:
            return ValidationResult(False, s)
        if (s := _import_scan(src, own_modules)) is not None:
            return ValidationResult(False, s)

    test_filename = candidate["test_file"]["path"]
    if not test_filename.endswith(".py") or "/" in test_filename:
        return ValidationResult(False, f"test_file.path must be a flat .py file: {test_filename!r}")

    with tempfile.TemporaryDirectory(prefix="noop-validate-") as td:
        td_path = Path(td)
        buggy_dir = td_path / "buggy"
        fixed_dir = td_path / "fixed"
        _write_dir(buggy_dir, candidate["buggy_files"], candidate["test_file"])
        _write_dir(fixed_dir, candidate["fixed_files"], candidate["test_file"])

        b_rc, b_out = _run_pytest(buggy_dir, test_filename)
        f_rc, f_out = _run_pytest(fixed_dir, test_filename)

    result = ValidationResult(
        valid=False,
        buggy_returncode=b_rc,
        fixed_returncode=f_rc,
        buggy_test_output=b_out,
        fixed_test_output=f_out,
    )

    if f_rc != 0:
        result.reason = f"fixed must pass; got exit={f_rc}"
        return result
    if b_rc == 0:
        result.reason = "buggy must fail at least one test; got exit=0 (all passed)"
        return result
    if b_rc not in (1,):
        result.reason = f"buggy exit code {b_rc} is not 1 (likely collection error / no tests)"
        return result

    # Sanity: both runs should report the same test count, otherwise the test
    # set diverged between conditions and our labels aren't clean.
    b_count = _count_passed_failed(b_out)
    f_count = _count_passed_failed(f_out)
    if b_count is not None and f_count is not None:
        if (b_count[0] + b_count[1]) != (f_count[0] + f_count[1]):
            result.reason = (
                f"test set diverged between buggy({sum(b_count)}) and fixed({sum(f_count)})"
            )
            return result

    result.valid = True
    return result


_SUMMARY_RE = re.compile(
    r"(\d+)\s+failed.*?(\d+)\s+passed|(\d+)\s+passed",
    re.IGNORECASE,
)


def _count_passed_failed(output: str) -> tuple[int, int] | None:
    m = _SUMMARY_RE.search(output)
    if not m:
        return None
    if m.group(1) is not None:  # "X failed, Y passed"
        return int(m.group(2)), int(m.group(1))
    if m.group(3) is not None:  # "Y passed"
        return int(m.group(3)), 0
    return None


def short_summary(result: ValidationResult) -> str:
    return json.dumps(
        {
            "valid": result.valid,
            "reason": result.reason,
            "buggy_exit": result.buggy_returncode,
            "fixed_exit": result.fixed_returncode,
        }
    )
