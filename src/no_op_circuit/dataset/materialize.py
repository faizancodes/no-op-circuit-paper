"""Write a validated TaskCandidate to data/tasks/<id>/ in the on-disk layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config import TASKS_DIR
from .validator import ValidationResult


def _resolve_unique_id(proposed: str, tasks_dir: Path) -> str:
    base = proposed
    suffix = 1
    while (tasks_dir / proposed).exists():
        suffix += 1
        proposed = f"{base}_{suffix}"
    return proposed


def materialize(
    candidate: dict[str, Any],
    result: ValidationResult,
    *,
    tasks_dir: Path = TASKS_DIR,
) -> Path:
    assert result.valid, "materialize() called on an invalid result"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_id = _resolve_unique_id(candidate["task_id"], tasks_dir)
    out = tasks_dir / task_id
    out.mkdir(parents=True, exist_ok=False)

    buggy_dir = out / "buggy"
    fixed_dir = out / "fixed"
    buggy_dir.mkdir()
    fixed_dir.mkdir()

    test_file = candidate["test_file"]

    for fdef in candidate["buggy_files"]:
        (buggy_dir / fdef["path"]).write_text(fdef["content"], encoding="utf-8")
    (buggy_dir / test_file["path"]).write_text(test_file["content"], encoding="utf-8")
    (buggy_dir / "tests_output.txt").write_text(
        result.buggy_test_output + "\n", encoding="utf-8"
    )

    for fdef in candidate["fixed_files"]:
        (fixed_dir / fdef["path"]).write_text(fdef["content"], encoding="utf-8")
    (fixed_dir / test_file["path"]).write_text(test_file["content"], encoding="utf-8")
    (fixed_dir / "tests_output.txt").write_text(
        result.fixed_test_output + "\n", encoding="utf-8"
    )

    meta = {
        "task_id": task_id,
        "description": candidate.get("description", "").strip() or None,
        "issue_text": candidate["issue_text"].strip() + "\n",
        "primary_file": candidate["primary_file"],
        "test_command": candidate["test_command"],
        "expected_action": {"buggy": "edit", "fixed": "noop"},
        "source_archetype": candidate.get("_archetype"),
    }
    # Drop None-valued keys for tidier YAML.
    meta = {k: v for k, v in meta.items() if v is not None}
    (out / "meta.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return out
