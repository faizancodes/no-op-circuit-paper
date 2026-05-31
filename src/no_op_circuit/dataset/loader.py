"""Load TaskPair objects from disk."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import yaml

from ..config import TASKS_DIR
from .schema import FileSnapshot, TaskPair


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _collect_files(root: Path) -> list[FileSnapshot]:
    files: list[FileSnapshot] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel == "tests_output.txt":
            continue  # carried as test transcript, not source
        files.append(FileSnapshot(path=rel, content=_read(p)))
    return files


def load_task(task_id: str, tasks_dir: Path = TASKS_DIR) -> TaskPair:
    task_dir = tasks_dir / task_id
    meta_path = task_dir / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing meta.yaml for task {task_id!r}: {meta_path}")
    meta = yaml.safe_load(_read(meta_path))

    buggy_dir = task_dir / "buggy"
    fixed_dir = task_dir / "fixed"
    pair = TaskPair(
        task_id=meta["task_id"],
        description=meta.get("description", "").strip(),
        issue_text=meta["issue_text"].strip(),
        primary_file=meta["primary_file"],
        test_command=meta["test_command"],
        buggy_files=_collect_files(buggy_dir),
        fixed_files=_collect_files(fixed_dir),
        buggy_test_output=_read(buggy_dir / "tests_output.txt"),
        fixed_test_output=_read(fixed_dir / "tests_output.txt"),
    )
    flaky_path = fixed_dir / "tests_output_flaky.txt"
    if flaky_path.is_file():
        pair.fixed_flaky_test_output = _read(flaky_path)
    return pair


def iter_tasks(tasks_dir: Path = TASKS_DIR) -> Iterator[TaskPair]:
    for child in sorted(tasks_dir.iterdir()):
        if (child / "meta.yaml").is_file():
            yield load_task(child.name, tasks_dir=tasks_dir)
