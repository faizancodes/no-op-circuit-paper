"""Local helpers for reading from / writing to Modal Volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import ACTIVATIONS_VOLUME


def download_activations(
    remote_paths: Iterable[str],
    local_root: Path,
    *,
    volume_name: str = ACTIVATIONS_VOLUME,
) -> list[Path]:
    """Download a set of files from the activations Volume to a local directory.

    `remote_paths` are paths relative to the volume root, e.g.
    ``"smoke-20260515T130000Z/parser_empty_input/buggy__code_tests.pt"``.
    Returns the list of local file paths written.
    """
    import modal  # local-only import; modal is in pyproject deps

    vol = modal.Volume.from_name(volume_name)
    written: list[Path] = []
    for remote in remote_paths:
        local = local_root / remote
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.open("wb") as fh:
            for chunk in vol.read_file(remote):
                fh.write(chunk)
        written.append(local)
    return written


def list_activations_run(run_id: str, *, volume_name: str = ACTIVATIONS_VOLUME) -> list[str]:
    """List files under a given run_id in the activations Volume."""
    import modal

    vol = modal.Volume.from_name(volume_name)
    return [entry.path for entry in vol.iterdir(run_id, recursive=True)]
