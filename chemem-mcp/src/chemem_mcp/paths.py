from __future__ import annotations

import re
from pathlib import Path

_VALID_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]{0,63}$")


class UnsafePathError(ValueError):
    pass


def validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not _VALID_RUN_ID.fullmatch(run_id):
        raise UnsafePathError(
            f"run_id must match [A-Za-z0-9][A-Za-z0-9_\\-.]{{0,63}}, got {run_id!r}"
        )


def safe_run_dir(run_id: str, run_root: Path) -> Path:
    validate_run_id(run_id)
    run_root = Path(run_root).resolve()
    candidate = (run_root / run_id).resolve()
    try:
        candidate.relative_to(run_root)
    except ValueError as exc:
        raise UnsafePathError(f"resolved run dir {candidate} escapes {run_root}") from exc
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def safe_child(parent: Path, *parts: str) -> Path:
    parent = Path(parent).resolve()
    candidate = parent.joinpath(*parts).resolve()
    try:
        candidate.relative_to(parent)
    except ValueError as exc:
        raise UnsafePathError(f"path {candidate} escapes {parent}") from exc
    return candidate
