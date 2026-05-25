from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .paths import safe_child

ALWAYS_KEEP_PATTERNS: tuple[str, ...] = (
    "inputs",
    "inputs/**",
    "state.json",
    "decision_log.jsonl",
    "resource_usage.json",
    "final",
    "final/**",
    "report.json",
    "*.controller.json",
    "*.decision.json",
    "*.reviewer.json",
)

DEFAULT_DELETE_GLOBS: tuple[str, ...] = (
    "docking_scratch/**",
    "derived_maps/**",
    "tmp/**",
    "scratch/**",
    "*.tmp",
)


class UnsafeCleanupError(RuntimeError):
    pass


@dataclass
class CleanupReport:
    run_dir: str
    dry_run: bool
    deleted_files: list[str] = field(default_factory=list)
    kept_protected: list[str] = field(default_factory=list)
    freed_bytes: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": "ok",
            "run_dir": self.run_dir,
            "dry_run": self.dry_run,
            "deleted_file_count": len(self.deleted_files),
            "freed_gb": round(self.freed_bytes / (1024**3), 4),
            "freed_bytes": self.freed_bytes,
            "deleted_files": self.deleted_files,
            "kept_protected_count": len(self.kept_protected),
            "warnings": self.warnings,
        }


def _matches_any(rel_path: str, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(rel_path, pat):
            return True
        if "/" in pat and fnmatch.fnmatch(rel_path, pat):
            return True
        # Handle directory-style patterns
        prefix = pat.rstrip("/").rstrip("*").rstrip("/")
        if prefix and (rel_path == prefix or rel_path.startswith(prefix + "/")):
            return True
    return False


def cleanup_run_dir(
    run_dir: Path,
    *,
    dry_run: bool = True,
    extra_keep: Optional[Iterable[str]] = None,
    delete_globs: Optional[Iterable[str]] = None,
) -> CleanupReport:
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise UnsafeCleanupError(f"run_dir {run_dir} does not exist or is not a directory")

    keep_patterns = list(ALWAYS_KEEP_PATTERNS) + list(extra_keep or ())
    delete_patterns = list(delete_globs) if delete_globs is not None else list(DEFAULT_DELETE_GLOBS)

    report = CleanupReport(run_dir=str(run_dir), dry_run=dry_run)

    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        # Guard against symlinks escaping the run dir.
        try:
            safe_child(run_dir, *path.relative_to(run_dir).parts)
        except Exception as exc:
            report.warnings.append(f"skipped unsafe path {path}: {exc}")
            continue

        rel = path.relative_to(run_dir).as_posix()

        if _matches_any(rel, keep_patterns):
            report.kept_protected.append(rel)
            continue

        if not _matches_any(rel, delete_patterns):
            continue

        try:
            size = path.stat().st_size
        except OSError:
            size = 0

        report.deleted_files.append(rel)
        report.freed_bytes += size

        if not dry_run:
            try:
                path.unlink()
            except OSError as exc:
                report.warnings.append(f"failed to delete {rel}: {exc}")

    return report
