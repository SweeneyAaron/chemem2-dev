from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .rate_limiter import RateLimiter


def _initial_run_root() -> Path:
    return Path(os.environ.get("CHEMEM_MCP_RUN_ROOT", "runs")).resolve()


_run_root: Path = _initial_run_root()
_rate_limiter: RateLimiter = RateLimiter.with_defaults()


def get_run_root() -> Path:
    return _run_root


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter


def set_runtime(
    *,
    run_root: Optional[Path] = None,
    rate_limiter: Optional[RateLimiter] = None,
) -> None:
    global _run_root, _rate_limiter
    if run_root is not None:
        _run_root = Path(run_root).resolve()
    if rate_limiter is not None:
        _rate_limiter = rate_limiter


@contextmanager
def runtime_override(
    *,
    run_root: Optional[Path] = None,
    rate_limiter: Optional[RateLimiter] = None,
) -> Iterator[None]:
    global _run_root, _rate_limiter
    prev_root, prev_rl = _run_root, _rate_limiter
    if run_root is not None:
        _run_root = Path(run_root).resolve()
    if rate_limiter is not None:
        _rate_limiter = rate_limiter
    try:
        yield
    finally:
        _run_root, _rate_limiter = prev_root, prev_rl
