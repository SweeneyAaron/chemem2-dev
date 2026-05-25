from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from chemem_mcp.rate_limiter import RateLimiter
from chemem_mcp.runtime import runtime_override


@pytest.fixture
def runtime(tmp_path: Path) -> Iterator[tuple[Path, RateLimiter]]:
    """Override runtime singletons with a per-test tmp dir + fresh limiter."""
    rl = RateLimiter.with_defaults()
    with runtime_override(run_root=tmp_path, rate_limiter=rl):
        yield tmp_path, rl
