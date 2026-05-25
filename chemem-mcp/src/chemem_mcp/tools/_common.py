"""Shared helpers for stub tool implementations."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

from ..paths import safe_run_dir
from ..rate_limiter import RateLimitResult, RateLimiter
from ..runtime import get_rate_limiter, get_run_root
from ..state import CaseState


def case_dir(run_id: str, run_root: Optional[Path] = None) -> Path:
    return safe_run_dir(run_id, run_root or get_run_root())


def load_state(run_id: str, run_root: Optional[Path] = None) -> CaseState:
    return CaseState.load_or_create(run_id, case_dir(run_id, run_root))


def check_limit(
    tool_name: str,
    key: str,
    *,
    rate_limiter: Optional[RateLimiter] = None,
) -> RateLimitResult:
    rl = rate_limiter or get_rate_limiter()
    return rl.check_and_increment(tool_name, key)


def rate_limited_envelope(tool_name: str, result: RateLimitResult) -> dict[str, Any]:
    return {
        "status": "rate_limited",
        "tool_name": tool_name,
        "rate_limit": result.to_dict(),
        "warnings": [f"{tool_name} rate limit exhausted for key={result.key}"],
    }


def stub_unit01(*parts: Any) -> float:
    """Deterministic float in [0, 1) derived from inputs (cross-run stable)."""
    s = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.md5(s).digest()[:4], "big") / 0xFFFFFFFF


def stub_int(low: int, high: int, *parts: Any) -> int:
    return low + int(stub_unit01(*parts) * (high - low + 1))


def stub_range(lo: float, hi: float, *parts: Any) -> float:
    return lo + stub_unit01(*parts) * (hi - lo)


STUB_WARNING = "stub implementation"


# ---------------------------------------------------------------------------
# Stub "ground truth" — shared across all scoring metrics so that one pose
# per segment is a clear cross-metric winner. Without this, every metric is
# an independent hash draw and the LLM has no coherent signal to latch onto.
# ---------------------------------------------------------------------------


def parse_pose_id(pose_id: str) -> tuple[str, str, int]:
    """Decompose ``f"{segment_id}_{ligand_id}_pose_{n:03d}"``.

    Returns ``(segment_id, ligand_id, pose_index)``. Falls back to
    ``("", "", 0)`` for ids that don't match the convention so callers can
    still produce a deterministic-but-degraded result.
    """
    if "_pose_" not in pose_id:
        return "", "", 0
    head, tail = pose_id.rsplit("_pose_", 1)
    try:
        pose_index = int(tail)
    except ValueError:
        pose_index = 0
    if "_" not in head:
        return head, "", pose_index
    segment_id, ligand_id = head.rsplit("_", 1)
    return segment_id, ligand_id, pose_index


def truth_quality(segment_id: str, ligand_id: str, pose_index: int) -> float:
    """Hidden per-pose 'true' quality in [0, 1).

    Skewed so that among ~10 poses per segment, one stands out clearly:
    ``s = stub_unit01(...)`` is uniform, then we cube it. Expected max over
    10 draws lands around 0.9; the runner-up sits well below that.
    """
    s = stub_unit01(segment_id, ligand_id, pose_index, "truth")
    return s ** 3


def metric_from_truth(
    lo: float,
    hi: float,
    pose_id: str,
    metric_salt: str,
    *,
    jitter_frac: float = 0.06,
) -> float:
    """Map ``truth_quality`` into a metric range with small per-metric jitter.

    All scoring primitives derive from this so they agree on which pose is
    best. ``jitter_frac`` keeps the metrics from being suspiciously identical
    without scrambling the rank order.
    """
    segment_id, ligand_id, pose_index = parse_pose_id(pose_id)
    q = truth_quality(segment_id, ligand_id, pose_index)
    base = lo + (hi - lo) * q
    jitter = (stub_unit01(pose_id, metric_salt) - 0.5) * (hi - lo) * jitter_frac
    return base + jitter
