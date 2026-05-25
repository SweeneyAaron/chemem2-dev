"""Rendering tools — snapshot stubs that just emit placeholder file paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import safe_child
from ._common import STUB_WARNING, case_dir, check_limit, rate_limited_envelope


def _snapshot_dir(run_id: str) -> Path:
    out = safe_child(case_dir(run_id), "snapshots")
    out.mkdir(parents=True, exist_ok=True)
    return out


def render_snapshot(run_id: str, pose_id: str, view: str = "default") -> dict[str, Any]:
    limit = check_limit("render_snapshot", f"{run_id}:{pose_id}:{view}")
    if not limit.allowed:
        return rate_limited_envelope("render_snapshot", limit)
    out = safe_child(_snapshot_dir(run_id), f"{pose_id}_{view}.stub.png")
    out.write_text(f"# stub snapshot: {pose_id} {view}\n")
    return {
        "status": "ok",
        "tool_name": "render_snapshot",
        "pose_id": pose_id,
        "snapshot_path": str(out),
        "warnings": [STUB_WARNING],
    }


def render_ion_site_snapshot(run_id: str, ion_candidate_id: str) -> dict[str, Any]:
    limit = check_limit("render_ion_site_snapshot", f"{run_id}:{ion_candidate_id}")
    if not limit.allowed:
        return rate_limited_envelope("render_ion_site_snapshot", limit)
    out = safe_child(_snapshot_dir(run_id), f"{ion_candidate_id}.stub.png")
    out.write_text(f"# stub ion snapshot: {ion_candidate_id}\n")
    return {
        "status": "ok",
        "tool_name": "render_ion_site_snapshot",
        "ion_candidate_id": ion_candidate_id,
        "snapshot_path": str(out),
        "warnings": [STUB_WARNING],
    }


def render_refined_snapshots(run_id: str, refinement_id: str) -> dict[str, Any]:
    limit = check_limit("render_refined_snapshots", f"{run_id}:{refinement_id}")
    if not limit.allowed:
        return rate_limited_envelope("render_refined_snapshots", limit)
    out = safe_child(_snapshot_dir(run_id), f"{refinement_id}_refined.stub.png")
    out.write_text(f"# stub refined snapshot: {refinement_id}\n")
    return {
        "status": "ok",
        "tool_name": "render_refined_snapshots",
        "refinement_id": refinement_id,
        "snapshot_path": str(out),
        "warnings": [STUB_WARNING],
    }
