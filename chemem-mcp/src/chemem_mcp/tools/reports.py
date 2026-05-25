"""Reporting tools — final report assembly + case state summary."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..paths import safe_child
from ..schemas import (
    AcceptedIon,
    AcceptedLigand,
    FinalReport,
    FinalStatus,
)
from ._common import (
    STUB_WARNING,
    case_dir,
    check_limit,
    load_state,
    rate_limited_envelope,
)


def generate_final_report(
    run_id: str,
    final_status: FinalStatus = "accepted_with_warnings",
    accepted_ligand_pose_ids: Optional[list[str]] = None,
    accepted_ion_candidate_ids: Optional[list[str]] = None,
    noise_segment_ids: Optional[list[str]] = None,
    unexplained_segment_ids: Optional[list[str]] = None,
    recommended_human_checks: Optional[list[str]] = None,
) -> dict[str, Any]:
    limit = check_limit("generate_final_report", run_id)
    if not limit.allowed:
        return rate_limited_envelope("generate_final_report", limit)
    state = load_state(run_id)
    rd = case_dir(run_id)
    final_dir = safe_child(rd, "final")
    final_dir.mkdir(parents=True, exist_ok=True)

    accepted_ligands: list[AcceptedLigand] = []
    for pid in accepted_ligand_pose_ids or []:
        seg_id = ""
        ligand_id = "LIG"
        for sid, seg in state.segments.items():
            for cand in seg.candidate_assignments:
                if cand.pose_id == pid:
                    seg_id = sid
                    ligand_id = cand.ligand_id
                    break
        accepted_ligands.append(
            AcceptedLigand(
                ligand_id=ligand_id,
                segment_id=seg_id or "unknown",
                final_pose_id=pid,
                confidence=0.7,
                supporting_evidence=["stub final report"],
                warnings=[],
            )
        )

    accepted_ions: list[AcceptedIon] = []
    for iid in accepted_ion_candidate_ids or []:
        cand = state.ion_candidates.get(iid)
        accepted_ions.append(
            AcceptedIon(
                ion_id=iid,
                segment_id=cand.segment_id if cand else "unknown",
                confidence=cand.confidence if cand else 0.5,
                supporting_evidence=["stub final report"],
                warnings=[],
            )
        )

    report = FinalReport(
        run_id=run_id,
        final_status=final_status,
        accepted_ligands=accepted_ligands,
        accepted_ions=accepted_ions,
        noise_or_solvent_segments=list(noise_segment_ids or []),
        unexplained_segments=list(unexplained_segment_ids or []),
        recommended_human_checks=list(recommended_human_checks or []),
        files_to_keep=[
            str(rd / "inputs"),
            str(rd / "state.json"),
            str(rd / "decision_log.jsonl"),
            str(final_dir / "report.json"),
        ],
        files_safe_to_clean=[
            "intermediate docking poses beyond top candidates",
            "derived map crops",
            "failed refinement scratch directories",
        ],
    )

    report_path = safe_child(final_dir, "report.json")
    report_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))

    state.final_status = final_status
    state.accepted_ligands = accepted_ligands
    state.accepted_ions = accepted_ions
    state.set_stage("final_report")

    return {
        "status": "ok",
        "tool_name": "generate_final_report",
        "run_id": run_id,
        "report_path": str(report_path),
        "report": report.model_dump(mode="json"),
        "warnings": [STUB_WARNING],
    }


def summarise_case_state(run_id: str) -> dict[str, Any]:
    limit = check_limit("summarise_case_state", run_id)
    if not limit.allowed:
        return rate_limited_envelope("summarise_case_state", limit)
    state = load_state(run_id)
    return {
        "status": "ok",
        "tool_name": "summarise_case_state",
        "run_id": run_id,
        "stage": state.stage,
        "final_status": state.final_status,
        "counts": {
            "segments": len(state.segments),
            "ion_candidates": len(state.ion_candidates),
            "refinements": len(state.refinements),
            "controller_steps": state.controller_step_count,
            "tool_calls": state.tool_call_count,
        },
        "segment_ids": list(state.segments.keys()),
        "ion_candidate_ids": list(state.ion_candidates.keys()),
        "refinement_ids": list(state.refinements.keys()),
        "tweak_depth": dict(state.tweak_depth),
        "warnings": [STUB_WARNING],
    }


def get_case_disk_usage(run_id: str) -> dict[str, Any]:
    limit = check_limit("get_case_disk_usage", run_id)
    if not limit.allowed:
        return rate_limited_envelope("get_case_disk_usage", limit)
    rd = case_dir(run_id)
    total = 0
    n_files = 0
    for p in rd.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
            n_files += 1
    return {
        "status": "ok",
        "tool_name": "get_case_disk_usage",
        "run_id": run_id,
        "file_count": n_files,
        "total_bytes": total,
        "total_gb": round(total / (1024**3), 6),
        "warnings": [],
    }
