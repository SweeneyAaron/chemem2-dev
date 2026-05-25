"""Tier 2 — atomic primitives. Stubs only in Block 3."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..paths import safe_child
from ..schemas import (
    CandidateAssignment,
    CandidateScores,
    InteractionSummary,
    NonLigandHypothesis,
    SegmentEvidence,
)
from ._common import (
    STUB_WARNING,
    case_dir,
    check_limit,
    load_state,
    metric_from_truth,
    parse_pose_id,
    rate_limited_envelope,
    stub_int,
    stub_range,
    stub_unit01,
    truth_quality,
)


def _persist_pose_score(run_id: str, pose_id: str, **fields: Any) -> None:
    """Write computed metric(s) into ``state.poses[pose_id].scores`` and save.

    Silently no-ops if the pose isn't registered (e.g. atomic primitive called
    on a pose_id that was never docked). Keeps the agent's per-pose visibility
    in sync with what's been measured so the model can avoid recomputing.
    """
    state = load_state(run_id)
    pose = state.poses.get(pose_id)
    if pose is None:
        return
    for key, value in fields.items():
        if value is None:
            continue
        setattr(pose.scores, key, value)
    state.save()


# ---------------------------------------------------------------------------
# Map metric primitives (callable against any map path)
# ---------------------------------------------------------------------------


def _score_envelope(run_id: str, pose_id: str, map_path: str, tool: str) -> dict[str, Any]:
    case_dir(run_id)
    return {
        "status": "ok",
        "tool_name": tool,
        "pose_id": pose_id,
        "map_path": map_path,
        "warnings": [STUB_WARNING],
    }


def compute_ccc(run_id: str, pose_id: str, map_path: str) -> dict[str, Any]:
    limit = check_limit("compute_ccc", f"{run_id}:{pose_id}:{map_path}")
    if not limit.allowed:
        return rate_limited_envelope("compute_ccc", limit)
    env = _score_envelope(run_id, pose_id, map_path, "compute_ccc")
    value = round(metric_from_truth(0.45, 0.92, pose_id, "ccc"), 3)
    env["value"] = value
    _persist_pose_score(run_id, pose_id, ccc=value)
    return env


def compute_mi(run_id: str, pose_id: str, map_path: str) -> dict[str, Any]:
    limit = check_limit("compute_mi", f"{run_id}:{pose_id}:{map_path}")
    if not limit.allowed:
        return rate_limited_envelope("compute_mi", limit)
    env = _score_envelope(run_id, pose_id, map_path, "compute_mi")
    value = round(metric_from_truth(0.10, 0.50, pose_id, "mi"), 3)
    env["value"] = value
    _persist_pose_score(run_id, pose_id, mi=value)
    return env


def compute_mapq(run_id: str, pose_id: str, map_path: str) -> dict[str, Any]:
    limit = check_limit("compute_mapq", f"{run_id}:{pose_id}:{map_path}")
    if not limit.allowed:
        return rate_limited_envelope("compute_mapq", limit)
    env = _score_envelope(run_id, pose_id, map_path, "compute_mapq")
    mean = round(metric_from_truth(0.40, 0.85, pose_id, "mapqmean"), 3)
    minv = round(metric_from_truth(0.15, 0.65, pose_id, "mapqmin"), 3)
    if minv > mean:
        minv = round(mean - 0.05, 3)
    env["mean"] = mean
    env["min"] = minv
    _persist_pose_score(run_id, pose_id, mapq_mean=mean, mapq_min=minv)
    return env


def compute_sci(run_id: str, pose_id: str, map_path: str) -> dict[str, Any]:
    limit = check_limit("compute_sci", f"{run_id}:{pose_id}:{map_path}")
    if not limit.allowed:
        return rate_limited_envelope("compute_sci", limit)
    env = _score_envelope(run_id, pose_id, map_path, "compute_sci")
    value = round(metric_from_truth(0.30, 0.85, pose_id, "sci"), 3)
    env["value"] = value
    _persist_pose_score(run_id, pose_id, sci=value)
    return env


def compute_qscore_per_atom(
    run_id: str, pose_id: str, map_path: str, n_atoms: int = 30
) -> dict[str, Any]:
    limit = check_limit("compute_qscore_per_atom", f"{run_id}:{pose_id}:{map_path}")
    if not limit.allowed:
        return rate_limited_envelope("compute_qscore_per_atom", limit)
    case_dir(run_id)
    n = max(1, int(n_atoms))
    pose_mean = metric_from_truth(0.25, 0.85, pose_id, "qscore")
    scores = []
    for i in range(n):
        jitter = (stub_unit01(pose_id, map_path, "atom", i) - 0.5) * 0.30
        scores.append(round(max(0.0, min(1.0, pose_mean + jitter)), 3))
    mean = round(sum(scores) / len(scores), 3)
    _persist_pose_score(run_id, pose_id, qscore_mean=mean)
    return {
        "status": "ok",
        "tool_name": "compute_qscore_per_atom",
        "pose_id": pose_id,
        "map_path": map_path,
        "qscores": scores,
        "mean": mean,
        "warnings": [STUB_WARNING],
    }


# ---------------------------------------------------------------------------
# Map utilities — emit derived maps under runs/<run_id>/derived_maps/
# ---------------------------------------------------------------------------


def _derived_map_path(run_id: str, name: str) -> Path:
    dirpath = safe_child(case_dir(run_id), "derived_maps")
    dirpath.mkdir(parents=True, exist_ok=True)
    return safe_child(dirpath, name)


def _touch_stub_map(path: Path, marker: str) -> None:
    path.write_text(f"# stub derived map: {marker}\n")


def crop_map_around_segment(
    run_id: str, map_path: str, segment_id: str, padding_angstrom: float = 3.0
) -> dict[str, Any]:
    key = f"{run_id}:{segment_id}:{padding_angstrom}"
    limit = check_limit("crop_map_around_segment", key)
    if not limit.allowed:
        return rate_limited_envelope("crop_map_around_segment", limit)
    out = _derived_map_path(run_id, f"crop_{segment_id}_pad{padding_angstrom}.stub.mrc")
    _touch_stub_map(out, f"crop:{map_path}:{segment_id}:{padding_angstrom}")
    return {
        "status": "ok",
        "tool_name": "crop_map_around_segment",
        "source_map_path": map_path,
        "segment_id": segment_id,
        "padding_angstrom": padding_angstrom,
        "derived_map_path": str(out),
        "warnings": [STUB_WARNING],
    }


def mask_map_by_segment(
    run_id: str, map_path: str, segment_id: str
) -> dict[str, Any]:
    limit = check_limit("mask_map_by_segment", f"{run_id}:{segment_id}")
    if not limit.allowed:
        return rate_limited_envelope("mask_map_by_segment", limit)
    out = _derived_map_path(run_id, f"mask_{segment_id}.stub.mrc")
    _touch_stub_map(out, f"mask:{map_path}:{segment_id}")
    return {
        "status": "ok",
        "tool_name": "mask_map_by_segment",
        "source_map_path": map_path,
        "segment_id": segment_id,
        "derived_map_path": str(out),
        "warnings": [STUB_WARNING],
    }


def subtract_ligand_density(
    run_id: str, map_path: str, pose_id: str, resolution: float = 3.0
) -> dict[str, Any]:
    limit = check_limit("subtract_ligand_density", f"{run_id}:{pose_id}")
    if not limit.allowed:
        return rate_limited_envelope("subtract_ligand_density", limit)
    out = _derived_map_path(run_id, f"residual_{pose_id}.stub.mrc")
    _touch_stub_map(out, f"sub:{map_path}:{pose_id}:{resolution}")
    return {
        "status": "ok",
        "tool_name": "subtract_ligand_density",
        "source_map_path": map_path,
        "pose_id": pose_id,
        "resolution": resolution,
        "derived_map_path": str(out),
        "warnings": [STUB_WARNING],
    }


def compute_ligand_volume(run_id: str, ligand_id: str) -> dict[str, Any]:
    limit = check_limit("compute_ligand_volume", f"{run_id}:{ligand_id}")
    if not limit.allowed:
        return rate_limited_envelope("compute_ligand_volume", limit)
    case_dir(run_id)
    return {
        "status": "ok",
        "tool_name": "compute_ligand_volume",
        "ligand_id": ligand_id,
        "volume_angstrom3": round(stub_range(200.0, 600.0, ligand_id, "lvol"), 2),
        "warnings": [STUB_WARNING],
    }


def compute_segment_volume(run_id: str, segment_id: str) -> dict[str, Any]:
    limit = check_limit("compute_segment_volume", f"{run_id}:{segment_id}")
    if not limit.allowed:
        return rate_limited_envelope("compute_segment_volume", limit)
    state = load_state(run_id)
    vol: Optional[float] = None
    if segment_id in state.segments:
        vol = state.segments[segment_id].volume_angstrom3
    return {
        "status": "ok",
        "tool_name": "compute_segment_volume",
        "segment_id": segment_id,
        "volume_angstrom3": vol if vol is not None else round(stub_range(100.0, 700.0, segment_id), 2),
        "warnings": [STUB_WARNING],
    }


# ---------------------------------------------------------------------------
# Non-map analysis primitives
# ---------------------------------------------------------------------------


def analyse_clashes(run_id: str, pose_id: str) -> dict[str, Any]:
    limit = check_limit("analyse_clashes", f"{run_id}:{pose_id}")
    if not limit.allowed:
        return rate_limited_envelope("analyse_clashes", limit)
    case_dir(run_id)
    # Inverse correlation with truth_quality: the winner has few clashes.
    seg, lig, idx = parse_pose_id(pose_id)
    q = truth_quality(seg, lig, idx)
    clash_count = int(round((1.0 - q) * 5))
    severe = int(round((1.0 - q) * 2))
    max_overlap = round((1.0 - q) * 0.6, 3)
    _persist_pose_score(
        run_id, pose_id, clash_count=clash_count, severe_clash_count=severe
    )
    return {
        "status": "ok",
        "tool_name": "analyse_clashes",
        "pose_id": pose_id,
        "clash_count": clash_count,
        "severe_clash_count": severe,
        "max_overlap_angstrom": max_overlap,
        "warnings": [STUB_WARNING],
    }


def analyse_ligand_strain(run_id: str, pose_id: str) -> dict[str, Any]:
    limit = check_limit("analyse_ligand_strain", f"{run_id}:{pose_id}")
    if not limit.allowed:
        return rate_limited_envelope("analyse_ligand_strain", limit)
    case_dir(run_id)
    seg, lig, idx = parse_pose_id(pose_id)
    q = truth_quality(seg, lig, idx)
    strain = round(0.5 + (1.0 - q) * 14.5, 3)
    _persist_pose_score(run_id, pose_id, ligand_strain_kcal_mol=strain)
    return {
        "status": "ok",
        "tool_name": "analyse_ligand_strain",
        "pose_id": pose_id,
        "strain_kcal_mol": strain,
        "warnings": [STUB_WARNING],
    }


def analyse_interactions(run_id: str, pose_id: str) -> dict[str, Any]:
    limit = check_limit("analyse_interactions", f"{run_id}:{pose_id}")
    if not limit.allowed:
        return rate_limited_envelope("analyse_interactions", limit)
    case_dir(run_id)
    summary = InteractionSummary(
        hbonds=stub_int(0, 5, pose_id, "hb"),
        salt_bridges=stub_int(0, 2, pose_id, "sb"),
        pi_stacking=stub_int(0, 2, pose_id, "pi"),
        metal_coordination=stub_int(0, 1, pose_id, "mc"),
        key_residues=["LYS42", "ASP87", "TYR120"],
    )
    return {
        "status": "ok",
        "tool_name": "analyse_interactions",
        "pose_id": pose_id,
        "summary": summary.model_dump(mode="json"),
        "warnings": [STUB_WARNING],
    }


# ---------------------------------------------------------------------------
# Ion primitives
# ---------------------------------------------------------------------------


def detect_ion_like_density(run_id: str, sigma_threshold: float = 3.0) -> dict[str, Any]:
    limit = check_limit("detect_ion_like_density", run_id)
    if not limit.allowed:
        return rate_limited_envelope("detect_ion_like_density", limit)
    state = load_state(run_id)
    candidate_segments = [
        sid for sid in state.segments if state.segments[sid].peak_density >= sigma_threshold
    ]
    return {
        "status": "ok",
        "tool_name": "detect_ion_like_density",
        "sigma_threshold": sigma_threshold,
        "candidate_segment_ids": candidate_segments[:3],
        "warnings": [STUB_WARNING],
    }


def analyse_coordination_geometry(run_id: str, ion_candidate_id: str) -> dict[str, Any]:
    limit = check_limit("analyse_coordination_geometry", f"{run_id}:{ion_candidate_id}")
    if not limit.allowed:
        return rate_limited_envelope("analyse_coordination_geometry", limit)
    state = load_state(run_id)
    cand = state.ion_candidates.get(ion_candidate_id)
    if cand is None:
        return {
            "status": "ok",
            "tool_name": "analyse_coordination_geometry",
            "ion_candidate_id": ion_candidate_id,
            "geometry": "unknown",
            "coordination_number": 0,
            "warnings": [STUB_WARNING, "ion_candidate_id not found in state"],
        }
    return {
        "status": "ok",
        "tool_name": "analyse_coordination_geometry",
        "ion_candidate_id": ion_candidate_id,
        "geometry": cand.coordination.geometry,
        "coordination_number": cand.coordination.coordination_number,
        "mean_distance_angstrom": cand.coordination.mean_distance_angstrom,
        "warnings": [STUB_WARNING],
    }


def propose_ion_identity(
    run_id: str,
    coordination_number: int,
    mean_distance_angstrom: float,
) -> dict[str, Any]:
    key = f"{run_id}:cn={coordination_number}:d={mean_distance_angstrom}"
    limit = check_limit("propose_ion_identity", key)
    if not limit.allowed:
        return rate_limited_envelope("propose_ion_identity", limit)
    case_dir(run_id)
    if mean_distance_angstrom < 2.1:
        identity = "Mg2+"
    elif mean_distance_angstrom < 2.45:
        identity = "Zn2+"
    else:
        identity = "Ca2+"
    return {
        "status": "ok",
        "tool_name": "propose_ion_identity",
        "proposed_identity": identity,
        "confidence": round(stub_range(0.4, 0.85, identity, key), 3),
        "warnings": [STUB_WARNING],
    }


def score_ion_assignment(run_id: str, ion_candidate_id: str) -> dict[str, Any]:
    limit = check_limit("score_ion_assignment", f"{run_id}:{ion_candidate_id}")
    if not limit.allowed:
        return rate_limited_envelope("score_ion_assignment", limit)
    case_dir(run_id)
    return {
        "status": "ok",
        "tool_name": "score_ion_assignment",
        "ion_candidate_id": ion_candidate_id,
        "score": round(stub_range(0.3, 0.9, ion_candidate_id, "ion"), 3),
        "warnings": [STUB_WARNING],
    }


# ---------------------------------------------------------------------------
# Refinement-result primitives
# ---------------------------------------------------------------------------


def monitor_refinement_stability(run_id: str, refinement_id: str) -> dict[str, Any]:
    limit = check_limit("monitor_refinement_stability", f"{run_id}:{refinement_id}")
    if not limit.allowed:
        return rate_limited_envelope("monitor_refinement_stability", limit)
    state = load_state(run_id)
    refine = state.refinements.get(refinement_id)
    if refine is None:
        return {
            "status": "ok",
            "tool_name": "monitor_refinement_stability",
            "refinement_id": refinement_id,
            "stability": None,
            "warnings": [STUB_WARNING, "refinement_id not found in state"],
        }
    return {
        "status": "ok",
        "tool_name": "monitor_refinement_stability",
        "refinement_id": refinement_id,
        "stability": refine.stability.model_dump(mode="json"),
        "warnings": [STUB_WARNING],
    }


def score_refined_model(run_id: str, refinement_id: str) -> dict[str, Any]:
    limit = check_limit("score_refined_model", f"{run_id}:{refinement_id}")
    if not limit.allowed:
        return rate_limited_envelope("score_refined_model", limit)
    case_dir(run_id)
    return {
        "status": "ok",
        "tool_name": "score_refined_model",
        "refinement_id": refinement_id,
        "ccc": round(stub_range(0.6, 0.9, refinement_id, "ccc"), 3),
        "mapq_mean": round(stub_range(0.55, 0.8, refinement_id, "mapq"), 3),
        "warnings": [STUB_WARNING],
    }


def analyse_refined_ligands(run_id: str, refinement_id: str) -> dict[str, Any]:
    limit = check_limit("analyse_refined_ligands", f"{run_id}:{refinement_id}")
    if not limit.allowed:
        return rate_limited_envelope("analyse_refined_ligands", limit)
    state = load_state(run_id)
    refine = state.refinements.get(refinement_id)
    pose_ids = refine.inputs.pose_ids if refine else []
    return {
        "status": "ok",
        "tool_name": "analyse_refined_ligands",
        "refinement_id": refinement_id,
        "per_ligand": [
            {
                "pose_id": pid,
                "ccc": round(stub_range(0.6, 0.85, pid, "rccc"), 3),
                "strain_kcal_mol": round(stub_range(0.5, 5.0, pid, "rstrain"), 3),
            }
            for pid in pose_ids
        ],
        "warnings": [STUB_WARNING],
    }


def analyse_refined_ions(run_id: str, refinement_id: str) -> dict[str, Any]:
    limit = check_limit("analyse_refined_ions", f"{run_id}:{refinement_id}")
    if not limit.allowed:
        return rate_limited_envelope("analyse_refined_ions", limit)
    state = load_state(run_id)
    refine = state.refinements.get(refinement_id)
    ion_ids = refine.inputs.ion_candidate_ids if refine else []
    return {
        "status": "ok",
        "tool_name": "analyse_refined_ions",
        "refinement_id": refinement_id,
        "per_ion": [
            {
                "ion_id": iid,
                "coordination_stable": True,
                "coordination_score": round(stub_range(0.5, 0.9, iid, "ricr"), 3),
            }
            for iid in ion_ids
        ],
        "warnings": [STUB_WARNING],
    }


def analyse_residual_density(run_id: str, refinement_id: str) -> dict[str, Any]:
    limit = check_limit("analyse_residual_density", f"{run_id}:{refinement_id}")
    if not limit.allowed:
        return rate_limited_envelope("analyse_residual_density", limit)
    case_dir(run_id)
    return {
        "status": "ok",
        "tool_name": "analyse_residual_density",
        "refinement_id": refinement_id,
        "unexplained_segment_ids": [],
        "warnings": [STUB_WARNING],
    }


def compare_refined_to_initial(run_id: str, refinement_id: str) -> dict[str, Any]:
    limit = check_limit("compare_refined_to_initial", f"{run_id}:{refinement_id}")
    if not limit.allowed:
        return rate_limited_envelope("compare_refined_to_initial", limit)
    state = load_state(run_id)
    refine = state.refinements.get(refinement_id)
    deltas = refine.score_deltas.model_dump(mode="json") if refine else {}
    return {
        "status": "ok",
        "tool_name": "compare_refined_to_initial",
        "refinement_id": refinement_id,
        "score_deltas": deltas,
        "warnings": [STUB_WARNING],
    }


def summarise_refinement_outcome(run_id: str, refinement_id: str) -> dict[str, Any]:
    limit = check_limit("summarise_refinement_outcome", f"{run_id}:{refinement_id}")
    if not limit.allowed:
        return rate_limited_envelope("summarise_refinement_outcome", limit)
    state = load_state(run_id)
    refine = state.refinements.get(refinement_id)
    return {
        "status": "ok" if refine else "missing",
        "tool_name": "summarise_refinement_outcome",
        "refinement_id": refinement_id,
        "outcome": refine.model_dump(mode="json") if refine else None,
        "warnings": [STUB_WARNING],
    }


# ---------------------------------------------------------------------------
# Segment-level summary
# ---------------------------------------------------------------------------


def _scores_from_truth(pose_id: str) -> CandidateScores:
    """Build a complete CandidateScores from the per-pose ground truth.

    Used to fill fields the controller hasn't explicitly measured yet, so
    summarise_segment_candidates can return a coherent row per pose.
    """
    seg, lig, idx = parse_pose_id(pose_id)
    q = truth_quality(seg, lig, idx)
    return CandidateScores(
        dock_score=round(-4.0 - q * 8.0, 3),
        minimised_score=round(-5.0 - q * 9.0, 3),
        ccc=round(metric_from_truth(0.45, 0.92, pose_id, "ccc"), 3),
        mi=round(metric_from_truth(0.10, 0.50, pose_id, "mi"), 3),
        sci=round(metric_from_truth(0.30, 0.85, pose_id, "sci"), 3),
        mapq_mean=round(metric_from_truth(0.40, 0.85, pose_id, "mapqmean"), 3),
        mapq_min=round(metric_from_truth(0.15, 0.65, pose_id, "mapqmin"), 3),
        density_coverage=round(0.4 + q * 0.55, 3),
        clash_count=int(round((1.0 - q) * 5)),
        severe_clash_count=int(round((1.0 - q) * 2)),
        ligand_strain_kcal_mol=round(0.5 + (1.0 - q) * 7.5, 3),
    )


def _merge_scores(measured: CandidateScores, derived: CandidateScores) -> CandidateScores:
    """Prefer measured fields; fall back to derived for anything unset."""
    out = measured.model_copy(deep=True)
    derived_dump = derived.model_dump()
    for key, val in derived_dump.items():
        if getattr(out, key) is None:
            setattr(out, key, val)
    return out


def _aggregate_score(scores: CandidateScores) -> float:
    """Rank-friendly aggregate across the metrics the controller cares about.

    Mean of normalised CCC, MI, SCI, MapQ_mean (only those that are set), so
    the row with the highest measured-or-derived agreement wins.
    """
    pieces: list[float] = []
    if scores.ccc is not None:
        pieces.append(scores.ccc)
    if scores.mi is not None:
        pieces.append(scores.mi / 0.5)  # rescale to roughly [0, 1]
    if scores.sci is not None:
        pieces.append(scores.sci)
    if scores.mapq_mean is not None:
        pieces.append(scores.mapq_mean)
    return sum(pieces) / len(pieces) if pieces else 0.0


def summarise_segment_candidates(
    run_id: str,
    segment_id: str,
    pose_ids: Optional[list[str]] = None,
    top_n: int = 5,
) -> dict[str, Any]:
    limit = check_limit("summarise_segment_candidates", f"{run_id}:{segment_id}")
    if not limit.allowed:
        return rate_limited_envelope("summarise_segment_candidates", limit)
    state = load_state(run_id)
    if segment_id not in state.segments:
        return {
            "status": "missing",
            "tool_name": "summarise_segment_candidates",
            "segment_id": segment_id,
            "warnings": [STUB_WARNING, "segment_id not found in state"],
        }
    seg = state.segments[segment_id]

    if pose_ids:
        target_pids = [pid for pid in pose_ids if pid]
    else:
        target_pids = [
            p.pose_id for p in state.poses.values() if p.segment_id == segment_id
        ]

    candidates: list[CandidateAssignment] = []
    for pid in target_pids:
        pose = state.poses.get(pid)
        ligand_id = pose.ligand_id if pose else (parse_pose_id(pid)[1] or "LIG")
        measured = pose.scores if pose else CandidateScores()
        derived = _scores_from_truth(pid)
        scores = _merge_scores(measured, derived)
        candidates.append(
            CandidateAssignment(
                assignment_id=f"{segment_id}_{pid}",
                segment_id=segment_id,
                ligand_id=ligand_id,
                pose_id=pid,
                scores=scores,
                interactions=InteractionSummary(
                    hbonds=stub_int(0, 5, pid, "hb"),
                    salt_bridges=stub_int(0, 2, pid, "sb"),
                    key_residues=seg.nearby_residues or ["LYS42", "ASP87"],
                ),
            )
        )

    candidates.sort(key=lambda c: _aggregate_score(c.scores), reverse=True)
    top = candidates[: max(1, int(top_n))] if candidates else []

    enriched = seg.model_copy(
        update={
            "candidate_assignments": top,
            "non_ligand_hypotheses": [
                NonLigandHypothesis(
                    type="solvent_or_noise",
                    confidence=round(stub_range(0.05, 0.35, segment_id, "noise"), 3),
                    reason="stub: shape + volume are weakly compatible with solvent",
                )
            ],
        }
    )
    state.add_segment(enriched)
    return {
        "status": "ok",
        "tool_name": "summarise_segment_candidates",
        "segment_id": segment_id,
        "evidence": enriched.model_dump(mode="json"),
        "warnings": [STUB_WARNING],
    }
