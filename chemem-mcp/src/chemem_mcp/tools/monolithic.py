"""Tier 1 — monolithic protocol wrappers. Stubs only in Block 3."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from ..schemas import (
    AlphaMaskParameters,
    BindingSiteRecord,
    IonCandidate,
    IonCoordination,
    IonDensity,
    LigandLibraryEntry,
    PoseRecord,
    RefinementFiles,
    RefinementInputs,
    RefinementOutcome,
    RefinementScoreDeltas,
    RefinementStability,
    SegmentEvidence,
)
from ._common import (
    STUB_WARNING,
    case_dir,
    check_limit,
    load_state,
    rate_limited_envelope,
    stub_int,
    stub_range,
    stub_unit01,
)


# ---------------------------------------------------------------------------
# Stage 1 — preparation
# ---------------------------------------------------------------------------


def prepare_case(
    run_id: str,
    protein_pdb: str,
    map_mrc: str,
    ligand_library: str,
    site_hint_json: Optional[str] = None,
) -> dict[str, Any]:
    """Validate inputs and stage them under runs/<run_id>/inputs/."""
    limit = check_limit("prepare_case", run_id)
    if not limit.allowed:
        return rate_limited_envelope("prepare_case", limit)
    run_dir = case_dir(run_id)
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    staged: dict[str, str] = {}
    for label, source in (
        ("protein_pdb", protein_pdb),
        ("map_mrc", map_mrc),
        ("ligand_library", ligand_library),
        ("site_hint_json", site_hint_json),
    ):
        if source is None:
            continue
        src = Path(source)
        if src.exists():
            dst = inputs_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            staged[label] = str(dst)
        else:
            staged[label] = source  # record as-given; real impl will validate later

    state = load_state(run_id)
    state.set_stage("preparation")

    return {
        "status": "ok",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "files": staged,
        "summary": {"message": "stub prepare_case completed"},
        "warnings": [STUB_WARNING],
    }


def generate_binding_sites(run_id: str, max_sites: int = 4) -> dict[str, Any]:
    limit = check_limit("generate_binding_sites", run_id)
    if not limit.allowed:
        return rate_limited_envelope("generate_binding_sites", limit)
    state = load_state(run_id)
    n = min(max(1, max_sites), 4)
    sites: list[dict[str, Any]] = []
    for i in range(n):
        sid = f"site_{i + 1:03d}"
        record = BindingSiteRecord(
            site_id=sid,
            center_xyz=(
                round(stub_range(-10, 10, run_id, sid, "x"), 3),
                round(stub_range(-10, 10, run_id, sid, "y"), 3),
                round(stub_range(-10, 10, run_id, sid, "z"), 3),
            ),
            radius_angstrom=round(stub_range(5.0, 8.0, run_id, sid, "r"), 3),
        )
        sites.append(record.model_dump(mode="json"))
        state.add_binding_site(record, persist=False)
    state.save()
    return {
        "status": "ok",
        "run_id": run_id,
        "binding_sites": sites,
        "warnings": [STUB_WARNING],
    }


def generate_alpha_masks(
    run_id: str,
    threshold_sigma: float = 2.5,
    smoothing: float = 1.0,
    min_blob_volume: float = 50.0,
    site_ids: Optional[list[str]] = None,
    blobs_per_site: int = 3,
) -> dict[str, Any]:
    key = f"{run_id}:thr={threshold_sigma}:smo={smoothing}:vol={min_blob_volume}"
    limit = check_limit("generate_alpha_masks", key)
    if not limit.allowed:
        return rate_limited_envelope("generate_alpha_masks", limit)

    state = load_state(run_id)
    targets = site_ids or [f"site_{i + 1:03d}" for i in range(2)]
    alpha_params = AlphaMaskParameters(
        threshold_sigma=threshold_sigma,
        smoothing=smoothing,
        min_blob_volume=min_blob_volume,
    )
    segments: list[SegmentEvidence] = []
    for site_id in targets:
        for n in range(blobs_per_site):
            sid = f"{site_id}_blob_{n + 1:03d}"
            seg = SegmentEvidence(
                segment_id=sid,
                center_xyz=(
                    round(stub_range(-15, 15, sid, "x"), 3),
                    round(stub_range(-15, 15, sid, "y"), 3),
                    round(stub_range(-15, 15, sid, "z"), 3),
                ),
                volume_angstrom3=round(stub_range(100.0, 700.0, sid, "vol"), 2),
                peak_density=round(stub_range(2.0, 6.0, sid, "peak"), 3),
                mean_density=round(stub_range(0.5, 2.5, sid, "mean"), 3),
                local_background_mean=round(stub_range(0.1, 0.6, sid, "bg"), 3),
                local_background_std=round(stub_range(0.05, 0.2, sid, "bgstd"), 3),
                nearby_residues=[],
                possible_binding_site_ids=[site_id],
                alpha_mask_parameters=alpha_params,
            )
            segments.append(seg)
            state.add_segment(seg, persist=False)
    state.save()

    return {
        "status": "ok",
        "run_id": run_id,
        "parameters": {
            "threshold_sigma": threshold_sigma,
            "smoothing": smoothing,
            "min_blob_volume": min_blob_volume,
        },
        "segments": [s.model_dump(mode="json") for s in segments],
        "warnings": [STUB_WARNING],
    }


def segment_density_regions(
    run_id: str, padding_angstrom: float = 3.0
) -> dict[str, Any]:
    """Thin wrapper exposing alpha-mask segments under the brief's name."""
    limit = check_limit("segment_density_regions", run_id)
    if not limit.allowed:
        return rate_limited_envelope("segment_density_regions", limit)
    state = load_state(run_id)
    return {
        "status": "ok",
        "run_id": run_id,
        "segment_ids": list(state.segments.keys()),
        "warnings": [STUB_WARNING],
    }


def prepare_ligand_library(
    run_id: str, ligand_input: str, ligand_ids: Optional[list[str]] = None
) -> dict[str, Any]:
    limit = check_limit("prepare_ligand_library", run_id)
    if not limit.allowed:
        return rate_limited_envelope("prepare_ligand_library", limit)
    state = load_state(run_id)
    ids = ligand_ids or ["ATP", "ADP"]
    for ligand_id in ids:
        state.add_ligand(
            LigandLibraryEntry(ligand_id=ligand_id, source=ligand_input),
            persist=False,
        )
    state.save()
    return {
        "status": "ok",
        "run_id": run_id,
        "source": ligand_input,
        "ligand_ids": ids,
        "warnings": [STUB_WARNING],
    }


def prepare_ligand_segments(
    run_id: str, ligand_id: str, n_fragments: int = 3
) -> dict[str, Any]:
    limit = check_limit("prepare_ligand_segments", f"{run_id}:{ligand_id}")
    if not limit.allowed:
        return rate_limited_envelope("prepare_ligand_segments", limit)
    case_dir(run_id)
    fragments = [f"{ligand_id}_frag_{i + 1:02d}" for i in range(n_fragments)]
    return {
        "status": "ok",
        "run_id": run_id,
        "ligand_id": ligand_id,
        "fragment_ids": fragments,
        "warnings": [STUB_WARNING],
    }


# ---------------------------------------------------------------------------
# Stage 1 — docking + minimisation
# ---------------------------------------------------------------------------


def dock_ligands_to_segment(
    run_id: str,
    segment_id: str,
    ligand_ids: list[str],
    max_poses: int = 5,
    search_radius_angstrom: float = 6.0,
    increase_pose_diversity: bool = False,
) -> dict[str, Any]:
    state = load_state(run_id)
    pose_ids: list[str] = []
    blocked: list[dict] = []
    n_poses = max(1, min(int(max_poses), 20))
    for ligand_id in ligand_ids:
        key = f"{run_id}:{segment_id}:{ligand_id}"
        limit = check_limit("dock_ligands_to_segment", key)
        if not limit.allowed:
            blocked.append(limit.to_dict())
            continue
        for n in range(n_poses):
            pid = f"{segment_id}_{ligand_id}_pose_{n + 1:03d}"
            pose_ids.append(pid)
            state.add_pose(
                PoseRecord(pose_id=pid, segment_id=segment_id, ligand_id=ligand_id),
                persist=False,
            )
    state.save()
    if not pose_ids and blocked:
        return {
            "status": "rate_limited",
            "tool_name": "dock_ligands_to_segment",
            "rate_limits": blocked,
            "warnings": ["all (segment, ligand) docking budgets exhausted"],
        }
    return {
        "status": "ok",
        "run_id": run_id,
        "segment_id": segment_id,
        "ligand_ids": ligand_ids,
        "parameters": {
            "max_poses": n_poses,
            "search_radius_angstrom": search_radius_angstrom,
            "increase_pose_diversity": increase_pose_diversity,
        },
        "pose_ids": pose_ids,
        "rate_limited_pairs": blocked,
        "warnings": [STUB_WARNING],
    }


def dock_ligand_segments_to_segment(
    run_id: str,
    segment_id: str,
    fragment_ids: list[str],
    max_poses: int = 5,
) -> dict[str, Any]:
    case_dir(run_id)
    pose_ids: list[str] = []
    blocked: list[dict] = []
    for fragment_id in fragment_ids:
        key = f"{run_id}:{segment_id}:{fragment_id}"
        limit = check_limit("dock_ligand_segments_to_segment", key)
        if not limit.allowed:
            blocked.append(limit.to_dict())
            continue
        for n in range(max_poses):
            pose_ids.append(f"{segment_id}_{fragment_id}_pose_{n + 1:03d}")
    return {
        "status": "ok" if pose_ids else "rate_limited",
        "run_id": run_id,
        "segment_id": segment_id,
        "fragment_ids": fragment_ids,
        "pose_ids": pose_ids,
        "rate_limited_pairs": blocked,
        "warnings": [STUB_WARNING],
    }


def minimise_candidate_pose(run_id: str, pose_id: str) -> dict[str, Any]:
    limit = check_limit("minimise_candidate_pose", f"{run_id}:{pose_id}")
    if not limit.allowed:
        return rate_limited_envelope("minimise_candidate_pose", limit)
    case_dir(run_id)
    return {
        "status": "ok",
        "run_id": run_id,
        "pose_id": pose_id,
        "minimised_pose_id": f"{pose_id}_min",
        "minimised_energy_kcal_mol": round(stub_range(-20.0, -2.0, pose_id, "E"), 3),
        "warnings": [STUB_WARNING],
    }


# ---------------------------------------------------------------------------
# Stage 3 — ion analysis (monolithic entrypoints)
# ---------------------------------------------------------------------------


def run_ion_search(
    run_id: str,
    confidence_threshold: float = 0.65,
    target_segment_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    limit = check_limit("run_ion_search", run_id)
    if not limit.allowed:
        return rate_limited_envelope("run_ion_search", limit)
    state = load_state(run_id)
    targets = target_segment_ids or list(state.segments.keys())[:1] or ["site_001_blob_001"]
    candidates: list[IonCandidate] = []
    for seg_id in targets:
        conf = stub_range(0.4, 0.9, seg_id, "ionconf")
        if conf < confidence_threshold:
            continue
        cand = IonCandidate(
            ion_candidate_id=f"ion_{seg_id}_001",
            segment_id=seg_id,
            proposed_identity="Mg2+",
            confidence=round(conf, 3),
            density=IonDensity(
                peak_density=round(stub_range(3.0, 6.0, seg_id, "peak"), 3),
                shape="compact_spherical",
                volume_angstrom3=round(stub_range(15.0, 30.0, seg_id, "vol"), 2),
            ),
            coordination=IonCoordination(
                geometry="octahedral_partial",
                coordination_number=stub_int(4, 6, seg_id, "cn"),
                donor_atoms=["ASP87:OD1", "ASP87:OD2", "HOH301:O"],
                mean_distance_angstrom=round(stub_range(2.0, 2.3, seg_id, "d"), 3),
                distance_std_angstrom=round(stub_range(0.05, 0.25, seg_id, "ds"), 3),
            ),
        )
        candidates.append(cand)
        state.add_ion_candidate(cand, persist=False)
    state.save()
    return {
        "status": "ok",
        "run_id": run_id,
        "confidence_threshold": confidence_threshold,
        "ion_candidates": [c.model_dump(mode="json") for c in candidates],
        "warnings": [STUB_WARNING],
    }


def build_ion_site(run_id: str, ion_candidate_id: str) -> dict[str, Any]:
    limit = check_limit("build_ion_coordination_site", f"{run_id}:{ion_candidate_id}")
    if not limit.allowed:
        return rate_limited_envelope("build_ion_coordination_site", limit)
    case_dir(run_id)
    return {
        "status": "ok",
        "run_id": run_id,
        "ion_candidate_id": ion_candidate_id,
        "built_ion_id": f"{ion_candidate_id}_built",
        "warnings": [STUB_WARNING],
    }


# ---------------------------------------------------------------------------
# Stage 4 — refinement
# ---------------------------------------------------------------------------


def run_smart_refine_2(
    run_id: str,
    assignment_set_id: str,
    pose_ids: list[str],
    ion_candidate_ids: Optional[list[str]] = None,
    optimisation_score: str = "qscore",
    final_minimise: bool = True,
) -> dict[str, Any]:
    limit = check_limit("run_smart_refine_2", f"{run_id}:{assignment_set_id}")
    if not limit.allowed:
        return rate_limited_envelope("run_smart_refine_2", limit)
    state = load_state(run_id)
    rid = f"smart_refine_2_{assignment_set_id}_{stub_int(100, 999, assignment_set_id, 'rid')}"
    outcome = RefinementOutcome(
        refinement_id=rid,
        status="ok",
        inputs=RefinementInputs(
            pose_ids=list(pose_ids), ion_candidate_ids=list(ion_candidate_ids or [])
        ),
        stability=RefinementStability(
            nan_detected=False,
            max_ligand_rmsd_from_start=round(stub_range(0.3, 1.5, rid, "lrmsd"), 3),
            max_protein_local_rmsd=round(stub_range(0.1, 0.6, rid, "prmsd"), 3),
            energy_exploded=False,
            coordination_restraints_stable=True,
        ),
        score_deltas=RefinementScoreDeltas(
            mean_ligand_ccc_delta=round(stub_range(-0.05, 0.15, rid, "ccc"), 3),
            mean_mapq_delta=round(stub_range(-0.02, 0.12, rid, "mapq"), 3),
            severe_clash_delta=stub_int(-2, 1, rid, "clash"),
            ligand_strain_delta=round(stub_range(-1.0, 1.5, rid, "strain"), 3),
        ),
        files=RefinementFiles(
            refined_model_pdb=str(case_dir(run_id) / "refine" / rid / "model.pdb"),
            refined_ligands_sdf=str(case_dir(run_id) / "refine" / rid / "ligands.sdf"),
            summary_json=str(case_dir(run_id) / "refine" / rid / "summary.json"),
        ),
    )
    state.add_refinement(outcome)
    return {
        "status": "ok",
        "run_id": run_id,
        "assignment_set_id": assignment_set_id,
        "outcome": outcome.model_dump(mode="json"),
        "warnings": [STUB_WARNING],
    }


def run_search_refine(run_id: str, pose_id: str) -> dict[str, Any]:
    limit = check_limit("run_search_refine", f"{run_id}:{pose_id}")
    if not limit.allowed:
        return rate_limited_envelope("run_search_refine", limit)
    case_dir(run_id)
    return {
        "status": "ok",
        "run_id": run_id,
        "pose_id": pose_id,
        "refined_pose_id": f"{pose_id}_sr",
        "delta_score": round(stub_range(-0.1, 0.2, pose_id, "delta"), 3),
        "warnings": [STUB_WARNING],
    }
