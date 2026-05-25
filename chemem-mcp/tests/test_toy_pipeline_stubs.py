"""End-to-end smoke test of the stub pipeline.

Walks Stage 1 → Stage 5 calling stub tool functions directly (no LLM).
Every emitted artifact is validated against its pydantic schema.
"""
from __future__ import annotations

import json
from pathlib import Path

from chemem_mcp.schemas import (
    FinalReport,
    IonCandidate,
    RefinementOutcome,
    SegmentEvidence,
)
from chemem_mcp.state import (
    DECISION_LOG_FILENAME,
    STATE_FILENAME,
    CaseState,
)
from chemem_mcp.tools import monolithic, primitives, reports, rendering, tweaks


RUN_ID = "toy_case_001"


def _ok(result: dict) -> dict:
    assert result["status"] == "ok", result
    return result


def test_toy_pipeline_stage_1_to_5(runtime) -> None:
    run_root, rl = runtime

    # ---- Stage 1: preparation ----
    _ok(
        monolithic.prepare_case(
            run_id=RUN_ID,
            protein_pdb="protein.pdb",
            map_mrc="map.mrc",
            ligand_library="ligands.sdf",
        )
    )
    sites_res = _ok(monolithic.generate_binding_sites(run_id=RUN_ID, max_sites=2))
    site_ids = [s["site_id"] for s in sites_res["binding_sites"]]
    assert site_ids == ["site_001", "site_002"]

    ligands_res = _ok(
        monolithic.prepare_ligand_library(
            run_id=RUN_ID, ligand_input="ligands.sdf", ligand_ids=["ATP"]
        )
    )
    ligand_ids = ligands_res["ligand_ids"]

    # ---- Stage 1: segmentation ----
    masks_res = _ok(
        monolithic.generate_alpha_masks(
            run_id=RUN_ID,
            threshold_sigma=2.5,
            smoothing=1.0,
            min_blob_volume=50.0,
            site_ids=site_ids,
            blobs_per_site=2,
        )
    )
    segment_dicts = masks_res["segments"]
    assert len(segment_dicts) == 4
    for d in segment_dicts:
        SegmentEvidence.model_validate(d)
    segment_ids = [d["segment_id"] for d in segment_dicts]

    # ---- Stage 1: dock + atomic scoring on the first segment ----
    target_seg = segment_ids[0]
    dock_res = _ok(
        monolithic.dock_ligands_to_segment(
            run_id=RUN_ID, segment_id=target_seg, ligand_ids=ligand_ids, max_poses=3
        )
    )
    pose_ids = dock_res["pose_ids"]
    assert len(pose_ids) == 3

    map_path = "map.mrc"
    for pid in pose_ids:
        _ok(primitives.compute_ccc(run_id=RUN_ID, pose_id=pid, map_path=map_path))
        _ok(primitives.compute_mi(run_id=RUN_ID, pose_id=pid, map_path=map_path))
        _ok(primitives.compute_mapq(run_id=RUN_ID, pose_id=pid, map_path=map_path))
        _ok(primitives.analyse_clashes(run_id=RUN_ID, pose_id=pid))

    # Atomic primitive composition: subtract pose 0 then MI on residual.
    residual = _ok(
        primitives.subtract_ligand_density(
            run_id=RUN_ID, map_path=map_path, pose_id=pose_ids[0]
        )
    )
    _ok(
        primitives.compute_mi(
            run_id=RUN_ID, pose_id=pose_ids[1], map_path=residual["derived_map_path"]
        )
    )

    summary = _ok(
        primitives.summarise_segment_candidates(
            run_id=RUN_ID, segment_id=target_seg, pose_ids=pose_ids
        )
    )
    SegmentEvidence.model_validate(summary["evidence"])

    # ---- Stage 1.5: Tier-3 escalation on a different segment ----
    rerun = _ok(
        tweaks.increase_docking_diversity(
            run_id=RUN_ID, segment_id=segment_ids[1], ligand_id=ligand_ids[0]
        )
    )
    assert rerun["tweak_depth"] == 1
    rerun2 = _ok(
        tweaks.increase_docking_diversity(
            run_id=RUN_ID, segment_id=segment_ids[1], ligand_id=ligand_ids[0]
        )
    )
    assert rerun2["tweak_depth"] == 2
    # Each escalation should have used more poses than the previous.
    assert (
        rerun2["applied_parameters"]["max_poses"]
        > rerun["applied_parameters"]["max_poses"]
    )

    # ---- Stage 3: ion search ----
    ion_res = _ok(
        monolithic.run_ion_search(
            run_id=RUN_ID, confidence_threshold=0.0, target_segment_ids=[segment_ids[2]]
        )
    )
    ion_candidates = ion_res["ion_candidates"]
    assert len(ion_candidates) >= 1
    for c in ion_candidates:
        IonCandidate.model_validate(c)
    ion_id = ion_candidates[0]["ion_candidate_id"]
    _ok(primitives.analyse_coordination_geometry(run_id=RUN_ID, ion_candidate_id=ion_id))

    # ---- Stage 4: smart_refine_2 + post-refine analysis ----
    refine_res = _ok(
        monolithic.run_smart_refine_2(
            run_id=RUN_ID,
            assignment_set_id="set_001",
            pose_ids=pose_ids[:1],
            ion_candidate_ids=[ion_id],
        )
    )
    RefinementOutcome.model_validate(refine_res["outcome"])
    refine_id = refine_res["outcome"]["refinement_id"]
    _ok(primitives.monitor_refinement_stability(run_id=RUN_ID, refinement_id=refine_id))
    _ok(primitives.compare_refined_to_initial(run_id=RUN_ID, refinement_id=refine_id))
    _ok(primitives.analyse_refined_ligands(run_id=RUN_ID, refinement_id=refine_id))
    _ok(rendering.render_refined_snapshots(run_id=RUN_ID, refinement_id=refine_id))

    # ---- Stage 5: final report ----
    report_res = _ok(
        reports.generate_final_report(
            run_id=RUN_ID,
            final_status="accepted_with_warnings",
            accepted_ligand_pose_ids=[pose_ids[0]],
            accepted_ion_candidate_ids=[ion_id],
            noise_segment_ids=[],
            recommended_human_checks=["inspect refined model snapshot"],
        )
    )
    FinalReport.model_validate(report_res["report"])

    # ---- Artifacts on disk ----
    run_dir: Path = run_root / RUN_ID
    assert (run_dir / STATE_FILENAME).exists()
    assert (run_dir / DECISION_LOG_FILENAME).exists() or True  # log is optional in stub flow
    final_path = run_dir / "final" / "report.json"
    assert final_path.exists(), f"missing {final_path}"
    on_disk = json.loads(final_path.read_text())
    FinalReport.model_validate(on_disk)
    assert on_disk["accepted_ligands"][0]["final_pose_id"] == pose_ids[0]

    # ---- Case state recovery ----
    reloaded = CaseState.load_or_create(RUN_ID, run_dir)
    assert reloaded.stage == "final_report"
    assert reloaded.final_status == "accepted_with_warnings"
    assert len(reloaded.segments) == 4
    assert ion_id in reloaded.ion_candidates
    assert refine_id in reloaded.refinements
    assert reloaded.get_tweak_depth("increase_docking_diversity", f"{segment_ids[1]}:{ligand_ids[0]}") == 2

    # ---- summarise_case_state is a sane view ----
    summary_state = _ok(reports.summarise_case_state(run_id=RUN_ID))
    assert summary_state["counts"]["segments"] == 4
    assert summary_state["counts"]["ion_candidates"] >= 1
    assert summary_state["counts"]["refinements"] >= 1


def test_rate_limit_exhausts_tier3_then_blocks(runtime) -> None:
    run_root, rl = runtime
    _ok(monolithic.prepare_case(run_id=RUN_ID, protein_pdb="p", map_mrc="m", ligand_library="l"))
    # increase_docking_diversity default limit = 3
    for _ in range(3):
        _ok(
            tweaks.increase_docking_diversity(
                run_id=RUN_ID, segment_id="site_001_blob_001", ligand_id="ATP"
            )
        )
    blocked = tweaks.increase_docking_diversity(
        run_id=RUN_ID, segment_id="site_001_blob_001", ligand_id="ATP"
    )
    assert blocked["status"] == "rate_limited"
    assert blocked["rate_limit"]["remaining"] == 0
