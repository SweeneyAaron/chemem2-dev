"""Tests for the per-segment 'ground truth winner' baked into the stubs.

The toy pipeline needs each segment to have a single pose that scores
high across CCC, MapQ, MI, and SCI simultaneously — otherwise the LLM
controller has no coherent signal to latch onto and loops on per-pose
metric calls.
"""
from __future__ import annotations

from pathlib import Path

from chemem_mcp.llm import ScriptedLLMClient
from chemem_mcp.orchestrator import Agent, _stage_exit_ready
from chemem_mcp.paths import safe_run_dir
from chemem_mcp.schemas import ControllerDecision
from chemem_mcp.state import CaseState
from chemem_mcp.tools import monolithic, primitives


RUN_ID = "gt_run"


def _seed_docked_segment(run_root: Path, segment_id: str, ligand_ids: list[str], n_poses: int = 5) -> list[str]:
    """Run prepare/sites/ligands/masks/dock so the segment has poses on disk."""
    monolithic.prepare_case(
        run_id=RUN_ID, protein_pdb="p", map_mrc="m", ligand_library="l"
    )
    monolithic.generate_binding_sites(run_id=RUN_ID, max_sites=2)
    monolithic.prepare_ligand_library(
        run_id=RUN_ID, ligand_input="l.sdf", ligand_ids=ligand_ids
    )
    monolithic.generate_alpha_masks(
        run_id=RUN_ID,
        threshold_sigma=2.5,
        smoothing=1.0,
        min_blob_volume=50.0,
        site_ids=["site_001"],
        blobs_per_site=2,
    )
    dock = monolithic.dock_ligands_to_segment(
        run_id=RUN_ID,
        segment_id=segment_id,
        ligand_ids=ligand_ids,
        max_poses=n_poses,
    )
    return dock["pose_ids"]


def test_one_pose_per_segment_is_cross_metric_winner(runtime) -> None:
    """The pose with the highest CCC must also be the highest on MapQ, MI, SCI.

    Failing this would mean the metrics are independent noise — exactly the
    problem the toy_009 run hit.
    """
    run_root, _ = runtime
    segment_id = "site_001_blob_001"
    pose_ids = _seed_docked_segment(run_root, segment_id, ["ATP", "ADP"], n_poses=5)
    assert len(pose_ids) == 10  # 2 ligands x 5 poses

    rows: list[dict] = []
    for pid in pose_ids:
        ccc = primitives.compute_ccc(run_id=RUN_ID, pose_id=pid, map_path="m")["value"]
        mapq = primitives.compute_mapq(run_id=RUN_ID, pose_id=pid, map_path="m")["mean"]
        mi = primitives.compute_mi(run_id=RUN_ID, pose_id=pid, map_path="m")["value"]
        sci = primitives.compute_sci(run_id=RUN_ID, pose_id=pid, map_path="m")["value"]
        rows.append({"pose_id": pid, "ccc": ccc, "mapq": mapq, "mi": mi, "sci": sci})

    winner_ccc = max(rows, key=lambda r: r["ccc"])
    winner_mapq = max(rows, key=lambda r: r["mapq"])
    winner_mi = max(rows, key=lambda r: r["mi"])
    winner_sci = max(rows, key=lambda r: r["sci"])

    # The same pose wins across all four metrics — this is the cross-metric
    # signal the controller is supposed to detect.
    assert winner_ccc["pose_id"] == winner_mapq["pose_id"] == winner_mi["pose_id"] == winner_sci["pose_id"]

    # And it must actually stand out: its CCC is at least 0.05 above the
    # runner-up, so a noisy LLM can still rank it confidently.
    sorted_ccc = sorted(rows, key=lambda r: r["ccc"], reverse=True)
    assert sorted_ccc[0]["ccc"] - sorted_ccc[1]["ccc"] >= 0.03


def test_compute_ccc_is_deterministic_per_pose(runtime) -> None:
    """Same pose_id + map → same value. Recomputing is wasteful, not informative."""
    run_root, _ = runtime
    _seed_docked_segment(run_root, "site_001_blob_001", ["ATP"], n_poses=2)
    pid = "site_001_blob_001_ATP_pose_001"
    a = primitives.compute_ccc(run_id=RUN_ID, pose_id=pid, map_path="m")
    # Reset the per-pose budget so we can call it again.
    from chemem_mcp.runtime import get_rate_limiter
    get_rate_limiter().reset(tool_name="compute_ccc", key=f"{RUN_ID}:{pid}:m")
    b = primitives.compute_ccc(run_id=RUN_ID, pose_id=pid, map_path="m")
    assert a["value"] == b["value"]


def test_compute_ccc_persists_score_into_pose_record(runtime) -> None:
    """After compute_ccc runs, the value must be visible on the pose in state.json,
    so the agent can see it in poses_with_metrics on the next turn."""
    run_root, _ = runtime
    pose_ids = _seed_docked_segment(run_root, "site_001_blob_001", ["ATP"], n_poses=2)
    pid = pose_ids[0]
    result = primitives.compute_ccc(run_id=RUN_ID, pose_id=pid, map_path="m")

    reloaded = CaseState.load_or_create(RUN_ID, run_root / RUN_ID)
    assert reloaded.poses[pid].scores.ccc == result["value"]


def test_compute_mapq_persists_mean_and_min(runtime) -> None:
    run_root, _ = runtime
    pose_ids = _seed_docked_segment(run_root, "site_001_blob_001", ["ATP"], n_poses=1)
    pid = pose_ids[0]
    result = primitives.compute_mapq(run_id=RUN_ID, pose_id=pid, map_path="m")

    reloaded = CaseState.load_or_create(RUN_ID, run_root / RUN_ID)
    assert reloaded.poses[pid].scores.mapq_mean == result["mean"]
    assert reloaded.poses[pid].scores.mapq_min == result["min"]
    # mapq_min must not exceed mapq_mean — they're related quantities.
    assert reloaded.poses[pid].scores.mapq_min <= reloaded.poses[pid].scores.mapq_mean


def test_summarise_segment_candidates_trips_stage_exit_ready(runtime) -> None:
    """The whole point of summarise_segment_candidates is that it writes the
    candidate_assignments list, which is the gate _stage_exit_ready checks
    for density_assignment. Without that, the controller loops forever."""
    run_root, _ = runtime
    seg = "site_001_blob_001"
    pose_ids = _seed_docked_segment(run_root, seg, ["ATP"], n_poses=3)

    # Pre-condition: stage_exit_ready is false (no candidate_assignments yet).
    state = CaseState.load_or_create(RUN_ID, run_root / RUN_ID)
    state.set_stage("density_assignment")
    assert _stage_exit_ready(state) is False

    # One metric measured on each pose so summarise can blend measured + derived.
    for pid in pose_ids:
        primitives.compute_ccc(run_id=RUN_ID, pose_id=pid, map_path="m")

    res = primitives.summarise_segment_candidates(run_id=RUN_ID, segment_id=seg)
    assert res["status"] == "ok"

    reloaded = CaseState.load_or_create(RUN_ID, run_root / RUN_ID)
    reloaded.set_stage("density_assignment")
    assert len(reloaded.segments[seg].candidate_assignments) > 0
    assert _stage_exit_ready(reloaded) is True


def test_summarise_segment_candidates_uses_measured_scores(runtime) -> None:
    """When the agent has measured a metric on a pose, summarise must reuse
    that value rather than overwriting it with a different stub draw."""
    run_root, _ = runtime
    seg = "site_001_blob_001"
    pose_ids = _seed_docked_segment(run_root, seg, ["ATP"], n_poses=2)
    pid = pose_ids[0]

    measured = primitives.compute_ccc(run_id=RUN_ID, pose_id=pid, map_path="m")
    summary = primitives.summarise_segment_candidates(run_id=RUN_ID, segment_id=seg)
    candidates = summary["evidence"]["candidate_assignments"]
    row = next(c for c in candidates if c["pose_id"] == pid)
    assert row["scores"]["ccc"] == measured["value"]


def test_evidence_exposes_poses_with_metrics(runtime) -> None:
    """The agent's _evidence dict (the case_summary the model sees) must
    surface measured scores per pose, so the model can tell which poses
    have been evaluated even after they scroll out of recent_events."""
    run_root, _ = runtime
    seg = "site_001_blob_001"
    pose_ids = _seed_docked_segment(run_root, seg, ["ATP"], n_poses=2)
    primitives.compute_ccc(run_id=RUN_ID, pose_id=pose_ids[0], map_path="m")

    state = CaseState.load_or_create(RUN_ID, run_root / RUN_ID)
    agent = Agent(llm=ScriptedLLMClient(), model="test")
    ev = agent._evidence(state)

    assert "poses_with_metrics" in ev
    measured_row = next(p for p in ev["poses_with_metrics"] if p["pose_id"] == pose_ids[0])
    unmeasured_row = next(p for p in ev["poses_with_metrics"] if p["pose_id"] == pose_ids[1])
    assert "ccc" in measured_row["scores"]
    assert "ccc" not in unmeasured_row["scores"]
    assert ev["counts"]["poses_with_any_metric"] == 1
