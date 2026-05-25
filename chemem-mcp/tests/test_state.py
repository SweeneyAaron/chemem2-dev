from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemem_mcp.paths import safe_run_dir
from chemem_mcp.schemas import (
    SCHEMA_VERSION,
    AlphaMaskParameters,
    BindingSiteRecord,
    ControllerDecision,
    LigandLibraryEntry,
    PoseRecord,
    SegmentEvidence,
)
from chemem_mcp.state import (
    DECISION_LOG_FILENAME,
    STATE_FILENAME,
    CaseState,
)

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"


def _make_state(tmp_path: Path) -> CaseState:
    run_dir = safe_run_dir("case001", tmp_path)
    return CaseState.load_or_create("case001", run_dir)


def _make_decision(step: int = 1) -> ControllerDecision:
    raw = json.loads((GOLDEN_DIR / "controller_decision.json").read_text())
    raw["controller_step_id"] = f"controller_{step:04d}"
    return ControllerDecision.model_validate(raw)


def test_load_or_create_writes_state_file(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    assert (Path(state.run_dir) / STATE_FILENAME).exists()
    assert state.run_id == "case001"
    assert state.schema_version == SCHEMA_VERSION
    assert state.stage == "preparation"


def test_load_or_create_is_idempotent(tmp_path: Path) -> None:
    a = _make_state(tmp_path)
    a.controller_step_count = 7
    a.save()
    b = CaseState.load_or_create("case001", Path(a.run_dir))
    assert b.controller_step_count == 7


def test_save_is_atomic(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    state.save()
    # leftover temp files should not linger
    leftovers = list(Path(state.run_dir).glob(".state.*.tmp"))
    assert leftovers == []


def test_log_decision_appends_and_advances_stage(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    d1 = _make_decision(1)
    d1.stage = "density_assignment"
    state.log_decision(d1)
    d2 = _make_decision(2)
    d2.stage = "ion_analysis"
    state.log_decision(d2)
    assert state.controller_step_count == 2
    assert state.stage == "ion_analysis"

    log_path = Path(state.run_dir) / DECISION_LOG_FILENAME
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    entries = [json.loads(line) for line in lines]
    assert entries[0]["kind"] == "decision"
    assert entries[0]["payload"]["controller_step_id"] == "controller_0001"
    assert entries[1]["payload"]["controller_step_id"] == "controller_0002"


def test_log_tool_call_appended(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    state.log_tool_call(
        "dock_ligands_to_segment",
        {"site_id": "site_001", "ligand_id": "ATP"},
        {"pose_count": 12, "best_dock_score": -8.4},
    )
    assert state.tool_call_count == 1
    lines = (Path(state.run_dir) / DECISION_LOG_FILENAME).read_text().strip().splitlines()
    entry = json.loads(lines[0])
    assert entry["kind"] == "tool_call"
    assert entry["payload"]["tool_name"] == "dock_ligands_to_segment"
    assert entry["payload"]["result_summary"]["pose_count"] == 12


def test_decision_log_survives_reload(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    state.log_decision(_make_decision(1))
    state.log_decision(_make_decision(2))
    reloaded = CaseState.load_or_create("case001", Path(state.run_dir))
    log_path = Path(reloaded.run_dir) / DECISION_LOG_FILENAME
    assert len(log_path.read_text().strip().splitlines()) == 2
    assert reloaded.controller_step_count == 2


def test_bump_tweak_increments_and_persists(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    assert state.get_tweak_depth("increase_docking_diversity", "site_001:ATP") == 0
    assert state.bump_tweak("increase_docking_diversity", "site_001:ATP") == 1
    assert state.bump_tweak("increase_docking_diversity", "site_001:ATP") == 2

    reloaded = CaseState.load_or_create("case001", Path(state.run_dir))
    assert reloaded.get_tweak_depth("increase_docking_diversity", "site_001:ATP") == 2


def test_tweak_scopes_independent(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    state.bump_tweak("increase_docking_diversity", "site_001:ATP")
    state.bump_tweak("increase_docking_diversity", "site_002:ATP")
    state.bump_tweak("loosen_alpha_mask", "site_001")
    assert state.get_tweak_depth("increase_docking_diversity", "site_001:ATP") == 1
    assert state.get_tweak_depth("increase_docking_diversity", "site_002:ATP") == 1
    assert state.get_tweak_depth("loosen_alpha_mask", "site_001") == 1


def test_reset_tweak_clears_only_target(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    state.bump_tweak("increase_docking_diversity", "site_001:ATP")
    state.bump_tweak("loosen_alpha_mask", "site_001")
    state.reset_tweak("increase_docking_diversity", "site_001:ATP")
    assert state.get_tweak_depth("increase_docking_diversity", "site_001:ATP") == 0
    assert state.get_tweak_depth("loosen_alpha_mask", "site_001") == 1


def test_tweak_key_rejects_separator_collision(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    with pytest.raises(ValueError):
        state.bump_tweak("bad::tool", "scope")
    with pytest.raises(ValueError):
        state.bump_tweak("tool", "bad::scope")


def test_add_binding_site_persists(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    site = BindingSiteRecord(
        site_id="site_001", center_xyz=(1.0, 2.0, 3.0), radius_angstrom=6.5
    )
    state.add_binding_site(site)
    reloaded = CaseState.load_or_create("case001", Path(state.run_dir))
    assert "site_001" in reloaded.binding_sites
    assert reloaded.binding_sites["site_001"].radius_angstrom == 6.5


def test_add_ligand_persists(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    state.add_ligand(LigandLibraryEntry(ligand_id="ATP", source="ligs.sdf"))
    state.add_ligand(LigandLibraryEntry(ligand_id="ADP", source="ligs.sdf"))
    reloaded = CaseState.load_or_create("case001", Path(state.run_dir))
    assert set(reloaded.ligand_library.keys()) == {"ATP", "ADP"}


def test_add_pose_persists(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    state.add_pose(
        PoseRecord(pose_id="seg1_ATP_pose_001", segment_id="seg1", ligand_id="ATP")
    )
    reloaded = CaseState.load_or_create("case001", Path(state.run_dir))
    assert "seg1_ATP_pose_001" in reloaded.poses
    assert reloaded.poses["seg1_ATP_pose_001"].ligand_id == "ATP"


def test_add_segment_persists(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    seg = SegmentEvidence(
        segment_id="site_001_blob_001",
        center_xyz=(1.0, 2.0, 3.0),
        volume_angstrom3=100.0,
        peak_density=3.5,
        mean_density=1.2,
        alpha_mask_parameters=AlphaMaskParameters(
            threshold_sigma=2.0, smoothing=1.0, min_blob_volume=50.0
        ),
    )
    state.add_segment(seg)
    reloaded = CaseState.load_or_create("case001", Path(state.run_dir))
    assert "site_001_blob_001" in reloaded.segments
    assert reloaded.segments["site_001_blob_001"].peak_density == 3.5


def test_add_binding_site_persists(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    site = BindingSiteRecord(
        site_id="site_001", center_xyz=(1.0, 2.0, 3.0), radius_angstrom=6.0
    )
    state.add_binding_site(site)
    reloaded = CaseState.load_or_create("case001", Path(state.run_dir))
    assert "site_001" in reloaded.binding_sites
    assert reloaded.binding_sites["site_001"].radius_angstrom == 6.0


def test_add_ligand_persists(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    state.add_ligand(LigandLibraryEntry(ligand_id="ATP", source="lib.sdf"))
    reloaded = CaseState.load_or_create("case001", Path(state.run_dir))
    assert "ATP" in reloaded.ligand_library
    assert reloaded.ligand_library["ATP"].source == "lib.sdf"


def test_add_pose_persists(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    state.add_pose(
        PoseRecord(pose_id="seg_blob_ATP_pose_001", segment_id="seg_blob", ligand_id="ATP")
    )
    reloaded = CaseState.load_or_create("case001", Path(state.run_dir))
    assert "seg_blob_ATP_pose_001" in reloaded.poses
    assert reloaded.poses["seg_blob_ATP_pose_001"].ligand_id == "ATP"


def test_set_stage_persists(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    state.set_stage("smart_refine_2")
    reloaded = CaseState.load_or_create("case001", Path(state.run_dir))
    assert reloaded.stage == "smart_refine_2"


def test_updated_at_changes_on_save(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    first = state.updated_at
    # bump_tweak persists
    state.bump_tweak("loosen_alpha_mask", "site_001")
    assert state.updated_at != first
