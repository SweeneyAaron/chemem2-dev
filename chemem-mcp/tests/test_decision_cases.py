"""Validate the 10 Block 3 decision-case fixtures.

These fixtures are the frozen Stage-1 inputs that Block 4's agent battery
will consume. For now we only assert structural validity + that the
expected tool names refer to tools registered with the MCP server.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemem_mcp import mcp_server
from chemem_mcp.schemas import DecisionCase

FIXTURE_DIR = Path(__file__).parent / "data" / "decision_cases"
EXPECTED_FIXTURE_COUNT = 10


@pytest.fixture(scope="module")
def fixture_paths() -> list[Path]:
    paths = sorted(FIXTURE_DIR.glob("*.json"))
    assert len(paths) == EXPECTED_FIXTURE_COUNT, paths
    return paths


def test_fixture_count(fixture_paths: list[Path]) -> None:
    assert len(fixture_paths) == EXPECTED_FIXTURE_COUNT


@pytest.mark.parametrize(
    "filename",
    [
        "01_single_clear_ligand_density.json",
        "02_ligand_core_fits_tail_weak.json",
        "03_multiple_ligands_same_segment.json",
        "04_density_solvent_or_noise.json",
        "05_ion_like_spherical_density.json",
        "06_severe_clashes_good_map_score.json",
        "07_refinement_improves_score_breaks_chemistry.json",
        "08_refinement_generates_nans.json",
        "09_alpha_mask_misses_weak_density.json",
        "10_docking_diversity_too_low_elongated.json",
    ],
)
def test_fixture_validates(filename: str) -> None:
    raw = json.loads((FIXTURE_DIR / filename).read_text())
    parsed = DecisionCase.model_validate(raw)
    assert parsed.case_id == raw["case_id"]


def test_unique_case_ids(fixture_paths: list[Path]) -> None:
    case_ids = [
        DecisionCase.model_validate(json.loads(p.read_text())).case_id
        for p in fixture_paths
    ]
    assert len(case_ids) == len(set(case_ids))


def test_expected_tools_are_registered(fixture_paths: list[Path]) -> None:
    """Every named tool in an expected_decision must exist on the server."""
    registered = set(mcp_server.registered_tool_names())
    for p in fixture_paths:
        case = DecisionCase.model_validate(json.loads(p.read_text()))
        tool = case.expected_decision.tool_name
        if tool is None:
            continue  # advance_stage / stop_ambiguous etc.
        assert tool in registered, f"{case.case_id} references unknown tool {tool}"


def test_stages_are_within_known_set(fixture_paths: list[Path]) -> None:
    seen = set()
    for p in fixture_paths:
        case = DecisionCase.model_validate(json.loads(p.read_text()))
        seen.add(case.stage)
    # at minimum we want both density_assignment and smart_refine_2 covered
    assert "density_assignment" in seen
    assert "smart_refine_2" in seen
