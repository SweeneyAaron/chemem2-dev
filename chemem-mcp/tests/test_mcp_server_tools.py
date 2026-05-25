"""Sanity checks on MCP tool registration."""
from __future__ import annotations

import asyncio

from chemem_mcp import mcp_server


EXPECTED_MIN_TOOL_COUNT = 50


def _list_tool_names() -> list[str]:
    tools = asyncio.run(mcp_server.mcp.list_tools())
    return [t.name for t in tools]


def test_registered_count_matches_mcp_list() -> None:
    listed = _list_tool_names()
    assert sorted(listed) == sorted(mcp_server.registered_tool_names())


def test_minimum_tool_surface() -> None:
    listed = set(_list_tool_names())
    assert len(listed) >= EXPECTED_MIN_TOOL_COUNT, listed


def test_core_tools_present() -> None:
    listed = set(_list_tool_names())
    required = {
        # housekeeping
        "get_rate_limit_status",
        "cleanup_intermediate_files",
        # tier 1
        "prepare_case",
        "generate_binding_sites",
        "generate_alpha_masks",
        "dock_ligands_to_segment",
        "run_smart_refine_2",
        "run_ion_search",
        "build_ion_site",
        # tier 2
        "compute_ccc",
        "compute_mi",
        "compute_mapq",
        "subtract_ligand_density",
        "summarise_segment_candidates",
        # tier 3
        "increase_docking_diversity",
        "loosen_alpha_mask",
        "tighten_alpha_mask",
        "rerun_smart_refine_softer_restraints",
        "rerun_smart_refine_alt_score",
        "rerun_ion_search_lower_threshold",
        # reporting
        "generate_final_report",
        "summarise_case_state",
    }
    missing = required - listed
    assert not missing, f"missing tools: {missing}"


def test_no_duplicate_tool_names() -> None:
    listed = _list_tool_names()
    assert len(listed) == len(set(listed))
