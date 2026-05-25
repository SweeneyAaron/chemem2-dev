from __future__ import annotations

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from .cleanup import cleanup_run_dir
from .paths import safe_run_dir
from .runtime import get_rate_limiter, get_run_root
from .tools import monolithic, primitives, reports, rendering, tweaks

mcp = FastMCP("chemem-ligand-fitting")

_REGISTERED_TOOL_NAMES: list[str] = []


def _register(*fns) -> None:
    for fn in fns:
        mcp.tool()(fn)
        _REGISTERED_TOOL_NAMES.append(fn.__name__)


# ---------------------------------------------------------------------------
# Housekeeping tools — defined inline so they can read live runtime state.
# ---------------------------------------------------------------------------


@mcp.tool()
def get_rate_limit_status(run_id: str) -> dict[str, Any]:
    """Return rate-limit usage filtered to one case (run_id)."""
    safe_run_dir(run_id, get_run_root())
    return get_rate_limiter().summary(run_id=run_id)


_REGISTERED_TOOL_NAMES.append("get_rate_limit_status")


@mcp.tool()
def cleanup_intermediate_files(
    run_id: str,
    dry_run: bool = True,
    extra_keep: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Delete temporary files while preserving inputs, decision logs, and final outputs."""
    result = get_rate_limiter().check_and_increment(
        tool_name="cleanup_intermediate_files", key=run_id
    )
    if not result.allowed:
        return {
            "status": "rate_limited",
            "rate_limit": result.to_dict(),
            "warnings": ["cleanup rate limit exhausted for this case"],
        }
    run_dir = safe_run_dir(run_id, get_run_root())
    report = cleanup_run_dir(run_dir, dry_run=dry_run, extra_keep=extra_keep)
    return report.to_dict()


_REGISTERED_TOOL_NAMES.append("cleanup_intermediate_files")


# ---------------------------------------------------------------------------
# Tier 1 — monolithic protocol wrappers
# ---------------------------------------------------------------------------

_register(
    monolithic.prepare_case,
    monolithic.generate_binding_sites,
    monolithic.generate_alpha_masks,
    monolithic.segment_density_regions,
    monolithic.prepare_ligand_library,
    monolithic.prepare_ligand_segments,
    monolithic.dock_ligands_to_segment,
    monolithic.dock_ligand_segments_to_segment,
    monolithic.minimise_candidate_pose,
    monolithic.run_ion_search,
    monolithic.build_ion_site,
    monolithic.run_smart_refine_2,
    monolithic.run_search_refine,
)

# ---------------------------------------------------------------------------
# Tier 2 — atomic primitives
# ---------------------------------------------------------------------------

_register(
    primitives.compute_ccc,
    primitives.compute_mi,
    primitives.compute_mapq,
    primitives.compute_sci,
    primitives.compute_qscore_per_atom,
    primitives.crop_map_around_segment,
    primitives.mask_map_by_segment,
    primitives.subtract_ligand_density,
    primitives.compute_ligand_volume,
    primitives.compute_segment_volume,
    primitives.analyse_clashes,
    primitives.analyse_ligand_strain,
    primitives.analyse_interactions,
    primitives.detect_ion_like_density,
    primitives.analyse_coordination_geometry,
    primitives.propose_ion_identity,
    primitives.score_ion_assignment,
    primitives.monitor_refinement_stability,
    primitives.score_refined_model,
    primitives.analyse_refined_ligands,
    primitives.analyse_refined_ions,
    primitives.analyse_residual_density,
    primitives.compare_refined_to_initial,
    primitives.summarise_refinement_outcome,
    primitives.summarise_segment_candidates,
)

# ---------------------------------------------------------------------------
# Tier 3 — parameter-tweak / rerun
# ---------------------------------------------------------------------------

_register(
    tweaks.increase_docking_diversity,
    tweaks.narrow_docking_search,
    tweaks.enable_flexible_rings,
    tweaks.loosen_alpha_mask,
    tweaks.tighten_alpha_mask,
    tweaks.rerun_smart_refine_stronger_density,
    tweaks.rerun_smart_refine_softer_restraints,
    tweaks.rerun_smart_refine_alt_score,
    tweaks.rerun_ion_search_lower_threshold,
)

# ---------------------------------------------------------------------------
# Reporting + rendering
# ---------------------------------------------------------------------------

_register(
    reports.generate_final_report,
    reports.summarise_case_state,
    reports.get_case_disk_usage,
    rendering.render_snapshot,
    rendering.render_ion_site_snapshot,
    rendering.render_refined_snapshots,
)


def registered_tool_names() -> list[str]:
    """Return the list of MCP-registered tool names (for tests)."""
    return list(_REGISTERED_TOOL_NAMES)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
