"""Tier 3 — named parameter-tweak / rerun tools.

Each tool applies a fixed input-mutation strategy that escalates with the
current tweak depth (stored in CaseState) and re-invokes the underlying
Tier-1 monolithic tool. Rate-limit keys match the (tool, scope) granularity
documented in the plan.
"""
from __future__ import annotations

from typing import Any

from ._common import check_limit, load_state, rate_limited_envelope
from . import monolithic


def _docking_params_for_depth(depth: int) -> dict[str, Any]:
    base_max_poses = 5
    base_radius = 6.0
    return {
        "max_poses": base_max_poses + 5 * depth,
        "search_radius_angstrom": round(base_radius + 2.0 * depth, 2),
        "increase_pose_diversity": True,
    }


def _alpha_params_for_depth(direction: str, depth: int) -> dict[str, float]:
    base = {"threshold_sigma": 2.5, "smoothing": 1.0, "min_blob_volume": 50.0}
    if direction == "loosen":
        return {
            "threshold_sigma": round(base["threshold_sigma"] - 0.3 * depth, 3),
            "smoothing": round(base["smoothing"] + 0.3 * depth, 3),
            "min_blob_volume": round(max(10.0, base["min_blob_volume"] - 10.0 * depth), 3),
        }
    return {
        "threshold_sigma": round(base["threshold_sigma"] + 0.3 * depth, 3),
        "smoothing": round(max(0.1, base["smoothing"] - 0.2 * depth), 3),
        "min_blob_volume": round(base["min_blob_volume"] + 10.0 * depth, 3),
    }


def _smart_refine_weights(direction: str, depth: int) -> dict[str, Any]:
    if direction == "stronger_density":
        return {"density_weight_multiplier": round(1.5 + 0.5 * depth, 3)}
    if direction == "softer_restraints":
        return {"restraint_weight_multiplier": round(max(0.1, 1.0 - 0.25 * depth), 3)}
    return {}


# ---------------------------------------------------------------------------
# Docking tweaks
# ---------------------------------------------------------------------------


def increase_docking_diversity(
    run_id: str, segment_id: str, ligand_id: str
) -> dict[str, Any]:
    scope = f"{segment_id}:{ligand_id}"
    key = f"{run_id}:{scope}"
    limit = check_limit("increase_docking_diversity", key)
    if not limit.allowed:
        return rate_limited_envelope("increase_docking_diversity", limit)

    state = load_state(run_id)
    depth = state.bump_tweak("increase_docking_diversity", scope)
    params = _docking_params_for_depth(depth)

    rerun = monolithic.dock_ligands_to_segment(
        run_id=run_id,
        segment_id=segment_id,
        ligand_ids=[ligand_id],
        max_poses=params["max_poses"],
        search_radius_angstrom=params["search_radius_angstrom"],
        increase_pose_diversity=params["increase_pose_diversity"],
    )

    return {
        "status": "ok",
        "tool_name": "increase_docking_diversity",
        "tool_tier": "parameter_tweak",
        "scope": scope,
        "tweak_depth": depth,
        "applied_parameters": params,
        "rerun": rerun,
        "rate_limit": limit.to_dict(),
    }


def narrow_docking_search(
    run_id: str,
    segment_id: str,
    ligand_id: str,
    target_radius_angstrom: float = 4.0,
) -> dict[str, Any]:
    scope = f"{segment_id}:{ligand_id}"
    key = f"{run_id}:{scope}"
    limit = check_limit("narrow_docking_search", key)
    if not limit.allowed:
        return rate_limited_envelope("narrow_docking_search", limit)

    state = load_state(run_id)
    depth = state.bump_tweak("narrow_docking_search", scope)
    radius = max(2.5, target_radius_angstrom - 0.5 * (depth - 1))

    rerun = monolithic.dock_ligands_to_segment(
        run_id=run_id,
        segment_id=segment_id,
        ligand_ids=[ligand_id],
        search_radius_angstrom=radius,
        max_poses=5,
        increase_pose_diversity=False,
    )

    return {
        "status": "ok",
        "tool_name": "narrow_docking_search",
        "tool_tier": "parameter_tweak",
        "scope": scope,
        "tweak_depth": depth,
        "applied_parameters": {"search_radius_angstrom": round(radius, 3)},
        "rerun": rerun,
        "rate_limit": limit.to_dict(),
    }


def enable_flexible_rings(
    run_id: str, segment_id: str, ligand_id: str
) -> dict[str, Any]:
    scope = f"{segment_id}:{ligand_id}"
    key = f"{run_id}:{scope}"
    limit = check_limit("enable_flexible_rings", key)
    if not limit.allowed:
        return rate_limited_envelope("enable_flexible_rings", limit)

    state = load_state(run_id)
    depth = state.bump_tweak("enable_flexible_rings", scope)

    rerun = monolithic.dock_ligands_to_segment(
        run_id=run_id,
        segment_id=segment_id,
        ligand_ids=[ligand_id],
        max_poses=5,
        increase_pose_diversity=True,
    )

    return {
        "status": "ok",
        "tool_name": "enable_flexible_rings",
        "tool_tier": "parameter_tweak",
        "scope": scope,
        "tweak_depth": depth,
        "applied_parameters": {"flexible_rings": True},
        "rerun": rerun,
        "rate_limit": limit.to_dict(),
    }


# ---------------------------------------------------------------------------
# Alpha-mask tweaks
# ---------------------------------------------------------------------------


def loosen_alpha_mask(run_id: str, site_id: str) -> dict[str, Any]:
    key = f"{run_id}:{site_id}"
    limit = check_limit("loosen_alpha_mask", key)
    if not limit.allowed:
        return rate_limited_envelope("loosen_alpha_mask", limit)

    state = load_state(run_id)
    depth = state.bump_tweak("loosen_alpha_mask", site_id)
    params = _alpha_params_for_depth("loosen", depth)

    rerun = monolithic.generate_alpha_masks(
        run_id=run_id,
        threshold_sigma=params["threshold_sigma"],
        smoothing=params["smoothing"],
        min_blob_volume=params["min_blob_volume"],
        site_ids=[site_id],
    )

    return {
        "status": "ok",
        "tool_name": "loosen_alpha_mask",
        "tool_tier": "parameter_tweak",
        "scope": site_id,
        "tweak_depth": depth,
        "applied_parameters": params,
        "rerun": rerun,
        "rate_limit": limit.to_dict(),
    }


def tighten_alpha_mask(run_id: str, site_id: str) -> dict[str, Any]:
    key = f"{run_id}:{site_id}"
    limit = check_limit("tighten_alpha_mask", key)
    if not limit.allowed:
        return rate_limited_envelope("tighten_alpha_mask", limit)

    state = load_state(run_id)
    depth = state.bump_tweak("tighten_alpha_mask", site_id)
    params = _alpha_params_for_depth("tighten", depth)

    rerun = monolithic.generate_alpha_masks(
        run_id=run_id,
        threshold_sigma=params["threshold_sigma"],
        smoothing=params["smoothing"],
        min_blob_volume=params["min_blob_volume"],
        site_ids=[site_id],
    )

    return {
        "status": "ok",
        "tool_name": "tighten_alpha_mask",
        "tool_tier": "parameter_tweak",
        "scope": site_id,
        "tweak_depth": depth,
        "applied_parameters": params,
        "rerun": rerun,
        "rate_limit": limit.to_dict(),
    }


# ---------------------------------------------------------------------------
# Refinement tweaks
# ---------------------------------------------------------------------------


def rerun_smart_refine_stronger_density(
    run_id: str, assignment_set_id: str, pose_ids: list[str]
) -> dict[str, Any]:
    key = f"{run_id}:{assignment_set_id}"
    limit = check_limit("rerun_smart_refine_stronger_density", key)
    if not limit.allowed:
        return rate_limited_envelope("rerun_smart_refine_stronger_density", limit)

    state = load_state(run_id)
    depth = state.bump_tweak("rerun_smart_refine_stronger_density", assignment_set_id)
    params = _smart_refine_weights("stronger_density", depth)

    rerun = monolithic.run_smart_refine_2(
        run_id=run_id,
        assignment_set_id=f"{assignment_set_id}_strong{depth}",
        pose_ids=pose_ids,
        optimisation_score="ccc",
    )

    return {
        "status": "ok",
        "tool_name": "rerun_smart_refine_stronger_density",
        "tool_tier": "parameter_tweak",
        "scope": assignment_set_id,
        "tweak_depth": depth,
        "applied_parameters": params,
        "rerun": rerun,
        "rate_limit": limit.to_dict(),
    }


def rerun_smart_refine_softer_restraints(
    run_id: str, assignment_set_id: str, pose_ids: list[str]
) -> dict[str, Any]:
    key = f"{run_id}:{assignment_set_id}"
    limit = check_limit("rerun_smart_refine_softer_restraints", key)
    if not limit.allowed:
        return rate_limited_envelope("rerun_smart_refine_softer_restraints", limit)

    state = load_state(run_id)
    depth = state.bump_tweak("rerun_smart_refine_softer_restraints", assignment_set_id)
    params = _smart_refine_weights("softer_restraints", depth)

    rerun = monolithic.run_smart_refine_2(
        run_id=run_id,
        assignment_set_id=f"{assignment_set_id}_soft{depth}",
        pose_ids=pose_ids,
    )

    return {
        "status": "ok",
        "tool_name": "rerun_smart_refine_softer_restraints",
        "tool_tier": "parameter_tweak",
        "scope": assignment_set_id,
        "tweak_depth": depth,
        "applied_parameters": params,
        "rerun": rerun,
        "rate_limit": limit.to_dict(),
    }


def rerun_smart_refine_alt_score(
    run_id: str, assignment_set_id: str, pose_ids: list[str], score: str = "ccc"
) -> dict[str, Any]:
    key = f"{run_id}:{assignment_set_id}:{score}"
    limit = check_limit("rerun_smart_refine_alt_score", key)
    if not limit.allowed:
        return rate_limited_envelope("rerun_smart_refine_alt_score", limit)

    if score not in ("qscore", "ccc", "mi", "sci"):
        return {
            "status": "invalid_arguments",
            "tool_name": "rerun_smart_refine_alt_score",
            "warnings": [f"unknown optimisation score {score!r}"],
        }

    state = load_state(run_id)
    depth = state.bump_tweak("rerun_smart_refine_alt_score", f"{assignment_set_id}:{score}")

    rerun = monolithic.run_smart_refine_2(
        run_id=run_id,
        assignment_set_id=f"{assignment_set_id}_{score}{depth}",
        pose_ids=pose_ids,
        optimisation_score=score,
    )

    return {
        "status": "ok",
        "tool_name": "rerun_smart_refine_alt_score",
        "tool_tier": "parameter_tweak",
        "scope": f"{assignment_set_id}:{score}",
        "tweak_depth": depth,
        "applied_parameters": {"optimisation_score": score},
        "rerun": rerun,
        "rate_limit": limit.to_dict(),
    }


def rerun_ion_search_lower_threshold(
    run_id: str, current_threshold: float = 0.65
) -> dict[str, Any]:
    key = f"{run_id}"
    limit = check_limit("rerun_ion_search_lower_threshold", key)
    if not limit.allowed:
        return rate_limited_envelope("rerun_ion_search_lower_threshold", limit)

    state = load_state(run_id)
    depth = state.bump_tweak("rerun_ion_search_lower_threshold", run_id)
    new_threshold = max(0.1, current_threshold - 0.1 * depth)

    rerun = monolithic.run_ion_search(
        run_id=run_id, confidence_threshold=round(new_threshold, 3)
    )

    return {
        "status": "ok",
        "tool_name": "rerun_ion_search_lower_threshold",
        "tool_tier": "parameter_tweak",
        "scope": run_id,
        "tweak_depth": depth,
        "applied_parameters": {"confidence_threshold": round(new_threshold, 3)},
        "rerun": rerun,
        "rate_limit": limit.to_dict(),
    }
