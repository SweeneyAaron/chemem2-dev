from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

DEFAULT_LIMITS: dict[str, int] = {
    "prepare_case": 1,
    "generate_binding_sites": 5,
    "generate_alpha_masks": 5,
    "segment_density_regions": 5,
    "prepare_ligand_library": 1,
    "prepare_ligand_segments": 1,
    "dock_ligands_to_segment": 3,
    "dock_ligand_segments_to_segment": 3,
    "minimise_candidate_pose": 2,
    "batch_score_candidates": 5,
    "compute_ccc": 3,
    "compute_mi": 3,
    "compute_mapq": 3,
    "compute_sci": 3,
    "compute_qscore_per_atom": 3,
    "compute_ligand_volume": 3,
    "compute_segment_volume": 3,
    "crop_map_around_segment": 5,
    "mask_map_by_segment": 5,
    "subtract_ligand_density": 5,
    "analyse_clashes": 3,
    "analyse_ligand_strain": 3,
    "analyse_interactions": 3,
    "summarise_segment_candidates": 5,
    "detect_ion_like_density": 5,
    "analyse_coordination_geometry": 5,
    "propose_ion_identity": 5,
    "build_ion_coordination_site": 3,
    "score_ion_assignment": 5,
    "render_ion_site_snapshot": 3,
    "run_ion_search": 3,
    "run_smart_refine_2": 5,
    "monitor_refinement_stability": 5,
    "score_refined_model": 5,
    "analyse_refined_ligands": 5,
    "analyse_refined_ions": 5,
    "analyse_residual_density": 5,
    "compare_refined_to_initial": 5,
    "summarise_refinement_outcome": 5,
    "render_snapshot": 200,
    "render_refined_snapshots": 50,
    "increase_docking_diversity": 3,
    "narrow_docking_search": 3,
    "enable_flexible_rings": 1,
    "loosen_alpha_mask": 3,
    "tighten_alpha_mask": 3,
    "rerun_smart_refine_stronger_density": 2,
    "rerun_smart_refine_softer_restraints": 2,
    "rerun_smart_refine_alt_score": 2,
    "rerun_ion_search_lower_threshold": 2,
    "cleanup_intermediate_files": 20,
    "generate_final_report": 1,
    "summarise_case_state": 1000,
    "get_rate_limit_status": 1000,
    "qwen_controller_step": 100,
}

DEFAULT_FALLBACK_LIMIT = 10


@dataclass(frozen=True)
class RateLimitResult:
    tool_name: str
    key: str
    allowed: bool
    used: int
    limit: int
    remaining: int

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "key": self.key,
            "allowed": self.allowed,
            "used": self.used,
            "limit": self.limit,
            "remaining": self.remaining,
        }


@dataclass
class RateLimiter:
    limits: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    fallback_limit: int = DEFAULT_FALLBACK_LIMIT
    usage: dict[tuple[str, str], int] = field(default_factory=dict)

    @classmethod
    def with_defaults(cls) -> "RateLimiter":
        return cls()

    def get_limit(self, tool_name: str) -> int:
        return self.limits.get(tool_name, self.fallback_limit)

    def peek(self, tool_name: str, key: str) -> RateLimitResult:
        limit = self.get_limit(tool_name)
        used = self.usage.get((tool_name, key), 0)
        return RateLimitResult(
            tool_name=tool_name,
            key=key,
            allowed=used < limit,
            used=used,
            limit=limit,
            remaining=max(0, limit - used),
        )

    def check_and_increment(self, tool_name: str, key: str) -> RateLimitResult:
        limit = self.get_limit(tool_name)
        usage_key = (tool_name, key)
        used = self.usage.get(usage_key, 0)
        if used >= limit:
            return RateLimitResult(
                tool_name=tool_name,
                key=key,
                allowed=False,
                used=used,
                limit=limit,
                remaining=0,
            )
        used += 1
        self.usage[usage_key] = used
        return RateLimitResult(
            tool_name=tool_name,
            key=key,
            allowed=True,
            used=used,
            limit=limit,
            remaining=limit - used,
        )

    def reset(self, tool_name: Optional[str] = None, key: Optional[str] = None) -> int:
        before = len(self.usage)
        if tool_name is None and key is None:
            self.usage.clear()
        else:
            for usage_key in list(self.usage.keys()):
                t, k = usage_key
                if tool_name is not None and t != tool_name:
                    continue
                if key is not None and k != key:
                    continue
                del self.usage[usage_key]
        return before - len(self.usage)

    def summary(self, run_id: Optional[str] = None) -> dict:
        rows = []
        prefix = f"{run_id}:" if run_id else None
        for (tool_name, key), used in sorted(self.usage.items()):
            if prefix is not None and not (key == run_id or key.startswith(prefix)):
                continue
            limit = self.get_limit(tool_name)
            rows.append(
                {
                    "tool_name": tool_name,
                    "key": key,
                    "used": used,
                    "limit": limit,
                    "remaining": max(0, limit - used),
                }
            )
        return {"status": "ok", "run_id": run_id, "rate_limits": rows}


class RateLimitExceeded(RuntimeError):
    def __init__(self, result: RateLimitResult):
        self.result = result
        super().__init__(
            f"rate limit exceeded for {result.tool_name} key={result.key}: "
            f"{result.used}/{result.limit}"
        )
