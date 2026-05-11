from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class AtomClass(Enum):
    ANCHOR = "anchor"
    REPAIR = "repair"
    WEAK_OR_ABSENT = "weak_or_absent"
    UNCERTAIN = "uncertain"


@dataclass
class AtomMapMetrics:
    atom_index: int
    q_score: float
    local_ccc: float
    map_value: float
    map_gradient: np.ndarray
    halfmap_agreement: Optional[float] = None
    positive_diff_density: Optional[float] = None
    negative_diff_density: Optional[float] = None


@dataclass
class LigandMapMetrics:
    atom_metrics: Dict[int, AtomMapMetrics]
    ligand_ccc: float
    low_tail_q: float
    mean_q: float
    worst_atom_indices: List[int]


@dataclass
class TorsionInfo:
    torsion_id: int
    atom_indices: Tuple[int, int, int, int]
    bond_atoms: Tuple[int, int]
    downstream_atoms: List[int]
    current_angle_deg: float


@dataclass
class TorsionProfile:
    torsion_id: int
    minima_deg: List[float]
    relative_energies: List[float]
    scan_angles_deg: List[float]
    scan_energies: List[float]
    source: str = "unknown"
    atom_indices: Optional[Tuple[int, int, int, int]] = None


@dataclass
class CandidatePose:
    coords: np.ndarray
    move_type: str
    move_metadata: Dict[str, Any]
    map_metrics: Optional[LigandMapMetrics] = None
    geometry_metrics: Optional[Dict[str, Any]] = None
    score: Optional[float] = None


@dataclass
class MoveLogEntry:
    macrocycle: int
    move_type: str
    accepted: bool
    torsion_id: Optional[int]
    old_angle_deg: Optional[float]
    new_angle_deg: Optional[float]
    delta_ligand_ccc: float
    delta_low_tail_q: float
    delta_mean_q: float
    anchor_rmsd: float
    delta_openmm_energy: float
    clash_delta: float
    rejection_reason: Optional[str]
    move_metadata: Dict[str, Any] = field(default_factory=dict)


def _default_schedule() -> List[Dict[str, Any]]:
    return [
        {
            "name": "broad_repair",
            "torsion_offsets_deg": [0],
            "rigid_translations_A": [0.1, 0.2, 0.4],
            "rigid_rotations_deg": [2, 5, 10],
            "max_torsions": 8,
        },
        {
            "name": "medium_repair",
            "torsion_offsets_deg": [0],
            "rigid_translations_A": [0.1, 0.2],
            "rigid_rotations_deg": [2, 5],
            "max_torsions": 6,
        },
        {
            "name": "polish",
            "torsion_offsets_deg": [0, -5, 5],
            "rigid_translations_A": [0.05, 0.1],
            "rigid_rotations_deg": [1, 2],
            "max_torsions": 4,
        },
    ]


@dataclass
class SmartLigandRefinementConfig:
    anchor_q_min: float = 0.75
    anchor_local_ccc_min: float = 0.60
    repair_q_max: float = 0.55
    min_halfmap_agreement: float = 0.50
    min_density_value: float = 1e-6
    min_positive_diff_density: float = 1e-6

    target_q: float = 0.75
    min_torsion_badness: float = 0.05

    torsion_scan_step_deg: float = 10.0
    max_relative_minimum_energy_kcal: float = 5.0
    max_torsion_profile_count: int = 64
    torsion_profile_source: str = "auto"

    branch_beam_width: int = 16
    max_branch_torsions: int = 6
    enable_branch_rebuild: bool = False
    branch_minimum_offsets_deg: List[float] = field(
        default_factory=lambda: [-20.0, -10.0, 0.0, 10.0, 20.0]
    )
    enable_ring_flip_proposals: bool = True
    branch_rebuild_verbose: bool = False
    branch_aggressive_angle_search: bool = False
    branch_aggressive_grid_step_deg: float = 30.0
    branch_aggressive_flip_offsets_deg: List[float] = field(
        default_factory=lambda: [-120.0, -60.0, 60.0, 120.0, 180.0]
    )
    branch_relative_q_seeding: bool = False
    branch_relative_q_percentile: float = 0.25
    branch_use_anchor_seeds: bool = False
    branch_min_acceptance_improvement: float = -1e-3
    branch_max_anchor_rmsd: float = 0.6
    branch_accept_on_q_improvement: bool = True

    enable_directed_torsion_proposals: bool = True
    directed_torsion_step_deg: float = 20.0
    enable_subregion_proposals: bool = True
    subregion_min_size: int = 2
    subregion_max_size: int = 8
    subregion_step_A: float = 0.25

    w_ccc: float = 1.0
    w_q: float = 0.5
    w_local: float = 0.5
    w_diff: float = 0.25
    w_half: float = 0.25
    w_anchor: float = 1.0
    w_geom: float = 0.5
    w_clash: float = 1.0

    max_anchor_rmsd: float = 0.25
    min_acceptance_improvement: float = 1e-4
    max_low_tail_q_degrade: float = 0.02
    max_ligand_strain_increase_kcal: float = 5.0
    max_clash_score_increase: float = 1.0
    max_bond_deviation_A: float = 0.20
    max_angle_deviation_deg: float = 25.0

    max_macrocycles: int = 3
    convergence_ligand_ccc_delta: float = 1e-4
    convergence_low_tail_q_delta: float = 1e-4
    macrocycle_schedule: List[Dict[str, Any]] = field(default_factory=_default_schedule)

    qscore_sigma_ref: float = 0.6
    local_ccc_radius_A: float = 1.5
    local_ccc_sigma_A: float = 0.6
    resolution_A: float = 3.0
    sigma_coeff: float = 0.356
    score_hydrogens: bool = False

    clean_candidates: bool = False
    clean_accepted_each_macrocycle: bool = True
    clean_max_iterations: int = 100

    protein_ligand_clash_distance_A: float = 1.6
    max_candidates_per_stage: int = 256
    profile_timings: bool = False
    no_para: bool = False
    cpus_per_site: int = 1
    max_parallel_workers: Optional[int] = None

    output_dir: Optional[str] = None
    move_log_name: str = "smart_ligand_refine_log.json"
    write_accepted_sdf: bool = False
    reference_sdf_path: Optional[str] = None
    write_diagnostics: bool = False
    diagnostics_name: str = "smart_ligand_refine_diagnostics.json"
    progress: bool = False
    debug: bool = False
    debug_candidate_limit: int = 5
    debug_printer: Optional[Any] = None

    ligand_context_indices: Optional[Sequence[int]] = None
    protein_coords_A: Optional[np.ndarray] = None
