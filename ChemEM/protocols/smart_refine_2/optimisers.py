from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .scorers import BaseScorer, ScoreResult, get_scorer

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional in light tests
    cKDTree = None


VDW_RADII_A = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "MG": 1.73,
    "CA": 2.31,
    "ZN": 1.39,
    "MN": 1.61,
    "FE": 1.56,
    "CU": 1.40,
    "CO": 1.53,
    "NI": 1.63,
    "NA": 2.27,
    "K": 2.75,
}


@dataclass
class FitInMapConfig:
    initial_step_size_A: float | None = None
    min_step_size_A: float | None = None
    fd_delta_A: float | None = None
    step_decay: float = 0.5
    max_steps: int = 64
    clash_cutoff_A: float = 5.0
    max_vdw_overlap_A: float = 0.1
    max_hbond_overlap_A: float = 0.7
    ligand_clash_topological_exclusion_bonds: int = 3
    clash_mode: str = "off"
    clash_weight: float = 1.0
    backtracking_max_halves: int = 8
    improvement_tol: float = 1e-12
    # If set, fit_in_map stops early once the moving average of the last
    # ``early_stop_window`` accepted-step relative objective gains drops below
    # ``early_stop_tol``. Default None disables the check (preserves prior behaviour).
    early_stop_tol: float | None = None
    early_stop_window: int = 3
    debug: bool = False
    debug_stride: int = 1
    clash_diagnostics: bool = False
    clash_diagnostic_limit: int = 5
    progress: bool = True
    progress_leave: bool = False


@dataclass
class FitInMapResult:
    best_coords_A: np.ndarray
    initial_raw_score: float
    best_raw_score: float
    initial_objective: float
    best_objective: float
    initial_clash_penalty: float
    best_clash_penalty: float
    initial_clash_count: int
    best_clash_count: int
    best_max_overlap_A: float
    steps: int
    evaluations: int
    converged: bool
    final_step_size_A: float
    score_terms: dict[str, Any] = field(default_factory=dict)
    clash_terms: dict[str, Any] = field(default_factory=dict)
    best_rotation_matrix: np.ndarray | None = None
    best_translation_A: np.ndarray | None = None

    @property
    def delta_raw_score(self) -> float:
        return float(self.best_raw_score - self.initial_raw_score)

    @property
    def delta_objective(self) -> float:
        return float(self.best_objective - self.initial_objective)


@dataclass
class ClashResult:
    penalty: float
    count: int
    max_overlap_A: float
    max_allowed_overlap_A: float
    max_excess_overlap_A: float
    pairs_checked: int
    worst_pairs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.count == 0


@dataclass
class _Evaluation:
    raw_score: float
    objective: float
    score_terms: dict[str, Any]
    clash: ClashResult


def _map_reference(refine_ligand):
    return getattr(
        refine_ligand,
        "_map_reference",
        getattr(refine_ligand, "_map_referece", None),
    )


def _apix_scale_A(refine_ligand) -> float:
    map_reference = _map_reference(refine_ligand)
    apix = getattr(map_reference, "apix", 1.0)
    arr = np.asarray(apix, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    return float(np.min(arr)) if arr.size else 1.0


def _resolved_steps(refine_ligand, config: FitInMapConfig) -> tuple[float, float, float]:
    apix = _apix_scale_A(refine_ligand)
    initial = (
        0.5 * apix
        if config.initial_step_size_A is None
        else float(config.initial_step_size_A)
    )
    minimum = (
        0.01 * apix
        if config.min_step_size_A is None
        else float(config.min_step_size_A)
    )
    fd_delta = 0.1 * apix if config.fd_delta_A is None else float(config.fd_delta_A)
    return (
        max(initial, 1e-12),
        max(minimum, 1e-12),
        max(fd_delta, 1e-12),
    )


def _as_coords(coords_A) -> np.ndarray:
    coords = np.asarray(coords_A, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords_A must have shape (N, 3), got {coords.shape}")
    return coords


def _element_key(element) -> str:
    text = str(element or "C").upper().strip()
    if text.isdigit():
        by_atomic_number = {
            "1": "H",
            "6": "C",
            "7": "N",
            "8": "O",
            "9": "F",
            "15": "P",
            "16": "S",
            "17": "CL",
            "35": "BR",
            "53": "I",
        }
        return by_atomic_number.get(text, "C")
    return text


def _vdw_radius_A(element) -> float:
    return float(VDW_RADII_A.get(_element_key(element), 1.70))


def _rotation_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-15:
        return np.eye(3, dtype=np.float64)
    axis = rv / angle
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _transform_coords(
    base_coords_A: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_A: np.ndarray,
    pivot: np.ndarray | None = None,
) -> np.ndarray:
    # Rotate about `pivot` when given (covalent: the warhead, so it stays fixed),
    # else the ligand centroid. Default pivot=None reproduces the original behaviour.
    center = np.mean(base_coords_A, axis=0) if pivot is None else np.asarray(pivot, dtype=np.float64)
    return ((base_coords_A - center) @ rotation_matrix.T) + center + translation_A


def _ligand_radius_A(coords_A: np.ndarray) -> float:
    if coords_A.shape[0] <= 1:
        return 1.0
    centroid = np.mean(coords_A, axis=0)
    radius = float(np.max(np.linalg.norm(coords_A - centroid, axis=1)))
    return max(radius, 1.0)


def _progress(iterable, total: int, enabled: bool, leave: bool):
    if not enabled:
        return iterable, False
    try:
        from tqdm.auto import tqdm

        return (
            tqdm(iterable, total=total, desc="fit-in-map", unit="step", leave=leave),
            False,
        )
    except Exception:
        print(f"[fit-in-map] running up to {total} steps")
        return iterable, True


def protein_ligand_clash(
    ligand_coords_A: np.ndarray,
    ligand_elements,
    protein_coords_A: np.ndarray,
    protein_elements=None,
    *,
    cutoff_A: float = 5.0,
    max_vdw_overlap_A: float = 0.1,
    max_hbond_overlap_A: float = 0.7,
    ligand_atom_indices=None,
    protein_atom_indices=None,
    diagnose_pairs: bool = False,
    diagnostic_limit: int = 5,
) -> ClashResult:
    lig = _as_coords(ligand_coords_A)
    prot = np.asarray(protein_coords_A, dtype=np.float64).reshape((-1, 3))
    if lig.size == 0 or prot.size == 0:
        return ClashResult(0.0, 0, 0.0, 0.0, 0.0, 0)

    lig_elements = np.asarray(ligand_elements, dtype=object).reshape(-1)
    if lig_elements.shape[0] != lig.shape[0]:
        lig_elements = np.full(lig.shape[0], "C", dtype=object)

    if protein_elements is None:
        prot_elements = np.full(prot.shape[0], "C", dtype=object)
    else:
        prot_elements = np.asarray(protein_elements, dtype=object).reshape(-1)
        if prot_elements.shape[0] != prot.shape[0]:
            prot_elements = np.full(prot.shape[0], "C", dtype=object)
    lig_atom_indices = (
        np.arange(lig.shape[0], dtype=int)
        if ligand_atom_indices is None
        else np.asarray(ligand_atom_indices, dtype=int).reshape(-1)
    )
    if lig_atom_indices.shape[0] != lig.shape[0]:
        lig_atom_indices = np.arange(lig.shape[0], dtype=int)
    prot_atom_indices = (
        np.arange(prot.shape[0], dtype=int)
        if protein_atom_indices is None
        else np.asarray(protein_atom_indices, dtype=int).reshape(-1)
    )
    if prot_atom_indices.shape[0] != prot.shape[0]:
        prot_atom_indices = np.arange(prot.shape[0], dtype=int)

    pairs: list[tuple[int, int]] = []
    cutoff = float(cutoff_A)
    if cKDTree is not None:
        hits = cKDTree(prot).query_ball_point(lig, r=cutoff)
        pairs = [(i, int(j)) for i, rows in enumerate(hits) for j in rows]
    else:
        for i, xyz in enumerate(lig):
            d = np.linalg.norm(prot - xyz, axis=1)
            pairs.extend((i, int(j)) for j in np.flatnonzero(d <= cutoff))

    if not pairs:
        return ClashResult(0.0, 0, 0.0, 0.0, 0.0, 0)

    penalty = 0.0
    count = 0
    max_overlap = 0.0
    max_allowed_overlap = 0.0
    max_excess_overlap = 0.0
    worst_pairs = []
    for i, j in pairs:
        li = _element_key(lig_elements[i])
        pj = _element_key(prot_elements[j])
        allowed_overlap = (
            float(max_hbond_overlap_A)
            if {li, pj} <= {"N", "O"}
            else float(max_vdw_overlap_A)
        )
        dist = float(np.linalg.norm(lig[i] - prot[j]))
        overlap = max(0.0, _vdw_radius_A(li) + _vdw_radius_A(pj) - dist)
        max_overlap = max(max_overlap, overlap)
        max_allowed_overlap = max(max_allowed_overlap, allowed_overlap)
        excess_overlap = max(0.0, overlap - allowed_overlap)
        if excess_overlap <= 0.0:
            continue
        penalty += excess_overlap * excess_overlap
        count += 1
        max_excess_overlap = max(max_excess_overlap, excess_overlap)
        if diagnose_pairs:
            worst_pairs.append(
                {
                    "ligand_row": int(i),
                    "protein_row": int(j),
                    "ligand_atom_index": int(lig_atom_indices[i]),
                    "protein_atom_index": int(prot_atom_indices[j]),
                    "ligand_element": li,
                    "protein_element": pj,
                    "distance_A": float(dist),
                    "overlap_A": float(overlap),
                    "allowed_overlap_A": float(allowed_overlap),
                    "excess_overlap_A": float(excess_overlap),
                }
            )
    if worst_pairs:
        worst_pairs.sort(key=lambda pair: pair["excess_overlap_A"], reverse=True)
        worst_pairs = worst_pairs[: max(0, int(diagnostic_limit))]

    return ClashResult(
        penalty=float(penalty),
        count=int(count),
        max_overlap_A=float(max_overlap),
        max_allowed_overlap_A=float(max_allowed_overlap),
        max_excess_overlap_A=float(max_excess_overlap),
        pairs_checked=int(len(pairs)),
        worst_pairs=worst_pairs,
    )


def ligand_self_clash(
    ligand_coords_A: np.ndarray,
    ligand_elements,
    mol=None,
    *,
    ligand_atom_indices=None,
    query_atom_indices=None,
    reference_atom_indices=None,
    cutoff_A: float = 5.0,
    max_vdw_overlap_A: float = 0.1,
    max_hbond_overlap_A: float = 0.7,
    topological_exclusion_bonds: int = 3,
    diagnose_pairs: bool = False,
    diagnostic_limit: int = 5,
) -> ClashResult:
    lig = _as_coords(ligand_coords_A)
    if lig.size == 0 or mol is None:
        return ClashResult(0.0, 0, 0.0, 0.0, 0.0, 0)

    try:
        from rdkit import Chem

        topology = np.asarray(Chem.GetDistanceMatrix(mol), dtype=np.float64)
    except Exception:
        return ClashResult(0.0, 0, 0.0, 0.0, 0.0, 0)

    lig_elements = np.asarray(ligand_elements, dtype=object).reshape(-1)
    if lig_elements.shape[0] != lig.shape[0]:
        lig_elements = np.full(lig.shape[0], "C", dtype=object)

    ligand_atom_indices = _resolve_ligand_atom_indices(mol, ligand_atom_indices, lig.shape[0])
    if ligand_atom_indices.shape[0] != lig.shape[0]:
        return ClashResult(0.0, 0, 0.0, 0.0, 0.0, 0)

    row_by_atom = {int(atom_idx): row for row, atom_idx in enumerate(ligand_atom_indices)}
    query_rows = _rows_for_ligand_atom_selection(
        row_by_atom,
        ligand_atom_indices,
        query_atom_indices,
    )
    reference_rows = _rows_for_ligand_atom_selection(
        row_by_atom,
        ligand_atom_indices,
        reference_atom_indices,
    )
    if query_rows.size == 0 or reference_rows.size == 0:
        return ClashResult(0.0, 0, 0.0, 0.0, 0.0, 0)

    query = lig[query_rows]
    reference = lig[reference_rows]
    pairs: list[tuple[int, int]] = []
    cutoff = float(cutoff_A)
    if cKDTree is not None:
        hits = cKDTree(reference).query_ball_point(query, r=cutoff)
        pairs = [
            (int(query_rows[i]), int(reference_rows[j]))
            for i, rows in enumerate(hits)
            for j in rows
        ]
    else:
        for q_local, xyz in enumerate(query):
            d = np.linalg.norm(reference - xyz, axis=1)
            pairs.extend(
                (int(query_rows[q_local]), int(reference_rows[r_local]))
                for r_local in np.flatnonzero(d <= cutoff)
            )

    if not pairs:
        return ClashResult(0.0, 0, 0.0, 0.0, 0.0, 0)

    penalty = 0.0
    count = 0
    max_overlap = 0.0
    max_allowed_overlap = 0.0
    max_excess_overlap = 0.0
    worst_pairs = []
    pairs_checked = 0
    seen_pairs = set()
    topo_exclusion = int(topological_exclusion_bonds)

    for query_row, reference_row in pairs:
        query_atom_idx = int(ligand_atom_indices[query_row])
        reference_atom_idx = int(ligand_atom_indices[reference_row])
        if query_atom_idx == reference_atom_idx:
            continue

        pair_key = tuple(sorted((query_atom_idx, reference_atom_idx)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        if (
            topo_exclusion >= 0
            and 0 <= query_atom_idx < topology.shape[0]
            and 0 <= reference_atom_idx < topology.shape[1]
            and float(topology[query_atom_idx, reference_atom_idx]) <= topo_exclusion
        ):
            continue

        pairs_checked += 1
        qi = _element_key(lig_elements[query_row])
        rj = _element_key(lig_elements[reference_row])
        allowed_overlap = (
            float(max_hbond_overlap_A)
            if {qi, rj} <= {"N", "O"}
            else float(max_vdw_overlap_A)
        )
        dist = float(np.linalg.norm(lig[query_row] - lig[reference_row]))
        overlap = max(0.0, _vdw_radius_A(qi) + _vdw_radius_A(rj) - dist)
        max_overlap = max(max_overlap, overlap)
        max_allowed_overlap = max(max_allowed_overlap, allowed_overlap)
        excess_overlap = max(0.0, overlap - allowed_overlap)
        if excess_overlap <= 0.0:
            continue
        penalty += excess_overlap * excess_overlap
        count += 1
        max_excess_overlap = max(max_excess_overlap, excess_overlap)
        if diagnose_pairs:
            worst_pairs.append(
                {
                    "kind": "ligand_self",
                    "query_ligand_row": int(query_row),
                    "reference_ligand_row": int(reference_row),
                    "query_ligand_atom_index": int(query_atom_idx),
                    "reference_ligand_atom_index": int(reference_atom_idx),
                    "query_ligand_element": qi,
                    "reference_ligand_element": rj,
                    "distance_A": float(dist),
                    "overlap_A": float(overlap),
                    "allowed_overlap_A": float(allowed_overlap),
                    "excess_overlap_A": float(excess_overlap),
                }
            )
    if worst_pairs:
        worst_pairs.sort(key=lambda pair: pair["excess_overlap_A"], reverse=True)
        worst_pairs = worst_pairs[: max(0, int(diagnostic_limit))]

    return ClashResult(
        penalty=float(penalty),
        count=int(count),
        max_overlap_A=float(max_overlap),
        max_allowed_overlap_A=float(max_allowed_overlap),
        max_excess_overlap_A=float(max_excess_overlap),
        pairs_checked=int(pairs_checked),
        worst_pairs=worst_pairs,
    )


def combine_clash_results(*clashes: ClashResult) -> ClashResult:
    valid = [clash for clash in clashes if clash is not None]
    if not valid:
        return ClashResult(0.0, 0, 0.0, 0.0, 0.0, 0)

    worst_pairs = [
        dict(pair)
        for clash in valid
        for pair in list(getattr(clash, "worst_pairs", []) or [])
    ]
    if worst_pairs:
        worst_pairs.sort(
            key=lambda pair: float(pair.get("excess_overlap_A", 0.0)),
            reverse=True,
        )

    return ClashResult(
        penalty=float(sum(float(clash.penalty) for clash in valid)),
        count=int(sum(int(clash.count) for clash in valid)),
        max_overlap_A=float(max(float(clash.max_overlap_A) for clash in valid)),
        max_allowed_overlap_A=float(
            max(float(clash.max_allowed_overlap_A) for clash in valid)
        ),
        max_excess_overlap_A=float(
            max(float(clash.max_excess_overlap_A) for clash in valid)
        ),
        pairs_checked=int(sum(int(clash.pairs_checked) for clash in valid)),
        worst_pairs=worst_pairs,
    )


def _resolve_ligand_atom_indices(mol, ligand_atom_indices, n_coord_rows: int) -> np.ndarray:
    if ligand_atom_indices is not None:
        arr = np.asarray(ligand_atom_indices, dtype=int).reshape(-1)
        return arr if arr.shape[0] == int(n_coord_rows) else np.zeros(0, dtype=int)

    try:
        heavy = [
            int(atom.GetIdx())
            for atom in mol.GetAtoms()
            if int(atom.GetAtomicNum()) > 1
        ]
    except Exception:
        heavy = []
    if len(heavy) == int(n_coord_rows):
        return np.asarray(heavy, dtype=int)
    return np.arange(int(n_coord_rows), dtype=int)


def _rows_for_ligand_atom_selection(
    row_by_atom: dict[int, int],
    ligand_atom_indices: np.ndarray,
    atom_selection,
) -> np.ndarray:
    if atom_selection is None:
        return np.arange(ligand_atom_indices.shape[0], dtype=int)
    rows = [
        int(row_by_atom[int(atom_idx)])
        for atom_idx in np.asarray(atom_selection, dtype=int).reshape(-1)
        if int(atom_idx) in row_by_atom
    ]
    return np.asarray(rows, dtype=int)


def _clash_objective(raw: float, clash: ClashResult, mode: str, weight: float, n_ligand_atoms: int) -> float:
    if not np.isfinite(float(raw)):
        return float("-inf")
    if mode == "off":
        return float(raw)
    if mode == "hard":
        return float(raw) if clash.accepted else float("-inf")
    normalized_penalty = float(clash.penalty) / max(1, int(n_ligand_atoms))
    return float(raw) - float(weight) * normalized_penalty


def _finite_difference_delta(plus_value: float, minus_value: float, scale: float) -> float:
    plus_finite = np.isfinite(plus_value)
    minus_finite = np.isfinite(minus_value)
    if plus_finite and minus_finite:
        return float((plus_value - minus_value) / scale)
    if plus_finite and not minus_finite:
        return float(1.0 / scale)
    if not plus_finite and minus_finite:
        return float(-1.0 / scale)
    return 0.0


def fit_in_map(
    refine_ligand,
    scorer: BaseScorer | str | None = None,
    config: FitInMapConfig | None = None,
) -> FitInMapResult:
    config = config or FitInMapConfig()
    scorer = get_scorer(scorer)
    clash_mode = str(config.clash_mode).lower().strip()
    if clash_mode not in {"off", "soft", "hard"}:
        raise ValueError(
            "FitInMapConfig.clash_mode must be one of "
            f"'off', 'soft', or 'hard', got {config.clash_mode!r}"
        )

    base_coords = _as_coords(getattr(refine_ligand, "_atom_positions"))
    base_coords = base_coords.copy()
    ligand_elements = np.asarray(
        getattr(refine_ligand, "_atom_elements", np.full(base_coords.shape[0], "C")),
        dtype=object,
    )
    protein_coords = np.asarray(
        getattr(refine_ligand, "local_coords_A", np.zeros((0, 3))),
        dtype=np.float64,
    ).reshape((-1, 3))
    protein_elements = np.asarray(
        getattr(refine_ligand, "local_elements", np.full(protein_coords.shape[0], "C")),
        dtype=object,
    )
    ligand_atom_indices = getattr(refine_ligand, "_atom_indices", None)
    protein_atom_indices = getattr(refine_ligand, "local_rows", None)
    ligand_mol = getattr(getattr(refine_ligand, "_ligand", None), "mol", None)

    step_size, min_step, fd_delta = _resolved_steps(refine_ligand, config)
    step_decay = float(config.step_decay)
    if not 0.0 < step_decay < 1.0:
        raise ValueError("FitInMapConfig.step_decay must be between 0 and 1")
    max_backtracks = max(0, int(config.backtracking_max_halves))
    improvement_tol = max(0.0, float(config.improvement_tol))

    early_stop_tol = (
        float(config.early_stop_tol) if config.early_stop_tol is not None else None
    )
    early_stop_window = max(1, int(config.early_stop_window))
    recent_rel_gains: list[float] = []

    ligand_radius = _ligand_radius_A(base_coords)
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    # Covalent ligand: pin the bonded warhead. Rotate the rigid body about the warhead
    # (not the centroid) and freeze the translation DOF, so rigid-body moves can never
    # displace the warhead off the protein anchor. Gated on _covalent_warhead_row, which
    # is set only for covalent ligands and rides copy.copy clones -> non-covalent
    # ligands keep pivot=None (centroid) and free translation, i.e. unchanged behaviour.
    warhead_row = getattr(refine_ligand, "_covalent_warhead_row", None)
    covalent_pivot = base_coords[int(warhead_row)].copy() if warhead_row is not None else None
    freeze_translation = warhead_row is not None
    # With MORE than one covalent bond the ligand is pinned at two points, so the
    # only rigid move that keeps every bond intact is a rotation about the axis
    # through the bonded atoms. Rotation gradients are projected onto that axis
    # below; with three or more anchors no rotation is legal at all. The tether
    # penalty alone is not enough here -- it is a soft flat-bottom term and the
    # search will happily trade a broken bond for a better Q-score.
    covalent_axis = None
    freeze_rotation = False
    _anchors = getattr(refine_ligand, "_covalent_anchors", None) or []
    if len(_anchors) == 2:
        rows = [int(a[0]) for a in _anchors]
        if all(0 <= r < base_coords.shape[0] for r in rows):
            axis = base_coords[rows[1]] - base_coords[rows[0]]
            norm = float(np.linalg.norm(axis))
            if norm > 1e-9:
                covalent_axis = axis / norm
            else:
                freeze_rotation = True
    elif len(_anchors) > 2:
        freeze_rotation = True
    steps = 0
    evaluations = 0
    trace = []

    def debug_log(message: str) -> None:
        if bool(config.debug):
            print(f"[fit-in-map debug] {message}")

    def evaluate(coords_A: np.ndarray) -> _Evaluation:
        nonlocal evaluations
        scorer_out = scorer.score(refine_ligand, coords_A)
        score_result = (
            scorer_out
            if isinstance(scorer_out, ScoreResult)
            else ScoreResult(value=float(scorer_out), terms={})
        )
        raw = float(score_result.value)
        if not np.isfinite(raw):
            raw = float("-inf")
        protein_clash = protein_ligand_clash(
            coords_A,
            ligand_elements,
            protein_coords,
            protein_elements,
            cutoff_A=float(config.clash_cutoff_A),
            max_vdw_overlap_A=float(config.max_vdw_overlap_A),
            max_hbond_overlap_A=float(config.max_hbond_overlap_A),
            ligand_atom_indices=ligand_atom_indices,
            protein_atom_indices=protein_atom_indices,
            diagnose_pairs=bool(config.clash_diagnostics),
            diagnostic_limit=int(config.clash_diagnostic_limit),
        )
        self_clash = ligand_self_clash(
            coords_A,
            ligand_elements,
            ligand_mol,
            ligand_atom_indices=ligand_atom_indices,
            cutoff_A=float(config.clash_cutoff_A),
            max_vdw_overlap_A=float(config.max_vdw_overlap_A),
            max_hbond_overlap_A=float(config.max_hbond_overlap_A),
            topological_exclusion_bonds=int(
                getattr(config, "ligand_clash_topological_exclusion_bonds", 3)
            ),
            diagnose_pairs=bool(config.clash_diagnostics),
            diagnostic_limit=int(config.clash_diagnostic_limit),
        )
        clash = combine_clash_results(protein_clash, self_clash)
        objective = _clash_objective(
            raw,
            clash,
            clash_mode,
            float(config.clash_weight),
            base_coords.shape[0],
        )
        evaluations += 1
        return _Evaluation(
            raw_score=raw,
            objective=float(objective),
            score_terms=dict(score_result.terms),
            clash=clash,
        )

    current_coords = _transform_coords(base_coords, rotation, translation, covalent_pivot)
    current = evaluate(current_coords)
    initial = current
    best = current
    best_coords = current_coords.copy()
    best_rotation = rotation.copy()
    best_translation = translation.copy()
    debug_log(
        "initial "
        f"mode={clash_mode} raw={initial.raw_score:+.5f} "
        f"objective={initial.objective:+.5f} "
        f"clashes={initial.clash.count} penalty={initial.clash.penalty:.5f}"
    )
    if bool(config.clash_diagnostics) and initial.clash.worst_pairs:
        debug_log(f"initial worst_clashes={initial.clash.worst_pairs}")

    iterator, fallback_progress = _progress(
        range(int(config.max_steps)),
        int(config.max_steps),
        bool(config.progress),
        bool(config.progress_leave),
    )
    for _ in iterator:
        if step_size <= min_step:
            break

        if steps % 2 == 0 and not freeze_translation:
            move_type = "translation"
            gradient = np.zeros(3, dtype=np.float64)
            for axis in range(3):
                plus_translation = translation.copy()
                minus_translation = translation.copy()
                plus_translation[axis] += fd_delta
                minus_translation[axis] -= fd_delta
                plus_coords = _transform_coords(base_coords, rotation, plus_translation, covalent_pivot)
                minus_coords = _transform_coords(base_coords, rotation, minus_translation, covalent_pivot)
                gradient[axis] = _finite_difference_delta(
                    evaluate(plus_coords).objective,
                    evaluate(minus_coords).objective,
                    2.0 * fd_delta,
                )
            gradient = np.where(np.isfinite(gradient), gradient, 0.0)
            norm = float(np.linalg.norm(gradient))
            if norm > 0.0:
                direction = gradient / norm

                def proposed(step):
                    trial_translation = translation + float(step) * direction
                    return rotation, trial_translation
            else:
                proposed = None
        elif freeze_rotation:
            #Three or more covalent bonds leave no rigid-body freedom at all;
            #the pose is refined by torsions only.
            move_type = "rotation"
            proposed = None
        else:
            move_type = "rotation"
            gradient = np.zeros(3, dtype=np.float64)
            angle_delta = fd_delta / ligand_radius
            for axis in range(3):
                plus_rv = np.zeros(3, dtype=np.float64)
                minus_rv = np.zeros(3, dtype=np.float64)
                plus_rv[axis] = angle_delta
                minus_rv[axis] = -angle_delta
                plus_rotation = _rotation_matrix(plus_rv) @ rotation
                minus_rotation = _rotation_matrix(minus_rv) @ rotation
                plus_coords = _transform_coords(base_coords, plus_rotation, translation, covalent_pivot)
                minus_coords = _transform_coords(base_coords, minus_rotation, translation, covalent_pivot)
                gradient[axis] = _finite_difference_delta(
                    evaluate(plus_coords).objective,
                    evaluate(minus_coords).objective,
                    2.0 * angle_delta,
                )
            gradient = np.where(np.isfinite(gradient), gradient, 0.0)
            # Two covalent bonds: keep only the rotation component about the axis
            # through the bonded atoms, the one rotation that holds both bonds.
            if covalent_axis is not None:
                gradient = covalent_axis * float(np.dot(gradient, covalent_axis))
            norm = float(np.linalg.norm(gradient))
            if norm > 0.0:
                direction = gradient / norm

                def proposed(step):
                    rv = (float(step) / ligand_radius) * direction
                    trial_rotation = _rotation_matrix(rv) @ rotation
                    return trial_rotation, translation
            else:
                proposed = None

        accepted_step = False
        n_backtracks = 0
        trial = current
        objective_before_step = current.objective
        if proposed is None:
            step_size *= step_decay
        else:
            trial_step = float(step_size)
            for n_backtracks in range(max_backtracks + 1):
                trial_rotation, trial_translation = proposed(trial_step)
                trial_coords = _transform_coords(
                    base_coords,
                    trial_rotation,
                    trial_translation,
                    covalent_pivot,
                )
                trial = evaluate(trial_coords)
                if trial.objective > current.objective + improvement_tol:
                    rotation = trial_rotation
                    translation = trial_translation
                    current_coords = trial_coords
                    current = trial
                    step_size = trial_step
                    accepted_step = True
                    break
                trial_step *= step_decay
                if trial_step <= min_step:
                    break
            if not accepted_step:
                step_size = min(step_size * step_decay, trial_step)
        steps += 1
        if current.objective > best.objective:
            best = current
            best_coords = current_coords.copy()
            best_rotation = rotation.copy()
            best_translation = translation.copy()

        early_stopped = False
        if early_stop_tol is not None and accepted_step:
            gain = float(current.objective - objective_before_step)
            recent_rel_gains.append(gain)
            if len(recent_rel_gains) > early_stop_window:
                recent_rel_gains.pop(0)
            if (
                len(recent_rel_gains) >= early_stop_window
                and (sum(recent_rel_gains) / early_stop_window) < early_stop_tol
            ):
                early_stopped = True
                debug_log(
                    f"early-stop at step={steps} "
                    f"mean_gain={sum(recent_rel_gains) / early_stop_window:+.3e} "
                    f"tol={early_stop_tol:+.3e}"
                )
        if bool(config.debug) and (
            steps == 1 or steps % max(1, int(config.debug_stride)) == 0
        ):
            trace_item = {
                "step": int(steps),
                "move": move_type,
                "accepted": bool(accepted_step),
                "raw": float(current.raw_score),
                "objective": float(current.objective),
                "clash_count": int(current.clash.count),
                "clash_penalty": float(current.clash.penalty),
                "step_size_A": float(step_size),
                "gradient": [float(x) for x in gradient.tolist()],
                "translation_A": [float(x) for x in translation.tolist()],
                "backtracks": int(n_backtracks),
            }
            trace.append(trace_item)
            debug_log(
                f"step={steps} move={move_type} accepted={accepted_step} "
                f"raw={current.raw_score:+.5f} objective={current.objective:+.5f} "
                f"clashes={current.clash.count} penalty={current.clash.penalty:.5f} "
                f"step_size={step_size:.5f} gradient={np.round(gradient, 5).tolist()} "
                f"translation={np.round(translation, 5).tolist()}"
            )

        if accepted_step and steps % 4 == 0:
            step_size *= step_decay

        if early_stopped:
            break

    converged = bool(step_size <= min_step or early_stopped)
    if fallback_progress:
        print(
            "[fit-in-map] finished "
            f"steps={steps} best={best.raw_score:+.5f} "
            f"objective={best.objective:+.5f}"
        )

    score_terms = dict(best.score_terms)
    score_terms.update(
        {
            "fit_clash_mode": clash_mode,
            "fit_clash_weight": float(config.clash_weight),
            "fit_initial_raw_score": float(initial.raw_score),
            "fit_initial_objective": float(initial.objective),
            "fit_initial_clash_count": int(initial.clash.count),
            "fit_initial_clash_penalty": float(initial.clash.penalty),
            "fit_translation_A": [float(x) for x in best_translation.tolist()],
            "fit_trace": trace,
        }
    )

    return FitInMapResult(
        best_coords_A=best_coords,
        initial_raw_score=float(initial.raw_score),
        best_raw_score=float(best.raw_score),
        initial_objective=float(initial.objective),
        best_objective=float(best.objective),
        initial_clash_penalty=float(initial.clash.penalty),
        best_clash_penalty=float(best.clash.penalty),
        initial_clash_count=int(initial.clash.count),
        best_clash_count=int(best.clash.count),
        best_max_overlap_A=float(best.clash.max_overlap_A),
        steps=int(steps),
        evaluations=int(evaluations),
        converged=converged,
        final_step_size_A=float(step_size),
        score_terms=score_terms,
        clash_terms={
            "penalty": float(best.clash.penalty),
            "count": int(best.clash.count),
            "max_overlap_A": float(best.clash.max_overlap_A),
            "max_allowed_overlap_A": float(best.clash.max_allowed_overlap_A),
            "max_excess_overlap_A": float(best.clash.max_excess_overlap_A),
            "pairs_checked": int(best.clash.pairs_checked),
            "accepted": bool(best.clash.accepted),
            "worst_pairs": list(best.clash.worst_pairs),
        },
        best_rotation_matrix=best_rotation,
        best_translation_A=best_translation,
    )
