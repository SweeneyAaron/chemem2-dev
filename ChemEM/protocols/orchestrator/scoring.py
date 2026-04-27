# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Stateless per-pose scoring helpers used by the orchestrator's gates.

Both helpers wrap existing ChemEM primitives without mutating system or
ligand state, so the orchestrator can score arbitrary candidate poses
without disturbing the protocols that ship those primitives.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ChemEM.protocols.mapQ_score.mapq_utils import compute_qscores_from_emmap
from ChemEM.protocols._docking.mmgbsa_score import score_single_pose


def _heavy_atom_indices(mol) -> np.ndarray:
    return np.asarray(
        [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"], dtype=int
    )


def _within_map_bounds(positions: np.ndarray, density_map) -> bool:
    try:
        origin = np.asarray(density_map.origin, dtype=float)
        apix = np.asarray(density_map.apix, dtype=float)
        shape = np.asarray(density_map.density_map.shape, dtype=float)
    except AttributeError:
        return True  # fail open if map shape isn't introspectable
    max_bounds = origin + (shape - 1.0) * apix
    return bool(np.all(positions >= origin) and np.all(positions <= max_bounds))


def qscore_pose(
    coords: np.ndarray,
    mol,
    density_map,
    sigma_ref: float = 0.6,
) -> Optional[float]:
    """Mean Q-score of one ligand pose against the full map.

    coords: (n_atoms, 3) Å — must align with mol's atom indexing.
    Returns None if the pose lies outside the map or has no heavy atoms.
    """
    heavy = _heavy_atom_indices(mol)
    if heavy.size == 0:
        return None
    heavy_xyz = np.asarray(coords, dtype=float)[heavy]
    if not _within_map_bounds(heavy_xyz, density_map):
        return None
    qs = compute_qscores_from_emmap(
        atoms_xyz=heavy_xyz, emmap=density_map, sigma_ref=sigma_ref
    )
    return float(np.mean(qs))


def mmgbsa_single_frame(
    coords: np.ndarray,
    ligand,
    protein,
    pose_idx: int = 0,
):
    """Single-frame MMGBSA on one pose. Returns PoseScore or None on failure.

    Wraps mmgbsa_score.score_single_pose(), which already builds a
    one-frame mdtraj trajectory and evaluates OpenMM Context energies
    without integrating — no MD sampling.
    """
    try:
        return score_single_pose(np.asarray(coords, dtype=float), ligand, protein, pose_idx)
    except Exception:
        return None
