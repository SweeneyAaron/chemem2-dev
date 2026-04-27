# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

from .diagnostic import apply_dihedral_delta, apply_rigid_transform


def build_targets_from_gradient(
    accepted_pos_nm: np.ndarray,
    ligand_heavy_idx: np.ndarray,
    grad_heavy: np.ndarray,
    cap_A: float,
    proposal_scale: float,
    proposal_mode: str,
    rng: np.random.Generator,
    bad_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Build per-atom target positions (nm) for the CustomExternalForce tether.

    Convention: ``grad_heavy = ∇score`` with higher-is-better; ``"plus"`` moves
    atoms along +gradient (toward higher score). ``"minus"`` reverses the
    direction; ``"*_jitter"`` blends in random perturbation; ``"random_*"``
    replaces direction with a random unit vector.

    If ``bad_mask`` is provided (1D indices into ``grad_heavy``/``ligand_heavy_idx``
    row space, or a bool array of the same length), only atoms at those indices
    are displaced along the gradient. Atoms outside the mask keep their current
    position as the target (the tether then pulls them back to where they are,
    effectively pinning the well-fit subset in place).
    """
    target_pos_nm = np.asarray(accepted_pos_nm, dtype=np.float64).copy()
    lig_idx = np.asarray(ligand_heavy_idx, dtype=int)

    cap_A_eff = max(0.0, float(cap_A)) * float(proposal_scale)
    cap_nm = cap_A_eff * 0.1

    stats = {
        "cap_A": float(cap_A_eff),
        "moved_atoms": 0,
        "mean_target_disp_A": 0.0,
        "max_target_disp_A": 0.0,
        "mean_grad_norm": 0.0,
        "max_grad_norm": 0.0,
        "targeted_atoms": int(lig_idx.size),
    }

    if cap_nm <= 0.0 or lig_idx.size == 0:
        return target_pos_nm, stats

    grad = np.asarray(grad_heavy, dtype=np.float64)
    if grad.shape != (lig_idx.size, 3):
        return target_pos_nm, stats

    grad = np.where(np.isfinite(grad), grad, 0.0)
    norms = np.linalg.norm(grad, axis=1)
    if norms.size == 0:
        return target_pos_nm, stats

    max_norm = float(np.max(norms))
    mean_norm = float(np.mean(norms))
    stats["mean_grad_norm"] = mean_norm
    stats["max_grad_norm"] = max_norm
    if (not np.isfinite(max_norm)) or max_norm <= 1e-12:
        return target_pos_nm, stats

    unit_dirs = np.zeros_like(grad)
    nz = norms > 1e-12
    unit_dirs[nz] = grad[nz] / norms[nz, None]

    mode = str(proposal_mode or "plus").lower()
    dirs = unit_dirs.copy()

    if mode.startswith("minus"):
        dirs = -dirs

    if ("jitter" in mode) or mode.startswith("random"):
        if rng is None:
            rng = np.random.default_rng(0)
        noise = rng.normal(size=dirs.shape)
        noise_norm = np.linalg.norm(noise, axis=1)
        noise_u = np.zeros_like(noise)
        nnz = noise_norm > 1e-12
        noise_u[nnz] = noise[nnz] / noise_norm[nnz, None]

        if mode.startswith("random"):
            dirs = noise_u
            if mode.endswith("minus"):
                dirs = -dirs
        else:
            jitter_alpha = 0.35
            dirs = dirs + (jitter_alpha * noise_u)
            dnorm = np.linalg.norm(dirs, axis=1)
            dnz = dnorm > 1e-12
            dirs[dnz] = dirs[dnz] / dnorm[dnz, None]
            dirs[~dnz] = 0.0

    move_scale = np.clip(norms / max_norm, 0.0, 1.0)
    if mode.startswith("random"):
        move_scale = np.maximum(move_scale, 0.35)

    delta_nm = dirs * (cap_nm * move_scale)[:, None]

    if bad_mask is not None:
        bm = np.asarray(bad_mask)
        if bm.dtype == bool:
            if bm.shape != (lig_idx.size,):
                return target_pos_nm, stats
            keep = bm
        else:
            keep = np.zeros(lig_idx.size, dtype=bool)
            valid = bm[(bm >= 0) & (bm < lig_idx.size)]
            keep[valid.astype(int)] = True
        delta_nm = np.where(keep[:, None], delta_nm, 0.0)
        stats["targeted_atoms"] = int(np.count_nonzero(keep))

    target_pos_nm[lig_idx] = target_pos_nm[lig_idx] + delta_nm

    disp_nm = np.linalg.norm(delta_nm, axis=1)
    moved = disp_nm > 1e-12
    stats["moved_atoms"] = int(np.count_nonzero(moved))
    if stats["moved_atoms"] > 0:
        stats["mean_target_disp_A"] = float(np.mean(disp_nm[moved]) * 10.0)
        stats["max_target_disp_A"] = float(np.max(disp_nm[moved]) * 10.0)

    return target_pos_nm, stats


def build_targets_from_dihedral(
    accepted_pos_nm: np.ndarray,
    ligand_heavy_idx: np.ndarray,
    heavy_coords_A: np.ndarray,
    bond: Tuple[int, int],
    delta_theta: float,
    side_atoms: Iterable[int],
) -> Tuple[np.ndarray, dict]:
    """Build tether targets by rotating one side of a rotatable bond.

    ``heavy_coords_A`` are the current ligand heavy-atom coordinates in Å,
    in the same ordering as ``ligand_heavy_idx``. Rotation is applied to
    atoms listed in ``side_atoms`` (heavy-local indices) around the bond
    axis by ``delta_theta`` radians. Non-rotated atoms' targets equal their
    current positions, so the tether effectively pins them in place.
    """
    target_pos_nm = np.asarray(accepted_pos_nm, dtype=np.float64).copy()
    lig_idx = np.asarray(ligand_heavy_idx, dtype=int)
    heavy = np.asarray(heavy_coords_A, dtype=np.float64)

    stats = {
        "bond": (int(bond[0]), int(bond[1])),
        "delta_theta_deg": float(np.degrees(delta_theta)),
        "side_size": int(np.asarray(list(side_atoms), dtype=int).size),
        "moved_atoms": 0,
        "mean_target_disp_A": 0.0,
        "max_target_disp_A": 0.0,
    }

    if lig_idx.size == 0 or heavy.shape[0] != lig_idx.size:
        return target_pos_nm, stats

    side = np.asarray(list(side_atoms), dtype=int)
    if side.size == 0:
        return target_pos_nm, stats

    new_heavy_A = apply_dihedral_delta(heavy, bond, delta_theta, side)

    # Only the side atoms change; others keep their current position as target.
    target_pos_nm[lig_idx[side]] = new_heavy_A[side] * 0.1

    disp_A = np.linalg.norm(new_heavy_A[side] - heavy[side], axis=1)
    moved = disp_A > 1e-6
    stats["moved_atoms"] = int(np.count_nonzero(moved))
    if stats["moved_atoms"] > 0:
        stats["mean_target_disp_A"] = float(np.mean(disp_A[moved]))
        stats["max_target_disp_A"] = float(np.max(disp_A[moved]))

    return target_pos_nm, stats


def build_targets_from_subregion(
    accepted_pos_nm: np.ndarray,
    ligand_heavy_idx: np.ndarray,
    heavy_coords_A: np.ndarray,
    cluster_atoms: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """Build tether targets by applying ``R @ x + t`` to a bad-atom cluster.

    Atoms outside the cluster keep their current positions as targets, so the
    tether pins them in place. The per-atom cap on displacement is enforced by
    the caller's choice of ``epsilon`` when computing ``(R, t)`` via
    :func:`diagnostic.best_rigid_transform_for_cluster`.
    """
    target_pos_nm = np.asarray(accepted_pos_nm, dtype=np.float64).copy()
    lig_idx = np.asarray(ligand_heavy_idx, dtype=int)
    heavy = np.asarray(heavy_coords_A, dtype=np.float64)
    cluster = np.asarray(cluster_atoms, dtype=int)

    stats = {
        "cluster_size": int(cluster.size),
        "moved_atoms": 0,
        "mean_target_disp_A": 0.0,
        "max_target_disp_A": 0.0,
    }

    if lig_idx.size == 0 or heavy.shape[0] != lig_idx.size or cluster.size == 0:
        return target_pos_nm, stats

    new_heavy_A = apply_rigid_transform(heavy, cluster, R, t)
    target_pos_nm[lig_idx[cluster]] = new_heavy_A[cluster] * 0.1

    disp_A = np.linalg.norm(new_heavy_A[cluster] - heavy[cluster], axis=1)
    moved = disp_A > 1e-6
    stats["moved_atoms"] = int(np.count_nonzero(moved))
    if stats["moved_atoms"] > 0:
        stats["mean_target_disp_A"] = float(np.mean(disp_A[moved]))
        stats["max_target_disp_A"] = float(np.max(disp_A[moved]))

    return target_pos_nm, stats
