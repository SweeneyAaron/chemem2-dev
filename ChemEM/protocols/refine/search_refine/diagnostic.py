# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""
Per-atom fit diagnostics for search_refine.

Combines Q-score (per-atom density coverage) with per-atom CCC gradient norm
into a "badness" score used to identify atoms that are mispositioned and
should be the target of proposal generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ChemEM.protocols.mapQ_score.mapq_utils import compute_qscores_from_emmap


@dataclass
class AtomFitQuality:
    q_score: np.ndarray      # (N,) per-atom Q-score, higher = better
    grad_norm: np.ndarray    # (N,) per-atom |∇CCC|, higher = atom "wants to move more"
    badness: np.ndarray      # (N,) combined score, higher = worse fit


@dataclass
class AtomClassification:
    good_idx: np.ndarray     # atoms with q >= q_good_thresh
    bad_idx: np.ndarray      # atoms with q <= q_bad_thresh
    neutral_idx: np.ndarray  # atoms in between


def atom_fit_quality(
    heavy_coords_A: np.ndarray,
    atom_gradient: np.ndarray,
    local_map,
    *,
    sigma_ref: float = 0.6,
    radii: Optional[np.ndarray] = None,
    protein_xyz_A: Optional[np.ndarray] = None,
) -> AtomFitQuality:
    """Compute per-atom Q-score and gradient norm; combine into a badness score.

    badness = (1 - q) * (1 + grad_norm / median(grad_norm))

    Low Q and high gradient norm ⇒ high badness ⇒ prime candidate to move.

    Args:
        heavy_coords_A: (N, 3) ligand heavy atom coords in Å.
        atom_gradient:  (N, 3) per-atom CCC gradient (higher-is-better).
        local_map:      EMMap-like (density_map, origin, apix).
        sigma_ref:      Q-score reference Gaussian width.
        radii:          Optional explicit Q-score radii array.
        protein_xyz_A:  Optional protein heavy-atom coords used to populate
                        the neighbor KDTree inside compute_qscores_from_emmap,
                        so the Q sampling avoids protein atom positions.
                        Mirrors the QScorer's combined-input behavior.
    """
    coords = np.asarray(heavy_coords_A, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"heavy_coords_A must be (N,3), got {coords.shape}")

    grad = np.asarray(atom_gradient, dtype=np.float64)
    if grad.shape != coords.shape:
        raise ValueError(
            f"atom_gradient shape {grad.shape} does not match coords {coords.shape}"
        )

    n_lig = int(coords.shape[0])

    if protein_xyz_A is not None and np.asarray(protein_xyz_A).size:
        prot = np.asarray(protein_xyz_A, dtype=np.float64)
        combined = np.concatenate([coords, prot], axis=0)
    else:
        combined = coords

    try:
        q_all = compute_qscores_from_emmap(
            atoms_xyz=combined,
            emmap=local_map,
            sigma_ref=float(sigma_ref),
            radii=radii,
        )
        q = np.asarray(q_all[:n_lig], dtype=np.float64)
        q = np.where(np.isfinite(q), q, 0.0)
    except Exception:
        q = np.zeros(n_lig, dtype=np.float64)

    grad_norm = np.linalg.norm(grad, axis=1)
    med = float(np.median(grad_norm))
    rel = grad_norm / med if med > 1e-12 else np.zeros_like(grad_norm)

    q_clip = np.clip(q, 0.0, 1.0)
    badness = (1.0 - q_clip) * (1.0 + rel)

    return AtomFitQuality(q_score=q, grad_norm=grad_norm, badness=badness)


def classify_atoms(
    fit_quality: AtomFitQuality,
    *,
    q_good_thresh: float = 0.7,
    q_bad_thresh: float = 0.3,
) -> AtomClassification:
    """Partition atoms by Q-score into good / bad / neutral index sets."""
    q = np.asarray(fit_quality.q_score, dtype=np.float64)
    good = np.nonzero(q >= float(q_good_thresh))[0].astype(int)
    bad = np.nonzero(q <= float(q_bad_thresh))[0].astype(int)
    neutral = np.nonzero((q > float(q_bad_thresh)) & (q < float(q_good_thresh)))[0].astype(int)
    return AtomClassification(good_idx=good, bad_idx=bad, neutral_idx=neutral)


def format_diagnostic(
    fit_quality: AtomFitQuality,
    classification: AtomClassification,
) -> str:
    """One-line summary for per-iter logs."""
    q = fit_quality.q_score
    g = fit_quality.grad_norm
    q_mean = float(np.mean(q)) if q.size else 0.0
    q_min = float(np.min(q)) if q.size else 0.0
    g_max = float(np.max(g)) if g.size else 0.0
    return (
        f"diag: q(mean={q_mean:.3f}, min={q_min:.3f}) "
        f"g_max={g_max:.3g} "
        f"good={classification.good_idx.size} "
        f"bad={classification.bad_idx.size} "
        f"neutral={classification.neutral_idx.size}"
    )


# ----------------------------------------------------------------------
# Dihedral rearrangement helpers
# ----------------------------------------------------------------------


@dataclass
class DihedralCandidate:
    bond: Tuple[int, int]        # (heavy_a, heavy_b) — heavy-atom local indices
    alignment: float             # cosine of motion direction with target direction, in [-1, 1]
    side_atoms: np.ndarray       # heavy indices of the side containing the targeted atom
    v_norm: float                # |axis × (x_atom - pivot)|; distance from axis


def _heavy_index_map(mol) -> Tuple[List[int], dict]:
    """Return (heavy_atoms_mol_idx_list, mol_idx_to_heavy_idx dict).

    heavy_atoms_mol_idx_list[k] is the RDKit atom index of the k-th heavy atom.
    The k-th heavy atom corresponds to row k in heavy_coords_A / the k-th entry
    in ligand_heavy_indices.
    """
    heavy = [int(a.GetIdx()) for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    mol_to_heavy = {m: h for h, m in enumerate(heavy)}
    return heavy, mol_to_heavy


def enumerate_rotatable_heavy_bonds(mol) -> List[Tuple[int, int]]:
    """Return rotatable single bonds between heavy atoms, in heavy-local indices.

    A rotatable bond here is a single, non-ring, non-aromatic bond between two
    heavy atoms that each have at least one other heavy-atom neighbor (i.e.
    neither endpoint is a heavy terminal). Operates directly on the heavy-atom
    subgraph, so the result is unchanged whether the input mol has explicit
    hydrogens or not.
    """
    from rdkit.Chem import BondType

    _, mol_to_heavy = _heavy_index_map(mol)
    n_heavy = len(mol_to_heavy)

    heavy_degree = [0] * n_heavy
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        if i in mol_to_heavy and j in mol_to_heavy:
            heavy_degree[mol_to_heavy[i]] += 1
            heavy_degree[mol_to_heavy[j]] += 1

    out: List[Tuple[int, int]] = []
    seen = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() != BondType.SINGLE:
            continue
        if bond.GetIsAromatic() or bond.IsInRing():
            continue
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        if i not in mol_to_heavy or j not in mol_to_heavy:
            continue
        hi = int(mol_to_heavy[i])
        hj = int(mol_to_heavy[j])
        if heavy_degree[hi] <= 1 or heavy_degree[hj] <= 1:
            continue
        key = (hi, hj) if hi < hj else (hj, hi)
        if key in seen:
            continue
        seen.add(key)
        out.append((hi, hj))
    return out


def bond_sides_heavy(
    mol,
    bond_heavy: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Partition heavy atoms into two sides after removing the given bond.

    Returns (side_a_heavy_idx, side_b_heavy_idx) as np.int arrays, where
    side_a is the side containing bond_heavy[0]. Returns (None, None) for
    ring bonds (where both sides remain connected via another path).
    """
    heavy_list, mol_to_heavy = _heavy_index_map(mol)
    a_h, b_h = bond_heavy
    a_mol = heavy_list[a_h]
    b_mol = heavy_list[b_h]

    visited = set()
    stack = [a_mol]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        atom = mol.GetAtomWithIdx(v)
        for nb in atom.GetNeighbors():
            nb_idx = int(nb.GetIdx())
            # Skip the bond being broken, in either direction.
            if v == a_mol and nb_idx == b_mol:
                continue
            if v == b_mol and nb_idx == a_mol:
                continue
            if nb_idx not in visited:
                stack.append(nb_idx)

    if b_mol in visited:
        # Ring bond — both ends are still connected without this edge.
        return None, None

    side_a = sorted(mol_to_heavy[m] for m in visited if m in mol_to_heavy)
    all_heavy = set(range(len(heavy_list)))
    side_b = sorted(all_heavy - set(side_a))

    return np.asarray(side_a, dtype=int), np.asarray(side_b, dtype=int)


def rank_dihedrals_for_atom(
    atom_heavy_idx: int,
    target_dir: np.ndarray,
    heavy_coords_A: np.ndarray,
    rotatable_bonds: List[Tuple[int, int]],
    mol,
) -> List[DihedralCandidate]:
    """Rank rotatable bonds by how well their rotation moves ``atom_heavy_idx``
    in the given target direction. Returns a list sorted by ``|alignment|``
    descending. Bonds where ``atom_heavy_idx`` lies on the rotation axis
    (v_norm ≈ 0) are skipped.
    """
    coords = np.asarray(heavy_coords_A, dtype=np.float64)
    td = np.asarray(target_dir, dtype=np.float64)
    td_norm = float(np.linalg.norm(td))
    if td_norm < 1e-12:
        return []
    td = td / td_norm

    out: List[DihedralCandidate] = []
    for bond in rotatable_bonds:
        a_h, b_h = bond
        side_a, side_b = bond_sides_heavy(mol, bond)
        if side_a is None:
            continue

        if atom_heavy_idx in side_a.tolist():
            side = side_a
            pivot = coords[b_h]
        elif atom_heavy_idx in side_b.tolist():
            side = side_b
            pivot = coords[a_h]
        else:
            continue

        axis = coords[b_h] - coords[a_h]
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            continue
        axis = axis / axis_norm

        v = np.cross(axis, coords[atom_heavy_idx] - pivot)
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-9:
            continue

        align = float(np.dot(v / v_norm, td))
        out.append(
            DihedralCandidate(
                bond=(int(a_h), int(b_h)),
                alignment=align,
                side_atoms=side,
                v_norm=v_norm,
            )
        )

    out.sort(key=lambda d: abs(d.alignment), reverse=True)
    return out


def _heavy_adjacency(mol) -> List[set]:
    """Heavy-atom bond graph: adj[h] = set of heavy-local indices bonded to h."""
    _, mol_to_heavy = _heavy_index_map(mol)
    n_heavy = len(mol_to_heavy)
    adj: List[set] = [set() for _ in range(n_heavy)]
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        if i in mol_to_heavy and j in mol_to_heavy:
            hi = mol_to_heavy[i]
            hj = mol_to_heavy[j]
            adj[hi].add(hj)
            adj[hj].add(hi)
    return adj


def cluster_bad_atoms(
    bad_idx: np.ndarray,
    mol,
) -> List[np.ndarray]:
    """Group bad-atom heavy indices into contiguous sub-regions.

    Two bad atoms belong to the same cluster iff they are directly bonded in
    the heavy-atom graph, OR connected through a chain of other bad atoms.
    Returns a list of np.int arrays of heavy indices (one per connected
    component within the bad-atom set). Empty input yields an empty list.
    """
    bad = np.asarray(bad_idx, dtype=int).ravel()
    if bad.size == 0:
        return []

    adj = _heavy_adjacency(mol)
    bad_set = set(int(x) for x in bad.tolist())

    clusters: List[np.ndarray] = []
    visited: set = set()
    for seed in bad.tolist():
        s = int(seed)
        if s in visited:
            continue
        component = []
        stack = [s]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            component.append(v)
            for nb in adj[v]:
                if nb in bad_set and nb not in visited:
                    stack.append(nb)
        clusters.append(np.asarray(sorted(component), dtype=int))
    return clusters


def best_rigid_transform_for_cluster(
    cluster_coords_A: np.ndarray,
    target_dirs: np.ndarray,
    epsilon_A: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form Kabsch fit aligning ``cluster_coords_A`` to ``coords + ε·dirs``.

    Returns ``(R, t)`` such that ``R @ x + t`` approximates
    ``x + epsilon_A * target_dir(x)`` for atoms in the cluster. Target
    directions are expected per-atom (shape (M, 3)); zero rows are tolerated.
    For a cluster of 1 atom the rotation collapses to identity and ``t`` is
    the pure translation ``ε·dir``.
    """
    coords = np.asarray(cluster_coords_A, dtype=np.float64)
    dirs = np.asarray(target_dirs, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"cluster_coords_A must be (M,3), got {coords.shape}")
    if dirs.shape != coords.shape:
        raise ValueError(
            f"target_dirs shape {dirs.shape} does not match coords {coords.shape}"
        )
    eps = float(epsilon_A)

    desired = coords + eps * dirs
    c_src = coords.mean(axis=0)
    c_dst = desired.mean(axis=0)

    P = coords - c_src
    Q = desired - c_dst

    H = P.T @ Q
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64), c_dst - c_src

    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    if d == 0.0:
        d = 1.0
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = c_dst - R @ c_src
    return R, t


def apply_rigid_transform(
    heavy_coords_A: np.ndarray,
    cluster_atoms: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Apply ``R @ x + t`` to rows ``cluster_atoms`` of ``heavy_coords_A``."""
    coords = np.asarray(heavy_coords_A, dtype=np.float64)
    idx = np.asarray(cluster_atoms, dtype=int)
    out = coords.copy()
    if idx.size == 0:
        return out
    out[idx] = (coords[idx] @ np.asarray(R, dtype=np.float64).T) + np.asarray(t, dtype=np.float64)
    return out


def directed_kick_proposal(
    heavy_coords_A: np.ndarray,
    atom_gradient: np.ndarray,
    q_score: np.ndarray,
    rotatable_bonds: List[Tuple[int, int]],
    mol,
    kick_angle_deg: float,
) -> Optional[dict]:
    """Build a directed basin-hopping kick: a dihedral delta on the worst-Q atom
    aligned with that atom's gradient direction.

    Returns ``None`` when no viable dihedral exists (no rotatable bonds, empty
    Q array, near-zero gradient, or the worst atom lies on every candidate
    axis). The returned dict has keys:

    - ``bond``: heavy-local (a, b) of the selected rotatable bond
    - ``bad_atom``: heavy index of the worst-Q atom
    - ``side_atoms``: atoms on the bad-atom's side of the bond
    - ``delta_theta_rad``, ``delta_theta_deg``: signed rotation magnitude
    - ``alignment``: alignment score from :func:`rank_dihedrals_for_atom`
    - ``new_heavy_coords_A``: rotated heavy-atom coordinates
    """
    q = np.asarray(q_score, dtype=np.float64).ravel()
    grad = np.asarray(atom_gradient, dtype=np.float64)
    coords = np.asarray(heavy_coords_A, dtype=np.float64)
    if q.size == 0 or not rotatable_bonds or coords.shape[0] == 0:
        return None

    worst_idx = int(np.argmin(q))
    g = grad[worst_idx]
    g_norm = float(np.linalg.norm(g))
    if g_norm < 1e-12:
        return None
    target_dir = g / g_norm

    try:
        ranked = rank_dihedrals_for_atom(
            atom_heavy_idx=worst_idx,
            target_dir=target_dir,
            heavy_coords_A=coords,
            rotatable_bonds=rotatable_bonds,
            mol=mol,
        )
    except Exception:
        return None
    if not ranked:
        return None

    entry = ranked[0]
    sign = 1.0 if entry.alignment > 0 else -1.0
    dtheta_rad = float(np.radians(float(sign) * float(kick_angle_deg)))
    new_coords = apply_dihedral_delta(coords, entry.bond, dtheta_rad, entry.side_atoms)

    return {
        "bond": entry.bond,
        "bad_atom": worst_idx,
        "side_atoms": entry.side_atoms,
        "delta_theta_rad": dtheta_rad,
        "delta_theta_deg": float(sign) * float(kick_angle_deg),
        "alignment": float(entry.alignment),
        "new_heavy_coords_A": new_coords,
    }


def apply_dihedral_delta(
    heavy_coords_A: np.ndarray,
    bond: Tuple[int, int],
    delta_theta: float,
    side_atoms: np.ndarray,
) -> np.ndarray:
    """Rotate ``side_atoms`` around the bond axis by ``delta_theta`` (radians).

    Atoms NOT in ``side_atoms`` are untouched. Returns a new (N, 3) array.
    """
    coords = np.asarray(heavy_coords_A, dtype=np.float64)
    side = np.asarray(side_atoms, dtype=int)
    a_h, b_h = bond

    x1 = coords[a_h]
    axis = coords[b_h] - x1
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9 or side.size == 0:
        return coords.copy()
    axis = axis / norm

    c = float(np.cos(delta_theta))
    s = float(np.sin(delta_theta))
    ux, uy, uz = float(axis[0]), float(axis[1]), float(axis[2])
    K = np.array(
        [[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]],
        dtype=np.float64,
    )
    R = np.eye(3) * c + s * K + (1.0 - c) * np.outer(axis, axis)

    new_coords = coords.copy()
    offsets = coords[side] - x1
    new_coords[side] = offsets @ R.T + x1
    return new_coords
