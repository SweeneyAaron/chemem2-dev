# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


import numpy as np 
from openmm import unit

from typing import Dict, List, Tuple, Any

# -------------------------
# Residue mapping by coords
# -------------------------
def to_angstrom_positions(positions) -> np.ndarray:
    if hasattr(positions, "value_in_unit"):
        return np.asarray(positions.value_in_unit(unit.angstrom), dtype=float)
    try:
        return np.array([[p.x, p.y, p.z] for p in positions], dtype=float)
    except Exception:
        raise TypeError("Unsupported positions format for conversion to Å.")

def chain_id(res) -> str:
    try:
        return (getattr(res.chain, "id", None)).strip() or "?"
    except Exception:
        return (getattr(res, "chain", None) or getattr(res, "segid", None) or "").strip() or "?"

def atomname_to_idx(res) -> Dict[str, List[int]]:
    m: Dict[str, List[int]] = {}
    for a in res.atoms():
        try:
            m.setdefault(a.name, []).append(a.index)
        except Exception:
            m.setdefault(a.name, []).append(a.idx)
    return m

def max_namewise_delta(posA, idxsA, posB, idxsB) -> float:
    PA = posA[idxsA]
    PB = posB[idxsB]
    return max(np.linalg.norm(PB - p, axis=1).min() for p in PA)

def residues_match_by_backbone(rA, rB, posA, posB, tol_ang: float = 1e-5) -> bool:
    mapA = atomname_to_idx(rA)
    mapB = atomname_to_idx(rB)
    common_bb = [n for n in ("N", "CA", "C", "O") if (n in mapA and n in mapB)]
    if len(common_bb) < 2:
        return False
    for name in common_bb:
        d = max_namewise_delta(posA, mapA[name], posB, mapB[name])
        if d > tol_ang:
            return False
    return True



def build_residue_map_by_positions(orig_topology, orig_positions,
                                   final_topology, final_positions,
                                   tol_ang: float = 1e-3) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """
    Build a residue label map by comparing backbone atom coordinates (Å).

    Returns:
      res_map: {(orig_chain, orig_resid) -> (final_chain, final_resid)}
    """
    
    posA = to_angstrom_positions(orig_positions)
    posB = to_angstrom_positions(final_positions)

    orig_residues = list(orig_topology.residues())
    final_residues = list(final_topology.residues())

    buckets: Dict[str, List[Any]] = {}
    for r in final_residues:
        buckets.setdefault(r.name, []).append(r)

    res_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
    matched = 0

    for rA in orig_residues:
        keyA = (chain_id(rA), str(rA.id))
        candidates = buckets.get(rA.name, [])

        hit = None
        for rB in candidates:
            if residues_match_by_backbone(rA, rB, posA, posB, tol_ang=1e-5):
                hit = rB
                break

        if hit is not None:
            res_map[keyA] = (chain_id(hit), str(hit.id))
            candidates.remove(hit)
            matched += 1

    print(f'[INFO] Matched {matched} original residues of {len(orig_residues)} to final {len(final_residues)}')
    return res_map


def get_openmm_mappings(topology) -> List[Tuple[str, str]]:
    return [(res.chain.id, res.id) for res in topology.residues()]
