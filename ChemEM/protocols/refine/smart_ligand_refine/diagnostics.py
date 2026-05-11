from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional at import time
    Chem = None


def _ligand_mol(ligand):
    return getattr(ligand, "mol", ligand)


def _formal_charge(mol) -> Optional[int]:
    if mol is None or not hasattr(mol, "GetAtoms"):
        return None
    try:
        return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))
    except Exception:
        return None


def _heavy_indices(mol, n_coords: int) -> List[int]:
    if mol is None or not hasattr(mol, "GetAtoms"):
        return list(range(int(n_coords)))
    out = []
    for atom in mol.GetAtoms():
        idx = int(atom.GetIdx())
        if idx < int(n_coords) and int(atom.GetAtomicNum()) > 1:
            out.append(idx)
    return out or list(range(int(n_coords)))


def _symbols(mol, indices: Sequence[int]) -> List[str]:
    if mol is None or not hasattr(mol, "GetAtomWithIdx"):
        return ["?"] * len(indices)
    out = []
    for idx in indices:
        try:
            out.append(str(mol.GetAtomWithIdx(int(idx)).GetSymbol()))
        except Exception:
            out.append("?")
    return out


def read_sdf_mol(path: str):
    if Chem is None:
        return None
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    for mol in supplier:
        if mol is not None and mol.GetNumConformers():
            return mol
    return None


def read_sdf_coords(path: str) -> Optional[np.ndarray]:
    mol = read_sdf_mol(path)
    if mol is None:
        return None
    return np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)


def compare_reference_sdf(
    ligand,
    coords_A: np.ndarray,
    reference_sdf_path: Optional[str],
) -> Dict[str, object]:
    """Compare current ligand coordinates with a reference SDF by heavy-atom order."""
    if not reference_sdf_path:
        return {}

    mol = _ligand_mol(ligand)
    coords = np.asarray(coords_A, dtype=np.float64)
    ref_mol = read_sdf_mol(str(reference_sdf_path))
    if ref_mol is None:
        return {
            "reference_sdf_path": str(reference_sdf_path),
            "available": False,
            "reason": "reference_sdf_unreadable",
        }

    ref_coords = np.asarray(ref_mol.GetConformer(0).GetPositions(), dtype=np.float64)
    heavy = _heavy_indices(mol, coords.shape[0])
    ref_heavy = _heavy_indices(ref_mol, ref_coords.shape[0])
    n = min(len(heavy), len(ref_heavy))

    out: Dict[str, object] = {
        "reference_sdf_path": str(reference_sdf_path),
        "available": True,
        "input_atom_count": int(coords.shape[0]),
        "reference_atom_count": int(ref_coords.shape[0]),
        "input_heavy_atom_count": int(len(heavy)),
        "reference_heavy_atom_count": int(len(ref_heavy)),
        "input_formal_charge": _formal_charge(mol),
        "reference_formal_charge": _formal_charge(ref_mol),
        "compared_heavy_atom_count": int(n),
        "atom_count_mismatch": bool(coords.shape[0] != ref_coords.shape[0]),
        "heavy_atom_count_mismatch": bool(len(heavy) != len(ref_heavy)),
        "formal_charge_mismatch": bool(_formal_charge(mol) != _formal_charge(ref_mol)),
    }
    if n == 0:
        out["reason"] = "no_comparable_heavy_atoms"
        return out

    cur_idx = heavy[:n]
    dep_idx = ref_heavy[:n]
    cur = coords[cur_idx]
    dep = ref_coords[dep_idx]
    disp = np.linalg.norm(cur - dep, axis=1)
    centroid_shift = float(np.linalg.norm(np.mean(cur, axis=0) - np.mean(dep, axis=0)))
    rmsd = float(np.sqrt(np.mean(disp * disp)))
    symbols = _symbols(mol, cur_idx)
    ref_symbols = _symbols(ref_mol, dep_idx)

    worst = []
    for rank, row in enumerate(np.argsort(-disp)[: min(10, n)], start=1):
        worst.append(
            {
                "rank": int(rank),
                "atom_index": int(cur_idx[int(row)]),
                "reference_atom_index": int(dep_idx[int(row)]),
                "symbol": symbols[int(row)],
                "reference_symbol": ref_symbols[int(row)],
                "distance_A": float(disp[int(row)]),
            }
        )

    out.update(
        {
            "heavy_rmsd_A": rmsd,
            "heavy_centroid_shift_A": centroid_shift,
            "max_heavy_displacement_A": float(np.max(disp)),
            "worst_heavy_displacements": worst,
            "symbol_mismatches": [
                {
                    "atom_index": int(cur_idx[i]),
                    "reference_atom_index": int(dep_idx[i]),
                    "symbol": symbols[i],
                    "reference_symbol": ref_symbols[i],
                }
                for i in range(n)
                if symbols[i] != ref_symbols[i]
            ],
        }
    )
    return out
