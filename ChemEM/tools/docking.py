# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

import os
import numpy as np
from typing import Iterable, List, Tuple, Optional

from rdkit import Chem
from rdkit.Geometry import Point3D
from ChemEM.tools.ligand import write_to_sdf
from ChemEM.tools.geometry import rmsd_cluster
from ChemEM.data.data import HALOGEN_DONOR_ATOM_IDXS
Pose = Tuple[float, np.ndarray]  # (score, coords[N,3])

def compute_halogen_bond_data(mol,atom_types, valid_atom_types = HALOGEN_DONOR_ATOM_IDXS):
    
    ligand_donor_indices = []
    ligand_donor_root_indices = []
    
    # Identify halogen bond donors in the ligand (e.g., Cl, Br, I)
    for atom, atom_type in zip(mol.GetAtoms(), atom_types):
        if atom_type in valid_atom_types:
            donor_idx = atom.GetIdx()
            # Get heavy (non-hydrogen) neighbors; assume first is the donor root
            heavy_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() != 'H']
            donor_root_idx = heavy_neighbors[0].GetIdx() if heavy_neighbors else None
            
            ligand_donor_indices.append(donor_idx)
            ligand_donor_root_indices.append(donor_root_idx)
    
    
    return np.array(ligand_donor_indices), np.array(ligand_donor_root_indices)
    


#keep in docking
def energy_cutoff(sorted_poses: List[Pose], delta: float = 0.1) -> List[Pose]:
    """
    Keep poses with score <= best_score + delta.
    Assumes 'lower is better'.
    """
    if not sorted_poses:
        return []
    thresh = float(sorted_poses[0][0] + delta)
    return [p for p in sorted_poses if p[0] <= thresh]

#move to ligan tools
def mol_with_positions(template_mol: Chem.Mol, coords: np.ndarray, conf_id: int = 0) -> Chem.Mol:
    """Return a copy of template_mol with a single conformer built from coords."""
    newmol = Chem.Mol(template_mol)
    newmol.RemoveAllConformers()

    n_atoms = newmol.GetNumAtoms()
    if coords.shape[0] != n_atoms or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N,3) with N={n_atoms}; got {coords.shape}")

    conf = Chem.Conformer(n_atoms)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    conf.SetId(conf_id)
    newmol.AddConformer(conf, assignId=True)
    return newmol


#keep in docking
def write_results(template_ligand, docked_solutions: List[Pose], out_dir: str) -> None:
    """
    Write poses as SDFs + a results.txt.
    template_ligand is your Ligand wrapper with `.mol`.
    """
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    for num, (score, pos) in enumerate(docked_solutions):
        mol_id = f"Ligand_{num}"
        lines.append(f"{mol_id} : {score}\n")
        new_mol = mol_with_positions(template_ligand.mol, pos)
        write_to_sdf(new_mol, os.path.join(out_dir, mol_id + ".sdf"))

    with open(os.path.join(out_dir, "results.txt"), "w") as f:
        f.writelines(lines)

#keep in docking
def dock_worker(block, echo_serialised, centroid, radius, cpus_per_site: int):
    """
    One split-site docking run executed in its own process.
    NOTE: echo_serialised.copy() must be picklable.
    """
    os.environ["OMP_NUM_THREADS"] = str(cpus_per_site)
    os.environ["MKL_NUM_THREADS"] = str(cpus_per_site)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpus_per_site)

    pre = echo_serialised.copy()
    pre.add_multi_site_bias(centroid, radius)

    # Keep import inside worker so the subprocess starts cleanly.
    from ChemEM import docking2
    return docking2.run_aco_docking(pre, block)
