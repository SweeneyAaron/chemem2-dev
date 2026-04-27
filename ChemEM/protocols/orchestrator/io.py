# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Output writers for SmartOrchestrator.

Per-gate JSON audit (gate1/gate2/gate3.json) lets a user retrace why each
final assignment was picked. final_complex.pdb is the assembled answer:
receptor + every assigned ligand pose, with distinct chain IDs when the
same ligand is placed in multiple sites.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Dict, List

from openmm import unit
from rdkit import Chem
from rdkit.Geometry import Point3D

from ChemEM.parsers.writers import save_structure_parmed, write_to_sdf

from .state import FinalAssignment, PoseCandidate


def write_gate_json(
    candidates_by_site: Dict[str, List[PoseCandidate]],
    path: str,
) -> None:
    payload = {
        str(site_id): [c.to_dict() for c in cands]
        for site_id, cands in candidates_by_site.items()
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_assignments_json(assignments: List[FinalAssignment], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([a.to_dict() for a in assignments], f, indent=2)


def write_summary_json(
    assignments: List[FinalAssignment],
    gate_counts: Dict[str, Dict[str, int]],
    path: str,
) -> None:
    payload = {
        "gate_counts": gate_counts,
        "assignments": [a.to_dict() for a in assignments],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _ligand_with_coords(ligand, coords) -> Chem.Mol:
    """Return a deep-copy of ligand.mol with conformer 0 set to coords."""
    rdmol = Chem.Mol(ligand.mol)
    conf = rdmol.GetConformer(0)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return rdmol


def write_assignment_sdfs(
    assignments: List[FinalAssignment],
    ligands,
    out_dir: str,
) -> None:
    """One SDF per assignment under <out_dir>/Ligand_{i}/site_{s}_chain_{c}.sdf."""
    for a in assignments:
        ligand = ligands[a.ligand_idx]
        sub = os.path.join(out_dir, f"Ligand_{a.ligand_idx}")
        os.makedirs(sub, exist_ok=True)
        rdmol = _ligand_with_coords(ligand, a.coords)
        sdf_path = os.path.join(
            sub, f"site_{a.site_id}_chain_{a.chain_id}.sdf"
        )
        write_to_sdf(rdmol, sdf_path)


def write_final_complex_pdb(
    assignments: List[FinalAssignment],
    system,
    path: str,
) -> None:
    """Write receptor + every assigned ligand pose into one PDB.

    Each ligand copy gets a distinct chain ID (FinalAssignment.chain_id)
    so the same ligand placed in multiple sites is unambiguous in the file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    assembled = copy.deepcopy(system.protein.complex_structure)

    for a in assignments:
        ligand = system.ligand[a.ligand_idx]
        lig_struct = copy.deepcopy(ligand.complex_structure)
        # Plant the assignment coordinates onto the ligand structure copy.
        for atom, (x, y, z) in zip(lig_struct.atoms, a.coords):
            atom.xx, atom.xy, atom.xz = float(x), float(y), float(z)
        # Stamp the chain ID so duplicate ligands are distinguishable.
        for residue in lig_struct.residues:
            residue.chain = a.chain_id
        for atom in lig_struct.atoms:
            atom.residue.chain = a.chain_id
        assembled = assembled + lig_struct

    save_structure_parmed(assembled, path)
