# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>
"""
Covalent-ligand OUTPUT helpers.

When a covalent ligand is refined, the reacted product (leaving group removed, the
S-O/S-N/... junction bond formed) only exists inside the merged parmed structure that
`inject_covalent_bonds` builds. The plain refiners instead serialize `ligand.mol`, which
still carries the leaving group and has no protein bond, so their covalent outputs are
"pre-reaction" and wrong.

This module produces the two covalent deliverables from the refined state:
  1. a full receptor+ligand complex PDB (protein + F-removed, covalently-bonded ligand);
  2. a product-form ligand SDF (leaving group removed, warhead at the bonded geometry)
     for an external tool (Coot/Phenix/...).

Everything here is gated by the caller on `ligand.covalent_link`; non-covalent output is
untouched.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from .writers import write_to_sdf, save_structure_parmed


def ligand_residue_coords_by_name(structure, lig_res_name: str) -> Dict[str, Tuple[float, float, float]]:
    """Map ``atom name -> (x, y, z)`` for the ligand residue in a merged parmed
    structure. Atom names follow the element+1-based-count scheme, matching the
    ``chemem_name`` tags set by ``_truncate_ligand_for_fragment``.
    """
    coords: Dict[str, Tuple[float, float, float]] = {}
    for res in structure.residues:
        if res.name != lig_res_name:
            continue
        for atom in res.atoms:
            coords[atom.name] = (float(atom.xx), float(atom.xy), float(atom.xz))
    return coords


def build_full_ligand_mol_at_refined_pose(ligand, coords_by_name: Dict[str, Tuple[float, float, float]]):
    """Return a copy of the FULL ligand.mol (leaving group RETAINED) moved to the
    refined pose. The covalent bond/reaction is represented only in the complex PDB;
    the ligand SDF stays a complete, valid molecule.

    Atoms present in ``coords_by_name`` (matched by the element+count atom-name scheme,
    which is how the merged covalent structure names ligand atoms) are moved to their
    refined coordinates. Leaving atoms (absent from ``coords_by_name`` because they were
    stripped from the reacted structure, e.g. the sulfonyl fluoride's F) are translated
    by a bonded neighbour's displacement so their bond geometry is preserved.
    ``ligand.mol`` itself is not mutated.
    """
    mol = Chem.Mol(ligand.mol)
    conf = mol.GetConformer()
    names = ligand.atom_names  # per-atom-index element+count names
    old = np.asarray(conf.GetPositions(), dtype=float)

    updated = [False] * mol.GetNumAtoms()
    for i, name in enumerate(names):
        xyz = coords_by_name.get(name)
        if xyz is not None:
            conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
            updated[i] = True

    # Translate leaving atoms with a refined bonded neighbour (keep bond geometry).
    for i in range(mol.GetNumAtoms()):
        if updated[i]:
            continue
        nbr = next((nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors()
                    if updated[nb.GetIdx()]), None)
        if nbr is not None:
            disp = np.asarray(conf.GetAtomPosition(nbr)) - old[nbr]
            new_pos = old[i] + disp
            conf.SetAtomPosition(i, Point3D(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))
    return mol


def build_covalent_complex_structure(protein_structure, ligand):
    """Merge the protein with the covalent ligand (at its current pose) and inject the
    covalent junction (which strips the leaving atoms). Returns the merged parmed
    ``Structure`` (protein + F-removed bonded ligand).

    The protein is copied INTERNALLY via parmed's ``.copy()`` (which — unlike
    ``copy.deepcopy`` — preserves the ``symmetry`` attribute the PDB writer needs),
    so the caller's live ``system.protein.complex_structure`` is never mutated by the
    in-place ``+=`` merge. ``ligand.complex_structure.positions`` must already be at
    the refined pose.
    """
    # Lazy imports: parsers must not import protocols at module load (circular).
    from ChemEM.protocols.core.simulation import merge_structures
    from ChemEM.parsers.covalent_fragment import inject_covalent_bonds

    protein_copy = protein_structure.copy(type(protein_structure))
    merged = merge_structures(protein_copy, [ligand.complex_structure])
    inject_covalent_bonds(merged, [ligand])
    return merged


def write_full_ligand_sdf_at_refined_pose(ligand, path: str,
                                          coords_by_name: Dict[str, Tuple[float, float, float]]) -> None:
    """Write the FULL ligand SDF (leaving group retained) at the refined pose."""
    mol = build_full_ligand_mol_at_refined_pose(ligand, coords_by_name)
    write_to_sdf(mol, path)


def _append_ligand_conect_records(structure, path: str, ligand_res_prefix: str = "LIG") -> None:
    """Append CONECT records for every bond involving a ligand-residue atom (the
    intra-ligand bonds AND the covalent junction bond to the protein), so viewers
    like ChimeraX draw the protein<->ligand covalent bond. parmed's PDB writer emits
    no CONECT records, and viewers will not connect a HETATM ligand to a protein ATOM
    across residues by distance.

    Serials are read back from the written PDB (order = structure-atom order), so no
    assumption is made about parmed's renumbering.
    """
    # Map structure-atom order -> written PDB serial.
    serials = []
    with open(path) as fh:
        pdb_lines = fh.readlines()
    for line in pdb_lines:
        if line.startswith(("ATOM", "HETATM")):
            try:
                serials.append(int(line[6:11]))
            except ValueError:
                serials.append(None)
    if len(serials) != len(structure.atoms):
        print(f"[covalent] CONECT records skipped: PDB atom count {len(serials)} "
              f"!= structure atom count {len(structure.atoms)}")
        return

    lig_atom_idx = set()
    for res in structure.residues:
        if str(res.name).startswith(ligand_res_prefix):
            for atom in res.atoms:
                lig_atom_idx.add(atom.idx)
    if not lig_atom_idx:
        return

    from collections import defaultdict
    neighbours = defaultdict(set)
    for bond in structure.bonds:
        i, j = bond.atom1.idx, bond.atom2.idx
        if i in lig_atom_idx:
            neighbours[i].add(j)
        if j in lig_atom_idx:
            neighbours[j].add(i)

    conect_lines = []
    for i in sorted(lig_atom_idx):
        if i not in neighbours or serials[i] is None:
            continue
        ns = [n for n in sorted(neighbours[i]) if serials[n] is not None]
        for k in range(0, len(ns), 4):  # max 4 bonded serials per CONECT line
            chunk = ns[k:k + 4]
            conect_lines.append(
                "CONECT" + "%5d" % serials[i] + "".join("%5d" % serials[n] for n in chunk)
            )
    if not conect_lines:
        return

    # Insert CONECT records before END/MASTER (PDB ordering).
    kept = [l for l in pdb_lines if not l.startswith(("END", "MASTER"))]
    kept.extend(line + "\n" for line in conect_lines)
    kept.append("END\n")
    with open(path, "w") as fh:
        fh.writelines(kept)


def write_covalent_complex_pdb(structure, path: str) -> None:
    """Write a merged protein+ligand parmed structure (the covalent complex) to PDB,
    with CONECT records for the ligand + covalent junction so viewers draw the bond."""
    save_structure_parmed(structure, path)
    try:
        _append_ligand_conect_records(structure, path)
    except Exception as exc:
        print(f"[covalent] could not append CONECT records: {exc}")
