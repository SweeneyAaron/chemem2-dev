#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:08:10 2026

@author: aaron.sweeney
"""

from rdkit import Chem

def check_unassigned_chirality(mol):
    
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    unassigned = [i for i in chiral_centers if i[1] == "?"]

    if unassigned:
        print(f"[WARNING] [ChemEM] Unassigned chiral centers detected. Attempting automated structural assignment.")
        print(f"          Initial chiral state (Atom Index, Designation): {chiral_centers}")
        try:
            Chem.AssignAtomChiralTagsFromStructure(mol)
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            print(f"[INFO]    [ChemEM] Chirality assignment successful. Updated state: {chiral_centers}")
        except Exception as e:
            print(f"[ERROR]   [ChemEM] Automated structural assignment failed: {e}")
            print(f"[WARNING] [ChemEM] Fallback initiated: Proceeding to load ligand with unassigned chiral centers.")
            
def set_ligand_rings(mol):
    try:
        _ = Chem.GetSymmSSSR(mol)
    
    except Exception as e:
        print(
            "ChemEM- Non-Fatal warning ring info assignment failed "
            f"with GetSymmSSSR. Full Error: {e}"
        )

def transfer_mol_coords(ref_mol, new_mol):
    """Copy ref_mol's coordinates onto new_mol via a substructure match.

    new_mol does not need a conformer: every matched position is overwritten
    here, so an empty one is allocated rather than requiring the caller to
    embed first (ETKDG can fail on large flexible ligands, and the embedded
    coordinates would be discarded anyway).

    Returns None if the coordinates can't be transferred faithfully, letting
    callers fall back rather than propagate a half-populated conformer.
    """

    if not ref_mol.GetNumConformers():
        return None

    match = new_mol.GetSubstructMatch(ref_mol)
    if not match:
        return None

    # GetSubstructMatch always covers the whole query, so the meaningful check
    # is on new_mol: any atom of it left out of the match would keep the origin
    # placeholder coordinates of a freshly allocated conformer.
    if len(match) != new_mol.GetNumAtoms():
        return None

    if not new_mol.GetNumConformers():
        new_mol.AddConformer(Chem.Conformer(new_mol.GetNumAtoms()), assignId=True)

    ref_mol_conformer = ref_mol.GetConformer()
    new_mol_conformer = new_mol.GetConformer()

    # match[i] is already a new_mol atom index for ref_mol atom i.
    for ref_atom_idx, new_atom_idx in enumerate(match):
        new_mol_conformer.SetAtomPosition(new_atom_idx,
                                          ref_mol_conformer.GetAtomPosition(ref_atom_idx))

    return new_mol




def get_charged_atoms(mol):
    return [(atom.GetIdx(), atom.GetFormalCharge()) for atom in mol.GetAtoms()]

def get_aromatic_rings(mol):
    """
    Returns:
      aromatic_rings: list[list[Atom]]
      aromatic_indices: list[tuple[int,...]]
    """
    aromatic_rings = []
    aromatic_indices = []
    ring_info = mol.GetRingInfo()

    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic_rings.append([mol.GetAtomWithIdx(idx) for idx in ring])
            aromatic_indices.append(ring)

    return aromatic_rings, aromatic_indices