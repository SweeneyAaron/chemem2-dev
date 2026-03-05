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
    
    match = new_mol.GetSubstructMatch(ref_mol)
    if not match:
        return None
    
    #if mol.GetNumConformers()
    
    ref_mol_conformer = ref_mol.GetConformer()
    new_mol_conformer = new_mol.GetConformer()
    
    # Map heavy atoms: mol_noH atom i corresponds to prot_noH atom match[i]
    ref_heavy = [a.GetIdx() for a in ref_mol.GetAtoms() if a.GetSymbol() != "H"]
    new_heavy = [a.GetIdx() for a in new_mol.GetAtoms() if a.GetSymbol() != "H"]

    for ref_mol_idx, new_mol_idx in enumerate(match):
        ref_atom_idx = ref_heavy[ref_mol_idx]
        new_atom_idx = new_heavy[new_mol_idx]
        new_mol_conformer.SetAtomPosition(new_mol_idx, ref_mol_conformer.GetAtomPosition(ref_atom_idx))
    
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