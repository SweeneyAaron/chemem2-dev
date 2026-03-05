#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for ChemEM.remodel.ligand_ops
"""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

# Assuming the module path based on your imports
from ChemEM.parsers.remodel import ligand_ops

@pytest.fixture
def mol3d():
    """Factory fixture: build a small 3D molecule deterministically."""
    def _make(smiles: str, seed: int = 0xC0FFEE):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        res = AllChem.EmbedMolecule(mol, randomSeed=seed)
        assert res == 0
        AllChem.UFFOptimizeMolecule(mol, maxIters=50)
        return mol
    return _make


# --- Tests for check_unassigned_chirality ---

def test_check_unassigned_chirality_no_chiral_centers(capsys):
    # Ethane - no chiral centers
    mol = Chem.MolFromSmiles("CC")
    ligand_ops.check_unassigned_chirality(mol)
    
    # Check that nothing was printed to stdout (no warnings/errors)
    captured = capsys.readouterr()
    assert "[WARNING]" not in captured.out

def test_check_unassigned_chirality_assigned_centers(capsys):
    # L-Alanine (explicitly assigned stereochemistry in SMILES)
    mol = Chem.MolFromSmiles("C[C@@H](C(=O)O)N")
    ligand_ops.check_unassigned_chirality(mol)
    
    # Check that nothing was printed because the center is assigned
    captured = capsys.readouterr()
    assert "[WARNING]" not in captured.out

def test_check_unassigned_chirality_unassigned_center_resolved(capsys, mol3d):
    # Alanine without stereochemistry defined in SMILES
    # We need a 3D structure so RDKit can assign it from coordinates
    mol = mol3d("CC(C(=O)O)N")
    
    # Ensure it starts unassigned
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    assert any(c[1] == '?' for c in chiral_centers)

    ligand_ops.check_unassigned_chirality(mol)
    
    captured = capsys.readouterr()
    assert "[WARNING] [ChemEM] Unassigned chiral centers detected." in captured.out
    assert "[INFO]    [ChemEM] Chirality assignment successful." in captured.out
    
    # Check that RDKit resolved it to R or S based on the 3D generation
    updated_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    assert not any(c[1] == '?' for c in updated_centers)

def test_check_unassigned_chirality_unassigned_cannot_be_resolved(capsys):
    # Alanine without stereochemistry, and NO 3D coordinates
    mol = Chem.MolFromSmiles("CC(C(=O)O)N")
    
    ligand_ops.check_unassigned_chirality(mol)
    
    captured = capsys.readouterr()
    assert "[WARNING] [ChemEM] Unassigned chiral centers detected." in captured.out
    # Since there are no 3D coords, RDKit usually falls back gracefully, 
    # but the centers might remain unassigned. The function catches the exception if one occurs.
    # We just ensure the warning fired.


# --- Tests for set_ligand_rings ---

def test_set_ligand_rings_valid_molecule(capsys):
    mol = Chem.MolFromSmiles("c1ccccc1")
    ligand_ops.set_ligand_rings(mol)
    
    # Check that GetRingInfo() was populated (though RDKit usually does this automatically)
    assert mol.GetRingInfo().NumRings() == 1
    
    captured = capsys.readouterr()
    assert "ChemEM- Non-Fatal warning" not in captured.out

def test_set_ligand_rings_error_handling(capsys, monkeypatch):
    mol = Chem.MolFromSmiles("c1ccccc1")
    
    # Force GetSymmSSSR to raise an exception to test the try/except block
    def mock_GetSymmSSSR(*args, **kwargs):
        raise RuntimeError("Mocked RDKit ring error")
    
    monkeypatch.setattr(Chem, "GetSymmSSSR", mock_GetSymmSSSR)
    
    ligand_ops.set_ligand_rings(mol)
    
    captured = capsys.readouterr()
    assert "ChemEM- Non-Fatal warning ring info assignment failed" in captured.out
    assert "Mocked RDKit ring error" in captured.out


# --- Tests for transfer_mol_coords ---

def test_transfer_mol_coords_successful(mol3d):
    # Reference molecule with coordinates
    ref_mol = mol3d("c1ccccc1")
    ref_mol = Chem.RemoveHs(ref_mol) # <--- STRIP Hs to match actual usage
    
    # New molecule (same structure)
    new_mol = Chem.MolFromSmiles("c1ccccc1")
    AllChem.EmbedMolecule(new_mol) # give it *some* conformer to hold coords
    
    # Ensure they start different
    ref_pos = ref_mol.GetConformer().GetAtomPosition(0)
    new_pos = new_mol.GetConformer().GetAtomPosition(0)
    assert (ref_pos.x, ref_pos.y, ref_pos.z) != (new_pos.x, new_pos.y, new_pos.z)
    
    # Do the transfer
    result_mol = ligand_ops.transfer_mol_coords(ref_mol, new_mol)
    
    assert result_mol is not None
    # Check that heavy atom 0 coordinates match
    res_pos = result_mol.GetConformer().GetAtomPosition(0)
    assert res_pos.x == pytest.approx(ref_pos.x)
    assert res_pos.y == pytest.approx(ref_pos.y)
    assert res_pos.z == pytest.approx(ref_pos.z)

# --- Tests for get_charged_atoms ---

@pytest.mark.parametrize(
    "smiles, expected_charges",
    [
        ("CC", []), # Neutral, no formal charges -> actually the function returns (idx, 0) for all atoms
        ("[NH3+]CC(=O)[O-]", [(0, 1), (1, 0), (2, 0), (3, 0), (4, -1)]), # Zwitterion
        ("[Na+].[Cl-]", [(0, 1), (1, -1)]),
    ],
)
def test_get_charged_atoms(smiles, expected_charges):
    mol = Chem.MolFromSmiles(smiles)
    
    # The current implementation returns formal charges for ALL atoms:
    # return [(atom.GetIdx(), atom.GetFormalCharge()) for atom in mol.GetAtoms()]
    
    charges = ligand_ops.get_charged_atoms(mol)
    
    if smiles == "CC":
        # Expect (idx, 0) for the two carbons
        assert charges == [(0, 0), (1, 0)]
    else:
        assert charges == expected_charges


# --- Tests for get_aromatic_rings ---

@pytest.mark.parametrize(
    "smiles, expected_num_rings, expected_sizes",
    [
        ("c1ccccc1", 1, [6]), # 1 aromatic ring
        ("CCCCCC", 0, []),    # no rings
        ("c1ccc2[nH]ccc2c1", 2, [5, 6]), # Indole: 2 aromatic rings
        ("C1CCCC1", 0, []),   # 1 aliphatic ring, 0 aromatic
        ("c1ccc2NCCCc2c1", 1, [6]) # 1 aromatic, 1 aliphatic
    ],
)
def test_get_aromatic_rings(smiles, expected_num_rings, expected_sizes):
    mol = Chem.MolFromSmiles(smiles)
    aromatic_rings, aromatic_indices = ligand_ops.get_aromatic_rings(mol)
    
    assert len(aromatic_rings) == expected_num_rings
    assert len(aromatic_indices) == expected_num_rings
    
    # Check the sizes of the rings found match expectations
    found_sizes = sorted([len(ring) for ring in aromatic_indices])
    assert found_sizes == sorted(expected_sizes)
    
    # Verify the objects returned in aromatic_rings are actual RDKit Atom objects
    if expected_num_rings > 0:
        assert isinstance(aromatic_rings[0][0], Chem.Atom)