#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for ChemEM.remodel.protonation
"""

import pytest
from unittest.mock import patch, MagicMock

pytest.importorskip("rdkit")
from rdkit import Chem
from rdkit.Chem import AllChem

# Import the module to test
from ChemEM.parsers.remodel import protonation

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


# --- Tests for OpenMM Modeller functions ---

def test_delete_all_hydrogens():
    # Mock the OpenMM modeller and topology
    mock_modeller = MagicMock()
    
    # Create mock atoms
    atom_c = MagicMock()
    atom_c.element = MagicMock() # Some non-hydrogen element
    
    atom_h1 = MagicMock()
    atom_h1.element = protonation.omm_element.hydrogen
    
    atom_h2 = MagicMock()
    atom_h2.element = protonation.omm_element.hydrogen

    # Set up the topology to return our mock atoms
    mock_modeller.topology.atoms.return_value = [atom_c, atom_h1, atom_h2]

    # Run the function
    deleted_count = protonation.delete_all_hydrogens(mock_modeller)

    # Assertions
    assert deleted_count == 2
    mock_modeller.delete.assert_called_once_with([atom_h1, atom_h2])

def test_delete_all_hydrogens_none_present():
    mock_modeller = MagicMock()
    
    atom_c = MagicMock()
    atom_c.element = MagicMock() # Non-hydrogen
    
    mock_modeller.topology.atoms.return_value = [atom_c]

    deleted_count = protonation.delete_all_hydrogens(mock_modeller)

    assert deleted_count == 0
    mock_modeller.delete.assert_not_called()

def test_add_hydrogens():
    mock_modeller = MagicMock()
    mock_forcefield = MagicMock()
    
    protonation.add_hydrogens(mock_modeller, mock_forcefield, pH=8.0)
    
    mock_modeller.addHydrogens.assert_called_once_with(mock_forcefield, pH=8.0)


# --- Tests for set_smiles_protonation_state (Argument Parsing Logic) ---

@patch("ChemEM.parsers.remodel.protonation.protonate_smiles")
def test_set_smiles_protonation_state_ph_float(mock_protonate):
    mock_protonate.return_value = ["mock_smiles"]
    
    result = protonation.set_smiles_protonation_state("CCO", pH=7.4, pka_prec=1.0)
    
    assert result == ["mock_smiles"]
    mock_protonate.assert_called_once_with(
        "CCO", ph_min=7.4, ph_max=7.4, max_variants=128, label_states=False, precision=1.0
    )

@patch("ChemEM.parsers.remodel.protonation.protonate_smiles")
def test_set_smiles_protonation_state_ph_list(mock_protonate):
    mock_protonate.return_value = ["mock_smiles"]
    
    # Passing a list for pH, unsorted
    result = protonation.set_smiles_protonation_state("CCO", pH=[8.0, 6.0])
    
    assert result == ["mock_smiles"]
    # Function should sort the list to [6.0, 8.0]
    mock_protonate.assert_called_once_with(
        "CCO", ph_min=6.0, ph_max=8.0, max_variants=128, label_states=False, precision=1.0
    )

def test_set_smiles_protonation_state_invalid_list_length():
    with pytest.raises(ValueError, match="pH as list must be of lenght 2"):
        protonation.set_smiles_protonation_state("CCO", pH=[7.0])
        
    with pytest.raises(ValueError, match="pH as list must be of lenght 2"):
        protonation.set_smiles_protonation_state("CCO", pH=[6.0, 7.0, 8.0])

def test_set_smiles_protonation_state_invalid_type():
    with pytest.raises(ValueError, match="pH must be list"):
        protonation.set_smiles_protonation_state("CCO", pH="7.4")


# --- Tests for set_mol_protonatation_state ---

@patch("ChemEM.parsers.remodel.protonation.set_smiles_protonation_state")
def test_set_mol_protonatation_state_success(mock_set_smiles, mol3d):
    # Neutral methylamine
    mol = mol3d("CN") 
    
    # Mock dimorphite to return the protonated form
    mock_set_smiles.return_value = ["[NH3+]C"]
    
    prot_mol = protonation.set_mol_protonatation_state(mol)
    
    assert prot_mol is not None
    # Original C(1) + N(1) + H(5) = 7. Protonated C(1) + N(1) + H(6) = 8
    assert prot_mol.GetNumAtoms() == 8 
    
    # Ensure RDKit recognized the charge
    charges = [a.GetFormalCharge() for a in prot_mol.GetAtoms()]
    assert sum(charges) == 1
    
    # Check that it has 3D coordinates (from the coordinate transfer)
    assert prot_mol.GetNumConformers() == 1

@patch("ChemEM.parsers.remodel.protonation.set_smiles_protonation_state")
def test_set_mol_protonatation_state_fallback_none_smiles(mock_set_smiles, mol3d, capsys):
    mol = mol3d("CN")
    
    # Force the SMILES protonator to fail/return None
    mock_set_smiles.return_value = None
    
    prot_mol = protonation.set_mol_protonatation_state(mol)
    
    # Should fall back to simply adding hydrogens to the original molecule
    assert prot_mol is not None
    assert prot_mol.GetNumAtoms() == 7  # Neutral methylamine
    
    captured = capsys.readouterr()
    assert "Can't protonate smiles" in captured.out

@patch("ChemEM.parsers.remodel.protonation.set_smiles_protonation_state")
def test_set_mol_protonatation_state_fallback_invalid_smiles(mock_set_smiles, mol3d, capsys):
    mol = mol3d("CN")
    
    # Mock dimorphite returning gibberish that RDKit can't parse
    mock_set_smiles.return_value = ["INVALID_SMILES_STRING"]
    
    prot_mol = protonation.set_mol_protonatation_state(mol)
    
    # RDKit will fail to make a mol from "INVALID_SMILES_STRING", triggering fallback
    assert prot_mol is not None
    assert prot_mol.GetNumAtoms() == 7 
    
    captured = capsys.readouterr()
    assert "Can't protonate smiles" in captured.out