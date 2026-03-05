#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for ChemEM.remodel.ligand_parser
"""

import os
import pytest
from unittest.mock import patch, MagicMock

pytest.importorskip("rdkit")
from rdkit import Chem

# Import the module to test
from ChemEM.parsers import ligand_parser
from ChemEM.parsers.ligand_parser import LigandParser, ligand_from_smiles, ligand_from_sdf, _make_ligand_from_rd_mol


# --- Fixtures ---

@pytest.fixture
def tmp_sdf_file(tmp_path):
    """Creates a temporary SDF file with one valid and one invalid molecule."""
    sdf_path = tmp_path / "test_ligands.sdf"
    
    # Write a valid molecule (ethanol)
    mol1 = Chem.MolFromSmiles("CCO")
    mol1.SetProp("_Name", "Ethanol")
    
    # Write an invalid/empty molecule
    mol2 = Chem.MolFromSmiles("INVALID") 
    
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol1)
    if mol2: # RDKit won't even write None, so we'll just test the valid one here
        writer.write(mol2)
    writer.close()
    
    return str(sdf_path)


# --- Tests for LigandParser.load_ligands ---

@patch("ChemEM.parsers.ligand_parser.ligand_from_sdf")
@patch("os.path.exists")
def test_load_ligands_routes_to_sdf(mock_exists, mock_from_sdf):
    # Setup: mock os.path.exists to return True (simulating a file path)
    mock_exists.return_value = True
    
    # Mock the return value of the SDF loader
    mock_ligand = MagicMock()
    mock_from_sdf.return_value = [mock_ligand]
    
    # Execute
    result = LigandParser.load_ligands("fake_file.sdf")
    
    # Assert
    assert result == [mock_ligand]
    mock_from_sdf.assert_called_once()
    assert mock_from_sdf.call_args[0][0] == "fake_file.sdf"

@patch("ChemEM.parsers.ligand_parser.ligand_from_smiles")
@patch("os.path.exists")
def test_load_ligands_routes_to_smiles(mock_exists, mock_from_smiles):
    # Setup: mock os.path.exists to return False (simulating a SMILES string)
    mock_exists.return_value = False
    
    # Mock the return value of the SMILES loader
    mock_ligand = MagicMock()
    mock_from_smiles.return_value = mock_ligand
    
    # Execute
    result = LigandParser.load_ligands("CCO")
    
    # Assert
    assert result == [mock_ligand]
    mock_from_smiles.assert_called_once()
    assert mock_from_smiles.call_args[0][0] == "CCO"

@patch("ChemEM.parsers.ligand_parser.ligand_from_smiles")
@patch("os.path.exists")
def test_load_ligands_raises_runtime_error_on_none(mock_exists, mock_from_smiles):
    mock_exists.return_value = False
    mock_from_smiles.return_value = None  # Simulating a failed load
    
    with pytest.raises(RuntimeError, match="Failed to load ligands from input"):
        LigandParser.load_ligands("INVALID_SMILES")

@patch("ChemEM.parsers.ligand_parser.ligand_from_smiles")
@patch("os.path.exists")
def test_load_ligands_raises_runtime_error_on_exception(mock_exists, mock_from_smiles):
    mock_exists.return_value = False
    mock_from_smiles.side_effect = Exception("Some inner parsing error")
    
    with pytest.raises(RuntimeError, match="Failed to load ligands from input"):
        LigandParser.load_ligands("CCO")


# --- Tests for ligand_from_smiles ---

@patch("ChemEM.parsers.ligand_parser._make_ligand_from_rd_mol")
@patch("ChemEM.parsers.ligand_parser.set_ligand_rings")
@patch("ChemEM.parsers.ligand_parser.check_unassigned_chirality")
@patch("ChemEM.parsers.ligand_parser.set_smiles_protonation_state")
def test_ligand_from_smiles_full_pipeline(
    mock_protonate, mock_chirality, mock_rings, mock_make_ligand
):
    # Setup mocks
    mock_protonate.return_value = ["CCO"]  # Return standard SMILES
    mock_make_ligand.return_value = "MockLigandObject"
    
    # Execute
    result = ligand_from_smiles(
        "CCO", protonation=True, chirality=True, rings=True
    )
    
    # Assertions
    assert result == "MockLigandObject"
    mock_protonate.assert_called_once()
    mock_chirality.assert_called_once()
    mock_rings.assert_called_once()
    mock_make_ligand.assert_called_once()
    
    # Verify a 3D RDKit mol with Hydrogens was passed to the final maker
    passed_mol = mock_make_ligand.call_args[0][0]
    assert passed_mol.GetNumAtoms() > 3  # CCO + Hydrogens = 9 atoms
    assert passed_mol.GetNumConformers() == 1  # 3D embedded

def test_ligand_from_smiles_invalid_smiles():
    # It should raise an Exception from RDKit which the parent function will later catch
    with pytest.raises(Exception):
        ligand_from_smiles("INVALID", protonation=False)


# --- Tests for ligand_from_sdf ---

@patch("ChemEM.parsers.ligand_parser._make_ligand_from_rd_mol")
@patch("ChemEM.parsers.ligand_parser.set_mol_protonatation_state")
def test_ligand_from_sdf_success(mock_protonate, mock_make_ligand, tmp_sdf_file):
    # Setup
    mock_protonate.return_value = None # Force fallback to Chem.AddHs
    mock_make_ligand.return_value = "MockLigandObject"
    
    # Execute
    results = ligand_from_sdf(tmp_sdf_file, protonation=True, chirality=False, rings=False)
    
    # Assertions
    assert len(results) == 1
    assert results[0] == "MockLigandObject"
    mock_protonate.assert_called_once()
    mock_make_ligand.assert_called_once()


# --- Tests for _make_ligand_from_rd_mol ---

@patch("ChemEM.parsers.ligand_parser.RingType")
@patch("ChemEM.parsers.ligand_parser.AtomType")
@patch("ChemEM.parsers.ligand_parser.Ligand")
@patch("ChemEM.parsers.ligand_parser.get_aromatic_rings")
@patch("ChemEM.parsers.ligand_parser.load_ligand_structure")
@patch("ChemEM.parsers.ligand_parser.get_charged_atoms")
def test_make_ligand_from_rd_mol(
    mock_get_charges,
    mock_load_structure,
    mock_get_rings,
    mock_ligand_class,
    mock_atom_type,
    mock_ring_type
):
    # Mocking the RDKit molecule
    rd_mol = Chem.MolFromSmiles("c1ccccc1") # Benzene
    
    # Mock the return values of our helper functions
    mock_get_charges.return_value = [(0, 0)]
    
    mock_openff_structure = MagicMock()
    mock_openff_structure.residues = [MagicMock()] # Mocking the residue list
    mock_openff_system = MagicMock()
    mock_load_structure.return_value = (mock_openff_structure, mock_openff_system)
    
    mock_get_rings.return_value = ([["mock_ring_atom"]], [[0, 1, 2, 3, 4, 5]])
    
    mock_atom_type.from_atom.return_value = "MockAtomType"
    mock_ring_type.from_ring.return_value = "MockRingType"
    mock_ligand_class.return_value = "FinalLigandInstance"
    
    # Execute
    result = _make_ligand_from_rd_mol(rd_mol, "source_id_123", name="LIG")
    
    # Assertions
    assert result == "FinalLigandInstance"
    
    # Ensure residue name was set
    assert mock_openff_structure.residues[0].name == "LIG"
    
    # Ensure the final Ligand object was instantiated with the right arguments
    mock_ligand_class.assert_called_once_with(
        "source_id_123",
        rd_mol,
        mock_openff_system,
        mock_openff_structure,
        ["MockAtomType"] * 6, # 6 heavy atoms in benzene
        ["MockRingType"],     # 1 ring
        [[0, 1, 2, 3, 4, 5]], # 1 ring index list
        [(0, 0)],             # charges
    )