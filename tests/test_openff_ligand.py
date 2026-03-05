#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for ChemEM.parsers.openff_ligand
"""

import pytest
from unittest.mock import patch, MagicMock

pytest.importorskip("rdkit")
from rdkit import Chem
from rdkit.Chem import AllChem

# Import the module to test
from ChemEM.parsers import openff_ligand

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


# --- 1. Fast Mocked Unit Test ---

@patch("ChemEM.parsers.openff_ligand.parmed")
@patch("ChemEM.parsers.openff_ligand.PDBFile")
@patch("ChemEM.parsers.openff_ligand.Interchange")
@patch("ChemEM.parsers.openff_ligand.ForceField")
@patch("ChemEM.parsers.openff_ligand.Molecule")
def test_load_ligand_structure_flow_mocked(
    mock_molecule_class,
    mock_forcefield_class,
    mock_interchange_class,
    mock_pdbfile_class,
    mock_parmed,
    mol3d
):
    """
    Tests the data flow and function calls without running actual 
    OpenFF parameterization (which is slow and requires offxml files).
    """
    # Setup our mock returns
    mock_off_mol = MagicMock()
    mock_molecule_class.from_rdkit.return_value = mock_off_mol
    
    mock_ff = MagicMock()
    mock_forcefield_class.return_value = mock_ff
    
    mock_interchange = MagicMock()
    mock_interchange_class.from_smirnoff.return_value = mock_interchange
    
    mock_system = MagicMock()
    mock_interchange.to_openmm.return_value = mock_system
    
    mock_pdb_instance = MagicMock()
    mock_pdbfile_class.return_value = mock_pdb_instance
    
    mock_parmed_structure = MagicMock()
    mock_parmed.openmm.load_topology.return_value = mock_parmed_structure

    # Generate a simple 3D molecule (Ethanol)
    rd_mol = mol3d("CCO") 
    
    # Execute the function
    struct, system = openff_ligand.load_ligand_structure(rd_mol)

    # Asserts - Did it return what we expect?
    assert struct is mock_parmed_structure
    assert system is mock_system
    
    # Verify OpenFF Molecule creation and charge assignment
    mock_molecule_class.from_rdkit.assert_called_once_with(rd_mol, allow_undefined_stereo=True)
    mock_off_mol.assign_partial_charges.assert_called_once_with("mmff94")
    
    # Verify ForceField setup (checking that the specific version you requested is used)
    mock_forcefield_class.assert_called_once_with("openff_unconstrained-2.0.0.offxml")
    
    # Verify Interchange setup
    mock_interchange_class.from_smirnoff.assert_called_once_with(
        topology=[mock_off_mol],
        force_field=mock_ff,
        charge_from_molecules=[mock_off_mol],
    )
    
    # Verify ParmEd structure loading
    mock_parmed.openmm.load_topology.assert_called_once_with(
        mock_pdb_instance.topology,
        mock_system,
        xyz=mock_pdb_instance.positions
    )


# --- 2. Full Integration Test (Skipped if OpenFF/ParmEd missing) ---

@pytest.mark.slow
def test_load_ligand_structure_integration(mol3d):
    """
    Actually attempts to run OpenFF parameterization.
    This test will automatically skip if the required libraries are not installed.
    """
    # Skip if heavy dependencies aren't present
    pytest.importorskip("openff.toolkit")
    pytest.importorskip("openff.interchange")
    pytest.importorskip("parmed")
    
    # Create a small, valid 3D molecule (Methanol)
    rd_mol = mol3d("CO") 
    
    try:
        struct, system = openff_ligand.load_ligand_structure(rd_mol)
    except Exception as e:
        pytest.fail(f"Integration test failed with error: {e}")
    
    # Check that we got realistic objects back
    assert struct is not None
    assert system is not None
    
    # Verify the OpenMM system has the expected number of particles
    # Methanol (CO) has 1 Carbon, 1 Oxygen, 4 Hydrogens = 6 atoms/particles
    assert system.getNumParticles() == 6
    
    # Verify ParmEd structure correctly interpreted the atoms
    assert len(struct.atoms) == 6