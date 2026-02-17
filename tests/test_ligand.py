#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 23:26:28 2026

@author: aaron.sweeney
"""

import numpy as np
import pytest
pytest.importorskip("rdkit")
from rdkit import Chem
from rdkit.Chem import AllChem
from ChemEM.tools import ligand


@pytest.mark.parametrize(
    "smiles,num_rings,size_rings",
        [("c1ccccc1", 1, [6]),#1 aromatic ring
         ("CCCCCC", 0,[]), #no rings
         ("c1ccc2[nH]ccc2c1", 2, [5,6]), #2 aromatic rings
         ("c1ccc2NCCCc2c1", 1, [6]) #1 aromatic 1 aliphatic ring
        ],
    )
def test_get_aromatic_rings(smiles, num_rings, size_rings):
    
    mol = Chem.MolFromSmiles(smiles)
    aromatic_rings, aromatic_indices = ligand.get_aromatic_rings(mol)
    assert len(aromatic_indices) == len(aromatic_rings) == num_rings
    assert sorted(size_rings) == sorted([len(i) for i in aromatic_indices])
        
@pytest.mark.parametrize(
    "smiles,n_tor,check_tors",
    [
        ("CC(C)NC[C@H](O)c1ccc(O)c(O)c1", 11, True),          # normal
        ("CC(C)[NH2+]C[C@H](O)c1ccc([O-])c([O-])c1", 7, True),# protonated
        ("c1ccccc1", 0, False),                               # no torsions
    ],
)
def test_get_torsion_lists(mol3d, smiles, n_tor, check_tors):
    m = mol3d(smiles)
    torsions_lists = ligand.get_torsion_lists(m)

    assert len(torsions_lists) == n_tor
    if check_tors:
        assert all(len(t) == 4 for t in torsions_lists)

@pytest.mark.parametrize("elem,rad", [('O', 1.55), ('N', 1.6), ('C', 1.7), ('I', 2.1)])
def test_get_van_der_waals_radius_known_elements(elem, rad):
    assert ligand.get_van_der_waals_radius(elem) == rad

@pytest.mark.parametrize("elem", ["Xx", "Q", "", "12"])
def test_get_van_der_waals_radius_unknown_element_raises(elem):
    with pytest.raises(RuntimeError):
        ligand.get_van_der_waals_radius(elem)
    
def test_get_ligand_heavy_atom_indexes_excludes_h_and_returns_indices():
    # Ethanol with explicit Hs so we can really test exclusion
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)

    idxs = ligand.get_ligand_heavy_atom_indexes(mol)

    # Heavy atoms in CCO are 3: C, C, O
    assert isinstance(idxs, np.ndarray)
    assert idxs.shape == (3,)
    assert all(mol.GetAtomWithIdx(int(i)).GetSymbol() != "H" for i in idxs)

    # Should match RDKit's idea of heavy atom indices (non-H)
    expected = np.array([a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"])
    assert np.array_equal(idxs, expected)

def test_compute_bond_distances_linear_chain():
    # butane: 4 heavy atoms in a chain: 0-1-2-3
    mol = Chem.MolFromSmiles("CCCC")
    heavy = ligand.get_ligand_heavy_atom_indexes(mol)

    d = ligand.compute_bond_distances(mol, heavy)

    expected = np.array(
        [
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ],
        dtype=np.int32,
    )
    assert d.dtype == np.int32
    assert d.shape == (4, 4)
    assert np.array_equal(d, expected)


def test_compute_bond_distances_ring():
    # cyclohexane ring: distances wrap around
    mol = Chem.MolFromSmiles("C1CCCCC1")
    heavy = ligand.get_ligand_heavy_atom_indexes(mol)

    d = ligand.compute_bond_distances(mol, heavy)

    expected = np.array(
        [
            [0, 1, 2, 3, 2, 1],
            [1, 0, 1, 2, 3, 2],
            [2, 1, 0, 1, 2, 3],
            [3, 2, 1, 0, 1, 2],
            [2, 3, 2, 1, 0, 1],
            [1, 2, 3, 2, 1, 0],
        ],
        dtype=np.int32,
    )
    assert d.shape == (6, 6)
    assert np.array_equal(d, expected)

def test_compute_bond_distances_disconnected_heavy_atoms_are_999():
    # Two disconnected fragments; heavy-heavy distances across fragments should stay 999
    mol = Chem.MolFromSmiles("CC.CC")  # ethane + ethane
    heavy = ligand.get_ligand_heavy_atom_indexes(mol)

    d = ligand.compute_bond_distances(mol, heavy)

    # heavy atom indices will be [0,1,2,3] with bonds only (0-1) and (2-3)
    assert d.shape == (4, 4)

    # within fragments:
    assert d[0, 1] == 1 and d[1, 0] == 1
    assert d[2, 3] == 1 and d[3, 2] == 1

    # across fragments should remain the sentinel 999
    for i in (0, 1):
        for j in (2, 3):
            assert d[i, j] == 999
            assert d[j, i] == 999


@pytest.mark.parametrize(
    "smiles, expected_counts_sorted",
    [
        ("CC",           [3, 3]),                 # ethane: CH3-CH3
        ("c1ccccc1",     [1, 1, 1, 1, 1, 1]),     # benzene: 6 x CH
        ("CC(C)(C)O",    [0, 1, 3, 3, 3]),        # tert-butanol: central C=0H, O=1H, three CH3
    ],
)
def test_get_ligand_hydrogen_reference_counts(smiles, expected_counts_sorted):
    mol = Chem.MolFromSmiles(smiles)
    refs = ligand.get_ligand_hydrogen_reference(mol)

    assert isinstance(refs, list)
    assert all(isinstance(r, np.ndarray) for r in refs)
    assert all(r.dtype == np.int32 for r in refs)

    counts = sorted(len(r) for r in refs)  # order-independent
    assert counts == sorted(expected_counts_sorted)

@pytest.mark.parametrize(
    "smiles",
    ["CC", "c1ccccc1", "CC(C)(C)O", "CCO"],
)
def test_get_ligand_hydrogen_reference_indices_are_hydrogens(smiles):
    mol = Chem.MolFromSmiles(smiles)
    molH = Chem.AddHs(mol, addCoords=True)

    refs = ligand.get_ligand_hydrogen_reference(mol)

    for arr in refs:
        for idx in arr.tolist():
            assert 0 <= idx < molH.GetNumAtoms()
            assert molH.GetAtomWithIdx(int(idx)).GetSymbol() == "H"


def _expected_best_bonds(mol):
    """Spec-style oracle mirroring the documented rules (not hardcoding bond indices)."""
    ring_info = mol.GetRingInfo()
    expected = []

    for ring_bonds, ring_atoms in zip(ring_info.BondRings(), ring_info.AtomRings()):
        # ignore tiny rings
        if len(ring_atoms) < 4:
            continue

        # skip fully aromatic rings
        if all(mol.GetBondWithIdx(idx).GetIsAromatic() for idx in ring_bonds):
            continue

        # skip rings containing any double bond (your code treats these as "kekulized aromatic")
        if any(mol.GetBondWithIdx(idx).GetBondType() == Chem.BondType.DOUBLE for idx in ring_bonds):
            continue

        # candidates: single bonds, and bond must belong to exactly one ring (not shared/fused)
        candidates = []
        for bidx in ring_bonds:
            b = mol.GetBondWithIdx(bidx)
            if b.GetBondType() != Chem.BondType.SINGLE:
                continue
            if ring_info.NumBondRings(bidx) != 1:
                continue
            candidates.append(bidx)

        if not candidates:
            continue

        # prefer C-C bonds
        cc = []
        for bidx in candidates:
            b = mol.GetBondWithIdx(bidx)
            if b.GetBeginAtom().GetAtomicNum() == 6 and b.GetEndAtom().GetAtomicNum() == 6:
                cc.append(bidx)

        final = cc if cc else candidates
        expected.append(min(final))  # deterministic

    return sorted(set(expected))


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",            # benzene: aromatic -> []
        "C1CC1",               # cyclopropane: <4 atoms -> []
        "C1=CCCCC1",           # cyclohexene: has double -> []
        "c1ccc2c(c1)C=CC2",    # indene: 5-ring has double -> []
    ],
)
def test_find_best_ring_bond_to_break_returns_empty_for_skipped_rings(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert ligand.find_best_ring_bond_to_break(mol) == []


@pytest.mark.parametrize(
    "smiles, expected_n",
    [
        ("C1CCCCC1", 1),            # cyclohexane: one non-aromatic ring
        ("C1CCOC1", 1),             # THF: one non-aromatic ring, should prefer a C-C bond
        ("C1CCC2CCCCC2C1", 2),      # decalin: two rings, should give one bond per ring
    ],
)
def test_find_best_ring_bond_to_break_matches_rule_based_expectation(smiles, expected_n):
    mol = Chem.MolFromSmiles(smiles)

    out = ligand.find_best_ring_bond_to_break(mol)
    exp = _expected_best_bonds(mol)

    assert out == exp
    assert len(out) == expected_n

    # Invariants on returned bonds
    ring_info = mol.GetRingInfo()
    all_ring_bonds = {b for ring in ring_info.BondRings() for b in ring}

    for bidx in out:
        b = mol.GetBondWithIdx(bidx)
        assert bidx in all_ring_bonds
        assert b.GetBondType() == Chem.BondType.SINGLE
        assert not b.GetIsAromatic()
        assert ring_info.NumBondRings(bidx) == 1  # not a shared/fused bond


def test_find_best_ring_bond_to_break_prefers_cc_when_available():
    # THF has both C-C and C-O single bonds in the ring; should pick a C-C bond
    mol = Chem.MolFromSmiles("C1CCOC1")
    out = ligand.find_best_ring_bond_to_break(mol)
    assert len(out) == 1

    b = mol.GetBondWithIdx(out[0])
    assert b.GetBeginAtom().GetAtomicNum() == 6 and b.GetEndAtom().GetAtomicNum() == 6





def test_remove_bonds_from_mol_ring_opening_returns_mol_and_constraints(mol3d):
    # cyclohexane: breaking one ring bond should not fragment, just open the ring
    mol = mol3d("C1CCCCC1")

    # pick a bond index that exists
    bond_idx = 0
    assert mol.GetNumBonds() > bond_idx

    out_mol, atoms_to_constrain, distances = ligand.remove_bonds_from_mol(mol, [bond_idx])

    assert out_mol is not None
    assert atoms_to_constrain is not None and distances is not None
    assert len(atoms_to_constrain) == 1
    assert len(distances) == 1

    i, j = atoms_to_constrain[0]
    assert isinstance(i, int) and isinstance(j, int)
    assert isinstance(distances[0], float)

    # should remain a single fragment after removing one bond in a ring
    assert len(Chem.GetMolFrags(out_mol)) == 1

    # bond count decreased by 1
    assert out_mol.GetNumBonds() == mol.GetNumBonds() - 1


def test_remove_bonds_from_mol_chain_break_fragments_returns_none(mol3d):
    # ethane: breaking the only heavy-heavy bond fragments into 2 pieces -> should return None triplet
    mol = mol3d("CC")

    # find the C-C bond index (ignore H bonds)
    cc_bond_idx = None
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetSymbol() == "C" and a2.GetSymbol() == "C":
            cc_bond_idx = b.GetIdx()
            break
    assert cc_bond_idx is not None

    out_mol, atoms_to_constrain, distances = ligand.remove_bonds_from_mol(mol, [cc_bond_idx])

    assert out_mol is None
    assert atoms_to_constrain is None
    assert distances is None


def test_remove_bonds_from_mol_distance_matches_conformer(mol3d):
    # Check the computed constraint distance equals the actual conformer distance (rounded to 3 dp)
    mol = mol3d("C1CCCCC1")

    bond_idx = 0
    bond = mol.GetBondWithIdx(bond_idx)
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

    conf = mol.GetConformer()
    expected = round(conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j)), 3)

    out_mol, atoms_to_constrain, distances = ligand.remove_bonds_from_mol(mol, [bond_idx])

    assert out_mol is not None
    assert atoms_to_constrain == [(i, j)] or atoms_to_constrain == [(j, i)]  # order should match idx1/idx2
    assert distances == [expected]

