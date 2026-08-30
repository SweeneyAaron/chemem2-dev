"""The `--reference-ligand` recall diagnostic must measure what it claims to.

`rmsd_to_ref` in the docking_v2 engine compares the docked pose to the reference **by
atom index** -- no superposition, no symmetry correction, no atom mapping. A reference
SDF is written in whatever order the depositor used (element-sorted, typically), while
the docked ligand is built from SMILES, so the two orders have nothing to do with each
other. Feeding the raw SDF coordinates through therefore reports a large RMSD for a
*perfect* pose: on 9DMU/GAD a pose identical to the native measures 7.70 A, which sits
inside the range of genuine results and reads as a plausible answer rather than as a
broken instrument.

`Docking._attach_reference_ligand` permutes the reference into the docked atom order so
the number means something. These tests pin that, and pin the failure mode that
motivated it.

Run with:  pytest ChemEM/tests/test_reference_ligand_diag.py     (env: chemem2-run)
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from ChemEM.protocols._docking.docking import Docking

GAD = ("Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]2O[C@@H]"
       "([n+]3cccc(NC(=O)c4cccs4)c3)[C@H](O)[C@@H]2O)[C@@H](O)[C@H]1O")


def _docked_ligand(smiles=GAD):
    """A ligand mol shaped like ChemEM's: built from SMILES, hydrogens appended."""
    mol = Chem.AddHs(Chem.MolFromSmiles(Chem.CanonSmiles(smiles)))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) == 0
    return types.SimpleNamespace(mol=mol)


def _element_sorted_copy(mol):
    """Rewrite a mol with atoms sorted by element -- how deposited SDFs usually come."""
    order = sorted(range(mol.GetNumAtoms()),
                   key=lambda i: (mol.GetAtomWithIdx(i).GetSymbol(), i))
    return Chem.RenumberAtoms(mol, order)


def _write(mol, path):
    w = Chem.SDWriter(str(path))
    w.write(mol)
    w.close()
    return str(path)


def _protocol(ref_path):
    logs = []
    system = types.SimpleNamespace(
        options=types.SimpleNamespace(reference_ligand=ref_path),
        log=logs.append,
    )
    d = Docking(system)
    return d, logs


def _index_wise_rmsd(a, b):
    """Exactly what the engine computes: no mapping, no superposition, no symmetry."""
    return float(np.sqrt(np.mean(np.sum((np.asarray(a) - np.asarray(b)) ** 2, axis=1))))


# --------------------------------------------------------------------------------------
def test_perfect_pose_measures_zero_after_mapping(tmp_path):
    """The headline property: pose == native must report ~0, whatever the SDF order."""
    ligand = _docked_ligand()
    heavy = Chem.RemoveHs(Chem.Mol(ligand.mol))
    native_xyz = np.asarray(heavy.GetConformer().GetPositions(), dtype=float)

    ref_path = _write(_element_sorted_copy(heavy), tmp_path / "native.sdf")

    combined = types.SimpleNamespace()
    d, _ = _protocol(ref_path)
    d._attach_reference_ligand(ligand, combined)

    # Tolerance is set by the SDF format, not by the mapping: coordinates are written
    # to 4 decimal places, which floors the achievable agreement at ~1e-4 A.
    assert _index_wise_rmsd(combined.reference_heavy_coords, native_xyz) < 1e-3


def test_unmapped_reference_would_have_been_badly_wrong(tmp_path):
    """Guard the premise: without mapping the same perfect pose looks like a failure."""
    ligand = _docked_ligand()
    heavy = Chem.RemoveHs(Chem.Mol(ligand.mol))
    native_xyz = np.asarray(heavy.GetConformer().GetPositions(), dtype=float)

    shuffled = _element_sorted_copy(heavy)
    raw = np.asarray(shuffled.GetConformer().GetPositions(), dtype=float)

    # This is what the old code handed to the engine.
    assert _index_wise_rmsd(raw, native_xyz) > 3.0


def test_shape_and_order_match_the_docked_heavy_atoms(tmp_path):
    """rmsd_to_ref walks the first ref.rows() atom indices, so those must be the
    docked ligand's heavy atoms, in its own order."""
    ligand = _docked_ligand()
    heavy = Chem.RemoveHs(Chem.Mol(ligand.mol))
    ref_path = _write(_element_sorted_copy(heavy), tmp_path / "native.sdf")

    combined = types.SimpleNamespace()
    d, _ = _protocol(ref_path)
    d._attach_reference_ligand(ligand, combined)

    mapped = combined.reference_heavy_coords
    assert mapped.shape == (heavy.GetNumAtoms(), 3)
    # AddHs appends hydrogens, so heavy atoms occupy 0..n_heavy-1 in ligand.mol
    assert heavy.GetNumAtoms() < ligand.mol.GetNumAtoms()
    for i in range(heavy.GetNumAtoms()):
        assert ligand.mol.GetAtomWithIdx(i).GetSymbol() != "H"


def test_a_mismatched_reference_raises_instead_of_reporting_nonsense(tmp_path):
    ligand = _docked_ligand()
    other = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    assert AllChem.EmbedMolecule(other, randomSeed=1) == 0
    ref_path = _write(Chem.RemoveHs(other), tmp_path / "wrong.sdf")

    d, _ = _protocol(ref_path)
    with pytest.raises(ValueError, match="not a substructure"):
        d._attach_reference_ligand(ligand, types.SimpleNamespace())


def test_symmetric_mappings_are_reported_as_an_upper_bound(tmp_path):
    """GAD's phosphate oxygens are automorphic; the engine gets one fixed array and
    cannot take a min over them, so the user must be told the number is a bound."""
    ligand = _docked_ligand()
    heavy = Chem.RemoveHs(Chem.Mol(ligand.mol))
    ref_path = _write(heavy, tmp_path / "native.sdf")

    d, logs = _protocol(ref_path)
    d._attach_reference_ligand(ligand, types.SimpleNamespace())

    msg = " ".join(logs)
    assert "recall diagnostic on" in msg
    n_matches = len(heavy.GetSubstructMatches(heavy, uniquify=False, maxMatches=50000))
    if n_matches > 1:
        assert "upper bound" in msg


def test_no_reference_flag_is_a_no_op():
    ligand = _docked_ligand("CCO")
    system = types.SimpleNamespace(
        options=types.SimpleNamespace(reference_ligand=None), log=lambda *_: None)
    combined = types.SimpleNamespace()
    Docking(system)._attach_reference_ligand(ligand, combined)
    assert not hasattr(combined, "reference_heavy_coords")
