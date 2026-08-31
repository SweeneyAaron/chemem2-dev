#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Protein hydrogen coordinates: where they come from, and rotating them.

Two things are under test, both feeding the same consumer -- ECHO's H-bond term
reads protein hydrogen coordinates directly and gates on the D-H...A angle
(ScoringFunctions.cpp:312-344), so where those coordinates come from and whether
they are allowed to rotate both change the score:

  * ``write_residues_to_pdb(return_hydrogens=True)`` -- carrying the *prepared*
    hydrogen placement past the ``RemoveHs`` that builds the heavy-atom site mol,
    instead of regenerating it with ``Chem.AddHs``.
  * ``protein_hydrogen_torsions`` -- which donors may rotate, and the geometry of
    rotating them.

The index alignment between the two is load-bearing and easy to break silently:
``protein_hydrogens[i]`` is indexed by the same ``i`` as ``protein_positions`` in
the C++ (PreComputedData.cpp:148), so a mismatch does not raise, it just scores
the wrong hydrogens against the wrong atoms.
"""

import numpy as np
import pytest

from rdkit import Chem
from rdkit.Chem import AllChem

from ChemEM.tools.biomolecule import write_residues_to_pdb
from ChemEM.protocols.score.protein_hydrogen_torsions import (
    ProteinHydrogenRelaxer,
    _rotate_about,
    _wrap180,
    donors_near_ligand,
    min_ligand_distance,
    rotatable_protein_donors,
)


# --------------------------------------------------------------------------------------
# minimal ParmEd-shaped fakes, matching test_manual_binding_site's
# --------------------------------------------------------------------------------------
class _Pos:
    def __init__(self, xyz):
        self.x, self.y, self.z = (float(v) for v in xyz)


class _Res:
    def __init__(self, name, number):
        self.name = name
        self.number = number
        self.idx = number - 1
        self.chain = "A"
        self.atoms = []


class _Atom:
    def __init__(self, idx, name, element_name, atomic_number, residue):
        self.idx = idx
        self.name = name
        self.element_name = element_name
        self.element = atomic_number
        self.atomic_number = atomic_number
        self.bond_partners = []
        self.residue = residue
        residue.atoms.append(self)


class _Structure:
    def __init__(self, positions):
        self.positions = [_Pos(p) for p in positions]


def _serine_residue(with_hydrogens=True):
    """A SER-shaped residue: CA-CB-OG, with HG on OG when protonated.

    Only the CB-OG-HG end matters -- that is the rotatable donor. Coordinates are
    made up but geometrically sane (roughly tetrahedral at OG).
    """
    res = _Res("SER", 1)
    spec = [
        ("CA", "C", 6, (0.000, 0.000, 0.000)),
        ("CB", "C", 6, (1.520, 0.000, 0.000)),
        ("OG", "O", 8, (2.100, 1.280, 0.000)),
    ]
    if with_hydrogens:
        spec.append(("HG", "H", 1, (3.060, 1.230, 0.000)))

    atoms = [_Atom(i, n, e, z, res) for i, (n, e, z, _) in enumerate(spec)]
    coords = np.asarray([xyz for _, _, _, xyz in spec], dtype=float)

    def bond(i, j):
        atoms[i].bond_partners.append(atoms[j])
        atoms[j].bond_partners.append(atoms[i])

    bond(0, 1)
    bond(1, 2)
    if with_hydrogens:
        bond(2, 3)

    return res, coords


# --------------------------------------------------------------------------------------
# write_residues_to_pdb: carrying the prepared hydrogens through
# --------------------------------------------------------------------------------------
def test_return_hydrogens_is_opt_in():
    """Default stays a bare mol; six call sites depend on that."""
    res, coords = _serine_residue()
    out = write_residues_to_pdb([res], _Structure(coords).positions)
    assert isinstance(out, Chem.Mol)


def test_prepared_hydrogens_align_with_the_heavy_mol():
    res, coords = _serine_residue()
    mol, hydrogens = write_residues_to_pdb(
        [res], _Structure(coords).positions, return_hydrogens=True
    )

    # The alignment the C++ assumes: one entry per heavy atom, same order.
    assert mol.GetNumAtoms() == 3
    assert len(hydrogens) == mol.GetNumAtoms()
    assert [a.GetSymbol() for a in mol.GetAtoms()] == ["C", "C", "O"]

    # Only OG carries a hydrogen, and it is the one the structure placed --
    # not something AddHs invented.
    assert [len(h) for h in hydrogens] == [0, 0, 1]
    assert np.allclose(hydrogens[2][0], coords[3])


def test_unprotonated_structure_reports_none_rather_than_empty_lists():
    """None means "cannot know"; empty lists would mean "no hydrogens here".

    PreCompDataProtein has to tell those apart to decide whether falling back to
    Chem.AddHs is required or would be discarding real information.
    """
    res, coords = _serine_residue(with_hydrogens=False)
    mol, hydrogens = write_residues_to_pdb(
        [res], _Structure(coords).positions, return_hydrogens=True
    )
    assert mol.GetNumAtoms() == 3
    assert hydrogens is None


def test_prepared_hydrogens_differ_from_the_rdkit_default():
    """The whole point: AddHs does not reproduce the placement it replaces.

    If these ever agreed, --protein-hydrogens would be a no-op and the rotatable
    donors would already be where preparation put them.
    """
    from ChemEM.tools.biomolecule import get_protein_hydrogen_reference

    res, coords = _serine_residue()
    mol, prepared = write_residues_to_pdb(
        [res], _Structure(coords).positions, return_hydrogens=True
    )
    rdkit_placed = get_protein_hydrogen_reference(mol)

    assert len(rdkit_placed) == len(prepared)
    assert len(rdkit_placed[2]) == 1
    assert not np.allclose(rdkit_placed[2][0], prepared[2][0], atol=1e-3)


# --------------------------------------------------------------------------------------
# donor classification
# --------------------------------------------------------------------------------------
def _molecule(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _as_site(mol):
    """Split an all-atom mol into the (heavy mol, per-heavy-atom H coords) pair.

    That is the shape PreCompDataProtein hands the scorer: the mol is heavy-atom
    only, so hydrogen *counts* have to come from the coordinate lists.
    """
    pos = mol.GetConformer().GetPositions()
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]

    rw = Chem.RWMol(mol)
    for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1),
                      reverse=True):
        rw.RemoveAtom(idx)
    heavy_mol = rw.GetMol()
    Chem.SanitizeMol(heavy_mol)

    conf = Chem.Conformer(heavy_mol.GetNumAtoms())
    for new_idx, old_idx in enumerate(heavy):
        conf.SetAtomPosition(new_idx, pos[old_idx].tolist())
    heavy_mol.RemoveAllConformers()
    heavy_mol.AddConformer(conf, assignId=True)

    for atom in heavy_mol.GetAtoms():
        atom.SetProp("resName", "XXX")
        atom.SetProp("resId", "1")
        atom.SetProp("atomName", f"{atom.GetSymbol()}{atom.GetIdx()}")

    hydrogens = [[pos[n.GetIdx()] for n in mol.GetAtomWithIdx(h).GetNeighbors()
                  if n.GetAtomicNum() == 1] for h in heavy]
    return heavy_mol, hydrogens


@pytest.mark.parametrize("smiles,n_donors,kind", [
    ("CCO",           1, "free"),      # Ser/Thr hydroxyl
    ("c1ccc(O)cc1",   1, "planar"),    # Tyr: conjugated to the ring
    ("CCS",           1, "free"),      # Cys thiol
    ("CC[NH3+]",      1, "free"),      # Lys ammonium
    ("CC(N)=O",       0, None),        # Asn/Gln amide: planar, H fixed by chemistry
    ("NC(=[NH2+])N",  0, None),        # Arg guanidinium: same
    ("CC(=O)[O-]",    0, None),        # carboxylate: no hydrogen to turn
    ("COC",           0, None),        # ether: two heavy neighbours, not a rotation
    ("O",             0, None),        # water: no heavy neighbour to rotate about
])
def test_rotatable_donor_classification(smiles, n_donors, kind):
    donors = rotatable_protein_donors(*_as_site(_molecule(smiles)))
    assert len(donors) == n_donors
    if kind is not None:
        assert ["planar" if d.allowed is not None else "free"
                for d in donors] == [kind]


def test_classification_needs_the_hydrogen_lists():
    """No hydrogen coordinates means no donors -- a heavy-only mol cannot say."""
    heavy_mol, hydrogens = _as_site(_molecule("CCO"))
    assert rotatable_protein_donors(heavy_mol, None) == []
    assert rotatable_protein_donors(heavy_mol, [[] for _ in hydrogens]) == []


# --------------------------------------------------------------------------------------
# rotation geometry
# --------------------------------------------------------------------------------------
def test_rotation_is_rigid_about_the_bond():
    donor = rotatable_protein_donors(*_as_site(_molecule("CCO")))[0]
    ref = donor.h_ref[0]
    r0 = np.linalg.norm(ref - donor.origin)
    cos0 = (ref - donor.origin) @ donor.axis / r0

    for angle in (37.0, 90.0, 180.0, -120.0):
        moved = np.asarray(donor.rotated(angle)[0])
        r1 = np.linalg.norm(moved - donor.origin)
        # Bond length and the angle to the rotation axis are both invariants;
        # together they say only the torsion changed.
        assert r1 == pytest.approx(r0, abs=1e-9)
        assert (moved - donor.origin) @ donor.axis / r1 == pytest.approx(cos0, abs=1e-9)
        assert np.linalg.norm(moved - ref) > 1e-3


def test_zero_rotation_is_the_identity():
    donor = rotatable_protein_donors(*_as_site(_molecule("CCO")))[0]
    assert np.allclose(np.asarray(donor.rotated(0.0)), donor.h_ref)


def test_conjugated_donor_is_restricted_to_the_ring_plane():
    mol = _molecule("c1ccc(O)cc1")
    heavy_mol, hydrogens = _as_site(mol)
    donor = rotatable_protein_donors(heavy_mol, hydrogens)[0]

    assert donor.allowed is not None
    assert len(donor.allowed) == 2

    pos = heavy_mol.GetConformer().GetPositions()
    cz = [n.GetIdx() for n in heavy_mol.GetAtomWithIdx(donor.heavy_idx).GetNeighbors()][0]
    ring = [n.GetIdx() for n in heavy_mol.GetAtomWithIdx(cz).GetNeighbors()
            if n.GetIdx() != donor.heavy_idx]
    normal = np.cross(pos[ring[0]] - pos[cz], pos[ring[1]] - pos[cz])
    normal /= np.linalg.norm(normal)

    # The optimised geometry is not planar to machine precision, so the floor is
    # set by the input, not by the rotation.
    floor = abs((donor.h_ref[0] - pos[cz]) @ normal)
    for angle in donor.allowed:
        h = np.asarray(donor.rotated(float(angle))[0])
        assert abs((h - pos[cz]) @ normal) <= max(2.0 * floor, 1e-9)

    # ... and the restriction has to actually bite: an unsnapped rotation leaves.
    off_plane = np.asarray(
        _rotate_about(donor.h_ref - donor.origin, donor.axis, np.radians(90.0))[0]
    ) + donor.origin
    assert abs((off_plane - pos[cz]) @ normal) > 0.5

    # The two in-plane orientations are the same bond flipped end for end.
    assert np.allclose(np.asarray(donor.rotated(float(donor.allowed[0]) + 180.0)),
                       np.asarray(donor.rotated(float(donor.allowed[1]))), atol=1e-9)


@pytest.mark.parametrize("angle", [0.0, 47.0, -95.0, 173.0])
def test_snapping_collapses_onto_an_allowed_orientation(angle):
    donor = rotatable_protein_donors(*_as_site(_molecule("c1ccc(O)cc1")))[0]
    snapped = donor.snap(angle)
    assert min(abs(_wrap180(snapped - a)) for a in donor.allowed) < 1e-9


def test_free_donor_does_not_snap():
    donor = rotatable_protein_donors(*_as_site(_molecule("CCO")))[0]
    assert donor.snap(47.0) == pytest.approx(47.0)


# --------------------------------------------------------------------------------------
# proximity filter
# --------------------------------------------------------------------------------------
def test_only_donors_the_scorer_can_see_are_relaxed():
    """The H-bond branch is gated at 6 A, so a distant donor cannot move the score.

    Relaxing one anyway costs ~200 ms an evaluation to prove nothing changed.
    """
    donor = rotatable_protein_donors(*_as_site(_molecule("CCO")))[0]

    near = donor.origin + np.array([[2.8, 0.0, 0.0]])
    far = donor.origin + np.array([[20.0, 0.0, 0.0]])

    assert donors_near_ligand([donor], near, cutoff=6.0) == [donor]
    assert donors_near_ligand([donor], far, cutoff=6.0) == []
    assert min_ligand_distance(donor, near) == pytest.approx(2.8)


# --------------------------------------------------------------------------------------
# the relaxer
# --------------------------------------------------------------------------------------
class _Receptor:
    """Stands in for `combined.protein_hydrogens` plus a scorer over it."""

    def __init__(self, donors, score_fn):
        self.hydrogens = {d.heavy_idx: [np.array(h) for h in d.h_ref] for d in donors}
        self._score_fn = score_fn
        self.calls = 0

    def apply(self, idx, coords):
        self.hydrogens[idx] = coords

    def score(self):
        self.calls += 1
        return self._score_fn(self.hydrogens)


def _target_seeking_score(target):
    """Lower is better as the hydrogen approaches `target` -- a fake H-bond."""
    def score(hydrogens):
        return float(min(np.linalg.norm(np.asarray(h) - target)
                         for coords in hydrogens.values() for h in coords))
    return score


def test_relaxer_finds_the_orientation_and_reports_the_starting_score():
    donor = rotatable_protein_donors(*_as_site(_molecule("CCO")))[0]

    # Put the target where a 180 degree turn would take the hydrogen.
    target = np.asarray(donor.rotated(180.0)[0])
    receptor = _Receptor([donor], _target_seeking_score(target))

    relaxer = ProteinHydrogenRelaxer([donor], receptor.apply, receptor.score,
                                     grid_deg=30.0, passes=1, maxiter=50)
    score, start_score, n_evals, max_delta, per_donor, best = relaxer.relax()

    assert score < start_score           # it improved on what it was given
    assert score == pytest.approx(0.0, abs=1e-3)
    assert abs(abs(max_delta) - 180.0) < 5.0
    assert n_evals > 0
    assert len(per_donor) == 1
    assert per_donor[0]["delta_echo"] < 0


def test_relaxer_always_restores_the_reference_placement():
    """Poses share one precompute, so a pose must not inherit the last one's protein."""
    donor = rotatable_protein_donors(*_as_site(_molecule("CCO")))[0]
    receptor = _Receptor([donor], _target_seeking_score(
        np.asarray(donor.rotated(180.0)[0])))

    relaxer = ProteinHydrogenRelaxer([donor], receptor.apply, receptor.score,
                                     grid_deg=60.0, passes=1, maxiter=20)
    _, _, _, _, _, best = relaxer.relax()

    assert np.allclose(np.asarray(receptor.hydrogens[donor.heavy_idx]), donor.h_ref)

    # ... and the caller can put the winning orientation back deliberately.
    relaxer.apply(best)
    assert not np.allclose(np.asarray(receptor.hydrogens[donor.heavy_idx]),
                           donor.h_ref)
    relaxer.restore()
    assert np.allclose(np.asarray(receptor.hydrogens[donor.heavy_idx]), donor.h_ref)


def test_no_improvement_leaves_the_pose_untouched():
    donor = rotatable_protein_donors(*_as_site(_molecule("CCO")))[0]
    receptor = _Receptor([donor], lambda _hydrogens: 1.0)   # perfectly flat

    relaxer = ProteinHydrogenRelaxer([donor], receptor.apply, receptor.score,
                                     grid_deg=60.0, passes=1, maxiter=20)
    score, start_score, _, max_delta, _, _ = relaxer.relax()

    assert score == pytest.approx(start_score)
    assert max_delta == pytest.approx(0.0)
    assert np.allclose(np.asarray(receptor.hydrogens[donor.heavy_idx]), donor.h_ref)


def test_a_throwing_scorer_costs_the_pose_nothing():
    donor = rotatable_protein_donors(*_as_site(_molecule("CCO")))[0]

    def boom(_hydrogens):
        raise RuntimeError("unscorable")

    receptor = _Receptor([donor], boom)
    relaxer = ProteinHydrogenRelaxer([donor], receptor.apply, receptor.score,
                                     grid_deg=120.0, passes=1, maxiter=5)
    score, start_score, _, max_delta, per_donor, best = relaxer.relax()

    assert score is None and start_score is None and best is None
    assert max_delta == pytest.approx(0.0)
    assert per_donor == []
    assert np.allclose(np.asarray(receptor.hydrogens[donor.heavy_idx]), donor.h_ref)


def test_nothing_to_relax_is_a_clean_skip():
    relaxer = ProteinHydrogenRelaxer([], lambda *_: None, lambda: 0.0)
    assert relaxer.relax() == (None, None, 0, 0.0, [], None)


def test_conjugated_donors_are_scanned_by_slot_not_by_grid():
    """Two orientations, so the coarse scan must not evaluate 360/grid of them.

    At 30 degrees a full sweep would be 12 evaluations per pass to explore two
    distinct placements -- ten of them duplicates, at ~200 ms each.
    """
    donor = rotatable_protein_donors(*_as_site(_molecule("c1ccc(O)cc1")))[0]
    receptor = _Receptor([donor], _target_seeking_score(
        np.asarray(donor.rotated(float(donor.allowed[1]))[0])))

    relaxer = ProteinHydrogenRelaxer([donor], receptor.apply, receptor.score,
                                     grid_deg=30.0, passes=1, maxiter=100)
    _, _, n_evals, _, _, _ = relaxer.relax()

    # 1 start + 2 slots + 1 attribution. The polish is skipped outright because
    # every coordinate is discrete.
    assert n_evals == 4
