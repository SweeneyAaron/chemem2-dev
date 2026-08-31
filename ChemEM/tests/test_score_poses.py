"""Tests for the ECHO scorer of ``--score --score-with echo``.

Covers the things that are easy to get silently wrong:
  - the weighted/raw term bookkeeping must reproduce the C++ total exactly
    (`breakdown`), including keeping the lumped aromatic/nonbond duplicates out
    of the weighted sum;
  - `resolve_rep_max` must default to --repulsion-cap-polish, not the
    run_echo_score pybind default of 5.0, or every pose scores several units
    better than docking said it did;
  - the donor-H torsion filter must only ever admit torsions that move a single
    hydrogen, which is what makes "ligand H only" a topological guarantee.
"""
import argparse
import types
import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

try:
    from ChemEM.protocols.score import hydrogen_torsions as ht
    from ChemEM.protocols.score import poses as sp
    from ChemEM.protocols.score.scorers import echo as rp
except ModuleNotFoundError:
    from protocols.score import hydrogen_torsions as ht
    from protocols.score import poses as sp
    from protocols.score.scorers import echo as rp


def _weights(**overrides):
    """Stand-in for ECHOWeights: any attribute defaults to 0.0."""
    w = types.SimpleNamespace()
    for name in rp.RAW_TERMS:
        setattr(w, name, 0.0)
    for name, value in overrides.items():
        setattr(w, name, value)
    return w


class TestBreakdown(unittest.TestCase):
    def test_weighted_terms_sum_to_echo_linear(self):
        weights = _weights(hbond_raw=-0.5, clash=-2.0, nonbond_attr=-0.25)
        terms = {name: 0.0 for name in rp.RAW_TERMS}
        terms.update(hbond_raw=-10.0, clash=3.0, nonbond_attr=-8.0)

        raw, weighted, echo_linear, _map = rp.breakdown(terms, 0.0, weights)

        self.assertAlmostEqual(weighted["hbond_raw"], 5.0)
        self.assertAlmostEqual(weighted["clash"], -6.0)
        self.assertAlmostEqual(weighted["nonbond_attr"], 2.0)
        self.assertAlmostEqual(echo_linear, sum(weighted.values()))
        self.assertAlmostEqual(echo_linear, 1.0)
        self.assertEqual(raw["hbond_raw"], -10.0)

    def test_lumped_channels_are_never_weighted(self):
        """`aromatic`/`nonbond` duplicate their split channels; weighting both
        would double-count them into the total."""
        self.assertNotIn("aromatic", rp.LINEAR_TERMS)
        self.assertNotIn("nonbond", rp.LINEAR_TERMS)

        # Even with a non-zero weight on the lumped names, they contribute nothing.
        weights = _weights(aromatic=-9.0, nonbond=-9.0)
        terms = {name: 0.0 for name in rp.RAW_TERMS}
        terms.update(aromatic=5.0, nonbond=5.0)
        _raw, weighted, echo_linear, _map = rp.breakdown(terms, 0.0, weights)

        self.assertEqual(echo_linear, 0.0)
        self.assertNotIn("aromatic", weighted)
        self.assertNotIn("nonbond", weighted)

    def test_map_score_inverts_the_cpp_total(self):
        """ECHOScore::score computes
        total = -(linear + map) + bias + constraint + covalent,
        so recovering `map` from the total must invert exactly that."""
        weights = _weights(hbond_raw=2.0)
        terms = {name: 0.0 for name in rp.RAW_TERMS}
        terms.update(hbond_raw=3.0, bias=0.5, constraint=0.25, covalent=0.125)

        echo_linear_expected = 6.0
        map_expected = -1.75
        offsets = 0.875
        echo_total = -(echo_linear_expected + map_expected) + offsets

        _raw, _w, echo_linear, map_score = rp.breakdown(terms, echo_total, weights)

        self.assertAlmostEqual(echo_linear, echo_linear_expected)
        self.assertAlmostEqual(map_score, map_expected)

    def test_missing_channel_defaults_to_zero(self):
        """`covalent` is absent from older engines' term dicts."""
        _raw, _w, echo_linear, _map = rp.breakdown({}, 0.0, _weights())
        self.assertEqual(echo_linear, 0.0)


class TestRepMax(unittest.TestCase):
    def test_defaults_to_repulsion_cap_polish(self):
        """run_aco_docking ranks its returned poses with repCap_final_nm, so the
        re-scorer must too -- not the pybind rep_max=5.0 default."""
        opts = argparse.Namespace(score_echo_rep_max=None, repulsion_cap_polish=15.0)
        self.assertEqual(rp.resolve_rep_max(opts), 15.0)

    def test_tracks_a_changed_dock_flag(self):
        opts = argparse.Namespace(score_echo_rep_max=None, repulsion_cap_polish=22.0)
        self.assertEqual(rp.resolve_rep_max(opts), 22.0)

    def test_explicit_override_wins(self):
        opts = argparse.Namespace(score_echo_rep_max=5.0, repulsion_cap_polish=15.0)
        self.assertEqual(rp.resolve_rep_max(opts), 5.0)


def _embed(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    return mol


class TestDonorHTorsions(unittest.TestCase):
    def test_every_admitted_torsion_moves_exactly_one_hydrogen(self):
        """The property the whole 'ligand H only' guarantee rests on."""
        for smiles in ("CCO", "OCC(O)CO", "c1ccccc1CCO", "CC(=O)NCCO"):
            mol = _embed(smiles)
            for torsion in ht.donor_h_torsions(mol):
                self.assertEqual(mol.GetAtomWithIdx(torsion[3]).GetSymbol(), "H",
                                 f"{smiles}: torsion {torsion} does not end on an H")

                before = mol.GetConformer(0).GetPositions().copy()
                conf = mol.GetConformer(0)
                rdMolTransforms.SetDihedralDeg(
                    conf, *torsion,
                    rdMolTransforms.GetDihedralDeg(conf, *torsion) + 37.0,
                )
                moved = np.linalg.norm(conf.GetPositions() - before, axis=1) > 1e-6

                self.assertEqual(moved.sum(), 1, f"{smiles}: {torsion} moved {moved.sum()} atoms")
                self.assertTrue(moved[torsion[3]])

    def test_each_hydrogen_appears_at_most_once(self):
        mol = _embed("OCC(O)CO")
        moved = [t[3] for t in ht.donor_h_torsions(mol)]
        self.assertEqual(len(moved), len(set(moved)))

    def test_molecule_without_donor_hydrogens(self):
        self.assertEqual(ht.donor_h_torsions(_embed("c1ccccc1")), [])
        self.assertEqual(ht.donor_h_torsions(_embed("CCOCC")), [])

    def test_conformerless_molecule_is_not_an_error(self):
        self.assertEqual(ht.donor_h_torsions(Chem.AddHs(Chem.MolFromSmiles("CCO"))), [])
        self.assertEqual(ht.donor_h_torsions(None), [])


class TestRelaxer(unittest.TestCase):
    def test_no_torsions_is_a_clean_skip(self):
        relaxer = ht.HydrogenTorsionRelaxer([], lambda mol: 0.0)
        score, evals, delta = relaxer.relax(_embed("c1ccccc1"), 0)
        self.assertIsNone(score)
        self.assertEqual((evals, delta), (0, 0.0))

    def test_finds_the_minimum_and_leaves_heavy_atoms_alone(self):
        mol = _embed("OCC(O)CO")
        torsions = ht.donor_h_torsions(mol)
        self.assertTrue(torsions)

        target = 77.0
        conf_of = lambda m: m.GetConformer(0)

        def score(work):
            # Minimised when the first torsion sits at `target`.
            angle = rdMolTransforms.GetDihedralDeg(conf_of(work), *torsions[0])
            return abs((angle - target + 180.0) % 360.0 - 180.0)

        heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
        before = mol.GetConformer(0).GetPositions().copy()

        relaxer = ht.HydrogenTorsionRelaxer(torsions, score, grid_deg=30.0, passes=1)
        best, evals, _delta = relaxer.relax(mol, 0)

        self.assertLess(best, 1.0)
        self.assertGreater(evals, 0)
        after = mol.GetConformer(0).GetPositions()
        # Heavy atoms are held fixed by topology, not by a restraint, so the only
        # movement possible is float noise from RDKit's rigid transform (~1 ULP).
        self.assertLess(np.abs(before[heavy] - after[heavy]).max(), 1e-12)
        moved = np.linalg.norm(after - before, axis=1) > 1e-9
        self.assertTrue(all(mol.GetAtomWithIdx(int(i)).GetSymbol() == "H"
                            for i in np.where(moved)[0]))

    def test_a_failing_objective_leaves_the_pose_untouched(self):
        mol = _embed("CCO")
        torsions = ht.donor_h_torsions(mol)
        self.assertTrue(torsions)
        before = mol.GetConformer(0).GetPositions().copy()

        def boom(work):
            raise RuntimeError("scoring blew up")

        score, _evals, delta = ht.HydrogenTorsionRelaxer(torsions, boom).relax(mol, 0)

        self.assertIsNone(score)
        self.assertEqual(delta, 0.0)
        np.testing.assert_array_equal(before, mol.GetConformer(0).GetPositions())

    def test_never_returns_worse_than_the_input_placement(self):
        """A pose must never come out of relaxation scoring worse than it went in."""
        mol = _embed("CCO")
        torsions = ht.donor_h_torsions(mol)
        conf = mol.GetConformer(0)
        start = rdMolTransforms.GetDihedralDeg(conf, *torsions[0])

        # Rewards precisely the incoming angle; every scan point is worse.
        def score(work):
            angle = rdMolTransforms.GetDihedralDeg(work.GetConformer(0), *torsions[0])
            return 0.0 if abs((angle - start + 180.0) % 360.0 - 180.0) < 1e-6 else 5.0

        best, _evals, delta = ht.HydrogenTorsionRelaxer(
            torsions, score, grid_deg=90.0, passes=1, maxiter=5
        ).relax(mol, 0)

        self.assertEqual(best, 0.0)
        self.assertAlmostEqual(delta, 0.0, places=6)


class TestRepMaxDeprecatedSpelling(unittest.TestCase):
    def test_old_flag_lands_on_the_new_dest(self):
        """--rescore-rep-max must keep working, and must keep meaning rep_max."""
        from ChemEM.__main__ import build_parser

        args = build_parser().parse_args(
            ["conf.txt", "--score", "--rescore-rep-max", "7.5"]
        )
        self.assertEqual(args.score_echo_rep_max, 7.5)
        self.assertEqual(rp.resolve_rep_max(args), 7.5)

    def test_unused_old_flag_does_not_clobber_the_new_default(self):
        """The alias shares a dest with the new flag; if it also carried a default,
        it would overwrite `None` and silently pin rep_max."""
        from ChemEM.__main__ import build_parser

        args = build_parser().parse_args(["conf.txt", "--score"])
        self.assertIsNone(args.score_echo_rep_max)


class _FakeLigand:
    def __init__(self, source, mol, identifier):
        self.input = source
        self.mol = mol
        self.identifier = identifier


class TestPoseNumbering(unittest.TestCase):
    """A multi-record SDF loads as one Ligand per record, so the pose number has to
    come from the source grouping, not from ligand_idx or conf_id."""

    def _poses(self, sources):
        mol = _embed("CCO")
        system = types.SimpleNamespace(
            ligand=[_FakeLigand(s, mol, f"L{i}") for i, s in enumerate(sources)]
        )
        out = []
        for _lig_ctx, poses in sp.iter_ligands(system, ctx=None):
            out.extend(poses)
        return out

    def test_poses_are_numbered_per_source(self):
        poses = self._poses(["a.sdf", "a.sdf", "b.sdf", "a.sdf"])
        self.assertEqual([p.pose for p in poses], [0, 1, 0, 2])
        self.assertEqual([p.ligand_idx for p in poses], [0, 1, 2, 3])

    def test_a_conformerless_ligand_yields_no_poses(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))  # never embedded
        system = types.SimpleNamespace(ligand=[_FakeLigand("a.sdf", mol, "L0")])
        pairs = list(sp.iter_ligands(system, ctx=None))
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][1], [])

    def test_source_stem_falls_back_for_smiles(self):
        self.assertEqual(sp.source_stem("c1ccccc1CCO", 7), "Ligand_7")
        self.assertEqual(sp.source_stem("", 2), "Ligand_2")


if __name__ == "__main__":
    unittest.main()
