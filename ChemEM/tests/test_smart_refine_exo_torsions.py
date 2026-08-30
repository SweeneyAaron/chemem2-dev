"""Tests for SmartRefine2 exocyclic ring-branch torsions ("exo torsions").

Covers:
  - extract_exo_torsions enumeration (finds ring->substituent torsions; excludes
    aromatic rings and trivial sub-min_downstream caps).
  - the rotation geometry: rotating downstream atoms about the ring bond j-k
    leaves ring atoms i,j,k fixed but MOVES the first branch atom l — the exact
    DOF the normal ring->branch torsion lacks.
  - _downstream_poorly_fit gate.
  - --sr2-exo-* CLI flags.
"""
import argparse
import importlib
import types
import unittest

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    _HAS_RDKIT = True
except ModuleNotFoundError:
    Chem = None
    AllChem = None
    _HAS_RDKIT = False

if _HAS_RDKIT:
    try:
        from ChemEM.protocols.smart_refine_2.smart_utils import (
            extract_exo_torsions,
            build_semantic_anchor_blocks,
            build_directional_torsion_walks,
        )
        from ChemEM.protocols.smart_refine_2 import smart_refine as sr
        from ChemEM.protocols.smart_refine_2.optimisers import _rotation_matrix
    except ModuleNotFoundError:
        from protocols.smart_refine_2.smart_utils import (
            extract_exo_torsions,
            build_semantic_anchor_blocks,
            build_directional_torsion_walks,
        )
        from protocols.smart_refine_2 import smart_refine as sr
        from protocols.smart_refine_2.optimisers import _rotation_matrix
else:
    extract_exo_torsions = None
    build_semantic_anchor_blocks = None
    build_directional_torsion_walks = None
    sr = None
    _rotation_matrix = None


def _embed(smiles, seed=1):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


# Cyclohexane bearing a -CH2-O-P(=O)(O)O substituent: a saturated ring with a
# flexible >3-heavy-atom branch, the ADP/ATP ribose->phosphate failure mode.
_PHOSPHATE_RING = "C1CCCCC1COP(=O)(O)O"


@unittest.skipUnless(_HAS_RDKIT, "rdkit required")
class TestExtractExoTorsions(unittest.TestCase):
    def test_finds_ring_branch_torsion_with_phosphate_downstream(self):
        mol = _embed(_PHOSPHATE_RING)
        exos = extract_exo_torsions(mol)
        self.assertTrue(exos, "expected at least one exo torsion")
        # Every torsion's k (torsion[2]) is a ring atom; l (torsion[3]) is the
        # exocyclic first branch atom; downstream includes the phosphorus.
        p_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "P"][0]
        ring_atoms = set(mol.GetRingInfo().AtomRings()[0])
        found_phosphate = False
        for exo in exos:
            self.assertIn(int(exo.torsion[2]), ring_atoms)        # k in ring
            self.assertNotIn(int(exo.torsion[3]), ring_atoms)     # l exocyclic
            self.assertGreater(len(exo.downstream_atoms), 3)
            if p_idx in exo.downstream_atoms:
                found_phosphate = True
        self.assertTrue(found_phosphate, "phosphate not in any downstream set")

    def test_two_quads_per_substituent(self):
        # Left+right ring-bond axes -> exactly two torsions for the single branch.
        mol = _embed(_PHOSPHATE_RING)
        self.assertEqual(len(extract_exo_torsions(mol)), 2)

    def test_excludes_aromatic_ring(self):
        self.assertEqual(extract_exo_torsions(_embed("Cc1ccccc1COP(=O)(O)O")), [])

    def test_excludes_small_downstream(self):
        # methylcyclohexane: the methyl branch has only 1 downstream heavy atom.
        self.assertEqual(extract_exo_torsions(_embed("CC1CCCCC1")), [])


@unittest.skipUnless(_HAS_RDKIT, "rdkit required")
class TestExoRotationGeometry(unittest.TestCase):
    def test_rotation_moves_first_branch_atom_keeps_ring_fixed(self):
        mol = _embed(_PHOSPHATE_RING)
        exo = next(e for e in extract_exo_torsions(mol)
                   if any(mol.GetAtomWithIdx(a).GetSymbol() == "P"
                          for a in e.downstream_atoms))
        pos = mol.GetConformer().GetPositions()
        i, j, k, l = exo.torsion

        pJ, pK = pos[j], pos[k]
        axis = pK - pJ
        axis = axis / np.linalg.norm(axis)
        R = _rotation_matrix(axis * np.deg2rad(90.0))

        down = list(exo.downstream_atoms)
        new = pos.copy()
        new[down] = (pos[down] - pJ) @ R.T + pJ

        # Ring reference atoms i, j, k are untouched.
        for ring_atom in (i, j, k):
            np.testing.assert_allclose(new[ring_atom], pos[ring_atom], atol=1e-9)
        # The first branch atom l moves (it is off the j-k axis) — the DOF the
        # normal ring->branch torsion (axis k-l, l on-axis) cannot provide.
        self.assertGreater(float(np.linalg.norm(new[l] - pos[l])), 0.1)


@unittest.skipUnless(_HAS_RDKIT, "rdkit required")
class TestExoWalkSeeding(unittest.TestCase):
    """build_directional_torsion_walks prepends an exo step to the branch walk."""

    def _blocks_and_root(self, mol):
        blocks = build_semantic_anchor_blocks(mol)
        ring = set(mol.GetRingInfo().AtomRings()[0])
        root = max(blocks, key=lambda b: len(set(b.atom_indices) & ring))
        targets = [b.block_id for b in blocks if b.block_id != root.block_id]
        return blocks, int(root.block_id), targets

    def test_exo_step_seeded_as_first_step(self):
        mol = _embed(_PHOSPHATE_RING)
        blocks, root_id, targets = self._blocks_and_root(mol)
        exos = extract_exo_torsions(mol)
        ring = set(mol.GetRingInfo().AtomRings()[0])

        walks = build_directional_torsion_walks(
            mol, blocks, root_block_id=root_id,
            target_block_ids=targets, exo_torsions=exos,
        )
        exo_first = [w for w in walks if w.steps and w.steps[0].is_exo]
        self.assertTrue(exo_first, "no walk was seeded with an exo first step")
        # The seeded step rotates about a ring bond.
        self.assertTrue(set(exo_first[0].steps[0].axis) <= ring)

    def test_no_exo_steps_without_exo_torsions(self):
        mol = _embed(_PHOSPHATE_RING)
        blocks, root_id, targets = self._blocks_and_root(mol)
        walks = build_directional_torsion_walks(
            mol, blocks, root_block_id=root_id, target_block_ids=targets,
        )
        self.assertFalse(any(s.is_exo for w in walks for s in w.steps))


@unittest.skipUnless(_HAS_RDKIT, "rdkit required")
class TestDownstreamPoorlyFit(unittest.TestCase):
    def _rl(self, per_atom_q, block_q, best_idx):
        return types.SimpleNamespace(
            _per_atom_qscores=np.asarray(per_atom_q, dtype=float),
            _block_qscores=list(block_q),
            get_best_block_by_qscore=lambda: best_idx,
        )

    def test_true_when_branch_below_best_block(self):
        rl = self._rl([0.9, 0.9, 0.2, 0.2], block_q=[0.9, 0.2], best_idx=0)
        self.assertTrue(sr._downstream_poorly_fit(rl, np.array([2, 3])))

    def test_false_when_branch_fits_well(self):
        rl = self._rl([0.9, 0.9, 0.95, 0.95], block_q=[0.9, 0.95], best_idx=0)
        self.assertFalse(sr._downstream_poorly_fit(rl, np.array([2, 3])))

    def test_empty_rows_false(self):
        rl = self._rl([0.9], block_q=[0.9], best_idx=0)
        self.assertFalse(sr._downstream_poorly_fit(rl, np.array([], dtype=int)))


@unittest.skipUnless(_HAS_RDKIT, "rdkit required")
class TestExoCliFlags(unittest.TestCase):
    def _parser(self):
        protocol_spec = importlib.import_module("protocol_spec")
        parser = argparse.ArgumentParser()
        protocol_spec.add_smart_ligand_refine2_args(parser)
        return parser

    def test_defaults(self):
        ns = self._parser().parse_args([])
        self.assertTrue(ns.sr2_exo_torsions)
        self.assertAlmostEqual(ns.sr2_exo_step_deg, 20.0)
        self.assertEqual(ns.sr2_exo_min_downstream, 3)

    def test_disable_and_overrides(self):
        ns = self._parser().parse_args(
            ["--no-sr2-exo-torsions", "--sr2-exo-step-deg", "30", "--sr2-exo-min-downstream", "4"]
        )
        self.assertFalse(ns.sr2_exo_torsions)
        self.assertAlmostEqual(ns.sr2_exo_step_deg, 30.0)
        self.assertEqual(ns.sr2_exo_min_downstream, 4)


if __name__ == "__main__":
    unittest.main()
