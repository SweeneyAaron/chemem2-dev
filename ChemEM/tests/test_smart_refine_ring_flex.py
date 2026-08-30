import argparse
import importlib
import unittest

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
            aliphatic_ring_systems,
            generate_ring_conformers,
        )
    except ModuleNotFoundError:
        from protocols.smart_refine_2.smart_utils import (
            aliphatic_ring_systems,
            generate_ring_conformers,
        )
else:
    aliphatic_ring_systems = None
    generate_ring_conformers = None


def _embed(smiles, seed=1):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


@unittest.skipUnless(_HAS_RDKIT, "rdkit required")
class TestAliphaticRingSystems(unittest.TestCase):
    def test_detects_saturated_ring(self):
        # benzyl-cyclohexane: one aliphatic ring (the cyclohexane), the phenyl
        # is aromatic and must be excluded.
        mol = _embed("c1ccccc1CC2CCCCC2")
        systems = aliphatic_ring_systems(mol)
        self.assertEqual(len(systems), 1)
        self.assertEqual(len(systems[0]), 6)

    def test_excludes_aromatic_only(self):
        mol = _embed("c1ccccc1C")
        self.assertEqual(aliphatic_ring_systems(mol), [])

    def test_excludes_acyclic(self):
        mol = _embed("CCCCCO")
        self.assertEqual(aliphatic_ring_systems(mol), [])

    def test_merges_fused_saturated_rings(self):
        # decalin: two fused saturated rings -> a single ring system.
        mol = _embed("C1CCC2CCCCC2C1")
        systems = aliphatic_ring_systems(mol)
        self.assertEqual(len(systems), 1)
        self.assertEqual(len(systems[0]), 10)


@unittest.skipUnless(_HAS_RDKIT, "rdkit required")
class TestGenerateRingConformers(unittest.TestCase):
    def test_generates_distinct_puckers(self):
        mol = _embed("c1ccccc1CC2CCCCC2")
        ring = aliphatic_ring_systems(mol)[0]
        confs = generate_ring_conformers(mol, ring, n_confs=6, min_ring_rmsd=0.3)
        self.assertGreater(len(confs), 0)
        self.assertLessEqual(len(confs), 6)
        # Every conformer maps exactly the ring atoms.
        for conf in confs:
            self.assertEqual(set(conf.keys()), set(ring))

    def test_too_small_ring_returns_empty(self):
        mol = _embed("C1CC1C")  # cyclopropane (size 3) is excluded
        self.assertEqual(generate_ring_conformers(mol, (0, 1, 2), n_confs=4), [])


@unittest.skipUnless(_HAS_RDKIT, "rdkit required")
class TestRingFlexCliFlags(unittest.TestCase):
    def test_flag_defaults(self):
        protocol_spec = importlib.import_module("protocol_spec")
        parser = argparse.ArgumentParser()
        protocol_spec.add_smart_ligand_refine2_args(parser)
        defaults = parser.parse_args([])
        self.assertFalse(defaults.sr2_pre_minimise)
        self.assertTrue(defaults.sr2_ring_flex)
        self.assertEqual(defaults.sr2_ring_flex_confs, 8)
        self.assertAlmostEqual(defaults.sr2_ring_flex_rmsd, 0.3)

    def test_flag_overrides(self):
        protocol_spec = importlib.import_module("protocol_spec")
        parser = argparse.ArgumentParser()
        protocol_spec.add_smart_ligand_refine2_args(parser)
        ns = parser.parse_args(
            [
                "--sr2-pre-minimise",
                "--no-sr2-ring-flex",
                "--sr2-ring-flex-confs",
                "12",
                "--sr2-ring-flex-rmsd",
                "0.5",
            ]
        )
        self.assertTrue(ns.sr2_pre_minimise)
        self.assertFalse(ns.sr2_ring_flex)
        self.assertEqual(ns.sr2_ring_flex_confs, 12)
        self.assertAlmostEqual(ns.sr2_ring_flex_rmsd, 0.5)


if __name__ == "__main__":
    unittest.main()
