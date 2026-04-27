import unittest
from types import SimpleNamespace

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from ChemEM.protocols.orchestrator import scoring
except ModuleNotFoundError:
    from protocols.orchestrator import scoring


def _ethanol():
    """Embed a small RDKit molecule with explicit hydrogens at a known centroid."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    coords = mol.GetConformer().GetPositions()
    coords -= coords.mean(axis=0)            # centre at origin
    coords += np.array([10.0, 10.0, 10.0])   # shift into the stub map's interior
    from rdkit.Geometry import Point3D
    conf = mol.GetConformer()
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return mol


def _stub_density_map(shape=(20, 20, 20), apix=1.0, origin=(0.0, 0.0, 0.0)):
    """Stub object that quacks like ChemEM's EMMap for MapGrid.from_emmap."""
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        origin=np.asarray(origin, dtype=float),
        apix=np.asarray([apix, apix, apix], dtype=float),
        density_map=rng.standard_normal(shape).astype(np.float32),
    )


class TestQscorePoseStateless(unittest.TestCase):
    def test_returns_finite_float_for_in_bounds_pose(self):
        mol = _ethanol()
        dmap = _stub_density_map()
        coords = mol.GetConformer().GetPositions()
        score = scoring.qscore_pose(coords, mol, dmap)
        self.assertIsNotNone(score)
        self.assertTrue(np.isfinite(score))

    def test_returns_none_for_out_of_bounds_pose(self):
        mol = _ethanol()
        dmap = _stub_density_map()
        coords = mol.GetConformer().GetPositions() + np.array([1000.0, 0.0, 0.0])
        score = scoring.qscore_pose(coords, mol, dmap)
        self.assertIsNone(score)

    def test_does_not_mutate_input(self):
        mol = _ethanol()
        dmap = _stub_density_map()
        coords = mol.GetConformer().GetPositions()
        coords_before = coords.copy()
        scoring.qscore_pose(coords, mol, dmap)
        np.testing.assert_array_equal(coords, coords_before)
        # Conformer untouched too.
        np.testing.assert_array_almost_equal(
            mol.GetConformer().GetPositions(), coords_before
        )


if __name__ == "__main__":
    unittest.main()
