import unittest
from types import SimpleNamespace

import numpy as np

try:
    from openmm import System as OpenMMSystem
    from ChemEM.protocols.refine.ion_fixer import IonFixer
    _HAS_OPENMM = True
except ModuleNotFoundError:
    OpenMMSystem = None
    IonFixer = None
    _HAS_OPENMM = False


@unittest.skipUnless(_HAS_OPENMM, "OpenMM is required for IonFixer tests")
class TestIonFixerDensityMap(unittest.TestCase):
    @staticmethod
    def _fixer(*, density_map=None, no_map=False):
        system = SimpleNamespace(
            density_map=density_map,
            options=SimpleNamespace(no_map=no_map),
        )
        return IonFixer(system)

    def test_standard_system_density_map_enables_restraint(self):
        density_map = object()
        fixer = self._fixer(density_map=density_map)

        self.assertIs(fixer._get_density_map(), density_map)
        self.assertTrue(fixer.should_apply_map_restraint())

    def test_no_map_option_disables_restraint(self):
        fixer = self._fixer(density_map=object(), no_map=True)

        self.assertFalse(fixer.should_apply_map_restraint())

    def test_legacy_density_attribute_remains_supported(self):
        density_map = object()
        system = SimpleNamespace(
            density=density_map,
            options=SimpleNamespace(no_map=False),
        )
        fixer = IonFixer(system)

        self.assertIs(fixer._get_density_map(), density_map)
        self.assertTrue(fixer.should_apply_map_restraint())

    def test_map_force_is_added_to_openmm_system(self):
        density_map = SimpleNamespace(
            density_map=np.ones((2, 2, 2), dtype=np.float32),
            origin=(0.0, 0.0, 0.0),
            apix=(1.0, 1.0, 1.0),
            map_contour=None,
        )
        fixer = self._fixer(density_map=density_map)
        fixer.openmm_system = OpenMMSystem()
        fixer.openmm_system.addParticle(12.0)
        fixer.local_density_map = density_map
        fixer.map_atom_indices = [0]
        fixer.added_water_resnums = []
        atom = SimpleNamespace(idx=0, residue=SimpleNamespace(number=1))
        fixer._get_atom_by_idx = lambda atom_idx: atom

        map_force = fixer.apply_map_restraint(
            k_map=75.0,
            smooth_sigma_A=0.0,
            smooth_sigma_vox=0.0,
            normalise=True,
        )

        self.assertIs(map_force, fixer.map_force)
        self.assertEqual(fixer.openmm_system.getNumForces(), 1)
        self.assertEqual(map_force.getNumBonds(), 1)
        self.assertEqual(map_force.getForceGroup(), 7)
        self.assertEqual(map_force.getGlobalParameterName(0), "global_k")
        self.assertAlmostEqual(map_force.getGlobalParameterDefaultValue(0), 75.0)


if __name__ == "__main__":
    unittest.main()
