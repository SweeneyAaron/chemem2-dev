"""Tests for IonFixer's existing-ion mode (--ion-spec).

Covers refining an ion that is already present in the input structure instead of
placing a new one, plus the atom-index-based identity that mode depends on.
"""

import unittest
from types import SimpleNamespace

import numpy as np

try:
    from ChemEM.parsers.models import AtomSpec
    from ChemEM.protocols.refine import ion_fixer as ion_fixer_module
    from ChemEM.protocols.refine.ion_fixer import IonFixer
    _HAS_OPENMM = True
except ModuleNotFoundError:
    AtomSpec = None
    ion_fixer_module = None
    IonFixer = None
    _HAS_OPENMM = False


class _FakeAtom:
    def __init__(self, idx, name, element_name, xyz, residue):
        self.idx = idx
        self.name = name
        self.element_name = element_name
        self.xx, self.xy, self.xz = (float(v) for v in xyz)
        self.residue = residue


class _FakeResidue:
    def __init__(self, chain, name, number):
        self.chain = chain
        self.name = name
        self.number = number
        self.atoms = []


class _FakeStructure:
    """Minimal stand-in for the ParmEd Structure surface IonFixer actually uses."""

    def __init__(self):
        self.residues = []

    @property
    def atoms(self):
        return [a for res in self.residues for a in res.atoms]

    def add_residue(self, chain, name, number, atoms):
        res = _FakeResidue(chain, name, number)
        next_idx = len(self.atoms)
        for offset, (atom_name, element, xyz) in enumerate(atoms):
            res.atoms.append(_FakeAtom(next_idx + offset, atom_name, element, xyz, res))
        self.residues.append(res)
        return res

    def __iadd__(self, other):
        for res in other.residues:
            self.add_residue(
                res.chain,
                res.name,
                res.number,
                [(a.name, a.element_name, (a.xx, a.xy, a.xz)) for a in res.atoms],
            )
        return self


def _water_structure(resnum, o_xyz):
    struct = _FakeStructure()
    o_xyz = np.asarray(o_xyz, dtype=float)
    struct.add_residue(
        "W",
        "HOH",
        resnum,
        [
            ("O", "O", o_xyz),
            ("H1", "H", o_xyz + np.array([0.96, 0.0, 0.0])),
            ("H2", "H", o_xyz + np.array([-0.24, 0.93, 0.0])),
        ],
    )
    return struct


class _FakeProtein:
    """Resolves spec strings the way Protein.get_atom_idx_from_spec does."""

    def __init__(self, structure):
        self.complex_structure = structure

    def get_atom_idx_from_spec(self, spec):
        chain, resname, resnum, atom_name = spec.split(":")
        for atom in self.complex_structure.atoms:
            res = atom.residue
            if (
                str(res.chain) == chain
                and str(res.name) == resname
                and int(res.number) == int(resnum)
                and str(atom.name) == atom_name
            ):
                return AtomSpec(structure=self, atom_idx=int(atom.idx))
        raise RuntimeError(f"[ERROR] no residue matches atom spec {spec!r}.")


def _zinc_site_structure():
    """A tetrahedral ZN site: three HIS NE2 donors plus the deposited ion.

    The ALA is deliberately numbered 301 in chain B -- the same residue number the
    ion carries in chain A -- to exercise the resnum-collision regression.
    """
    struct = _FakeStructure()
    struct.add_residue("A", "HIS", 94, [("NE2", "N", (2.1, 0.0, 0.0))])
    struct.add_residue("A", "HIS", 96, [("NE2", "N", (-0.7, 1.98, 0.0))])
    struct.add_residue("A", "HIS", 119, [("NE2", "N", (-0.7, -0.99, 1.71))])
    struct.add_residue(
        "B",
        "ALA",
        301,
        [
            ("N", "N", (8.0, 0.0, 0.0)),
            ("CA", "C", (9.0, 0.0, 0.0)),
            ("C", "C", (10.0, 0.0, 0.0)),
            ("O", "O", (11.0, 0.0, 0.0)),
        ],
    )
    struct.add_residue("A", "ZN", 301, [("ZN", "Zn", (0.0, 0.0, 0.0))])
    return struct


@unittest.skipUnless(_HAS_OPENMM, "OpenMM is required for IonFixer tests")
class TestIonFixerExistingIon(unittest.TestCase):
    def _fixer(self, *, ion_spec="A:ZN:301:ZN", ion_type=None, atom_specs=None,
               exclude_specs=None, coordination_geometry="Tetrahedral"):
        structure = _zinc_site_structure()
        if atom_specs is None:
            atom_specs = ["A:HIS:94:NE2", "A:HIS:96:NE2", "A:HIS:119:NE2"]
        system = SimpleNamespace(
            protein=_FakeProtein(structure),
            ligand=[],
            density_map=None,
            options=SimpleNamespace(
                ion_spec=ion_spec,
                ion_type=ion_type,
                coordination_geometry=coordination_geometry,
                atom_specs=list(atom_specs),
                exclude_specs=list(exclude_specs or []),
                no_map=False,
            ),
        )
        fixer = IonFixer(system)
        fixer.structure = structure
        return fixer

    # --- spec resolution and ion-type inference -------------------------------

    def test_ion_spec_enables_existing_ion_mode(self):
        fixer = self._fixer()
        fixer.get_spec_atoms()

        self.assertTrue(fixer.use_existing_ion)
        self.assertIsNotNone(fixer.existing_ion_spec)
        np.testing.assert_allclose(fixer.existing_ion_spec.get_point(), [0.0, 0.0, 0.0])
        self.assertEqual(len(fixer.spec_atoms), 3)

    def test_no_ion_spec_leaves_placement_mode(self):
        fixer = self._fixer(ion_spec=None, ion_type="ZN")
        fixer.get_spec_atoms()

        self.assertFalse(fixer.use_existing_ion)
        self.assertIsNone(fixer.existing_ion_spec)

    def test_ion_type_is_inferred_from_supplied_ion(self):
        fixer = self._fixer(ion_type=None)
        fixer.get_spec_atoms()

        self.assertEqual(fixer.resolve_existing_ion_type(), "ZN")
        self.assertEqual(fixer.system.options.ion_type, "ZN")

    def test_matching_explicit_ion_type_is_accepted(self):
        fixer = self._fixer(ion_type="ZN")
        fixer.get_spec_atoms()

        self.assertEqual(fixer.resolve_existing_ion_type(), "ZN")
        self.assertEqual(fixer.system.options.ion_type, "ZN")

    def test_mismatched_explicit_ion_type_raises(self):
        fixer = self._fixer(ion_type="MG")
        fixer.get_spec_atoms()

        with self.assertRaises(RuntimeError) as ctx:
            fixer.resolve_existing_ion_type()
        self.assertIn("disagrees with --ion-spec", str(ctx.exception))

    def test_ion_spec_overlapping_atom_spec_raises(self):
        fixer = self._fixer(atom_specs=["A:HIS:94:NE2", "A:ZN:301:ZN"])

        with self.assertRaises(RuntimeError) as ctx:
            fixer.get_spec_atoms()
        self.assertIn("its own coordinating atoms", str(ctx.exception))

    def test_ion_spec_overlapping_exclude_spec_raises(self):
        fixer = self._fixer(exclude_specs=["A:ZN:301:ZN"])

        with self.assertRaises(RuntimeError) as ctx:
            fixer.get_spec_atoms()
        self.assertIn("excluded/pinned", str(ctx.exception))

    def test_non_ion_residue_raises(self):
        fixer = self._fixer(ion_spec="B:ALA:301:CA", atom_specs=["A:HIS:94:NE2"])
        fixer.get_spec_atoms()

        with self.assertRaises(RuntimeError) as ctx:
            fixer.resolve_existing_ion_type()
        self.assertIn("must be monatomic", str(ctx.exception))

    # --- placement is bypassed -------------------------------------------------

    def test_initial_position_is_the_deposited_position(self):
        fixer = self._fixer()
        fixer.get_spec_atoms()

        def _boom(**kwargs):
            raise AssertionError("placement optimiser must not run in existing-ion mode")

        original = ion_fixer_module.generate_initial_ion_position
        ion_fixer_module.generate_initial_ion_position = _boom
        try:
            fixer.get_initial_position()
        finally:
            ion_fixer_module.generate_initial_ion_position = original

        np.testing.assert_allclose(fixer.initial_pos, [0.0, 0.0, 0.0])

    def test_locate_existing_ion_records_the_existing_atom(self):
        fixer = self._fixer()
        fixer.get_spec_atoms()
        fixer.selected_structure = fixer.structure

        ion_atom = fixer.locate_existing_ion()

        self.assertIsNone(fixer.ion_structure)
        self.assertEqual(fixer.added_ion_atom_idx, int(ion_atom.idx))
        self.assertEqual(fixer.added_ion_resnum, 301)
        self.assertEqual(ion_atom.residue.name, "ZN")

    def test_merge_system_does_not_duplicate_the_ion(self):
        fixer = self._fixer()
        fixer.get_spec_atoms()
        fixer.selected_structure = fixer.structure
        fixer.locate_existing_ion()

        n_residues_before = len(fixer.selected_structure.residues)
        fixer.waters = [_water_structure(500, (0.0, 0.0, 2.1))]
        fixer.merge_system()

        zinc_residues = [r for r in fixer.selected_structure.residues if r.name == "ZN"]
        self.assertEqual(len(zinc_residues), 1)
        self.assertEqual(len(fixer.selected_structure.residues), n_residues_before + 1)
        self.assertEqual(len(fixer.added_water_oxygen_indices), 1)
        self.assertEqual(len(fixer.added_water_atom_indices), 3)

    # --- identity by atom index, not residue number ----------------------------

    def test_pinning_survives_a_residue_number_collision_with_the_ion(self):
        fixer = self._fixer()
        fixer.get_spec_atoms()
        fixer.selected_structure = fixer.structure
        fixer.locate_existing_ion()
        fixer.waters = [_water_structure(500, (0.0, 0.0, 2.1))]
        fixer.merge_system()
        fixer.cache_spec_atom_indices_in_selected_structure()

        result = fixer.identify_atoms_to_pin()
        pinned = set(result["pin_atom_indices"])

        # The ion (chain A, resnum 301) must stay mobile...
        self.assertNotIn(fixer.added_ion_atom_idx, pinned)
        # ...but ALA 301 in chain B, which shares that residue number, must be pinned.
        ala = next(r for r in fixer.selected_structure.residues if r.name == "ALA")
        for atom in ala.atoms:
            self.assertIn(int(atom.idx), pinned)
        # Dummy waters and the coordinating donors stay mobile.
        for idx in fixer.added_water_atom_indices:
            self.assertNotIn(idx, pinned)
        for idx in fixer.spec_atom_indices_selected:
            self.assertNotIn(idx, pinned)

    def test_map_selection_keeps_the_ion_and_drops_dummy_waters(self):
        fixer = self._fixer()
        fixer.get_spec_atoms()
        fixer.selected_structure = fixer.structure
        fixer.locate_existing_ion()
        fixer.waters = [_water_structure(500, (0.0, 0.0, 2.1))]
        fixer.merge_system()
        fixer.cache_spec_atom_indices_in_selected_structure()
        fixer.coord_atom_indices = list(fixer.spec_atom_indices_selected)

        map_indices = set(fixer.identify_map_restrained_atoms()["map_atom_indices"])

        self.assertIn(fixer.added_ion_atom_idx, map_indices)
        for idx in fixer.added_water_atom_indices:
            self.assertNotIn(idx, map_indices)

    # --- output sync -----------------------------------------------------------

    def test_verify_existing_ion_synced_accepts_matching_coordinates(self):
        fixer = self._fixer()
        fixer.get_spec_atoms()
        fixer.selected_structure = fixer.structure
        fixer.locate_existing_ion()

        target = _zinc_site_structure()
        refined = fixer._find_ion_atom_in_selected_structure()
        refined.xx, refined.xy, refined.xz = 0.05, -0.02, 0.01
        target_ion = next(a for a in target.atoms if a.residue.name == "ZN")
        target_ion.xx, target_ion.xy, target_ion.xz = 0.05, -0.02, 0.01

        matched = fixer._verify_existing_ion_synced(target, "target")
        self.assertIs(matched, target_ion)

    def test_verify_existing_ion_synced_rejects_stale_coordinates(self):
        fixer = self._fixer()
        fixer.get_spec_atoms()
        fixer.selected_structure = fixer.structure
        fixer.locate_existing_ion()

        target = _zinc_site_structure()
        refined = fixer._find_ion_atom_in_selected_structure()
        refined.xx, refined.xy, refined.xz = 0.5, -0.2, 0.1

        with self.assertRaises(RuntimeError) as ctx:
            fixer._verify_existing_ion_synced(target, "target")
        self.assertIn("did not propagate", str(ctx.exception))


@unittest.skipUnless(_HAS_OPENMM, "OpenMM is required for IonFixer tests")
class TestIonFixerPlacementModeUnchanged(unittest.TestCase):
    """The default (place a new ion) path must be unaffected by existing-ion mode."""

    def _fixer(self):
        structure = _zinc_site_structure()
        # Drop the deposited ion: placement mode creates its own.
        structure.residues = [r for r in structure.residues if r.name != "ZN"]
        system = SimpleNamespace(
            protein=_FakeProtein(structure),
            ligand=[],
            density_map=None,
            options=SimpleNamespace(
                ion_spec=None,
                ion_type="ZN",
                coordination_geometry="Tetrahedral",
                atom_specs=["A:HIS:94:NE2", "A:HIS:96:NE2", "A:HIS:119:NE2"],
                exclude_specs=[],
                no_map=False,
            ),
        )
        fixer = IonFixer(system)
        fixer.get_spec_atoms()
        fixer.selected_structure = structure

        ion_structure = _FakeStructure()
        ion_structure.add_residue("Z", "ZN", 900, [("ZN", "Zn", (0.0, 0.0, 0.0))])
        fixer.ion_structure = ion_structure
        fixer.waters = [_water_structure(901, (0.0, 0.0, 2.1))]
        return fixer

    def test_merge_system_appends_the_new_ion(self):
        fixer = self._fixer()
        n_before = len(fixer.selected_structure.residues)

        fixer.merge_system()

        self.assertFalse(fixer.use_existing_ion)
        self.assertEqual(len(fixer.selected_structure.residues), n_before + 2)
        self.assertEqual(fixer.added_ion_resnum, 900)
        ion_atom = fixer._find_ion_atom_in_selected_structure()
        self.assertEqual(ion_atom.residue.name, "ZN")
        self.assertEqual(len(fixer.added_water_atom_indices), 3)

    def test_pinning_leaves_placed_ion_and_waters_mobile(self):
        fixer = self._fixer()
        fixer.merge_system()
        fixer.cache_spec_atom_indices_in_selected_structure()

        pinned = set(fixer.identify_atoms_to_pin()["pin_atom_indices"])

        self.assertNotIn(fixer.added_ion_atom_idx, pinned)
        for idx in fixer.added_water_atom_indices:
            self.assertNotIn(idx, pinned)
        ala = next(r for r in fixer.selected_structure.residues if r.name == "ALA")
        for atom in ala.atoms:
            self.assertIn(int(atom.idx), pinned)


if __name__ == "__main__":
    unittest.main()
