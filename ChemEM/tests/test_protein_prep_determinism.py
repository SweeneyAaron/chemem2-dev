"""Tests for deterministic protein preparation and the prepared-protein cache.

The expensive end-to-end determinism check (prepare the same PDB in N processes,
assert identical coordinates) needs a real structure and minutes of OpenMM, so it
lives in the verification script rather than here. What is unit-testable is the
logic that decides *whether* a run is reproducible and whether a cache entry may
be trusted:

  - PrepOptions must reject seed 0, which OpenMM silently reads as "pick a fresh
    random seed" -- the exact failure the whole change exists to remove.
  - The cache key must change when, and only when, something that changes the
    prepared coordinates changes.
  - The topology round-trip must be lossless, including the duplicate chain ids
    that split_chains_on_breaks produces and the bonds prep adds, because a PDB
    round trip is not.
  - Positions must survive at full float64 precision: the residue map matches
    backbone atoms to 1e-5 A, so a lossy cache returns an EMPTY map with no error.
"""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

try:
    from ChemEM.parsers.remodel import determinism as det
    from ChemEM.parsers.remodel import prep_cache as pc
except ModuleNotFoundError:
    from parsers.remodel import determinism as det
    from parsers.remodel import prep_cache as pc


class TestPrepOptions(unittest.TestCase):
    def test_rejects_zero_seed(self):
        """OpenMM reads seed 0 as 'fresh random seed per Context'."""
        with self.assertRaises(ValueError):
            det.PrepOptions(seed=0)

    def test_zero_seed_allowed_when_determinism_is_off(self):
        self.assertEqual(det.PrepOptions(seed=0, deterministic=False).seed, 0)

    def test_default_seed_is_non_zero(self):
        self.assertNotEqual(det.DEFAULT_PREP_SEED, 0)
        self.assertEqual(det.PrepOptions().seed, det.DEFAULT_PREP_SEED)

    def test_key_fields_cover_everything_that_changes_coordinates(self):
        fields = det.PrepOptions().key_fields()
        for name in ("platform", "threads", "seed", "pH", "deterministic",
                     "clash_relief_steps", "h_placement_implicit"):
            self.assertIn(name, fields)

    def test_dead_add_missing_residues_flag_is_gone(self):
        """It never did anything: model_to_fixer_interchange round-trips through a
        temp PDB, PDBFile writes no SEQRES, so findMissingResidues always returns
        {} and loops are never rebuilt. Keeping it advertised control we lacked."""
        self.assertFalse(hasattr(det.PrepOptions(), "add_missing_residues"))
        self.assertNotIn("add_missing_residues", det.PrepOptions().key_fields())

    def test_clash_relief_is_uncapped_by_default(self):
        """Capping is a large speedup but is NOT safe as a default: the useful
        snapshot lands at a structure-dependent iteration. A 600-step cap
        reproduces the uncapped structure exactly on 9e26 yet leaves a 0.655 A
        worst contact on 7bxu against 1.052 A uncapped."""
        self.assertIsNone(det.DEFAULT_CLASH_RELIEF_STEPS)
        self.assertIsNone(det.PrepOptions().clash_relief_steps)

    def test_h_placement_keeps_implicit_solvent_by_default(self):
        """Dropping GB is ~2x faster but shifts echo_total by up to 0.63 units --
        the ECHO electrostatic grid uses per-atom charges including hydrogens, so
        moving them rewrites it. Too large to change silently."""
        self.assertTrue(det.PrepOptions().h_placement_implicit)


class TestBoundedClashRelief(unittest.TestCase):
    """The wrapper caps LangevinIntegrator.step and must always restore it."""

    def _integrator(self):
        try:
            from openmm import LangevinIntegrator
            return LangevinIntegrator
        except Exception:
            self.skipTest("OpenMM unavailable")

    def test_restores_step_afterwards(self):
        LangevinIntegrator = self._integrator()
        before = LangevinIntegrator.step
        with det.bounded_clash_relief(400):
            self.assertIsNot(LangevinIntegrator.step, before)
        self.assertIs(LangevinIntegrator.step, before)

    def test_restores_step_on_exception(self):
        LangevinIntegrator = self._integrator()
        before = LangevinIntegrator.step
        with self.assertRaises(RuntimeError):
            with det.bounded_clash_relief(400):
                raise RuntimeError("boom")
        self.assertIs(LangevinIntegrator.step, before)

    def test_none_leaves_step_untouched(self):
        LangevinIntegrator = self._integrator()
        before = LangevinIntegrator.step
        with det.bounded_clash_relief(None):
            self.assertIs(LangevinIntegrator.step, before)

    def test_budget_is_spent_then_exhausted(self):
        """Calls draw from a shared budget; once spent, further calls are no-ops
        so PDBFixer's loop finishes without doing more dynamics."""
        LangevinIntegrator = self._integrator()
        original = LangevinIntegrator.step
        calls = []
        LangevinIntegrator.step = lambda self, n: calls.append(n)
        try:
            with det.bounded_clash_relief(500):
                LangevinIntegrator.step(None, 200)   # full
                LangevinIntegrator.step(None, 200)   # full
                LangevinIntegrator.step(None, 200)   # clipped to 100
                LangevinIntegrator.step(None, 200)   # nothing left
            self.assertEqual(calls, [200, 200, 100])
        finally:
            LangevinIntegrator.step = original

    def test_zero_budget_skips_all_dynamics(self):
        LangevinIntegrator = self._integrator()
        original = LangevinIntegrator.step
        calls = []
        LangevinIntegrator.step = lambda self, n: calls.append(n)
        try:
            with det.bounded_clash_relief(0):
                LangevinIntegrator.step(None, 200)
            self.assertEqual(calls, [])
        finally:
            LangevinIntegrator.step = original


class TestPlatformScoping(unittest.TestCase):
    def test_thread_default_is_restored(self):
        """Platform property defaults are process-global: leaking Threads=1 would
        silently serialise every later docking Context."""
        try:
            from openmm import Platform
            platform = Platform.getPlatformByName("CPU")
        except Exception:
            self.skipTest("CPU platform unavailable")

        before = platform.getPropertyDefaultValue("Threads")
        with det.prep_platform(det.PrepOptions(platform="CPU", threads=1)) as plat:
            self.assertIsNotNone(plat)
            self.assertEqual(platform.getPropertyDefaultValue("Threads"), "1")
        self.assertEqual(platform.getPropertyDefaultValue("Threads"), before)

    def test_thread_default_restored_on_exception(self):
        try:
            from openmm import Platform
            platform = Platform.getPlatformByName("CPU")
        except Exception:
            self.skipTest("CPU platform unavailable")

        before = platform.getPropertyDefaultValue("Threads")
        with self.assertRaises(RuntimeError):
            with det.prep_platform(det.PrepOptions(platform="CPU", threads=1)):
                raise RuntimeError("boom")
        self.assertEqual(platform.getPropertyDefaultValue("Threads"), before)

    def test_non_deterministic_mode_yields_no_platform(self):
        with det.prep_platform(det.PrepOptions(deterministic=False)) as plat:
            self.assertIsNone(plat)


class _CacheCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pdb = os.path.join(self.tmp, "in.pdb")
        with open(self.pdb, "w") as fh:
            fh.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000\nEND\n")
        self.cache = pc.ProteinPrepCache(root=os.path.join(self.tmp, "cache"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _key(self, **kw):
        params = dict(forcefield_files=["amber14/protein.ff14SB.xml"],
                      request_implicit=True, force_ff=False,
                      prep=det.PrepOptions())
        params.update(kw)
        return self.cache.key(self.pdb, **params)


class TestCacheKey(_CacheCase):
    def test_stable_for_identical_inputs(self):
        self.assertEqual(self._key(), self._key())

    def test_changes_with_file_contents(self):
        first = self._key()
        with open(self.pdb, "a") as fh:
            fh.write("ATOM      2  CA  ALA A   1       1.000   0.000   0.000\n")
        self.assertNotEqual(first, self._key())

    def test_changes_with_forcefield_list(self):
        self.assertNotEqual(
            self._key(),
            self._key(forcefield_files=["amber14/protein.ff14SB.xml", "implicit/gbn2.xml"]),
        )

    def test_forcefield_order_matters(self):
        a = self._key(forcefield_files=["a.xml", "b.xml"])
        b = self._key(forcefield_files=["b.xml", "a.xml"])
        self.assertNotEqual(a, b)

    def test_changes_with_prep_seed_and_platform(self):
        base = self._key()
        self.assertNotEqual(base, self._key(prep=det.PrepOptions(seed=999)))
        self.assertNotEqual(base, self._key(prep=det.PrepOptions(platform="Reference")))
        self.assertNotEqual(base, self._key(prep=det.PrepOptions(deterministic=False)))

    def test_changes_with_settings_that_move_atoms(self):
        """Both of these alter prepared coordinates, so an entry written under one
        must never be served under the other."""
        base = self._key()
        self.assertNotEqual(base, self._key(prep=det.PrepOptions(clash_relief_steps=2000)))
        self.assertNotEqual(base, self._key(prep=det.PrepOptions(clash_relief_steps=0)))
        self.assertNotEqual(base, self._key(prep=det.PrepOptions(h_placement_implicit=False)))

    def test_schema_version_invalidates_old_entries(self):
        """Bumping the schema must retire every entry written before it, for prep
        changes not captured by any other keyed field."""
        base = self._key()
        original = det.PREP_SCHEMA_VERSION
        try:
            det.PREP_SCHEMA_VERSION = original + 1
            self.assertNotEqual(base, self._key())
        finally:
            det.PREP_SCHEMA_VERSION = original

    def test_independent_of_file_path(self):
        """Same structure at two paths should share one entry."""
        other = os.path.join(self.tmp, "copy.pdb")
        shutil.copy(self.pdb, other)
        moved = self.cache.key(other, forcefield_files=["amber14/protein.ff14SB.xml"],
                               request_implicit=True, force_ff=False,
                               prep=det.PrepOptions())
        self.assertEqual(self._key(), moved)


class TestCacheStoreLoad(_CacheCase):
    """Round-trip a Modeller through the cache."""

    def _modeller(self):
        from openmm import unit
        from openmm.app import Modeller, Topology, element

        top = Topology()
        # Two chains sharing an id, as split_chains_on_breaks produces -- the case
        # a PDB round trip cannot represent.
        positions = []
        atoms = []
        for _ in range(2):
            chain = top.addChain("A")
            res = top.addResidue("ALA", chain)
            for name, sym in (("N", "N"), ("CA", "C")):
                atoms.append(top.addAtom(name, element.get_by_symbol(sym), res))
        top.addBond(atoms[0], atoms[1])
        # A bond createStandardBonds() would never rebuild: across the two chains.
        top.addBond(atoms[1], atoms[2])

        # Coordinates with far more precision than PDB '%.3f' can hold.
        coords = np.array([[0.1234567890123, 0.2, 0.3],
                           [0.4, 0.5678901234567, 0.6],
                           [0.7, 0.8, 0.9012345678901],
                           [1.0, 1.1, 1.2]], dtype=np.float64)
        return Modeller(top, coords * unit.nanometer), coords

    def test_round_trip_is_bit_exact(self):
        """The 1e-5 A residue match means any rounding empties the residue map."""
        from openmm import unit

        modeller, coords = self._modeller()
        key = self._key()
        self.cache.store(key, modeller, n_mapped=7)

        loaded = self.cache.load(key)
        self.assertIsNotNone(loaded)
        back = np.asarray(loaded.positions.value_in_unit(unit.nanometer), dtype=np.float64)
        np.testing.assert_array_equal(back, coords)

    def test_round_trip_preserves_topology(self):
        modeller, _ = self._modeller()
        key = self._key()
        self.cache.store(key, modeller, n_mapped=7)
        loaded = self.cache.load(key)

        self.assertEqual(loaded.topology.getNumChains(), 2)
        self.assertEqual(loaded.topology.getNumAtoms(), 4)
        self.assertEqual(loaded.topology.getNumBonds(), 2)
        self.assertEqual([c.id for c in loaded.topology.chains()], ["A", "A"])
        self.assertEqual([a.name for a in loaded.topology.atoms()], ["N", "CA", "N", "CA"])

    def test_expected_mapped_is_recorded(self):
        modeller, _ = self._modeller()
        key = self._key()
        self.cache.store(key, modeller, n_mapped=7)
        self.assertEqual(self.cache.expected_mapped(key), 7)

    def test_miss_returns_none(self):
        self.assertIsNone(self.cache.load(self._key()))

    def test_incomplete_entry_is_a_miss(self):
        """A crashed or racing writer must never be read as valid."""
        modeller, _ = self._modeller()
        key = self._key()
        self.cache.store(key, modeller, n_mapped=7)
        os.remove(os.path.join(self.cache._dir(key), pc._COMPLETE_MARKER))
        self.assertIsNone(self.cache.load(key))

    def test_corrupt_entry_falls_back_rather_than_raising(self):
        modeller, _ = self._modeller()
        key = self._key()
        self.cache.store(key, modeller, n_mapped=7)
        with open(os.path.join(self.cache._dir(key), "positions.npy"), "w") as fh:
            fh.write("not a numpy file")
        self.assertIsNone(self.cache.load(key))

    def test_atom_count_mismatch_is_rejected(self):
        modeller, _ = self._modeller()
        key = self._key()
        self.cache.store(key, modeller, n_mapped=7)
        path = os.path.join(self.cache._dir(key), "positions.npy")
        np.save(path, np.zeros((3, 3)))
        self.assertIsNone(self.cache.load(key))

    def test_refresh_forces_a_miss(self):
        modeller, _ = self._modeller()
        key = self._key()
        self.cache.store(key, modeller, n_mapped=7)
        self.assertIsNotNone(self.cache.load(key))
        self.assertIsNone(pc.ProteinPrepCache(root=self.cache.root, refresh=True).load(key))

    def test_invalidate_removes_the_entry(self):
        modeller, _ = self._modeller()
        key = self._key()
        self.cache.store(key, modeller, n_mapped=7)
        self.cache.invalidate(key)
        self.assertIsNone(self.cache.load(key))

    def test_prepared_pdb_is_written_but_never_loaded_from(self):
        """It exists for humans. Loading from it would round coordinates to
        '%.3f' and empty the residue map."""
        modeller, coords = self._modeller()
        key = self._key()
        self.cache.store(key, modeller, n_mapped=7)
        self.assertTrue(os.path.exists(os.path.join(self.cache._dir(key), "prepared.pdb")))

        # Corrupting it must not affect a load.
        with open(os.path.join(self.cache._dir(key), "prepared.pdb"), "w") as fh:
            fh.write("GARBAGE\n")
        from openmm import unit
        back = np.asarray(self.cache.load(key).positions.value_in_unit(unit.nanometer),
                          dtype=np.float64)
        np.testing.assert_array_equal(back, coords)


class TestCacheMeta(_CacheCase):
    def test_meta_records_versions_and_counts(self):
        from openmm import unit
        from openmm.app import Modeller, Topology, element

        top = Topology()
        res = top.addResidue("ALA", top.addChain("A"))
        top.addAtom("N", element.get_by_symbol("N"), res)
        modeller = Modeller(top, np.zeros((1, 3)) * unit.nanometer)

        key = self._key()
        self.cache.store(key, modeller, n_mapped=1)
        with open(os.path.join(self.cache._dir(key), "meta.json")) as fh:
            meta = json.load(fh)

        self.assertEqual(meta["n_atoms"], 1)
        self.assertEqual(meta["n_residues"], 1)
        self.assertEqual(meta["n_mapped_residues"], 1)
        self.assertIn("openmm", meta["versions"])
        self.assertIn("pdbfixer", meta["versions"])


if __name__ == "__main__":
    unittest.main()
