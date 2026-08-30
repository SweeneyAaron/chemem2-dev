"""Tests for `--dock-seed`: the ACO search seed.

The engine itself is not exercised here (nothing in the suites loads the compiled
extension), so these cover the parts that decide whether the seed is honoured:

  - the flag surface and its random-by-default behaviour;
  - the seed being resolved ONCE per run and written back onto `system.options`,
    which is where `PreCompDataProtein` reads it -- if that write-back regresses,
    each site would silently draw its own seed and the run would stop being
    reproducible as a whole;
  - the seed being logged, since random-by-default is only defensible because the
    log makes any run repeatable.

The behavioural guarantees (same seed reproduces; independent of --ncpu and of
ligand order) need the real engine and are covered by the verification runs
recorded in the README.
"""
import argparse
import types
import unittest

try:
    from ChemEM.protocols._docking.docking import Docking
    from ChemEM.protocol_spec import add_dock_args
except ModuleNotFoundError:
    from protocols._docking.docking import Docking
    from protocol_spec import add_dock_args


def _protocol(**options):
    """A Docking instance with just enough system for _resolve_seed."""
    proto = Docking.__new__(Docking)
    proto._dock_seed = None
    logged = []
    proto.system = types.SimpleNamespace(
        options=argparse.Namespace(**options),
        log=logged.append,
    )
    proto._logged = logged
    return proto


class TestFlagSurface(unittest.TestCase):
    def test_dock_seed_defaults_to_unset(self):
        """Unset is what means 'random'; a non-None default would silently freeze
        every run to one trajectory again."""
        parser = argparse.ArgumentParser()
        add_dock_args(parser)
        self.assertIsNone(parser.parse_args([]).dock_seed)

    def test_dock_seed_accepts_an_int(self):
        parser = argparse.ArgumentParser()
        add_dock_args(parser)
        self.assertEqual(parser.parse_args(["--dock-seed", "42"]).dock_seed, 42)


class TestResolveSeed(unittest.TestCase):
    def test_explicit_seed_is_used_verbatim(self):
        proto = _protocol(dock_seed=42)
        self.assertEqual(proto._resolve_seed(), 42)
        self.assertEqual(proto.system.options.dock_seed, 42)

    def test_unset_draws_a_random_seed(self):
        proto = _protocol(dock_seed=None)
        seed = proto._resolve_seed()
        self.assertIsInstance(seed, int)
        self.assertGreater(seed, 0)

    def test_two_runs_draw_different_seeds(self):
        """Random-by-default exists to surface search variance; identical draws
        would defeat it."""
        seeds = {_protocol(dock_seed=None)._resolve_seed() for _ in range(8)}
        self.assertGreater(len(seeds), 1)

    def test_random_seed_fits_in_uint64(self):
        """The value is cast to uint64_t in C++; a negative or >64-bit int would
        wrap or throw."""
        for _ in range(20):
            seed = _protocol(dock_seed=None)._resolve_seed()
            self.assertGreaterEqual(seed, 0)
            self.assertLess(seed, 1 << 64)

    def test_zero_is_replaced(self):
        """0 is a legal mt19937 seed but is reserved so 'unset' stays unambiguous
        wherever the value is falsy-checked."""
        proto = _protocol(dock_seed=0)
        self.assertNotEqual(proto._resolve_seed(), 0)

    def test_seed_is_written_back_for_the_precompute_to_read(self):
        """PreCompDataProtein reads system.options.dock_seed. Without the
        write-back each site would draw its own seed and the run would not
        reproduce as a whole."""
        proto = _protocol(dock_seed=None)
        seed = proto._resolve_seed()
        self.assertEqual(proto.system.options.dock_seed, seed)
        # Resolving again (as a second protocol in the same run would) now sees the
        # fixed value and must not re-randomise.
        again = _protocol(dock_seed=proto.system.options.dock_seed)
        self.assertEqual(again._resolve_seed(), seed)

    def test_seed_is_logged_with_a_reproduction_hint(self):
        """Random-by-default is only safe because this line exists."""
        proto = _protocol(dock_seed=None)
        seed = proto._resolve_seed()
        logged = "\n".join(proto._logged)
        self.assertIn("[dock] seed:", logged)
        self.assertIn(str(seed), logged)
        self.assertIn("--dock-seed", logged)

    def test_log_says_where_the_seed_came_from(self):
        """So a log tells you whether the run was reproducible by construction or
        only after the fact."""
        drawn = _protocol(dock_seed=None)
        drawn._resolve_seed()
        self.assertIn("(random)", "\n".join(drawn._logged))

        given = _protocol(dock_seed=7)
        given._resolve_seed()
        self.assertIn("(--dock-seed)", "\n".join(given._logged))

    def test_seed_recorded_on_the_instance_for_the_summary(self):
        proto = _protocol(dock_seed=99)
        proto._resolve_seed()
        self.assertEqual(proto._dock_seed, 99)


class TestPrecomputeDefault(unittest.TestCase):
    def test_module_default_is_non_zero(self):
        """Ad-hoc system objects in benchmark/ and score_poses/ fall back to this."""
        try:
            from ChemEM.tools.precomputed_data import DEFAULT_DOCK_SEED
        except ModuleNotFoundError:
            from tools.precomputed_data import DEFAULT_DOCK_SEED
        self.assertNotEqual(DEFAULT_DOCK_SEED, 0)


if __name__ == "__main__":
    unittest.main()
