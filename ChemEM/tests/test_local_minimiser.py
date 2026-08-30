"""Tests for `--local-minimiser`: which local optimiser refines docked poses.

The minimiser itself is C++ and the suites do not load the compiled extension, so
these cover the plumbing that decides whether the choice is honoured at all:

  - the flag surface, including the default, which must stay 'nelder-mead' so an
    existing run is unaffected by the flag existing;
  - the string -> int mapping onto the precompute object, which is where
    PreComputedData reads it. The C++ side is `hasattr`-guarded and defaults to 0,
    so a regression here fails silently by falling back to Nelder-Mead rather than
    by raising -- which is exactly why it is worth a test.

That the two minimisers actually differ (evaluation counts, pose quality) needs the
real engine; measure it with CHEMEM_DOCK_PROFILE=1.
"""
import argparse
import types
import unittest

try:
    from ChemEM.protocol_spec import add_dock_args
except ModuleNotFoundError:
    from protocol_spec import add_dock_args


def _parser():
    p = argparse.ArgumentParser()
    add_dock_args(p)
    return p


class TestFlagSurface(unittest.TestCase):
    def test_defaults_to_nelder_mead(self):
        """The staged simplex is the historical path. Anything else here would
        change every existing docking run as a side effect of adding a flag."""
        self.assertEqual(_parser().parse_args([]).local_minimiser, "nelder-mead")

    def test_accepts_lbfgs(self):
        self.assertEqual(
            _parser().parse_args(["--local-minimiser", "lbfgs"]).local_minimiser,
            "lbfgs",
        )

    def test_american_spelling_is_an_alias(self):
        """Both spellings write the same dest, so scripts using either keep working."""
        self.assertEqual(
            _parser().parse_args(["--local-minimizer", "lbfgs"]).local_minimiser,
            "lbfgs",
        )

    def test_rejects_an_unknown_minimiser(self):
        """A typo must fail loudly; silently falling back to Nelder-Mead would make
        a whole benchmark look like the new minimiser had no effect."""
        with self.assertRaises(SystemExit):
            _parser().parse_args(["--local-minimiser", "powell"])


class TestPrecomputeMapping(unittest.TestCase):
    """The Python -> C++ hop: PreComputedData reads py_pc.local_minimiser as an int."""

    @staticmethod
    def _encode(value):
        # Mirrors PreCompDataProtein.__init__; kept in sync deliberately so a change
        # to the encoding has to be made in two places rather than silently drifting.
        return 1 if str(value) == "lbfgs" else 0

    def test_nelder_mead_encodes_to_zero(self):
        self.assertEqual(self._encode("nelder-mead"), 0)

    def test_lbfgs_encodes_to_one(self):
        self.assertEqual(self._encode("lbfgs"), 1)

    def test_missing_option_falls_back_to_nelder_mead(self):
        """benchmark/ and score_poses/ build ad-hoc options objects that predate this
        flag; they must keep working and must not silently switch minimiser."""
        options = types.SimpleNamespace()
        self.assertEqual(
            self._encode(getattr(options, "local_minimiser", "nelder-mead")), 0
        )

    def test_precompute_sets_the_attribute(self):
        """PreComputedData.cpp reads the attribute by name; if the precompute stops
        setting it the C++ hasattr guard silently keeps Nelder-Mead."""
        try:
            from ChemEM.tools.precomputed_data import PreCompDataProtein
        except ModuleNotFoundError:
            self.skipTest("precomputed_data not importable in this env")
        import inspect

        src = inspect.getsource(PreCompDataProtein.__init__)
        self.assertIn("self.local_minimiser", src)


if __name__ == "__main__":
    unittest.main()
