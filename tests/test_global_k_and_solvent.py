"""Tests for the shared --global-k and --implicit-solvent flags.

Both use a None sentinel: unset means every minimiser keeps the value it used
before the flags existed, so these tests pin down both the override path and the
"changes nothing" path.
"""

import unittest
from types import SimpleNamespace

try:
    from openmm import app
    from ChemEM.protocols.core.simulation import (
        resolve_global_k,
        resolve_implicit_solvent,
    )
    from ChemEM.parsers.parse_forcefield import (
        AMBER_FF,
        IMPLICIT_SOLVENT_MODELS,
    )
    from ChemEM.protocols.refine.lining_refine import _resolve_lr_global_k
    from ChemEM.protocols.refine.ion_fixer import IonFixer
    from ChemEM.__main__ import build_parser
    _HAS_OPENMM = True
except ModuleNotFoundError:
    app = None
    _HAS_OPENMM = False


@unittest.skipUnless(_HAS_OPENMM, "OpenMM is required for these tests")
class TestResolveGlobalK(unittest.TestCase):
    def test_unset_keeps_caller_default(self):
        self.assertEqual(resolve_global_k(SimpleNamespace(global_k=None), 150.0), 150.0)

    def test_missing_attribute_keeps_caller_default(self):
        self.assertEqual(resolve_global_k(SimpleNamespace(), 150.0), 150.0)

    def test_supplied_value_wins(self):
        self.assertEqual(resolve_global_k(SimpleNamespace(global_k=500.0), 150.0), 500.0)

    def test_zero_is_honoured_not_treated_as_unset(self):
        self.assertEqual(resolve_global_k(SimpleNamespace(global_k=0.0), 150.0), 0.0)

    def test_returns_float(self):
        self.assertIsInstance(resolve_global_k(SimpleNamespace(global_k=500), 150.0), float)


@unittest.skipUnless(_HAS_OPENMM, "OpenMM is required for these tests")
class TestResolveImplicitSolvent(unittest.TestCase):
    def test_unset_keeps_caller_default(self):
        self.assertIs(
            resolve_implicit_solvent(SimpleNamespace(implicit_solvent=None), app.GBn2),
            app.GBn2,
        )
        # ion_fixer / export pass None as their default: vacuum stays vacuum.
        self.assertIsNone(
            resolve_implicit_solvent(SimpleNamespace(implicit_solvent=None), None)
        )

    def test_missing_attribute_keeps_caller_default(self):
        self.assertIs(resolve_implicit_solvent(SimpleNamespace(), app.GBn2), app.GBn2)

    def test_every_choice_maps_to_its_constant(self):
        expected = {
            "none": None,
            "hct": app.HCT,
            "obc1": app.OBC1,
            "obc2": app.OBC2,
            "gbn": app.GBn,
            "gbn2": app.GBn2,
        }
        for name, constant in expected.items():
            with self.subTest(model=name):
                got = resolve_implicit_solvent(
                    SimpleNamespace(implicit_solvent=name), app.GBn2
                )
                self.assertIs(got, constant)

    def test_name_is_case_insensitive(self):
        self.assertIs(
            resolve_implicit_solvent(SimpleNamespace(implicit_solvent="GBn2"), None),
            app.GBn2,
        )

    def test_none_overrides_a_gb_default(self):
        """--implicit-solvent none must be able to turn GB OFF for refine/slr2."""
        self.assertIsNone(
            resolve_implicit_solvent(SimpleNamespace(implicit_solvent="none"), app.GBn2)
        )

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_implicit_solvent(SimpleNamespace(implicit_solvent="bogus"), None)
        self.assertIn("Unknown implicit solvent model", str(ctx.exception))

    def test_model_table_matches_the_forcefield_xml_list(self):
        """IMPLICIT_SOLVENT_MODELS and AMBER_FF.supported_implicit must not drift."""
        from_xml = {
            name.split("/")[-1].removesuffix(".xml")
            for name in AMBER_FF.supported_implicit
        }
        from_map = set(IMPLICIT_SOLVENT_MODELS) - {"none"}
        self.assertEqual(from_map, from_xml)


@unittest.skipUnless(_HAS_OPENMM, "OpenMM is required for these tests")
class TestLiningRefinePrecedence(unittest.TestCase):
    def test_lr_flag_wins_over_shared_flag(self):
        opts = SimpleNamespace(lr_global_k=900.0, global_k=300.0)
        self.assertEqual(_resolve_lr_global_k(opts), 900.0)

    def test_shared_flag_used_when_lr_unset(self):
        opts = SimpleNamespace(lr_global_k=None, global_k=300.0)
        self.assertEqual(_resolve_lr_global_k(opts), 300.0)

    def test_both_unset_falls_back_to_historical_default(self):
        opts = SimpleNamespace(lr_global_k=None, global_k=None)
        self.assertEqual(_resolve_lr_global_k(opts), 150.0)


class _CapturingStructure:
    """Records the kwargs IonFixer.build_system passes to createSystem."""

    def __init__(self):
        self.calls = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.impropers = []

    def createSystem(self, **kwargs):
        self.calls.append(kwargs)
        return object()


@unittest.skipUnless(_HAS_OPENMM, "OpenMM is required for these tests")
class TestIonFixerBuildSystemSolvent(unittest.TestCase):
    @staticmethod
    def _fixer(implicit_solvent=None):
        system = SimpleNamespace(
            options=SimpleNamespace(implicit_solvent=implicit_solvent),
        )
        fixer = IonFixer(system)
        fixer.selected_structure = _CapturingStructure()
        return fixer

    def test_default_is_vacuum_with_rigid_water(self):
        fixer = self._fixer()
        fixer.build_system()

        kwargs = fixer.selected_structure.calls[0]
        self.assertTrue(kwargs["rigidWater"])
        self.assertNotIn("implicitSolvent", kwargs)
        self.assertIs(kwargs["nonbondedMethod"], app.NoCutoff)
        self.assertIs(kwargs["constraints"], app.HBonds)

    def test_gbn2_adds_implicit_solvent_and_drops_rigid_water(self):
        fixer = self._fixer(implicit_solvent="gbn2")
        fixer.build_system()

        kwargs = fixer.selected_structure.calls[0]
        self.assertIs(kwargs["implicitSolvent"], app.GBn2)
        self.assertNotIn("rigidWater", kwargs)

    def test_explicit_none_is_still_vacuum(self):
        fixer = self._fixer(implicit_solvent="none")
        fixer.build_system()

        kwargs = fixer.selected_structure.calls[0]
        self.assertTrue(kwargs["rigidWater"])
        self.assertNotIn("implicitSolvent", kwargs)


@unittest.skipUnless(_HAS_OPENMM, "OpenMM is required for these tests")
class TestCliFlags(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_both_flags_default_to_none(self):
        args = self.parser.parse_args(["cfg", "--refine"])
        self.assertIsNone(args.global_k)
        self.assertIsNone(args.implicit_solvent)
        self.assertIsNone(args.lr_global_k)

    def test_global_k_is_parsed_as_float(self):
        args = self.parser.parse_args(["cfg", "--refine", "--global-k", "500"])
        self.assertEqual(args.global_k, 500.0)

    def test_flags_are_available_to_every_protocol(self):
        """Both live in the shared group, so protocol choice must not matter."""
        for flag in ("--refine", "--ion-fixer", "--dock", "--lining-refine"):
            with self.subTest(protocol=flag):
                args = self.parser.parse_args(
                    ["cfg", flag, "--global-k", "500", "--implicit-solvent", "gbn2"]
                )
                self.assertEqual(args.global_k, 500.0)
                self.assertEqual(args.implicit_solvent, "gbn2")

    def test_invalid_solvent_choice_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["cfg", "--refine", "--implicit-solvent", "bogus"])


if __name__ == "__main__":
    unittest.main()
