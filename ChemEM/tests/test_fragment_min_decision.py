"""Tests for the fragment-pinned minimiser's pure decision logic and CLI flags.

The OpenMM-driven `fragment_pinned_polish_ligand` itself needs a real density map
and is validated server-side; here we unit-test the accept/pin decision and the
flag defaults (no OpenMM required).
"""
import argparse
import importlib
import unittest

try:
    from ChemEM.protocols.smart_refine_2 import smart_refine as sr
except ModuleNotFoundError:
    from protocols.smart_refine_2 import smart_refine as sr


class TestFragmentMinDecision(unittest.TestCase):
    def test_all_stable_accepts(self):
        before = [0.8, 0.7, 0.6]
        after = [0.8, 0.7, 0.6]
        self.assertEqual(
            sr._fragment_min_decision(before, after, 0.70, 0.70), ("accept", [])
        )

    def test_overall_up_small_block_drop_accepts(self):
        # Overall improves; one block drops 0.05 (<= 0.1 tol) -> accept.
        before = [0.8, 0.7, 0.6]
        after = [0.9, 0.65, 0.6]
        self.assertEqual(
            sr._fragment_min_decision(before, after, 0.70, 0.75, block_tol=0.1),
            ("accept", []),
        )

    def test_overall_up_large_block_drop_pins_that_block(self):
        # Overall improves but block 1 drops 0.3 (> tol) -> pin block 1.
        before = [0.8, 0.7, 0.6]
        after = [0.95, 0.40, 0.6]
        decision, rows = sr._fragment_min_decision(
            before, after, 0.70, 0.78, block_tol=0.1
        )
        self.assertEqual(decision, "pin")
        self.assertEqual(rows, [1])

    def test_two_blocks_beyond_tol_pins_both(self):
        before = [0.8, 0.7, 0.6]
        after = [0.55, 0.45, 0.6]  # blocks 0 and 1 drop > 0.1
        decision, rows = sr._fragment_min_decision(
            before, after, 0.70, 0.72, block_tol=0.1
        )
        self.assertEqual(decision, "pin")
        self.assertEqual(sorted(rows), [0, 1])

    def test_overall_down_within_block_tol_pins_worst(self):
        # Overall regressed but every block within tol -> pin the worst dropper.
        before = [0.80, 0.70, 0.60]
        after = [0.78, 0.64, 0.59]  # drops 0.02, 0.06, 0.01 (all <= 0.1)
        decision, rows = sr._fragment_min_decision(
            before, after, 0.70, 0.60, block_tol=0.1
        )
        self.assertEqual(decision, "pin")
        self.assertEqual(rows, [1])  # block 1 dropped the most (0.06)


class TestMinimiserCliFlags(unittest.TestCase):
    def _parser(self):
        protocol_spec = importlib.import_module("protocol_spec")
        parser = argparse.ArgumentParser()
        protocol_spec.add_smart_ligand_refine2_args(parser)
        return parser

    def test_defaults(self):
        ns = self._parser().parse_args([])
        self.assertEqual(ns.sr2_minimiser, "standard")
        self.assertAlmostEqual(ns.sr2_fragmin_block_tol, 0.1)
        self.assertEqual(ns.sr2_fragmin_max_iters, 4)

    def test_fragment_overrides(self):
        ns = self._parser().parse_args(
            ["--sr2-minimiser", "fragment", "--sr2-fragmin-block-tol", "0.2",
             "--sr2-fragmin-max-iters", "6"]
        )
        self.assertEqual(ns.sr2_minimiser, "fragment")
        self.assertAlmostEqual(ns.sr2_fragmin_block_tol, 0.2)
        self.assertEqual(ns.sr2_fragmin_max_iters, 6)


if __name__ == "__main__":
    unittest.main()
