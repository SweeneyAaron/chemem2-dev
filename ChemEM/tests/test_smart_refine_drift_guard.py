import argparse
import importlib
import types
import unittest

import numpy as np

try:
    from ChemEM.protocols.smart_refine_2 import smart_refine as sr
except ModuleNotFoundError:
    from protocols.smart_refine_2 import smart_refine as sr

try:
    from ChemEM.protocols.mapQ_score.mapq_utils import MapGrid
except ModuleNotFoundError:
    from protocols.mapQ_score.mapq_utils import MapGrid


def _blob_mapgrid():
    """10^3 grid (apix 1, origin 0) with a high-density cube at index 2..5."""
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[2:5, 2:5, 2:5] = 10.0  # (z, y, x)
    return MapGrid(data=data, origin_xyz=np.zeros(3), apix_xyz=np.ones(3))


def _result(best_raw, *, best_coords, initial_raw=0.0):
    """Minimal stand-in for FitInMapResult covering the fields the guard reads."""
    return types.SimpleNamespace(
        best_raw_score=float(best_raw),
        initial_raw_score=float(initial_raw),
        best_clash_penalty=0.0,
        initial_clash_penalty=0.0,
        best_clash_count=0,
        initial_clash_count=0,
        best_objective=float(best_raw),
        best_coords_A=np.asarray(best_coords, dtype=float),
    )


def _rl_with_guard(cfg, *, anchor_centroid=None, mapgrid=None, threshold=None, coverage=None):
    return types.SimpleNamespace(
        _drift_guard=cfg,
        _drift_anchor_centroid=anchor_centroid,
        _drift_mapgrid=mapgrid,
        _drift_threshold=threshold,
        _drift_anchor_coverage=coverage,
    )


class TestDensityCoverage(unittest.TestCase):
    def test_inside_blob_full_coverage(self):
        mg = _blob_mapgrid()
        mean, std = mg.stats()
        thr = mean + std
        inside = np.array([[3.0, 3.0, 3.0], [3.5, 3.0, 3.0]])
        self.assertEqual(sr._density_coverage(inside, mg, thr), 1.0)

    def test_outside_blob_zero_coverage(self):
        mg = _blob_mapgrid()
        mean, std = mg.stats()
        thr = mean + std
        outside = np.array([[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]])
        self.assertEqual(sr._density_coverage(outside, mg, thr), 0.0)


class TestCentroidPenalty(unittest.TestCase):
    def setUp(self):
        self.cfg = sr.DriftGuardConfig(
            centroid_trust=True, centroid_radius_A=2.0, centroid_k=0.05
        )

    def test_within_radius_no_penalty(self):
        coords = np.array([[1.0, 0, 0], [-1.0, 0, 0]])  # centroid at origin
        self.assertEqual(sr._centroid_penalty(coords, np.zeros(3), self.cfg), 0.0)

    def test_beyond_radius_linear(self):
        coords = np.array([[5.0, 0, 0], [5.0, 0, 0]])  # centroid 5 -> (5-2)*0.05
        self.assertAlmostEqual(
            sr._centroid_penalty(coords, np.zeros(3), self.cfg), 0.15
        )

    def test_disabled_is_zero(self):
        coords = np.array([[5.0, 0, 0], [5.0, 0, 0]])
        off = sr.DriftGuardConfig()
        self.assertEqual(sr._centroid_penalty(coords, np.zeros(3), off), 0.0)


class TestDriftAssess(unittest.TestCase):
    def test_no_guard_is_noop(self):
        rl = types.SimpleNamespace(_drift_guard=None)
        pen, ok = sr._drift_assess(rl, np.zeros((2, 3)))
        self.assertEqual((pen, ok), (0.0, True))

    def test_envelope_fail_when_leaving_blob(self):
        mg = _blob_mapgrid()
        mean, std = mg.stats()
        thr = mean + std
        cfg = sr.DriftGuardConfig(envelope_gate=True, envelope_slack=0.1)
        rl = _rl_with_guard(
            cfg, mapgrid=mg, threshold=thr, coverage=1.0,
        )
        # Pose fully outside the blob -> coverage 0 < 1.0 - 0.1.
        _, ok = sr._drift_assess(rl, np.array([[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]]))
        self.assertFalse(ok)
        # Pose still in the blob -> passes.
        _, ok2 = sr._drift_assess(rl, np.array([[3.0, 3.0, 3.0], [3.5, 3.0, 3.0]]))
        self.assertTrue(ok2)


class TestShouldAcceptWithGuard(unittest.TestCase):
    def test_envelope_rejects_even_if_raw_improves(self):
        mg = _blob_mapgrid()
        mean, std = mg.stats()
        thr = mean + std
        cfg = sr.DriftGuardConfig(envelope_gate=True, envelope_slack=0.1)
        rl = _rl_with_guard(cfg, mapgrid=mg, threshold=thr, coverage=1.0)
        # Big raw improvement but the pose left the blob.
        res = _result(0.9, initial_raw=0.1, best_coords=[[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]])
        self.assertFalse(sr.should_accept_refinement(res, refine_ligand=rl))
        # Same numbers without a guard would accept.
        self.assertTrue(sr.should_accept_refinement(res, refine_ligand=None))

    def test_centroid_penalty_raises_acceptance_bar(self):
        cfg = sr.DriftGuardConfig(
            centroid_trust=True, centroid_radius_A=1.0, centroid_k=1.0
        )
        rl = _rl_with_guard(cfg, anchor_centroid=np.zeros(3))
        # Tiny raw gain (0.05) but centroid moved to ~5 -> penalty (5-1)*1 = 4.
        res = _result(0.15, initial_raw=0.10, best_coords=[[5.0, 0, 0], [5.0, 0, 0]])
        self.assertFalse(sr.should_accept_refinement(res, refine_ligand=rl))
        # Without the guard the 0.05 gain is accepted.
        self.assertTrue(sr.should_accept_refinement(res, refine_ligand=None))


class TestPickBestLigandWithGuard(unittest.TestCase):
    def test_centroid_penalty_prefers_local_candidate(self):
        cfg = sr.DriftGuardConfig(
            centroid_trust=True, centroid_radius_A=1.0, centroid_k=1.0
        )
        rl_local = _rl_with_guard(cfg, anchor_centroid=np.zeros(3))
        rl_far = _rl_with_guard(cfg, anchor_centroid=np.zeros(3))
        local = _result(0.50, best_coords=[[0.0, 0, 0], [0.0, 0, 0]])      # pen 0
        far = _result(0.60, best_coords=[[10.0, 0, 0], [10.0, 0, 0]])      # pen (10-1)=9
        # far has higher raw (0.60 > 0.50) but its penalty (9) sinks it.
        winner = sr._pick_best_ligand([(rl_local, local), (rl_far, far)])
        self.assertIs(winner, rl_local)
        # Without a guard the higher-raw far candidate would win.
        rl_a = types.SimpleNamespace(_drift_guard=None)
        rl_b = types.SimpleNamespace(_drift_guard=None)
        self.assertIs(
            sr._pick_best_ligand([(rl_a, local), (rl_b, far)]), rl_b
        )

    def test_envelope_drops_out_of_blob_candidate(self):
        mg = _blob_mapgrid()
        mean, std = mg.stats()
        thr = mean + std
        cfg = sr.DriftGuardConfig(envelope_gate=True, envelope_slack=0.1)
        rl_in = _rl_with_guard(cfg, mapgrid=mg, threshold=thr, coverage=1.0)
        rl_out = _rl_with_guard(cfg, mapgrid=mg, threshold=thr, coverage=1.0)
        good = _result(0.40, best_coords=[[3.0, 3.0, 3.0], [3.5, 3.0, 3.0]])  # in blob
        bad = _result(0.95, best_coords=[[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]])   # out, higher raw
        winner = sr._pick_best_ligand([(rl_in, good), (rl_out, bad)])
        self.assertIs(winner, rl_in)

    def test_no_guard_is_noop_picks_highest_raw(self):
        rl_a = types.SimpleNamespace(_drift_guard=None)
        rl_b = types.SimpleNamespace(_drift_guard=None)
        a = _result(0.40, best_coords=[[0.0, 0, 0]])
        b = _result(0.70, best_coords=[[9.0, 0, 0]])
        self.assertIs(sr._pick_best_ligand([(rl_a, a), (rl_b, b)]), rl_b)


class TestDriftGuardCliFlags(unittest.TestCase):
    def _parser(self):
        protocol_spec = importlib.import_module("protocol_spec")
        parser = argparse.ArgumentParser()
        protocol_spec.add_smart_ligand_refine2_args(parser)
        return parser

    def test_defaults(self):
        # Guards are ON by default (tight radius / strong k) so SR2 resists
        # drift into neighbouring density without an explicit flag.
        ns = self._parser().parse_args([])
        self.assertTrue(ns.sr2_centroid_trust)
        self.assertTrue(ns.sr2_envelope_gate)
        self.assertAlmostEqual(ns.sr2_centroid_trust_radius, 5.0)
        self.assertAlmostEqual(ns.sr2_centroid_trust_k, 0.4)
        self.assertAlmostEqual(ns.sr2_envelope_threshold_sigma, 1.0)
        self.assertAlmostEqual(ns.sr2_envelope_slack, 0.15)

    def test_disable_flags(self):
        ns = self._parser().parse_args(
            ["--no-sr2-centroid-trust", "--no-sr2-envelope-gate"]
        )
        self.assertFalse(ns.sr2_centroid_trust)
        self.assertFalse(ns.sr2_envelope_gate)

    def test_overrides(self):
        ns = self._parser().parse_args(
            [
                "--sr2-centroid-trust",
                "--sr2-centroid-trust-radius", "3.0",
                "--sr2-centroid-trust-k", "0.2",
                "--sr2-envelope-gate",
                "--sr2-envelope-threshold-sigma", "1.5",
                "--sr2-envelope-slack", "0.05",
            ]
        )
        self.assertTrue(ns.sr2_centroid_trust)
        self.assertTrue(ns.sr2_envelope_gate)
        self.assertAlmostEqual(ns.sr2_centroid_trust_radius, 3.0)
        self.assertAlmostEqual(ns.sr2_centroid_trust_k, 0.2)
        self.assertAlmostEqual(ns.sr2_envelope_threshold_sigma, 1.5)
        self.assertAlmostEqual(ns.sr2_envelope_slack, 0.05)


if __name__ == "__main__":
    unittest.main()
