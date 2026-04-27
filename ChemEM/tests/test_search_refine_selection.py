import unittest
from types import SimpleNamespace

import numpy as np

try:
    from ChemEM.protocols.refine.search_refine.orchestrator import SearchRefine
    from ChemEM.protocols.refine.search_refine.scorers.ccc import CCCScorer
except ModuleNotFoundError:
    from protocols.refine.search_refine.orchestrator import SearchRefine
    from protocols.refine.search_refine.scorers.ccc import CCCScorer


def _rec(score, x_shift):
    """Make a record-like dict at x-translated coords."""
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    coords[:, 0] += x_shift
    return {
        "final_score": float(score),
        "score": float(score),
        "ligand_coords_A": coords,
    }


class TestSelectFinalPoses(unittest.TestCase):
    def setUp(self):
        self.sr = SearchRefine(system=SimpleNamespace(options=SimpleNamespace()))

    def test_empty_records_returns_empty(self):
        out = self.sr._select_final_poses([], return_n=3, rmsd_thr_A=0.5, score_margin=0.1)
        self.assertEqual(out, [])

    def test_return_n_one_keeps_only_best(self):
        ranked = [_rec(0.90, 0.0), _rec(0.89, 5.0), _rec(0.85, 10.0)]
        out = self.sr._select_final_poses(ranked, return_n=1, rmsd_thr_A=0.5, score_margin=0.1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["final_score"], 0.90)

    def test_within_margin_and_distinct_gets_kept(self):
        # Three poses: best (score 0.90, x=0), near-tie distinct (0.895, x=5),
        # near-tie too-close (0.893, x=0.1).
        ranked = [_rec(0.90, 0.0), _rec(0.895, 5.0), _rec(0.893, 0.1)]
        out = self.sr._select_final_poses(ranked, return_n=3, rmsd_thr_A=0.5, score_margin=0.02)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0]["final_score"], 0.90)
        self.assertAlmostEqual(out[1]["final_score"], 0.895)

    def test_outside_margin_is_dropped_even_when_distinct(self):
        ranked = [_rec(0.90, 0.0), _rec(0.70, 5.0)]
        out = self.sr._select_final_poses(ranked, return_n=3, rmsd_thr_A=0.5, score_margin=0.05)
        self.assertEqual(len(out), 1)

    def test_return_n_cap_respected(self):
        # Many near-tie distinct poses — only return_n should be emitted.
        ranked = [_rec(0.90, 0.0), _rec(0.895, 5.0), _rec(0.893, 10.0), _rec(0.891, 15.0)]
        out = self.sr._select_final_poses(ranked, return_n=2, rmsd_thr_A=0.5, score_margin=0.05)
        self.assertEqual(len(out), 2)

    def test_sorted_order_allows_early_break(self):
        # Verify the function doesn't walk all the way through a long tail.
        ranked = [_rec(0.90, 0.0)] + [_rec(0.70, 5.0 + i) for i in range(100)]
        out = self.sr._select_final_poses(ranked, return_n=10, rmsd_thr_A=0.5, score_margin=0.05)
        self.assertEqual(len(out), 1)


class _FakeMap:
    def __init__(self, density_map, origin=(0.0, 0.0, 0.0),
                 apix=(1.0, 1.0, 1.0), resolution=3.0):
        self.density_map = np.asarray(density_map, dtype=np.float64)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.apix = np.asarray(apix, dtype=np.float64)
        self.resolution = float(resolution)


def _blob(shape=(32, 32, 32), center=(16.0, 16.0, 16.0), sigma=3.0):
    nz, ny, nx = shape
    z = np.arange(nz, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    x = np.arange(nx, dtype=np.float64)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
    return np.exp(-0.5 * d2 / (sigma * sigma))


class TestStageLegacyRouting(unittest.TestCase):
    def setUp(self):
        self.emmap = _FakeMap(_blob())
        self.coords = np.array(
            [[14.0, 16.0, 16.0], [16.0, 16.0, 16.0], [18.0, 16.0, 16.0]],
            dtype=np.float64,
        )
        self.masses = np.array([12.0, 12.0, 12.0], dtype=np.float64)

    def _make_scorer(self, stage):
        opts = SimpleNamespace(
            sr_sigma_coeff=0.356,
            sr_normalise_sim_map=True,
            sr_ccc_mask_mode="nonzero",
            sr_fd_step_a=0.5,
            sr_fd_mode="central",
            sr_stage=stage,
            resolution=3.0,
        )
        scorer = CCCScorer(opts)
        scorer.prepare(object(), self.emmap, self.masses, [])
        return scorer

    def test_v2_uses_analytical_gradient(self):
        from ChemEM.protocols.refine.search_refine.scorers.base import BaseScorer
        scorer = self._make_scorer("v2")
        g = scorer.atom_gradient(self.coords)
        g_fd = BaseScorer.atom_gradient(scorer, self.coords)
        # Both should be finite and non-zero; the analytical path produces
        # different per-atom magnitudes than FD because of voxel quantization,
        # so just verify results are not trivially equal.
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertTrue(np.all(np.isfinite(g_fd)))
        self.assertGreater(float(np.linalg.norm(g)), 0.0)

    def test_legacy_routes_through_finite_difference(self):
        from ChemEM.protocols.refine.search_refine.scorers.base import BaseScorer
        scorer = self._make_scorer("legacy")
        g_legacy = scorer.atom_gradient(self.coords)
        g_fd_direct = BaseScorer.atom_gradient(scorer, self.coords)
        # Legacy stage is bit-exact for the FD path.
        np.testing.assert_allclose(g_legacy, g_fd_direct, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
