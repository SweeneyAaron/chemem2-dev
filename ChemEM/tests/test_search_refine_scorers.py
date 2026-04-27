import unittest
from types import SimpleNamespace

import numpy as np

try:
    from ChemEM.protocols.refine.search_refine.scorers.sci import SCIScorer
    from ChemEM.protocols.refine.search_refine.scorers.ccc import CCCScorer
    from ChemEM.protocols.refine.search_refine.scorers.mi import MIScorer
except ModuleNotFoundError:
    from protocols.refine.search_refine.scorers.sci import SCIScorer
    from protocols.refine.search_refine.scorers.ccc import CCCScorer
    from protocols.refine.search_refine.scorers.mi import MIScorer


class _FakeMap:
    """Minimal EMMap-like fake with just the fields scorers read."""

    def __init__(self, density_map, origin=(0.0, 0.0, 0.0),
                 apix=(1.0, 1.0, 1.0), resolution=3.0):
        self.density_map = np.asarray(density_map, dtype=np.float64)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.apix = np.asarray(apix, dtype=np.float64)
        self.resolution = float(resolution)


def _blob(shape=(32, 32, 32), center_xyz=(16.0, 16.0, 16.0), sigma=2.5):
    nz, ny, nx = shape
    z = np.arange(nz, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    x = np.arange(nx, dtype=np.float64)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    d2 = (xx - center_xyz[0]) ** 2 + (yy - center_xyz[1]) ** 2 + (zz - center_xyz[2]) ** 2
    return np.exp(-0.5 * d2 / (sigma * sigma))


def _options(**kwargs):
    defaults = dict(
        sr_sigma_coeff=0.356,
        sr_normalise_sim_map=True,
        sr_use_amp_eq=False,
        sr_sci_sigma=1.0,
        sr_w0=1.0,
        sr_w1=1.0,
        sr_w2=1.0,
        sr_sci_eps=1e-8,
        sr_ccc_mask_mode="nonzero",
        sr_mi_nbins=64,
        sr_mi_normalized=False,
        sr_mi_fd_step_a=0.1,
        sr_fd_step_a=0.2,
        sr_fd_mode="central",
        resolution=3.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class _FakeEnv:
    """Only used by BaseScorer.prepare — the grid-based scorers don't touch it."""
    pass


class TestGridScorers(unittest.TestCase):
    def setUp(self):
        self.shape = (24, 24, 24)
        self.center = np.array([12.0, 12.0, 12.0])
        self.emmap = _FakeMap(
            density_map=_blob(shape=self.shape, center_xyz=tuple(self.center), sigma=2.5),
            origin=(0.0, 0.0, 0.0),
            apix=(1.0, 1.0, 1.0),
            resolution=3.0,
        )

        # 3 heavy atoms clustered at the blob centre (coords in Å).
        self.coords_aligned = np.array(
            [[12.0, 12.0, 12.0], [12.5, 12.0, 12.0], [12.0, 12.5, 12.0]],
            dtype=np.float64,
        )
        self.coords_shifted = self.coords_aligned + np.array([2.0, 0.0, 0.0])
        self.heavy_masses = np.array([12.0, 12.0, 12.0], dtype=np.float64)

    def _prepare(self, scorer_cls, **opt_overrides):
        scorer = scorer_cls(_options(**opt_overrides))
        scorer.prepare(_FakeEnv(), self.emmap, self.heavy_masses, [])
        return scorer

    def test_sci_aligned_beats_shifted(self):
        scorer = self._prepare(SCIScorer)
        s_a = scorer.score(self.coords_aligned)
        s_s = scorer.score(self.coords_shifted)
        self.assertTrue(np.isfinite(s_a))
        self.assertTrue(np.isfinite(s_s))
        self.assertGreater(s_a, s_s)

    def test_ccc_aligned_beats_shifted(self):
        scorer = self._prepare(CCCScorer)
        s_a = scorer.score(self.coords_aligned)
        s_s = scorer.score(self.coords_shifted)
        self.assertTrue(np.isfinite(s_a))
        self.assertTrue(np.isfinite(s_s))
        self.assertGreater(s_a, s_s)

    def test_mi_aligned_beats_shifted(self):
        scorer = self._prepare(MIScorer)
        s_a = scorer.score(self.coords_aligned)
        s_s = scorer.score(self.coords_shifted)
        self.assertTrue(np.isfinite(s_a))
        self.assertTrue(np.isfinite(s_s))
        self.assertGreater(s_a, s_s)

    def test_sci_fd_gradient_points_uphill(self):
        scorer = self._prepare(SCIScorer, sr_fd_step_a=0.3)
        # Place atoms +x of the true centre; ∇score w.r.t. x should be negative
        # (moving +x decreases score, so gradient points −x back toward centre).
        displaced = self.coords_aligned + np.array([1.5, 0.0, 0.0])
        grad = scorer.atom_gradient(displaced)
        self.assertEqual(grad.shape, displaced.shape)
        # Net x-gradient should be strictly negative.
        self.assertLess(float(np.sum(grad[:, 0])), 0.0)

    def test_ccc_analytical_gradient_points_uphill(self):
        scorer = self._prepare(CCCScorer)
        displaced = self.coords_aligned + np.array([1.5, 0.0, 0.0])
        grad = scorer.atom_gradient(displaced)
        self.assertEqual(grad.shape, displaced.shape)
        self.assertTrue(np.all(np.isfinite(grad)))
        # Analytical gradient: moving +x lowers CCC, so ∂CCC/∂x < 0.
        self.assertLess(float(np.sum(grad[:, 0])), 0.0)
        # And y, z components should be ~zero for a pure-x displacement
        # (symmetry of the isotropic blob). Allow small numerical noise.
        self.assertLess(float(abs(np.sum(grad[:, 1]))), abs(float(np.sum(grad[:, 0]))))
        self.assertLess(float(abs(np.sum(grad[:, 2]))), abs(float(np.sum(grad[:, 0]))))

    def test_ccc_analytical_vs_fd_agreement(self):
        # Build a finer grid so the central-FD step reliably crosses voxel
        # boundaries despite the nearest-voxel rasterization in the simulator.
        shape = (40, 40, 40)
        center = np.array([20.0, 20.0, 20.0])
        apix = (0.5, 0.5, 0.5)
        emmap = _FakeMap(
            density_map=_blob(shape=shape, center_xyz=tuple(center), sigma=4.0),
            origin=(0.0, 0.0, 0.0),
            apix=apix,
            resolution=2.0,
        )
        coords = np.array(
            [[10.5, 10.0, 10.0], [11.2, 10.3, 10.1], [10.0, 10.8, 10.4]],
            dtype=np.float64,
        )
        masses = np.array([12.0, 12.0, 12.0], dtype=np.float64)

        analytical = CCCScorer(_options(sr_fd_step_a=0.5))
        analytical.prepare(_FakeEnv(), emmap, masses, [])
        g_ana = analytical.atom_gradient(coords)

        # Force the FD path by calling the base implementation directly.
        from ChemEM.protocols.refine.search_refine.scorers.base import BaseScorer
        g_fd = BaseScorer.atom_gradient(analytical, coords)

        self.assertEqual(g_ana.shape, g_fd.shape)
        self.assertTrue(np.all(np.isfinite(g_ana)))
        self.assertTrue(np.all(np.isfinite(g_fd)))

        # Net direction must agree: the sum of gradients over atoms (the
        # rigid-body direction) should be well-aligned between methods.
        net_ana = g_ana.sum(axis=0)
        net_fd = g_fd.sum(axis=0)
        norm = float(np.linalg.norm(net_ana) * np.linalg.norm(net_fd))
        self.assertGreater(norm, 1e-12)
        cos_net = float(np.dot(net_ana, net_fd) / norm)
        self.assertGreater(cos_net, 0.9)

        # Per-atom direction agreement: mean cosine similarity over atoms
        # with a nontrivial FD gradient should be > 0.7.
        cosines = []
        for i in range(coords.shape[0]):
            na = float(np.linalg.norm(g_ana[i]))
            nf = float(np.linalg.norm(g_fd[i]))
            if na < 1e-10 or nf < 1e-10:
                continue
            cosines.append(float(np.dot(g_ana[i], g_fd[i]) / (na * nf)))
        self.assertGreater(len(cosines), 0)
        self.assertGreater(float(np.mean(cosines)), 0.7)


if __name__ == "__main__":
    unittest.main()
