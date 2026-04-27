import unittest

import numpy as np

try:
    from ChemEM.protocols.refine.search_refine.diagnostic import (
        atom_fit_quality,
        classify_atoms,
        format_diagnostic,
    )
except ModuleNotFoundError:
    from protocols.refine.search_refine.diagnostic import (
        atom_fit_quality,
        classify_atoms,
        format_diagnostic,
    )


class _FakeMap:
    def __init__(self, density_map, origin=(0.0, 0.0, 0.0),
                 apix=(1.0, 1.0, 1.0), resolution=3.0):
        self.density_map = np.asarray(density_map, dtype=np.float64)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.apix = np.asarray(apix, dtype=np.float64)
        self.resolution = float(resolution)


def _gaussian_blob_map(shape, center_xyz, sigma):
    nz, ny, nx = shape
    z = np.arange(nz, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    x = np.arange(nx, dtype=np.float64)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    d2 = (
        (xx - center_xyz[0]) ** 2
        + (yy - center_xyz[1]) ** 2
        + (zz - center_xyz[2]) ** 2
    )
    return np.exp(-0.5 * d2 / (sigma * sigma))


class TestAtomFitQuality(unittest.TestCase):
    def setUp(self):
        self.shape = (32, 32, 32)
        self.center = (16.0, 16.0, 16.0)
        self.emmap = _FakeMap(
            density_map=_gaussian_blob_map(self.shape, self.center, sigma=3.0),
            origin=(0.0, 0.0, 0.0),
            apix=(1.0, 1.0, 1.0),
            resolution=3.0,
        )

    def test_classify_atoms_flags_vacuum_atom_as_bad(self):
        # Five atoms: 4 clustered in the density basin, 1 shoved out to the
        # edge of the map where density is essentially zero.
        coords = np.array(
            [
                [16.0, 16.0, 16.0],  # centre of blob
                [16.8, 16.0, 16.0],
                [15.5, 16.3, 16.0],
                [16.0, 16.0, 16.5],
                [3.0,  3.0,  3.0],   # vacuum
            ],
            dtype=np.float64,
        )
        # Dummy gradient — diagnostic should still classify by Q alone.
        grad = np.zeros_like(coords)

        fq = atom_fit_quality(coords, grad, self.emmap, sigma_ref=0.6)
        self.assertEqual(fq.q_score.shape, (5,))
        self.assertTrue(np.all(np.isfinite(fq.q_score)))
        # The vacuum atom's Q should be far below the in-basin atoms'.
        self.assertLess(fq.q_score[4], fq.q_score[0])
        self.assertLess(fq.q_score[4], 0.3)

        classification = classify_atoms(fq, q_good_thresh=0.5, q_bad_thresh=0.3)
        # Vacuum atom must be in bad_idx.
        self.assertIn(4, classification.bad_idx.tolist())
        # At least one in-basin atom must be in good_idx.
        self.assertTrue(any(i in classification.good_idx.tolist() for i in range(4)))

    def test_badness_scales_with_grad_norm(self):
        coords = np.array(
            [[16.0, 16.0, 16.0], [16.5, 16.0, 16.0]],
            dtype=np.float64,
        )
        # Atom 1 has a larger "wants to move" gradient than atom 0.
        grad = np.array([[0.1, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
        fq = atom_fit_quality(coords, grad, self.emmap)
        # With comparable Q scores, larger gradient => larger badness.
        if abs(fq.q_score[0] - fq.q_score[1]) < 0.15:
            self.assertGreater(fq.badness[1], fq.badness[0])

    def test_format_diagnostic_produces_string(self):
        coords = np.array([[16.0, 16.0, 16.0]], dtype=np.float64)
        grad = np.zeros_like(coords)
        fq = atom_fit_quality(coords, grad, self.emmap)
        cls = classify_atoms(fq)
        msg = format_diagnostic(fq, cls)
        self.assertIsInstance(msg, str)
        self.assertIn("q(", msg)
        self.assertIn("good=", msg)
        self.assertIn("bad=", msg)


if __name__ == "__main__":
    unittest.main()
