import unittest
import numpy as np

try:
    from ChemEM.protocols.core.sci_score import amplitude_equalize_pair, sci_score_3d
except ModuleNotFoundError:
    from protocols.core.sci_score import amplitude_equalize_pair, sci_score_3d


class TestSCIScore3D(unittest.TestCase):
    def _blob(self, shape=(32, 32, 32), center=(16.0, 16.0, 16.0), sigma=3.0):
        z = np.arange(shape[0], dtype=np.float64)
        y = np.arange(shape[1], dtype=np.float64)
        x = np.arange(shape[2], dtype=np.float64)
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
        return np.exp(-0.5 * d2 / (sigma * sigma))

    def test_perfect_alignment_scores_better_than_shifted(self):
        exp_map = self._blob()
        sim_perfect = exp_map.copy()
        sim_shift = np.roll(exp_map, shift=3, axis=2)

        sci_perfect, _ = sci_score_3d(exp_map, sim_perfect, use_amp_eq=False, sigma=1.0)
        sci_shift, _ = sci_score_3d(exp_map, sim_shift, use_amp_eq=False, sigma=1.0)

        self.assertTrue(np.isfinite(sci_perfect))
        self.assertTrue(np.isfinite(sci_shift))
        self.assertGreater(sci_perfect, sci_shift)

    def test_amplitude_equalization_outputs_finite_values(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=(20, 18, 16))
        b = rng.normal(size=(20, 18, 16))

        a_eq, b_eq = amplitude_equalize_pair(a, b)
        self.assertEqual(a_eq.shape, a.shape)
        self.assertEqual(b_eq.shape, b.shape)
        self.assertTrue(np.isfinite(a_eq).all())
        self.assertTrue(np.isfinite(b_eq).all())


if __name__ == "__main__":
    unittest.main()
