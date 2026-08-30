import unittest

import numpy as np

from ChemEM.protocols.smart_refine_2.shape_metrics import (
    density_shape_metrics,
    skeleton_graph_metrics,
)


def _ellipsoid(shape=(40, 40, 40), center=(20, 20, 20), radii=(4, 8, 3)):
    z = np.arange(shape[0], dtype=float)
    y = np.arange(shape[1], dtype=float)
    x = np.arange(shape[2], dtype=float)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    cz, cy, cx = [float(v) for v in center]
    rz, ry, rx = [float(v) for v in radii]
    value = (
        ((zz - cz) / rz) ** 2
        + ((yy - cy) / ry) ** 2
        + ((xx - cx) / rx) ** 2
    )
    return (value <= 1.0).astype(float)


def _line(shape=(32, 32, 32), *, axis=0, center=(16, 16, 16), start=7, stop=25):
    arr = np.zeros(shape, dtype=float)
    slc = [int(center[0]), int(center[1]), int(center[2])]
    slc[int(axis)] = slice(int(start), int(stop))
    arr[tuple(slc)] = 1.0
    return arr


def _t_shape(shape=(32, 32, 32)):
    arr = _line(shape, axis=0, center=(16, 16, 16), start=7, stop=25)
    arr[16, 16, 16:26] = 1.0
    return arr


class TestDensityShapeMetrics(unittest.TestCase):
    def test_same_shape_translated_or_rotated_scores_high(self):
        exp = _ellipsoid(radii=(4, 8, 3))
        shifted = _ellipsoid(center=(22, 19, 21), radii=(4, 8, 3))
        rotated = np.rot90(exp, k=1, axes=(1, 2))

        shifted_metrics = density_shape_metrics(exp, shifted)
        rotated_metrics = density_shape_metrics(exp, rotated)

        self.assertGreater(shifted_metrics["shape_zernike_similarity"], 0.85)
        self.assertGreater(rotated_metrics["shape_zernike_similarity"], 0.85)
        self.assertGreater(shifted_metrics["shape_spharm_similarity"], 0.75)
        self.assertGreater(rotated_metrics["shape_spharm_similarity"], 0.75)

    def test_different_shapes_score_lower_than_same_shape(self):
        exp = _ellipsoid(radii=(5, 5, 5))
        same = _ellipsoid(radii=(5, 5, 5))
        rod = _ellipsoid(radii=(2, 12, 2))

        same_metrics = density_shape_metrics(exp, same)
        rod_metrics = density_shape_metrics(exp, rod)

        self.assertGreater(
            same_metrics["shape_zernike_similarity"],
            rod_metrics["shape_zernike_similarity"] + 0.05,
        )
        self.assertGreater(
            same_metrics["shape_spharm_similarity"],
            rod_metrics["shape_spharm_similarity"] + 0.05,
        )

    def test_disconnected_noise_does_not_dominate_overlapping_component(self):
        ligand_shape = _ellipsoid(center=(14, 14, 14), radii=(3, 5, 3))
        far_noise = _ellipsoid(center=(29, 29, 29), radii=(5, 2, 2))
        exp = ligand_shape + far_noise
        sim = ligand_shape.copy()

        metrics = density_shape_metrics(exp, sim)

        self.assertEqual(metrics["shape_exp_component_count"], 2)
        self.assertEqual(metrics["shape_selected_component_mode"], "overlap")
        self.assertLess(metrics["shape_selected_component_fraction"], 1.0)
        self.assertGreater(metrics["shape_zernike_similarity"], 0.85)

    def test_no_overlap_fallback_records_nearest_component(self):
        exp = _ellipsoid(center=(8, 8, 8), radii=(2, 2, 2))
        exp += _ellipsoid(center=(31, 31, 31), radii=(2, 2, 2))
        sim = _ellipsoid(center=(26, 26, 26), radii=(2, 2, 2))

        metrics = density_shape_metrics(exp, sim)

        self.assertEqual(metrics["shape_exp_component_count"], 2)
        self.assertEqual(metrics["shape_selected_component_mode"], "nearest")
        self.assertEqual(metrics["shape_zernike_backend"], "internal")

    def test_skeleton_same_and_slightly_shifted_rods_score_high(self):
        exp = _line()
        shifted = _line(center=(16, 17, 16))

        metrics = skeleton_graph_metrics(exp, shifted, tolerance_A=1.5)

        self.assertEqual(metrics["density_skeleton_backend"], "skimage_skeletonize_proxy")
        self.assertGreater(metrics["density_skeleton_f1"], 0.9)
        self.assertGreater(metrics["density_skeleton_precision"], 0.9)
        self.assertGreater(metrics["density_skeleton_recall"], 0.9)

    def test_skeleton_different_shapes_score_lower_than_same_shape(self):
        sphere = _ellipsoid(shape=(32, 32, 32), center=(16, 16, 16), radii=(5, 5, 5))
        rod = _line()

        same = skeleton_graph_metrics(sphere, sphere)
        different = skeleton_graph_metrics(sphere, rod)

        self.assertGreater(
            same["density_skeleton_f1"],
            different["density_skeleton_f1"] + 0.2,
        )

    def test_skeleton_missing_branch_has_high_precision_low_recall(self):
        exp = _t_shape()
        sim = _line()

        metrics = skeleton_graph_metrics(exp, sim, tolerance_A=1.5)

        self.assertGreater(metrics["density_skeleton_precision"], 0.9)
        self.assertLess(metrics["density_skeleton_recall"], 0.8)
        self.assertGreater(metrics["density_skeleton_unmatched_density_length_A"], 0.0)

    def test_skeleton_ligand_sticking_out_lowers_precision(self):
        exp = _line()
        sim = _t_shape()

        metrics = skeleton_graph_metrics(exp, sim, tolerance_A=1.5)

        self.assertGreater(metrics["density_skeleton_recall"], 0.9)
        self.assertLess(metrics["density_skeleton_precision"], 0.8)
        self.assertGreater(metrics["density_skeleton_unmatched_ligand_length_A"], 0.0)

    def test_skeleton_noise_does_not_dominate_overlapping_component(self):
        exp = _line()
        exp += _line(center=(26, 26, 26), start=20, stop=30)
        sim = _line()

        metrics = skeleton_graph_metrics(exp, sim)

        self.assertEqual(metrics["density_skeleton_component_mode"], "overlap")
        self.assertGreater(metrics["density_skeleton_f1"], 0.9)


if __name__ == "__main__":
    unittest.main()
