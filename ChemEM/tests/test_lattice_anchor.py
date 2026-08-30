"""Tests for the site-anchored ECHO grid lattice (`--echo-lattice-anchor`).

`make_zero_grid` puts the grid origin at `min(all atoms) - padding`, so the
lattice PHASE is set by whichever atom sits at the protein's extremity. Those are
often rebuilt atoms tens of Angstrom from the binding site; move one and the whole
sampling lattice slides, changing `electro_attractive` through trilinear
interpolation even though no physics changed.

`make_lattice_grid` snaps the origin onto an absolute lattice so the phase depends
only on `lattice_origin`. The properties that matter:

  - coverage is preserved (the grid still contains coords +/- padding);
  - the extent grows by at most one voxel per axis;
  - the phase is invariant to a distant atom moving;
  - `off` remains bit-for-bit identical to the old behaviour.
"""
import unittest

import numpy as np

try:
    from ChemEM.tools.precomputed_data import (make_zero_grid, make_lattice_grid,
                                               snap_point_to_grid)
except ModuleNotFoundError:
    from tools.precomputed_data import (make_zero_grid, make_lattice_grid,
                                        snap_point_to_grid)


def _rng_coords(seed, n=200, scale=60.0, offset=(120.0, 140.0, 150.0)):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, scale, size=(n, 3)) + np.asarray(offset)


class TestSnapPointToGrid(unittest.TestCase):
    def test_snaps_to_the_nearest_node(self):
        got = snap_point_to_grid([1.1, 2.4, -0.6], origin=[0.0, 0.0, 0.0], spacing=0.5)
        np.testing.assert_allclose(got, [1.0, 2.5, -0.5])

    def test_a_node_maps_to_itself(self):
        origin = np.array([0.3, -1.2, 4.0])
        node = origin + np.array([3, -2, 7]) * 0.375
        np.testing.assert_allclose(snap_point_to_grid(node, origin, 0.375), node)

    def test_result_is_always_on_the_lattice(self):
        origin, spacing = np.array([0.13, -2.7, 5.5]), 0.375
        for point in _rng_coords(7, n=50):
            snapped = snap_point_to_grid(point, origin, spacing)
            steps = (snapped - origin) / spacing
            np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)


class TestMakeLatticeGrid(unittest.TestCase):
    SPACING = 0.375
    PADDING = 10.0

    def _covers(self, origin, grid, coords, padding):
        lo = coords.min(axis=0) - padding
        hi = coords.max(axis=0) + padding
        nz, ny, nx = grid.shape
        top = origin + (np.array([nx, ny, nz]) - 1) * self.SPACING
        return np.all(origin <= lo + 1e-9) and np.all(top >= hi - 1e-9)

    def test_covers_the_padded_bounding_box(self):
        for seed in range(6):
            coords = _rng_coords(seed)
            origin, grid = make_lattice_grid(coords, spacing=self.SPACING,
                                             padding=self.PADDING)
            self.assertTrue(self._covers(origin, grid, coords, self.PADDING),
                            f"seed {seed}: grid does not cover coords +/- padding")

    def test_covers_the_top_face(self):
        """Counting voxels from `mins` rather than the snapped origin loses up to
        one voxel of coverage at the top."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        origin, grid = make_lattice_grid(coords, spacing=0.3, padding=0.0,
                                         lattice_origin=(0.05, 0.05, 0.05))
        nz, ny, nx = grid.shape
        top = origin + (np.array([nx, ny, nz]) - 1) * 0.3
        self.assertTrue(np.all(top >= 1.0 - 1e-9))

    def test_extent_within_one_voxel_of_make_zero_grid(self):
        for seed in range(6):
            coords = _rng_coords(seed)
            _o1, g1 = make_zero_grid(coords, spacing=self.SPACING, padding=self.PADDING)
            _o2, g2 = make_lattice_grid(coords, spacing=self.SPACING, padding=self.PADDING)
            for a, b in zip(g1.shape, g2.shape):
                self.assertLessEqual(abs(a - b), 1)

    def test_origin_is_on_the_requested_lattice(self):
        lat0 = np.array([1.7, -0.4, 2.2])
        origin, _grid = make_lattice_grid(_rng_coords(3), spacing=self.SPACING,
                                          padding=self.PADDING, lattice_origin=lat0)
        steps = (origin - lat0) / self.SPACING
        np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)

    def test_centroid_mode_makes_the_centroid_a_lattice_node(self):
        centroid = np.array([114.65753125, 138.51878125, 152.905])
        origin, _grid = make_lattice_grid(_rng_coords(1), spacing=self.SPACING,
                                          padding=self.PADDING, lattice_origin=centroid)
        steps = (centroid - origin) / self.SPACING
        np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)


class TestDistantAtomInvariance(unittest.TestCase):
    """The property the whole change exists for."""

    SPACING = 0.375
    PADDING = 10.0

    def setUp(self):
        self.coords = _rng_coords(11)
        # An atom far outside the cluster, at the bounding-box extremity -- the
        # rebuilt-loop case. Perturb it and see whose lattice moves.
        self.coords = np.vstack([self.coords, [[40.0, 40.0, 40.0]]])
        self.moved = self.coords.copy()
        self.moved[-1] += 0.9

    def _voxel_offset(self, o1, o2):
        """Offset between two origins, in voxels. Integer <=> the two grids sample
        the SAME physical points, so every trilinear lookup is unchanged; a
        fractional part means the sample points moved under the ligand."""
        return (np.asarray(o1) - np.asarray(o2)) / self.SPACING

    def test_old_behaviour_shifts_the_sample_points(self):
        """The leak: the origin moves by a non-integer number of voxels, so the
        grid is resampled at different physical locations."""
        o1, _ = make_zero_grid(self.coords, spacing=self.SPACING, padding=self.PADDING)
        o2, _ = make_zero_grid(self.moved, spacing=self.SPACING, padding=self.PADDING)
        offset = self._voxel_offset(o1, o2)
        self.assertGreater(np.abs(offset - np.round(offset)).max(), 1e-6,
                           "expected a sub-voxel shift under the old behaviour")

    def test_global_anchor_keeps_the_same_sample_points(self):
        """The origin may still track the bounding box, but only in WHOLE voxels,
        so the set of sampled locations is identical."""
        o1, _ = make_lattice_grid(self.coords, spacing=self.SPACING, padding=self.PADDING)
        o2, _ = make_lattice_grid(self.moved, spacing=self.SPACING, padding=self.PADDING)
        offset = self._voxel_offset(o1, o2)
        np.testing.assert_allclose(offset, np.round(offset), atol=1e-9)

    def test_centroid_anchor_keeps_the_same_sample_points(self):
        centroid = self.coords[:-1].mean(axis=0)
        o1, _ = make_lattice_grid(self.coords, spacing=self.SPACING,
                                  padding=self.PADDING, lattice_origin=centroid)
        o2, _ = make_lattice_grid(self.moved, spacing=self.SPACING,
                                  padding=self.PADDING, lattice_origin=centroid)
        offset = self._voxel_offset(o1, o2)
        np.testing.assert_allclose(offset, np.round(offset), atol=1e-9)
        # And the centroid stays an exact node of both lattices.
        for origin in (o1, o2):
            steps = (centroid - origin) / self.SPACING
            np.testing.assert_allclose(steps, np.round(steps), atol=1e-9)

    def test_phase_survives_a_sub_voxel_perturbation(self):
        """A 0.01 A wobble of a distant atom -- the realistic case -- must not move
        the lattice at all, not even by a whole voxel."""
        nudged = self.coords.copy()
        nudged[-1] += 0.01
        o1, g1 = make_lattice_grid(self.coords, spacing=self.SPACING, padding=self.PADDING)
        o2, g2 = make_lattice_grid(nudged, spacing=self.SPACING, padding=self.PADDING)
        np.testing.assert_array_equal(o1, o2)
        self.assertEqual(g1.shape, g2.shape)

        # The same wobble does move the old lattice, sub-voxel.
        p1, _ = make_zero_grid(self.coords, spacing=self.SPACING, padding=self.PADDING)
        p2, _ = make_zero_grid(nudged, spacing=self.SPACING, padding=self.PADDING)
        self.assertGreater(np.abs(p1 - p2).max(), 0.0)


class TestOffIsUnchanged(unittest.TestCase):
    def test_make_zero_grid_is_untouched(self):
        """`--echo-lattice-anchor off` must be a byte-for-byte no-op, so the old
        function has to keep its exact previous semantics."""
        coords = _rng_coords(5)
        origin, grid = make_zero_grid(coords, spacing=0.375, padding=10.0)
        np.testing.assert_allclose(origin, coords.min(axis=0) - 10.0)

        lengths = (coords.max(axis=0) + 10.0) - (coords.min(axis=0) - 10.0)
        expected = [int(np.ceil(lengths[i] / 0.375)) + 1 for i in (2, 1, 0)]
        self.assertEqual(list(grid.shape), expected)


if __name__ == "__main__":
    unittest.main()
