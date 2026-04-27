import unittest

import numpy as np

try:
    from ChemEM.protocols.refine.search_refine.direction import (
        build_targets_from_gradient,
    )
except ModuleNotFoundError:
    from protocols.refine.search_refine.direction import (
        build_targets_from_gradient,
    )


class TestBuildTargetsFromGradient(unittest.TestCase):
    def setUp(self):
        # 5-atom ligand embedded at the start of a 6-atom full-system positions
        # array (1 trailing "protein" atom that must never be touched).
        self.full_pos_nm = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
                [0.1, 0.1, 0.0],
                [9.9, 9.9, 9.9],
            ],
            dtype=np.float64,
        )
        self.lig_idx = np.array([0, 1, 2, 3, 4], dtype=int)
        # All atoms have the same unit-x gradient so move magnitude is uniform.
        self.grad = np.tile(np.array([1.0, 0.0, 0.0]), (5, 1)).astype(np.float64)
        self.cap_A = 1.0

    def _run(self, bad_mask=None, mode="plus", scale=1.0):
        return build_targets_from_gradient(
            accepted_pos_nm=self.full_pos_nm,
            ligand_heavy_idx=self.lig_idx,
            grad_heavy=self.grad,
            cap_A=self.cap_A,
            proposal_scale=scale,
            proposal_mode=mode,
            rng=np.random.default_rng(0),
            bad_mask=bad_mask,
        )

    def test_no_mask_moves_all_atoms(self):
        target, stats = self._run(bad_mask=None)
        self.assertEqual(stats["moved_atoms"], 5)
        self.assertEqual(stats["targeted_atoms"], 5)
        # Every ligand atom's target differs from its input.
        for i in self.lig_idx:
            self.assertFalse(np.allclose(target[i], self.full_pos_nm[i]))
        # The "protein" atom must not move.
        self.assertTrue(np.allclose(target[5], self.full_pos_nm[5]))

    def test_index_mask_moves_only_flagged_atoms(self):
        target, stats = self._run(bad_mask=np.array([2], dtype=int))
        self.assertEqual(stats["targeted_atoms"], 1)
        self.assertEqual(stats["moved_atoms"], 1)
        # Only ligand atom 2 moves.
        for i in [0, 1, 3, 4]:
            self.assertTrue(np.allclose(target[i], self.full_pos_nm[i]))
        self.assertFalse(np.allclose(target[2], self.full_pos_nm[2]))
        # Protein untouched.
        self.assertTrue(np.allclose(target[5], self.full_pos_nm[5]))

    def test_bool_mask_works(self):
        mask = np.array([False, True, False, True, False])
        target, stats = self._run(bad_mask=mask)
        self.assertEqual(stats["targeted_atoms"], 2)
        self.assertEqual(stats["moved_atoms"], 2)
        for i in [0, 2, 4]:
            self.assertTrue(np.allclose(target[i], self.full_pos_nm[i]))
        for i in [1, 3]:
            self.assertFalse(np.allclose(target[i], self.full_pos_nm[i]))

    def test_empty_mask_moves_nothing(self):
        target, stats = self._run(bad_mask=np.array([], dtype=int))
        self.assertEqual(stats["targeted_atoms"], 0)
        self.assertEqual(stats["moved_atoms"], 0)
        for i in range(self.full_pos_nm.shape[0]):
            self.assertTrue(np.allclose(target[i], self.full_pos_nm[i]))

    def test_out_of_range_indices_ignored(self):
        target, stats = self._run(bad_mask=np.array([2, 99, -1], dtype=int))
        self.assertEqual(stats["targeted_atoms"], 1)
        self.assertFalse(np.allclose(target[2], self.full_pos_nm[2]))
        for i in [0, 1, 3, 4]:
            self.assertTrue(np.allclose(target[i], self.full_pos_nm[i]))

    def test_wrong_length_bool_mask_falls_through(self):
        # Malformed bool mask: do not crash, do not move anything.
        target, stats = self._run(bad_mask=np.array([True, False], dtype=bool))
        self.assertEqual(stats["moved_atoms"], 0)
        for i in range(self.full_pos_nm.shape[0]):
            self.assertTrue(np.allclose(target[i], self.full_pos_nm[i]))

    def test_displacement_direction_matches_gradient(self):
        # With "plus" mode and +x gradient, flagged atom should move in +x.
        target, _ = self._run(bad_mask=np.array([1]))
        delta = target[1] - self.full_pos_nm[1]
        self.assertGreater(delta[0], 0.0)
        self.assertAlmostEqual(delta[1], 0.0, places=10)
        self.assertAlmostEqual(delta[2], 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
