import unittest

import numpy as np

try:
    from ChemEM.protocols.refine.search_refine.diagnostic import (
        apply_rigid_transform,
        best_rigid_transform_for_cluster,
        cluster_bad_atoms,
    )
    from ChemEM.protocols.refine.search_refine.direction import (
        build_targets_from_subregion,
    )
except ModuleNotFoundError:
    from protocols.refine.search_refine.diagnostic import (
        apply_rigid_transform,
        best_rigid_transform_for_cluster,
        cluster_bad_atoms,
    )
    from protocols.refine.search_refine.direction import (
        build_targets_from_subregion,
    )


def _pentane_mol():
    """n-pentane: 5 heavy atoms in a linear chain."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles("CCCCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


class TestClusterBadAtoms(unittest.TestCase):
    def test_empty_bad_idx_returns_empty(self):
        mol = _pentane_mol()
        clusters = cluster_bad_atoms(np.array([], dtype=int), mol)
        self.assertEqual(clusters, [])

    def test_contiguous_bad_atoms_form_single_cluster(self):
        mol = _pentane_mol()
        # Pentane is heavy chain 0-1-2-3-4; atoms 1, 2, 3 are all bonded.
        clusters = cluster_bad_atoms(np.array([1, 2, 3], dtype=int), mol)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(sorted(clusters[0].tolist()), [1, 2, 3])

    def test_non_contiguous_bad_atoms_form_separate_clusters(self):
        mol = _pentane_mol()
        # Atoms 0 and 4 are not bonded directly and not via other bad atoms.
        clusters = cluster_bad_atoms(np.array([0, 4], dtype=int), mol)
        self.assertEqual(len(clusters), 2)
        sizes = sorted(c.size for c in clusters)
        self.assertEqual(sizes, [1, 1])

    def test_two_clusters_with_a_gap(self):
        mol = _pentane_mol()
        # Atoms {0,1} and {3,4} — atom 2 is not bad, so the two sides are separate.
        clusters = cluster_bad_atoms(np.array([0, 1, 3, 4], dtype=int), mol)
        self.assertEqual(len(clusters), 2)
        got = {tuple(sorted(c.tolist())) for c in clusters}
        self.assertEqual(got, {(0, 1), (3, 4)})


class TestBestRigidTransformForCluster(unittest.TestCase):
    def test_uniform_direction_yields_pure_translation(self):
        # Three atoms in an L-shape, all wanting to move in +x by ε.
        cluster = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        dirs = np.tile(np.array([1.0, 0.0, 0.0]), (3, 1))
        R, t = best_rigid_transform_for_cluster(cluster, dirs, epsilon_A=0.5)
        # Rotation should be (very close to) identity; translation = (0.5, 0, 0).
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(t, np.array([0.5, 0.0, 0.0]), atol=1e-10)

    def test_desired_positions_are_reached_when_rigid_solution_exists(self):
        # Rotate the cluster by a known small angle around z; the best
        # rigid fit must recover exactly (Procrustes is exact for rigid cases).
        cluster = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        centroid = cluster.mean(axis=0)
        theta = np.radians(15.0)
        Rz = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta),  np.cos(theta), 0.0],
                [0.0,            0.0,            1.0],
            ],
            dtype=np.float64,
        )
        target_positions = (cluster - centroid) @ Rz.T + centroid + np.array([0.1, 0.0, 0.0])
        dirs = target_positions - cluster  # per-atom displacement vectors
        # epsilon = 1 so that desired = coords + dirs = target_positions.
        R, t = best_rigid_transform_for_cluster(cluster, dirs, epsilon_A=1.0)
        applied = cluster @ R.T + t
        np.testing.assert_allclose(applied, target_positions, atol=1e-8)

    def test_apply_rigid_transform_only_moves_cluster(self):
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        cluster = np.array([1, 2], dtype=int)
        R = np.eye(3)
        t = np.array([0.0, 0.7, 0.0])
        new = apply_rigid_transform(coords, cluster, R, t)
        # Only atoms 1 and 2 get the translation.
        np.testing.assert_allclose(new[[0, 3, 4]], coords[[0, 3, 4]], atol=1e-12)
        np.testing.assert_allclose(
            new[cluster], coords[cluster] + np.array([0.0, 0.7, 0.0]), atol=1e-12
        )


class TestBuildTargetsFromSubregion(unittest.TestCase):
    def test_builds_targets_only_for_cluster(self):
        # 5 ligand heavies + 2 protein atoms.
        heavy_A = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        n_full = 7
        full_pos_nm = np.zeros((n_full, 3), dtype=np.float64)
        full_pos_nm[:5] = heavy_A * 0.1
        full_pos_nm[5:] = np.array([[9.0, 9.0, 9.0], [9.1, 9.0, 9.0]])

        cluster = np.array([2, 3], dtype=int)
        R = np.eye(3)
        t = np.array([0.0, 0.5, 0.0])

        target, stats = build_targets_from_subregion(
            accepted_pos_nm=full_pos_nm,
            ligand_heavy_idx=np.arange(5, dtype=int),
            heavy_coords_A=heavy_A,
            cluster_atoms=cluster,
            R=R,
            t=t,
        )
        # Non-cluster ligand atoms + protein atoms must be untouched.
        np.testing.assert_allclose(target[[0, 1, 4]], full_pos_nm[[0, 1, 4]], atol=1e-12)
        np.testing.assert_allclose(target[5:], full_pos_nm[5:], atol=1e-12)
        # Cluster atoms get +y displacement of 0.5 Å = 0.05 nm.
        np.testing.assert_allclose(
            target[cluster] - full_pos_nm[cluster],
            np.tile(np.array([0.0, 0.05, 0.0]), (2, 1)),
            atol=1e-12,
        )
        self.assertEqual(stats["moved_atoms"], 2)
        self.assertAlmostEqual(stats["max_target_disp_A"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
