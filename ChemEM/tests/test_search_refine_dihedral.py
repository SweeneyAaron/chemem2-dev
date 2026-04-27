import unittest

import numpy as np

try:
    from ChemEM.protocols.refine.search_refine.diagnostic import (
        apply_dihedral_delta,
        bond_sides_heavy,
        directed_kick_proposal,
        enumerate_rotatable_heavy_bonds,
        rank_dihedrals_for_atom,
    )
    from ChemEM.protocols.refine.search_refine.direction import (
        build_targets_from_dihedral,
    )
except ModuleNotFoundError:
    from protocols.refine.search_refine.diagnostic import (
        apply_dihedral_delta,
        bond_sides_heavy,
        directed_kick_proposal,
        enumerate_rotatable_heavy_bonds,
        rank_dihedrals_for_atom,
    )
    from protocols.refine.search_refine.direction import (
        build_targets_from_dihedral,
    )


def _butane_mol_and_coords():
    """Build an n-butane-like RDKit mol with a known 3D conformer.

    Heavy atoms are laid out C0-C1-C2-C3 along x, with all H atoms implicit
    (added via Chem.AddHs), then embedded to 3D via RDKit.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles("CCCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


class TestDihedralHelpers(unittest.TestCase):
    def test_enumerate_rotatable_bonds_on_butane(self):
        mol = _butane_mol_and_coords()
        bonds = enumerate_rotatable_heavy_bonds(mol)
        # n-butane has one rotatable central bond (C1-C2). C0-C1 and C2-C3 are
        # terminal (D1) bonds and excluded by the SMARTS.
        self.assertEqual(len(bonds), 1)
        a, b = bonds[0]
        # Both endpoints are heavy-atom-local indices 0..3 (exactly 4 heavies).
        self.assertIn(a, (1, 2))
        self.assertIn(b, (1, 2))
        self.assertNotEqual(a, b)

    def test_bond_sides_partition_sums_to_all_heavy(self):
        mol = _butane_mol_and_coords()
        bonds = enumerate_rotatable_heavy_bonds(mol)
        bond = bonds[0]
        side_a, side_b = bond_sides_heavy(mol, bond)
        self.assertIsNotNone(side_a)
        self.assertIsNotNone(side_b)
        # 4 heavy atoms total; the single rotatable bond splits them 2+2.
        self.assertEqual(side_a.size + side_b.size, 4)
        self.assertEqual(set(side_a.tolist()) & set(side_b.tolist()), set())
        # Each endpoint is on its own side.
        self.assertIn(bond[0], side_a.tolist())
        self.assertIn(bond[1], side_b.tolist())

    def test_apply_dihedral_delta_rotates_only_side_atoms(self):
        mol = _butane_mol_and_coords()
        heavy_mol_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
        conf = mol.GetConformer(0)
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in heavy_mol_idx],
            dtype=np.float64,
        )

        bonds = enumerate_rotatable_heavy_bonds(mol)
        bond = bonds[0]
        side_a, side_b = bond_sides_heavy(mol, bond)

        # Rotate side_b by 90° around the bond axis.
        new = apply_dihedral_delta(coords, bond, np.radians(90.0), side_b)
        # Atoms in side_a (including bond endpoint a) must be unchanged.
        for i in side_a.tolist():
            np.testing.assert_allclose(new[i], coords[i], atol=1e-10)
        # At least one atom in side_b must have moved meaningfully.
        moved = any(
            float(np.linalg.norm(new[i] - coords[i])) > 1e-3 for i in side_b.tolist()
        )
        self.assertTrue(moved)
        # The endpoint of the bond that lies on the axis itself doesn't move.
        # bond[1] is an endpoint and IS on the axis; it stays fixed.
        np.testing.assert_allclose(new[bond[1]], coords[bond[1]], atol=1e-10)

    def test_rank_dihedrals_picks_axis_aware_bond(self):
        mol = _butane_mol_and_coords()
        heavy_mol_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
        conf = mol.GetConformer(0)
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in heavy_mol_idx],
            dtype=np.float64,
        )
        bonds = enumerate_rotatable_heavy_bonds(mol)

        # For a C0 bad atom, a direction perpendicular to the C1-C2 axis should
        # have high alignment; parallel direction should have low alignment.
        axis = coords[bonds[0][1]] - coords[bonds[0][0]]
        axis = axis / np.linalg.norm(axis)
        # Pick a perpendicular direction.
        ref = np.array([0.0, 1.0, 0.0])
        perp = ref - np.dot(ref, axis) * axis
        perp = perp / np.linalg.norm(perp)

        ranked_perp = rank_dihedrals_for_atom(0, perp, coords, bonds, mol)
        ranked_par = rank_dihedrals_for_atom(0, axis, coords, bonds, mol)
        self.assertTrue(len(ranked_perp) >= 1)
        # An atom exactly on the axis has v_norm=0 and the bond is skipped.
        # Butane's C0 is typically slightly off-axis after MMFF optimisation,
        # so we expect the bond to be present. Either way, perp alignment must
        # exceed parallel alignment (latter ~0 by geometry).
        align_perp = abs(ranked_perp[0].alignment) if ranked_perp else 0.0
        align_par = abs(ranked_par[0].alignment) if ranked_par else 0.0
        self.assertGreater(align_perp, align_par)

    def test_directed_kick_returns_none_when_no_rotatable_bonds(self):
        mol = _butane_mol_and_coords()
        heavy_mol_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
        conf = mol.GetConformer(0)
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in heavy_mol_idx],
            dtype=np.float64,
        )
        grad = np.tile(np.array([1.0, 0.0, 0.0]), (coords.shape[0], 1)).astype(np.float64)
        q = np.array([0.9, 0.8, 0.2, 0.9], dtype=np.float64)
        result = directed_kick_proposal(
            heavy_coords_A=coords,
            atom_gradient=grad,
            q_score=q,
            rotatable_bonds=[],
            mol=mol,
            kick_angle_deg=90.0,
        )
        self.assertIsNone(result)

    def test_directed_kick_picks_worst_q_atom(self):
        mol = _butane_mol_and_coords()
        heavy_mol_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
        conf = mol.GetConformer(0)
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in heavy_mol_idx],
            dtype=np.float64,
        )
        bonds = enumerate_rotatable_heavy_bonds(mol)
        # Point every atom's gradient perpendicular to the C1-C2 bond axis so
        # the dihedral ranking is nontrivial.
        axis = coords[bonds[0][1]] - coords[bonds[0][0]]
        axis = axis / np.linalg.norm(axis)
        perp = np.array([0.0, 1.0, 0.0]) - np.dot(np.array([0.0, 1.0, 0.0]), axis) * axis
        perp = perp / np.linalg.norm(perp)
        grad = np.tile(perp, (coords.shape[0], 1)).astype(np.float64)

        # Atom 0 is the worst-Q atom; its side should be picked and moved.
        q = np.array([0.05, 0.9, 0.9, 0.9], dtype=np.float64)
        result = directed_kick_proposal(
            heavy_coords_A=coords,
            atom_gradient=grad,
            q_score=q,
            rotatable_bonds=bonds,
            mol=mol,
            kick_angle_deg=60.0,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["bad_atom"], 0)
        self.assertEqual(abs(result["delta_theta_deg"]), 60.0)
        # Atom 0 should have been displaced from its input position.
        new = result["new_heavy_coords_A"]
        self.assertGreater(float(np.linalg.norm(new[0] - coords[0])), 1e-3)
        # The bond endpoint on the other side should be untouched (on the axis
        # as the pivot OR on the fixed side).
        other_endpoint = [a for a in result["bond"] if a != 0]  # not atom 0
        for ep in other_endpoint:
            np.testing.assert_allclose(new[ep], coords[ep], atol=1e-10)

    def test_directed_kick_returns_none_for_zero_gradient(self):
        mol = _butane_mol_and_coords()
        heavy_mol_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
        conf = mol.GetConformer(0)
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in heavy_mol_idx],
            dtype=np.float64,
        )
        bonds = enumerate_rotatable_heavy_bonds(mol)
        q = np.array([0.1, 0.9, 0.9, 0.9], dtype=np.float64)
        # Worst-Q atom has zero gradient, so no target direction.
        grad = np.zeros((coords.shape[0], 3), dtype=np.float64)
        result = directed_kick_proposal(
            heavy_coords_A=coords,
            atom_gradient=grad,
            q_score=q,
            rotatable_bonds=bonds,
            mol=mol,
            kick_angle_deg=90.0,
        )
        self.assertIsNone(result)

    def test_build_targets_from_dihedral_pins_other_atoms(self):
        mol = _butane_mol_and_coords()
        heavy_mol_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
        conf = mol.GetConformer(0)
        heavy_A = np.array(
            [list(conf.GetAtomPosition(i)) for i in heavy_mol_idx],
            dtype=np.float64,
        )
        bonds = enumerate_rotatable_heavy_bonds(mol)
        bond = bonds[0]
        _, side_b = bond_sides_heavy(mol, bond)

        # Simulate a 7-atom system: 4 ligand heavies + 3 protein atoms.
        n_full = 7
        full_pos_nm = np.zeros((n_full, 3), dtype=np.float64)
        full_pos_nm[0:4] = heavy_A * 0.1  # Å → nm
        full_pos_nm[4:] = np.array([[9.0, 9.0, 9.0], [9.1, 9.0, 9.0], [9.0, 9.1, 9.0]])

        lig_idx = np.array([0, 1, 2, 3], dtype=int)

        target, stats = build_targets_from_dihedral(
            accepted_pos_nm=full_pos_nm,
            ligand_heavy_idx=lig_idx,
            heavy_coords_A=heavy_A,
            bond=bond,
            delta_theta=np.radians(90.0),
            side_atoms=side_b,
        )

        # Protein atoms must be untouched.
        np.testing.assert_allclose(target[4:], full_pos_nm[4:], atol=1e-12)
        # Side-A atoms (not rotated) must be at their current positions.
        side_a = np.array([i for i in range(4) if i not in side_b.tolist()], dtype=int)
        for i in side_a.tolist():
            np.testing.assert_allclose(target[i], full_pos_nm[i], atol=1e-12)
        # At least one side-B atom must have a moved target.
        moved_any = any(
            not np.allclose(target[i], full_pos_nm[i]) for i in side_b.tolist()
        )
        self.assertTrue(moved_any)
        self.assertGreater(stats["moved_atoms"], 0)
        self.assertGreater(stats["max_target_disp_A"], 0.0)


if __name__ == "__main__":
    unittest.main()
