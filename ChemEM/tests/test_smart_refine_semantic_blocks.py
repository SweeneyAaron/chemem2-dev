import unittest

try:
    from rdkit import Chem

    _HAS_RDKIT = True
except ModuleNotFoundError:
    Chem = None
    _HAS_RDKIT = False

if _HAS_RDKIT:
    try:
        from ChemEM.protocols.smart_refine_2.smart_utils import (
            _component_after_cut,
            _heavy_adjacency,
            build_directional_torsion_walks,
            build_semantic_anchor_blocks,
        )
    except ModuleNotFoundError:
        from protocols.smart_refine_2.smart_utils import (
            _component_after_cut,
            _heavy_adjacency,
            build_directional_torsion_walks,
            build_semantic_anchor_blocks,
        )
else:
    build_semantic_anchor_blocks = None
    build_directional_torsion_walks = None
    _component_after_cut = None
    _heavy_adjacency = None


@unittest.skipUnless(_HAS_RDKIT, "rdkit is required for semantic block tests")
class TestSemanticBlockGroupTorsions(unittest.TestCase):
    def setUp(self):
        self.mol = Chem.MolFromSmiles("CC(C)NCC(c1ccc(c(c1)O)O)O")
        self.blocks = build_semantic_anchor_blocks(self.mol)
        self.heavy_atoms, self.adj = _heavy_adjacency(self.mol)

    def test_directed_torsions_move_real_ligand_cut_sides(self):
        for block in self.blocks:
            for torsion in block.torsions:
                real_moved = _component_after_cut(
                    adj=self.adj,
                    start=torsion.axis[1],
                    cut=torsion.axis,
                    allowed=self.heavy_atoms,
                )

                self.assertEqual(tuple(sorted(real_moved)), torsion.moved_atom_indices)
                self.assertEqual(
                    tuple(sorted(self.heavy_atoms - real_moved)),
                    torsion.fixed_atom_indices,
                )

    def test_catechol_sidechain_torsion_ownership(self):
        self.assertEqual(
            [block.role for block in self.blocks],
            ["core", "path", "branch_bundle", "branch"],
        )

        core, path, branch_bundle, branch = self.blocks

        self.assertEqual([torsion.axis for torsion in core.torsions], [(5, 6)])
        self.assertEqual(core.torsions[0].moved_atom_indices, core.atom_indices)
        self.assertEqual(core.torsions[0].moved_block_ids, (core.block_id,))

        self.assertEqual([torsion.axis for torsion in branch_bundle.torsions], [(3, 1)])
        self.assertEqual(branch_bundle.torsions[0].moved_atom_indices, branch_bundle.atom_indices)
        self.assertEqual(
            branch_bundle.torsions[0].moved_block_ids,
            (branch_bundle.block_id,),
        )

        self.assertEqual(branch.torsions, ())

        path_torsions = {torsion.axis: torsion for torsion in path.torsions}
        self.assertEqual(
            set(path_torsions),
            {(1, 3), (3, 4), (4, 3), (4, 5), (5, 4), (6, 5)},
        )

        sidechain_move = path_torsions[(6, 5)]
        self.assertTrue({0, 2, 14}.issubset(sidechain_move.moved_atom_indices))
        self.assertEqual(sidechain_move.moved_block_ids, (path.block_id, 2, 3))

        core_branch_move = path_torsions[(4, 5)]
        self.assertTrue({7, 14}.issubset(core_branch_move.moved_atom_indices))
        self.assertEqual(core_branch_move.moved_block_ids, (core.block_id, branch.block_id))

    def test_directional_walk_keeps_only_root_to_tip_torsions(self):
        walks = build_directional_torsion_walks(
            self.mol,
            self.blocks,
            root_block_id=0,
            target_block_ids=[1, 2, 3],
        )

        self.assertEqual(len(walks), 1)
        self.assertEqual(
            [step.torsion.axis for step in walks[0].steps],
            [(6, 5), (5, 4), (4, 3), (3, 1)],
        )
        self.assertNotIn((5, 6), [step.torsion.axis for step in walks[0].steps])
        self.assertNotIn((4, 5), [step.torsion.axis for step in walks[0].steps])
        self.assertNotIn((3, 4), [step.torsion.axis for step in walks[0].steps])
        self.assertNotIn((1, 3), [step.torsion.axis for step in walks[0].steps])

    def test_directional_walk_handles_linear_chain_both_directions(self):
        mol = Chem.MolFromSmiles("CCCCCCC")
        blocks = build_semantic_anchor_blocks(mol)
        raw_axes = {
            torsion.axis
            for block in blocks
            for torsion in block.torsions
        }

        self.assertTrue({(2, 3), (3, 2)}.issubset(raw_axes))

        walks = build_directional_torsion_walks(
            mol,
            blocks,
            root_block_id=0,
            target_block_ids=[1, 2],
        )
        axes_by_walk = [
            [step.torsion.axis for step in walk.steps]
            for walk in walks
        ]

        self.assertIn([(4, 3), (3, 2), (2, 1)], axes_by_walk)
        self.assertIn([(2, 3), (3, 4), (4, 5)], axes_by_walk)
        self.assertNotIn([(3, 2), (2, 1)], axes_by_walk)

    def test_directional_walk_frontier_targets(self):
        walks = build_directional_torsion_walks(
            self.mol,
            self.blocks,
            root_block_id=0,
            target_block_ids=[1, 2, 3],
        )

        self.assertEqual(
            [step.frontier_atom_indices for step in walks[0].steps],
            [(4, 14), (3,), (1,), (0, 2)],
        )
        self.assertEqual(walks[0].frontier_atom_indices, (0, 1, 2, 3, 4, 14))
        self.assertEqual(walks[0].axes, ((6, 5), (5, 4), (4, 3), (3, 1)))
        self.assertEqual(
            walks[0].frontier_by_torsion,
            (
                ((7, 6, 5, 4), (4, 14)),
                ((6, 5, 4, 3), (3,)),
                ((5, 4, 3, 1), (1,)),
                ((4, 3, 1, 0), (0, 2)),
            ),
        )

    def test_directional_walk_repr_is_compact(self):
        walks = build_directional_torsion_walks(
            self.mol,
            self.blocks,
            root_block_id=0,
            target_block_ids=[1, 2, 3],
        )
        text = repr(walks[0])

        self.assertIn("frontier_by_torsion=", text)
        self.assertIn("rdkit_torsion=", text)
        self.assertNotIn("reason=", text)
        self.assertFalse(hasattr(walks[0].steps[0].torsion, "reason"))

    def test_directional_walk_includes_terminal_alcohol_branch_upstream(self):
        walks = build_directional_torsion_walks(
            self.mol,
            self.blocks,
            root_block_id=0,
            target_block_ids=[1, 2, 3],
        )

        self.assertIn(14, walks[0].steps[0].frontier_atom_indices)

    def test_directional_walk_uses_coordinate_moving_atoms_for_frontiers(self):
        from rdkit.Chem import AllChem, rdMolTransforms

        mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)NCC(c1ccc(c(c1)O)O)O"))
        AllChem.EmbedMolecule(mol, randomSeed=17)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=50)

        blocks = build_semantic_anchor_blocks(mol)
        walk = build_directional_torsion_walks(
            mol,
            blocks,
            root_block_id=0,
            target_block_ids=[1, 2, 3],
        )[0]

        for step in walk.steps:
            work = Chem.Mol(mol)
            conf = work.GetConformer()
            before = {
                atom_idx: conf.GetAtomPosition(int(atom_idx))
                for atom_idx in step.torsion.moved_atom_indices
            }
            current = rdMolTransforms.GetDihedralDeg(conf, *step.rdkit_torsion)
            rdMolTransforms.SetDihedralDeg(conf, *step.rdkit_torsion, current + 20.0)
            changed = []
            for atom_idx, before_xyz in before.items():
                after_xyz = conf.GetAtomPosition(int(atom_idx))
                dx = after_xyz.x - before_xyz.x
                dy = after_xyz.y - before_xyz.y
                dz = after_xyz.z - before_xyz.z
                if ((dx * dx) + (dy * dy) + (dz * dz)) ** 0.5 > 1e-5:
                    changed.append(int(atom_idx))

            self.assertEqual(tuple(sorted(changed)), step.moved_atom_indices)

    def test_directional_walk_reroots_from_best_path(self):
        walks = build_directional_torsion_walks(
            self.mol,
            self.blocks,
            root_block_id=1,
            target_block_ids=[0, 2, 3],
        )

        axes_by_walk = [
            [step.torsion.axis for step in walk.steps]
            for walk in walks
        ]
        self.assertIn([(5, 6)], axes_by_walk)
        self.assertIn([(3, 1)], axes_by_walk)
        self.assertEqual(
            [walk.target_block_ids for walk in walks],
            [(0,), (2,)],
        )

        root_atoms = set(self.blocks[1].atom_indices)
        for walk in walks:
            for step in walk.steps:
                self.assertFalse(set(step.moved_atom_indices) & root_atoms)


if __name__ == "__main__":
    unittest.main()
