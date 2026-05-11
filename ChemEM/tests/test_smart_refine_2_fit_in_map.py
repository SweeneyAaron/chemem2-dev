import contextlib
import importlib
import importlib.util
import io
import os
import tempfile
import unittest


_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None

if _HAS_NUMPY:
    import numpy as np

    from protocols.smart_refine_2 import optimisers, scorers
else:
    np = None
    optimisers = None
    scorers = None


class DummyMap:
    def __init__(self, apix=1.0):
        self.apix = apix


class DummyRefineLigand:
    def __init__(
        self,
        coords,
        *,
        elements=None,
        protein_coords=None,
        protein_elements=None,
        apix=1.0,
    ):
        self._atom_positions = np.asarray(coords, dtype=np.float64)
        self._atom_elements = np.asarray(
            elements or ["C"] * self._atom_positions.shape[0], dtype=object
        )
        self.local_coords_A = np.asarray(
            protein_coords if protein_coords is not None else np.zeros((0, 3)),
            dtype=np.float64,
        )
        self.local_elements = np.asarray(
            protein_elements or ["C"] * self.local_coords_A.shape[0], dtype=object
        )
        self._map_reference = DummyMap(apix)

    def qscore_context_coords_A(self, ligand_coords_A=None):
        if ligand_coords_A is None:
            ligand_coords_A = self._atom_positions
        return np.concatenate(
            [np.asarray(ligand_coords_A, dtype=np.float64), self.local_coords_A],
            axis=0,
        )

    def ligand_score_indices(self):
        return np.arange(self._atom_positions.shape[0], dtype=int)


def _heavy_coords_from_sdf(path):
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    counts = lines[3].split()
    n_atoms = int(counts[0])
    coords = []
    for line in lines[4 : 4 + n_atoms]:
        parts = line.split()
        if len(parts) < 4 or parts[3] == "H":
            continue
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(coords, dtype=np.float64)


def _fit_result(raw, coords=None):
    if coords is None:
        coords = np.zeros((1, 3), dtype=np.float64)
    return optimisers.FitInMapResult(
        best_coords_A=np.asarray(coords, dtype=np.float64),
        initial_raw_score=float(raw),
        best_raw_score=float(raw),
        initial_objective=float(raw),
        best_objective=float(raw),
        initial_clash_penalty=0.0,
        best_clash_penalty=0.0,
        initial_clash_count=0,
        best_clash_count=0,
        best_max_overlap_A=0.0,
        steps=1,
        evaluations=2,
        converged=False,
        final_step_size_A=0.5,
    )


class _DummyBlock:
    block_id = 0


def _make_refinable_ligand():
    ligand = DummyRefineLigand([[0, 0, 0]], apix=1.0)
    ligand._rotor_tree = [_DummyBlock()]
    ligand.get_best_block_by_qscore = lambda: 0
    return ligand


@unittest.skipUnless(_HAS_NUMPY, "numpy is required for SR2 fit-in-map tests")
class TestSmartRefine2FitInMap(unittest.TestCase):
    def test_optimizer_improves_fake_scorer_without_mutating_ligand(self):
        class TargetScorer(scorers.BaseScorer):
            def score(self, refine_ligand, coords_A):
                centre = np.mean(np.asarray(coords_A, dtype=np.float64), axis=0)
                value = -float(np.sum((centre - np.array([2.0, 0.0, 0.0])) ** 2))
                return scorers.ScoreResult(value=value, terms={"target": value})

        ligand = DummyRefineLigand([[0, 0, 0], [1, 0, 0]], apix=1.0)
        before = ligand._atom_positions.copy()
        result = optimisers.fit_in_map(
            ligand,
            scorer=TargetScorer(),
            config=optimisers.FitInMapConfig(
                max_steps=8,
                progress=False,
            ),
        )

        self.assertGreater(result.delta_raw_score, 0.0)
        np.testing.assert_allclose(ligand._atom_positions, before)

    def test_apix_derived_step_defaults_use_minimum_apix(self):
        ligand = DummyRefineLigand([[0, 0, 0]], apix=[2.0, 1.5, 3.0])
        initial, minimum, fd_delta = optimisers._resolved_steps(
            ligand,
            optimisers.FitInMapConfig(),
        )

        self.assertAlmostEqual(initial, 0.75)
        self.assertAlmostEqual(minimum, 0.015)
        self.assertAlmostEqual(fd_delta, 0.15)

    def test_clash_gate_uses_vdw_overlap_tolerance(self):
        clash = optimisers.protein_ligand_clash(
            [[0.0, 0.0, 0.0]],
            ["C"],
            [[3.0, 0.0, 0.0]],
            ["C"],
            cutoff_A=5.0,
        )
        clear = optimisers.protein_ligand_clash(
            [[0.0, 0.0, 0.0]],
            ["C"],
            [[3.4, 0.0, 0.0]],
            ["C"],
            cutoff_A=5.0,
        )

        self.assertGreater(clash.penalty, 0.0)
        self.assertEqual(clash.count, 1)
        self.assertFalse(clash.accepted)
        self.assertEqual(clear.penalty, 0.0)
        self.assertEqual(clear.count, 0)
        self.assertTrue(clear.accepted)

    def test_hbond_allowance_relaxes_n_o_overlap(self):
        no_pair = optimisers.protein_ligand_clash(
            [[0.0, 0.0, 0.0]],
            ["O"],
            [[2.4, 0.0, 0.0]],
            ["N"],
            cutoff_A=5.0,
        )
        cc_pair = optimisers.protein_ligand_clash(
            [[0.0, 0.0, 0.0]],
            ["C"],
            [[2.4, 0.0, 0.0]],
            ["C"],
            cutoff_A=5.0,
        )

        self.assertEqual(no_pair.penalty, 0.0)
        self.assertTrue(no_pair.accepted)
        self.assertGreater(cc_pair.penalty, 0.0)
        self.assertFalse(cc_pair.accepted)

    def test_hard_clash_mode_reproduces_old_stall(self):
        class XScorer(scorers.BaseScorer):
            def score(self, refine_ligand, coords_A):
                value = float(np.mean(np.asarray(coords_A, dtype=np.float64)[:, 0]))
                return scorers.ScoreResult(value=value)

        ligand = DummyRefineLigand(
            [[0.0, 0.0, 0.0]],
            protein_coords=[[1.0, 0.0, 0.0]],
            apix=1.0,
        )
        result = optimisers.fit_in_map(
            ligand,
            scorer=XScorer(),
            config=optimisers.FitInMapConfig(
                initial_step_size_A=0.5,
                clash_mode="hard",
                max_steps=2,
                progress=False,
            ),
        )

        self.assertEqual(result.best_raw_score, 0.0)
        self.assertEqual(result.best_objective, float("-inf"))
        self.assertGreater(result.best_clash_count, 0)

    def test_default_clash_off_mode_can_move_from_clashing_start(self):
        class XScorer(scorers.BaseScorer):
            def score(self, refine_ligand, coords_A):
                value = float(np.mean(np.asarray(coords_A, dtype=np.float64)[:, 0]))
                return scorers.ScoreResult(value=value)

        ligand = DummyRefineLigand(
            [[0.0, 0.0, 0.0]],
            protein_coords=[[1.0, 0.0, 0.0]],
            apix=1.0,
        )
        result = optimisers.fit_in_map(
            ligand,
            scorer=XScorer(),
            config=optimisers.FitInMapConfig(
                initial_step_size_A=0.5,
                max_steps=2,
                progress=False,
            ),
        )

        self.assertEqual(result.score_terms["fit_clash_mode"], "off")
        self.assertGreater(result.best_raw_score, result.initial_raw_score)
        self.assertGreater(np.linalg.norm(result.best_translation_A), 0.1)
        self.assertFalse(np.allclose(result.best_coords_A, ligand._atom_positions))
        self.assertGreater(result.best_clash_count, 0)

    def test_soft_clash_mode_penalizes_without_killing_gradient(self):
        class XScorer(scorers.BaseScorer):
            def score(self, refine_ligand, coords_A):
                value = float(np.mean(np.asarray(coords_A, dtype=np.float64)[:, 0]))
                return scorers.ScoreResult(value=value)

        ligand = DummyRefineLigand(
            [[0.0, 0.0, 0.0]],
            protein_coords=[[0.0, 0.0, 0.0]],
            apix=1.0,
        )
        result = optimisers.fit_in_map(
            ligand,
            scorer=XScorer(),
            config=optimisers.FitInMapConfig(
                initial_step_size_A=0.5,
                clash_mode="soft",
                clash_weight=0.1,
                max_steps=2,
                progress=False,
            ),
        )

        self.assertEqual(result.score_terms["fit_clash_mode"], "soft")
        self.assertGreater(result.best_raw_score, result.initial_raw_score)
        self.assertLessEqual(result.best_clash_penalty, result.initial_clash_penalty)
        self.assertGreater(np.linalg.norm(result.best_translation_A), 0.1)

    def test_sdf_centroid_benchmark_fixture_moves_toward_reference(self):
        fixture_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "test", "test_output")
        )
        reference_path = os.path.join(fixture_dir, "Ligand_1_2_moved.sdf")
        moved_path = os.path.join(fixture_dir, "Ligand_1_2_moved_1.sdf")
        if not os.path.exists(reference_path) or not os.path.exists(moved_path):
            self.skipTest("local moved ligand benchmark SDFs are unavailable")

        reference = _heavy_coords_from_sdf(reference_path)
        moved = _heavy_coords_from_sdf(moved_path)
        reference_centroid = np.mean(reference, axis=0)

        class CentroidScorer(scorers.BaseScorer):
            def score(self, refine_ligand, coords_A):
                centre = np.mean(np.asarray(coords_A, dtype=np.float64), axis=0)
                value = -float(np.sum((centre - reference_centroid) ** 2))
                return scorers.ScoreResult(value=value)

        ligand = DummyRefineLigand(moved, apix=1.0)
        initial_distance = float(np.linalg.norm(np.mean(moved, axis=0) - reference_centroid))
        result = optimisers.fit_in_map(
            ligand,
            scorer=CentroidScorer(),
            config=optimisers.FitInMapConfig(
                initial_step_size_A=0.75,
                max_steps=24,
                progress=False,
            ),
        )
        final_distance = float(
            np.linalg.norm(np.mean(result.best_coords_A, axis=0) - reference_centroid)
        )

        self.assertGreater(result.best_raw_score, result.initial_raw_score)
        self.assertGreater(np.linalg.norm(result.best_translation_A), 0.1)
        self.assertLess(final_distance, initial_distance)

    def test_qscore_scorer_scores_ligand_with_protein_context(self):
        calls = {}

        def fake_qscores_from_emmap(**kwargs):
            calls.update(kwargs)
            return np.array([0.2, 0.4], dtype=np.float32)

        old_func = scorers.compute_qscores_from_emmap
        scorers.compute_qscores_from_emmap = fake_qscores_from_emmap
        try:
            ligand = DummyRefineLigand(
                [[0, 0, 0], [1, 0, 0]],
                protein_coords=[[2, 0, 0]],
                apix=1.0,
            )
            result = scorers.QScoreScorer().score(
                ligand,
                np.array([[0.1, 0, 0], [1.1, 0, 0]], dtype=np.float64),
            )
        finally:
            scorers.compute_qscores_from_emmap = old_func

        self.assertAlmostEqual(result.value, 0.3, places=6)
        self.assertEqual(calls["atoms_xyz"].shape, (3, 3))
        np.testing.assert_array_equal(calls["score_indices"], np.array([0, 1]))

    def test_accepter_updates_ligand_when_score_improves_and_no_clash(self):
        smart_refine = importlib.import_module("protocols.smart_refine_2.smart_refine")
        ligand = DummyRefineLigand([[0, 0, 0], [1, 0, 0]], apix=1.0)
        result = optimisers.FitInMapResult(
            best_coords_A=np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float64),
            initial_raw_score=0.1,
            best_raw_score=0.2,
            initial_objective=0.1,
            best_objective=0.2,
            initial_clash_penalty=0.0,
            best_clash_penalty=0.0,
            initial_clash_count=0,
            best_clash_count=0,
            best_max_overlap_A=0.0,
            steps=1,
            evaluations=2,
            converged=False,
            final_step_size_A=0.5,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            out = smart_refine.accepter(ligand, result)

        self.assertIs(out, ligand)
        np.testing.assert_allclose(ligand._atom_positions, result.best_coords_A)

    def test_accepter_rejects_new_clash_from_clean_start(self):
        smart_refine = importlib.import_module("protocols.smart_refine_2.smart_refine")
        ligand = DummyRefineLigand([[0, 0, 0]], apix=1.0)
        before = ligand._atom_positions.copy()
        result = optimisers.FitInMapResult(
            best_coords_A=np.array([[1, 0, 0]], dtype=np.float64),
            initial_raw_score=0.1,
            best_raw_score=0.2,
            initial_objective=0.1,
            best_objective=float("-inf"),
            initial_clash_penalty=0.0,
            best_clash_penalty=1.0,
            initial_clash_count=0,
            best_clash_count=1,
            best_max_overlap_A=1.0,
            steps=1,
            evaluations=2,
            converged=False,
            final_step_size_A=0.5,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            out = smart_refine.accepter(ligand, result)

        self.assertIs(out, ligand)
        np.testing.assert_allclose(ligand._atom_positions, before)

    def test_accepter_allows_improved_preclashed_pose_when_penalty_does_not_worsen(self):
        smart_refine = importlib.import_module("protocols.smart_refine_2.smart_refine")
        ligand = DummyRefineLigand([[0, 0, 0]], apix=1.0)
        result = optimisers.FitInMapResult(
            best_coords_A=np.array([[1, 0, 0]], dtype=np.float64),
            initial_raw_score=0.1,
            best_raw_score=0.2,
            initial_objective=0.1,
            best_objective=0.2,
            initial_clash_penalty=2.0,
            best_clash_penalty=1.5,
            initial_clash_count=3,
            best_clash_count=2,
            best_max_overlap_A=1.0,
            steps=1,
            evaluations=2,
            converged=False,
            final_step_size_A=0.5,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            out = smart_refine.accepter(ligand, result)

        self.assertIs(out, ligand)
        np.testing.assert_allclose(ligand._atom_positions, result.best_coords_A)

    @unittest.skipUnless(_HAS_RDKIT, "rdkit is required for SDF debug writer tests")
    def test_debug_sdf_writer_uses_refined_heavy_coords(self):
        from rdkit import Chem
        from rdkit.Geometry import Point3D

        smart_refine = importlib.import_module("protocols.smart_refine_2.smart_refine")

        mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom_idx in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(
                atom_idx,
                Point3D(float(atom_idx), 0.0, 0.0),
            )
        mol.AddConformer(conf, assignId=True)

        class Ligand:
            pass

        class RefineLigandForWrite(DummyRefineLigand):
            pass

        ligand = Ligand()
        ligand.mol = mol
        refine_ligand = RefineLigandForWrite(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            elements=["C", "C"],
        )
        refine_ligand._ligand = ligand
        refine_ligand._atom_indices = np.array([0, 1], dtype=int)

        result = optimisers.FitInMapResult(
            best_coords_A=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            initial_raw_score=0.0,
            best_raw_score=1.0,
            initial_objective=0.0,
            best_objective=1.0,
            initial_clash_penalty=0.0,
            best_clash_penalty=0.0,
            initial_clash_count=0,
            best_clash_count=0,
            best_max_overlap_A=0.0,
            steps=1,
            evaluations=2,
            converged=False,
            final_step_size_A=0.5,
            best_rotation_matrix=np.eye(3),
            best_translation_A=np.array([1.0, 0.0, 0.0]),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                path = smart_refine.write_refined_ligand_sdf(
                    refine_ligand,
                    result,
                    tmpdir,
                    "debug.sdf",
                )
            self.assertTrue(os.path.exists(path))
            read_mol = Chem.SDMolSupplier(path, removeHs=False)[0]
            self.assertIsNotNone(read_mol)
            self.assertEqual(read_mol.GetNumAtoms(), mol.GetNumAtoms())
            self.assertEqual(
                [atom.GetSymbol() for atom in read_mol.GetAtoms()],
                [atom.GetSymbol() for atom in mol.GetAtoms()],
            )
            coords = np.asarray(read_mol.GetConformer().GetPositions())
            np.testing.assert_allclose(coords[0], [1.0, 0.0, 0.0])
            np.testing.assert_allclose(coords[1], [2.0, 0.0, 0.0])

            hydrogen_indices = [
                int(atom.GetIdx())
                for atom in read_mol.GetAtoms()
                if int(atom.GetAtomicNum()) == 1
            ]
            self.assertEqual(hydrogen_indices, list(range(2, mol.GetNumAtoms())))
            stale_hydrogen_coords = np.asarray(
                [[float(atom_idx + 1), 0.0, 0.0] for atom_idx in hydrogen_indices],
                dtype=np.float64,
            )
            self.assertFalse(
                np.allclose(coords[hydrogen_indices], stale_hydrogen_coords)
            )
            for h_idx in hydrogen_indices:
                heavy_neighbors = [
                    int(neighbor.GetIdx())
                    for neighbor in read_mol.GetAtomWithIdx(h_idx).GetNeighbors()
                    if int(neighbor.GetAtomicNum()) > 1
                ]
                self.assertEqual(len(heavy_neighbors), 1)
                dist_A = float(
                    np.linalg.norm(coords[h_idx] - coords[heavy_neighbors[0]])
                )
                self.assertLess(dist_A, 1.35)
            self.assertAlmostEqual(float(read_mol.GetProp("sr2_best_raw_score")), 1.0)

    @unittest.skipUnless(_HAS_RDKIT, "rdkit is required for hydrogen refresh tests")
    def test_hydrogen_refresh_mismatch_does_not_reorder_atoms(self):
        from rdkit import Chem
        from rdkit.Geometry import Point3D

        smart_refine = importlib.import_module("protocols.smart_refine_2.smart_refine")

        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        rw_mol = Chem.RWMol(mol)
        rw_mol.RemoveAtom(mol.GetNumAtoms() - 1)
        mol = rw_mol.GetMol()
        mol.UpdatePropertyCache(strict=False)

        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom_idx in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(
                atom_idx,
                Point3D(float(atom_idx), 0.0, 0.0),
            )
        mol.AddConformer(conf, assignId=True)

        symbols_before = [atom.GetSymbol() for atom in mol.GetAtoms()]
        with contextlib.redirect_stdout(io.StringIO()):
            refreshed = smart_refine._refresh_hydrogen_positions_from_heavy_geometry(
                mol, Chem
            )

        self.assertIs(refreshed, mol)
        self.assertEqual(mol.GetNumAtoms(), len(symbols_before))
        self.assertEqual([atom.GetSymbol() for atom in mol.GetAtoms()], symbols_before)

    def test_refine_ligand_returns_accepted_ligand_and_stores_result(self):
        try:
            smart_refine = importlib.import_module(
                "protocols.smart_refine_2.smart_refine"
            )
        except Exception as exc:
            self.skipTest(f"smart_refine import unavailable: {exc}")

        class FakeScorer:
            pass

        fake_result = optimisers.FitInMapResult(
            best_coords_A=np.zeros((1, 3)),
            initial_raw_score=0.0,
            best_raw_score=1.0,
            initial_objective=0.0,
            best_objective=1.0,
            initial_clash_penalty=0.0,
            best_clash_penalty=0.0,
            initial_clash_count=0,
            best_clash_count=0,
            best_max_overlap_A=0.0,
            steps=1,
            evaluations=2,
            converged=False,
            final_step_size_A=0.5,
        )

        old_get_scorer = smart_refine.get_scorer
        old_fit_in_map = smart_refine.fit_in_map
        smart_refine.get_scorer = lambda scorer: FakeScorer()
        smart_refine.fit_in_map = lambda ligand, scorer=None, config=None: fake_result
        ligand = DummyRefineLigand([[0, 0, 0]], apix=1.0)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIs(smart_refine.refine_ligand(ligand), ligand)
            self.assertIs(ligand._last_fit_in_map_result, fake_result)
            np.testing.assert_allclose(ligand._atom_positions, fake_result.best_coords_A)
            self.assertEqual(ligand._sr2_iterations_completed, 0)
            self.assertEqual(ligand._sr2_stop_reason, "no_rotor_tree")
        finally:
            smart_refine.get_scorer = old_get_scorer
            smart_refine.fit_in_map = old_fit_in_map

    def test_refine_ligand_patience_stops_after_no_improvements(self):
        smart_refine = importlib.import_module("protocols.smart_refine_2.smart_refine")

        class FakeScorer:
            pass

        calls = {"refit": 0}

        def fake_refit(base_rl, *args, **kwargs):
            calls["refit"] += 1
            base_rl._last_fit_in_map_result = _fit_result(1.0)
            return base_rl

        old_get_scorer = smart_refine.get_scorer
        old_fit_in_map = smart_refine.fit_in_map
        old_branch_walker = smart_refine.branch_walker
        old_refit = smart_refine._refit_branch_candidates
        smart_refine.get_scorer = lambda scorer: FakeScorer()
        smart_refine.fit_in_map = lambda ligand, scorer=None, config=None: _fit_result(1.0)
        smart_refine.branch_walker = lambda ligand, scorer=None, config=None: [object()]
        smart_refine._refit_branch_candidates = fake_refit
        ligand = _make_refinable_ligand()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = smart_refine.refine_ligand(ligand, max_iters=10, patience=2)
            self.assertIs(out, ligand)
            self.assertEqual(calls["refit"], 2)
            self.assertEqual(ligand._sr2_iterations_completed, 2)
            self.assertEqual(ligand._sr2_no_improve_iters, 2)
            self.assertEqual(ligand._sr2_stop_reason, "patience")
        finally:
            smart_refine.get_scorer = old_get_scorer
            smart_refine.fit_in_map = old_fit_in_map
            smart_refine.branch_walker = old_branch_walker
            smart_refine._refit_branch_candidates = old_refit

    def test_refine_ligand_patience_resets_after_improvement(self):
        smart_refine = importlib.import_module("protocols.smart_refine_2.smart_refine")

        class FakeScorer:
            pass

        raw_scores = iter([1.2, 1.2, 1.2])
        calls = {"refit": 0}

        def fake_refit(base_rl, *args, **kwargs):
            calls["refit"] += 1
            base_rl._last_fit_in_map_result = _fit_result(next(raw_scores))
            return base_rl

        old_get_scorer = smart_refine.get_scorer
        old_fit_in_map = smart_refine.fit_in_map
        old_branch_walker = smart_refine.branch_walker
        old_refit = smart_refine._refit_branch_candidates
        smart_refine.get_scorer = lambda scorer: FakeScorer()
        smart_refine.fit_in_map = lambda ligand, scorer=None, config=None: _fit_result(1.0)
        smart_refine.branch_walker = lambda ligand, scorer=None, config=None: [object()]
        smart_refine._refit_branch_candidates = fake_refit
        ligand = _make_refinable_ligand()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = smart_refine.refine_ligand(ligand, max_iters=10, patience=2)
            self.assertIs(out, ligand)
            self.assertEqual(calls["refit"], 3)
            self.assertEqual(ligand._sr2_iterations_completed, 3)
            self.assertEqual(ligand._sr2_no_improve_iters, 2)
            self.assertEqual(ligand._sr2_stop_reason, "patience")
        finally:
            smart_refine.get_scorer = old_get_scorer
            smart_refine.fit_in_map = old_fit_in_map
            smart_refine.branch_walker = old_branch_walker
            smart_refine._refit_branch_candidates = old_refit

    def test_refine_ligand_patience_can_be_disabled(self):
        smart_refine = importlib.import_module("protocols.smart_refine_2.smart_refine")

        class FakeScorer:
            pass

        calls = {"refit": 0}

        def fake_refit(base_rl, *args, **kwargs):
            calls["refit"] += 1
            base_rl._last_fit_in_map_result = _fit_result(1.0)
            return base_rl

        old_get_scorer = smart_refine.get_scorer
        old_fit_in_map = smart_refine.fit_in_map
        old_branch_walker = smart_refine.branch_walker
        old_refit = smart_refine._refit_branch_candidates
        smart_refine.get_scorer = lambda scorer: FakeScorer()
        smart_refine.fit_in_map = lambda ligand, scorer=None, config=None: _fit_result(1.0)
        smart_refine.branch_walker = lambda ligand, scorer=None, config=None: [object()]
        smart_refine._refit_branch_candidates = fake_refit
        ligand = _make_refinable_ligand()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = smart_refine.refine_ligand(ligand, max_iters=3, patience=None)
            self.assertIs(out, ligand)
            self.assertEqual(calls["refit"], 3)
            self.assertEqual(ligand._sr2_iterations_completed, 3)
            self.assertEqual(ligand._sr2_no_improve_iters, 3)
            self.assertEqual(ligand._sr2_stop_reason, "max_iters")
        finally:
            smart_refine.get_scorer = old_get_scorer
            smart_refine.fit_in_map = old_fit_in_map
            smart_refine.branch_walker = old_branch_walker
            smart_refine._refit_branch_candidates = old_refit

    def test_refine_ligand_records_no_branch_results_stop_reason(self):
        smart_refine = importlib.import_module("protocols.smart_refine_2.smart_refine")

        class FakeScorer:
            pass

        old_get_scorer = smart_refine.get_scorer
        old_fit_in_map = smart_refine.fit_in_map
        old_branch_walker = smart_refine.branch_walker
        smart_refine.get_scorer = lambda scorer: FakeScorer()
        smart_refine.fit_in_map = lambda ligand, scorer=None, config=None: _fit_result(1.0)
        smart_refine.branch_walker = lambda ligand, scorer=None, config=None: []
        ligand = _make_refinable_ligand()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = smart_refine.refine_ligand(ligand, max_iters=10, patience=2)
            self.assertIs(out, ligand)
            self.assertEqual(ligand._sr2_iterations_completed, 0)
            self.assertEqual(ligand._sr2_no_improve_iters, 0)
            self.assertEqual(ligand._sr2_stop_reason, "no_branch_results")
        finally:
            smart_refine.get_scorer = old_get_scorer
            smart_refine.fit_in_map = old_fit_in_map
            smart_refine.branch_walker = old_branch_walker


if __name__ == "__main__":
    unittest.main()
