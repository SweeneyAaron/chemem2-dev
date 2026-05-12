import contextlib
import importlib
import importlib.util
import io
import os
import tempfile
import unittest
from types import SimpleNamespace


_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None

if _HAS_NUMPY:
    import numpy as np

    from protocols.smart_refine_2 import map_metrics as sr2_map_metrics
    from protocols.smart_refine_2 import optimisers, scorers
else:
    np = None
    sr2_map_metrics = None
    optimisers = None
    scorers = None


class DummyMap:
    def __init__(
        self,
        apix=1.0,
        *,
        density_map=None,
        origin=(0.0, 0.0, 0.0),
        resolution=3.0,
    ):
        self.apix = apix
        self.origin = np.asarray(origin, dtype=np.float64)
        self.resolution = float(resolution)
        if density_map is not None:
            self.density_map = np.asarray(density_map, dtype=np.float64)


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


def _blob(shape=(24, 24, 24), center_xyz=(12.0, 12.0, 12.0), sigma=2.5):
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

        old_func = sr2_map_metrics.compute_qscores_from_emmap
        sr2_map_metrics.compute_qscores_from_emmap = fake_qscores_from_emmap
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
            sr2_map_metrics.compute_qscores_from_emmap = old_func

        self.assertAlmostEqual(result.value, 0.3, places=6)
        self.assertEqual(calls["atoms_xyz"].shape, (3, 3))
        np.testing.assert_array_equal(calls["score_indices"], np.array([0, 1]))

    def test_score_spec_parser_handles_weighted_combinations(self):
        names, weights = scorers.parse_score_spec("CCC,MI", "0.2, 0.8")

        self.assertEqual(names, ["ccc", "mi"])
        self.assertEqual(weights, [0.2, 0.8])
        self.assertEqual(scorers.parse_score_spec(None, None), (["qscore"], [1.0]))
        with self.assertRaises(ValueError):
            scorers.parse_score_spec("CCC,MI", "1.0")
        with self.assertRaises(ValueError):
            scorers.parse_score_spec("not_a_score", None)

    def test_density_metric_scorers_prefer_aligned_coordinates(self):
        density = _blob()
        emmap = DummyMap(
            apix=(1.0, 1.0, 1.0),
            density_map=density,
            origin=(0.0, 0.0, 0.0),
            resolution=3.0,
        )
        aligned = np.array(
            [[12.0, 12.0, 12.0], [12.5, 12.0, 12.0], [12.0, 12.5, 12.0]],
            dtype=np.float64,
        )
        shifted = aligned + np.array([2.0, 0.0, 0.0])
        ligand = DummyRefineLigand(aligned, apix=(1.0, 1.0, 1.0))
        ligand._map_reference = emmap

        options = SimpleNamespace(sr2_use_amp_eq=False)
        for scorer_cls in (scorers.CCCScorer, scorers.MIScorer, scorers.SCIScorer):
            scorer = scorer_cls(options=options)
            aligned_score = scorer.score(ligand, aligned).value
            shifted_score = scorer.score(ligand, shifted).value
            self.assertTrue(np.isfinite(aligned_score))
            self.assertTrue(np.isfinite(shifted_score))
            self.assertGreater(aligned_score, shifted_score)

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

    @unittest.skipUnless(_HAS_RDKIT, "rdkit is required for clone sync tests")
    def test_clone_refine_ligand_preserves_real_ligand_object(self):
        from rdkit import Chem

        smart_refine = importlib.import_module("protocols.smart_refine_2.smart_refine")

        real_ligand = SimpleNamespace(
            mol=Chem.MolFromSmiles("CC"),
            set_positions=lambda coords: None,
        )
        refine_ligand = SimpleNamespace(
            _ligand=real_ligand,
            _ligand_object=real_ligand,
            _atom_positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
            _excluded_root_blocks={7},
        )

        clone = smart_refine._clone_refine_ligand(refine_ligand)

        self.assertIs(clone._ligand_object, real_ligand)
        self.assertIsNot(clone._ligand, real_ligand)
        self.assertTrue(hasattr(clone._ligand, "mol"))
        self.assertEqual(clone._excluded_root_blocks, {7})
        np.testing.assert_allclose(clone._atom_positions, refine_ligand._atom_positions)
        self.assertIsNot(clone._atom_positions, refine_ligand._atom_positions)

    @unittest.skipUnless(_HAS_RDKIT, "rdkit is required for clone sync tests")
    def test_sync_ligand_object_uses_real_ligand_from_clone(self):
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

        class RealLigand:
            def __init__(self, mol):
                self.mol = Chem.Mol(mol)
                self.positions = None

            def set_positions(self, coords):
                self.positions = np.asarray(coords, dtype=np.float64)

        real_ligand = RealLigand(mol)
        clone = SimpleNamespace(
            _ligand=SimpleNamespace(mol=Chem.Mol(mol)),
            _ligand_object=real_ligand,
            _atom_indices=np.array([0, 1], dtype=int),
            _atom_positions=np.array([[5, 0, 0], [6, 0, 0]], dtype=np.float64),
        )

        synced = smart_refine._sync_ligand_object_from_refine_ligand(clone)

        self.assertIs(synced, real_ligand)
        self.assertIsInstance(real_ligand.mol, Chem.Mol)
        self.assertIsNotNone(real_ligand.positions)
        np.testing.assert_allclose(real_ligand.positions[:2], [[5, 0, 0], [6, 0, 0]])

        class RefreshableClone:
            def __init__(self, old_mol, ligand_object):
                self._ligand = SimpleNamespace(mol=Chem.Mol(old_mol))
                self._ligand_object = ligand_object
                self._atom_positions = None

            def _init_atoms(self):
                self._atom_positions = np.asarray(
                    self._ligand.mol.GetConformer(0).GetPositions()[:2],
                    dtype=np.float64,
                )

        refreshable = RefreshableClone(mol, real_ligand)
        smart_refine._refresh_refine_ligand_from_ligand_object(refreshable)

        self.assertIs(refreshable._ligand.mol, real_ligand.mol)
        np.testing.assert_allclose(refreshable._atom_positions, [[5, 0, 0], [6, 0, 0]])

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

    def test_acceptance_scorer_can_reject_optimisation_improvement(self):
        smart_refine = importlib.import_module("protocols.smart_refine_2.smart_refine")

        class OptimisationScorer(scorers.BaseScorer):
            def score(self, refine_ligand, coords_A):
                return scorers.ScoreResult(value=float(np.mean(coords_A[:, 0])))

        class AcceptanceScorer(scorers.BaseScorer):
            def score(self, refine_ligand, coords_A):
                return scorers.ScoreResult(value=-float(np.mean(coords_A[:, 0])))

        fit_result = optimisers.FitInMapResult(
            best_coords_A=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
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
        ligand = DummyRefineLigand([[0, 0, 0]], apix=1.0)
        before = ligand._atom_positions.copy()

        old_fit_in_map = smart_refine.fit_in_map
        smart_refine.fit_in_map = lambda ligand, scorer=None, config=None: fit_result
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = smart_refine.refine_ligand(
                    ligand,
                    optimisation_scorer=OptimisationScorer(),
                    acceptance_scorer=AcceptanceScorer(),
                )
            self.assertIs(out, ligand)
            np.testing.assert_allclose(ligand._atom_positions, before)
            self.assertAlmostEqual(fit_result.initial_raw_score, 0.0)
            self.assertAlmostEqual(fit_result.best_raw_score, -1.0)
            self.assertAlmostEqual(
                fit_result.score_terms["optimisation_best_raw_score"],
                1.0,
            )
        finally:
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

    def test_smart_refine2_final_minimise_flag_defaults_off(self):
        import argparse

        protocol_spec = importlib.import_module("protocol_spec")
        parser = argparse.ArgumentParser()
        protocol_spec.add_smart_ligand_refine2_args(parser)

        defaults = parser.parse_args([])
        self.assertEqual(defaults.sr2_optimisation_score, "qscore")
        self.assertIsNone(defaults.sr2_optimisation_weights)
        self.assertEqual(defaults.sr2_acceptance_score, "qscore")
        self.assertIsNone(defaults.sr2_acceptance_weights)
        self.assertFalse(defaults.sr2_final_minimise)
        self.assertTrue(
            parser.parse_args(["--sr2-final-minimise"]).sr2_final_minimise
        )
        args = parser.parse_args(
            [
                "--sr2-optimisation-score",
                "CCC,MI",
                "--sr2-optimisation-weights",
                "0.2,",
                "0.8",
                "--sr2-acceptance-score",
                "qscore,SCI",
                "--sr2-acceptance-weights",
                "1,",
                "0.5",
            ]
        )
        self.assertEqual(args.sr2_optimisation_score, ["CCC,MI"])
        self.assertEqual(args.sr2_optimisation_weights, ["0.2,", "0.8"])
        self.assertEqual(args.sr2_acceptance_score, ["qscore,SCI"])
        self.assertEqual(args.sr2_acceptance_weights, ["1,", "0.5"])

    def test_final_minimisation_flag_off_does_not_sync_ligand(self):
        smart_refine = importlib.import_module("protocols.smart_refine_2.smart_refine")

        system = SimpleNamespace(
            options=SimpleNamespace(sr2_final_minimise=False),
            density_map=object(),
            protein=SimpleNamespace(complex_structure=object()),
        )
        runner = smart_refine.SmartRefine2(system)
        ligand = DummyRefineLigand([[0, 0, 0]], apix=1.0)
        result = _fit_result(1.0)

        old_sync = smart_refine._sync_ligand_object_from_refine_ligand
        smart_refine._sync_ligand_object_from_refine_ligand = (
            lambda _: self.fail("final minimisation should not sync when disabled")
        )
        try:
            out_ligand, out_result = runner._final_map_minimise_ligand(ligand, result)
        finally:
            smart_refine._sync_ligand_object_from_refine_ligand = old_sync

        self.assertIs(out_ligand, ligand)
        self.assertIs(out_result, result)

    def test_final_minimisation_flag_on_without_map_skips_cleanly(self):
        smart_refine = importlib.import_module("protocols.smart_refine_2.smart_refine")

        system = SimpleNamespace(
            options=SimpleNamespace(sr2_final_minimise=True, no_map=False),
            density_map=None,
            protein=SimpleNamespace(complex_structure=object()),
        )
        runner = smart_refine.SmartRefine2(system)
        ligand = DummyRefineLigand([[0, 0, 0]], apix=1.0)
        result = _fit_result(1.0)

        old_sync = smart_refine._sync_ligand_object_from_refine_ligand
        smart_refine._sync_ligand_object_from_refine_ligand = (
            lambda _: self.fail("final minimisation should not sync without a map")
        )
        try:
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                out_ligand, out_result = runner._final_map_minimise_ligand(
                    ligand,
                    result,
                )
        finally:
            smart_refine._sync_ligand_object_from_refine_ligand = old_sync

        self.assertIs(out_ligand, ligand)
        self.assertIs(out_result, result)
        self.assertIn("no density map", stdout.getvalue())

    def test_final_minimisation_success_updates_wrapper_and_result(self):
        smart_refine = importlib.import_module("protocols.smart_refine_2.smart_refine")

        calls = {}
        protein_structure = object()
        local_structure = object()
        env_structure = object()
        local_map = object()
        density_map = object()

        class FakeLigand:
            def __init__(self):
                self.complex_structure = SimpleNamespace(residues=[object()])
                self.set_positions_calls = []

            def set_positions(self, coords):
                self.set_positions_calls.append(np.asarray(coords, dtype=np.float64))

        class FinalRefineLigand(DummyRefineLigand):
            def __init__(self, ligand_obj):
                super().__init__([[1, 0, 0], [2, 0, 0]], apix=1.0)
                self._ligand = ligand_obj
                self._atom_indices = np.array([0, 1], dtype=int)
                self._protein_index = "old-index"
                self.local_refreshes = []
                self.qscore_updates = 0
                self.best_block_updates = 0
                self.init_atoms_called = False

            def _init_atoms(self):
                self.init_atoms_called = True
                self._atom_positions = np.array(
                    [[8, 0, 0], [9, 0, 0]],
                    dtype=np.float64,
                )
                self._atom_elements = np.array(["C", "C"], dtype=object)
                self._atom_indices = np.array([0, 1], dtype=int)
                self._atom_row_by_mol_index = {0: 0, 1: 1}

            def _init_local_protein(self, protein_index):
                self._protein_index = protein_index
                self.local_refreshes.append(protein_index)

            def update_atom_qscores(self):
                self.qscore_updates += 1

            def get_best_block_by_qscore(self):
                self.best_block_updates += 1
                return 0

        class FakeSetup:
            def __init__(self, **kwargs):
                calls["setup_kwargs"] = kwargs
                self.complex_structure = env_structure

        class FakeMinimise:
            def __init__(self, env):
                calls["minimise_env"] = env

            def run(self, **kwargs):
                calls["run_kwargs"] = kwargs
                return 12.5

        ligand_obj = FakeLigand()
        refine_ligand = FinalRefineLigand(ligand_obj)
        result = _fit_result(
            1.0,
            coords=np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float64),
        )
        result.best_rotation_matrix = np.eye(3)
        result.best_translation_A = np.array([1, 0, 0], dtype=np.float64)

        system = SimpleNamespace(
            options=SimpleNamespace(
                sr2_final_minimise=True,
                no_map=False,
                local_radius=9.5,
                do_biased_md=True,
                pin_specs=["A:GLY:1:CA"],
                distance_specs=["A:GLY:1:CA;LIG:0:C1;2.0"],
            ),
            density_map=density_map,
            protein=SimpleNamespace(complex_structure=protein_structure),
            platform="CPU",
        )
        runner = smart_refine.SmartRefine2(system)
        runner.ligands = [refine_ligand]

        def fake_get_protein_complex():
            calls["get_protein_complex"] = calls.get("get_protein_complex", 0) + 1
            runner._protein_index = "new-index"

        def fake_sync(rl):
            calls["synced"] = rl
            ligand_obj.set_positions(np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float64))
            return ligand_obj

        def fake_residue_positions(residue):
            calls["residue"] = residue
            return np.array([[1, 0, 0]], dtype=np.float64)

        def fake_residue_subset(points, structure, distance_cutoff):
            calls["subset"] = (points.copy(), structure, distance_cutoff)
            return local_structure

        def fake_submap(structure, map_obj):
            calls["submap"] = (structure, map_obj)
            return local_map

        def fake_update_global_positions(full_structure, local_structure):
            calls["global_copy"] = (full_structure, local_structure)
            return 1

        def fake_update_ligand_positions(local_structure, ligand_objects):
            calls["ligand_copy"] = (local_structure, ligand_objects)
            ligand_objects[0].set_positions(
                np.array([[8, 0, 0], [9, 0, 0]], dtype=np.float64)
            )
            return 1

        patched = {
            "ChemEMSimulationSetup": FakeSetup,
            "MinimiseInPlace": FakeMinimise,
            "submap_from_structure": fake_submap,
            "update_global_positions": fake_update_global_positions,
            "update_ligand_positions": fake_update_ligand_positions,
            "get_residue_positions": fake_residue_positions,
            "get_residue_subset_from_points": fake_residue_subset,
            "_sync_ligand_object_from_refine_ligand": fake_sync,
        }
        old_values = {name: getattr(smart_refine, name) for name in patched}
        runner.get_protein_complex = fake_get_protein_complex
        try:
            for name, value in patched.items():
                setattr(smart_refine, name, value)
            with contextlib.redirect_stdout(io.StringIO()):
                out_ligand, out_result = runner._final_map_minimise_ligand(
                    refine_ligand,
                    result,
                )
        finally:
            for name, value in old_values.items():
                setattr(smart_refine, name, value)

        self.assertIs(out_ligand, refine_ligand)
        self.assertIs(out_result, result)
        self.assertIs(calls["synced"], refine_ligand)
        np.testing.assert_allclose(calls["subset"][0], [[1, 0, 0]])
        self.assertIs(calls["subset"][1], protein_structure)
        self.assertEqual(calls["subset"][2], 9.5)
        self.assertEqual(calls["submap"], (local_structure, density_map))

        setup_kwargs = calls["setup_kwargs"]
        self.assertIs(setup_kwargs["protein_structure"], local_structure)
        self.assertEqual(setup_kwargs["ligand_structure"], [ligand_obj])
        self.assertIs(setup_kwargs["density_map"], local_map)
        self.assertEqual(setup_kwargs["protein_restraint"], "protein")
        self.assertEqual(setup_kwargs["pin_k"], 5000.0)
        self.assertFalse(setup_kwargs["localise"])
        self.assertEqual(setup_kwargs["global_k"], 150.0)
        self.assertEqual(setup_kwargs["pin_specs"], ["A:GLY:1:CA"])
        self.assertEqual(
            setup_kwargs["distance_specs"],
            ["A:GLY:1:CA;LIG:0:C1;2.0"],
        )
        self.assertEqual(
            calls["run_kwargs"],
            {"do_biased_md": True, "md_ps": 5.0, "max_iters": 200},
        )
        self.assertEqual(calls["global_copy"], (protein_structure, env_structure))
        self.assertEqual(calls["ligand_copy"], (env_structure, [ligand_obj]))
        self.assertEqual(calls["get_protein_complex"], 1)

        self.assertTrue(refine_ligand.init_atoms_called)
        self.assertEqual(refine_ligand._protein_index, "new-index")
        self.assertGreaterEqual(refine_ligand.local_refreshes.count("new-index"), 1)
        self.assertEqual(refine_ligand.qscore_updates, 1)
        self.assertEqual(refine_ligand.best_block_updates, 1)
        np.testing.assert_allclose(refine_ligand._atom_positions, [[8, 0, 0], [9, 0, 0]])
        np.testing.assert_allclose(result.best_coords_A, [[8, 0, 0], [9, 0, 0]])
        self.assertIsNone(result.best_rotation_matrix)
        self.assertIsNone(result.best_translation_A)
        self.assertEqual(result.score_terms["final_minimise_energy_kcal"], 12.5)


if __name__ == "__main__":
    unittest.main()
