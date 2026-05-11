import json
import os
import tempfile
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from ChemEM import ligand_fitting
except Exception:
    ligand_fitting = None

try:
    from ChemEM.protocols.core.sci_score import (
        simulate_ligand_density_on_map_grid,
        simulate_ligand_density_subgrid,
        truncated_cc,
    )
    from ChemEM.protocols.refine.smart_ligand_refine import (
        AtomClass,
        AtomMapMetrics,
        BranchRebuildMoveGenerator,
        CandidatePose,
        GeometryOracle,
        LigandMapMetrics,
        MapMetricEvaluator,
        MoveAcceptor,
        SmartLigandRefinementConfig,
        SmartLigandRefiner,
        TorsionInfo,
        TorsionProfile,
        TorsionProfileBuilder,
        build_profile_from_scan,
        classify_atom,
        detect_rotatable_torsions,
        low_tail_q,
        rank_torsions_by_badness,
        set_torsion_angle,
        smart_ligand_refine,
    )
    from ChemEM.protocols.refine.smart_ligand_refine.torsion_profiles import (
        dihedral_angle_deg,
    )
    from ChemEM.protocols.refine.smart_ligand_refine.map_metrics import (
        _local_ccc_at_atom,
        _local_ccc_batch,
    )
except ModuleNotFoundError:
    from protocols.core.sci_score import (
        simulate_ligand_density_on_map_grid,
        simulate_ligand_density_subgrid,
        truncated_cc,
    )
    from protocols.refine.smart_ligand_refine import (
        AtomClass,
        AtomMapMetrics,
        BranchRebuildMoveGenerator,
        CandidatePose,
        GeometryOracle,
        LigandMapMetrics,
        MapMetricEvaluator,
        MoveAcceptor,
        SmartLigandRefinementConfig,
        SmartLigandRefiner,
        TorsionInfo,
        TorsionProfile,
        TorsionProfileBuilder,
        build_profile_from_scan,
        classify_atom,
        detect_rotatable_torsions,
        low_tail_q,
        rank_torsions_by_badness,
        set_torsion_angle,
        smart_ligand_refine,
    )
    from protocols.refine.smart_ligand_refine.torsion_profiles import (
        dihedral_angle_deg,
    )
    from protocols.refine.smart_ligand_refine.map_metrics import (
        _local_ccc_at_atom,
        _local_ccc_batch,
    )


class _FakeMap:
    def __init__(self, density_map, origin=(0.0, 0.0, 0.0),
                 apix=(1.0, 1.0, 1.0), resolution=3.0):
        self.density_map = np.asarray(density_map, dtype=np.float64)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.apix = np.asarray(apix, dtype=np.float64)
        self.resolution = float(resolution)


def _blob(shape=(24, 24, 24), center=(12.0, 12.0, 12.0), sigma=2.0):
    z, y, x = [np.arange(n, dtype=np.float64) for n in shape]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    d2 = (
        (xx - center[0]) ** 2
        + (yy - center[1]) ** 2
        + (zz - center[2]) ** 2
    )
    return np.exp(-0.5 * d2 / (sigma * sigma))


def _simulate_full_reference(
    coords_xyz_A,
    atom_masses,
    map_origin_xyz_A,
    map_apix_xyz_A,
    map_shape_zyx,
    *,
    resolution_A,
    sigma_coeff=0.356,
    normalise=True,
):
    coords = np.asarray(coords_xyz_A, dtype=np.float64)
    masses = np.asarray(atom_masses, dtype=np.float64)
    origin = np.asarray(map_origin_xyz_A, dtype=np.float64).reshape(3)
    apix = np.asarray(map_apix_xyz_A, dtype=np.float64).reshape(3)
    nz, ny, nx = [int(i) for i in map_shape_zyx]
    grid = np.zeros((nz, ny, nx), dtype=np.float64)
    for xyz, mass in zip(coords, masses):
        ix = int(np.rint((float(xyz[0]) - origin[0]) / apix[0]))
        iy = int(np.rint((float(xyz[1]) - origin[1]) / apix[1]))
        iz = int(np.rint((float(xyz[2]) - origin[2]) / apix[2]))
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            grid[iz, iy, ix] += float(mass)
    sigma_A = float(sigma_coeff) * float(resolution_A)
    sigma_zyx = np.array(
        [
            sigma_A / max(apix[2], 1e-12),
            sigma_A / max(apix[1], 1e-12),
            sigma_A / max(apix[0], 1e-12),
        ],
        dtype=np.float64,
    )
    sim = gaussian_filter(grid, sigma=sigma_zyx, mode="constant", cval=0.0)
    if normalise:
        vmax = float(np.max(sim))
        if vmax > 0.0:
            sim = sim / vmax
    return np.asarray(sim, dtype=np.float64)


def _butane():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _pentane():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=11)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _linear_branch_fixture():
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.2, 0.0],
            [3.0, 0.2, 0.5],
            [4.0, 0.5, 0.2],
            [5.0, 0.2, 0.8],
        ],
        dtype=np.float64,
    )
    torsions = [
        TorsionInfo(0, (0, 1, 2, 3), (1, 2), [2, 3, 4, 5], 0.0),
        TorsionInfo(1, (1, 2, 3, 4), (2, 3), [3, 4, 5], 0.0),
        TorsionInfo(2, (2, 3, 4, 5), (3, 4), [4, 5], 0.0),
    ]
    profiles = {
        int(t.torsion_id): TorsionProfile(
            torsion_id=int(t.torsion_id),
            minima_deg=[60.0, 120.0],
            relative_energies=[0.0, 0.5],
            scan_angles_deg=[0.0, 60.0, 120.0],
            scan_energies=[1.0, 0.0, 0.5],
            source="test",
            atom_indices=t.atom_indices,
        )
        for t in torsions
    }
    atom_metrics = {
        idx: AtomMapMetrics(idx, 0.10, 0.10, 1.0, np.zeros(3))
        for idx in range(coords.shape[0])
    }
    metrics = LigandMapMetrics(
        atom_metrics=atom_metrics,
        ligand_ccc=0.0,
        low_tail_q=0.10,
        mean_q=0.10,
        worst_atom_indices=[5],
    )
    classes = {
        0: AtomClass.ANCHOR,
        1: AtomClass.ANCHOR,
        2: AtomClass.UNCERTAIN,
        3: AtomClass.REPAIR,
        4: AtomClass.REPAIR,
        5: AtomClass.REPAIR,
    }
    return coords, torsions, profiles, metrics, classes


def _angle_delta_deg(target, current):
    return abs(((float(target) - float(current) + 180.0) % 360.0) - 180.0)


class TestSmartLigandRefineMVP(unittest.TestCase):
    def test_low_tail_q_uses_worst_fraction(self):
        self.assertAlmostEqual(low_tail_q([0.9, 0.1, 0.8, 0.2], fraction=0.5), 0.15)

    def test_atom_classification_respects_density_support(self):
        cfg = SmartLigandRefinementConfig(
            anchor_q_min=0.75,
            anchor_local_ccc_min=0.60,
            repair_q_max=0.55,
            min_density_value=0.2,
        )
        anchor = AtomMapMetrics(0, 0.85, 0.70, 0.3, np.zeros(3))
        repair = AtomMapMetrics(1, 0.20, 0.20, 0.3, np.zeros(3))
        weak = AtomMapMetrics(2, 0.20, 0.20, 0.0, np.zeros(3))

        self.assertEqual(classify_atom(anchor, cfg), AtomClass.ANCHOR)
        self.assertEqual(classify_atom(repair, cfg), AtomClass.REPAIR)
        self.assertEqual(classify_atom(weak, cfg), AtomClass.WEAK_OR_ABSENT)

    def test_subgrid_density_wrapper_matches_full_grid_reference(self):
        coords = np.asarray(
            [[10.2, 11.7, 8.9], [13.6, 10.1, 9.2], [11.0, 14.0, 10.5]],
            dtype=np.float64,
        )
        masses = np.asarray([12.0, 16.0, 14.0], dtype=np.float64)
        kwargs = dict(
            map_origin_xyz_A=np.zeros(3),
            map_apix_xyz_A=np.ones(3),
            map_shape_zyx=(24, 25, 26),
            resolution_A=2.5,
            sigma_coeff=0.356,
            normalise=True,
        )
        ref = _simulate_full_reference(coords, masses, **kwargs)
        got = simulate_ligand_density_on_map_grid(coords, masses, **kwargs)
        np.testing.assert_allclose(got, ref, atol=1e-12, rtol=1e-12)

    def test_decomposed_ligand_ccc_matches_truncated_cc(self):
        density = _blob(shape=(28, 27, 26), center=(11.0, 13.0, 12.0), sigma=2.4)
        density[::5, ::4, ::3] = 0.0
        emmap = _FakeMap(density)
        cfg = SmartLigandRefinementConfig(score_hydrogens=True)
        evaluator = MapMetricEvaluator(emmap, ligand=None, config=cfg)
        coords = np.asarray(
            [[10.2, 12.8, 11.5], [12.4, 13.1, 12.2], [9.9, 14.2, 13.0]],
            dtype=np.float64,
        )
        masses = np.ones(coords.shape[0], dtype=np.float64)
        sim_sub, lo, hi = simulate_ligand_density_subgrid(
            coords,
            masses,
            emmap.origin,
            emmap.apix,
            emmap.density_map.shape,
            resolution_A=emmap.resolution,
            sigma_coeff=cfg.sigma_coeff,
            normalise=True,
        )
        full = np.zeros_like(emmap.density_map, dtype=np.float64)
        z0, y0, x0 = [int(i) for i in lo]
        z1, y1, x1 = [int(i) for i in hi]
        full[z0:z1, y0:y1, x0:x1] = sim_sub
        self.assertAlmostEqual(
            evaluator._ligand_ccc_from_subgrid(sim_sub, lo, hi),
            truncated_cc(emmap.density_map, full, mask=None),
            places=12,
        )

    def test_batched_local_ccc_matches_scalar_helper(self):
        density = _blob(shape=(18, 19, 20), center=(9.0, 9.0, 9.0), sigma=2.0)
        emmap = _FakeMap(density)
        cfg = SmartLigandRefinementConfig(score_hydrogens=True)
        evaluator = MapMetricEvaluator(emmap, ligand=None, config=cfg)
        coords = np.asarray(
            [[8.0, 8.0, 8.0], [0.0, 0.0, 0.0], [15.0, 14.0, 13.0]],
            dtype=np.float64,
        )
        scalar = np.asarray(
            [
                _local_ccc_at_atom(
                    coord,
                    evaluator._density,
                    evaluator._origin,
                    evaluator._apix,
                    evaluator._kernel,
                    evaluator._kernel_radius_vox,
                )
                for coord in coords
            ],
            dtype=np.float64,
        )
        batched = _local_ccc_batch(
            coords,
            evaluator._density,
            evaluator._origin,
            evaluator._apix,
            evaluator._kernel,
            evaluator._kernel_radius_vox,
            evaluator._density_padded,
            evaluator._density_windows,
            evaluator._kernel_centred_flat,
            evaluator._kernel_norm,
        )
        np.testing.assert_allclose(batched, scalar, atol=1e-12, rtol=1e-12)

    @unittest.skipIf(
        ligand_fitting is None
        or not hasattr(ligand_fitting, "compute_ligand_ccc_decomposed"),
        "ligand_fitting decomposed CCC extension unavailable",
    )
    def test_cpp_ligand_ccc_matches_python_reference(self):
        density = _blob(shape=(20, 21, 22), center=(9.0, 10.0, 11.0), sigma=2.0)
        emmap = _FakeMap(density)
        evaluator = MapMetricEvaluator(
            emmap,
            ligand=None,
            config=SmartLigandRefinementConfig(score_hydrogens=True),
        )
        coords = np.asarray([[9.0, 10.0, 11.0], [10.0, 10.5, 11.5]], dtype=np.float64)
        sim_sub, lo, hi = simulate_ligand_density_subgrid(
            coords,
            np.ones(coords.shape[0], dtype=np.float64),
            emmap.origin,
            emmap.apix,
            emmap.density_map.shape,
            resolution_A=emmap.resolution,
        )
        full = np.zeros_like(emmap.density_map, dtype=np.float64)
        z0, y0, x0 = [int(i) for i in lo]
        z1, y1, x1 = [int(i) for i in hi]
        full[z0:z1, y0:y1, x0:x1] = sim_sub
        exp_sub = emmap.density_map[z0:z1, y0:y1, x0:x1]
        got = ligand_fitting.compute_ligand_ccc_decomposed(
            exp_sub,
            sim_sub,
            int(evaluator._density_nonzero_stats[0]),
            float(evaluator._density_nonzero_stats[1]),
            float(evaluator._density_nonzero_stats[2]),
            int(evaluator._density_finite_stats[0]),
            float(evaluator._density_finite_stats[1]),
            float(evaluator._density_finite_stats[2]),
        )
        self.assertAlmostEqual(got, truncated_cc(emmap.density_map, full), places=10)

    def test_torsion_detection_and_absolute_setter(self):
        mol = _butane()
        coords = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
        torsions = detect_rotatable_torsions(mol, coords)
        self.assertGreaterEqual(len(torsions), 1)

        torsion = torsions[0]
        new_coords = set_torsion_angle(coords, torsion, 60.0)
        new_angle = dihedral_angle_deg(new_coords, torsion.atom_indices)
        delta = abs(((new_angle - 60.0 + 180.0) % 360.0) - 180.0)
        self.assertLess(delta, 1e-6)

    def test_torsion_profile_builder_finds_minima(self):
        mol = _butane()
        coords = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
        torsions = detect_rotatable_torsions(mol, coords)[:1]
        cfg = SmartLigandRefinementConfig(
            torsion_scan_step_deg=30.0,
            max_torsion_profile_count=1,
        )

        profiles = TorsionProfileBuilder(cfg).build_profiles(mol, torsions)
        self.assertIn(torsions[0].torsion_id, profiles)
        self.assertGreaterEqual(len(profiles[torsions[0].torsion_id].minima_deg), 1)

    def test_scan_profile_exposes_openff_minima_and_torsion_quad(self):
        torsion = TorsionInfo(
            torsion_id=7,
            atom_indices=(0, 1, 2, 3),
            bond_atoms=(1, 2),
            downstream_atoms=[2, 3],
            current_angle_deg=0.0,
        )
        cfg = SmartLigandRefinementConfig(max_relative_minimum_energy_kcal=1.0)
        profile = build_profile_from_scan(
            torsion,
            angles=[0, 60, 120, 180, 240, 300],
            energies=[2.0, 0.0, 2.0, 4.0, 0.5, 4.0],
            config=cfg,
            source="openff_openmm",
        )

        self.assertEqual(profile.source, "openff_openmm")
        self.assertEqual(profile.atom_indices, torsion.atom_indices)
        self.assertEqual(profile.minima_deg, [60.0, 240.0])

    def test_torsion_ranking_includes_weak_and_uncertain_bad_atoms(self):
        torsion = TorsionInfo(
            torsion_id=1,
            atom_indices=(0, 1, 2, 3),
            bond_atoms=(1, 2),
            downstream_atoms=[2, 3],
            current_angle_deg=0.0,
        )
        coords = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        metrics = LigandMapMetrics(
            atom_metrics={
                2: AtomMapMetrics(2, 0.20, 0.10, 0.0, np.zeros(3)),
                3: AtomMapMetrics(3, 0.60, 0.10, 0.2, np.zeros(3)),
            },
            ligand_ccc=0.0,
            low_tail_q=0.0,
            mean_q=0.0,
            worst_atom_indices=[2, 3],
        )
        classes = {
            2: AtomClass.WEAK_OR_ABSENT,
            3: AtomClass.UNCERTAIN,
        }
        cfg = SmartLigandRefinementConfig(min_torsion_badness=0.01, target_q=0.75)

        ranked = rank_torsions_by_badness([torsion], coords, metrics, classes, cfg)
        self.assertEqual([t.torsion_id for t in ranked], [1])

    def test_targeted_branch_path_orders_linear_torsions_proximal_to_distal(self):
        coords, torsions, profiles, metrics, classes = _linear_branch_fixture()
        cfg = SmartLigandRefinementConfig(max_branch_torsions=3)

        paths = BranchRebuildMoveGenerator(cfg)._targeted_branch_paths(
            CandidatePose(coords, "initial", {}),
            torsions,
            profiles,
            metrics,
            classes,
        )

        self.assertTrue(paths)
        self.assertEqual([s.torsion.torsion_id for s in paths[0].steps], [0, 1, 2])
        self.assertEqual([s.target_atom for s in paths[0].steps], [3, 4, 5])

    def test_targeted_branch_path_uses_seed_side_not_anchor_side(self):
        coords, torsions, profiles, metrics, classes = _linear_branch_fixture()
        reversed_torsions = [
            TorsionInfo(10, (3, 2, 1, 0), (2, 1), [0, 1], 0.0),
            TorsionInfo(11, (4, 3, 2, 1), (3, 2), [0, 1, 2], 0.0),
            TorsionInfo(12, (5, 4, 3, 2), (4, 3), [0, 1, 2, 3], 0.0),
        ]
        all_torsions = torsions + reversed_torsions
        for t in reversed_torsions:
            profiles[int(t.torsion_id)] = TorsionProfile(
                torsion_id=int(t.torsion_id),
                minima_deg=[60.0],
                relative_energies=[0.0],
                scan_angles_deg=[0.0, 60.0],
                scan_energies=[1.0, 0.0],
                source="test",
                atom_indices=t.atom_indices,
            )
        cfg = SmartLigandRefinementConfig(max_branch_torsions=3)

        paths = BranchRebuildMoveGenerator(cfg)._targeted_branch_paths(
            CandidatePose(coords, "initial", {}),
            all_torsions,
            profiles,
            metrics,
            classes,
        )

        selected = paths[0].steps
        self.assertEqual([s.torsion.torsion_id for s in selected], [0, 1, 2])
        for step in selected:
            self.assertIn(5, step.torsion.downstream_atoms)
            self.assertNotIn(0, step.torsion.downstream_atoms)

    def test_targeted_branch_intermediate_score_keeps_target_atom_improvement(self):
        coords, torsions, profiles, metrics, classes = _linear_branch_fixture()
        torsions = torsions[:1]
        profiles = {0: profiles[0]}
        cfg = SmartLigandRefinementConfig(
            branch_beam_width=1,
            max_branch_torsions=1,
            branch_minimum_offsets_deg=[0.0],
        )
        torsion = torsions[0]

        class AngleMapEvaluator:
            def evaluate(self, cand_coords):
                angle = dihedral_angle_deg(cand_coords, torsion.atom_indices)
                q3 = 0.90 if _angle_delta_deg(60.0, angle) < 1e-4 else 0.20
                atom_metrics = dict(metrics.atom_metrics)
                atom_metrics[3] = AtomMapMetrics(3, q3, q3, 1.0, np.zeros(3))
                return LigandMapMetrics(
                    atom_metrics=atom_metrics,
                    ligand_ccc=-1.0,
                    low_tail_q=-1.0,
                    mean_q=-1.0,
                    worst_atom_indices=[5],
                )

        class AcceptAllGeometry:
            def evaluate(self, candidate):
                candidate.geometry_metrics = {
                    "protein_ligand_clash_score": 0.0,
                    "delta_ligand_internal_energy": 0.0,
                    "max_bond_deviation": 0.0,
                    "max_angle_deviation": 0.0,
                    "has_chirality_error": False,
                    "has_ring_planarity_error": False,
                }
                return candidate.geometry_metrics

            def is_acceptable(self, candidate):
                return True

        class NegativeFinalScorer:
            def __init__(self):
                self.map_metric_evaluator = AngleMapEvaluator()
                self.geometry_oracle = AcceptAllGeometry()

            def score(self, candidate, *_args):
                candidate.map_metrics = self.map_metric_evaluator.evaluate(candidate.coords)
                candidate.geometry_metrics = self.geometry_oracle.evaluate(candidate)
                candidate.score = -10.0
                candidate.move_metadata["score_terms"] = {
                    "delta_ligand_ccc": -1.0,
                }
                return candidate

        out = BranchRebuildMoveGenerator(cfg).generate_scored(
            CandidatePose(coords, "initial", {}),
            torsions,
            profiles,
            metrics,
            classes,
            baseline_geometry={"protein_ligand_clash_score": 0.0},
            scorer=NegativeFinalScorer(),
        )

        self.assertTrue(out)
        sequence = out[0].move_metadata["torsion_sequence"]
        self.assertEqual(len(sequence), 1)
        self.assertAlmostEqual(sequence[0]["new_angle_deg"], 60.0)
        self.assertGreater(out[0].move_metadata["branch_target_score"], 0.0)
        self.assertLess(out[0].score, 0.0)

    def test_branch_rebuild_final_candidate_still_uses_normal_acceptance(self):
        coords, _torsions, _profiles, metrics, classes = _linear_branch_fixture()

        class AcceptAllGeometry:
            def is_acceptable(self, candidate):
                return True

        candidate = CandidatePose(
            coords=coords.copy(),
            move_type="branch_rebuild",
            move_metadata={},
            map_metrics=metrics,
            geometry_metrics={"protein_ligand_clash_score": 0.0},
            score=-1.0,
        )
        decision = MoveAcceptor(
            AcceptAllGeometry(),
            SmartLigandRefinementConfig(min_acceptance_improvement=1e-4),
        ).accept_move(
            CandidatePose(coords, "initial", {}),
            candidate,
            metrics,
            classes,
            old_geometry={"protein_ligand_clash_score": 0.0},
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "score_not_improved")

    def test_exocyclic_ring_flip_candidate_is_added_to_branch_stage(self):
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except Exception:
            self.skipTest("RDKit is required for ring-flip detection")

        mol = Chem.AddHs(Chem.MolFromSmiles("COc1cc(CC)ccc1F"))
        AllChem.EmbedMolecule(mol, randomSeed=19)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
        coords = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
        torsions = detect_rotatable_torsions(mol, coords)
        cfg = SmartLigandRefinementConfig(
            branch_beam_width=8,
            max_branch_torsions=1,
            branch_minimum_offsets_deg=[0.0],
            enable_ring_flip_proposals=True,
        )
        generator = BranchRebuildMoveGenerator(cfg, ligand=mol)
        ring_torsions = [
            t for t in torsions if generator._is_exocyclic_ring_flip_torsion(t)
        ]
        self.assertTrue(ring_torsions)
        torsion = ring_torsions[0]
        seed = next(
            int(i)
            for i in torsion.downstream_atoms
            if mol.GetAtomWithIdx(int(i)).GetAtomicNum() > 1
        )
        metrics = LigandMapMetrics(
            atom_metrics={
                int(i): AtomMapMetrics(int(i), 0.10, 0.10, 1.0, np.zeros(3))
                for i in torsion.downstream_atoms
                if mol.GetAtomWithIdx(int(i)).GetAtomicNum() > 1
            },
            ligand_ccc=0.0,
            low_tail_q=0.10,
            mean_q=0.10,
            worst_atom_indices=[seed],
        )
        classes = {int(i): AtomClass.REPAIR for i in metrics.atom_metrics}
        profile = TorsionProfile(
            torsion_id=int(torsion.torsion_id),
            minima_deg=[dihedral_angle_deg(coords, torsion.atom_indices)],
            relative_energies=[0.0],
            scan_angles_deg=[0.0],
            scan_energies=[0.0],
            source="test",
            atom_indices=torsion.atom_indices,
        )

        candidates = generator.generate(
            CandidatePose(coords, "initial", {}),
            [torsion],
            {int(torsion.torsion_id): profile},
            metrics,
            classes,
        )

        self.assertTrue(candidates)
        kinds = [
            entry["proposal_kind"]
            for cand in candidates
            for entry in cand.move_metadata["torsion_sequence"]
        ]
        self.assertIn("ring_flip", kinds)

    def test_geometry_oracle_rejects_bad_bond_deviation(self):
        mol = _butane()
        coords = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
        cfg = SmartLigandRefinementConfig(max_bond_deviation_A=0.2)
        oracle = GeometryOracle(ligand=mol, reference_coords_A=coords, config=cfg)

        stretched = coords.copy()
        stretched[0, 0] += 1.0
        candidate = CandidatePose(stretched, "test", {})
        oracle.evaluate(candidate)
        self.assertFalse(oracle.is_acceptable(candidate))

    def test_public_api_returns_report_without_map_degradation_loop(self):
        mol = _butane()
        coords = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
        emmap = _FakeMap(_blob())
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SmartLigandRefinementConfig(
                max_macrocycles=0,
                output_dir=tmpdir,
                score_hydrogens=False,
            )
            refined, report = smart_ligand_refine(
                protein=None,
                ligand=mol,
                ligand_coords=coords,
                em_map=emmap,
                openmm_system=None,
                openmm_context=None,
                config=cfg,
            )
            np.testing.assert_allclose(refined, coords)
            self.assertIn("initial_ligand_ccc", report)
            self.assertIn("final_low_tail_q", report)
            self.assertTrue(os.path.exists(report["move_log_path"]))

    def test_profile_timings_emit_without_changing_scores(self):
        mol = _butane()
        coords = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
        emmap = _FakeMap(_blob())
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg = SmartLigandRefinementConfig(
                max_macrocycles=0,
                output_dir=os.path.join(tmpdir, "base"),
                score_hydrogens=False,
            )
            timed_cfg = SmartLigandRefinementConfig(
                max_macrocycles=0,
                output_dir=os.path.join(tmpdir, "timed"),
                score_hydrogens=False,
                profile_timings=True,
            )
            _, base_report = smart_ligand_refine(
                protein=None,
                ligand=mol,
                ligand_coords=coords,
                em_map=emmap,
                openmm_system=None,
                openmm_context=None,
                config=base_cfg,
            )
            _, timed_report = smart_ligand_refine(
                protein=None,
                ligand=mol,
                ligand_coords=coords,
                em_map=emmap,
                openmm_system=None,
                openmm_context=None,
                config=timed_cfg,
            )
            self.assertAlmostEqual(
                timed_report["final_ligand_ccc"],
                base_report["final_ligand_ccc"],
                places=12,
            )
            self.assertIn("timings_s", timed_report)
            self.assertIn("map_metrics", timed_report["timings_s"])

    def test_serial_and_parallel_stage_scoring_match(self):
        mol = _butane()
        coords = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
        emmap = _FakeMap(_blob())
        candidates = [
            CandidatePose(coords + np.asarray([0.05, 0.0, 0.0]), "test", {}),
            CandidatePose(coords + np.asarray([0.0, 0.05, 0.0]), "test", {}),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            def run_stage(max_workers):
                cfg = SmartLigandRefinementConfig(
                    max_macrocycles=0,
                    output_dir=os.path.join(tmpdir, str(max_workers)),
                    score_hydrogens=False,
                    max_parallel_workers=max_workers,
                    no_para=(max_workers == 1),
                    profile_timings=True,
                    clean_accepted_each_macrocycle=False,
                )
                refiner = SmartLigandRefiner(
                    protein=None,
                    ligand=mol,
                    ligand_coords=coords,
                    em_map=emmap,
                    openmm_system=None,
                    openmm_context=None,
                    config=cfg,
                )
                pose = CandidatePose(coords.copy(), "initial", {})
                metrics = refiner.map_metric_evaluator.evaluate(pose.coords)
                classes = refiner.atom_classifier.classify(metrics)
                geometry = refiner.geometry_oracle.evaluate(pose)
                best = refiner._try_stage(
                    0,
                    pose,
                    [CandidatePose(c.coords.copy(), c.move_type, dict(c.move_metadata)) for c in candidates],
                    metrics,
                    classes,
                    geometry,
                )[0]
                return best, refiner._stage_timings[-1]["timings_s"]

            serial, _ = run_stage(1)
            parallel, parallel_timings = run_stage(2)
            self.assertIn(parallel_timings.get("mode"), {"parallel", "serial"})
            if parallel_timings.get("mode") == "serial":
                self.assertIn("parallel_fallback_reason", parallel_timings)
            np.testing.assert_allclose(parallel.coords, serial.coords, atol=1e-9, rtol=0.0)

    def test_progress_events_are_written_to_diagnostics(self):
        mol = _butane()
        coords = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
        emmap = _FakeMap(_blob())
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SmartLigandRefinementConfig(
                max_macrocycles=0,
                output_dir=tmpdir,
                score_hydrogens=False,
                progress=True,
                write_diagnostics=True,
            )
            _, report = smart_ligand_refine(
                protein=None,
                ligand=mol,
                ligand_coords=coords,
                em_map=emmap,
                openmm_system=None,
                openmm_context=None,
                config=cfg,
            )

            self.assertIn("diagnostics_path", report)
            with open(report["diagnostics_path"]) as handle:
                diagnostics = json.load(handle)
            events = diagnostics["events"]
            progress = [event for event in events if event["type"] == "progress"]
            self.assertTrue(progress)
            self.assertTrue(any(event.get("stage") == "torsion_profiles" for event in progress))


if __name__ == "__main__":
    unittest.main()
