import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from ChemEM.protocols.orchestrator import io as orch_io
from ChemEM.protocols.orchestrator import scoring
from ChemEM.protocols.orchestrator.orchestrator import SmartOrchestrator
from ChemEM.protocols.orchestrator.state import PoseCandidate


def _mol(smiles, coords):
    mol = Chem.MolFromSmiles(smiles)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(np.asarray(coords, dtype=float)):
        conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.AddConformer(conf, assignId=True)
    return mol


class TestCoverageMetrics(unittest.TestCase):
    def test_larger_ligand_explains_more_site_density_than_compact_ligand(self):
        density = np.zeros((20, 20, 20), dtype=float)
        density[4, 4, 4] = 1.0
        density[4, 4, 12] = 1.0
        site_map = SimpleNamespace(
            density_map=density,
            origin=np.array([0.0, 0.0, 0.0]),
            apix=np.array([1.0, 1.0, 1.0]),
            resolution=2.0,
        )

        small = scoring.density_fit_metrics(
            np.asarray([[4.0, 4.0, 4.0]], dtype=float),
            _mol("C", [[4.0, 4.0, 4.0]]),
            [site_map],
            threshold_frac=0.05,
        )
        large = scoring.density_fit_metrics(
            np.asarray([[4.0, 4.0, 4.0], [12.0, 4.0, 4.0]], dtype=float),
            _mol("CC", [[4.0, 4.0, 4.0], [12.0, 4.0, 4.0]]),
            [site_map],
            threshold_frac=0.05,
        )

        self.assertIsNotNone(small)
        self.assertIsNotNone(large)
        self.assertLess(small["density_coverage"], large["density_coverage"])
        self.assertIn("density_envelope_iou", large)
        self.assertIn("density_excess_fraction", large)

    def test_missing_site_map_returns_none(self):
        metrics = scoring.density_fit_metrics(
            np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            _mol("C", [[0.0, 0.0, 0.0]]),
            None,
        )
        self.assertIsNone(metrics)

    def test_density_sci_and_shape_metrics_are_available(self):
        density = np.zeros((20, 20, 20), dtype=float)
        density[8:12, 8:12, 8:12] = 1.0
        site_map = SimpleNamespace(
            density_map=density,
            origin=np.array([0.0, 0.0, 0.0]),
            apix=np.array([1.0, 1.0, 1.0]),
            resolution=2.0,
        )

        metrics = scoring.density_fit_metrics(
            np.asarray([[10.0, 10.0, 10.0]], dtype=float),
            _mol("C", [[10.0, 10.0, 10.0]]),
            [site_map],
            threshold_frac=0.05,
            compute_sci=True,
            compute_shape=True,
        )

        self.assertIsNotNone(metrics)
        self.assertIn("density_mi", metrics)
        self.assertIn("density_normalized_mi", metrics)
        self.assertEqual(metrics["density_mi_nbins"], 64)
        self.assertIn("density_sci", metrics)
        self.assertIn("density_sci_cc0", metrics)
        self.assertIn("shape_zernike_similarity", metrics)
        self.assertEqual(metrics["shape_zernike_backend"], "internal")
        self.assertIn("density_skeleton_f1", metrics)
        self.assertEqual(
            metrics["density_skeleton_backend"], "skimage_skeletonize_proxy"
        )

    def test_orchestrator_sci_auto_and_shape_mode_helpers(self):
        options = SimpleNamespace(
            orch_score_mode="absolute",
            orch_audit_mode="full",
            orch_density_sci_mode="auto",
            orch_shape_metrics="gate3",
        )
        system = SimpleNamespace(options=options, log=lambda _msg: None)
        orch = SmartOrchestrator(system)

        self.assertTrue(orch._density_sci_enabled())
        self.assertFalse(orch._shape_metrics_enabled("gate1"))
        self.assertTrue(orch._shape_metrics_enabled("gate3"))

        options.orch_density_sci_mode = "off"
        options.orch_shape_metrics = "all"
        self.assertFalse(orch._density_sci_enabled())
        self.assertTrue(orch._shape_metrics_enabled("gate1"))

    def test_skeleton_metric_failure_is_recorded_as_note(self):
        density = np.zeros((20, 20, 20), dtype=float)
        density[8:12, 8:12, 8:12] = 1.0
        site_map = SimpleNamespace(
            density_map=density,
            origin=np.array([0.0, 0.0, 0.0]),
            apix=np.array([1.0, 1.0, 1.0]),
            resolution=2.0,
        )
        ligand = SimpleNamespace(mol=_mol("C", [[10.0, 10.0, 10.0]]))
        options = SimpleNamespace(
            orch_score_mode="absolute",
            orch_audit_mode="full",
            orch_density_sci_mode="off",
            orch_shape_metrics="gate3",
            orch_density_threshold_frac=0.05,
            orch_w_density_coverage=5.0,
            orch_w_density_ccc=1.0,
        )
        system = SimpleNamespace(
            options=options,
            ligand=[ligand],
            binding_site_maps={"S1": [site_map]},
            log=lambda _msg: None,
        )
        candidate = PoseCandidate(
            site_id="S1",
            ligand_idx=0,
            pose_idx=0,
            coords=np.asarray([[10.0, 10.0, 10.0]], dtype=float),
            dock_score=0.0,
        )
        orch = SmartOrchestrator(system)

        with patch(
            "ChemEM.protocols.orchestrator.scoring.density_shape_metrics",
            return_value={"skeleton_metrics_failed": "SyntheticError"},
        ):
            orch._score_density_fit({"S1": [candidate]}, stage_name="gate3")

        self.assertIn("skeleton_metrics_failed:SyntheticError", candidate.notes)


class TestAuditWriters(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="orch_audit_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_candidate_audit_writes_json_csv_and_sdf(self):
        ligand = SimpleNamespace(mol=_mol("CC", [[0, 0, 0], [1, 0, 0]]))
        candidate = PoseCandidate(
            site_id="S1",
            ligand_idx=0,
            pose_idx=2,
            coords=np.asarray([[2, 2, 2], [3, 3, 3]], dtype=float),
            dock_score=-1.5,
            qscore=0.7,
            rank_score=1.25,
            metrics={"q_low_tail": 0.6, "density_coverage": 0.8},
        )

        orch_io.write_audit_candidates(
            {"S1": [candidate]},
            [ligand],
            self.tmpdir,
            "01_gate1_scored",
            include_sdfs=True,
        )

        stage_dir = os.path.join(self.tmpdir, "01_gate1_scored")
        self.assertTrue(os.path.exists(os.path.join(stage_dir, "scores.json")))
        self.assertTrue(os.path.exists(os.path.join(stage_dir, "scores.csv")))
        sdf_path = os.path.join(stage_dir, "sdf", "site_S1", "ligand_0_pose_2.sdf")
        self.assertTrue(os.path.exists(sdf_path))

        with open(os.path.join(stage_dir, "scores.json")) as f:
            payload = json.load(f)
        row = payload["candidates"][0]
        self.assertEqual(row["density_coverage"], 0.8)
        self.assertEqual(row["rank_score"], 1.25)
        self.assertEqual(row["sdf_path"], sdf_path)

    def test_candidate_audit_writes_skeleton_columns(self):
        ligand = SimpleNamespace(mol=_mol("C", [[0, 0, 0]]))
        candidate = PoseCandidate(
            site_id="S1",
            ligand_idx=0,
            pose_idx=0,
            coords=np.asarray([[0, 0, 0]], dtype=float),
            dock_score=0.0,
            metrics={
                "density_skeleton_f1": 0.75,
                "density_skeleton_backend": "skimage_skeletonize_proxy",
                "density_mi": 0.12,
                "density_normalized_mi": 0.34,
                "density_mi_nbins": 64,
            },
        )

        orch_io.write_audit_candidates(
            {"S1": [candidate]},
            [ligand],
            self.tmpdir,
            "skeleton_stage",
            include_sdfs=False,
        )

        csv_path = os.path.join(self.tmpdir, "skeleton_stage", "scores.csv")
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
            row = f.readline().strip().split(",")
        values = dict(zip(header, row))
        self.assertEqual(values["density_mi"], "0.12")
        self.assertEqual(values["density_normalized_mi"], "0.34")
        self.assertEqual(values["density_skeleton_f1"], "0.75")
        self.assertEqual(
            values["density_skeleton_backend"], "skimage_skeletonize_proxy"
        )

    def test_scores_only_candidate_audit_skips_sdf(self):
        ligand = SimpleNamespace(mol=_mol("C", [[0, 0, 0]]))
        candidate = PoseCandidate(
            site_id="S1",
            ligand_idx=0,
            pose_idx=0,
            coords=np.asarray([[0, 0, 0]], dtype=float),
            dock_score=0.0,
        )

        orch_io.write_audit_candidates(
            {"S1": [candidate]},
            [ligand],
            self.tmpdir,
            "scores_only",
            include_sdfs=False,
        )

        stage_dir = os.path.join(self.tmpdir, "scores_only")
        self.assertTrue(os.path.exists(os.path.join(stage_dir, "scores.json")))
        self.assertFalse(os.path.exists(os.path.join(stage_dir, "sdf")))

    def test_assignment_eval_audit_writes_json_and_csv(self):
        rows = [
            {
                "site_id": "7",
                "expected_ligand_idx": 1,
                "actual_ligand_idx": 1,
                "label": "true_positive",
                "assignment_score": 4.2,
            }
        ]
        summary = {"counts": {"true_positive": 1}}
        orch_io.write_assignment_eval(rows, summary, self.tmpdir)

        stage_dir = os.path.join(self.tmpdir, "09_assignment_eval")
        self.assertTrue(os.path.exists(os.path.join(stage_dir, "scores.json")))
        self.assertTrue(os.path.exists(os.path.join(stage_dir, "scores.csv")))
        with open(os.path.join(stage_dir, "scores.json")) as f:
            payload = json.load(f)
        self.assertEqual(payload["summary"]["counts"]["true_positive"], 1)

    def test_binding_site_maps_are_persisted_with_manifest(self):
        class FakeMap:
            density_map = np.zeros((3, 4, 5), dtype=float)
            origin = np.array([1.0, 2.0, 3.0])
            apix = np.array([0.5, 0.5, 0.75])
            resolution = 2.5

            def write_mrc(self, path):
                with open(path, "w") as f:
                    f.write("fake mrc")

        orch_io.write_audit_maps({"S1": [(FakeMap(), {"kind": "feature"})]}, self.tmpdir)

        map_path = os.path.join(self.tmpdir, "maps", "site_S1", "feature_0.mrc")
        manifest_path = os.path.join(self.tmpdir, "maps", "manifest.json")
        self.assertTrue(os.path.exists(map_path))
        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path) as f:
            manifest = json.load(f)
        entry = manifest["maps"][0]
        self.assertEqual(entry["site_id"], "S1")
        self.assertEqual(entry["feature_idx"], 0)
        self.assertEqual(entry["shape"], [3, 4, 5])
        self.assertTrue(entry["written"])


if __name__ == "__main__":
    unittest.main()
