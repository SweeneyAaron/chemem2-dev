import argparse
import importlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D


def _prefer_local_chemem():
    root = Path(__file__).resolve().parents[1]
    parent = str(root.parent)
    if parent in sys.path:
        sys.path.remove(parent)
    sys.path.insert(0, parent)

    loaded = sys.modules.get("ChemEM")
    loaded_file = str(getattr(loaded, "__file__", "") or "") if loaded else ""
    if loaded is not None and not loaded_file.startswith(str(root)):
        for name in list(sys.modules):
            if name == "ChemEM" or name.startswith("ChemEM."):
                del sys.modules[name]


def _load_protocol_spec():
    _prefer_local_chemem()
    return importlib.import_module("ChemEM.protocol_spec")


def _load_orchestrator_module():
    _prefer_local_chemem()
    return importlib.import_module("ChemEM.protocols.orchestrator.orchestrator")


def _mol_with_coords(coords):
    coords = np.asarray(coords, dtype=float)
    mol = Chem.MolFromSmiles("CC")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.AddConformer(conf, assignId=True)
    return mol


class FakeLigand:
    def __init__(self, coords):
        self.mol = _mol_with_coords(coords)
        self.complex_structure = SimpleNamespace(
            positions=np.asarray(coords, dtype=float).copy()
        )

    def set_positions(self, coords):
        coords = np.asarray(coords, dtype=float)
        self.complex_structure.positions = coords.copy()
        conf = self.mol.GetConformer(0)
        for i, xyz in enumerate(coords):
            conf.SetAtomPosition(
                i,
                Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            )


def _system(tmpdir, *, final_refiner=None):
    options = SimpleNamespace(local_refine=False)
    if final_refiner is not None:
        options.orch_final_refiner = final_refiner
    logs = []
    return SimpleNamespace(
        options=options,
        output=tmpdir,
        protein=SimpleNamespace(
            complex_structure=SimpleNamespace(
                positions=np.asarray([[10.0, 10.0, 10.0]], dtype=float)
            )
        ),
        ligand=[FakeLigand([[0, 0, 0], [1, 0, 0]])],
        log=logs.append,
        logs=logs,
    )


class TestOrchestratorFinalRefinerArgs(unittest.TestCase):
    def test_orchestrator_final_refiner_parser_defaults_and_aliases(self):
        protocol_spec = _load_protocol_spec()
        parser = argparse.ArgumentParser()
        protocol_spec.add_orchestrate_args(parser)

        self.assertEqual(
            parser.parse_args([]).orch_final_refiner,
            "smart_refine_2",
        )
        self.assertEqual(
            parser.parse_args(["--orch-final-refiner", "search_refine"]).orch_final_refiner,
            "search_refine",
        )
        self.assertEqual(
            parser.parse_args(["--orch-skip-final-refine"]).orch_final_refiner,
            "none",
        )
        self.assertEqual(
            parser.parse_args(["--orch-skip-search-refine"]).orch_final_refiner,
            "none",
        )
        self.assertEqual(parser.parse_args([]).orch_score_mode, "absolute")
        self.assertEqual(
            parser.parse_args(["--orch-score-mode", "coverage"]).orch_score_mode,
            "coverage",
        )
        self.assertEqual(
            parser.parse_args(["--orch-expected-assignments", "7:1"]).orch_expected_assignments,
            "7:1",
        )

    def test_smart_refine2_registry_name_matches_key(self):
        protocol_spec = _load_protocol_spec()
        self.assertEqual(
            protocol_spec.REGISTRY["smart_ligand_refine2"].name,
            "smart_ligand_refine2",
        )


class TestOrchestratorFinalRefinerDispatch(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="orch_final_refiner_")
        self.orch_mod = _load_orchestrator_module()
        self.old_smart = self.orch_mod.SmartRefine2
        self.old_search = self.orch_mod.SearchRefine
        self.old_refine = self.orch_mod.Refine

    def tearDown(self):
        self.orch_mod.SmartRefine2 = self.old_smart
        self.orch_mod.SearchRefine = self.old_search
        self.orch_mod.Refine = self.old_refine
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _orchestrator(self, final_refiner=None):
        system = _system(self.tmpdir, final_refiner=final_refiner)
        orch = self.orch_mod.SmartOrchestrator(system)
        orch._setup_output()
        return orch, system

    def _candidate(self):
        return self.orch_mod.PoseCandidate(
            site_id="S1",
            ligand_idx=0,
            pose_idx=7,
            coords=np.asarray([[2, 2, 2], [3, 3, 3]], dtype=float),
            dock_score=-1.0,
        )

    def test_default_final_refiner_is_smart_refine2_with_isolated_output_and_metrics(self):
        orch, system = self._orchestrator()
        candidate = self._candidate()

        class FakeSmartRefine2:
            calls = 0
            outputs = []

            def __init__(self, system):
                self.system = system
                self.ligands = [
                    SimpleNamespace(
                        _sr2_stop_reason="patience",
                        _sr2_iterations_completed=2,
                        _sr2_no_improve_iters=1,
                    )
                ]

            def run(self):
                type(self).calls += 1
                type(self).outputs.append(getattr(self, "output", None))
                self.system.ligand[0].set_positions([[9, 8, 7], [6, 5, 4]])
                self.fit_results = [
                    SimpleNamespace(
                        best_raw_score=2.0,
                        delta_raw_score=1.5,
                        best_objective=1.8,
                        delta_objective=1.2,
                        best_clash_count=0,
                        steps=4,
                        evaluations=10,
                        score_terms={"final_minimise_energy_kcal": -12.5},
                    )
                ]
                return self.fit_results

        class ShouldNotRun:
            def __init__(self, system):
                self.system = system

            def run(self):
                raise AssertionError("SearchRefine should not run by default")

        self.orch_mod.SmartRefine2 = FakeSmartRefine2
        self.orch_mod.SearchRefine = ShouldNotRun

        orch._final_refine_candidates({"S1": [candidate]})

        self.assertEqual(FakeSmartRefine2.calls, 1)
        self.assertEqual(candidate.stage, "smart_refine_2")
        np.testing.assert_allclose(candidate.coords, [[9, 8, 7], [6, 5, 4]])
        self.assertEqual(
            FakeSmartRefine2.outputs[0],
            os.path.join(
                self.tmpdir,
                "orchestrator",
                "smart_refine_2",
                "site_S1",
                "ligand_0_pose_7",
            ),
        )
        self.assertEqual(candidate.refine_metrics["best_raw_score"], 2.0)
        self.assertEqual(candidate.refine_metrics["delta_raw_score"], 1.5)
        self.assertEqual(candidate.refine_metrics["delta_objective"], 1.2)
        self.assertEqual(candidate.refine_metrics["best_clash_count"], 0)
        self.assertEqual(candidate.refine_metrics["steps"], 4)
        self.assertEqual(candidate.refine_metrics["evaluations"], 10)
        self.assertEqual(candidate.refine_metrics["stop_reason"], "patience")
        self.assertEqual(candidate.refine_metrics["iterations_completed"], 2)
        self.assertEqual(candidate.refine_metrics["no_improve_iters"], 1)
        self.assertEqual(candidate.refine_metrics["final_minimise_energy_kcal"], -12.5)
        self.assertEqual(candidate.to_dict()["refine_metrics"], candidate.refine_metrics)

        np.testing.assert_allclose(
            system.ligand[0].mol.GetConformer(0).GetPositions(),
            [[0, 0, 0], [1, 0, 0]],
        )
        np.testing.assert_allclose(
            system.ligand[0].complex_structure.positions,
            [[0, 0, 0], [1, 0, 0]],
        )
        self.assertFalse(system.options.local_refine)

    def test_search_refine_can_still_be_selected(self):
        orch, _system_obj = self._orchestrator(final_refiner="search_refine")
        candidate = self._candidate()

        class FakeSearchRefine:
            calls = 0

            def __init__(self, system):
                self.system = system

            def run(self):
                type(self).calls += 1
                self.system.ligand[0].set_positions([[4, 4, 4], [5, 5, 5]])

        class ShouldNotRun:
            def __init__(self, system):
                self.system = system

            def run(self):
                raise AssertionError("SmartRefine2 should not run")

        self.orch_mod.SearchRefine = FakeSearchRefine
        self.orch_mod.SmartRefine2 = ShouldNotRun

        orch._final_refine_candidates({"S1": [candidate]})

        self.assertEqual(FakeSearchRefine.calls, 1)
        self.assertEqual(candidate.stage, "search_refined")
        np.testing.assert_allclose(candidate.coords, [[4, 4, 4], [5, 5, 5]])

    def test_none_skips_final_refinement(self):
        orch, _system_obj = self._orchestrator(final_refiner="none")
        candidate = self._candidate()

        class ShouldNotRun:
            def __init__(self, system):
                self.system = system

            def run(self):
                raise AssertionError("final refiner should be skipped")

        self.orch_mod.SmartRefine2 = ShouldNotRun
        self.orch_mod.SearchRefine = ShouldNotRun

        orch._final_refine_candidates({"S1": [candidate]})

        self.assertEqual(candidate.stage, "docked")
        np.testing.assert_allclose(candidate.coords, [[2, 2, 2], [3, 3, 3]])

    def test_explicit_refine_stage_label_and_error_note(self):
        orch, _system_obj = self._orchestrator()
        candidate = self._candidate()

        class FakeRefine:
            def __init__(self, system):
                self.system = system

            def run(self):
                self.system.ligand[0].set_positions([[7, 7, 7], [8, 8, 8]])

        self.orch_mod.Refine = FakeRefine
        orch._refine_one(candidate, refiner_cls=self.orch_mod.Refine)

        self.assertEqual(candidate.stage, "refined")
        np.testing.assert_allclose(candidate.coords, [[7, 7, 7], [8, 8, 8]])

        class Boom:
            def __init__(self, system):
                self.system = system

            def run(self):
                raise RuntimeError("boom")

        failing_candidate = self._candidate()
        orch._refine_one(failing_candidate, refiner_cls=Boom)
        self.assertEqual(failing_candidate.stage, "docked")
        self.assertIn("Boom_error:RuntimeError", failing_candidate.notes)


if __name__ == "__main__":
    unittest.main()
