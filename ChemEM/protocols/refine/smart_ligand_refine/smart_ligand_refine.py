from __future__ import annotations

import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ChemEM.messages import Messages

from .atom_classifier import AtomClassifier
from .diagnostics import compare_reference_sdf, read_sdf_coords
from .geometry_oracle import GeometryOracle
from .logging import RefinementLogger, _jsonable
from .map_metrics import MapMetricEvaluator
from .move_generators import (
    BranchRebuildMoveGenerator,
    RigidMicroJiggleMoveGenerator,
    SubregionRigidBodyMoveGenerator,
    TorsionMinimaMoveGenerator,
    anchor_rmsd,
    rank_torsions_by_badness,
)
from .scoring import CandidateScorer, MoveAcceptor
from .torsion_profiles import TorsionProfileBuilder, detect_rotatable_torsions
from .types import (
    AtomClass,
    CandidatePose,
    LigandMapMetrics,
    MoveLogEntry,
    SmartLigandRefinementConfig,
)

try:
    from openmm import unit
except Exception:  # pragma: no cover
    unit = None


class _SharedMapProxy:
    def __init__(self, density_map, origin, apix, resolution):
        self.density_map = density_map
        self.origin = np.asarray(origin, dtype=np.float64)
        self.apix = np.asarray(apix, dtype=np.float64)
        self.resolution = float(resolution)


_SCORING_WORKER = {}
_SCORING_WORKER_SHMS = []


def _attach_shared_map(spec):
    if spec is None:
        return None
    shm = shared_memory.SharedMemory(name=str(spec["name"]))
    _SCORING_WORKER_SHMS.append(shm)
    arr = np.ndarray(
        tuple(int(i) for i in spec["shape"]),
        dtype=np.dtype(spec["dtype"]),
        buffer=shm.buf,
    )
    return _SharedMapProxy(
        arr,
        np.asarray(spec["origin"], dtype=np.float64),
        np.asarray(spec["apix"], dtype=np.float64),
        float(spec["resolution"]),
    )


def _build_worker_openmm_context(payload):
    if not payload:
        return None, None, None
    try:
        from openmm import Context, Platform, VerletIntegrator, XmlSerializer, unit as omm_unit

        system = XmlSerializer.deserialize(str(payload["system_xml"]))
        integrator = VerletIntegrator(1.0 * omm_unit.femtoseconds)
        platform_name = payload.get("platform_name")
        try:
            platform = Platform.getPlatformByName(str(platform_name))
            context = Context(system, integrator, platform)
        except Exception:
            context = Context(system, integrator)
        context.setPositions(np.asarray(payload["positions_nm"], dtype=np.float64) * omm_unit.nanometer)
        return system, context, integrator
    except Exception:
        return None, None, None


def _init_scoring_worker(
    primary_map_spec,
    half_map_specs,
    difference_map_spec,
    ligand_payload,
    protein_payload,
    config,
    openmm_payload,
    baseline_pose,
    baseline_metrics,
    atom_classes,
    baseline_geometry,
):
    em_map = _attach_shared_map(primary_map_spec)
    half_maps = [_attach_shared_map(spec) for spec in list(half_map_specs or [])]
    difference_map = _attach_shared_map(difference_map_spec)
    openmm_system, openmm_context, integrator = _build_worker_openmm_context(openmm_payload)
    geometry_oracle = GeometryOracle(
        protein=protein_payload,
        ligand=ligand_payload,
        openmm_system=openmm_system,
        openmm_context=openmm_context,
        ligand_context_indices=getattr(config, "ligand_context_indices", None),
        protein_coords_A=getattr(config, "protein_coords_A", None),
        reference_coords_A=np.asarray(baseline_pose.coords, dtype=np.float64),
        config=config,
    )
    map_metric_evaluator = MapMetricEvaluator(
        em_map,
        ligand=ligand_payload,
        half_maps=half_maps,
        difference_map=difference_map,
        protein_coords_A=getattr(config, "protein_coords_A", None),
        config=config,
    )
    _SCORING_WORKER.clear()
    _SCORING_WORKER.update(
        {
            "geometry_oracle": geometry_oracle,
            "map_metric_evaluator": map_metric_evaluator,
            "candidate_scorer": CandidateScorer(map_metric_evaluator, geometry_oracle, config),
            "move_acceptor": MoveAcceptor(geometry_oracle, config),
            "baseline_pose": baseline_pose,
            "baseline_metrics": baseline_metrics,
            "atom_classes": atom_classes,
            "baseline_geometry": baseline_geometry,
            "clean_candidates": bool(getattr(config, "clean_candidates", False)),
            "openmm_integrator": integrator,
        }
    )


def _score_candidate_worker(index_candidate):
    index, candidate = index_candidate
    cand = candidate
    evaluator = _SCORING_WORKER.get("map_metric_evaluator")
    before_timings = dict(getattr(evaluator, "_timings", {})) if evaluator is not None else {}
    before_counts = dict(getattr(evaluator, "_call_counts", {})) if evaluator is not None else {}
    try:
        geometry_oracle = _SCORING_WORKER["geometry_oracle"]
        if _SCORING_WORKER.get("clean_candidates", False):
            cand = geometry_oracle.clean(cand)
        cand = _SCORING_WORKER["candidate_scorer"].score(
            cand,
            _SCORING_WORKER["baseline_pose"],
            _SCORING_WORKER["baseline_metrics"],
            _SCORING_WORKER["atom_classes"],
            _SCORING_WORKER["baseline_geometry"],
        )
        decision = _SCORING_WORKER["move_acceptor"].accept_move(
            _SCORING_WORKER["baseline_pose"],
            cand,
            _SCORING_WORKER["baseline_metrics"],
            _SCORING_WORKER["atom_classes"],
            _SCORING_WORKER["baseline_geometry"],
        )
        timing_delta = {}
        if evaluator is not None and getattr(evaluator, "_profile_timings", False):
            timing_delta = {
                "timings_s": {
                    key: float(value - before_timings.get(key, 0.0))
                    for key, value in getattr(evaluator, "_timings", {}).items()
                },
                "call_counts": {
                    key: int(value - before_counts.get(key, 0))
                    for key, value in getattr(evaluator, "_call_counts", {}).items()
                },
            }
        return int(index), cand, bool(decision.accepted), decision.reason, timing_delta
    except Exception as exc:
        candidate.score = float("-inf")
        return int(index), candidate, False, f"evaluation_failed:{type(exc).__name__}", {}


def _ligand_coords(ligand, ligand_coords) -> np.ndarray:
    if ligand_coords is not None:
        return np.asarray(ligand_coords, dtype=np.float64)
    if hasattr(ligand, "get_positions"):
        coords = ligand.get_positions()
        if np.asarray(coords).size:
            return np.asarray(coords, dtype=np.float64)
    mol = getattr(ligand, "mol", ligand)
    if mol is not None and hasattr(mol, "GetNumConformers") and mol.GetNumConformers():
        return np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
    raise ValueError("ligand_coords are required when ligand has no conformer")


def _geometry_energy(metrics: Optional[dict]) -> float:
    if not metrics:
        return 0.0
    return float(metrics.get("openmm_energy", metrics.get("ligand_internal_energy", 0.0)))


def _clash(metrics: Optional[dict]) -> float:
    if not metrics:
        return 0.0
    return float(metrics.get("protein_ligand_clash_score", 0.0))


class SmartLigandRefiner:
    def __init__(
        self,
        protein,
        ligand,
        ligand_coords,
        em_map,
        openmm_system=None,
        openmm_context=None,
        *,
        half_maps=None,
        config: Optional[SmartLigandRefinementConfig] = None,
        ligand_context_indices: Optional[Sequence[int]] = None,
        protein_coords_A: Optional[np.ndarray] = None,
        difference_map=None,
    ):
        self.protein = protein
        self.ligand = ligand
        self.initial_coords = _ligand_coords(ligand, ligand_coords)
        self.em_map = em_map
        self.openmm_system = openmm_system
        self.openmm_context = openmm_context
        self.half_maps = list(half_maps or [])
        self.config = config or SmartLigandRefinementConfig()
        if ligand_context_indices is not None:
            self.config.ligand_context_indices = list(ligand_context_indices)
        if protein_coords_A is not None:
            self.config.protein_coords_A = np.asarray(protein_coords_A, dtype=np.float64)
        self.difference_map = difference_map

        self.map_metric_evaluator = MapMetricEvaluator(
            em_map,
            ligand=ligand,
            half_maps=self.half_maps,
            difference_map=difference_map,
            protein_coords_A=self.config.protein_coords_A,
            config=self.config,
        )
        self.atom_classifier = AtomClassifier(self.config)
        self.geometry_oracle = GeometryOracle(
            protein=protein,
            ligand=ligand,
            openmm_system=openmm_system,
            openmm_context=openmm_context,
            ligand_context_indices=self.config.ligand_context_indices,
            protein_coords_A=self.config.protein_coords_A,
            reference_coords_A=self.initial_coords,
            config=self.config,
        )
        self.candidate_scorer = CandidateScorer(
            self.map_metric_evaluator,
            self.geometry_oracle,
            self.config,
        )
        self.move_acceptor = MoveAcceptor(self.geometry_oracle, self.config)
        self.logger = RefinementLogger(
            self.config.output_dir,
            self.config.move_log_name,
            self.config.diagnostics_name,
        )
        self._debug_printer = self.config.debug_printer or print
        self._stage_timings: List[dict] = []

    def _profiling(self) -> bool:
        return bool(getattr(self.config, "profile_timings", False))

    @staticmethod
    def _map_shared_spec(map_obj):
        if map_obj is None or not hasattr(map_obj, "density_map"):
            return None, None
        density = np.ascontiguousarray(np.asarray(map_obj.density_map, dtype=np.float64))
        shm = shared_memory.SharedMemory(create=True, size=int(density.nbytes))
        arr = np.ndarray(density.shape, dtype=density.dtype, buffer=shm.buf)
        arr[:] = density
        resolution = getattr(map_obj, "resolution", 3.0)
        try:
            resolution = float(resolution)
        except Exception:
            resolution = 3.0
        spec = {
            "name": shm.name,
            "shape": tuple(int(i) for i in density.shape),
            "dtype": str(density.dtype),
            "origin": np.asarray(getattr(map_obj, "origin", np.zeros(3)), dtype=np.float64),
            "apix": np.asarray(getattr(map_obj, "apix", np.ones(3)), dtype=np.float64),
            "resolution": float(resolution),
        }
        return spec, shm

    def _openmm_worker_payload(self) -> Optional[dict]:
        if self.openmm_system is None or self.openmm_context is None or unit is None:
            return None
        try:
            from openmm import XmlSerializer

            state = self.openmm_context.getState(getPositions=True)
            positions_nm = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=np.float64,
            )
            platform_name = self.openmm_context.getPlatform().getName()
            return {
                "system_xml": XmlSerializer.serialize(self.openmm_system),
                "positions_nm": positions_nm,
                "platform_name": str(platform_name),
            }
        except Exception:
            return None

    def _parallel_worker_count(self, n_candidates: int) -> int:
        if int(n_candidates) < 2 or bool(getattr(self.config, "no_para", False)):
            return 1
        requested = getattr(self.config, "max_parallel_workers", None)
        if requested is None:
            cpus_per_site = max(1, int(getattr(self.config, "cpus_per_site", 1) or 1))
            requested = max(1, (os.cpu_count() or 1) // cpus_per_site)
        return max(1, min(int(requested), int(n_candidates)))

    def _serial_evaluate_candidates(
        self,
        pose: CandidatePose,
        candidates: Sequence[CandidatePose],
        baseline_metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
        baseline_geometry: Optional[dict],
        stage_timing: dict,
    ) -> List[Tuple[CandidatePose, bool, Optional[str]]]:
        evaluated: List[Tuple[CandidatePose, bool, Optional[str]]] = []
        for candidate in candidates:
            cand_t0 = time.perf_counter()
            cand = candidate
            if self.config.clean_candidates:
                clean_t0 = time.perf_counter()
                cand = self.geometry_oracle.clean(cand)
                if self._profiling():
                    stage_timing["clean_s"] = float(
                        stage_timing.get("clean_s", 0.0)
                        + (time.perf_counter() - clean_t0)
                    )
            try:
                score_t0 = time.perf_counter()
                cand = self.candidate_scorer.score(
                    cand,
                    pose,
                    baseline_metrics,
                    atom_classes,
                    baseline_geometry,
                )
                decision = self.move_acceptor.accept_move(
                    pose,
                    cand,
                    baseline_metrics,
                    atom_classes,
                    baseline_geometry,
                )
                if self._profiling():
                    stage_timing["score_accept_s"] = float(
                        stage_timing.get("score_accept_s", 0.0)
                        + (time.perf_counter() - score_t0)
                    )
                evaluated.append((cand, decision.accepted, decision.reason))
            except Exception as exc:
                candidate.score = float("-inf")
                evaluated.append((candidate, False, f"evaluation_failed:{type(exc).__name__}"))
            if self._profiling():
                stage_timing["candidate_body_s"] = float(
                    stage_timing.get("candidate_body_s", 0.0)
                    + (time.perf_counter() - cand_t0)
                )
        return evaluated

    def _parallel_evaluate_candidates(
        self,
        pose: CandidatePose,
        candidates: Sequence[CandidatePose],
        baseline_metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
        baseline_geometry: Optional[dict],
        max_workers: int,
        stage_timing: dict,
    ) -> List[Tuple[CandidatePose, bool, Optional[str]]]:
        shms = []
        try:
            primary_spec, shm = self._map_shared_spec(self.em_map)
            if shm is not None:
                shms.append(shm)
            half_specs = []
            for hmap in self.half_maps:
                spec, shm = self._map_shared_spec(hmap)
                if spec is not None:
                    half_specs.append(spec)
                if shm is not None:
                    shms.append(shm)
            diff_spec, shm = self._map_shared_spec(self.difference_map)
            if shm is not None:
                shms.append(shm)
            worker_config = replace(self.config, debug_printer=None)
            openmm_payload = self._openmm_worker_payload()
            baseline_pose = CandidatePose(
                coords=np.asarray(pose.coords, dtype=np.float64),
                move_type=pose.move_type,
                move_metadata=dict(pose.move_metadata),
                map_metrics=baseline_metrics,
                geometry_metrics=baseline_geometry,
                score=pose.score,
            )
            results = [None] * len(candidates)
            worker_timings: Dict[str, float] = {}
            worker_counts: Dict[str, int] = {}
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_scoring_worker,
                initargs=(
                    primary_spec,
                    half_specs,
                    diff_spec,
                    self.ligand,
                    self.protein,
                    worker_config,
                    openmm_payload,
                    baseline_pose,
                    baseline_metrics,
                    atom_classes,
                    baseline_geometry,
                ),
            ) as pool:
                futures = [
                    pool.submit(_score_candidate_worker, (idx, cand))
                    for idx, cand in enumerate(candidates)
                ]
                for fut in as_completed(futures):
                    idx, cand, accepted, reason, timing_delta = fut.result()
                    results[int(idx)] = (cand, bool(accepted), reason)
                    if self._profiling() and timing_delta:
                        for key, value in dict(timing_delta.get("timings_s", {})).items():
                            worker_timings[str(key)] = float(
                                worker_timings.get(str(key), 0.0) + float(value)
                            )
                        for key, value in dict(timing_delta.get("call_counts", {})).items():
                            worker_counts[str(key)] = int(
                                worker_counts.get(str(key), 0) + int(value)
                            )
            if any(item is None for item in results):
                raise RuntimeError("parallel candidate scoring returned incomplete results")
            if self._profiling():
                stage_timing["worker_map_timings_s"] = worker_timings
                stage_timing["worker_map_call_counts"] = worker_counts
            return list(results)
        finally:
            for shm in shms:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass

    def _debug(self, message: str) -> None:
        if not bool(self.config.debug):
            return
        self._debug_printer(f"[smart_ligand_refine] {message}")

    def _progress(self, message: str, **payload) -> None:
        self.logger.diagnostic("progress", message=message, **payload)
        if not (bool(self.config.progress) or bool(self.config.debug)):
            return
        self._debug_printer(f"[smart_ligand_refine] {message}")

    def _diagnostic(self, event_type: str, **payload) -> None:
        self.logger.diagnostic(event_type, **payload)

    def _beam_progress(self, macro_idx: int, event_type: str, **payload) -> None:
        macrocycle = int(macro_idx + 1)
        self._diagnostic(event_type, macrocycle=macrocycle, **payload)
        if event_type == "branch_beam":
            self._progress(
                "Branch walk depth "
                f"{payload.get('beam_depth')}: generated "
                f"{payload.get('candidates_generated')}, kept "
                f"{payload.get('candidates_kept')}, best score "
                f"{payload.get('best_score')}",
                macrocycle=macrocycle,
                stage="branch_beam",
                **payload,
            )

    @staticmethod
    def _format_metrics(metrics: LigandMapMetrics) -> str:
        return (
            f"CCC={metrics.ligand_ccc:.4f}, "
            f"mean Q={metrics.mean_q:.4f}, "
            f"low-tail Q={metrics.low_tail_q:.4f}, "
            f"worst atoms={metrics.worst_atom_indices[:5]}"
        )

    @staticmethod
    def _format_class_counts(atom_classes: Dict[int, AtomClass]) -> str:
        counts = Counter(cls.value for cls in atom_classes.values())
        return (
            f"anchors={counts.get(AtomClass.ANCHOR.value, 0)}, "
            f"repair={counts.get(AtomClass.REPAIR.value, 0)}, "
            f"weak/absent={counts.get(AtomClass.WEAK_OR_ABSENT.value, 0)}, "
            f"uncertain={counts.get(AtomClass.UNCERTAIN.value, 0)}"
        )

    @staticmethod
    def _format_score_terms(candidate: CandidatePose) -> str:
        terms = (candidate.move_metadata or {}).get("score_terms", {})
        if not terms:
            return ""
        return (
            f"dCCC={terms.get('delta_ligand_ccc', 0.0):+.4f}, "
            f"dLowQ={terms.get('delta_low_tail_q', 0.0):+.4f}, "
            f"dLocal={terms.get('delta_bad_atom_local_ccc', 0.0):+.4f}, "
            f"anchorRMSD={terms.get('anchor_disruption', 0.0):.3f} A, "
            f"geomPenalty={terms.get('geometry_penalty', 0.0):.3f}, "
            f"clashPenalty={terms.get('clash_penalty', 0.0):.3f}"
        )

    @staticmethod
    def _atom_metrics_payload(
        metrics: LigandMapMetrics,
        atom_classes: Optional[Dict[int, AtomClass]] = None,
    ) -> List[dict]:
        out = []
        atom_classes = atom_classes or {}
        for atom_idx, metric in sorted(metrics.atom_metrics.items()):
            out.append(
                {
                    "atom_index": int(atom_idx),
                    "class": (
                        atom_classes[int(atom_idx)].value
                        if int(atom_idx) in atom_classes
                        else None
                    ),
                    "q_score": float(metric.q_score),
                    "local_ccc": float(metric.local_ccc),
                    "map_value": float(metric.map_value),
                    "map_gradient": np.asarray(metric.map_gradient, dtype=np.float64),
                    "halfmap_agreement": metric.halfmap_agreement,
                    "positive_diff_density": metric.positive_diff_density,
                    "negative_diff_density": metric.negative_diff_density,
                }
            )
        return out

    @staticmethod
    def _torsion_profiles_payload(torsion_profiles: Dict[int, object]) -> List[dict]:
        out = []
        for torsion_id, profile in sorted(torsion_profiles.items()):
            out.append(
                {
                    "torsion_id": int(torsion_id),
                    "source": str(getattr(profile, "source", "unknown")),
                    "atom_indices": getattr(profile, "atom_indices", None),
                    "minima_deg": list(getattr(profile, "minima_deg", [])),
                    "relative_energies": list(getattr(profile, "relative_energies", [])),
                }
            )
        return out

    def run(self) -> Tuple[np.ndarray, dict]:
        pose = CandidatePose(
            coords=self.initial_coords.copy(),
            move_type="initial",
            move_metadata={},
        )
        pose.map_metrics = self.map_metric_evaluator.evaluate(pose.coords)
        pose.geometry_metrics = self.geometry_oracle.evaluate(pose)
        initial_metrics = pose.map_metrics
        initial_atom_classes = self.atom_classifier.classify(initial_metrics)
        self._progress(
            "Starting near-fit ligand repair: "
            f"{pose.coords.shape[0]} atoms, {self._format_metrics(initial_metrics)}",
            macrocycle=0,
            stage="start",
            atom_count=int(pose.coords.shape[0]),
            ligand_ccc=float(initial_metrics.ligand_ccc),
            low_tail_q=float(initial_metrics.low_tail_q),
            mean_q=float(initial_metrics.mean_q),
        )
        self._debug(
            "Starting near-fit ligand repair: "
            f"{pose.coords.shape[0]} atoms, {self._format_metrics(initial_metrics)}"
        )
        self._diagnostic(
            "reference_comparison",
            phase="initial",
            comparison=compare_reference_sdf(
                self.ligand,
                pose.coords,
                self.config.reference_sdf_path,
            ),
        )
        self._diagnostic(
            "atom_metrics",
            phase="initial",
            ligand_metrics={
                "ligand_ccc": float(initial_metrics.ligand_ccc),
                "low_tail_q": float(initial_metrics.low_tail_q),
                "mean_q": float(initial_metrics.mean_q),
                "worst_atom_indices": list(initial_metrics.worst_atom_indices),
            },
            atoms=self._atom_metrics_payload(initial_metrics, initial_atom_classes),
        )
        reference_coords = (
            read_sdf_coords(self.config.reference_sdf_path)
            if self.config.reference_sdf_path
            else None
        )
        if reference_coords is not None:
            try:
                reference_metrics = self.map_metric_evaluator.evaluate(reference_coords)
                reference_classes = self.atom_classifier.classify(reference_metrics)
                self._diagnostic(
                    "reference_pose_map_metrics",
                    ligand_metrics={
                        "ligand_ccc": float(reference_metrics.ligand_ccc),
                        "low_tail_q": float(reference_metrics.low_tail_q),
                        "mean_q": float(reference_metrics.mean_q),
                        "worst_atom_indices": list(reference_metrics.worst_atom_indices),
                    },
                    atoms=self._atom_metrics_payload(reference_metrics, reference_classes),
                )
            except Exception as exc:
                self._diagnostic(
                    "reference_pose_map_metrics",
                    error=type(exc).__name__,
                )

        torsions = detect_rotatable_torsions(self.ligand, pose.coords)
        torsion_profile_builder = TorsionProfileBuilder(
            self.config,
            geometry_oracle=self.geometry_oracle,
        )
        torsion_profiles = torsion_profile_builder.build_profiles(
            self.ligand,
            torsions,
            self.openmm_system,
            self.openmm_context,
        )
        self._debug(
            f"Detected {len(torsions)} rotatable torsion directions; "
            f"built {len(torsion_profiles)} torsion profiles."
        )
        self._progress(
            f"Detected {len(torsions)} torsion directions; "
            f"built {len(torsion_profiles)} torsion profiles.",
            macrocycle=0,
            stage="torsion_profiles",
            torsions_detected=int(len(torsions)),
            torsion_profiles_built=int(len(torsion_profiles)),
        )
        self._diagnostic(
            "torsion_profiles",
            torsions_detected=int(len(torsions)),
            profiles=self._torsion_profiles_payload(torsion_profiles),
        )

        accepted_moves = 0
        rejected_moves = 0
        previous_metrics = initial_metrics

        for macro_idx in range(int(self.config.max_macrocycles)):
            schedule = self.config.macrocycle_schedule[
                min(macro_idx, len(self.config.macrocycle_schedule) - 1)
            ]
            metrics = self.map_metric_evaluator.evaluate(pose.coords)
            atom_classes = self.atom_classifier.classify(metrics)
            geometry = self.geometry_oracle.evaluate(pose)
            self._debug(
                f"Macrocycle {macro_idx + 1}/{self.config.max_macrocycles} "
                f"({schedule.get('name', 'repair')}): "
                f"{self._format_metrics(metrics)}; "
                f"{self._format_class_counts(atom_classes)}; "
                f"geometry energy={_geometry_energy(geometry):.2f} kcal/mol"
            )
            self._progress(
                f"Macrocycle {macro_idx + 1}/{self.config.max_macrocycles} "
                f"({schedule.get('name', 'repair')}): {self._format_metrics(metrics)}; "
                f"{self._format_class_counts(atom_classes)}",
                macrocycle=int(macro_idx + 1),
                stage=str(schedule.get("name", "repair")),
                ligand_ccc=float(metrics.ligand_ccc),
                low_tail_q=float(metrics.low_tail_q),
                mean_q=float(metrics.mean_q),
                class_counts=dict(Counter(cls.value for cls in atom_classes.values())),
                geometry_energy=float(_geometry_energy(geometry)),
            )
            self._diagnostic(
                "atom_metrics",
                phase=f"macrocycle_{macro_idx + 1}_start",
                atoms=self._atom_metrics_payload(metrics, atom_classes),
            )

            rigid = RigidMicroJiggleMoveGenerator(
                self.config,
                translations_A=schedule.get("rigid_translations_A"),
                rotations_deg=schedule.get("rigid_rotations_deg"),
            )
            pose, acc, rej = self._try_stage(
                macro_idx,
                pose,
                rigid.generate(pose, metrics, atom_classes),
                metrics,
                atom_classes,
                geometry,
            )
            accepted_moves += acc
            rejected_moves += rej

            metrics = self.map_metric_evaluator.evaluate(pose.coords)
            atom_classes = self.atom_classifier.classify(metrics)
            geometry = self.geometry_oracle.evaluate(pose)

            if self.config.enable_subregion_proposals:
                subregion_candidates = SubregionRigidBodyMoveGenerator(
                    self.config
                ).generate(
                    self.ligand,
                    pose,
                    metrics,
                    atom_classes,
                )
                pose, acc, rej = self._try_stage(
                    macro_idx,
                    pose,
                    subregion_candidates,
                    metrics,
                    atom_classes,
                    geometry,
                )
                accepted_moves += acc
                rejected_moves += rej

            metrics = self.map_metric_evaluator.evaluate(pose.coords)
            atom_classes = self.atom_classifier.classify(metrics)
            geometry = self.geometry_oracle.evaluate(pose)

            ranked_torsions = rank_torsions_by_badness(
                torsions,
                pose.coords,
                metrics,
                atom_classes,
                self.config,
            )[: int(schedule.get("max_torsions", 4))]
            self._debug(
                f"Ranked {len(ranked_torsions)} torsion(s) for repair in this macrocycle."
            )
            self._progress(
                f"Ranked {len(ranked_torsions)} torsion(s) for repair.",
                macrocycle=int(macro_idx + 1),
                stage="torsion_ranking",
                torsions_ranked=int(len(ranked_torsions)),
                torsion_ids=[int(t.torsion_id) for t in ranked_torsions],
            )
            self._diagnostic(
                "torsion_ranking",
                macrocycle=int(macro_idx + 1),
                torsion_ids=[int(t.torsion_id) for t in ranked_torsions],
                torsions=[
                    {
                        "torsion_id": int(t.torsion_id),
                        "atom_indices": tuple(int(i) for i in t.atom_indices),
                        "bond_atoms": tuple(int(i) for i in t.bond_atoms),
                        "downstream_atoms": [int(i) for i in t.downstream_atoms],
                    }
                    for t in ranked_torsions
                ],
            )
            torsion_generator = TorsionMinimaMoveGenerator(
                self.config,
                offsets_deg=schedule.get("torsion_offsets_deg"),
            )
            for torsion in ranked_torsions:
                profile = torsion_profiles.get(int(torsion.torsion_id))
                if profile is None:
                    continue
                metrics = self.map_metric_evaluator.evaluate(pose.coords)
                atom_classes = self.atom_classifier.classify(metrics)
                geometry = self.geometry_oracle.evaluate(pose)
                candidates = torsion_generator.generate(
                    pose,
                    torsion,
                    profile,
                    metrics,
                    atom_classes,
                )
                pose, acc, rej = self._try_stage(
                    macro_idx,
                    pose,
                    candidates,
                    metrics,
                    atom_classes,
                    geometry,
                )
                accepted_moves += acc
                rejected_moves += rej

            if self.config.enable_branch_rebuild:
                metrics = self.map_metric_evaluator.evaluate(pose.coords)
                atom_classes = self.atom_classifier.classify(metrics)
                geometry = self.geometry_oracle.evaluate(pose)
                clean_fn = self.geometry_oracle.clean if self.config.clean_candidates else None
                branch_candidates = BranchRebuildMoveGenerator(
                    self.config,
                    ligand=self.ligand,
                ).generate_scored(
                    pose,
                    torsions,
                    torsion_profiles,
                    metrics,
                    atom_classes,
                    geometry,
                    self.candidate_scorer,
                    clean_fn=clean_fn,
                    progress_callback=lambda event_type, **payload: self._beam_progress(
                        macro_idx,
                        event_type,
                        **payload,
                    ),
                )
                pose, acc, rej = self._try_stage(
                    macro_idx,
                    pose,
                    branch_candidates,
                    metrics,
                    atom_classes,
                    geometry,
                )
                accepted_moves += acc
                rejected_moves += rej

            if self.config.clean_accepted_each_macrocycle:
                metrics = self.map_metric_evaluator.evaluate(pose.coords)
                atom_classes = self.atom_classifier.classify(metrics)
                geometry = self.geometry_oracle.evaluate(pose)
                clean_candidate = self.geometry_oracle.clean(
                    CandidatePose(
                        coords=pose.coords,
                        move_type="openmm_no_map_clean",
                        move_metadata={},
                    )
                )
                pose, acc, rej = self._try_stage(
                    macro_idx,
                    pose,
                    [clean_candidate],
                    metrics,
                    atom_classes,
                    geometry,
                )
                accepted_moves += acc
                rejected_moves += rej

            new_metrics = self.map_metric_evaluator.evaluate(pose.coords)
            if self._converged(previous_metrics, new_metrics):
                self._debug(
                    "Converged: map metrics changed less than the configured "
                    f"tolerances ({self._format_metrics(new_metrics)})."
                )
                previous_metrics = new_metrics
                break
            self._debug(
                f"End macrocycle {macro_idx + 1}: {self._format_metrics(new_metrics)}"
            )
            previous_metrics = new_metrics

        final_metrics = self.map_metric_evaluator.evaluate(pose.coords)
        final_geometry = self.geometry_oracle.evaluate(pose)
        log_path = self.logger.write()
        final_reference_comparison = compare_reference_sdf(
            self.ligand,
            pose.coords,
            self.config.reference_sdf_path,
        )
        self._diagnostic(
            "reference_comparison",
            phase="final",
            comparison=final_reference_comparison,
        )
        final_atom_classes = self.atom_classifier.classify(final_metrics)
        self._diagnostic(
            "atom_metrics",
            phase="final",
            ligand_metrics={
                "ligand_ccc": float(final_metrics.ligand_ccc),
                "low_tail_q": float(final_metrics.low_tail_q),
                "mean_q": float(final_metrics.mean_q),
                "worst_atom_indices": list(final_metrics.worst_atom_indices),
            },
            atoms=self._atom_metrics_payload(final_metrics, final_atom_classes),
        )
        report = {
            "initial_ligand_ccc": float(initial_metrics.ligand_ccc),
            "final_ligand_ccc": float(final_metrics.ligand_ccc),
            "initial_low_tail_q": float(initial_metrics.low_tail_q),
            "final_low_tail_q": float(final_metrics.low_tail_q),
            "accepted_moves": int(accepted_moves),
            "rejected_moves": int(rejected_moves),
            "final_openmm_energy": float(_geometry_energy(final_geometry)),
            "move_log_path": str(log_path),
        }
        if self._profiling():
            report["timings_s"] = {
                "map_metrics": self.map_metric_evaluator.timing_report(),
                "torsion_profiles": torsion_profile_builder.timing_report(),
                "stages": list(self._stage_timings),
            }
        if final_reference_comparison:
            report["reference_comparison"] = final_reference_comparison
        if bool(self.config.write_diagnostics):
            diagnostics_path = self.logger.write_diagnostics(report)
            report["diagnostics_path"] = str(diagnostics_path)
        self._debug(
            "Finished: "
            f"CCC {report['initial_ligand_ccc']:.4f} -> {report['final_ligand_ccc']:.4f}, "
            f"low-tail Q {report['initial_low_tail_q']:.4f} -> "
            f"{report['final_low_tail_q']:.4f}, "
            f"accepted={accepted_moves}, rejected={rejected_moves}. "
            f"Move log: {log_path}"
        )
        return np.asarray(pose.coords, dtype=np.float64), report

    def _try_stage(
        self,
        macro_idx: int,
        pose: CandidatePose,
        candidates: Sequence[CandidatePose],
        baseline_metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
        baseline_geometry: Optional[dict],
    ) -> Tuple[CandidatePose, int, int]:
        if not candidates:
            self._diagnostic(
                "stage_summary",
                macrocycle=int(macro_idx + 1),
                stage="empty",
                candidates_generated=0,
                candidates_scored=0,
                accepted=0,
                rejected=0,
            )
            return pose, 0, 0

        stage_name = str(candidates[0].move_type)
        self._debug(
            f"  Stage '{stage_name}': evaluating {len(candidates)} candidate move(s)."
        )

        stage_wall_t0 = time.perf_counter()
        stage_timing = {
            "mode": "serial",
            "candidate_count": int(len(candidates)),
        }
        max_workers = self._parallel_worker_count(len(candidates))
        if max_workers > 1:
            parallel_t0 = time.perf_counter()
            try:
                evaluated = self._parallel_evaluate_candidates(
                    pose,
                    candidates,
                    baseline_metrics,
                    atom_classes,
                    baseline_geometry,
                    max_workers,
                    stage_timing,
                )
                stage_timing["mode"] = "parallel"
                stage_timing["workers"] = int(max_workers)
                if self._profiling():
                    stage_timing["parallel_score_s"] = float(
                        time.perf_counter() - parallel_t0
                    )
            except Exception as exc:
                if self._profiling():
                    stage_timing["parallel_fallback_reason"] = type(exc).__name__
                    stage_timing["parallel_fallback_detail"] = str(exc)
                evaluated = self._serial_evaluate_candidates(
                    pose,
                    candidates,
                    baseline_metrics,
                    atom_classes,
                    baseline_geometry,
                    stage_timing,
                )
        else:
            evaluated = self._serial_evaluate_candidates(
                pose,
                candidates,
                baseline_metrics,
                atom_classes,
                baseline_geometry,
                stage_timing,
            )

        if self._profiling():
            stage_timing["wall_s"] = float(time.perf_counter() - stage_wall_t0)

        accepted = [
            cand
            for cand, ok, _ in evaluated
            if ok and cand.score is not None and np.isfinite(float(cand.score))
        ]
        best = max(accepted, key=lambda c: float(c.score), default=None)
        best_scored = max(
            (
                cand
                for cand, _, _ in evaluated
                if cand.score is not None and np.isfinite(float(cand.score))
            ),
            key=lambda c: float(c.score),
            default=None,
        )

        if bool(self.config.debug):
            ranked_debug = sorted(
                evaluated,
                key=lambda item: float(item[0].score)
                if item[0].score is not None and np.isfinite(float(item[0].score))
                else float("-inf"),
                reverse=True,
            )[: max(0, int(self.config.debug_candidate_limit))]
            for rank, (cand, ok, reason) in enumerate(ranked_debug, start=1):
                status = "acceptable" if ok else f"rejected: {reason}"
                score = float(cand.score) if cand.score is not None else float("nan")
                self._debug(
                    f"    candidate {rank}: {cand.move_type}, score={score:+.5f}, "
                    f"{status}; {self._format_score_terms(cand)}"
                )

        accepted_count = 0
        rejected_count = 0
        rejection_reasons = Counter()
        for cand, ok, reason in evaluated:
            is_best = best is cand
            if ok and is_best:
                accepted_count += 1
                self._log_candidate(
                    macro_idx,
                    pose,
                    cand,
                    baseline_metrics,
                    atom_classes,
                    baseline_geometry,
                    accepted=True,
                    reason=None,
                )
            else:
                rejected_count += 1
                rejection_reasons[reason if not ok else "not_best_candidate"] += 1
                self._log_candidate(
                    macro_idx,
                    pose,
                    cand,
                    baseline_metrics,
                    atom_classes,
                    baseline_geometry,
                    accepted=False,
                    reason=reason if not ok else "not_best_candidate",
                )

        progress_best = best or best_scored
        stage_payload = {
            "macrocycle": int(macro_idx + 1),
            "stage": stage_name,
            "candidates_generated": int(len(candidates)),
            "candidates_scored": int(len(evaluated)),
            "accepted": int(accepted_count),
            "rejected": int(rejected_count),
            "rejection_counts": dict(rejection_reasons),
            "best_score": (
                None
                if progress_best is None or progress_best.score is None
                else float(progress_best.score)
            ),
            "best_low_tail_q": (
                None
                if progress_best is None or progress_best.map_metrics is None
                else float(progress_best.map_metrics.low_tail_q)
            ),
            "best_ligand_ccc": (
                None
                if progress_best is None or progress_best.map_metrics is None
                else float(progress_best.map_metrics.ligand_ccc)
            ),
        }
        if self._profiling():
            stage_payload["timings_s"] = dict(stage_timing)
            self._stage_timings.append(dict(stage_payload))
        self._diagnostic("stage_summary", **stage_payload)
        self._progress(
            f"Stage '{stage_name}': generated {len(candidates)}, "
            f"accepted {accepted_count}, rejected {rejected_count}, "
            f"best score {stage_payload['best_score']}",
            **stage_payload,
        )

        if best is not None:
            self._debug(
                f"  Accepted '{best.move_type}' with score={float(best.score):+.5f}; "
                f"{self._format_score_terms(best)}"
            )
            return best, accepted_count, rejected_count
        if rejection_reasons:
            summary = ", ".join(
                f"{reason}={count}" for reason, count in rejection_reasons.most_common()
            )
            self._debug(f"  No move accepted in '{stage_name}'. Rejections: {summary}.")
        return pose, 0, rejected_count

    def _log_candidate(
        self,
        macro_idx: int,
        old_pose: CandidatePose,
        candidate: CandidatePose,
        old_metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
        old_geometry: Optional[dict],
        *,
        accepted: bool,
        reason: Optional[str],
    ) -> None:
        new_metrics = candidate.map_metrics or old_metrics
        new_geometry = candidate.geometry_metrics or {}
        metadata = candidate.move_metadata or {}
        self.logger.log(
            MoveLogEntry(
                macrocycle=int(macro_idx),
                move_type=str(candidate.move_type),
                accepted=bool(accepted),
                torsion_id=(
                    None
                    if metadata.get("torsion_id") is None
                    else int(metadata.get("torsion_id"))
                ),
                old_angle_deg=(
                    None
                    if metadata.get("old_angle_deg") is None
                    else float(metadata.get("old_angle_deg"))
                ),
                new_angle_deg=(
                    None
                    if metadata.get("new_angle_deg") is None
                    else float(metadata.get("new_angle_deg"))
                ),
                delta_ligand_ccc=float(new_metrics.ligand_ccc - old_metrics.ligand_ccc),
                delta_low_tail_q=float(new_metrics.low_tail_q - old_metrics.low_tail_q),
                delta_mean_q=float(new_metrics.mean_q - old_metrics.mean_q),
                anchor_rmsd=float(anchor_rmsd(old_pose.coords, candidate.coords, atom_classes)),
                delta_openmm_energy=float(
                    _geometry_energy(new_geometry) - _geometry_energy(old_geometry)
                ),
                clash_delta=float(_clash(new_geometry) - _clash(old_geometry)),
                rejection_reason=reason,
                move_metadata=dict(metadata),
            )
        )

    def _converged(
        self,
        old_metrics: LigandMapMetrics,
        new_metrics: LigandMapMetrics,
    ) -> bool:
        delta_ccc = abs(float(new_metrics.ligand_ccc) - float(old_metrics.ligand_ccc))
        delta_low = abs(float(new_metrics.low_tail_q) - float(old_metrics.low_tail_q))
        return (
            delta_ccc < float(self.config.convergence_ligand_ccc_delta)
            and delta_low < float(self.config.convergence_low_tail_q_delta)
        )


def smart_ligand_refine(
    protein,
    ligand,
    ligand_coords,
    em_map,
    openmm_system,
    openmm_context,
    half_maps=None,
    config=None,
    *,
    ligand_context_indices: Optional[Sequence[int]] = None,
    protein_coords_A: Optional[np.ndarray] = None,
    difference_map=None,
):
    """
    Refine a near-fit ligand in a cryoEM map using external map scoring,
    Q-score-guided local search, torsion-minima jumps, and OpenMM/OpenFF
    geometry validation.

    Density scoring remains external to OpenMM. OpenMM is only used by the
    GeometryOracle as a no-map geometry sanity check / optional minimiser.

    Returns
    -------
    refined_coords : np.ndarray
    report : dict
    """

    refiner = SmartLigandRefiner(
        protein,
        ligand,
        ligand_coords,
        em_map,
        openmm_system,
        openmm_context,
        half_maps=half_maps,
        config=config,
        ligand_context_indices=ligand_context_indices,
        protein_coords_A=protein_coords_A,
        difference_map=difference_map,
    )
    return refiner.run()


class SmartLigandRefinement:
    """ChemEM protocol wrapper for SmartLigandRefiner."""

    def __init__(self, system):
        self.system = system
        self.output = None

    def _opt(self, name: str, default):
        return getattr(self.system.options, name, default)

    @staticmethod
    def _first_map_like(obj):
        if obj is None:
            return None
        if hasattr(obj, "density_map"):
            return obj
        if isinstance(obj, (list, tuple)):
            for item in obj:
                if hasattr(item, "density_map"):
                    return item
        return None

    def _select_map(self):
        if bool(self._opt("no_map", False)):
            return None
        mode = str(self._opt("slr_map_source", "confidence")).lower()
        density_map = self._first_map_like(getattr(self.system, "density_map", None))
        confidence_map = self._first_map_like(getattr(self.system, "confidence_map", None))
        difference_map = self._first_map_like(getattr(self.system, "difference_map", None))
        if mode == "raw":
            return density_map or confidence_map or difference_map
        if mode == "difference":
            return difference_map or confidence_map or density_map
        return confidence_map or density_map or difference_map

    def _difference_map(self):
        return self._first_map_like(getattr(self.system, "difference_map", None))

    def _config(self, ligand_output: str) -> SmartLigandRefinementConfig:
        cfg = SmartLigandRefinementConfig(
            output_dir=ligand_output,
            max_macrocycles=int(self._opt("slr_max_macrocycles", 3)),
            anchor_q_min=float(self._opt("slr_anchor_q_min", 0.75)),
            anchor_local_ccc_min=float(self._opt("slr_anchor_local_ccc_min", 0.60)),
            repair_q_max=float(self._opt("slr_repair_q_max", 0.55)),
            min_halfmap_agreement=float(self._opt("slr_min_halfmap_agreement", 0.50)),
            min_density_value=float(self._opt("slr_min_density_value", 1e-6)),
            target_q=float(self._opt("slr_target_q", 0.75)),
            min_torsion_badness=float(self._opt("slr_min_torsion_badness", 0.05)),
            torsion_profile_source=str(self._opt("slr_torsion_profile_source", "auto")),
            clean_candidates=bool(self._opt("slr_clean_candidates", False)),
            clean_accepted_each_macrocycle=bool(
                self._opt("slr_clean_each_macrocycle", True)
            ),
            enable_branch_rebuild=bool(self._opt("slr_branch_rebuild", False)),
            branch_beam_width=int(self._opt("slr_branch_beam_width", 16)),
            max_branch_torsions=int(self._opt("slr_max_branch_torsions", 6)),
            branch_minimum_offsets_deg=list(
                self._opt("slr_branch_minimum_offsets_deg", None)
                or [-20.0, -10.0, 0.0, 10.0, 20.0]
            ),
            enable_ring_flip_proposals=bool(
                self._opt("slr_enable_ring_flip_proposals", True)
            ),
            branch_rebuild_verbose=bool(
                self._opt("slr_branch_rebuild_verbose", False)
            ),
            branch_aggressive_angle_search=bool(
                self._opt("slr_branch_aggressive_angle_search", False)
            ),
            branch_aggressive_grid_step_deg=float(
                self._opt("slr_branch_aggressive_grid_step_deg", 30.0)
            ),
            branch_aggressive_flip_offsets_deg=list(
                self._opt("slr_branch_aggressive_flip_offsets_deg", None)
                or [-120.0, -60.0, 60.0, 120.0, 180.0]
            ),
            branch_relative_q_seeding=bool(
                self._opt("slr_branch_relative_q_seeding", False)
            ),
            branch_relative_q_percentile=float(
                self._opt("slr_branch_relative_q_percentile", 0.25)
            ),
            branch_use_anchor_seeds=bool(
                self._opt("slr_branch_use_anchor_seeds", False)
            ),
            branch_min_acceptance_improvement=float(
                self._opt("slr_branch_min_acceptance_improvement", -1e-3)
            ),
            branch_max_anchor_rmsd=float(
                self._opt("slr_branch_max_anchor_rmsd", 0.6)
            ),
            branch_accept_on_q_improvement=bool(
                self._opt("slr_branch_accept_on_q_improvement", True)
            ),
            max_candidates_per_stage=int(self._opt("slr_max_candidates_per_stage", 256)),
            max_torsion_profile_count=int(self._opt("slr_max_torsion_profile_count", 64)),
            torsion_scan_step_deg=float(self._opt("slr_torsion_scan_step_deg", 10.0)),
            enable_subregion_proposals=bool(
                self._opt("slr_enable_subregion_proposals", True)
            ),
            write_accepted_sdf=bool(self._opt("slr_write_accepted_sdf", False)),
            protein_ligand_clash_distance_A=float(
                self._opt("slr_clash_distance_a", 1.6)
            ),
            reference_sdf_path=self._opt("slr_reference_sdf", None),
            write_diagnostics=bool(self._opt("slr_write_diagnostics", False)),
            progress=bool(self._opt("slr_progress", False)),
            debug=bool(self._opt("slr_debug", False)),
            debug_candidate_limit=int(self._opt("slr_debug_candidate_limit", 5)),
            profile_timings=bool(self._opt("slr_profile_timings", False)),
            no_para=bool(self._opt("no_para", False)),
            cpus_per_site=int(getattr(self.system, "CPUS_PER_SITE", 1) or 1),
            max_parallel_workers=(
                None
                if int(self._opt("ncpu", 0) or 0) <= 0
                else int(self._opt("ncpu", 0))
            ),
            debug_printer=self.system.log,
        )
        if bool(self._opt("slr_smoke_test", False)):
            cfg.max_macrocycles = min(int(cfg.max_macrocycles), 1)
            cfg.branch_beam_width = min(int(cfg.branch_beam_width), 2)
            cfg.max_branch_torsions = min(int(cfg.max_branch_torsions), 2)
            cfg.max_candidates_per_stage = min(int(cfg.max_candidates_per_stage), 16)
            cfg.max_torsion_profile_count = min(int(cfg.max_torsion_profile_count), 8)
            cfg.torsion_scan_step_deg = max(float(cfg.torsion_scan_step_deg), 60.0)
            cfg.branch_minimum_offsets_deg = [0.0]
            cfg.enable_subregion_proposals = False
            cfg.clean_candidates = False
            cfg.clean_accepted_each_macrocycle = False
        return cfg

    def _prepare_output(self) -> None:
        self.output = os.path.join(self.system.output, "smart_ligand_refine")
        os.makedirs(self.output, exist_ok=True)

    def _build_geometry_env(self, ligand, coords_A):
        if not bool(self._opt("slr_use_openmm_geometry", True)):
            return None, None, None, None
        try:
            from ChemEM.protocols.core.simulation import identify_indices_from_name
            from ChemEM.protocols.refine.pose_minimiser import ChemEMSimulationSetup
            from ChemEM.protocols.refine.refine_utils import (
                get_residue_positions,
                get_residue_subset_from_points,
            )

            ligand.set_positions(np.asarray(coords_A, dtype=np.float64))
            ligand_points = get_residue_positions(ligand.complex_structure.residues[0])
            protein_structure = self.system.protein.complex_structure
            local_structure = get_residue_subset_from_points(
                ligand_points,
                protein_structure,
                distance_cutoff=float(self._opt("slr_pocket_radius", 12.0)),
            )
            env = ChemEMSimulationSetup(
                protein_structure=local_structure,
                ligand_structure=[ligand],
                density_map=None,
                platform_name=getattr(self.system, "platform", "CPU"),
                protein_restraint="protein",
                pin_k=float(self._opt("slr_pin_k", 5000.0)),
                localise=False,
                global_k=0.0,
                pin_specs=getattr(self.system.options, "pin_specs", []),
                distance_specs=getattr(self.system.options, "distance_specs", []),
            )
            if unit is not None:
                state = env.simulation.context.getState(getPositions=True)
                pos_nm = np.asarray(
                    state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                    dtype=np.float64,
                )
                pos_nm[np.asarray(env.all_ligand_indices, dtype=int)] = (
                    np.asarray(coords_A, dtype=np.float64) * 0.1
                )
                env.simulation.context.setPositions(pos_nm * unit.nanometer)

                protein_indices = [
                    atom.idx
                    for atom in env.complex_structure.atoms
                    if (not str(atom.residue.name).startswith("LIG"))
                    and atom.element != 1
                ]
                protein_coords_A = pos_nm[np.asarray(protein_indices, dtype=int)] * 10.0
            else:
                protein_coords_A = None
            return (
                env.complex_system,
                env.simulation.context,
                list(env.all_ligand_indices),
                protein_coords_A,
            )
        except Exception as exc:
            self.system.log(
                Messages.chemem_warning(
                    self.__class__.__name__,
                    "_build_geometry_env",
                    f"OpenMM geometry environment unavailable ({exc}); using RDKit/simple checks",
                )
            )
            return None, None, None, None

    def run(self) -> None:
        self.system.log(Messages.create_centered_box("Smart Ligand Refinement"))
        self._prepare_output()
        em_map = self._select_map()
        if em_map is None:
            self.system.log("[smart_ligand_refine] no map available; skipping")
            return

        summaries = []
        for lig_idx, ligand in enumerate(self.system.ligand):
            t0 = time.perf_counter()
            lig_output = os.path.join(self.output, f"Ligand_{lig_idx}")
            os.makedirs(lig_output, exist_ok=True)
            self.system.log(f"[smart_ligand_refine] ligand {lig_idx}: start")
            try:
                coords = _ligand_coords(ligand, None)
                openmm_system, context, lig_context_idx, protein_coords = (
                    self._build_geometry_env(ligand, coords)
                )
                cfg = self._config(lig_output)
                cfg.ligand_context_indices = lig_context_idx
                cfg.protein_coords_A = protein_coords
                refined_coords, report = smart_ligand_refine(
                    protein=self.system.protein,
                    ligand=ligand,
                    ligand_coords=coords,
                    em_map=em_map,
                    openmm_system=openmm_system,
                    openmm_context=context,
                    half_maps=getattr(self.system, "half_maps", None),
                    config=cfg,
                    ligand_context_indices=lig_context_idx,
                    protein_coords_A=protein_coords,
                    difference_map=self._difference_map(),
                )
                ligand.set_positions(refined_coords)
                if hasattr(ligand, "write_sdf"):
                    ligand.write_sdf(os.path.join(lig_output, "smart_refined.sdf"))
                report["ligand_index"] = int(lig_idx)
                report["runtime_s"] = float(time.perf_counter() - t0)
                summaries.append(report)
                self.system.log(
                    "[smart_ligand_refine] ligand "
                    f"{lig_idx}: CCC {report['initial_ligand_ccc']:.4f} -> "
                    f"{report['final_ligand_ccc']:.4f}; low-tail Q "
                    f"{report['initial_low_tail_q']:.4f} -> "
                    f"{report['final_low_tail_q']:.4f}; accepted "
                    f"{report['accepted_moves']}"
                )
                if bool(cfg.profile_timings):
                    timings = report.get("timings_s", {})
                    map_timings = (
                        timings.get("map_metrics", {}).get("timings_s", {})
                        if isinstance(timings, dict)
                        else {}
                    )
                    torsion_timings = (
                        timings.get("torsion_profiles", {}).get("timings_s", {})
                        if isinstance(timings, dict)
                        else {}
                    )
                    stage_total = sum(
                        float(stage.get("timings_s", {}).get("wall_s", 0.0))
                        for stage in timings.get("stages", [])
                    ) if isinstance(timings, dict) else 0.0
                    self.system.log(
                        "[smart_ligand_refine] ligand "
                        f"{lig_idx}: runtime_s={report['runtime_s']:.3f}; "
                        f"map_evaluate_s={float(map_timings.get('evaluate.total', 0.0)):.3f}; "
                        f"stage_wall_s={stage_total:.3f}; "
                        f"torsion_scan_s={float(torsion_timings.get('scan.total', 0.0)):.3f}"
                    )
            except Exception as exc:
                self.system.log(
                    Messages.chemem_warning(
                        self.__class__.__name__,
                        "run",
                        f"ligand {lig_idx}: {exc}",
                    )
                )

        with open(os.path.join(self.output, "summary.json"), "w") as handle:
            json.dump(_jsonable(summaries), handle, indent=2)
