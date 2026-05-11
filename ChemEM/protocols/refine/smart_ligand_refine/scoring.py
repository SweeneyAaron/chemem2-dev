from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .geometry_oracle import GeometryOracle
from .map_metrics import MapMetricEvaluator
from .move_generators import anchor_rmsd
from .types import (
    AtomClass,
    CandidatePose,
    LigandMapMetrics,
    SmartLigandRefinementConfig,
)


def _mean_for_classes(
    metrics: LigandMapMetrics,
    atom_classes: Dict[int, AtomClass],
    classes,
    attr: str,
) -> float:
    vals = []
    wanted = set(classes)
    for atom_idx, atom_cls in atom_classes.items():
        if atom_cls not in wanted:
            continue
        metric = metrics.atom_metrics.get(int(atom_idx))
        if metric is None:
            continue
        val = getattr(metric, attr)
        if val is not None and np.isfinite(float(val)):
            vals.append(float(val))
    return float(np.mean(vals)) if vals else 0.0


def _mean_optional(metrics: LigandMapMetrics, attr: str) -> float:
    vals = []
    for metric in metrics.atom_metrics.values():
        val = getattr(metric, attr)
        if val is not None and np.isfinite(float(val)):
            vals.append(float(val))
    return float(np.mean(vals)) if vals else 0.0


def _difference_support(metrics: LigandMapMetrics) -> float:
    vals = []
    for metric in metrics.atom_metrics.values():
        pos = 0.0 if metric.positive_diff_density is None else metric.positive_diff_density
        neg = 0.0 if metric.negative_diff_density is None else metric.negative_diff_density
        vals.append(float(pos) - float(neg))
    return float(np.mean(vals)) if vals else 0.0


def geometry_penalty(
    geometry_metrics: Optional[dict],
    config: SmartLigandRefinementConfig,
) -> float:
    if not geometry_metrics:
        return 0.0
    penalty = 0.0
    delta_e = float(geometry_metrics.get("delta_ligand_internal_energy", 0.0))
    if delta_e > config.max_ligand_strain_increase_kcal:
        penalty += (delta_e - config.max_ligand_strain_increase_kcal) / max(
            config.max_ligand_strain_increase_kcal, 1e-12
        )
    bond = float(geometry_metrics.get("max_bond_deviation", 0.0))
    if bond > config.max_bond_deviation_A:
        penalty += (bond - config.max_bond_deviation_A) / max(
            config.max_bond_deviation_A, 1e-12
        )
    angle = float(geometry_metrics.get("max_angle_deviation", 0.0))
    if angle > config.max_angle_deviation_deg:
        penalty += (angle - config.max_angle_deviation_deg) / max(
            config.max_angle_deviation_deg, 1e-12
        )
    if geometry_metrics.get("has_chirality_error", False):
        penalty += 10.0
    if geometry_metrics.get("has_ring_planarity_error", False):
        penalty += 5.0
    return float(penalty)


class CandidateScorer:
    def __init__(
        self,
        map_metric_evaluator: MapMetricEvaluator,
        geometry_oracle: GeometryOracle,
        config: Optional[SmartLigandRefinementConfig] = None,
    ):
        self.map_metric_evaluator = map_metric_evaluator
        self.geometry_oracle = geometry_oracle
        self.config = config or SmartLigandRefinementConfig()

    def score(
        self,
        candidate: CandidatePose,
        baseline_pose: CandidatePose,
        baseline_metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
        baseline_geometry: Optional[dict] = None,
    ) -> CandidatePose:
        candidate.map_metrics = self.map_metric_evaluator.evaluate(candidate.coords)
        candidate.geometry_metrics = self.geometry_oracle.evaluate(candidate)

        repair_like = (
            AtomClass.REPAIR,
            AtomClass.WEAK_OR_ABSENT,
            AtomClass.UNCERTAIN,
        )
        old_bad_ccc = _mean_for_classes(
            baseline_metrics, atom_classes, repair_like, "local_ccc"
        )
        new_bad_ccc = _mean_for_classes(
            candidate.map_metrics, atom_classes, repair_like, "local_ccc"
        )

        delta_ligand_ccc = (
            float(candidate.map_metrics.ligand_ccc) - float(baseline_metrics.ligand_ccc)
        )
        delta_low_tail_q = (
            float(candidate.map_metrics.low_tail_q) - float(baseline_metrics.low_tail_q)
        )
        delta_local = float(new_bad_ccc) - float(old_bad_ccc)
        delta_diff = _difference_support(candidate.map_metrics) - _difference_support(
            baseline_metrics
        )
        delta_half = _mean_optional(
            candidate.map_metrics, "halfmap_agreement"
        ) - _mean_optional(baseline_metrics, "halfmap_agreement")
        anchor_disruption = anchor_rmsd(
            baseline_pose.coords, candidate.coords, atom_classes
        )
        geom_pen = geometry_penalty(candidate.geometry_metrics, self.config)

        old_clash = (
            0.0
            if baseline_geometry is None
            else float(baseline_geometry.get("protein_ligand_clash_score", 0.0))
        )
        new_clash = float(
            candidate.geometry_metrics.get("protein_ligand_clash_score", 0.0)
        )
        clash_penalty = max(0.0, new_clash - old_clash)

        candidate.score = float(
            (self.config.w_ccc * delta_ligand_ccc)
            + (self.config.w_q * delta_low_tail_q)
            + (self.config.w_local * delta_local)
            + (self.config.w_diff * delta_diff)
            + (self.config.w_half * delta_half)
            - (self.config.w_anchor * anchor_disruption)
            - (self.config.w_geom * geom_pen)
            - (self.config.w_clash * clash_penalty)
        )
        candidate.move_metadata["score_terms"] = {
            "delta_ligand_ccc": float(delta_ligand_ccc),
            "delta_low_tail_q": float(delta_low_tail_q),
            "delta_bad_atom_local_ccc": float(delta_local),
            "delta_difference_density_explained": float(delta_diff),
            "delta_halfmap_agreement": float(delta_half),
            "anchor_disruption": float(anchor_disruption),
            "geometry_penalty": float(geom_pen),
            "clash_penalty": float(clash_penalty),
        }
        return candidate


@dataclass
class AcceptDecision:
    accepted: bool
    reason: Optional[str] = None


class MoveAcceptor:
    def __init__(
        self,
        geometry_oracle: GeometryOracle,
        config: Optional[SmartLigandRefinementConfig] = None,
    ):
        self.geometry_oracle = geometry_oracle
        self.config = config or SmartLigandRefinementConfig()

    def accept_move(
        self,
        old_pose: CandidatePose,
        new_pose: CandidatePose,
        old_metrics: LigandMapMetrics,
        atom_classes: Dict[int, AtomClass],
        old_geometry: Optional[dict] = None,
    ) -> AcceptDecision:
        is_branch = str(getattr(new_pose, "move_type", "")) == "branch_rebuild"
        if is_branch:
            anchor_cap = float(
                getattr(
                    self.config,
                    "branch_max_anchor_rmsd",
                    self.config.max_anchor_rmsd,
                )
            )
            score_floor = float(
                getattr(
                    self.config,
                    "branch_min_acceptance_improvement",
                    self.config.min_acceptance_improvement,
                )
            )
        else:
            anchor_cap = float(self.config.max_anchor_rmsd)
            score_floor = float(self.config.min_acceptance_improvement)

        if not self.geometry_oracle.is_acceptable(new_pose):
            return AcceptDecision(False, "geometry_rejected")

        rmsd = anchor_rmsd(old_pose.coords, new_pose.coords, atom_classes)
        if rmsd > anchor_cap:
            return AcceptDecision(False, "anchor_rmsd_exceeded")

        if new_pose.map_metrics is None:
            return AcceptDecision(False, "missing_map_metrics")

        low_tail_delta = (
            float(new_pose.map_metrics.low_tail_q) - float(old_metrics.low_tail_q)
        )
        if low_tail_delta < -float(self.config.max_low_tail_q_degrade):
            return AcceptDecision(False, "low_tail_q_degraded")

        if old_geometry is not None and new_pose.geometry_metrics is not None:
            old_clash = float(old_geometry.get("protein_ligand_clash_score", 0.0))
            new_clash = float(
                new_pose.geometry_metrics.get("protein_ligand_clash_score", 0.0)
            )
            if (new_clash - old_clash) > float(self.config.max_clash_score_increase):
                return AcceptDecision(False, "clash_increased")

        if float(new_pose.score or 0.0) < score_floor:
            if is_branch and bool(
                getattr(self.config, "branch_accept_on_q_improvement", True)
            ):
                mean_delta = (
                    float(new_pose.map_metrics.mean_q) - float(old_metrics.mean_q)
                )
                if low_tail_delta > 0.0 and mean_delta > 0.0:
                    return AcceptDecision(True, None)
            return AcceptDecision(False, "score_not_improved")

        return AcceptDecision(True, None)
