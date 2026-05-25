# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Dataclasses passed between SmartOrchestrator stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PoseCandidate:
    """One docked / refined / final-refined ligand pose at one site."""

    site_id: str
    ligand_idx: int
    pose_idx: int
    coords: np.ndarray
    dock_score: float
    qscore: Optional[float] = None
    mmgbsa: Optional[float] = None
    rank_score: Optional[float] = None
    stage: str = "docked"   # "docked" | "refined" | "search_refined" | "smart_refine_2"
    notes: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    refine_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "site_id": str(self.site_id),
            "ligand_idx": int(self.ligand_idx),
            "pose_idx": int(self.pose_idx),
            "dock_score": float(self.dock_score),
            "qscore": None if self.qscore is None else float(self.qscore),
            "mmgbsa": None if self.mmgbsa is None else float(self.mmgbsa),
            "rank_score": None if self.rank_score is None else float(self.rank_score),
            "stage": self.stage,
            "notes": list(self.notes),
            "metrics": dict(self.metrics),
            "refine_metrics": dict(self.refine_metrics),
        }


@dataclass
class FinalAssignment:
    """One (site -> ligand pose) assignment in the converged assembly."""

    site_id: str
    ligand_idx: int
    pose_idx: int
    coords: np.ndarray
    qscore: float
    mmgbsa: Optional[float]
    composite: float
    stage: str
    chain_id: str
    metrics: dict = field(default_factory=dict)
    rank_score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "site_id": str(self.site_id),
            "ligand_idx": int(self.ligand_idx),
            "pose_idx": int(self.pose_idx),
            "qscore": float(self.qscore),
            "mmgbsa": None if self.mmgbsa is None else float(self.mmgbsa),
            "composite": float(self.composite),
            "stage": self.stage,
            "chain_id": self.chain_id,
            "metrics": dict(self.metrics),
            "rank_score": None if self.rank_score is None else float(self.rank_score),
        }


@dataclass
class AssignmentRejection:
    """One site for which Gate 3 found a best pose but made no assignment."""

    site_id: str
    best_ligand_idx: Optional[int]
    best_pose_idx: Optional[int]
    qscore: Optional[float]
    mmgbsa: Optional[float]
    assignment_score: Optional[float]
    assignment_margin: Optional[float]
    rejection_reason: str
    stage: str
    metrics: dict = field(default_factory=dict)
    rank_score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "site_id": str(self.site_id),
            "best_ligand_idx": (
                None if self.best_ligand_idx is None else int(self.best_ligand_idx)
            ),
            "best_pose_idx": (
                None if self.best_pose_idx is None else int(self.best_pose_idx)
            ),
            "qscore": None if self.qscore is None else float(self.qscore),
            "mmgbsa": None if self.mmgbsa is None else float(self.mmgbsa),
            "assignment_score": (
                None if self.assignment_score is None else float(self.assignment_score)
            ),
            "assignment_margin": (
                None if self.assignment_margin is None else float(self.assignment_margin)
            ),
            "rejection_reason": self.rejection_reason,
            "stage": self.stage,
            "metrics": dict(self.metrics),
            "rank_score": None if self.rank_score is None else float(self.rank_score),
        }
