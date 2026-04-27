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
    """One docked / refined / search-refined ligand pose at one site."""

    site_id: str
    ligand_idx: int
    pose_idx: int
    coords: np.ndarray
    dock_score: float
    qscore: Optional[float] = None
    mmgbsa: Optional[float] = None
    stage: str = "docked"   # "docked" | "refined" | "search_refined"
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "site_id": str(self.site_id),
            "ligand_idx": int(self.ligand_idx),
            "pose_idx": int(self.pose_idx),
            "dock_score": float(self.dock_score),
            "qscore": None if self.qscore is None else float(self.qscore),
            "mmgbsa": None if self.mmgbsa is None else float(self.mmgbsa),
            "stage": self.stage,
            "notes": list(self.notes),
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
        }
