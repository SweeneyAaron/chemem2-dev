# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class RefinedPose:
    score: float
    position: np.ndarray
    final_score: float
    sci: float
    energy_kcal: float
    clash_penalty: float


@dataclass
class ProposalRecord:
    outer_iter: int
    proposal_idx: int
    positions_nm: np.ndarray
    ligand_coords_A: np.ndarray
    score: float
    score_terms: Dict[str, Any] = field(default_factory=dict)
    energy_kcal: float = 0.0
    clash_penalty: float = 0.0
    final_score: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "outer_iter": int(self.outer_iter),
            "proposal_idx": int(self.proposal_idx),
            "positions_nm": self.positions_nm,
            "ligand_coords_A": self.ligand_coords_A,
            "score": float(self.score),
            "score_terms": self.score_terms,
            "energy_kcal": float(self.energy_kcal),
            "clash_penalty": float(self.clash_penalty),
            "final_score": float(self.final_score),
        }
