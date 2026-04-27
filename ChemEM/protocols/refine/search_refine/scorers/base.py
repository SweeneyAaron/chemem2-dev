# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

import numpy as np


class BaseScorer(ABC):
    """
    Abstract scorer. Higher score == better fit. The orchestrator uses
    ``atom_gradient`` (gradient of ``score``) to decide pull direction,
    and ``score`` to rank / accept proposals.

    Subclasses may override ``atom_gradient`` when they can compute it
    analytically or per-atom (e.g. Q-score).
    """

    name: ClassVar[str] = "base"

    def __init__(self, options: Any):
        self.options = options

    def _opt(self, name: str, default: Any) -> Any:
        return getattr(self.options, name, default)

    def prepare(
        self,
        env: Any,
        local_map: Any,
        heavy_masses: np.ndarray,
        protein_heavy_indices: Any,
    ) -> None:
        """Cache per-ligand state (maps, indices, KDTrees). Called once per ligand."""
        self.env = env
        self.local_map = local_map
        self.heavy_masses = np.asarray(heavy_masses, dtype=np.float64)
        self.protein_heavy_indices = protein_heavy_indices

    @abstractmethod
    def score(
        self,
        heavy_coords_A: np.ndarray,
        terms_out: Optional[dict] = None,
    ) -> float:
        """Return scalar score for ligand heavy-atom coordinates (Å). Higher = better."""

    def _fd_step_a(self) -> float:
        return float(self._opt("sr_fd_step_a", 0.05))

    def _fd_mode(self) -> str:
        return str(self._opt("sr_fd_mode", "central")).lower()

    def atom_gradient(self, heavy_coords_A: np.ndarray) -> np.ndarray:
        """Central (or forward) finite-difference gradient of ``score``."""
        coords = np.asarray(heavy_coords_A, dtype=np.float64).copy()
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"heavy_coords_A must be (N,3), got {coords.shape}")

        n = coords.shape[0]
        grad = np.zeros((n, 3), dtype=np.float64)
        step = max(float(self._fd_step_a()), 1e-6)
        mode = self._fd_mode()

        if mode == "forward":
            base = float(self.score(coords))
            for i in range(n):
                for k in range(3):
                    probe = coords.copy()
                    probe[i, k] += step
                    sp = float(self.score(probe))
                    grad[i, k] = (sp - base) / step
        else:
            for i in range(n):
                for k in range(3):
                    plus = coords.copy()
                    minus = coords.copy()
                    plus[i, k] += step
                    minus[i, k] -= step
                    sp = float(self.score(plus))
                    sm = float(self.score(minus))
                    grad[i, k] = (sp - sm) / (2.0 * step)

        grad = np.where(np.isfinite(grad), grad, 0.0)
        return grad
