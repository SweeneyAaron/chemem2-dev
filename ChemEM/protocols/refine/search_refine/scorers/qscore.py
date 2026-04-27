# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

from typing import Optional

import numpy as np
from openmm import unit

from ChemEM.protocols.mapQ_score.mapq_utils import compute_qscores_from_emmap

from .base import BaseScorer


class QScorer(BaseScorer):
    """
    Mean ligand Q-score against the experimental submap. Higher = better.
    """

    name = "qscore"

    def prepare(self, env, local_map, heavy_masses, protein_heavy_indices) -> None:
        super().prepare(env, local_map, heavy_masses, protein_heavy_indices)
        if local_map is None:
            raise ValueError("QScorer requires a local map")
        self._emmap = local_map

        prot_idx = list(protein_heavy_indices) if protein_heavy_indices else []
        if prot_idx:
            state = env.simulation.context.getState(getPositions=True)
            pos_nm = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=np.float64,
            )
            self._protein_xyz_A = np.asarray(pos_nm[prot_idx], dtype=np.float64) * 10.0
        else:
            self._protein_xyz_A = np.zeros((0, 3), dtype=np.float64)

        self._n_ligand = int(self.heavy_masses.shape[0])

        radii_csv = self._opt("sr_qscore_radii", None)
        if radii_csv:
            try:
                radii = np.asarray(
                    [float(x) for x in str(radii_csv).split(",") if x.strip()],
                    dtype=np.float64,
                )
                radii = radii if radii.size else None
            except Exception:
                radii = None
        else:
            radii = None
        self._radii = radii
        self._sigma_ref = float(self._opt("sr_qscore_sigma_ref", 0.6))

    def score(self, heavy_coords_A: np.ndarray, terms_out: Optional[dict] = None) -> float:
        coords = np.asarray(heavy_coords_A, dtype=np.float64)
        if coords.shape[0] != self._n_ligand:
            raise ValueError(
                f"QScorer: expected {self._n_ligand} ligand heavy atoms, got {coords.shape[0]}"
            )

        if self._protein_xyz_A.size:
            combined = np.concatenate([coords, self._protein_xyz_A], axis=0)
        else:
            combined = coords

        q_all = compute_qscores_from_emmap(
            atoms_xyz=combined,
            emmap=self._emmap,
            sigma_ref=self._sigma_ref,
            radii=self._radii,
        )

        q_lig = np.asarray(q_all[: self._n_ligand], dtype=np.float64)
        finite = np.isfinite(q_lig)
        q_mean = float(np.mean(q_lig[finite])) if finite.any() else 0.0

        if terms_out is not None:
            terms_out["q_mean"] = q_mean
            terms_out["q_per_atom"] = [float(v) for v in q_lig]

        return q_mean
