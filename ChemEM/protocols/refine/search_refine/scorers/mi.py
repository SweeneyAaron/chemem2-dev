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

from ChemEM.protocols.core.density import mutual_information_score
from ChemEM.protocols.core.sci_score import simulate_ligand_density_on_map_grid

from .base import BaseScorer


class MIScorer(BaseScorer):
    """
    Histogram mutual information between experimental submap and a simulated
    ligand-density map. Higher = better.

    Note: histogram MI is discontinuous. Use a larger FD step (``sr_mi_fd_step_a``)
    to get usable gradients.
    """

    name = "mi"

    def prepare(self, env, local_map, heavy_masses, protein_heavy_indices) -> None:
        super().prepare(env, local_map, heavy_masses, protein_heavy_indices)
        if local_map is None:
            raise ValueError("MIScorer requires a local map")

        self._map_origin = np.asarray(local_map.origin, dtype=np.float64)
        self._map_apix = np.asarray(local_map.apix, dtype=np.float64)
        self._map_shape = tuple(local_map.density_map.shape)
        self._exp_map = np.asarray(local_map.density_map, dtype=np.float64)

        resolution = getattr(local_map, "resolution", None)
        try:
            resolution = float(resolution)
        except (TypeError, ValueError):
            resolution = None
        if resolution is None or not np.isfinite(resolution) or resolution <= 0.0:
            resolution = float(self._opt("resolution", 3.0))
        self._resolution = float(resolution)

    def _fd_step_a(self) -> float:
        return float(self._opt("sr_mi_fd_step_a", 0.1))

    def score(self, heavy_coords_A: np.ndarray, terms_out: Optional[dict] = None) -> float:
        coords = np.asarray(heavy_coords_A, dtype=np.float64)
        sim_map = simulate_ligand_density_on_map_grid(
            coords_xyz_A=coords,
            atom_masses=self.heavy_masses,
            map_origin_xyz_A=self._map_origin,
            map_apix_xyz_A=self._map_apix,
            map_shape_zyx=self._map_shape,
            resolution_A=self._resolution,
            sigma_coeff=float(self._opt("sr_sigma_coeff", 0.356)),
            normalise=bool(self._opt("sr_normalise_sim_map", True)),
        )

        n_bins = int(self._opt("sr_mi_nbins", 64))
        normalized = bool(self._opt("sr_mi_normalized", False))

        mi = mutual_information_score(
            self._exp_map,
            sim_map,
            n_bins=n_bins,
            nonzero_union=True,
            normalized=False,
        )
        if normalized:
            nmi = mutual_information_score(
                self._exp_map,
                sim_map,
                n_bins=n_bins,
                nonzero_union=True,
                normalized=True,
            )
        else:
            nmi = 0.0

        if terms_out is not None:
            terms_out["mi"] = float(mi)
            terms_out["normalized_mi"] = float(nmi)

        return float(nmi if normalized else mi)
