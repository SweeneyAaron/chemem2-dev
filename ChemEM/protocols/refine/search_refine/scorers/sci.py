# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ChemEM.protocols.core.sci_score import (
    sci_score_3d,
    simulate_ligand_density_on_map_grid,
)

from .base import BaseScorer


class SCIScorer(BaseScorer):
    name = "sci"

    def prepare(self, env, local_map, heavy_masses, protein_heavy_indices) -> None:
        super().prepare(env, local_map, heavy_masses, protein_heavy_indices)
        if local_map is None:
            raise ValueError("SCIScorer requires a local map")

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

        sci, details = sci_score_3d(
            self._exp_map,
            sim_map,
            use_amp_eq=bool(self._opt("sr_use_amp_eq", True)),
            sigma=float(self._opt("sr_sci_sigma", 1.0)),
            w0=float(self._opt("sr_w0", 1.0)),
            w1=float(self._opt("sr_w1", 1.0)),
            w2=float(self._opt("sr_w2", 1.0)),
            eps=float(self._opt("sr_sci_eps", 1e-8)),
        )

        if terms_out is not None:
            for k in ("cc0", "ccx", "ccy", "ccz", "ccxx", "ccyy", "cczz", "sci"):
                terms_out[k] = float(details.get(k, 0.0))

        return float(sci)
