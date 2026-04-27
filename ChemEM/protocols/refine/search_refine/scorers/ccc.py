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
from scipy.ndimage import gaussian_filter, map_coordinates

from ChemEM.protocols.core.sci_score import (
    simulate_ligand_density_on_map_grid,
    truncated_cc,
)

from .base import BaseScorer


class CCCScorer(BaseScorer):
    name = "ccc"

    def prepare(self, env, local_map, heavy_masses, protein_heavy_indices) -> None:
        super().prepare(env, local_map, heavy_masses, protein_heavy_indices)
        if local_map is None:
            raise ValueError("CCCScorer requires a local map")

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

        self._sigma_coeff = float(self._opt("sr_sigma_coeff", 0.356))
        self._sigma_A = self._sigma_coeff * self._resolution
        # scipy's gaussian_filter takes sigma per array axis (z, y, x).
        eps_ap = 1e-12
        self._sigma_zyx = np.array(
            [
                self._sigma_A / max(float(self._map_apix[2]), eps_ap),
                self._sigma_A / max(float(self._map_apix[1]), eps_ap),
                self._sigma_A / max(float(self._map_apix[0]), eps_ap),
            ],
            dtype=np.float64,
        )

        mask_mode = str(self._opt("sr_ccc_mask_mode", "nonzero")).lower()
        if mask_mode == "full":
            self._mask = np.ones_like(self._exp_map, dtype=bool)
        else:
            self._mask = None  # truncated_cc will fall back to nonzero-union

    def score(self, heavy_coords_A: np.ndarray, terms_out: Optional[dict] = None) -> float:
        coords = np.asarray(heavy_coords_A, dtype=np.float64)
        sim_map = simulate_ligand_density_on_map_grid(
            coords_xyz_A=coords,
            atom_masses=self.heavy_masses,
            map_origin_xyz_A=self._map_origin,
            map_apix_xyz_A=self._map_apix,
            map_shape_zyx=self._map_shape,
            resolution_A=self._resolution,
            sigma_coeff=self._sigma_coeff,
            normalise=bool(self._opt("sr_normalise_sim_map", True)),
        )

        cc = truncated_cc(self._exp_map, sim_map, self._mask)

        if terms_out is not None:
            terms_out["cc"] = float(cc)

        return float(cc)

    def atom_gradient(self, heavy_coords_A: np.ndarray) -> np.ndarray:
        """Analytical per-atom gradient of CCC (higher-is-better convention).

        Falls back to the base-class finite-difference implementation if the
        analytical path fails for any reason (e.g. degenerate maps). The
        ``--sr-stage=legacy`` preset also routes through the FD path so the
        legacy behavior stays bit-exact for A/B regression.
        """
        if str(self._opt("sr_stage", "v2")).lower() == "legacy":
            return super().atom_gradient(heavy_coords_A)
        try:
            return self._analytical_gradient(heavy_coords_A)
        except Exception:
            return super().atom_gradient(heavy_coords_A)

    def _analytical_gradient(self, heavy_coords_A: np.ndarray) -> np.ndarray:
        coords = np.asarray(heavy_coords_A, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"heavy_coords_A must be (N,3), got {coords.shape}")

        # Derivation:
        #   rho_sim(r) = sum_i w_i * G_sigma(r - x_i)
        #   A = rho_sim - mean(rho_sim), B = rho_exp - mean(rho_exp), both on mask M
        #   CCC = <A, B> / (|A|*|B|)
        #   ∂CCC/∂x_i = (w_i / (|A|*|B|)) * grad(G_sigma * R)(x_i)
        #   where R = B - (<A,B>/|A|^2) * A, zeroed outside M.
        # grad(G_sigma * R) is computed on the grid via derivative-of-Gaussian
        # filters, then trilinearly sampled at each atom position. The extra
        # factor 1/apix converts scipy's voxel-space derivative to Å-space.
        # CCC is scale-invariant, so we simulate un-normalized here for a
        # clean analytical formula.
        sim_map = simulate_ligand_density_on_map_grid(
            coords_xyz_A=coords,
            atom_masses=self.heavy_masses,
            map_origin_xyz_A=self._map_origin,
            map_apix_xyz_A=self._map_apix,
            map_shape_zyx=self._map_shape,
            resolution_A=self._resolution,
            sigma_coeff=self._sigma_coeff,
            normalise=False,
        )

        exp_map = self._exp_map
        if self._mask is not None:
            mask = self._mask & np.isfinite(exp_map) & np.isfinite(sim_map)
        else:
            mask = (
                np.isfinite(exp_map)
                & np.isfinite(sim_map)
                & ((exp_map != 0.0) | (sim_map != 0.0))
            )
            if int(np.count_nonzero(mask)) < 64:
                mask = np.isfinite(exp_map) & np.isfinite(sim_map)

        if int(np.count_nonzero(mask)) < 4:
            return np.zeros_like(coords)

        e_vals = exp_map[mask]
        s_vals = sim_map[mask]
        e_mean = float(np.mean(e_vals))
        s_mean = float(np.mean(s_vals))

        Ac = np.where(mask, sim_map - s_mean, 0.0)
        Bc = np.where(mask, exp_map - e_mean, 0.0)

        norm_A = float(np.linalg.norm(Ac[mask]))
        norm_B = float(np.linalg.norm(Bc[mask]))
        if norm_A < 1e-12 or norm_B < 1e-12:
            return np.zeros_like(coords)

        num = float(np.dot(Ac[mask], Bc[mask]))
        lam = num / (norm_A * norm_A)
        R = np.where(mask, Bc - lam * Ac, 0.0)

        dR_dz = gaussian_filter(R, sigma=self._sigma_zyx, order=(1, 0, 0), mode="nearest")
        dR_dy = gaussian_filter(R, sigma=self._sigma_zyx, order=(0, 1, 0), mode="nearest")
        dR_dx = gaussian_filter(R, sigma=self._sigma_zyx, order=(0, 0, 1), mode="nearest")

        eps_ap = 1e-12
        inv_apix_x = 1.0 / max(float(self._map_apix[0]), eps_ap)
        inv_apix_y = 1.0 / max(float(self._map_apix[1]), eps_ap)
        inv_apix_z = 1.0 / max(float(self._map_apix[2]), eps_ap)

        # Fractional voxel coords per atom: sample axes are (z, y, x).
        fx = (coords[:, 0] - float(self._map_origin[0])) * inv_apix_x
        fy = (coords[:, 1] - float(self._map_origin[1])) * inv_apix_y
        fz = (coords[:, 2] - float(self._map_origin[2])) * inv_apix_z
        sample = np.vstack([fz, fy, fx])

        gx = map_coordinates(dR_dx, sample, order=1, mode="nearest") * inv_apix_x
        gy = map_coordinates(dR_dy, sample, order=1, mode="nearest") * inv_apix_y
        gz = map_coordinates(dR_dz, sample, order=1, mode="nearest") * inv_apix_z

        denom = norm_A * norm_B
        scale = self.heavy_masses / denom
        grad = np.column_stack([gx, gy, gz]) * scale[:, None]
        grad = np.where(np.isfinite(grad), grad, 0.0)
        return grad
