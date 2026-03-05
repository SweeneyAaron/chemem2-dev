#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


# ----------------------------
# Minimal helpers
# ----------------------------

def _ensure_xyz3(v) -> np.ndarray:
    """Accept scalar or (3,) iterable; return float (3,) array."""
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size == 1:
        return np.repeat(a, 3)
    if a.size != 3:
        raise ValueError(f"Expected scalar or length-3, got shape {a.shape}")
    return a


# ----------------------------
# Adapter around ChemEM EMMap
# ----------------------------

@dataclass
class MapGrid:
    """
    Adapter around a regular grid map.

    data: (nz, ny, nx) array (ZYX indexing)
    origin_xyz: coordinate (Å) of voxel index (0,0,0) in XYZ
    apix_xyz: voxel size (Å) along X,Y,Z
    """
    data: np.ndarray
    origin_xyz: np.ndarray
    apix_xyz: np.ndarray

    @staticmethod
    def from_emmap(emmap) -> "MapGrid":
        """
        Build a MapGrid from your ChemEM EMMap.

        Assumptions:
          - emmap.origin is XYZ Å of voxel (0,0,0)
          - emmap.apix is XYZ Å/voxel (scalar ok)
          - emmap.density_map is a numpy array in (Z,Y,X) order

        If your density_map is actually (X,Y,Z), fix with:
            data = np.transpose(emmap.density_map, (2,1,0))  # XYZ -> ZYX
        """
        origin = _ensure_xyz3(emmap.origin)
        apix = _ensure_xyz3(emmap.apix)

        data = np.asarray(emmap.density_map, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"density_map must be 3D, got shape {data.shape}")

        return MapGrid(data=data, origin_xyz=origin, apix_xyz=apix)

    def stats(self) -> Tuple[float, float]:
        """Global mean and std (used for A,B in the reference Gaussian)."""
        return float(np.mean(self.data)), float(np.std(self.data))

    def sample_trilinear(self, xyz: np.ndarray) -> np.ndarray:
        """
        Trilinear interpolation at xyz points (Å).
        Returns float32 array of shape (N,), with NaN for out-of-bounds.
        """
        xyz_in = np.asarray(xyz, dtype=float)
        xyz2 = np.atleast_2d(xyz_in)  # (N,3)
        if xyz2.shape[1] != 3:
            raise ValueError(f"xyz must have shape (...,3), got {xyz2.shape}")

        orig = self.origin_xyz.reshape(3)
        apix = self.apix_xyz.reshape(3)

        # fractional grid coords in (x,y,z)
        f = (xyz2 - orig[None, :]) / apix[None, :]
        x, y, z = f[:, 0], f[:, 1], f[:, 2]

        nx = self.data.shape[2]
        ny = self.data.shape[1]
        nz = self.data.shape[0]

        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        z0 = np.floor(z).astype(int)

        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        oob = (
            (x0 < 0) | (y0 < 0) | (z0 < 0) |
            (x1 >= nx) | (y1 >= ny) | (z1 >= nz)
        )

        xd = (x - x0).astype(float)
        yd = (y - y0).astype(float)
        zd = (z - z0).astype(float)

        # gather (data is z,y,x)
        c000 = self.data[z0, y0, x0]
        c100 = self.data[z0, y0, x1]
        c010 = self.data[z0, y1, x0]
        c110 = self.data[z0, y1, x1]
        c001 = self.data[z1, y0, x0]
        c101 = self.data[z1, y0, x1]
        c011 = self.data[z1, y1, x0]
        c111 = self.data[z1, y1, x1]

        c00 = c000 * (1 - xd) + c100 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        val = (c0 * (1 - zd) + c1 * zd).astype(np.float32)

        if np.any(oob):
            val = val.copy()
            val[oob] = np.nan

        return val


# ----------------------------
# Q-score core (unchanged)
# ----------------------------

def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=float)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * i / phi
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def greedy_farthest_point(dirs: np.ndarray, k: int) -> np.ndarray:
    dirs = np.asarray(dirs, dtype=float)
    if dirs.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    k = min(k, dirs.shape[0])

    chosen = [0]
    min_d = 1.0 - np.clip(dirs @ dirs[0], -1.0, 1.0)

    for _ in range(1, k):
        idx = int(np.argmax(min_d))
        chosen.append(idx)
        d_new = 1.0 - np.clip(dirs @ dirs[idx], -1.0, 1.0)
        min_d = np.minimum(min_d, d_new)

    return dirs[np.array(chosen, dtype=int)]


def reference_gaussian_values(
    radii: np.ndarray,
    *,
    map_mean: float,
    map_std: float,
    sigma_ref: float = 0.6,
    mu: float = 0.0,
) -> np.ndarray:
    A = map_mean + 10.0 * map_std
    B = map_mean - 1.0 * map_std
    x = np.asarray(radii, dtype=float)
    return (A * np.exp(-0.5 * ((x - mu) / sigma_ref) ** 2) + B).astype(np.float32)


def qscore_from_uv(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    mask = np.isfinite(u) & np.isfinite(v)
    u = u[mask]
    v = v[mask]
    if u.size < 2:
        return float("nan")

    u0 = u - np.mean(u)
    v0 = v - np.mean(v)

    nu = np.linalg.norm(u0)
    nv = np.linalg.norm(v0)
    if nu == 0.0 or nv == 0.0:
        return float("nan")

    return float(np.dot(u0, v0) / (nu * nv))


def compute_atom_qscore(
    atom_index: int,
    atom_xyz: np.ndarray,
    *,
    kdtree: cKDTree,
    mapgrid: MapGrid,
    radii: np.ndarray,
    dirs_unit: np.ndarray,
    n_points_per_shell: int = 8,
    sigma_ref: float = 0.6,
    map_mean: float,
    map_std: float,
) -> float:
    atom_xyz = np.asarray(atom_xyz, dtype=float).reshape(3)

    v_r = reference_gaussian_values(
        radii, map_mean=map_mean, map_std=map_std, sigma_ref=sigma_ref
    )

    u_vals: List[float] = []
    v_vals: List[float] = []

    for i, r in enumerate(radii):
        if np.isclose(r, 0.0):
            u0 = float(mapgrid.sample_trilinear(atom_xyz)[0])
            u_vals.extend([u0] * n_points_per_shell)
            v_vals.extend([float(v_r[i])] * n_points_per_shell)
            continue

        cand_pts = atom_xyz[None, :] + r * dirs_unit  # (K,3)
        nn_idx = kdtree.query(cand_pts, k=1)[1]        # (K,)
        allowed = dirs_unit[nn_idx == atom_index]      # (A,3)

        if allowed.shape[0] == 0:
            allowed = dirs_unit

        chosen_dirs = greedy_farthest_point(allowed, n_points_per_shell)

        if chosen_dirs.shape[0] < n_points_per_shell:
            reps = n_points_per_shell - chosen_dirs.shape[0]
            chosen_dirs = np.vstack([chosen_dirs, np.tile(chosen_dirs[:1], (reps, 1))])

        pts = atom_xyz[None, :] + r * chosen_dirs
        u_shell = mapgrid.sample_trilinear(pts)

        u_vals.extend([float(x) for x in u_shell])
        v_vals.extend([float(v_r[i])] * n_points_per_shell)

    return qscore_from_uv(np.array(u_vals), np.array(v_vals))


def compute_qscores_from_emmap(
    *,
    atoms_xyz: np.ndarray,   # (N,3) atom coords in Å
    emmap,                   # ChemEM EMMap instance
    sigma_ref: float = 0.6,
    radii: Optional[np.ndarray] = None,
    n_points_per_shell: int = 8,
    candidate_dirs: int = 256,
) -> np.ndarray:
    """
    Compute per-atom Q-scores given atom coordinates and a ChemEM EMMap.
    """
    if radii is None:
        radii = np.round(np.arange(0.0, 2.0 + 1e-6, 0.1), 3)  # 0..2Å inclusive

    atoms_xyz = np.asarray(atoms_xyz, dtype=float)
    if atoms_xyz.ndim != 2 or atoms_xyz.shape[1] != 3:
        raise ValueError(f"atoms_xyz must be (N,3), got {atoms_xyz.shape}")

    mapgrid = MapGrid.from_emmap(emmap)
    map_mean, map_std = mapgrid.stats()

    kdtree = cKDTree(atoms_xyz)
    dirs_unit = fibonacci_sphere(candidate_dirs)

    out = np.empty((atoms_xyz.shape[0],), dtype=np.float32)
    for i in range(atoms_xyz.shape[0]):
        out[i] = compute_atom_qscore(
            i,
            atoms_xyz[i],
            kdtree=kdtree,
            mapgrid=mapgrid,
            radii=radii,
            dirs_unit=dirs_unit,
            n_points_per_shell=n_points_per_shell,
            sigma_ref=sigma_ref,
            map_mean=map_mean,
            map_std=map_std,
        )
    return out


# ----------------------------
# Optional: residue averaging helper
# ----------------------------

def per_residue_average(
    residue_keys: List[Tuple[str, str, int, str]],  # same length as atoms
    qscores: np.ndarray
) -> Dict[Tuple[str, str, int, str], float]:
    qscores = np.asarray(qscores, dtype=float).ravel()
    acc: Dict[Tuple[str, str, int, str], List[float]] = {}
    for key, q in zip(residue_keys, qscores):
        acc.setdefault(key, []).append(float(q))
    return {k: float(np.nanmean(v)) for k, v in acc.items()}
