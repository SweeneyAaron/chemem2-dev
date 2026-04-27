# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _safe_mask(mask: np.ndarray | None, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if mask is None:
        m = np.isfinite(a) & np.isfinite(b) & ((a != 0.0) | (b != 0.0))
        # Fallback for sparse maps.
        if int(np.count_nonzero(m)) < 64:
            m = np.isfinite(a) & np.isfinite(b)
        return m

    m = np.asarray(mask, dtype=bool)
    if m.shape != a.shape:
        raise ValueError(f"mask shape {m.shape} does not match map shape {a.shape}")
    return m & np.isfinite(a) & np.isfinite(b)


def truncated_cc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Map shape mismatch: {a.shape} vs {b.shape}")

    m = _safe_mask(mask, a, b)
    av = a[m].ravel()
    bv = b[m].ravel()
    if av.size < 4:
        return 0.0

    av = av - float(np.mean(av))
    bv = bv - float(np.mean(bv))

    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom < eps:
        return 0.0

    cc = float(np.dot(av, bv) / denom)
    if not np.isfinite(cc):
        return 0.0
    return max(0.0, cc)


def amplitude_equalize_pair(map_a: np.ndarray, map_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fourier amplitude equalization:
    keep each map phase, replace both amplitudes with their average amplitude.
    """
    a = np.asarray(map_a, dtype=np.float64)
    b = np.asarray(map_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Map shape mismatch: {a.shape} vs {b.shape}")

    fa = np.fft.fftn(a)
    fb = np.fft.fftn(b)

    amp = 0.5 * (np.abs(fa) + np.abs(fb))

    a_eq = np.fft.ifftn(amp * np.exp(1j * np.angle(fa))).real
    b_eq = np.fft.ifftn(amp * np.exp(1j * np.angle(fb))).real

    return np.asarray(a_eq, dtype=np.float64), np.asarray(b_eq, dtype=np.float64)


def derivative_channels_3d(volume: np.ndarray, sigma: float = 1.0) -> dict[str, np.ndarray]:
    """
    Directional first/second derivatives for (z,y,x)-ordered volume.
    Returns base, dx, dy, dz, dxx, dyy, dzz.
    """
    v = np.asarray(volume, dtype=np.float64)
    s = max(float(sigma), 0.0)

    base = gaussian_filter(v, sigma=s, mode="nearest") if s > 0.0 else v

    # Axis order is z, y, x for ndarray maps used in ChemEM.
    dx = gaussian_filter(v, sigma=s, order=(0, 0, 1), mode="nearest")
    dy = gaussian_filter(v, sigma=s, order=(0, 1, 0), mode="nearest")
    dz = gaussian_filter(v, sigma=s, order=(1, 0, 0), mode="nearest")

    dxx = gaussian_filter(v, sigma=s, order=(0, 0, 2), mode="nearest")
    dyy = gaussian_filter(v, sigma=s, order=(0, 2, 0), mode="nearest")
    dzz = gaussian_filter(v, sigma=s, order=(2, 0, 0), mode="nearest")

    return {
        "base": np.asarray(base, dtype=np.float64),
        "dx": np.asarray(dx, dtype=np.float64),
        "dy": np.asarray(dy, dtype=np.float64),
        "dz": np.asarray(dz, dtype=np.float64),
        "dxx": np.asarray(dxx, dtype=np.float64),
        "dyy": np.asarray(dyy, dtype=np.float64),
        "dzz": np.asarray(dzz, dtype=np.float64),
    }


def sci_score_3d(
    exp_map: np.ndarray,
    sim_map: np.ndarray,
    *,
    use_amp_eq: bool = True,
    sigma: float = 1.0,
    w0: float = 1.0,
    w1: float = 1.0,
    w2: float = 1.0,
    eps: float = 1e-8,
    mask: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """
    3D SCI-like score with truncated channel correlations and log-domain fusion.

    SCI = exp(
        w0*log(eps+CC0)
        + w1*mean(log(eps+CCx), log(eps+CCy), log(eps+CCz))
        + w2*mean(log(eps+CCxx), log(eps+CCyy), log(eps+CCzz))
    )
    """
    e = np.asarray(exp_map, dtype=np.float64)
    r = np.asarray(sim_map, dtype=np.float64)
    if e.shape != r.shape:
        raise ValueError(f"Map shape mismatch: {e.shape} vs {r.shape}")

    if use_amp_eq:
        e, r = amplitude_equalize_pair(e, r)

    m = _safe_mask(mask, e, r)

    e_ch = derivative_channels_3d(e, sigma=sigma)
    r_ch = derivative_channels_3d(r, sigma=sigma)

    cc0 = truncated_cc(e_ch["base"], r_ch["base"], m)

    ccx = truncated_cc(e_ch["dx"], r_ch["dx"], m)
    ccy = truncated_cc(e_ch["dy"], r_ch["dy"], m)
    ccz = truncated_cc(e_ch["dz"], r_ch["dz"], m)

    ccxx = truncated_cc(e_ch["dxx"], r_ch["dxx"], m)
    ccyy = truncated_cc(e_ch["dyy"], r_ch["dyy"], m)
    cczz = truncated_cc(e_ch["dzz"], r_ch["dzz"], m)

    first_log_mean = float(np.mean([
        np.log(eps + ccx),
        np.log(eps + ccy),
        np.log(eps + ccz),
    ]))
    second_log_mean = float(np.mean([
        np.log(eps + ccxx),
        np.log(eps + ccyy),
        np.log(eps + cczz),
    ]))

    sci_log = (float(w0) * np.log(eps + cc0)) + (float(w1) * first_log_mean) + (float(w2) * second_log_mean)
    sci = float(np.exp(sci_log))
    if not np.isfinite(sci):
        sci = 0.0

    details = {
        "cc0": float(cc0),
        "ccx": float(ccx),
        "ccy": float(ccy),
        "ccz": float(ccz),
        "ccxx": float(ccxx),
        "ccyy": float(ccyy),
        "cczz": float(cczz),
        "first_log_mean": float(first_log_mean),
        "second_log_mean": float(second_log_mean),
        "sci_log": float(sci_log),
        "sci": float(sci),
    }
    return sci, details


def simulate_ligand_density_on_map_grid(
    coords_xyz_A: np.ndarray,
    atom_masses: np.ndarray,
    map_origin_xyz_A: np.ndarray,
    map_apix_xyz_A: np.ndarray,
    map_shape_zyx: tuple[int, int, int],
    *,
    resolution_A: float,
    sigma_coeff: float = 0.356,
    normalise: bool = True,
) -> np.ndarray:
    """
    Rasterize ligand atom masses onto map grid and blur by Gaussian sigma_coeff*resolution.
    Returns simulated map in (z,y,x).
    """
    coords = np.asarray(coords_xyz_A, dtype=np.float64)
    masses = np.asarray(atom_masses, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords_xyz_A must be shape (N,3), got {coords.shape}")
    if masses.shape[0] != coords.shape[0]:
        raise ValueError("atom_masses length does not match coordinates")

    origin = np.asarray(map_origin_xyz_A, dtype=np.float64).reshape(3)
    apix = np.asarray(map_apix_xyz_A, dtype=np.float64).reshape(3)
    nz, ny, nx = [int(i) for i in map_shape_zyx]

    grid = np.zeros((nz, ny, nx), dtype=np.float64)

    for xyz, mass in zip(coords, masses):
        fx = (float(xyz[0]) - origin[0]) / apix[0]
        fy = (float(xyz[1]) - origin[1]) / apix[1]
        fz = (float(xyz[2]) - origin[2]) / apix[2]

        ix = int(np.rint(fx))
        iy = int(np.rint(fy))
        iz = int(np.rint(fz))

        if ix < 0 or iy < 0 or iz < 0:
            continue
        if ix >= nx or iy >= ny or iz >= nz:
            continue

        grid[iz, iy, ix] += float(mass)

    sigma_A = float(sigma_coeff) * float(resolution_A)
    sigma_zyx = np.array([
        sigma_A / max(apix[2], 1e-12),
        sigma_A / max(apix[1], 1e-12),
        sigma_A / max(apix[0], 1e-12),
    ], dtype=np.float64)

    sim = gaussian_filter(grid, sigma=sigma_zyx, mode="constant", cval=0.0)

    if normalise:
        vmax = float(np.max(sim))
        if vmax > 0.0:
            sim = sim / vmax

    return np.asarray(sim, dtype=np.float64)
