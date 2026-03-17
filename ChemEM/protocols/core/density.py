#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:56:56 2026

@author: aaron.sweeney
"""

import numpy as np
from rdkit import Chem
from scipy.ndimage import gaussian_filter


def _as_xyz(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.size == 1:
        return np.array([float(v[0]), float(v[0]), float(v[0])], dtype=np.float64)
    if v.size != 3:
        raise ValueError(f"Expected scalar or length-3 value, got shape {v.shape}")
    return v


def _zscore_array(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < eps:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sd


def blur_ligand_to_emmap(
    mol,
    emmap,
    resolution: float,
    conf_id: int = 0,
    sigma_coeff: float = 0.356,
    weight_mode: str = "mass",
    normalise: bool = True,
    out_dtype=np.float32,
):
    """
    Rasterise and Gaussian-blur an RDKit ligand onto the grid of `emmap`.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Ligand with a conformer.
    emmap : EMMap-like
        Must provide:
          - origin : (x, y, z) in Å
          - apix : scalar or (ax, ay, az) in Å/voxel
          - density_map : ndarray with shape (z, y, x)
          - copy() : returns a copy
    resolution : float
        Target map resolution in Å.
    conf_id : int
        RDKit conformer id.
    sigma_coeff : float
        Blur sigma in Å is sigma_coeff * resolution.
    weight_mode : {"mass", "atomic_number", "ones"}
        Per-atom weight used before blurring.
    normalise : bool
        If True, z-score the blurred model map.
    out_dtype : dtype
        Output density dtype.

    Returns
    -------
    model_map : same type as `emmap`
        Copy of `emmap` with density_map replaced by the blurred ligand density.
    """
    if mol is None:
        raise ValueError("mol is None")
    if mol.GetNumConformers() == 0:
        raise ValueError("mol has no conformers")
    if emmap is None:
        raise ValueError("emmap is None")

    conf = mol.GetConformer(int(conf_id))
    origin = np.asarray(emmap.origin, dtype=np.float64)
    apix = _as_xyz(emmap.apix)  # (ax, ay, az)
    nz, ny, nx = np.asarray(emmap.density_map).shape

    grid = np.zeros((nz, ny, nx), dtype=np.float64)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        x, y, z = float(pos.x), float(pos.y), float(pos.z)

        ix = int(round((x - origin[0]) / apix[0]))
        iy = int(round((y - origin[1]) / apix[1]))
        iz = int(round((z - origin[2]) / apix[2]))

        if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
            continue

        if weight_mode == "mass":
            w = float(atom.GetMass())
        elif weight_mode == "atomic_number":
            w = float(atom.GetAtomicNum())
        elif weight_mode == "ones":
            w = 1.0
        else:
            raise ValueError(f"Unknown weight_mode={weight_mode!r}")

        grid[iz, iy, ix] += w

    sigma_A = float(sigma_coeff) * float(resolution)
    sigma_zyx = np.array(
        [sigma_A / apix[2], sigma_A / apix[1], sigma_A / apix[0]],
        dtype=np.float64,
    )

    blurred = gaussian_filter(grid, sigma=sigma_zyx, mode="constant", cval=0.0)

    if normalise:
        blurred = _zscore_array(blurred)

    model_map = emmap.copy()
    model_map.density_map = blurred.astype(out_dtype, copy=False)
    if hasattr(model_map, "resolution"):
        model_map.resolution = float(resolution)
    return model_map


def mutual_information_score(
    map_a,
    map_b,
    n_bins: int = 64,
    mask=None,
    nonzero_union: bool = True,
    zscore: bool = False,
    normalized: bool = False,
    eps: float = 1e-12,
):
    """
    Compute histogram-based mutual information between two maps on the same grid.

    Parameters
    ----------
    map_a, map_b : EMMap-like or ndarray
        If EMMap-like, `.density_map` is used.
        Arrays must have identical shape.
    n_bins : int
        Number of histogram bins.
    mask : ndarray[bool] or None
        Optional boolean mask on the grid.
    nonzero_union : bool
        If True and mask is None, only voxels where (a != 0) or (b != 0) are used.
    zscore : bool
        If True, z-score both voxel arrays before histogramming.
    normalized : bool
        If True, return normalized MI = MI / sqrt(H(A) * H(B)).
    eps : float
        Small epsilon for numerical safety.

    Returns
    -------
    float
        MI score. Higher is better.
    """
    a = np.asarray(map_a.density_map if hasattr(map_a, "density_map") else map_a, dtype=np.float64)
    b = np.asarray(map_b.density_map if hasattr(map_b, "density_map") else map_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(f"Map shape mismatch: {a.shape} vs {b.shape}")

    if mask is None:
        use_mask = np.ones(a.shape, dtype=bool)
        if nonzero_union:
            use_mask &= ((a != 0.0) | (b != 0.0))
    else:
        use_mask = np.asarray(mask, dtype=bool)
        if use_mask.shape != a.shape:
            raise ValueError(f"Mask shape mismatch: {use_mask.shape} vs {a.shape}")

    av = a[use_mask].ravel()
    bv = b[use_mask].ravel()

    if av.size == 0:
        return 0.0

    if zscore:
        av = _zscore_array(av, eps=eps)
        bv = _zscore_array(bv, eps=eps)

    joint_hist, _, _ = np.histogram2d(av, bv, bins=int(n_bins))
    total = float(joint_hist.sum())
    if total <= 0.0:
        return 0.0

    pxy = joint_hist / total
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]

    nz = pxy > 0.0
    mi = float(np.sum(pxy[nz] * np.log((pxy[nz] + eps) / (px_py[nz] + eps))))

    if not normalized:
        return mi

    px_nz = px > 0.0
    py_nz = py > 0.0
    hx = -float(np.sum(px[px_nz] * np.log(px[px_nz] + eps)))
    hy = -float(np.sum(py[py_nz] * np.log(py[py_nz] + eps)))
    denom = np.sqrt(max(hx * hy, eps))
    return float(mi / denom)


def ligand_map_mi_score(
    mol,
    emmap,
    resolution: float,
    conf_id: int = 0,
    sigma_coeff: float = 0.356,
    weight_mode: str = "mass",
    normalise_model: bool = True,
    n_bins: int = 64,
    mask=None,
    nonzero_union: bool = True,
    zscore: bool = False,
    normalized: bool = False,
):
    """
    Convenience wrapper:
      blur ligand onto `emmap` grid, then score MI against `emmap`.
    """
    model_map = blur_ligand_to_emmap(
        mol=mol,
        emmap=emmap,
        resolution=resolution,
        conf_id=conf_id,
        sigma_coeff=sigma_coeff,
        weight_mode=weight_mode,
        normalise=normalise_model,
    )

    score = mutual_information_score(
        emmap,
        model_map,
        n_bins=n_bins,
        mask=mask,
        nonzero_union=nonzero_union,
        zscore=zscore,
        normalized=normalized,
    )
    return score, model_map


def submap_from_structure(structure,
                          density_map,
                          pad_A = 3.0,
                          heavy_only=True,
                          pad_value=0.0):
    """
    Cut a local EM submap around the current selected_structure.

    The returned map keeps the original world-space coordinate frame,
    just cropped to a local box around the structure.
    """
    coords = []
    for atom in structure.atoms:
        if heavy_only and not atom.element <= 1:
            continue
        coords.append([atom.xx, atom.xy, atom.xz])
    
    if not coords:
        raise RuntimeError("[ERROR] No atoms available to define local map box.")

    coords = np.asarray(coords, dtype=float)
    mins = coords.min(axis=0) - float(pad_A)
    maxs = coords.max(axis=0) + float(pad_A)
    apix = np.asarray(density_map.apix, dtype=float) 
    extents = np.maximum(maxs - mins, apix)
    
    nxyz = np.ceil(extents / apix).astype(int) + 1
    nx, ny, nz = int(nxyz[0]), int(nxyz[1]), int(nxyz[2])
    
    submap = density_map.submap(
        origin=(float(mins[0]), float(mins[1]), float(mins[2])),
        box_size=(nz, ny, nx),   # EMMap.submap expects (z, y, x)
        pad_value=float(pad_value),
    )
    
    return submap
        
    



