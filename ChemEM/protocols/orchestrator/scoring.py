# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Stateless per-pose scoring helpers used by the orchestrator's gates.

Both helpers wrap existing ChemEM primitives without mutating system or
ligand state, so the orchestrator can score arbitrary candidate poses
without disturbing the protocols that ship those primitives.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ChemEM.protocols.mapQ_score.mapq_utils import compute_qscores_from_emmap
from ChemEM.protocols._docking.mmgbsa_score import score_single_pose
from ChemEM.protocols.core.density import mutual_information_score
from ChemEM.protocols.core.sci_score import (
    sci_score_3d,
    simulate_ligand_density_subgrid,
    truncated_cc,
)
from ChemEM.protocols.smart_refine_2.shape_metrics import density_shape_metrics


def _heavy_atom_indices(mol) -> np.ndarray:
    return np.asarray(
        [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"], dtype=int
    )


def _low_tail(values: np.ndarray, fraction: float = 0.3) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.sort(arr[np.isfinite(arr)])
    if finite.size == 0:
        return 0.0
    n_tail = max(1, int(np.ceil(float(fraction) * finite.size)))
    return float(np.mean(finite[:n_tail]))


def _atom_masses(mol, indices: np.ndarray) -> np.ndarray:
    masses = []
    for idx in np.asarray(indices, dtype=int).reshape(-1):
        atom = mol.GetAtomWithIdx(int(idx))
        mass = float(atom.GetMass())
        if not np.isfinite(mass) or mass <= 0.0:
            mass = float(max(atom.GetAtomicNum(), 1))
        masses.append(mass)
    return np.asarray(masses, dtype=float)


def _within_map_bounds(positions: np.ndarray, density_map) -> bool:
    try:
        origin = np.asarray(density_map.origin, dtype=float)
        apix = np.asarray(density_map.apix, dtype=float)
        shape = np.asarray(density_map.density_map.shape, dtype=float)
    except AttributeError:
        return True  # fail open if map shape isn't introspectable
    max_bounds = origin + (shape - 1.0) * apix
    return bool(np.all(positions >= origin) and np.all(positions <= max_bounds))


def qscore_pose(
    coords: np.ndarray,
    mol,
    density_map,
    sigma_ref: float = 0.6,
) -> Optional[float]:
    """Mean Q-score of one ligand pose against the full map.

    coords: (n_atoms, 3) Å — must align with mol's atom indexing.
    Returns None if the pose lies outside the map or has no heavy atoms.
    """
    metrics = qscore_pose_metrics(
        coords,
        mol,
        density_map,
        sigma_ref=sigma_ref,
    )
    if metrics is None:
        return None
    return float(metrics["qscore"])


def qscore_pose_metrics(
    coords: np.ndarray,
    mol,
    density_map,
    sigma_ref: float = 0.6,
    low_tail_fraction: float = 0.3,
) -> Optional[dict]:
    """Per-pose Q-score metrics for heavy atoms."""
    heavy = _heavy_atom_indices(mol)
    if heavy.size == 0:
        return None
    heavy_xyz = np.asarray(coords, dtype=float)[heavy]
    if not _within_map_bounds(heavy_xyz, density_map):
        return None
    qs = compute_qscores_from_emmap(
        atoms_xyz=heavy_xyz, emmap=density_map, sigma_ref=sigma_ref
    )
    qs = np.asarray(qs, dtype=float).reshape(-1)
    finite = qs[np.isfinite(qs)]
    if finite.size == 0:
        return None
    q_mean = float(np.mean(finite))
    q_low_tail = _low_tail(qs, low_tail_fraction)
    return {
        "qscore": q_mean,
        "q_mean": q_mean,
        "q_low_tail": q_low_tail,
        "q_per_atom": [float(v) for v in qs],
        "q_heavy_atom_indices": [int(i) for i in heavy.tolist()],
    }


def _iter_site_maps(site_maps):
    if site_maps is None:
        return []
    if hasattr(site_maps, "density_map"):
        return [(0, site_maps)]
    if isinstance(site_maps, dict):
        values = list(site_maps.values())
    elif isinstance(site_maps, (list, tuple)):
        values = list(site_maps)
    else:
        values = [site_maps]

    out = []
    for idx, item in enumerate(values):
        emmap = item
        if isinstance(item, (list, tuple)) and item:
            emmap = item[0]
        if hasattr(emmap, "density_map"):
            out.append((idx, emmap))
    return out


def _as_xyz(value, default):
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        arr = np.asarray(default, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.array([float(arr[0])] * 3, dtype=float)
    if arr.size != 3:
        return np.asarray(default, dtype=float).reshape(3)
    return arr.astype(float)


def _resolution_A(emmap, default: float = 3.0) -> float:
    try:
        val = float(getattr(emmap, "resolution", default))
    except Exception:
        val = default
    if not np.isfinite(val) or val <= 0.0:
        val = default
    return val


def _density_mask(density: np.ndarray, threshold_frac: float) -> tuple[np.ndarray, float]:
    positive = np.clip(np.asarray(density, dtype=float), 0.0, None)
    vmax = float(np.max(positive)) if positive.size else 0.0
    if vmax <= 0.0:
        return positive > 0.0, 0.0
    threshold = vmax * max(float(threshold_frac), 0.0)
    mask = positive > threshold
    if int(np.count_nonzero(mask)) == 0:
        mask = positive > 0.0
    return mask, threshold


def density_fit_metrics(
    coords: np.ndarray,
    mol,
    site_maps,
    *,
    threshold_frac: float = 0.05,
    sigma_coeff: float = 0.356,
    normalise: bool = True,
    feature_weights: Optional[dict] = None,
    compute_sci: bool = False,
    compute_shape: bool = False,
) -> Optional[dict]:
    """Coverage-aware ligand-vs-site-density metrics.

    The coverage denominator is the full site feature map, not only the
    ligand support subgrid. This penalizes compact ligands that explain only
    a small part of a larger density feature.
    """
    maps = _iter_site_maps(site_maps)
    if not maps:
        return None

    heavy = _heavy_atom_indices(mol)
    if heavy.size == 0:
        return None

    coords = np.asarray(coords, dtype=float)
    heavy_xyz = coords[heavy]
    masses = _atom_masses(mol, heavy)
    weights = {
        "density_coverage": 2.0,
        "density_precision": 0.5,
        "density_ccc": 0.5,
    }
    if feature_weights:
        weights.update(feature_weights)

    best = None
    feature_metrics = []
    for feature_idx, emmap in maps:
        density = np.asarray(getattr(emmap, "density_map"), dtype=float)
        if density.ndim != 3 or density.size == 0:
            continue

        origin = _as_xyz(getattr(emmap, "origin", (0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))
        apix = _as_xyz(getattr(emmap, "apix", (1.0, 1.0, 1.0)), (1.0, 1.0, 1.0))
        sim_map, lo_zyx, hi_zyx = simulate_ligand_density_subgrid(
            coords_xyz_A=heavy_xyz,
            atom_masses=masses,
            map_origin_xyz_A=origin,
            map_apix_xyz_A=apix,
            map_shape_zyx=tuple(int(i) for i in density.shape),
            resolution_A=_resolution_A(emmap),
            sigma_coeff=float(sigma_coeff),
            normalise=bool(normalise),
        )
        if sim_map.size == 0:
            continue

        z0, y0, x0 = [int(i) for i in lo_zyx]
        z1, y1, x1 = [int(i) for i in hi_zyx]
        exp_map = np.asarray(density[z0:z1, y0:y1, x0:x1], dtype=float)
        if exp_map.shape != sim_map.shape:
            continue

        full_positive = np.clip(density, 0.0, None)
        full_mask, threshold = _density_mask(full_positive, threshold_frac)
        total_site_weight = float(np.sum(full_positive[full_mask]))
        if total_site_weight <= 0.0:
            total_site_weight = float(np.sum(full_positive))
        if total_site_weight <= 0.0:
            continue

        exp_positive = np.clip(exp_map, 0.0, None)
        sim_positive = np.clip(np.asarray(sim_map, dtype=float), 0.0, None)
        sim_max = float(np.max(sim_positive)) if sim_positive.size else 0.0
        sim_norm = sim_positive / sim_max if sim_max > 0.0 else sim_positive

        coverage = float(np.sum(exp_positive * np.clip(sim_norm, 0.0, 1.0)) / total_site_weight)
        coverage = float(np.clip(coverage, 0.0, 1.0))

        sim_sum = float(np.sum(sim_positive))
        if sim_sum > 0.0:
            exp_mask = exp_positive > threshold
            if int(np.count_nonzero(exp_mask)) == 0:
                exp_mask = exp_positive > 0.0
            precision = float(np.sum(sim_positive[exp_mask]) / sim_sum)
        else:
            exp_mask = exp_positive > threshold
            precision = 0.0
        precision = float(np.clip(precision, 0.0, 1.0))

        sim_mask = sim_norm > max(float(threshold_frac), 0.0)
        if int(np.count_nonzero(sim_mask)) == 0:
            sim_mask = sim_positive > 0.0
        union = np.logical_or(exp_mask, sim_mask)
        intersection = np.logical_and(exp_mask, sim_mask)
        if int(np.count_nonzero(union)) > 0:
            envelope_iou = float(np.count_nonzero(intersection) / np.count_nonzero(union))
        else:
            envelope_iou = 0.0
        excess_fraction = float(np.clip(1.0 - precision, 0.0, 1.0))

        ccc = float(truncated_cc(exp_map, sim_map))
        mi = None
        normalized_mi = None
        mi_nbins = 64
        try:
            mi = float(
                mutual_information_score(
                    exp_map,
                    sim_map,
                    n_bins=mi_nbins,
                    nonzero_union=True,
                    normalized=False,
                )
            )
            normalized_mi = float(
                mutual_information_score(
                    exp_map,
                    sim_map,
                    n_bins=mi_nbins,
                    nonzero_union=True,
                    normalized=True,
                )
            )
        except Exception as exc:
            mi_failed = type(exc).__name__
        else:
            mi_failed = None
        sci = None
        sci_terms = None
        if bool(compute_sci):
            try:
                sci, sci_terms = sci_score_3d(exp_map, sim_map)
            except Exception as exc:
                sci = None
                sci_terms = {"density_sci_failed": type(exc).__name__}
        overlap = float(np.sqrt(max(coverage, 0.0) * max(precision, 0.0)))
        feature_score = (
            weights["density_coverage"] * coverage
            + weights["density_precision"] * precision
            + weights["density_ccc"] * ccc
        )
        metrics = {
            "selected_feature_idx": int(feature_idx),
            "density_coverage": coverage,
            "density_precision": precision,
            "density_ccc": ccc,
            "density_overlap": overlap,
            "density_envelope_iou": envelope_iou,
            "density_excess_fraction": excess_fraction,
            "density_feature_score": float(feature_score),
            "density_mi_nbins": int(mi_nbins),
            "density_threshold": float(threshold),
            "density_total_weight": float(total_site_weight),
            "density_subgrid_shape": tuple(int(i) for i in sim_map.shape),
            "ligand_heavy_atom_count": int(heavy.size),
        }
        if mi is not None:
            metrics["density_mi"] = float(mi)
        if normalized_mi is not None:
            metrics["density_normalized_mi"] = float(normalized_mi)
        if mi_failed is not None:
            metrics["density_mi_failed"] = mi_failed
        if sci is not None:
            metrics["density_sci"] = float(sci)
            if sci_terms:
                flat_terms = {
                    key: float(value)
                    for key, value in sci_terms.items()
                    if key != "sci"
                    and isinstance(value, (int, float, np.integer, np.floating))
                    and np.isfinite(float(value))
                }
                metrics["density_sci_terms"] = flat_terms
                for key, value in flat_terms.items():
                    metrics[f"density_sci_{key}"] = float(value)
        elif sci_terms and "density_sci_failed" in sci_terms:
            metrics["density_sci_failed"] = sci_terms["density_sci_failed"]
        if bool(compute_shape):
            try:
                shape = density_shape_metrics(
                    exp_map,
                    sim_map,
                    threshold_frac=float(threshold_frac),
                    voxel_size_xyz=tuple(float(v) for v in apix.tolist()),
                )
            except Exception as exc:
                shape = {"shape_metrics_failed": type(exc).__name__}
            metrics.update(shape)
        feature_metrics.append(dict(metrics))
        if best is None or feature_score > best["density_feature_score"]:
            best = metrics

    if best is None:
        return None
    best["density_feature_metrics"] = feature_metrics
    return best


def mmgbsa_single_frame(
    coords: np.ndarray,
    ligand,
    protein,
    pose_idx: int = 0,
    resource_owner=None,
):
    """Single-frame MMGBSA on one pose. Returns PoseScore or None on failure.

    Wraps mmgbsa_score.score_single_pose(), which already builds a
    one-frame mdtraj trajectory and evaluates OpenMM Context energies
    without integrating — no MD sampling.
    """
    try:
        return score_single_pose(
            np.asarray(coords, dtype=float),
            ligand,
            protein,
            pose_idx,
            resource_owner=resource_owner,
        )
    except Exception:
        return None
