"""Diagnostic density-shape descriptors for SmartRefine2/orchestrator audits.

The descriptors here are deliberately dependency-light. They compare a
thresholded experimental density component with the simulated ligand density on
the same grid, and report rotationally robust descriptor similarities for
benchmarking rather than assignment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

try:
    from skimage.morphology import skeletonize as _skeletonize
except Exception:  # pragma: no cover - optional runtime dependency
    _skeletonize = None

try:  # SciPy >= 1.15
    from scipy.special import sph_harm_y as _sph_harm_y
except Exception:  # pragma: no cover - older SciPy fallback
    _sph_harm_y = None
    from scipy.special import sph_harm as _sph_harm
else:  # pragma: no cover - keep name defined for type checkers
    _sph_harm = None


@dataclass(frozen=True)
class ComponentSelection:
    mask: np.ndarray
    component_count: int
    selected_fraction: float
    mode: str


def _positive(volume: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(volume, dtype=np.float64), 0.0, None)


def _density_mask(volume: np.ndarray, threshold_frac: float) -> tuple[np.ndarray, float]:
    positive = _positive(volume)
    vmax = float(np.max(positive)) if positive.size else 0.0
    if vmax <= 0.0:
        return positive > 0.0, 0.0
    threshold = vmax * max(float(threshold_frac), 0.0)
    mask = positive > threshold
    if int(np.count_nonzero(mask)) == 0:
        mask = positive > 0.0
    return mask, threshold


def _as_zyx_voxel_size(voxel_size_xyz: Optional[tuple[float, float, float]]) -> np.ndarray:
    if voxel_size_xyz is None:
        arr = np.ones(3, dtype=np.float64)
    else:
        try:
            arr = np.asarray(voxel_size_xyz, dtype=np.float64).reshape(-1)
        except Exception:
            arr = np.ones(3, dtype=np.float64)
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    if arr.size != 3:
        arr = np.ones(3, dtype=np.float64)
    arr = np.where(np.isfinite(arr) & (arr > 0.0), arr, 1.0)
    return arr[[2, 1, 0]].astype(np.float64)


def _centroid(
    mask: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    if weights is None:
        return np.mean(coords.astype(np.float64), axis=0)
    vals = np.asarray(weights, dtype=np.float64)[mask]
    total = float(np.sum(vals))
    if total <= 0.0 or not np.isfinite(total):
        return np.mean(coords.astype(np.float64), axis=0)
    return np.average(coords.astype(np.float64), axis=0, weights=vals)


def _select_component(
    exp_positive: np.ndarray,
    exp_mask: np.ndarray,
    sim_mask: np.ndarray,
) -> ComponentSelection:
    labels, n_components = ndimage.label(np.asarray(exp_mask, dtype=bool))
    n_components = int(n_components)
    if n_components == 0:
        return ComponentSelection(
            mask=np.zeros_like(exp_mask, dtype=bool),
            component_count=0,
            selected_fraction=0.0,
            mode="empty",
        )

    total_voxels = max(1, int(np.count_nonzero(exp_mask)))
    best_label = None
    best_overlap = 0
    for label_idx in range(1, n_components + 1):
        component = labels == label_idx
        overlap = int(np.count_nonzero(component & sim_mask))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label_idx

    if best_label is not None and best_overlap > 0:
        component = labels == int(best_label)
        return ComponentSelection(
            mask=component,
            component_count=n_components,
            selected_fraction=float(np.count_nonzero(component) / total_voxels),
            mode="overlap",
        )

    sim_center = _centroid(sim_mask)
    if sim_center is None:
        sim_center = np.asarray(exp_mask.shape, dtype=np.float64) / 2.0

    best_distance = float("inf")
    best_label = 1
    for label_idx in range(1, n_components + 1):
        component = labels == label_idx
        center = _centroid(component, exp_positive)
        if center is None:
            continue
        distance = float(np.linalg.norm(center - sim_center))
        if distance < best_distance:
            best_distance = distance
            best_label = label_idx

    component = labels == int(best_label)
    return ComponentSelection(
        mask=component,
        component_count=n_components,
        selected_fraction=float(np.count_nonzero(component) / total_voxels),
        mode="nearest",
    )


def _coords_weights(
    mask: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    coords_zyx = np.argwhere(mask)
    if coords_zyx.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    if weights is None:
        vals = np.ones(coords_zyx.shape[0], dtype=np.float64)
    else:
        vals = np.asarray(weights, dtype=np.float64)[mask].reshape(-1)
        vals = np.clip(vals, 0.0, None)
        if float(np.sum(vals)) <= 0.0:
            vals = np.ones(coords_zyx.shape[0], dtype=np.float64)
    return coords_zyx.astype(np.float64), vals.astype(np.float64)


def _normalised_cartesian(
    mask: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords_zyx, vals = _coords_weights(mask, weights)
    if coords_zyx.shape[0] == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return np.zeros((0, 3), dtype=np.float64), empty, empty

    total = float(np.sum(vals))
    if total <= 0.0 or not np.isfinite(total):
        vals = np.ones_like(vals, dtype=np.float64)
        total = float(np.sum(vals))
    norm_weights = vals / max(total, 1e-12)

    center = np.average(coords_zyx, axis=0, weights=norm_weights)
    centered_zyx = coords_zyx - center
    # Use x,y,z for spherical coordinates, even though arrays are z,y,x.
    cart = centered_zyx[:, [2, 1, 0]]
    radii = np.linalg.norm(cart, axis=1)
    scale = float(np.max(radii)) if radii.size else 0.0
    if scale <= 1e-12:
        scale = 1.0
    return cart / scale, norm_weights, radii / scale


def _angles(cart: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radii = np.linalg.norm(cart, axis=1)
    safe_r = np.maximum(radii, 1e-12)
    polar = np.arccos(np.clip(cart[:, 2] / safe_r, -1.0, 1.0))
    azimuth = np.mod(np.arctan2(cart[:, 1], cart[:, 0]), 2.0 * np.pi)
    return radii, polar, azimuth


def _spherical_harmonic(
    l: int,
    m: int,
    polar: np.ndarray,
    azimuth: np.ndarray,
) -> np.ndarray:
    if _sph_harm_y is not None:
        return _sph_harm_y(int(l), int(m), polar, azimuth)
    return _sph_harm(int(m), int(l), azimuth, polar)


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.zeros_like(arr)
    return arr / norm


def _descriptor_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    an = _l2_normalise(a)
    bn = _l2_normalise(b)
    if an.size != bn.size or an.size == 0:
        return 0.0, 0.0
    if float(np.linalg.norm(an)) <= 1e-12 or float(np.linalg.norm(bn)) <= 1e-12:
        return 0.0, float(np.linalg.norm(an - bn))
    sim = float(np.clip(np.dot(an, bn), 0.0, 1.0))
    return sim, float(np.linalg.norm(an - bn))


def zernike_style_descriptor(
    mask: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    order: int = 6,
) -> np.ndarray:
    """Return internal 3D Zernike-style rotational invariants.

    This is a radial-moment/spherical-harmonic invariant descriptor. It is not a
    full Novotni/Klein 3DZD implementation, but it has the invariances we need
    for diagnostic density-shape comparisons without adding a hard dependency.
    """
    cart, vals, _ = _normalised_cartesian(mask, weights)
    if cart.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    radii, polar, azimuth = _angles(cart)
    out = []
    max_order = max(0, int(order))
    for n in range(max_order + 1):
        radial = np.power(radii, n)
        for l in range(n + 1):
            if (n - l) % 2 != 0:
                continue
            if n == 0 and l == 0:
                continue
            coeff_norm = 0.0
            for m in range(-l, l + 1):
                basis = _spherical_harmonic(l, m, polar, azimuth)
                coeff = np.sum(vals * radial * np.conjugate(basis))
                coeff_norm += float(np.abs(coeff) ** 2)
            out.append(float(np.sqrt(max(coeff_norm, 0.0))))
    return np.asarray(out, dtype=np.float64)


def spherical_harmonic_shell_descriptor(
    mask: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    lmax: int = 6,
    n_shells: int = 5,
) -> np.ndarray:
    cart, vals, radii = _normalised_cartesian(mask, weights)
    if cart.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    _r, polar, azimuth = _angles(cart)
    max_l = max(0, int(lmax))
    shell_count = max(1, int(n_shells))
    edges = np.linspace(0.0, 1.0 + 1e-12, shell_count + 1)
    out = []
    for shell_idx in range(shell_count):
        shell = (radii >= edges[shell_idx]) & (radii < edges[shell_idx + 1])
        shell_weight = float(np.sum(vals[shell]))
        if shell_weight <= 1e-12:
            out.extend([0.0] * max_l)
            continue
        shell_vals = vals[shell] / shell_weight
        for l in range(1, max_l + 1):
            coeff_norm = 0.0
            for m in range(-l, l + 1):
                basis = _spherical_harmonic(l, m, polar[shell], azimuth[shell])
                coeff = np.sum(shell_vals * np.conjugate(basis))
                coeff_norm += float(np.abs(coeff) ** 2)
            out.append(float(np.sqrt(max(coeff_norm, 0.0))))
    return np.asarray(out, dtype=np.float64)


def radial_profile_descriptor(
    mask: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    n_bins: int = 8,
) -> np.ndarray:
    _cart, vals, radii = _normalised_cartesian(mask, weights)
    if radii.size == 0:
        return np.zeros((max(1, int(n_bins)),), dtype=np.float64)
    hist, _edges = np.histogram(
        radii,
        bins=max(1, int(n_bins)),
        range=(0.0, 1.0),
        weights=vals,
    )
    return np.asarray(hist, dtype=np.float64)


_NEIGHBOR_OFFSETS_26 = tuple(
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
)


def _clean_skeleton_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0 or int(np.count_nonzero(arr)) == 0:
        return np.zeros_like(arr, dtype=bool)
    return ndimage.binary_fill_holes(arr).astype(bool)


def _skeleton_points_A(mask: np.ndarray, voxel_size_zyx: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask).astype(np.float64)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return coords * np.asarray(voxel_size_zyx, dtype=np.float64).reshape(3)


def _skeleton_graph_stats(mask: np.ndarray, voxel_size_zyx: np.ndarray) -> dict:
    skeleton = np.asarray(mask, dtype=bool)
    coords = [tuple(int(v) for v in point) for point in np.argwhere(skeleton)]
    point_set = set(coords)
    point_count = len(coords)
    if point_count == 0:
        return {
            "point_count": 0,
            "endpoint_count": 0,
            "branchpoint_count": 0,
            "branch_count": 0,
            "length_A": 0.0,
        }

    neighbors_by_point = {}
    degrees = {}
    edge_lengths = {}
    for point in coords:
        neighbors = []
        for offset in _NEIGHBOR_OFFSETS_26:
            neighbor = (
                point[0] + offset[0],
                point[1] + offset[1],
                point[2] + offset[2],
            )
            if neighbor not in point_set:
                continue
            neighbors.append(neighbor)
            edge = tuple(sorted((point, neighbor)))
            if edge not in edge_lengths:
                edge_lengths[edge] = float(
                    np.linalg.norm(np.asarray(offset, dtype=np.float64) * voxel_size_zyx)
                )
        neighbors_by_point[point] = neighbors
        degrees[point] = len(neighbors)

    endpoint_count = sum(1 for degree in degrees.values() if degree == 1)
    branchpoint_count = sum(1 for degree in degrees.values() if degree >= 3)
    length_A = float(sum(edge_lengths.values()))

    critical = {point for point, degree in degrees.items() if degree != 2}
    if not critical:
        _labels, n_components = ndimage.label(skeleton)
        branch_count = int(n_components)
    else:
        visited_edges = set()
        branch_count = 0
        for start in sorted(critical):
            for neighbor in neighbors_by_point[start]:
                first_edge = tuple(sorted((start, neighbor)))
                if first_edge in visited_edges:
                    continue
                branch_count += 1
                previous = start
                current = neighbor
                visited_edges.add(first_edge)
                while current not in critical:
                    next_points = [
                        p
                        for p in neighbors_by_point[current]
                        if p != previous
                        and tuple(sorted((current, p))) not in visited_edges
                    ]
                    if not next_points:
                        break
                    next_point = sorted(next_points)[0]
                    visited_edges.add(tuple(sorted((current, next_point))))
                    previous, current = current, next_point

    return {
        "point_count": int(point_count),
        "endpoint_count": int(endpoint_count),
        "branchpoint_count": int(branchpoint_count),
        "branch_count": int(branch_count),
        "length_A": float(length_A),
    }


def skeleton_graph_metrics(
    exp_map: np.ndarray,
    sim_map: np.ndarray,
    *,
    threshold_frac: float = 0.05,
    tolerance_A: float = 1.5,
    voxel_size_xyz: Optional[tuple[float, float, float]] = None,
) -> dict:
    """Compare experimental and simulated density centerline skeletons.

    This is a TEASAR-like diagnostic built on scikit-image's 3D thinning
    skeleton. It reports precision/recall/F1 over nearest skeleton points, plus
    simple branch graph statistics. Exact TEASAR can later become another
    backend without changing the audit field names.
    """
    if _skeletonize is None:
        raise ImportError("skimage.morphology.skeletonize is unavailable")

    exp = _positive(exp_map)
    sim = _positive(sim_map)
    if exp.shape != sim.shape:
        raise ValueError(f"shape mismatch: {exp.shape} vs {sim.shape}")

    exp_mask, _exp_threshold = _density_mask(exp, threshold_frac)
    sim_mask, _sim_threshold = _density_mask(sim, threshold_frac)
    selection = _select_component(exp, exp_mask, sim_mask)
    exp_component = selection.mask

    exp_clean = _clean_skeleton_mask(exp_component)
    sim_clean = _clean_skeleton_mask(sim_mask)
    exp_skeleton = np.asarray(_skeletonize(exp_clean), dtype=bool)
    sim_skeleton = np.asarray(_skeletonize(sim_clean), dtype=bool)

    voxel_size_zyx = _as_zyx_voxel_size(voxel_size_xyz)
    exp_stats = _skeleton_graph_stats(exp_skeleton, voxel_size_zyx)
    sim_stats = _skeleton_graph_stats(sim_skeleton, voxel_size_zyx)
    exp_points = _skeleton_points_A(exp_skeleton, voxel_size_zyx)
    sim_points = _skeleton_points_A(sim_skeleton, voxel_size_zyx)

    tolerance = max(float(tolerance_A), 0.0)
    if exp_points.shape[0] == 0 or sim_points.shape[0] == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        mean_distance = None
        unmatched_density_length = exp_stats["length_A"]
        unmatched_ligand_length = sim_stats["length_A"]
    else:
        exp_tree = cKDTree(exp_points)
        sim_tree = cKDTree(sim_points)
        sim_to_exp, _ = exp_tree.query(sim_points, k=1)
        exp_to_sim, _ = sim_tree.query(exp_points, k=1)
        precision = float(np.mean(sim_to_exp <= tolerance))
        recall = float(np.mean(exp_to_sim <= tolerance))
        if precision + recall <= 1e-12:
            f1 = 0.0
        else:
            f1 = float(2.0 * precision * recall / (precision + recall))
        mean_distance = float((np.mean(sim_to_exp) + np.mean(exp_to_sim)) / 2.0)

        exp_step = exp_stats["length_A"] / max(exp_stats["point_count"], 1)
        sim_step = sim_stats["length_A"] / max(sim_stats["point_count"], 1)
        unmatched_density_length = float(np.sum(exp_to_sim > tolerance) * exp_step)
        unmatched_ligand_length = float(np.sum(sim_to_exp > tolerance) * sim_step)

    return {
        "density_skeleton_precision": float(precision),
        "density_skeleton_recall": float(recall),
        "density_skeleton_f1": float(f1),
        "density_skeleton_mean_distance_A": mean_distance,
        "density_skeleton_unmatched_density_length_A": float(unmatched_density_length),
        "density_skeleton_unmatched_ligand_length_A": float(unmatched_ligand_length),
        "density_skeleton_exp_branch_count": int(exp_stats["branch_count"]),
        "density_skeleton_sim_branch_count": int(sim_stats["branch_count"]),
        "density_skeleton_branch_count_diff": int(
            abs(exp_stats["branch_count"] - sim_stats["branch_count"])
        ),
        "density_skeleton_exp_point_count": int(exp_stats["point_count"]),
        "density_skeleton_sim_point_count": int(sim_stats["point_count"]),
        "density_skeleton_component_mode": str(selection.mode),
        "density_skeleton_backend": "skimage_skeletonize_proxy",
    }


def density_shape_metrics(
    exp_map: np.ndarray,
    sim_map: np.ndarray,
    *,
    threshold_frac: float = 0.05,
    zernike_order: int = 6,
    spharm_lmax: int = 6,
    spharm_shells: int = 5,
    radial_bins: int = 8,
    voxel_size_xyz: Optional[tuple[float, float, float]] = None,
    skeleton_tolerance_A: float = 1.5,
) -> dict:
    exp = _positive(exp_map)
    sim = _positive(sim_map)
    if exp.shape != sim.shape:
        raise ValueError(f"shape mismatch: {exp.shape} vs {sim.shape}")

    exp_mask, exp_threshold = _density_mask(exp, threshold_frac)
    sim_mask, sim_threshold = _density_mask(sim, threshold_frac)
    selection = _select_component(exp, exp_mask, sim_mask)
    exp_component = selection.mask

    exp_weights = exp * exp_component
    sim_weights = sim * sim_mask

    z_exp = zernike_style_descriptor(
        exp_component,
        exp_weights,
        order=zernike_order,
    )
    z_sim = zernike_style_descriptor(
        sim_mask,
        sim_weights,
        order=zernike_order,
    )
    z_simil, z_dist = _descriptor_similarity(z_exp, z_sim)

    sh_exp = spherical_harmonic_shell_descriptor(
        exp_component,
        exp_weights,
        lmax=spharm_lmax,
        n_shells=spharm_shells,
    )
    sh_sim = spherical_harmonic_shell_descriptor(
        sim_mask,
        sim_weights,
        lmax=spharm_lmax,
        n_shells=spharm_shells,
    )
    sh_simil, sh_dist = _descriptor_similarity(sh_exp, sh_sim)

    rp_exp = radial_profile_descriptor(exp_component, exp_weights, n_bins=radial_bins)
    rp_sim = radial_profile_descriptor(sim_mask, sim_weights, n_bins=radial_bins)
    rp_simil, _rp_dist = _descriptor_similarity(rp_exp, rp_sim)

    exp_volume = int(np.count_nonzero(exp_component))
    sim_volume = int(np.count_nonzero(sim_mask))
    max_volume = max(exp_volume, sim_volume, 1)
    volume_ratio = float(min(exp_volume, sim_volume) / max_volume)

    metrics = {
        "shape_zernike_similarity": float(z_simil),
        "shape_zernike_distance": float(z_dist),
        "shape_spharm_similarity": float(sh_simil),
        "shape_spharm_distance": float(sh_dist),
        "shape_radial_profile_similarity": float(rp_simil),
        "shape_volume_ratio": float(volume_ratio),
        "shape_exp_component_count": int(selection.component_count),
        "shape_selected_component_fraction": float(selection.selected_fraction),
        "shape_selected_component_mode": str(selection.mode),
        "shape_exp_threshold": float(exp_threshold),
        "shape_sim_threshold": float(sim_threshold),
        "shape_exp_component_volume": int(exp_volume),
        "shape_sim_volume": int(sim_volume),
        "shape_zernike_backend": "internal",
    }
    try:
        metrics.update(
            skeleton_graph_metrics(
                exp,
                sim,
                threshold_frac=threshold_frac,
                tolerance_A=skeleton_tolerance_A,
                voxel_size_xyz=voxel_size_xyz,
            )
        )
    except Exception as exc:
        metrics["skeleton_metrics_failed"] = type(exc).__name__
    return metrics
