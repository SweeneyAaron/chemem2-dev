from __future__ import annotations

import math
import time
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.ndimage import map_coordinates

try:
    from ChemEM.protocols.core.sci_score import (
        simulate_ligand_density_subgrid,
        truncated_cc,
    )
except ModuleNotFoundError:  # pragma: no cover - source-tree fallback
    from protocols.core.sci_score import (
        simulate_ligand_density_subgrid,
        truncated_cc,
    )

from .types import AtomMapMetrics, LigandMapMetrics, SmartLigandRefinementConfig

try:
    from ChemEM.protocols.mapQ_score.mapq_utils import compute_qscores_from_emmap
except Exception:  # pragma: no cover - import-time optional dependency guard
    compute_qscores_from_emmap = None

try:
    from ChemEM import ligand_fitting as _ligand_fitting
except Exception:  # pragma: no cover - optional compiled extension
    _ligand_fitting = None


def _as_xyz(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.array([float(arr[0]), float(arr[0]), float(arr[0])], dtype=np.float64)
    if arr.size != 3:
        raise ValueError(f"Expected scalar or length-3 value, got {arr.shape}")
    return arr.astype(np.float64)


def _ligand_mol(ligand):
    return getattr(ligand, "mol", ligand)


def _atom_masses(ligand, atom_indices: Sequence[int]) -> np.ndarray:
    mol = _ligand_mol(ligand)
    if mol is None or not hasattr(mol, "GetAtomWithIdx"):
        return np.ones(len(atom_indices), dtype=np.float64)
    masses = []
    for idx in atom_indices:
        try:
            masses.append(float(mol.GetAtomWithIdx(int(idx)).GetMass()))
        except Exception:
            masses.append(1.0)
    return np.asarray(masses, dtype=np.float64)


def _active_atom_indices(ligand, n_atoms: int, score_hydrogens: bool) -> np.ndarray:
    if score_hydrogens:
        return np.arange(int(n_atoms), dtype=int)
    mol = _ligand_mol(ligand)
    if mol is None or not hasattr(mol, "GetAtoms"):
        return np.arange(int(n_atoms), dtype=int)
    idx = [
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if int(atom.GetIdx()) < n_atoms and int(atom.GetAtomicNum()) > 1
    ]
    if not idx:
        return np.arange(int(n_atoms), dtype=int)
    return np.asarray(idx, dtype=int)


def low_tail_q(q_scores: Iterable[float], fraction: float = 0.3) -> float:
    q_sorted = sorted(float(q) for q in q_scores if np.isfinite(float(q)))
    if not q_sorted:
        return 0.0
    n = max(1, int(len(q_sorted) * float(fraction)))
    return float(np.mean(q_sorted[:n]))


def _sample_grid(
    density: np.ndarray,
    coords_A: np.ndarray,
    origin_A: np.ndarray,
    apix_A: np.ndarray,
    *,
    mode: str = "nearest",
) -> np.ndarray:
    coords = np.asarray(coords_A, dtype=np.float64)
    fx = (coords[:, 0] - origin_A[0]) / max(float(apix_A[0]), 1e-12)
    fy = (coords[:, 1] - origin_A[1]) / max(float(apix_A[1]), 1e-12)
    fz = (coords[:, 2] - origin_A[2]) / max(float(apix_A[2]), 1e-12)
    sample = np.vstack([fz, fy, fx])
    values = map_coordinates(
        np.asarray(density, dtype=np.float64),
        sample,
        order=1,
        mode=mode,
        cval=0.0,
    )
    return np.asarray(values, dtype=np.float64)


def _local_gaussian_kernel(radius_A: float, sigma_A: float, apix_A: np.ndarray) -> tuple:
    r_vox = int(max(1, np.ceil(float(radius_A) / max(float(np.min(apix_A)), 1e-12))))
    z = (np.arange(-r_vox, r_vox + 1, dtype=np.float64) * apix_A[2])[:, None, None]
    y = (np.arange(-r_vox, r_vox + 1, dtype=np.float64) * apix_A[1])[None, :, None]
    x = (np.arange(-r_vox, r_vox + 1, dtype=np.float64) * apix_A[0])[None, None, :]
    d2 = x * x + y * y + z * z
    sigma2 = max(float(sigma_A) * float(sigma_A), 1e-12)
    kernel = np.exp(-0.5 * d2 / sigma2).astype(np.float64)
    return kernel, r_vox


def _local_ccc_at_atom(
    coord_A: np.ndarray,
    density: np.ndarray,
    origin_A: np.ndarray,
    apix_A: np.ndarray,
    kernel: np.ndarray,
    radius_vox: int,
) -> float:
    ix = int(round((coord_A[0] - origin_A[0]) / max(float(apix_A[0]), 1e-12)))
    iy = int(round((coord_A[1] - origin_A[1]) / max(float(apix_A[1]), 1e-12)))
    iz = int(round((coord_A[2] - origin_A[2]) / max(float(apix_A[2]), 1e-12)))

    z0, z1 = iz - radius_vox, iz + radius_vox + 1
    y0, y1 = iy - radius_vox, iy + radius_vox + 1
    x0, x1 = ix - radius_vox, ix + radius_vox + 1
    nz, ny, nx = density.shape

    patch = np.zeros_like(kernel, dtype=np.float64)
    fz0, fz1 = max(z0, 0), min(z1, nz)
    fy0, fy1 = max(y0, 0), min(y1, ny)
    fx0, fx1 = max(x0, 0), min(x1, nx)
    if fz0 < fz1 and fy0 < fy1 and fx0 < fx1:
        kz0, ky0, kx0 = fz0 - z0, fy0 - y0, fx0 - x0
        patch[
            kz0 : kz0 + (fz1 - fz0),
            ky0 : ky0 + (fy1 - fy0),
            kx0 : kx0 + (fx1 - fx0),
        ] = density[fz0:fz1, fy0:fy1, fx0:fx1]

    f = patch.ravel()
    t = kernel.ravel()
    f = f - float(np.mean(f))
    t = t - float(np.mean(t))
    denom = float(np.linalg.norm(f) * np.linalg.norm(t))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(np.dot(f, t) / denom, -1.0, 1.0))


def _stats(values: np.ndarray, mask: np.ndarray) -> tuple[int, float, float]:
    vals = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if vals.size == 0:
        return 0, 0.0, 0.0
    return int(vals.size), float(np.sum(vals)), float(np.dot(vals.ravel(), vals.ravel()))


def _ccc_from_sums(
    n: int,
    sum_a: float,
    sumsq_a: float,
    sum_b: float,
    sumsq_b: float,
    sum_ab: float,
    *,
    eps: float = 1e-12,
) -> float:
    if int(n) < 4:
        return 0.0
    nf = float(n)
    numerator = float(sum_ab) - (float(sum_a) * float(sum_b) / nf)
    var_a = float(sumsq_a) - (float(sum_a) * float(sum_a) / nf)
    var_b = float(sumsq_b) - (float(sum_b) * float(sum_b) / nf)
    if var_a < 0.0 and var_a > -1e-9:
        var_a = 0.0
    if var_b < 0.0 and var_b > -1e-9:
        var_b = 0.0
    denom = math.sqrt(max(var_a, 0.0) * max(var_b, 0.0))
    if denom < eps:
        return 0.0
    cc = float(numerator / denom)
    if not np.isfinite(cc):
        return 0.0
    return max(0.0, cc)


def _local_ccc_batch(
    coords_A: np.ndarray,
    density: np.ndarray,
    origin_A: np.ndarray,
    apix_A: np.ndarray,
    kernel: np.ndarray,
    radius_vox: int,
    density_padded: Optional[np.ndarray],
    density_windows: Optional[np.ndarray],
    kernel_centred_flat: np.ndarray,
    kernel_norm: float,
) -> np.ndarray:
    coords = np.asarray(coords_A, dtype=np.float64)
    out = np.zeros(int(coords.shape[0]), dtype=np.float64)
    if coords.size == 0:
        return out
    if density_windows is None or kernel_norm < 1e-12:
        return np.asarray(
            [
                _local_ccc_at_atom(c, density, origin_A, apix_A, kernel, radius_vox)
                for c in coords
            ],
            dtype=np.float64,
        )

    ix = np.rint((coords[:, 0] - origin_A[0]) / max(float(apix_A[0]), 1e-12)).astype(int)
    iy = np.rint((coords[:, 1] - origin_A[1]) / max(float(apix_A[1]), 1e-12)).astype(int)
    iz = np.rint((coords[:, 2] - origin_A[2]) / max(float(apix_A[2]), 1e-12)).astype(int)
    nz, ny, nx = density.shape
    valid = (iz >= 0) & (iy >= 0) & (ix >= 0) & (iz < nz) & (iy < ny) & (ix < nx)

    if (
        _ligand_fitting is not None
        and density_padded is not None
        and np.all(valid)
        and hasattr(_ligand_fitting, "compute_local_ccc_per_atom")
    ):
        try:
            return np.asarray(
                _ligand_fitting.compute_local_ccc_per_atom(
                    coords,
                    density_padded,
                    origin_A,
                    apix_A,
                    kernel_centred_flat,
                    float(kernel_norm),
                    int(radius_vox),
                ),
                dtype=np.float64,
            )
        except Exception:
            pass

    if np.any(valid):
        patches = density_windows[iz[valid], iy[valid], ix[valid]]
        flat = np.asarray(patches, dtype=np.float64).reshape((int(np.count_nonzero(valid)), -1))
        flat = flat - np.mean(flat, axis=1, keepdims=True)
        norms = np.linalg.norm(flat, axis=1)
        denom = norms * float(kernel_norm)
        numer = flat @ kernel_centred_flat
        values = np.zeros_like(numer, dtype=np.float64)
        ok = denom >= 1e-12
        values[ok] = numer[ok] / denom[ok]
        out[valid] = np.clip(values, -1.0, 1.0)

    if np.any(~valid):
        out[~valid] = np.asarray(
            [
                _local_ccc_at_atom(c, density, origin_A, apix_A, kernel, radius_vox)
                for c in coords[~valid]
            ],
            dtype=np.float64,
        )
    return out


class MapMetricEvaluator:
    """
    External ChemEM map scoring for SmartLigandRefinement.

    This class samples density/Q-score/CCC/gradient information directly from
    the ChemEM map objects. It never installs map forces in OpenMM.
    """

    def __init__(
        self,
        em_map,
        ligand=None,
        *,
        half_maps: Optional[Sequence] = None,
        difference_map=None,
        protein_coords_A: Optional[np.ndarray] = None,
        config: Optional[SmartLigandRefinementConfig] = None,
    ):
        self.em_map = em_map
        self.ligand = ligand
        self.half_maps = list(half_maps or [])
        self.difference_map = difference_map
        self.config = config or SmartLigandRefinementConfig()
        if protein_coords_A is None:
            protein_coords_A = self.config.protein_coords_A
        self.protein_coords_A = (
            np.asarray(protein_coords_A, dtype=np.float64)
            if protein_coords_A is not None
            else None
        )

        self._origin = None
        self._apix = None
        self._density = None
        self._grad_fields = None
        self._kernel = None
        self._kernel_radius_vox = None
        self._kernel_centred_flat = None
        self._kernel_norm = 0.0
        self._density_padded = None
        self._density_windows = None
        self._density_finite_stats = (0, 0.0, 0.0)
        self._density_nonzero_stats = (0, 0.0, 0.0)
        self._timings: dict[str, float] = {}
        self._call_counts: dict[str, int] = {}
        self._profile_timings = bool(getattr(self.config, "profile_timings", False))
        if em_map is not None:
            self._origin = np.asarray(em_map.origin, dtype=np.float64)
            self._apix = _as_xyz(em_map.apix)
            self._density = np.asarray(em_map.density_map, dtype=np.float64)
            finite = np.isfinite(self._density)
            nonzero = finite & (self._density != 0.0)
            self._density_finite_stats = _stats(self._density, finite)
            self._density_nonzero_stats = _stats(self._density, nonzero)
            # np.gradient returns components in array-axis order z, y, x.
            gz, gy, gx = np.gradient(
                self._density,
                max(float(self._apix[2]), 1e-12),
                max(float(self._apix[1]), 1e-12),
                max(float(self._apix[0]), 1e-12),
            )
            self._grad_fields = (gx, gy, gz)
            self._kernel, self._kernel_radius_vox = _local_gaussian_kernel(
                self.config.local_ccc_radius_A,
                self.config.local_ccc_sigma_A,
                self._apix,
            )
            kernel_centred = np.asarray(self._kernel, dtype=np.float64).ravel()
            kernel_centred = kernel_centred - float(np.mean(kernel_centred))
            self._kernel_centred_flat = kernel_centred
            self._kernel_norm = float(np.linalg.norm(kernel_centred))
            r = int(self._kernel_radius_vox)
            self._density_padded = np.pad(
                self._density,
                ((r, r), (r, r), (r, r)),
                mode="constant",
                constant_values=0.0,
            )
            try:
                self._density_windows = np.lib.stride_tricks.sliding_window_view(
                    self._density_padded,
                    self._kernel.shape,
                )
            except Exception:
                self._density_windows = None

    def _add_timing(self, name: str, seconds: float) -> None:
        if not self._profile_timings:
            return
        self._timings[name] = float(self._timings.get(name, 0.0) + seconds)

    def _count_call(self, name: str) -> None:
        if not self._profile_timings:
            return
        self._call_counts[name] = int(self._call_counts.get(name, 0) + 1)

    def _time_call(self, name: str, func, *args, **kwargs):
        if not self._profile_timings:
            return func(*args, **kwargs)
        self._count_call(name)
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            self._add_timing(name, time.perf_counter() - t0)

    def timing_report(self) -> dict:
        return {
            "timings_s": {str(k): float(v) for k, v in sorted(self._timings.items())},
            "call_counts": {str(k): int(v) for k, v in sorted(self._call_counts.items())},
        }

    def active_atom_indices(self, coords_A: np.ndarray) -> np.ndarray:
        return _active_atom_indices(
            self.ligand,
            int(np.asarray(coords_A).shape[0]),
            score_hydrogens=bool(self.config.score_hydrogens),
        )

    def evaluate(self, coords_A: np.ndarray) -> LigandMapMetrics:
        coords = np.asarray(coords_A, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords_A must have shape (N, 3), got {coords.shape}")

        active_idx = self.active_atom_indices(coords)
        if active_idx.size == 0 or self.em_map is None:
            return LigandMapMetrics({}, 0.0, 0.0, 0.0, [])

        self._count_call("evaluate")
        eval_t0 = time.perf_counter() if self._profile_timings else None
        active_coords = coords[active_idx]
        q_scores = self._time_call("evaluate.q_scores", self._q_scores, active_coords)
        map_values = self._time_call(
            "evaluate.sample_grid",
            _sample_grid,
            self._density,
            active_coords,
            self._origin,
            self._apix,
        )

        gradients = self._time_call(
            "evaluate.gradients",
            lambda: np.column_stack(
                [
                    _sample_grid(field, active_coords, self._origin, self._apix)
                    for field in self._grad_fields
                ]
            ),
        )
        gradients = np.where(np.isfinite(gradients), gradients, 0.0)

        local_ccc = self._time_call(
            "evaluate.local_ccc",
            _local_ccc_batch,
            active_coords,
            self._density,
            self._origin,
            self._apix,
            self._kernel,
            self._kernel_radius_vox,
            self._density_padded,
            self._density_windows,
            self._kernel_centred_flat,
            self._kernel_norm,
        )

        half_agreement = self._time_call(
            "evaluate.halfmap",
            self._halfmap_agreement,
            active_coords,
        )
        pos_diff, neg_diff = self._time_call(
            "evaluate.diffmap",
            self._difference_density,
            active_coords,
        )

        atom_metrics = {}
        for row, atom_idx in enumerate(active_idx.tolist()):
            atom_metrics[int(atom_idx)] = AtomMapMetrics(
                atom_index=int(atom_idx),
                q_score=float(q_scores[row]),
                local_ccc=float(local_ccc[row]),
                map_value=float(map_values[row]),
                map_gradient=np.asarray(gradients[row], dtype=np.float64),
                halfmap_agreement=(
                    None if half_agreement is None else float(half_agreement[row])
                ),
                positive_diff_density=(None if pos_diff is None else float(pos_diff[row])),
                negative_diff_density=(None if neg_diff is None else float(neg_diff[row])),
            )

        ligand_ccc = self._time_call(
            "evaluate.ligand_ccc",
            self._ligand_ccc,
            active_coords,
            active_idx,
        )
        q_values = [m.q_score for m in atom_metrics.values()]
        mean_q = float(np.mean(q_values)) if q_values else 0.0
        low_q = low_tail_q(q_values, fraction=0.3)
        worst = sorted(atom_metrics, key=lambda i: atom_metrics[i].q_score)
        n_worst = max(1, int(len(worst) * 0.3)) if worst else 0

        out = LigandMapMetrics(
            atom_metrics=atom_metrics,
            ligand_ccc=float(ligand_ccc),
            low_tail_q=float(low_q),
            mean_q=float(mean_q),
            worst_atom_indices=[int(i) for i in worst[:n_worst]],
        )
        if self._profile_timings and eval_t0 is not None:
            self._add_timing("evaluate.total", time.perf_counter() - eval_t0)
        return out

    def _q_scores(self, active_coords_A: np.ndarray) -> np.ndarray:
        n = int(active_coords_A.shape[0])
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        if compute_qscores_from_emmap is not None:
            try:
                if self.protein_coords_A is not None and self.protein_coords_A.size:
                    all_coords = np.concatenate(
                        [active_coords_A, self.protein_coords_A], axis=0
                    )
                else:
                    all_coords = active_coords_A
                q_all = compute_qscores_from_emmap(
                    atoms_xyz=all_coords,
                    emmap=self.em_map,
                    sigma_ref=float(self.config.qscore_sigma_ref),
                    radii=None,
                    score_indices=np.arange(n, dtype=int),
                )
                q = np.asarray(q_all, dtype=np.float64)
                return np.where(np.isfinite(q), q, 0.0)
            except Exception:
                pass

        # Fallback: turn local sampled map amplitude into a bounded support
        # proxy. This keeps the protocol usable in light unit tests or stripped
        # installations where the Q-score extension is unavailable.
        vals = _sample_grid(self._density, active_coords_A, self._origin, self._apix)
        finite = np.asarray(vals[np.isfinite(vals)], dtype=np.float64)
        if finite.size == 0:
            return np.zeros(n, dtype=np.float64)
        lo, hi = np.percentile(finite, [5.0, 95.0])
        span = max(float(hi - lo), 1e-12)
        return np.clip((vals - lo) / span, 0.0, 1.0)

    def _ligand_ccc_from_subgrid(
        self,
        sim_subgrid: np.ndarray,
        lo_zyx: np.ndarray,
        hi_zyx: np.ndarray,
    ) -> float:
        sim = np.asarray(sim_subgrid, dtype=np.float64)
        if sim.size == 0:
            return 0.0

        z0, y0, x0 = [int(i) for i in lo_zyx]
        z1, y1, x1 = [int(i) for i in hi_zyx]
        exp_sub = np.asarray(self._density[z0:z1, y0:y1, x0:x1], dtype=np.float64)
        if exp_sub.shape != sim.shape:
            return 0.0

        if _ligand_fitting is not None and hasattr(
            _ligand_fitting, "compute_ligand_ccc_decomposed"
        ):
            try:
                return float(
                    _ligand_fitting.compute_ligand_ccc_decomposed(
                        exp_sub,
                        sim,
                        int(self._density_nonzero_stats[0]),
                        float(self._density_nonzero_stats[1]),
                        float(self._density_nonzero_stats[2]),
                        int(self._density_finite_stats[0]),
                        float(self._density_finite_stats[1]),
                        float(self._density_finite_stats[2]),
                    )
                )
            except Exception:
                pass

        exp_finite = np.isfinite(exp_sub)
        sim_finite = np.isfinite(sim)
        finite = exp_finite & sim_finite
        union = finite & ((exp_sub != 0.0) | (sim != 0.0))
        bbox_nonzero = finite & (exp_sub != 0.0)

        full_nz_n, full_nz_sum, full_nz_sumsq = self._density_nonzero_stats
        bbox_nz_n, bbox_nz_sum, bbox_nz_sumsq = _stats(exp_sub, bbox_nonzero)
        inside_n, inside_sum_a, inside_sumsq_a = _stats(exp_sub, union)
        sim_union = sim[union]
        exp_union = exp_sub[union]

        n = int(full_nz_n - bbox_nz_n + inside_n)
        sum_a = float(full_nz_sum - bbox_nz_sum + inside_sum_a)
        sumsq_a = float(full_nz_sumsq - bbox_nz_sumsq + inside_sumsq_a)
        sum_b = float(np.sum(sim_union)) if sim_union.size else 0.0
        sumsq_b = float(np.dot(sim_union.ravel(), sim_union.ravel())) if sim_union.size else 0.0
        sum_ab = float(np.dot(exp_union.ravel(), sim_union.ravel())) if sim_union.size else 0.0

        if n < 64:
            full_finite_n, full_finite_sum, full_finite_sumsq = self._density_finite_stats
            sim_finite_vals = sim[finite]
            exp_finite_vals = exp_sub[finite]
            n = int(full_finite_n)
            sum_a = float(full_finite_sum)
            sumsq_a = float(full_finite_sumsq)
            sum_b = (
                float(np.sum(sim_finite_vals)) if sim_finite_vals.size else 0.0
            )
            sumsq_b = (
                float(np.dot(sim_finite_vals.ravel(), sim_finite_vals.ravel()))
                if sim_finite_vals.size
                else 0.0
            )
            sum_ab = (
                float(np.dot(exp_finite_vals.ravel(), sim_finite_vals.ravel()))
                if sim_finite_vals.size
                else 0.0
            )

        return _ccc_from_sums(n, sum_a, sumsq_a, sum_b, sumsq_b, sum_ab)

    def _ligand_ccc(self, active_coords_A: np.ndarray, active_idx: np.ndarray) -> float:
        if self.em_map is None or active_coords_A.size == 0:
            return 0.0
        try:
            resolution = getattr(self.em_map, "resolution", None)
            try:
                resolution = float(resolution)
            except (TypeError, ValueError):
                resolution = None
            if resolution is None or not np.isfinite(resolution) or resolution <= 0.0:
                resolution = float(self.config.resolution_A)

            masses = _atom_masses(self.ligand, active_idx)
            sim_timings = self._timings if self._profile_timings else None
            sim_subgrid, lo_zyx, hi_zyx = simulate_ligand_density_subgrid(
                coords_xyz_A=active_coords_A,
                atom_masses=masses,
                map_origin_xyz_A=self._origin,
                map_apix_xyz_A=self._apix,
                map_shape_zyx=self._density.shape,
                resolution_A=float(resolution),
                sigma_coeff=float(self.config.sigma_coeff),
                normalise=True,
                timings=sim_timings,
            )
            return float(self._ligand_ccc_from_subgrid(sim_subgrid, lo_zyx, hi_zyx))
        except Exception:
            vals = _sample_grid(self._density, active_coords_A, self._origin, self._apix)
            if vals.size == 0:
                return 0.0
            return float(np.mean(vals))

    def _halfmap_agreement(self, active_coords_A: np.ndarray) -> Optional[np.ndarray]:
        if len(self.half_maps) < 2:
            return None
        try:
            h1, h2 = self.half_maps[:2]
            v1 = _sample_grid(
                np.asarray(h1.density_map, dtype=np.float64),
                active_coords_A,
                np.asarray(h1.origin, dtype=np.float64),
                _as_xyz(h1.apix),
            )
            v2 = _sample_grid(
                np.asarray(h2.density_map, dtype=np.float64),
                active_coords_A,
                np.asarray(h2.origin, dtype=np.float64),
                _as_xyz(h2.apix),
            )
            denom = np.abs(v1) + np.abs(v2) + 1e-12
            return np.clip(1.0 - (np.abs(v1 - v2) / denom), 0.0, 1.0)
        except Exception:
            return None

    def _difference_density(self, active_coords_A: np.ndarray) -> tuple:
        if self.difference_map is None:
            return None, None
        try:
            vals = _sample_grid(
                np.asarray(self.difference_map.density_map, dtype=np.float64),
                active_coords_A,
                np.asarray(self.difference_map.origin, dtype=np.float64),
                _as_xyz(self.difference_map.apix),
            )
            return np.clip(vals, 0.0, None), np.clip(-vals, 0.0, None)
        except Exception:
            return None, None
