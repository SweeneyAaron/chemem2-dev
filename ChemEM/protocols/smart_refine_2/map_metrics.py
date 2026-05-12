from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

try:
    from ChemEM.protocols.core.density import mutual_information_score
    from ChemEM.protocols.core.sci_score import (
        sci_score_3d,
        simulate_ligand_density_subgrid,
        truncated_cc,
    )
except ModuleNotFoundError:  # pragma: no cover - source-tree fallback
    from protocols.core.density import mutual_information_score
    from protocols.core.sci_score import (
        sci_score_3d,
        simulate_ligand_density_subgrid,
        truncated_cc,
    )

try:
    from ChemEM.protocols.mapQ_score.mapq_utils import compute_qscores_from_emmap
    _QSCORE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional runtime dependency
    try:
        from protocols.mapQ_score.mapq_utils import compute_qscores_from_emmap
        _QSCORE_IMPORT_ERROR = None
    except Exception as fallback_exc:  # pragma: no cover
        compute_qscores_from_emmap = None
        _QSCORE_IMPORT_ERROR = fallback_exc or exc


DENSITY_METRICS = {"ccc", "mi", "sci"}


@dataclass
class MetricValue:
    value: float
    terms: dict[str, Any]


@dataclass
class MapMetricContext:
    map_reference: Any
    origin_A: np.ndarray
    apix_A: np.ndarray
    density: np.ndarray
    shape_zyx: tuple[int, int, int]
    resolution_A: float
    masses: np.ndarray
    key: tuple


def _as_coords(coords_A: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords_A, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords_A must have shape (N, 3), got {coords.shape}")
    return coords


def _as_xyz(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.array([float(arr[0])] * 3, dtype=np.float64)
    if arr.size != 3:
        raise ValueError(f"Expected scalar or length-3 value, got {arr.shape}")
    return arr.astype(np.float64)


def _map_reference(refine_ligand):
    return getattr(
        refine_ligand,
        "_map_reference",
        getattr(refine_ligand, "_map_referece", None),
    )


def _ligand_mol(refine_ligand):
    ligand = getattr(refine_ligand, "_ligand", None)
    return getattr(ligand, "mol", getattr(refine_ligand, "mol", None))


def _atom_masses(refine_ligand, n_atoms: int) -> np.ndarray:
    atom_indices = np.asarray(
        getattr(refine_ligand, "_atom_indices", np.arange(int(n_atoms), dtype=int)),
        dtype=int,
    ).reshape(-1)
    if atom_indices.shape[0] != int(n_atoms):
        atom_indices = np.arange(int(n_atoms), dtype=int)

    mol = _ligand_mol(refine_ligand)
    if mol is None or not hasattr(mol, "GetAtomWithIdx"):
        return np.ones(int(n_atoms), dtype=np.float64)

    masses = []
    for idx in atom_indices:
        try:
            masses.append(float(mol.GetAtomWithIdx(int(idx)).GetMass()))
        except Exception:
            masses.append(1.0)
    return np.asarray(masses, dtype=np.float64)


def _option(options, names: str | Iterable[str], default):
    if isinstance(names, str):
        names = (names,)
    for name in names:
        if options is not None and hasattr(options, name):
            return getattr(options, name)
    return default


def _resolution_A(map_reference, options) -> float:
    resolution = getattr(map_reference, "resolution", None)
    try:
        resolution = float(resolution)
    except (TypeError, ValueError):
        resolution = None
    if resolution is None or not np.isfinite(resolution) or resolution <= 0.0:
        resolution = float(
            _option(options, ("sr2_resolution_A", "sr_resolution", "resolution"), 3.0)
        )
    return float(resolution)


def _low_tail(values: np.ndarray, fraction: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    n_tail = max(1, int(np.ceil(finite.size * float(fraction))))
    return float(np.mean(np.sort(finite)[:n_tail]))


def _context_key(refine_ligand, map_reference, coords: np.ndarray) -> tuple:
    atom_indices = tuple(
        int(i)
        for i in np.asarray(
            getattr(refine_ligand, "_atom_indices", np.arange(coords.shape[0])),
            dtype=int,
        ).reshape(-1).tolist()
    )
    return (id(map_reference), int(coords.shape[0]), atom_indices)


def map_metric_context(refine_ligand, coords_A: np.ndarray, options=None) -> MapMetricContext:
    coords = _as_coords(coords_A)
    map_reference = _map_reference(refine_ligand)
    if map_reference is None:
        raise ValueError("Map metrics require refine_ligand._map_reference")
    if not hasattr(map_reference, "density_map"):
        raise ValueError("CCC/MI/SCI metrics require an EM map with density_map")

    key = _context_key(refine_ligand, map_reference, coords)
    cached = getattr(refine_ligand, "_sr2_map_metric_context", None)
    if isinstance(cached, MapMetricContext) and cached.key == key:
        return cached

    density = np.asarray(map_reference.density_map, dtype=np.float64)
    context = MapMetricContext(
        map_reference=map_reference,
        origin_A=_as_xyz(getattr(map_reference, "origin", (0.0, 0.0, 0.0))),
        apix_A=_as_xyz(getattr(map_reference, "apix", 1.0)),
        density=density,
        shape_zyx=tuple(int(i) for i in density.shape),
        resolution_A=_resolution_A(map_reference, options),
        masses=_atom_masses(refine_ligand, coords.shape[0]),
        key=key,
    )
    refine_ligand._sr2_map_metric_context = context
    return context


def score_qscore(
    refine_ligand,
    coords_A: np.ndarray,
    *,
    sigma_ref: float = 0.6,
    radii: np.ndarray | None = None,
    low_tail_fraction: float = 0.3,
) -> MetricValue:
    if compute_qscores_from_emmap is None:
        raise RuntimeError(
            "Q-score support is unavailable; failed to import "
            "compute_qscores_from_emmap"
        ) from _QSCORE_IMPORT_ERROR

    coords = _as_coords(coords_A)
    map_reference = _map_reference(refine_ligand)
    if map_reference is None:
        raise ValueError("QScoreScorer requires refine_ligand._map_reference")

    local_coords = np.asarray(
        getattr(refine_ligand, "local_coords_A", np.zeros((0, 3))),
        dtype=np.float64,
    ).reshape((-1, 3))
    if hasattr(refine_ligand, "qscore_context_coords_A"):
        context_coords = refine_ligand.qscore_context_coords_A(coords)
    elif local_coords.size:
        context_coords = np.concatenate([coords, local_coords], axis=0)
    else:
        context_coords = coords

    ligand_indices = (
        refine_ligand.ligand_score_indices()
        if hasattr(refine_ligand, "ligand_score_indices")
        else np.arange(coords.shape[0], dtype=int)
    )
    q_scores = compute_qscores_from_emmap(
        atoms_xyz=context_coords,
        emmap=map_reference,
        sigma_ref=float(sigma_ref),
        radii=radii,
        score_indices=ligand_indices,
    )
    q_scores = np.asarray(q_scores, dtype=np.float64).reshape(-1)
    finite = q_scores[np.isfinite(q_scores)]
    q_mean = float(np.mean(finite)) if finite.size else 0.0
    q_low_tail = _low_tail(q_scores, float(low_tail_fraction))
    return MetricValue(
        value=q_mean,
        terms={
            "q_mean": q_mean,
            "q_low_tail": q_low_tail,
            "q_per_atom": [float(v) for v in q_scores],
        },
    )


def score_density_metrics(
    refine_ligand,
    coords_A: np.ndarray,
    metric_names: Iterable[str],
    *,
    options=None,
) -> dict[str, MetricValue]:
    names = [str(name).lower().strip() for name in metric_names]
    requested = {name for name in names if name in DENSITY_METRICS}
    if not requested:
        return {}

    coords = _as_coords(coords_A)
    context = map_metric_context(refine_ligand, coords, options=options)
    sigma_coeff = float(
        _option(options, ("sr2_sigma_coeff", "sr_sigma_coeff"), 0.356)
    )
    normalise = bool(
        _option(options, ("sr2_normalise_sim_map", "sr_normalise_sim_map"), True)
    )
    sim_map, lo_zyx, hi_zyx = simulate_ligand_density_subgrid(
        coords_xyz_A=coords,
        atom_masses=context.masses,
        map_origin_xyz_A=context.origin_A,
        map_apix_xyz_A=context.apix_A,
        map_shape_zyx=context.shape_zyx,
        resolution_A=context.resolution_A,
        sigma_coeff=sigma_coeff,
        normalise=normalise,
    )
    if sim_map.size == 0:
        return {
            name: MetricValue(value=0.0, terms={name: 0.0})
            for name in requested
        }

    z0, y0, x0 = [int(i) for i in lo_zyx]
    z1, y1, x1 = [int(i) for i in hi_zyx]
    exp_map = np.asarray(context.density[z0:z1, y0:y1, x0:x1], dtype=np.float64)
    if exp_map.shape != sim_map.shape:
        return {
            name: MetricValue(value=0.0, terms={name: 0.0})
            for name in requested
        }

    results: dict[str, MetricValue] = {}
    if "ccc" in requested:
        mask_mode = str(
            _option(options, ("sr2_ccc_mask_mode", "sr_ccc_mask_mode"), "nonzero")
        ).lower()
        mask = np.ones_like(exp_map, dtype=bool) if mask_mode == "full" else None
        ccc = float(truncated_cc(exp_map, sim_map, mask))
        results["ccc"] = MetricValue(
            value=ccc,
            terms={
                "ccc": ccc,
                "ccc_mask_mode": mask_mode,
                "metric_subgrid_shape": tuple(int(i) for i in sim_map.shape),
            },
        )

    if "mi" in requested:
        n_bins = int(_option(options, ("sr2_mi_nbins", "sr_mi_nbins"), 64))
        normalized = bool(
            _option(options, ("sr2_mi_normalized", "sr_mi_normalized"), False)
        )
        mi = float(
            mutual_information_score(
                exp_map,
                sim_map,
                n_bins=n_bins,
                nonzero_union=True,
                normalized=False,
            )
        )
        nmi = (
            float(
                mutual_information_score(
                    exp_map,
                    sim_map,
                    n_bins=n_bins,
                    nonzero_union=True,
                    normalized=True,
                )
            )
            if normalized
            else 0.0
        )
        results["mi"] = MetricValue(
            value=float(nmi if normalized else mi),
            terms={
                "mi": mi,
                "normalized_mi": nmi,
                "mi_nbins": int(n_bins),
                "mi_normalized": bool(normalized),
                "metric_subgrid_shape": tuple(int(i) for i in sim_map.shape),
            },
        )

    if "sci" in requested:
        sci, details = sci_score_3d(
            exp_map,
            sim_map,
            use_amp_eq=bool(
                _option(options, ("sr2_use_amp_eq", "sr_use_amp_eq"), True)
            ),
            sigma=float(_option(options, ("sr2_sci_sigma", "sr_sci_sigma"), 1.0)),
            w0=float(_option(options, ("sr2_w0", "sr_w0"), 1.0)),
            w1=float(_option(options, ("sr2_w1", "sr_w1"), 1.0)),
            w2=float(_option(options, ("sr2_w2", "sr_w2"), 1.0)),
            eps=float(_option(options, ("sr2_sci_eps", "sr_sci_eps"), 1e-8)),
        )
        terms = {
            key: float(details.get(key, 0.0))
            for key in ("cc0", "ccx", "ccy", "ccz", "ccxx", "ccyy", "cczz", "sci")
        }
        terms["metric_subgrid_shape"] = tuple(int(i) for i in sim_map.shape)
        results["sci"] = MetricValue(value=float(sci), terms=terms)

    return results
