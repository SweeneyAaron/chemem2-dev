from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

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


@dataclass
class ScoreResult:
    value: float
    terms: dict[str, Any] = field(default_factory=dict)


class BaseScorer:
    name: ClassVar[str] = "base"

    def score(self, refine_ligand, coords_A: np.ndarray) -> ScoreResult:
        raise NotImplementedError


def _as_coords(coords_A: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords_A, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords_A must have shape (N, 3), got {coords.shape}")
    return coords


def _map_reference(refine_ligand):
    return getattr(
        refine_ligand,
        "_map_reference",
        getattr(refine_ligand, "_map_referece", None),
    )


def _low_tail(values: np.ndarray, fraction: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    n_tail = max(1, int(np.ceil(finite.size * float(fraction))))
    return float(np.mean(np.sort(finite)[:n_tail]))


class QScoreScorer(BaseScorer):
    name = "qscore"

    def __init__(
        self,
        *,
        sigma_ref: float = 0.6,
        radii: np.ndarray | None = None,
        low_tail_fraction: float = 0.3,
    ):
        self.sigma_ref = float(sigma_ref)
        self.radii = None if radii is None else np.asarray(radii, dtype=np.float64)
        self.low_tail_fraction = float(low_tail_fraction)

    def score(self, refine_ligand, coords_A: np.ndarray) -> ScoreResult:
        if compute_qscores_from_emmap is None:
            raise RuntimeError(
                "Q-score support is unavailable; failed to import "
                "compute_qscores_from_emmap"
            ) from _QSCORE_IMPORT_ERROR

        coords = _as_coords(coords_A)
        local_coords = np.asarray(
            getattr(refine_ligand, "local_coords_A", np.zeros((0, 3))),
            dtype=np.float64,
        ).reshape((-1, 3))
        map_reference = _map_reference(refine_ligand)
        if map_reference is None:
            raise ValueError("QScoreScorer requires refine_ligand._map_reference")

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
            sigma_ref=self.sigma_ref,
            radii=self.radii,
            score_indices=ligand_indices,
        )
        q_scores = np.asarray(q_scores, dtype=np.float64).reshape(-1)
        finite = q_scores[np.isfinite(q_scores)]
        q_mean = float(np.mean(finite)) if finite.size else 0.0
        q_low_tail = _low_tail(q_scores, self.low_tail_fraction)

        return ScoreResult(
            value=q_mean,
            terms={
                "q_mean": q_mean,
                "q_low_tail": q_low_tail,
                "q_per_atom": [float(v) for v in q_scores],
            },
        )


SCORER_REGISTRY = {
    QScoreScorer.name: QScoreScorer,
}


def get_scorer(name: str | BaseScorer | None = None, **kwargs) -> BaseScorer:
    if isinstance(name, BaseScorer) or hasattr(name, "score"):
        return name
    key = str(name or QScoreScorer.name).lower()
    if key not in SCORER_REGISTRY:
        raise ValueError(
            f"Unknown scorer {name!r}. Available: {sorted(SCORER_REGISTRY)}"
        )
    return SCORER_REGISTRY[key](**kwargs)
