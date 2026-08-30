from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np

from .map_metrics import DENSITY_METRICS, score_density_metrics, score_qscore


@dataclass
class ScoreResult:
    value: float
    terms: dict[str, Any] = field(default_factory=dict)


class BaseScorer:
    name: ClassVar[str] = "base"

    def __init__(self, options: Any = None, **params):
        self.options = _merged_options(options, params)
        self.spec_key = (self.name, self.__class__.__name__)

    def _opt(self, names: str | tuple[str, ...], default):
        if isinstance(names, str):
            names = (names,)
        for name in names:
            if hasattr(self.options, name):
                return getattr(self.options, name)
        return default

    def score(self, refine_ligand, coords_A: np.ndarray) -> ScoreResult:
        raise NotImplementedError


def _merged_options(options: Any = None, params: dict[str, Any] | None = None):
    values = {}
    if options is not None:
        values.update(getattr(options, "__dict__", {}))
    values.update(params or {})
    return SimpleNamespace(**values)


def _csv_items(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, str) and "," in item:
                out.extend(_csv_items(item))
            elif str(item).strip():
                out.append(item)
        return out
    return [value]


def _canonical_score_name(name) -> str:
    key = str(name).strip().lower()
    aliases = {
        "q": "qscore",
        "mapq": "qscore",
        "q-score": "qscore",
        "cc": "ccc",
    }
    key = aliases.get(key, key)
    if key not in SCORER_REGISTRY:
        raise ValueError(
            f"Unknown scorer {name!r}. Available: {sorted(SCORER_REGISTRY)}"
        )
    return key


def parse_score_spec(score_spec=None, weights=None) -> tuple[list[str], list[float]]:
    names = [_canonical_score_name(item) for item in _csv_items(score_spec or "qscore")]
    if not names:
        names = ["qscore"]

    weight_items = _csv_items(weights)
    if not weight_items:
        parsed_weights = [1.0] * len(names)
    else:
        parsed_weights = [float(item) for item in weight_items]
        if len(parsed_weights) != len(names):
            raise ValueError(
                "Number of score weights must match number of score names: "
                f"{len(parsed_weights)} weights for {len(names)} scores"
            )
    return names, parsed_weights


def _parse_radii(value):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            values = [float(item) for item in value.split(",") if item.strip()]
        except Exception:
            return None
        return np.asarray(values, dtype=np.float64) if values else None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    return arr if arr.size else None


class QScoreScorer(BaseScorer):
    name = "qscore"

    def __init__(
        self,
        *,
        sigma_ref: float = 0.6,
        radii: np.ndarray | None = None,
        low_tail_fraction: float = 0.3,
        options: Any = None,
        **params,
    ):
        super().__init__(
            options=options,
            sigma_ref=sigma_ref,
            radii=radii,
            low_tail_fraction=low_tail_fraction,
            **params,
        )

    def score(self, refine_ligand, coords_A: np.ndarray) -> ScoreResult:
        metric = score_qscore(
            refine_ligand,
            coords_A,
            sigma_ref=float(
                self._opt(
                    ("sigma_ref", "sr2_qscore_sigma_ref", "sr_qscore_sigma_ref"),
                    0.6,
                )
            ),
            radii=_parse_radii(
                self._opt(("radii", "sr2_qscore_radii", "sr_qscore_radii"), None)
            ),
            low_tail_fraction=float(self._opt("low_tail_fraction", 0.3)),
        )
        return ScoreResult(value=float(metric.value), terms=dict(metric.terms))


class _DensityMetricScorer(BaseScorer):
    metric_name: ClassVar[str] = ""

    def score(self, refine_ligand, coords_A: np.ndarray) -> ScoreResult:
        results = score_density_metrics(
            refine_ligand,
            coords_A,
            [self.metric_name],
            options=self.options,
        )
        metric = results[self.metric_name]
        return ScoreResult(
            value=float(metric.value),
            terms=dict(metric.terms),
        )


class CCCScorer(_DensityMetricScorer):
    name = "ccc"
    metric_name = "ccc"


class MIScorer(_DensityMetricScorer):
    name = "mi"
    metric_name = "mi"


class SCIScorer(_DensityMetricScorer):
    name = "sci"
    metric_name = "sci"


class WeightedScorer(BaseScorer):
    name = "weighted"

    def __init__(self, names, weights, *, options: Any = None, **params):
        super().__init__(options=options, **params)
        self.names = [str(name).lower().strip() for name in names]
        self.weights = [float(weight) for weight in weights]
        self.spec_key = tuple(zip(self.names, self.weights))

    def score(self, refine_ligand, coords_A: np.ndarray) -> ScoreResult:
        metric_values = {}
        metric_terms = {}

        if "qscore" in self.names:
            metric = score_qscore(
                refine_ligand,
                coords_A,
                sigma_ref=float(
                    self._opt(
                        ("sigma_ref", "sr2_qscore_sigma_ref", "sr_qscore_sigma_ref"),
                        0.6,
                    )
                ),
                radii=_parse_radii(
                    self._opt(("radii", "sr2_qscore_radii", "sr_qscore_radii"), None)
                ),
                low_tail_fraction=float(self._opt("low_tail_fraction", 0.3)),
            )
            metric_values["qscore"] = float(metric.value)
            metric_terms["qscore"] = dict(metric.terms)

        density_names = sorted({name for name in self.names if name in DENSITY_METRICS})
        density_results = score_density_metrics(
            refine_ligand,
            coords_A,
            density_names,
            options=self.options,
        )
        for name, metric in density_results.items():
            metric_values[name] = float(metric.value)
            metric_terms[name] = dict(metric.terms)

        components = []
        total = 0.0
        for name, weight in zip(self.names, self.weights):
            raw = float(metric_values[name])
            weighted = float(weight) * raw
            total += weighted
            components.append(
                {
                    "name": name,
                    "weight": float(weight),
                    "value": raw,
                    "weighted": weighted,
                }
            )

        return ScoreResult(
            value=float(total),
            terms={
                "combined_score": float(total),
                "score_components": components,
                "score_metric_terms": metric_terms,
            },
        )


SCORER_REGISTRY = {
    QScoreScorer.name: QScoreScorer,
    CCCScorer.name: CCCScorer,
    MIScorer.name: MIScorer,
    SCIScorer.name: SCIScorer,
}


def get_scorer(
    name: str | BaseScorer | None = None,
    *,
    weights=None,
    options: Any = None,
    **kwargs,
) -> BaseScorer:
    if isinstance(name, BaseScorer) or hasattr(name, "score"):
        return name

    names, parsed_weights = parse_score_spec(name, weights)
    if len(names) == 1 and float(parsed_weights[0]) == 1.0:
        return SCORER_REGISTRY[names[0]](options=options, **kwargs)
    return WeightedScorer(names, parsed_weights, options=options, **kwargs)
