# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Tuple

import numpy as np


class AcceptOutcome(Enum):
    MAJOR = "major"
    MICRO = "micro"
    REJECT = "reject"
    PERTURB = "perturb"


def _score_of(record) -> float:
    if isinstance(record, dict):
        # During the search loop, ``score`` is always available immediately
        # while ``final_score`` is only globally normalized later.
        # Use the raw metric score for acceptance decisions.
        val = record.get("score", record.get("final_score", 0.0))
    else:
        val = getattr(record, "score", getattr(record, "final_score", 0.0))
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


class AcceptanceStrategy(ABC):
    def __init__(self, options: Any):
        self.options = options

    def _opt(self, name: str, default: Any) -> Any:
        return getattr(self.options, name, default)

    @abstractmethod
    def decide(
        self,
        accepted,
        proposals: List,
        iter_idx: int,
    ) -> Tuple[Any, AcceptOutcome]:
        ...


class GreedyAccept(AcceptanceStrategy):
    name = "greedy"

    def decide(self, accepted, proposals, iter_idx):
        if not proposals:
            return accepted, AcceptOutcome.REJECT

        min_delta = float(self._opt("sr_min_delta", 1e-4))
        accepted_score = _score_of(accepted)
        scores = [_score_of(p) for p in proposals]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        delta = best_score - accepted_score

        if delta <= 0.0:
            return accepted, AcceptOutcome.REJECT
        if delta > min_delta:
            return proposals[best_idx], AcceptOutcome.MAJOR
        return proposals[best_idx], AcceptOutcome.MICRO


class MetropolisAccept(AcceptanceStrategy):
    name = "metropolis"

    def __init__(self, options):
        super().__init__(options)
        seed = int(self._opt("sr_seed", 1))
        self._rng = np.random.default_rng(seed + 0xA11CE)

    def _temperature(self, iter_idx: int) -> float:
        t_start = float(self._opt("sr_accept_temp_start", 0.05))
        t_end = float(self._opt("sr_accept_temp_end", 0.005))
        total = max(1, int(self._opt("sr_max_outer_iter", 50)))
        frac = min(1.0, max(0.0, (float(iter_idx) - 1.0) / float(total)))
        return float(t_start + (t_end - t_start) * frac)

    def decide(self, accepted, proposals, iter_idx):
        if not proposals:
            return accepted, AcceptOutcome.REJECT

        min_delta = float(self._opt("sr_min_delta", 1e-4))
        accepted_score = _score_of(accepted)
        scores = [_score_of(p) for p in proposals]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        delta = best_score - accepted_score

        if delta >= 0.0:
            if delta > min_delta:
                return proposals[best_idx], AcceptOutcome.MAJOR
            if delta > 0.0:
                return proposals[best_idx], AcceptOutcome.MICRO
            return accepted, AcceptOutcome.REJECT

        T = self._temperature(iter_idx)
        if T <= 0.0:
            return accepted, AcceptOutcome.REJECT
        prob = float(np.exp(delta / T))
        if self._rng.random() < prob:
            return proposals[best_idx], AcceptOutcome.MICRO
        return accepted, AcceptOutcome.REJECT


class BasinHoppingAccept(AcceptanceStrategy):
    name = "basin_hopping"

    def __init__(self, options):
        super().__init__(options)
        self._inner = GreedyAccept(options)
        self._stale = 0

    def decide(self, accepted, proposals, iter_idx):
        picked, outcome = self._inner.decide(accepted, proposals, iter_idx)
        if outcome == AcceptOutcome.MAJOR:
            self._stale = 0
            return picked, outcome
        self._stale += 1

        stale_thr = int(self._opt("sr_basin_hop_stale", 3))
        if self._stale >= stale_thr:
            self._stale = 0
            return picked, AcceptOutcome.PERTURB
        return picked, outcome


_STRATEGIES = {
    GreedyAccept.name: GreedyAccept,
    MetropolisAccept.name: MetropolisAccept,
    BasinHoppingAccept.name: BasinHoppingAccept,
}


def get_acceptance(name: str, options: Any) -> AcceptanceStrategy:
    key = str(name or "greedy").lower()
    if key not in _STRATEGIES:
        raise ValueError(
            f"Unknown acceptance strategy '{name}'. Available: {sorted(_STRATEGIES.keys())}"
        )
    return _STRATEGIES[key](options)


__all__ = [
    "AcceptOutcome",
    "AcceptanceStrategy",
    "GreedyAccept",
    "MetropolisAccept",
    "BasinHoppingAccept",
    "get_acceptance",
]
