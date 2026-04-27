# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

from typing import Any

from .base import BaseScorer
from .ccc import CCCScorer
from .mi import MIScorer
from .qscore import QScorer
from .sci import SCIScorer


SCORER_REGISTRY = {
    SCIScorer.name: SCIScorer,
    CCCScorer.name: CCCScorer,
    MIScorer.name: MIScorer,
    QScorer.name: QScorer,
}


def get_scorer(name: str, options: Any) -> BaseScorer:
    key = str(name or "sci").lower()
    if key not in SCORER_REGISTRY:
        raise ValueError(
            f"Unknown scorer '{name}'. Available: {sorted(SCORER_REGISTRY.keys())}"
        )
    return SCORER_REGISTRY[key](options)


__all__ = ["BaseScorer", "SCIScorer", "CCCScorer", "MIScorer", "QScorer",
           "SCORER_REGISTRY", "get_scorer"]
