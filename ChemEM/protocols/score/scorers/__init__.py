#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Scorer registry for the ``--score`` protocol.

Same lazy ``"module:Class"`` trick as ``ChemEM.protocol_spec.ProtocolSpec``: this
module is imported by ``protocol_spec`` at CLI-build time to resolve
``--score-with`` into protocol dependencies, so it must stay import-light.
Selecting ``qscore`` must not drag in the compiled ECHO extension or OpenMM.

Adding a scorer
---------------
1) Write the class in ``scorers/<name>.py``, subclassing ``PoseScorer``.
2) Add one line to ``SCORER_REGISTRY``.
"""

from __future__ import annotations

from importlib import import_module

#: Registry key -> "module:Class". Order here is only the order ``--score-with all``
#: and ``--help`` present them in; the CSV follows the user's ``--score-with`` order.
SCORER_REGISTRY = {
    "echo": "ChemEM.protocols.score.scorers.echo:EchoScorer",
    "qscore": "ChemEM.protocols.score.scorers.qscore:QScoreScorer",
    "density": "ChemEM.protocols.score.scorers.density:DensityScorer",
    "mmgbsa": "ChemEM.protocols.score.scorers.mmgbsa:MMGBSAScorer",
    "strain": "ChemEM.protocols.score.scorers.strain:StrainScorer",
    "clash": "ChemEM.protocols.score.scorers.clash:ClashScorer",
}

SCORER_NAMES = tuple(SCORER_REGISTRY)


def load_scorer_cls(name: str) -> type:
    """Import and return the scorer class for ``name``."""
    try:
        path = SCORER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown scorer {name!r} (have: {', '.join(SCORER_NAMES)})"
        ) from None
    module_name, _, class_name = path.partition(":")
    return getattr(import_module(module_name), class_name)
