#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Q-score of a pose against the density map (``--score --score-with qscore``).

Needs no binding site and no segmentation -- just the map and the heavy atoms -- so
this is the cheap scorer, and ``score_deps`` returns ``()`` for it.

Scored against the **full** ``system.density_map``, not a site crop. Read that
alongside the density scorer's numbers with care: they may be measuring different
maps (see ``--score-density-region``).

Not bitwise identical to the ``--mapq-score`` protocol this replaces, by about 1e-7:
``compute_qscores_from_emmap`` returns float32, and the old protocol averaged in
float32 where ``qscore_pose_metrics`` promotes to float64 first. The float64 value is
the more accurate one and is what the orchestrator has always ranked on. Measured on
7jjo: 0.7816745042800903 (old) vs 0.7816745638847351 (new).
"""

from __future__ import annotations

from ChemEM.protocols.orchestrator.scoring import qscore_pose_metrics

from .base import PoseScorer

# Per-atom lists cannot go in a flat CSV row; they are routed to pose_scores.json.
JSON_ONLY = ("q_per_atom", "q_heavy_atom_indices")


class QScoreScorer(PoseScorer):
    NAME = "qscore"
    HELP = "Q-score of the pose against the density map"
    DEPS = ()
    HEADLINE = "qscore"
    HIGHER_IS_BETTER = True
    NEEDS_SITE = False

    COLUMNS = ("qscore", "q_mean", "q_low_tail")

    def setup_run(self, ctx) -> None:
        self.density_map = getattr(self.system, "density_map", None)
        if self.density_map is None:
            raise ValueError(
                "Q-score scoring requires a density map. Set `densmap =` in the "
                "config, and do not pass --no-map."
            )
        # --sigma-ref is shared with the orchestrator and smart_refine_2; the
        # per-scorer flag only overrides it here.
        self.sigma_ref = float(
            self._opt("score_qscore_sigma_ref", None)
            or self._opt("sigma_ref", None)
            or 0.6
        )
        self.low_tail = float(self._opt("score_qscore_low_tail_fraction", 0.3))
        self.per_atom = bool(self._opt("score_qscore_per_atom", False))
        self.system.log(
            f"[score:qscore] sigma_ref={self.sigma_ref} "
            f"low_tail_fraction={self.low_tail} map=full"
        )

    def score(self, pose, row) -> None:
        metrics = qscore_pose_metrics(
            pose.coords,
            pose.mol,
            self.density_map,
            sigma_ref=self.sigma_ref,
            low_tail_fraction=self.low_tail,
        )
        if metrics is None:
            # The pose lies outside the map, or has no heavy atoms. Not an error --
            # a decoy scattered outside the box is a legitimate thing to score.
            row["qscore_failed"] = 1
            return
        for key, value in metrics.items():
            if key in JSON_ONLY:
                if self.per_atom:
                    row.setdefault("_json", {})[key] = value
                continue
            row[key] = value

    def finish_run(self, ctx, rows) -> dict:
        return {"sigma_ref": self.sigma_ref, "region": "full",
                "per_atom": self.per_atom}
