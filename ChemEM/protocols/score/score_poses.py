#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""``--score``: score already-placed ligand poses with one or more scorers.

    python -m ChemEM conf.txt --score --score-with echo,mmgbsa,qscore,density

Poses come from the config -- ``ligand = poses.sdf`` (one conformer per pose) or
``ligands_from_dir = <dir>`` -- so ``--score`` also composes with ``--dock`` in a
single run.

This protocol owns the loop and the bookkeeping; the scorers own the physics. Each
scorer contributes its own columns to a shared per-pose row, and a scorer that fails
on one pose costs that pose only that scorer's columns.
"""

from __future__ import annotations

import os
import time

from ChemEM.messages import Messages
from ChemEM.protocols.score.cli import resolve_scorers
from ChemEM.protocols.score.context import ScoreContext
from ChemEM.protocols.score.poses import iter_ligands
from ChemEM.protocols.score.scorers import load_scorer_cls

from . import rows as rows_mod


class ScorePoses:
    """Run the selected scorers over every pose in the config."""

    def __init__(self, system):
        self.system = system
        self.output = None
        self.scorers = []
        self.rows = []

    # ------------------------------------------------------------------ setup

    def _opt(self, name, default=None):
        return getattr(self.system.options, name, default)

    def _get_output(self):
        base = getattr(self.system, "output", None) or "."
        self.system.output = base
        self.output = os.path.join(base, self._opt("score_out", "score") or "score")
        os.makedirs(self.output, exist_ok=True)
        return self.output

    def _build_scorers(self, names):
        return [load_scorer_cls(name)(self.system, self.system.options)
                for name in names]

    def _rank_by(self):
        """Column the SDFs are sorted by, and whether larger is better."""
        explicit = self._opt("score_rank_by", None)
        if explicit:
            for scorer in self.scorers:
                if scorer.HEADLINE == explicit:
                    return explicit, scorer.HIGHER_IS_BETTER
            # An arbitrary column: no scorer owns it, so assume lower is better,
            # which is the convention every ChemEM score already follows.
            return explicit, False
        for scorer in self.scorers:
            if scorer.HEADLINE:
                return scorer.HEADLINE, scorer.HIGHER_IS_BETTER
        return None, False

    # -------------------------------------------------------------------- run

    def run(self):
        self.system.log(Messages.create_centered_box("Score Poses"))

        if not getattr(self.system, "ligand", None):
            raise ValueError(
                "No poses to score. Supply them with `ligand = poses.sdf` or "
                "`ligands_from_dir = <dir>` in the config."
            )

        names = resolve_scorers(self.system.options)
        self._get_output()
        self.scorers = self._build_scorers(names)

        case_id = (self._opt("score_case_id", None)
                   or os.path.basename(os.path.normpath(self.system.output))
                   or "case")
        ctx = ScoreContext(self.system, self.system.options, self.output,
                           case_id, self.scorers)

        self.system.log(f"[score] scorers: {', '.join(names)}")
        self.system.log(f"[score] case={case_id} output={self.output}")

        timings = {name: 0.0 for name in names}
        failures = {name: 0 for name in names}

        for scorer in self.scorers:
            scorer.setup_run(ctx)

        for lig_ctx, poses in iter_ligands(self.system, ctx):
            if not poses:
                self.system.log(
                    f"[score] ligand {lig_ctx.ligand_idx} has no conformers; skipping."
                )
                continue
            self._score_ligand(ctx, lig_ctx, poses, timings, failures)

        meta = {}
        for scorer in self.scorers:
            meta[scorer.NAME] = dict(
                scorer.finish_run(ctx, self.rows) or {},
                seconds=round(timings[scorer.NAME], 3),
                failed=failures[scorer.NAME],
                scored=len(self.rows) - failures[scorer.NAME],
            )

        self._write(ctx, names, meta)

        total_failures = sum(failures.values())
        self.system.log(
            f"[score] {len(self.rows)} pose(s) x {len(names)} scorer(s)"
            + (f", {total_failures} scorer failure(s)" if total_failures else "")
            + f" -> {self.output}"
        )
        return self.rows

    def _score_ligand(self, ctx, lig_ctx, poses, timings, failures):
        for scorer in self.scorers:
            scorer.setup_ligand(lig_ctx)
        try:
            for pose in poses:
                self.rows.append(
                    self._score_pose(ctx, lig_ctx, pose, timings, failures)
                )
        finally:
            # Precompute caches are large; drop them even if a pose blew up.
            for scorer in self.scorers:
                scorer.teardown_ligand(lig_ctx)

    def _score_pose(self, ctx, lig_ctx, pose, timings, failures):
        row = {
            "case_id": ctx.case_id,
            "ligand": lig_ctx.ligand.identifier,
            "source": lig_ctx.source,
            "pose": pose.pose,
            "ligand_idx": pose.ligand_idx,
            "conf_id": pose.conf_id,
            "site_id": "",
            "error": "",
        }

        # Hydrogen relaxation happens here, before ANY scorer scores: it mutates
        # coordinates, and every scorer must read the geometry that was scored.
        for scorer in self.scorers:
            self._guard(scorer, "pre_score", pose, row, timings, failures)

        for scorer in self.scorers:
            self._guard(scorer, "score", pose, row, timings, failures)

        # ...and the receptor is restored only after every scorer has read it.
        for scorer in self.scorers:
            self._guard(scorer, "post_score", pose, row, timings, failures)

        return row

    def _guard(self, scorer, hook, pose, row, timings, failures):
        """Run one scorer hook, containing its failures to that scorer's columns."""
        t0 = time.perf_counter()
        try:
            getattr(scorer, hook)(pose, row)
        except Exception as exc:
            key = f"{scorer.NAME}_error"
            message = f"{type(exc).__name__}: {exc}"
            if hook != "score":
                message = f"{hook}: {message}"
            # Keep the first failure: a pre_score failure is usually the cause of
            # the score failure that follows it.
            if not row.get(key):
                row[key] = message
                failures[scorer.NAME] += 1
                self.system.log(
                    f"[score:{scorer.NAME}] ligand {pose.ligand_idx} "
                    f"pose {pose.pose} failed: {message}"
                )
        finally:
            timings[scorer.NAME] += time.perf_counter() - t0

    # ----------------------------------------------------------------- output

    def _write(self, ctx, names, meta):
        log = self.system.log

        rows_mod.write_csv(
            os.path.join(self.output, "pose_scores.csv"),
            self.scorers, self.rows, log=log,
        )

        rank_by, higher_is_better = self._rank_by()
        rows_mod.write_run_json(
            os.path.join(self.output, "score_run.json"),
            {
                "case_id": ctx.case_id,
                "scorers": list(names),
                "n_poses": len(self.rows),
                "rank_by": rank_by,
                "options": {
                    key: value
                    for key, value in sorted(vars(self.system.options).items())
                    if key.startswith("score_")
                },
                "scorer_meta": meta,
            },
            log=log,
        )

        # Per-atom Q-scores and the nested density payloads only exist here.
        wants_json = (self._opt("score_json", False)
                      or self._opt("score_qscore_per_atom", False))
        if wants_json:
            rows_mod.write_json(
                os.path.join(self.output, "pose_scores.json"), self.rows, log=log
            )

        if self._opt("score_sdf", False):
            if rank_by is None:
                log("[score] --score-sdf: no rankable column; writing in input order.")
            rows_mod.write_sdfs(
                self.output, self.system, self.rows, rank_by, higher_is_better, log=log
            )
