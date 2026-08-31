#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""The pose-scorer plugin interface used by the ``--score`` protocol.

A scorer answers one question about one pose. The protocol owns everything else:
loading poses, deciding which binding site a pose belongs to, assembling the row,
containing failures, and writing the output. That split is what lets
``--score-with qscore`` run without paying for the ECHO precompute, and what makes
adding a fifth scorer one file plus one ``SCORER_REGISTRY`` entry.

Every hook except ``score`` has a no-op default, so a scorer that needs no setup is
just a ``NAME``, a ``COLUMNS`` tuple and a ``score`` method.

Lifecycle, per run::

    setup_run(ctx)                      # once
      for each Ligand:
        setup_ligand(lig)
          for each pose of that Ligand:
            pre_score(pose, row)        # may mutate coordinates (ECHO H relaxation)
            score(pose, row)            # every scorer, in --score-with order
            post_score(pose, row)       # undo what pre_score did, after ALL scorers
        teardown_ligand(lig)
    finish_run(ctx, rows)               # extra artefacts

``pre_score``/``post_score`` are separated from ``score`` on purpose. ECHO's optional
hydrogen relaxation rewrites conformer and receptor hydrogen coordinates, and MM-GBSA
is all-atom, so it must see the same geometry ECHO scored -- and the receptor must not
be restored until every scorer has read it.
"""

from __future__ import annotations


class PoseScorer:
    """Base class for a pose scorer. Subclasses override what they need."""

    #: Registry key, and the prefix of this scorer's ``<name>_error`` column.
    NAME: str = ""
    #: One-line description for ``--help``.
    HELP: str = ""
    #: Protocols that must run before this scorer can work.
    DEPS: tuple[str, ...] = ()
    #: Declared output columns, in the order they should appear in the CSV.
    COLUMNS: tuple[str, ...] = ()
    #: The single column that ranks poses, used as the default ``--score-rank-by``.
    HEADLINE: str | None = None
    #: Whether a larger ``HEADLINE`` is a better pose.
    HIGHER_IS_BETTER: bool = False
    #: Set True only if ``score`` calls ``pose.site()``. When no selected scorer
    #: needs a site, the site list is never built and segmentation never runs.
    NEEDS_SITE: bool = False

    def __init__(self, system, opts):
        self.system = system
        self.opts = opts

    # ------------------------------------------------------------------ config

    @classmethod
    def deps_for(cls, opts) -> tuple[str, ...]:
        """Protocol dependencies for this scorer given the parsed options.

        Overridden where the answer depends on a flag -- ``density`` needs the
        segmentation chain only for ``--score-density-region site``. Must tolerate
        an ``opts`` with none of the attributes set: ``generate_custom_usage()``
        builds throwaway parsers and benchmark scripts hand-build namespaces.
        """
        return cls.DEPS

    def needs_site(self) -> bool:
        """Runtime form of ``NEEDS_SITE``; overridable when a flag decides it."""
        return self.NEEDS_SITE

    def extra_columns(self) -> tuple[str, ...]:
        """Columns known only after ``setup_run`` (e.g. behind an optional flag)."""
        return ()

    def _opt(self, name, default=None):
        return getattr(self.opts, name, default)

    # --------------------------------------------------------------- lifecycle

    def setup_run(self, ctx) -> None:
        """Once per run. Raise here for a misconfiguration -- a scorer that cannot
        work at all should say so before any pose is read, not once per pose."""

    def setup_ligand(self, lig) -> None:
        """Once per ``Ligand``. Where per-ligand parameterisation belongs."""

    def pre_score(self, pose, row: dict) -> None:
        """Before any scorer scores this pose. May mutate coordinates."""

    def score(self, pose, row: dict) -> None:
        """Score one pose, writing this scorer's columns into ``row``.

        The protocol catches whatever this raises and records it in
        ``row["<NAME>_error"]``, so one unscorable pose costs neither the other
        poses nor the other scorers.
        """
        raise NotImplementedError

    def post_score(self, pose, row: dict) -> None:
        """After every scorer has read this pose. Undo ``pre_score`` here."""

    def teardown_ligand(self, lig) -> None:
        """Drop per-ligand state. The precompute caches are large."""

    def finish_run(self, ctx, rows: list) -> dict:
        """Once per run, after the CSV rows exist. Write extra artefacts.

        Returns a dict merged into this scorer's section of ``score_run.json``.
        """
        return {}
