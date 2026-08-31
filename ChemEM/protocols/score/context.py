#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Per-run and per-pose state shared by the scorers."""

from __future__ import annotations

from dataclasses import dataclass, field

from .sites import iter_sites, site_for_pose


@dataclass
class LigandCtx:
    """One ``Ligand`` from the config, handed to ``setup_ligand``."""
    ligand: object
    ligand_idx: int
    source: str
    ctx: "ScoreContext"


@dataclass
class PoseRef:
    """One pose: a single conformer of a single ``Ligand``.

    A multi-record poses SDF loads as one ``Ligand`` per record, so neither
    ``ligand_idx`` nor ``conf_id`` alone reads as a pose number -- ``pose`` is that
    number, counted within the input source, in input order.
    """
    ligand: object
    ligand_idx: int
    conf_id: int
    pose: int
    source: str
    mol: object
    ctx: "ScoreContext"
    _molblock: str | None = field(default=None, repr=False)
    _site: tuple | None = field(default=None, repr=False)

    @property
    def coords(self):
        """Conformer positions in Angstrom, all atoms, in mol atom order.

        Read fresh each time on purpose: ECHO's ``pre_score`` hydrogen relaxation
        rewrites the conformer in place, and every scorer after it must see the
        geometry that was actually scored.
        """
        return self.mol.GetConformer(self.conf_id).GetPositions()

    def molblock(self) -> str:
        """The pose as a MolBlock -- the boundary every C++ scorer takes.

        Memoised per pose, and invalidated by ``touch()`` when a ``pre_score`` hook
        moves an atom.
        """
        if self._molblock is None:
            from rdkit import Chem
            self._molblock = Chem.MolToMolBlock(
                self.mol, includeStereo=True, confId=self.conf_id
            )
        return self._molblock

    def touch(self) -> None:
        """Declare that this pose's coordinates changed; drops cached derivations."""
        self._molblock = None

    def site(self):
        """``(site_id, binding_site)`` for this pose, or ``(None, None)``."""
        if self._site is None:
            self._site = self.ctx.site_for(self)
        return self._site


class ScoreContext:
    """Run-scoped state: the system, the resolved options and the site list."""

    def __init__(self, system, opts, output: str, case_id: str, scorers: list):
        self.system = system
        self.opts = opts
        self.output = output
        self.case_id = case_id
        self.scorers = scorers
        self._sites = None

    @property
    def needs_site(self) -> bool:
        return any(s.needs_site() for s in self.scorers)

    @property
    def sites(self):
        """The binding sites, built on first access and never before.

        This laziness is what makes "``--score-with qscore`` needs no segmentation"
        true at runtime rather than only in ``score_deps``: if no selected scorer
        calls ``pose.site()``, ``system.binding_sites`` is never touched.
        """
        if self._sites is None:
            sites = iter_sites(self.system)
            forced = getattr(self.opts, "score_site", None)
            if forced is not None:
                picked = [(k, s) for k, s in sites if str(k) == str(forced)]
                if not picked:
                    available = ", ".join(str(k) for k, _ in sites) or "none"
                    raise ValueError(
                        f"--score-site {forced} does not match any binding site "
                        f"(have: {available})"
                    )
                sites = picked
            if not sites:
                raise ValueError(
                    "No binding site available to score against. Run with a config "
                    "that yields a binding site (check `centroid =`, or use "
                    "--manual-site)."
                )
            self._sites = sites
        return self._sites

    def site_for(self, pose):
        return site_for_pose(self.sites, pose.coords)
