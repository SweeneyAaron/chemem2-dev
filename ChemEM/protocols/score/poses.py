#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Getting poses out of the config and into ``PoseRef``s.

Poses come from the config exactly as they do for docking input --
``ligand = poses.sdf`` (one conformer per record) or ``ligands_from_dir = <dir>``.
Going through the normal ligand parser rather than reading SDFs directly is what
gives the scorers force-field parameters, protonation and atom types; MM-GBSA and
ECHO both need them.
"""

from __future__ import annotations

import os

from .context import LigandCtx, PoseRef


def iter_ligands(system, ctx):
    """Yield ``(LigandCtx, [PoseRef, ...])`` for every ligand in the config.

    Pose numbers run per input source, not per ligand: a multi-record poses SDF
    loads as one ``Ligand`` per record, so ``ligand_idx`` is a record index and only
    the source grouping gives a meaningful pose number.
    """
    counters: dict[str, int] = {}
    for lig_idx, ligand in enumerate(getattr(system, "ligand", None) or []):
        source = str(getattr(ligand, "input", "") or "")
        lig_ctx = LigandCtx(ligand=ligand, ligand_idx=lig_idx, source=source, ctx=ctx)

        mol = getattr(ligand, "mol", None)
        if mol is None or mol.GetNumConformers() == 0:
            yield lig_ctx, []
            continue

        poses = []
        for conf in mol.GetConformers():
            pose = counters.get(source, 0)
            counters[source] = pose + 1
            poses.append(PoseRef(
                ligand=ligand,
                ligand_idx=lig_idx,
                conf_id=conf.GetId(),
                pose=pose,
                source=source,
                mol=mol,
                ctx=ctx,
            ))
        yield lig_ctx, poses


def source_stem(source, lig_idx):
    """Filesystem-safe stem for an input source, which may be a SMILES string."""
    source = str(source or "")
    stem = os.path.splitext(os.path.basename(source))[0] if os.path.exists(source) else ""
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)[:80]
    return safe or f"Ligand_{lig_idx}"
