#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Ligand internal strain (``--score --score-with strain``).

MMFF94 ``E(pose) - E(relaxed)``, in kcal/mol. No map, no protein, no binding site --
the cheapest scorer here, and orthogonal to all the others: it targets one of the
Q-score blind spots directly, because a ligand contorted to fit fake or wrong density
is strained even when its density fit looks excellent.
"""

from __future__ import annotations

from rdkit import Chem

from .base import PoseScorer


def ligand_mmff_strain(mol, conf_id=0):
    """MMFF94 internal strain of one conformer, or None if it cannot be computed.

    Returns None rather than raising for an unsanitizable pose or a ligand MMFF has
    no parameters for -- both are ordinary, and neither should cost the pose its
    other scores.
    """
    try:
        from rdkit.Chem import AllChem

        m = Chem.Mol(mol, confId=conf_id) if conf_id else Chem.Mol(mol)
        try:
            Chem.SanitizeMol(m)
        except Exception:
            return None
        m = Chem.AddHs(m, addCoords=True)
        props = AllChem.MMFFGetMoleculeProperties(m)
        if props is None:
            return None
        ff = AllChem.MMFFGetMoleculeForceField(m, props)
        if ff is None:
            return None
        e_pose = float(ff.CalcEnergy())

        relaxed = Chem.Mol(m)
        AllChem.MMFFOptimizeMolecule(relaxed, maxIters=1000)
        props_r = AllChem.MMFFGetMoleculeProperties(relaxed)
        ff_r = AllChem.MMFFGetMoleculeForceField(relaxed, props_r) if props_r else None
        if ff_r is None:
            return None
        return e_pose - float(ff_r.CalcEnergy())
    except Exception:
        return None


class StrainScorer(PoseScorer):
    NAME = "strain"
    HELP = "Ligand internal MMFF94 strain, E(pose) - E(relaxed), kcal/mol"
    DEPS = ()
    HEADLINE = "ligand_strain"
    HIGHER_IS_BETTER = False
    NEEDS_SITE = False

    COLUMNS = ("ligand_strain",)

    def score(self, pose, row) -> None:
        # Chem.Mol(mol) keeps every conformer, so select the pose explicitly.
        single = Chem.Mol(pose.mol)
        single.RemoveAllConformers()
        single.AddConformer(pose.mol.GetConformer(pose.conf_id), assignId=True)

        strain = ligand_mmff_strain(single)
        if strain is None:
            row["strain_failed"] = 1
            return
        row["ligand_strain"] = float(strain)
