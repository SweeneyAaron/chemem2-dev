#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Steric clash penalties (``--score --score-with clash``).

Protein-ligand and ligand-self overlap, reusing the smart_refine_2 clash utilities.
Needs the prepared receptor but no binding site and no map, so ``deps`` is ``()``.

Like ``strain``, this is a cheap orthogonal check on a pose that fits density well:
a ligand pushed into a receptor atom to explain a blob is clashing whatever its
Q-score says.
"""

from __future__ import annotations

import numpy as np

from .base import PoseScorer


def protein_clash_atoms(system):
    """``(coords Nx3, elements N)`` of receptor heavy atoms, ligand residues excluded.

    A per-case constant, so it is built once and cached on the system.
    """
    cache = getattr(system, "_score_clash_protein_atoms", None)
    if cache is not None:
        return cache

    result = (None, None)
    struct = getattr(getattr(system, "protein", None), "complex_structure", None)
    if struct is not None:
        lig_resnames = set()
        for lig in getattr(system, "ligand", None) or []:
            ls = getattr(lig, "complex_structure", None)
            for res in getattr(ls, "residues", []) or []:
                name = str(getattr(res, "name", "") or "").upper().strip()
                if name:
                    lig_resnames.add(name)

        coords, elems = [], []
        for atom in getattr(struct, "atoms", []) or []:
            element = (getattr(atom, "element_name", None)
                       or getattr(atom, "element", None) or "")
            z = getattr(atom, "atomic_number", None)
            if (isinstance(z, int) and z <= 1) or str(element).upper() == "H":
                continue
            resn = str(getattr(getattr(atom, "residue", None), "name", "") or "")
            resn = resn.upper().strip()
            if resn.startswith("LIG") or resn in lig_resnames:
                continue
            coords.append([atom.xx, atom.xy, atom.xz])
            elems.append(str(element) or "C")
        if coords:
            result = (np.asarray(coords, dtype=float),
                      np.asarray(elems, dtype=object))

    system._score_clash_protein_atoms = result
    return result


class ClashScorer(PoseScorer):
    NAME = "clash"
    HELP = "Protein-ligand and ligand-self steric clash penalties"
    DEPS = ()
    HEADLINE = "protein_ligand_clash_penalty"
    HIGHER_IS_BETTER = False
    NEEDS_SITE = False

    COLUMNS = (
        "protein_ligand_clash_penalty", "protein_ligand_clash_count",
        "protein_ligand_max_overlap_A",
        "ligand_self_clash_penalty", "ligand_self_clash_count",
    )

    def setup_run(self, ctx) -> None:
        from ChemEM.protocols.smart_refine_2.optimisers import (
            ligand_self_clash, protein_ligand_clash,
        )
        self._protein_ligand_clash = protein_ligand_clash
        self._ligand_self_clash = ligand_self_clash

        coords, _elems = protein_clash_atoms(self.system)
        if coords is None:
            self.system.log(
                "[score:clash] no receptor heavy atoms found; only the ligand "
                "self-clash terms will be reported."
            )

    def score(self, pose, row) -> None:
        lig_coords = np.asarray(pose.coords, dtype=float)
        atoms = list(pose.mol.GetAtoms())
        lig_elems = np.array([a.GetSymbol() for a in atoms], dtype=object)
        heavy = np.array([a.GetAtomicNum() > 1 for a in atoms])
        if heavy.any():
            hc, he = lig_coords[heavy], lig_elems[heavy]
        else:
            hc, he = lig_coords, lig_elems

        prot_coords, prot_elems = protein_clash_atoms(self.system)
        if prot_coords is not None:
            pc = self._protein_ligand_clash(hc, he, prot_coords, prot_elems)
            row["protein_ligand_clash_penalty"] = float(pc.penalty)
            row["protein_ligand_clash_count"] = int(pc.count)
            row["protein_ligand_max_overlap_A"] = float(pc.max_overlap_A)

        sc = self._ligand_self_clash(lig_coords, lig_elems, mol=pose.mol)
        row["ligand_self_clash_penalty"] = float(sc.penalty)
        row["ligand_self_clash_count"] = int(sc.count)
