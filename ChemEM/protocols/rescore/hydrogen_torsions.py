#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Polar-hydrogen relaxation for pose re-scoring.

Ligand hydrogens reach the ECHO score through three channels only -- measured,
by relaxing a pose and diffing every term: `hbond_raw` (donor geometry),
`ligand_torsion` (the donor-H torsions carry an OpenFF torsion energy) and
`nonbond_rep` (H-protein steric contact). Every heavy-atom-driven channel --
`clash`, `nonbond_attr`, `aromatic*`, `electro*`, the hydrophobic channels,
`saltbridge_raw`, `unsat_polar`, `ligand_intra`, `desolvation_penalty_scaled` --
comes back unchanged to the last bit. So relaxing hydrogens cannot flatter a
pose's shape, sterics or electrostatics; it can only fix its H placement.
(Heavy-atom *coordinates* can shift by ~1 ULP, since RDKit rebuilds them through
a rigid transform; that is float noise, not motion.)

The H degrees of freedom worth relaxing are therefore the *polar donor-H
torsions* -- hydroxyl, thiol and (mono-)amine rotations.

Those are exactly the torsions the ACO docking search already samples:
``get_torsion_lists`` folds ``get_donor_h_torsions`` into the ligand torsion set
that ``PreCompDataLigand`` hands to the engine. Relaxing them here therefore puts
an externally-supplied pose on the same footing as a docked one, rather than
scoring it with whatever H placement its SDF happened to carry.

Because every torsion is filtered through ``only_h_moves_on_rotation``, "ligand
hydrogens only" is guaranteed by topology -- no heavy atom *can* move. There is
no restraint, no OpenMM system and no drift to account for.

Known limits, by construction:
  * Only rotatable *donor* H's move. Aromatic C-H's, and any H whose rotation
    would drag a second atom along, stay put. A -NH2 is excluded for that reason
    (rotating about the pivot bond moves both H's), matching what docking does.
  * Bond lengths and X-H angles are never relaxed.
  * The optimiser minimises the ECHO score with the ECHO score, so the caller
    must report the pre-relaxation value alongside the post one.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from ChemEM.tools.precomputed_data import (
    get_donor_h_torsions,
    only_h_moves_on_rotation,
)

# Backstop so a pathological ligand cannot spin forever inside one pose.
#
# Kept deliberately low because evaluations are expensive: `run_echo_score`
# rebuilds the whole PreComputedData object from Python on every call, so one
# evaluation costs ~200 ms even though the scoring itself is ~0.05 ms. Every
# design choice below is about spending as few calls as possible.
MAX_EVALS = 600


def donor_h_torsions(mol):
    """Donor-H torsion quads for `mol`, one per movable hydrogen.

    Returns ``(A, B, C, D)`` tuples where ``B-C`` is the rotatable bond, ``C`` is
    an N/O/S donor and ``D`` is the hydrogen that -- and only that -- moves when
    the dihedral is set. Mirrors the filtering ``get_torsion_lists`` applies at
    ChemEM/tools/precomputed_data.py so the set matches what docking samples.

    Topology-driven, so compute this once per ligand rather than once per pose.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return []

    torsions = []
    moved_hs = set()
    for t in get_donor_h_torsions(mol):
        if t[3] in moved_hs:
            continue
        try:
            if not only_h_moves_on_rotation(mol, t):
                continue
        except Exception:
            # A degenerate dihedral (collinear atoms) makes RDKit throw; such a
            # torsion is not usable, so drop it rather than lose the ligand.
            continue
        torsions.append(tuple(int(i) for i in t))
        moved_hs.add(t[3])

    return torsions


def _single_conformer_copy(mol, conf_id):
    """A copy of `mol` carrying only conformer `conf_id`, renumbered to 0."""
    work = Chem.Mol(mol)
    conf = Chem.Conformer(mol.GetConformer(conf_id))
    work.RemoveAllConformers()
    work.AddConformer(conf, assignId=True)
    return work


class HydrogenTorsionRelaxer:
    """Minimise a score over a ligand's donor-H torsions.

    `score_fn` takes an RDKit mol holding a single conformer (id 0) and returns
    the value to minimise -- for ECHO that is ``run_echo_score``, where lower is
    better.
    """

    def __init__(self, torsions, score_fn, *, grid_deg=60.0, passes=2,
                 maxiter=100):
        self.torsions = list(torsions)
        self.score_fn = score_fn
        self.grid_deg = float(grid_deg)
        self.passes = int(passes)
        self.maxiter = int(maxiter)

    def relax(self, mol, conf_id):
        """Relax conformer `conf_id` of `mol` in place.

        Returns ``(score, n_evals, max_delta_deg)``. ``score`` is None when there
        is nothing to do (no torsions), which the caller reports as a clean skip.
        Heavy-atom coordinates are unchanged on every path, including failure.
        """
        if not self.torsions:
            return None, 0, 0.0

        work = _single_conformer_copy(mol, conf_id)
        conf = work.GetConformer(0)
        start = np.array(
            [rdMolTransforms.GetDihedralDeg(conf, *t) for t in self.torsions],
            dtype=float,
        )

        state = {"evals": 0}

        def objective(angles):
            if state["evals"] >= MAX_EVALS:
                return float("inf")
            state["evals"] += 1
            for t, a in zip(self.torsions, angles):
                rdMolTransforms.SetDihedralDeg(conf, *t, float(a))
            try:
                return float(self.score_fn(work))
            except Exception:
                return float("inf")

        best_angles, best_score, start_score = self._seed(objective, start)
        best_angles, best_score = self._polish(objective, best_angles, best_score)

        if not np.isfinite(best_score) or best_score >= start_score:
            # Nothing beat the input placement (or every trial failed): leave the
            # pose exactly as it came in.
            best_angles, best_score = start, start_score

        if not np.isfinite(best_score):
            return None, state["evals"], 0.0

        # Apply the winning angles to the caller's real conformer.
        real_conf = mol.GetConformer(conf_id)
        for t, a in zip(self.torsions, best_angles):
            rdMolTransforms.SetDihedralDeg(real_conf, *t, float(a))

        max_delta = float(np.max(np.abs(_wrap180(best_angles - start)))) if len(start) else 0.0
        return best_score, state["evals"], max_delta

    def _seed(self, objective, start):
        """Coarse search for a basin, since Nelder-Mead is local and a hydroxyl
        rotation has several minima.

        Scans one torsion at a time, holding the others at the current best,
        rather than over the full product grid. Cost is
        ``passes * n_torsions * (360/grid)`` instead of ``(360/grid) ** n``,
        which matters when an evaluation costs ~200 ms: two torsions at 60 deg
        is 24 calls rather than 36, and four torsions is 48 rather than 1296.
        A second pass picks up coupling between torsions that share a pivot.

        Returns ``(best_angles, best_score, start_score)``; the incoming
        placement is always evaluated first, so the caller can tell whether the
        search improved on it without paying for another evaluation.
        """
        best_angles = np.array(start, dtype=float)
        start_score = objective(best_angles)
        best_score = start_score

        if self.grid_deg <= 0:
            return best_angles, best_score, start_score

        n_steps = max(1, int(round(360.0 / self.grid_deg)))
        grid = [i * (360.0 / n_steps) - 180.0 for i in range(n_steps)]

        for _ in range(max(1, self.passes)):
            improved = False
            for i in range(len(self.torsions)):
                trial = best_angles.copy()
                for angle in grid:
                    trial[i] = angle
                    score = objective(trial)
                    if score < best_score:
                        best_score = score
                        best_angles = trial.copy()
                        improved = True
            if not improved:
                break

        return best_angles, best_score, start_score

    def _polish(self, objective, angles, score):
        try:
            from scipy.optimize import minimize
        except ImportError:
            return angles, score

        try:
            result = minimize(
                objective,
                np.asarray(angles, dtype=float),
                method="Nelder-Mead",
                options={"maxiter": self.maxiter, "xatol": 1.0, "fatol": 1e-4},
            )
        except Exception:
            return angles, score

        if np.isfinite(result.fun) and result.fun < score:
            return np.asarray(result.x, dtype=float), float(result.fun)
        return angles, score


def _wrap180(angles):
    """Map degree differences onto (-180, 180] so 359 deg reads as -1 deg."""
    return (np.asarray(angles, dtype=float) + 180.0) % 360.0 - 180.0
