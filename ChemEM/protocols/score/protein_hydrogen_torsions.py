#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Protein donor-hydrogen relaxation for pose re-scoring.

The counterpart to `hydrogen_torsions`, on the other side of the interface. That
module rotates the *ligand's* polar hydrogens; this one rotates the *protein's*.

Why the protein's are worth rotating at all
-------------------------------------------
ECHO scores hydrogen bonds from real donor-H geometry, not a heavy-atom
heuristic: ``compute_nonbond_hbond_salt`` (ScoringFunctions.cpp:312-334) reads
``prot.hydrogen_coords[pi]``, takes the D-H...A angle through
``score_max_angle``, gates on ``max_ang > 110`` (:344) and feeds the angle into
the Buckingham polynomials (:352-354). Those protein hydrogen coordinates are
frozen for the whole docking run.

By default they are also *not the prepared ones*. The site's RDKit mol is
heavy-atom only -- ``write_residues_to_pdb`` ends in ``RemoveHs`` -- so
``get_protein_hydrogen_reference`` rebuilds them with
``Chem.AddHs(addCoords=True)``. For donors whose hydrogen is fixed by topology
that is harmless. For the freely rotatable ones it is not: RDKit picks an
arbitrary torsion, and a genuine hydrogen bond whose H points the wrong way
fails the 110 degree gate, falls through to the plain Buckingham term, and at
~2.7 A that term is repulsive. The pose is charged for a clash where it should
have been credited with a hydrogen bond.

What counts as rotatable
------------------------
The classification is the one already written and justified for the geodock
pharmacophore sites (``geodock/geodock/sites.py:_rotatable_donor_axis``): exactly
one heavy neighbour and at least one hydrogen, and either O/S (Ser/Thr/Tyr OH,
Cys SH) or N carrying three hydrogens (Lys NH3+). Arg NH1/NH2 and Asn/Gln
ND2/NE2 also have a single heavy neighbour but are conjugated and planar, so
their hydrogens are fixed by chemistry rather than by the protonation guess;
backbone amide N has two heavy neighbours and falls out of the same test.

Tyr OH is rotatable but not *freely* rotatable -- the C-O bond is conjugated with
the ring, so the hydrogen wants to lie in the ring plane. Those donors get their
two in-plane orientations rather than a full sweep. Sweeping them freely would
manufacture hydrogen bonds the chemistry forbids and inflate the measured gain,
which is exactly the wrong error to make in a GO/NO-GO measurement.

Known limits, by construction
-----------------------------
  * Only the hydrogen moves. No heavy atom can, so this is not sidechain
    flexibility and it cannot invalidate the precomputed grids.
  * Rotation is free: there is no intra-protein term, so a hydroxyl that rotates
    away from a protein partner to reach the ligand is not charged for what it
    gave up. The relaxed score is therefore an *upper bound* on the available
    signal, and the pre-relaxation value must be reported next to it.
  * Waters are excluded (no heavy neighbour), as are the amide and guanidinium
    donors above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# Same backstop rationale as the ligand relaxer: one evaluation costs ~200 ms
# because run_echo_score rebuilds the whole PreComputedData from Python, so the
# budget is counted in evaluations, not seconds.
MAX_EVALS = 600


@dataclass
class ProteinDonor:
    """One rotatable protein donor, indexed into the site's heavy-atom arrays.

    ``heavy_idx`` indexes ``protein_positions`` / ``protein_hydrogens`` / the
    lining mol interchangeably -- they share an atom order by construction.
    """

    heavy_idx: int
    res_name: str
    atom_name: str
    res_id: str
    origin: np.ndarray                 # donor heavy-atom position
    axis: np.ndarray                   # unit vector the hydrogens rotate about
    h_ref: np.ndarray                  # (n_H, 3), the placement as it came in
    allowed: Optional[np.ndarray] = None   # discrete angles (deg), or None for free

    def label(self) -> str:
        return f"{self.res_name}{self.res_id}:{self.atom_name}"

    def rotated(self, angle_deg: float) -> List[np.ndarray]:
        """This donor's hydrogens turned `angle_deg` about its bond axis."""
        angle_deg = self.snap(angle_deg)
        return list(_rotate_about(self.h_ref - self.origin,
                                  self.axis,
                                  np.radians(angle_deg)) + self.origin)

    def snap(self, angle_deg: float) -> float:
        """Nearest chemically allowed angle.

        Identity for a freely rotatable donor. For a conjugated one (Tyr) this
        collapses the coordinate onto its two in-plane values, which is also what
        keeps a continuous optimiser from wandering out of the ring plane -- it
        simply sees a step function in that coordinate and leaves it alone.
        """
        if self.allowed is None:
            return float(angle_deg)
        delta = np.abs(_wrap180(self.allowed - float(angle_deg)))
        return float(self.allowed[int(np.argmin(delta))])


def rotatable_protein_donors(lining_mol, protein_hydrogens, positions=None):
    """Rotatable donors of the site, in lining-mol atom order.

    `protein_hydrogens` is the per-heavy-atom hydrogen coordinate list the C++
    reads (`PreComputedData.cpp:148`); it supplies the hydrogen count, because
    `lining_mol` is heavy-atom only and so has no hydrogen neighbours to count.
    """
    if lining_mol is None or protein_hydrogens is None:
        return []

    if positions is None:
        positions = lining_mol.GetConformer().GetPositions()
    positions = np.asarray(positions, dtype=float)

    donors = []
    for atom in lining_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx >= len(protein_hydrogens):
            continue

        h_coords = protein_hydrogens[idx]
        if not len(h_coords):
            continue

        heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
        if len(heavy_nbrs) != 1:
            # Zero (a water) or two (a backbone amide, a bridging atom): either
            # way not a single-bond rotation.
            continue

        symbol = atom.GetSymbol()
        if symbol in ("O", "S"):
            pass                                   # Ser/Thr/Tyr OH, Cys SH
        elif symbol == "N" and len(h_coords) == 3:
            pass                                   # Lys NH3+
        else:
            # N with 1 or 2 H here is Arg NH1/NH2 or Asn/Gln ND2/NE2: conjugated
            # and planar, so the hydrogens are fixed by chemistry.
            continue

        nbr = heavy_nbrs[0]
        origin = positions[idx]
        axis = _unit(origin - positions[nbr.GetIdx()])
        if not np.any(axis):
            continue

        h_ref = np.asarray([np.asarray(h, dtype=float) for h in h_coords], dtype=float)

        donors.append(ProteinDonor(
            heavy_idx=idx,
            res_name=_prop(atom, "resName"),
            atom_name=_prop(atom, "atomName"),
            res_id=_prop(atom, "resId"),
            origin=origin,
            axis=axis,
            h_ref=h_ref,
            allowed=_planar_angles(positions, atom, nbr, origin, axis, h_ref),
        ))

    return donors


def donors_near_ligand(donors, ligand_coords, cutoff=6.0):
    """Donors with a ligand heavy atom inside `cutoff` of the donor atom.

    Anything further away cannot change the score by rotating: the H-bond branch
    is gated at 6 A (`ScoringFunctions.cpp:312`), so relaxing those donors would
    spend ~200 ms an evaluation to prove the score does not move.
    """
    if not donors or ligand_coords is None or not len(ligand_coords):
        return []

    lig = np.asarray(ligand_coords, dtype=float)
    keep = []
    for donor in donors:
        if np.min(np.linalg.norm(lig - donor.origin, axis=1)) <= cutoff:
            keep.append(donor)
    return keep


def min_ligand_distance(donor, ligand_coords):
    lig = np.asarray(ligand_coords, dtype=float)
    return float(np.min(np.linalg.norm(lig - donor.origin, axis=1)))


class ProteinHydrogenRelaxer:
    """Minimise a score over a site's rotatable protein donor-H torsions.

    `apply_fn(heavy_idx, coords)` writes one donor's hydrogen coordinates into
    whatever the scorer reads; `score_fn()` returns the value to minimise for the
    *fixed* ligand pose. Mirrors `HydrogenTorsionRelaxer` -- coordinate-descent
    coarse scan then a Nelder-Mead polish -- because the cost structure is the
    same and a hydroxyl rotation is multi-minimum either way.
    """

    def __init__(self, donors, apply_fn, score_fn, *, grid_deg=30.0, passes=2,
                 maxiter=100):
        self.donors = list(donors)
        self.apply_fn = apply_fn
        self.score_fn = score_fn
        self.grid_deg = float(grid_deg)
        self.passes = int(passes)
        self.maxiter = int(maxiter)

    def relax(self):
        """Returns ``(score, start_score, n_evals, max_delta_deg, per_donor, best)``.

        ``score`` is None when there is nothing to do. The reference placement is
        always restored on the way out, whether or not the search improved on it:
        the caller reuses one precompute across every pose of a ligand, so a pose
        that left the protein rotated would score the next one against a receptor
        the previous pose chose. A caller that wants to score *with* the winning
        orientations calls ``apply(best)``, scores, then ``restore()``.
        """
        if not self.donors:
            return None, None, 0, 0.0, [], None

        state = {"evals": 0}
        start = np.zeros(len(self.donors), dtype=float)

        def objective(angles):
            if state["evals"] >= MAX_EVALS:
                return float("inf")
            state["evals"] += 1
            self.apply(angles)
            try:
                return float(self.score_fn())
            except Exception:
                return float("inf")

        try:
            best, best_score, start_score = self._seed(objective, start)
            best, best_score = self._polish(objective, best, best_score)

            if not np.isfinite(best_score) or best_score >= start_score:
                best, best_score = start, start_score

            if not np.isfinite(best_score):
                return None, None, state["evals"], 0.0, [], None

            per_donor = self._attribute(objective, best, start_score)
        finally:
            self.restore()

        snapped = np.array([d.snap(a) for d, a in zip(self.donors, best)])
        max_delta = float(np.max(np.abs(_wrap180(snapped)))) if len(snapped) else 0.0
        return best_score, start_score, state["evals"], max_delta, per_donor, best

    # ------------------------------------------------------------- internals

    def apply(self, angles):
        """Write the hydrogens for `angles` (degrees, one per donor)."""
        for donor, angle in zip(self.donors, angles):
            self.apply_fn(donor.heavy_idx, donor.rotated(float(angle)))

    def restore(self):
        """Put every donor back to the placement it came in with."""
        for donor in self.donors:
            self.apply_fn(donor.heavy_idx, [np.array(h) for h in donor.h_ref])


    def _seed(self, objective, start):
        """Coordinate-descent coarse scan; see `HydrogenTorsionRelaxer._seed`.

        Scanning donors one at a time costs ``passes * n_donors * (360/grid)``
        rather than ``(360/grid) ** n_donors``. Unlike ligand torsions sharing a
        pivot, two protein donors couple only through the ligand, so a second
        pass rarely finds anything -- but it is cheap insurance when two donors
        compete for the same ligand acceptor.
        """
        best = np.array(start, dtype=float)
        start_score = objective(best)
        best_score = start_score

        if self.grid_deg <= 0:
            return best, best_score, start_score

        n_steps = max(1, int(round(360.0 / self.grid_deg)))
        full = [i * (360.0 / n_steps) - 180.0 for i in range(n_steps)]

        for _ in range(max(1, self.passes)):
            improved = False
            for i, donor in enumerate(self.donors):
                # A conjugated donor has two orientations, not 360/grid of them;
                # scanning the full grid would evaluate each of them six times.
                grid = full if donor.allowed is None else list(donor.allowed)
                trial = best.copy()
                for angle in grid:
                    trial[i] = angle
                    score = objective(trial)
                    if score < best_score:
                        best_score = score
                        best = trial.copy()
                        improved = True
            if not improved:
                break

        return best, best_score, start_score

    def _polish(self, objective, angles, score):
        if all(d.allowed is not None for d in self.donors):
            # Every coordinate is discrete; there is nothing for a continuous
            # optimiser to do but burn evaluations on snapped duplicates.
            return angles, score

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

    def _attribute(self, objective, best, start_score):
        """Per-donor share of the gain: move one donor, leave the rest at zero.

        Costs one evaluation per donor. Worth it -- the aggregate number says
        whether relaxation helps, and only this says *which* residues are
        responsible, which is what tells a real defect apart from the optimiser
        finding noise.
        """
        rows = []
        for i, donor in enumerate(self.donors):
            solo = np.zeros(len(self.donors), dtype=float)
            solo[i] = best[i]
            score = objective(solo)
            delta = _wrap180(donor.snap(float(best[i])))
            rows.append({
                "donor": donor.label(),
                "heavy_idx": donor.heavy_idx,
                "planar": donor.allowed is not None,
                "delta_deg": round(float(delta), 2),
                "delta_echo": (round(float(score - start_score), 4)
                               if np.isfinite(score) else ""),
            })
        return rows


# ------------------------------------------------------------------ geometry


def _unit(v):
    n = np.linalg.norm(v)
    return np.zeros(3) if n < 1e-9 else np.asarray(v, dtype=float) / n


def _rotate_about(v, axis, theta):
    """Rodrigues rotation of row vectors `v` about unit `axis` by `theta` rad."""
    v = np.atleast_2d(np.asarray(v, dtype=float))
    axis = np.asarray(axis, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    return (v * c
            + np.cross(axis, v) * s
            + np.outer(v @ axis, axis) * (1.0 - c))


def _wrap180(angles):
    """Map degrees onto (-180, 180] so 359 reads as -1."""
    return (np.asarray(angles, dtype=float) + 180.0) % 360.0 - 180.0


def _planar_angles(positions, atom, nbr, origin, axis, h_ref):
    """The two in-plane rotations for a conjugated donor, or None if free.

    A Tyr OH hangs off an aromatic carbon, so the C-O bond is conjugated and the
    hydrogen wants to lie in the ring plane. The two allowed rotations are the
    ones putting the hydrogen's perpendicular component along +/- the in-plane
    reference direction.
    """
    if not nbr.GetIsAromatic() or len(h_ref) != 1:
        return None

    ring_nbrs = [a for a in nbr.GetNeighbors()
                 if a.GetIdx() != atom.GetIdx() and a.GetAtomicNum() != 1]
    if not ring_nbrs:
        return None

    # In-plane reference: the ring bond's component perpendicular to the rotation
    # axis. That component and the axis together span the ring plane.
    ref = _unit(_perp(positions[ring_nbrs[0].GetIdx()] - positions[nbr.GetIdx()], axis))
    if not np.any(ref):
        return None

    h_perp = _perp(h_ref[0] - origin, axis)
    if np.linalg.norm(h_perp) < 1e-9:
        # Hydrogen sits on the axis; rotating it does nothing.
        return None
    h_perp = _unit(h_perp)

    # Signed angle from the incoming hydrogen to the reference, about the axis.
    ortho = np.cross(axis, ref)
    to_ref = np.degrees(np.arctan2(float(h_perp @ ortho), float(h_perp @ ref)))

    # Rotating by -to_ref lands on +ref; the other in-plane option is 180 away.
    return _wrap180(np.array([-to_ref, -to_ref + 180.0]))


def _perp(v, axis):
    v = np.asarray(v, dtype=float)
    return v - float(v @ axis) * np.asarray(axis, dtype=float)


def _prop(atom, name):
    try:
        return atom.GetProp(name).strip()
    except KeyError:
        return ""
