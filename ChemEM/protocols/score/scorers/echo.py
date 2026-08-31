#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""ECHO scoring of ligand poses (``--score --score-with echo``).

Reports, for every pose, the weighted ECHO total plus every individual term, both
weighted and unweighted. The total is the same quantity docking optimises: the site
precompute, the engine and the weights all come from a ``Docking`` instance, so
``--echo-weights``, ``--dock-full-map``, ``--manual-site`` and the v2 engine all
behave identically here.

Two caveats on reproducing a dock run's printed score exactly:
  * ``run_aco_docking`` ranks its returned poses at ``repCap_final_nm``
    (``--repulsion-cap-polish``), not the ``run_echo_score`` pybind default of 5.0 --
    see ``resolve_rep_max``.
  * protein preparation is deterministic and cached, but a cold prep on a different
    machine still yields slightly different coordinates, and an ECHO score moves with
    them. That is upstream of this scorer and hits ``--dock`` ranking equally.

Term bookkeeping (see ScoringFunctions.cpp)::

    total = -(apply_weights(terms) + map_score) + bias + constraint + covalent

``apply_weights`` is a plain linear sum over LINEAR_TERMS. ``run_echo_terms`` returns
the raw channels, ``run_echo_score`` the weighted total, so the weighted per-term
breakdown and the map contribution are both recoverable in Python without touching
the C++.

Note that ``aromatic``/``nonbond`` are lumped duplicates of their split channels
(``aromatic == aromatic_attr + aromatic_clash``,
``nonbond == nonbond_attr + nonbond_rep + clash``). They are reported raw-only;
including them in the weighted sum would double-count.
"""

from __future__ import annotations

import csv
import os
import time

from ChemEM.tools.precomputed_data import PreCompDataLigand, PreCompDataProtein

from ..hydrogen_torsions import HydrogenTorsionRelaxer, donor_h_torsions
from ..protein_hydrogen_torsions import (
    ProteinHydrogenRelaxer,
    donors_near_ligand,
    min_ligand_distance,
    rotatable_protein_donors,
)
from .base import PoseScorer

# The strict H-bond cutoff the scorer itself uses (ScoringFunctions.cpp:244). A donor
# this close to the ligand is one whose orientation actually decides whether a
# hydrogen bond is scored, so counting them is the prevalence measure: if a benchmark
# has none, protein-H orientation cannot be what is wrong with it.
HBOND_CLOSE_A = 3.5

# The 17 channels apply_weights() actually sums, in its own order.
LINEAR_TERMS = [
    "hphobe_raw_hpil",
    "hphobe_raw_hpho",
    "hphob_enc_gt_7_only_hpil_unsat",
    "hphob_enc_gt_7_only_hpho",
    "desolvation_penalty_scaled",
    "electro_repulsive_clamp",
    "electro_attractive",
    "saltbridge_raw",
    "unsat_polar",
    "ligand_torsion",
    "ligand_intra",
    "nonbond_attr",
    "nonbond_rep",
    "clash",
    "hbond_raw",
    "aromatic_attr",
    "aromatic_clash",
]

# Reported raw but never weighted: the two lumped duplicates, and the three channels
# score() adds outside the linear sum.
LUMPED_TERMS = ["aromatic", "nonbond"]
OFFSET_TERMS = ["bias", "constraint", "covalent"]

RAW_TERMS = LINEAR_TERMS + LUMPED_TERMS + OFFSET_TERMS


def resolve_rep_max(opts):
    """The repulsion cap that reproduces dock's reported score.

    ``run_aco_docking`` ranks and returns its poses from a final Nelder-Mead polish
    run at ``config.repCap_final_nm`` (SearchFunctions.cpp), which is
    ``--repulsion-cap-polish``, *not* the rep_max=5.0 default baked into the
    ``run_echo_score`` pybind signature. Scoring at 5.0 makes every pose look several
    score units better than docking said it was.
    """
    override = getattr(opts, "score_echo_rep_max", None)
    if override is not None:
        return float(override)
    return float(getattr(opts, "repulsion_cap_polish", 15.0))


def breakdown(terms, echo_total, weights):
    """Weighted per-term contributions plus the derived map score.

    Returns ``(raw, weighted, echo_linear, map_score)``. ``map_score`` is recovered by
    inverting the C++ total, since ``run_echo_score`` does not report it separately.
    """
    raw = {name: float(terms.get(name, 0.0)) for name in RAW_TERMS}
    weighted = {
        name: raw[name] * float(getattr(weights, name, 0.0))
        for name in LINEAR_TERMS
    }
    echo_linear = float(sum(weighted.values()))
    offsets = sum(raw[name] for name in OFFSET_TERMS)
    # total = -(echo_linear + map_score) + offsets  ->  solve for map_score
    map_score = -(echo_total - offsets) - echo_linear
    return raw, weighted, echo_linear, map_score


_H_COLUMNS = (
    "echo_total_prehmin", "n_h_torsions", "h_delta_deg_max",
    "h_nm_evals", "h_relax_seconds",
)
_PROT_H_COLUMNS = (
    "echo_total_prot_h_pre", "n_prot_h_donors", "n_prot_h_donors_close",
    "prot_h_delta_deg_max", "prot_h_evals", "prot_h_relax_seconds",
)


class EchoScorer(PoseScorer):
    """The docking score, evaluated on a pose that is already placed."""

    NAME = "echo"
    HELP = "ECHO: the weighted total docking optimises, plus every term"
    # The ECHO precompute needs a binding site, and the map term needs the segmented
    # site maps -- without them the reported total would not be the number --dock
    # optimised, which is the whole point.
    DEPS = ("binding_site", "alpha_mask", "confidence_map")
    HEADLINE = "echo_total"
    HIGHER_IS_BETTER = False
    NEEDS_SITE = True

    COLUMNS = (
        ("echo_total", "echo_linear", "map_score")
        + tuple(OFFSET_TERMS)
        + tuple(f"raw_{name}" for name in RAW_TERMS)
        + tuple(f"w_{name}" for name in LINEAR_TERMS)
    )

    def __init__(self, system, opts):
        super().__init__(system, opts)
        self._dock = None
        self._ext = None
        self._weights = None
        self._rep_max = None
        self._site_precomp = {}
        self._combined_by_site = {}
        self._donors_by_site = {}
        self._precomp_lig = None
        self._h_torsions = []
        self._donor_rows = []
        self._restore_protein = None

    def extra_columns(self):
        cols = ()
        if self._opt("score_echo_minimise_hydrogens", False):
            cols += _H_COLUMNS
        if self._opt("score_echo_protein_h", False):
            cols += _PROT_H_COLUMNS
        return cols

    # ------------------------------------------------------------------ setup

    def setup_run(self, ctx) -> None:
        self._dock = self._get_docking()
        self._ext = self._dock._echo_ext()
        self._weights = self._dock._weights()
        self._rep_max = resolve_rep_max(self.opts)

        self.system.log(
            f"[score:echo] engine={self._dock.ENGINE_NAME} rep_max={self._rep_max}"
        )
        self._warn_unreproducible_map_term()

        if self._opt("score_echo_minimise_hydrogens", False):
            self.system.log(
                "[score:echo] relaxing polar-hydrogen torsions against ECHO before "
                "scoring; this rewrites the ligand conformers in place, so every "
                "other selected scorer -- and any protocol running after --score -- "
                "sees the relaxed poses."
            )
        if self._opt("score_echo_protein_h", False):
            self.system.log(
                "[score:echo] relaxing rotatable protein donor-H orientations "
                f"(source={self._opt('protein_hydrogens', 'rdkit')}). The protein is "
                "restored between poses, so each pose gets its own best receptor "
                "hydrogen arrangement and none inherits the previous one's. "
                "Rotation is unpenalised, so echo_total is an upper bound: read it "
                "against echo_total_prot_h_pre."
            )

    def _get_docking(self):
        """A ``Docking`` instance used purely as an accessor.

        ``Docking.__init__`` does no work, so this is a cheap way to inherit the
        ``--echo-weights`` loading and the engine selection instead of duplicating
        two subtle pieces of logic that would then be free to drift away from what
        ``--dock`` actually does.
        """
        if self._opt("score_echo_engine", "docking") == "docking_v2":
            from ChemEM.protocols._docking.docking_v2 import DockingV2
            return DockingV2(self.system)
        from ChemEM.protocols._docking.docking import Docking
        return Docking(self.system)

    def _warn_unreproducible_map_term(self):
        """Say so when the map term here cannot match the one dock ranked with.

        ``run_echo_score``'s pybind signature has no ``use_map_score`` argument, so it
        always scores with mutual information. That matches ``--outer-map-score 0``
        (the default) but not ``--outer-map-score 1`` (SCI).
        """
        if self._opt("no_map", False) or getattr(self.system, "density_map", None) is None:
            return
        if int(self._opt("outer_map_score", 0) or 0) != 0:
            self.system.log(
                "[score:echo] WARNING: --outer-map-score 1 (SCI) was requested, but "
                "run_echo_score always scores the map term with mutual information, "
                "so echo_total will not reproduce the score dock ranked these poses "
                "by. Every non-map term is unaffected. Note the density scorer's "
                "density_sci column is a different quantity against a different map."
            )

    def setup_ligand(self, lig) -> None:
        # Both of these are per-ligand, not per-pose: the precompute is a fixed
        # parameterisation and the torsion set is topology-driven.
        self._precomp_lig = PreCompDataLigand(
            lig.ligand,
            self.system.platform,
            flexible_rings=False,
            resource_owner=self.system,
        )
        # `combined` deep-copies the site precompute, so these are built once per
        # (site, ligand) pair and reused across that ligand's poses. Rebuilding them
        # per pose would turn an O(ligands x sites) cost into O(poses).
        self._combined_by_site = {}
        self._donors_by_site = {}

        relax_h = self._opt("score_echo_minimise_hydrogens", False)
        self._h_torsions = donor_h_torsions(lig.ligand.mol) if relax_h else []
        if relax_h:
            self.system.log(
                f"[score:echo] ligand {lig.ligand_idx} ({lig.ligand.identifier}): "
                f"{len(self._h_torsions)} donor-H torsion(s)"
            )

    def teardown_ligand(self, lig) -> None:
        self._precomp_lig = None
        self._combined_by_site = {}
        self._donors_by_site = {}
        self._h_torsions = []

    # ---------------------------------------------------------------- scoring

    def _precomp_for_site(self, site_id, binding_site):
        """Cached ``PreCompDataProtein``, built exactly as Docking.run() builds it."""
        if site_id in self._site_precomp:
            return self._site_precomp[site_id]

        if getattr(binding_site, "is_alpha_feature_site", False):
            bias_radius = float(self._opt("feature_site_radius"))
        else:
            bias_radius = float(self._opt("bias_radius"))

        precomp = PreCompDataProtein(
            binding_site,
            self.system,
            bias_radius=bias_radius,
            split_site=self._opt("split_site", False),
        )
        self._site_precomp[site_id] = precomp
        return precomp

    def _combined_for(self, pose):
        site_id, binding_site = pose.site()
        if site_id is None:
            raise ValueError("no binding site for this pose")
        if site_id not in self._combined_by_site:
            site_precomp = self._precomp_for_site(site_id, binding_site)
            combined = site_precomp + self._precomp_lig
            self._combined_by_site[site_id] = combined
            # Donors are derived from `combined`, not from the site precompute, so
            # the reference hydrogen coordinates they capture are the ones the
            # relaxer will write back into on restore.
            if self._opt("score_echo_protein_h", False):
                self._donors_by_site[site_id] = rotatable_protein_donors(
                    binding_site.rdkit_lining_mol,
                    combined.protein_hydrogens,
                    combined.protein_positions,
                )
                self.system.log(
                    f"[score:echo] site {site_id}: "
                    f"{len(self._donors_by_site[site_id])} rotatable protein donor(s)"
                )
            else:
                self._donors_by_site[site_id] = []
        return site_id, self._combined_by_site[site_id]

    def _score_kwargs(self):
        return {
            "interaction_cutoff": float(self._opt("score_echo_interaction_cutoff", 6.0)),
            "rep_max": self._rep_max,
            "electro_clamp": float(self._opt("score_echo_electro_clamp", 2.0)),
        }

    def _echo_total(self, combined, mol, conf_id=0):
        from rdkit import Chem
        block = Chem.MolToMolBlock(mol, includeStereo=True, confId=conf_id)
        return float(self._ext.run_echo_score(
            combined, block, confId=0, weights=self._weights, **self._score_kwargs()
        ))

    def _echo_terms(self, combined, mol, conf_id=0):
        from rdkit import Chem
        block = Chem.MolToMolBlock(mol, includeStereo=True, confId=conf_id)
        return self._ext.run_echo_terms(
            combined, block, confId=0, **self._score_kwargs()
        )

    def pre_score(self, pose, row) -> None:
        """Relax hydrogens before ANY scorer reads this pose.

        Ligand donor-H torsions are rewritten in the conformer; rotatable receptor
        donor-H's are rotated in the precompute. Both are undone (the receptor) or
        left in place (the ligand) by design -- see ``post_score``.
        """
        relax_h = bool(self._h_torsions)
        relax_prot = bool(self._opt("score_echo_protein_h", False))
        if not (relax_h or relax_prot):
            if self._opt("score_echo_minimise_hydrogens", False):
                row.update({
                    "echo_total_prehmin": "", "n_h_torsions": 0,
                    "h_nm_evals": 0, "h_delta_deg_max": 0.0, "h_relax_seconds": 0.0,
                })
            return

        site_id, combined = self._combined_for(pose)
        row["site_id"] = str(site_id)

        if relax_h:
            self._relax_ligand_hydrogens(pose, row, combined)
        if relax_prot:
            self._restore_protein = self._relax_protein_hydrogens(
                pose, row, combined, self._donors_by_site.get(site_id, [])
            )

    def _relax_ligand_hydrogens(self, pose, row, combined):
        row["echo_total_prehmin"] = self._echo_total(combined, pose.mol, pose.conf_id)

        t0 = time.perf_counter()
        relaxer = HydrogenTorsionRelaxer(
            self._h_torsions,
            lambda mol: self._echo_total(combined, mol, 0),
            grid_deg=float(self._opt("score_echo_h_min_grid", 60.0)),
            passes=int(self._opt("score_echo_h_min_passes", 2)),
            maxiter=int(self._opt("score_echo_h_min_maxiter", 100)),
        )
        _score, n_evals, max_delta = relaxer.relax(pose.mol, pose.conf_id)

        # Keep the ParmEd view in step with the conformer we just rotated.
        pose.ligand.set_positions(
            pose.mol.GetConformer(pose.conf_id).GetPositions(), conf_id=pose.conf_id
        )
        pose.touch()

        row["n_h_torsions"] = len(self._h_torsions)
        row["h_nm_evals"] = n_evals
        row["h_delta_deg_max"] = round(max_delta, 3)
        row["h_relax_seconds"] = round(time.perf_counter() - t0, 3)

    def _relax_protein_hydrogens(self, pose, row, combined, donors):
        """Rotate the site's rotatable donor hydrogens to suit this pose.

        Returns a callable that puts the protein back, or None if nothing moved. The
        caller invokes it in ``post_score`` -- after *every* scorer, not just this
        one, so MM-GBSA does not see a half-restored receptor.
        """
        # Heavy atoms only: the H-bond loop iterates the ligand's heavy atoms
        # (ScoringFunctions.cpp:246), so counting hydrogens here would report donors
        # as "in range" that the scorer never pairs with anything.
        coords = pose.coords
        heavy = [a.GetIdx() for a in pose.mol.GetAtoms() if a.GetAtomicNum() != 1]
        coords = coords[heavy]

        near = donors_near_ligand(donors, coords, cutoff=6.0)

        row["n_prot_h_donors"] = len(near)
        row["n_prot_h_donors_close"] = sum(
            1 for d in near if min_ligand_distance(d, coords) <= HBOND_CLOSE_A
        )
        row["prot_h_evals"] = 0
        row["prot_h_delta_deg_max"] = 0.0
        row["prot_h_relax_seconds"] = 0.0

        if not near:
            row["echo_total_prot_h_pre"] = ""
            return None

        t0 = time.perf_counter()
        relaxer = ProteinHydrogenRelaxer(
            near,
            lambda idx, xyz: combined.protein_hydrogens.__setitem__(idx, xyz),
            lambda: self._echo_total(combined, pose.mol, pose.conf_id),
            grid_deg=float(self._opt("score_echo_protein_h_grid", 30.0)),
            passes=int(self._opt("score_echo_protein_h_passes", 2)),
            maxiter=int(self._opt("score_echo_protein_h_maxiter", 100)),
        )
        score, start_score, n_evals, max_delta, per_donor, best = relaxer.relax()

        row["prot_h_evals"] = n_evals
        row["prot_h_delta_deg_max"] = round(max_delta, 3)
        row["prot_h_relax_seconds"] = round(time.perf_counter() - t0, 3)
        row["echo_total_prot_h_pre"] = "" if start_score is None else start_score

        for entry in per_donor:
            self._donor_rows.append(dict(
                entry,
                ligand=row.get("ligand", ""),
                ligand_idx=pose.ligand_idx,
                conf_id=pose.conf_id,
                pose=pose.pose,
                site_id=row.get("site_id", ""),
            ))

        if best is None or score is None:
            return None

        relaxer.apply(best)
        return relaxer.restore

    def score(self, pose, row) -> None:
        site_id, combined = self._combined_for(pose)
        row["site_id"] = str(site_id)

        echo_total = self._echo_total(combined, pose.mol, pose.conf_id)
        terms = self._echo_terms(combined, pose.mol, pose.conf_id)
        raw, weighted, echo_linear, map_score = breakdown(terms, echo_total, self._weights)

        row["echo_total"] = echo_total
        row["echo_linear"] = echo_linear
        row["map_score"] = map_score
        for name in OFFSET_TERMS:
            row[name] = raw[name]
        for name in RAW_TERMS:
            row[f"raw_{name}"] = raw[name]
        for name in LINEAR_TERMS:
            row[f"w_{name}"] = weighted[name]

        if row.get("n_h_torsions") == 0 and "echo_total_prehmin" in row:
            row["echo_total_prehmin"] = echo_total

    def post_score(self, pose, row) -> None:
        # Every pose is scored against the same reference receptor, so the winning
        # orientations must not leak into the next pose.
        if self._restore_protein is not None:
            self._restore_protein()
            self._restore_protein = None

    # ----------------------------------------------------------------- output

    def finish_run(self, ctx, rows) -> dict:
        meta = {
            "engine": self._dock.ENGINE_NAME if self._dock else None,
            "rep_max": self._rep_max,
            "weights": {
                name: float(getattr(self._weights, name, 0.0))
                for name in LINEAR_TERMS + LUMPED_TERMS
            } if self._weights is not None else {},
        }
        if self._donor_rows:
            meta["protein_h_csv"] = self._write_donor_csv(ctx)
        return meta

    def _write_donor_csv(self, ctx):
        """Per-donor attribution, one row per (pose, rotatable donor).

        The pose-level delta says whether relaxation helps; only this says which
        residues are responsible, which is what separates a real mis-oriented
        hydroxyl from the optimiser finding score noise.
        """
        path = os.path.join(ctx.output, "echo_protein_h.csv")
        cols = ["ligand", "ligand_idx", "conf_id", "pose", "site_id",
                "donor", "heavy_idx", "planar", "delta_deg", "delta_echo"]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for entry in self._donor_rows:
                writer.writerow({c: entry.get(c, "") for c in cols})
        self.system.log(
            f"[score:echo] wrote {path} ({len(self._donor_rows)} donor rows)"
        )
        return path
