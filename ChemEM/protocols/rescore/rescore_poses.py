#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Re-score docking solutions with the ECHO scoring function (``--rescore-poses``).

Takes poses from the config -- ``ligand = poses.sdf`` (one conformer per pose) or
``ligands_from_dir = <dir>`` -- and reports, for every pose, the weighted ECHO
total plus every individual term, both weighted and unweighted.

The total is the same quantity docking optimises: the site precompute, the engine
and the weights all come from a ``Docking`` instance, so `--echo-weights`,
`--dock-full-map`, `--manual-site` and the v2 engine all behave identically here.

Two caveats on reproducing a dock run's printed score exactly:
  * `run_aco_docking` ranks its returned poses at `repCap_final_nm`
    (`--repulsion-cap-polish`), not the run_echo_score pybind default of 5.0 --
    see `_resolve_rep_max`.
  * PDBFixer's `addMissingAtoms()` runs an OpenMM minimisation on a
    non-deterministic platform, so every ChemEM *process* builds slightly
    different protein coordinates and an ECHO score moves by 1-3 units between
    runs. That is upstream of this protocol and hits `--dock` ranking equally.
    Within one process this protocol is exactly reproducible.

Term bookkeeping (see ScoringFunctions.cpp)::

    total = -(apply_weights(terms) + map_score) + bias + constraint + covalent

``apply_weights`` is a plain linear sum over LINEAR_TERMS. ``run_echo_terms``
returns the raw channels, ``run_echo_score`` the weighted total, so the weighted
per-term breakdown and the map contribution are both recoverable in Python
without touching the C++.

Note that ``aromatic``/``nonbond`` are lumped duplicates of their split channels
(``aromatic == aromatic_attr + aromatic_clash``,
``nonbond == nonbond_attr + nonbond_rep + clash``). They are reported raw-only;
including them in the weighted sum would double-count.
"""

from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
from rdkit import Chem

from ChemEM.messages import Messages
from ChemEM.tools.precomputed_data import PreCompDataLigand, PreCompDataProtein

from .hydrogen_torsions import HydrogenTorsionRelaxer, donor_h_torsions
from .protein_hydrogen_torsions import (
    ProteinHydrogenRelaxer,
    donors_near_ligand,
    min_ligand_distance,
    rotatable_protein_donors,
)

# The strict H-bond cutoff the scorer itself uses (ScoringFunctions.cpp:244). A
# donor this close to the ligand is one whose orientation actually decides
# whether a hydrogen bond is scored, so counting them is the prevalence measure:
# if a benchmark has none, protein-H orientation cannot be what is wrong with it.
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

# Reported raw but never weighted: the two lumped duplicates, and the three
# channels score() adds outside the linear sum.
LUMPED_TERMS = ["aromatic", "nonbond"]
OFFSET_TERMS = ["bias", "constraint", "covalent"]

RAW_TERMS = LINEAR_TERMS + LUMPED_TERMS + OFFSET_TERMS


class RescorePoses:
    """ECHO re-scoring of ligand poses supplied through the config."""

    def __init__(self, system):
        self.system = system
        self.output = None
        self._dock = None
        self._weights = None
        self._ext = None
        self._site_precomp = {}
        self._sites = []
        self._donor_rows = []

    # ------------------------------------------------------------------ setup

    def _opt(self, name, default=None):
        return getattr(self.system.options, name, default)

    def _get_output(self):
        base = getattr(self.system, "output", None) or "."
        self.system.output = base
        self.output = os.path.join(base, self._opt("rescore_out", "rescore") or "rescore")
        os.makedirs(self.output, exist_ok=True)
        return self.output

    def _get_docking(self):
        """A `Docking` instance used purely as an accessor.

        `Docking.__init__` does no work, so this is a cheap way to inherit the
        map-aware site filter, the `--echo-weights` loading and the engine
        selection instead of duplicating three subtle pieces of logic that would
        then be free to drift away from what `--dock` actually does.
        """
        engine = self._opt("rescore_engine", "docking")
        if engine == "docking_v2":
            from ChemEM.protocols._docking.docking_v2 import DockingV2
            return DockingV2(self.system)
        from ChemEM.protocols._docking.docking import Docking
        return Docking(self.system)

    def _get_sites(self):
        """Sites to score against, mirroring what docking would have used."""
        sites = self._dock._iter_sites()
        if not sites:
            raise ValueError(
                "No binding site available to re-score against. Run with a "
                "config that yields a binding site (check `centroid =`, or use "
                "--manual-site)."
            )

        forced = self._opt("rescore_site", None)
        if forced is not None:
            picked = [(k, s) for k, s in sites if str(k) == str(forced)]
            if not picked:
                available = ", ".join(str(k) for k, _ in sites)
                raise ValueError(
                    f"--rescore-site {forced} does not match any site (have: {available})"
                )
            sites = picked

        return sites

    def _precomp_for_site(self, site_id, binding_site):
        """Cached `PreCompDataProtein`, built exactly as Docking.run() builds it."""
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

    # ------------------------------------------------------------ site choice

    def _site_for_pose(self, coords):
        """Pick the site a pose belongs to.

        Poses arrive with fixed coordinates, so unlike docking we have to work
        out which site they sit in: prefer a site whose bounding box contains the
        pose centroid, else fall back to the nearest site centroid.
        """
        if len(self._sites) == 1:
            return self._sites[0]

        centroid = np.asarray(coords, dtype=float).mean(axis=0)

        for site_id, binding_site in self._sites:
            lo = np.asarray(binding_site.min_coords, dtype=float)
            hi = np.asarray(binding_site.max_coords, dtype=float)
            if np.all(centroid >= lo) and np.all(centroid <= hi):
                return site_id, binding_site

        def distance(entry):
            site_centroid = np.asarray(entry[1].binding_site_centroid, dtype=float)
            return float(np.linalg.norm(centroid - site_centroid))

        return min(self._sites, key=distance)

    # ---------------------------------------------------------------- scoring

    def _resolve_rep_max(self):
        """The repulsion cap that reproduces dock's reported score.

        `run_aco_docking` ranks and returns its poses from a final Nelder-Mead
        polish run at `config.repCap_final_nm` (SearchFunctions.cpp), which is
        `--repulsion-cap-polish`, *not* the rep_max=5.0 default baked into the
        run_echo_score pybind signature. Scoring at 5.0 makes every pose look
        several score units better than docking said it was.
        """
        override = self._opt("rescore_rep_max", None)
        if override is not None:
            return float(override)
        return float(self._opt("repulsion_cap_polish", 15.0))

    def _warn_unreproducible_map_term(self):
        """Say so when the map term here cannot match the one dock ranked with.

        `run_echo_score`'s pybind signature has no `use_map_score` argument, so it
        always scores with mutual information. That matches `--outer-map-score 0`
        (the default) but not `--outer-map-score 1` (SCI).
        """
        if self._opt("no_map", False) or getattr(self.system, "density_map", None) is None:
            return
        if int(self._opt("outer_map_score", 0) or 0) != 0:
            self.system.log(
                "[rescore] WARNING: --outer-map-score 1 (SCI) was requested, but "
                "run_echo_score always scores the map term with mutual "
                "information, so echo_total will not reproduce the score dock "
                "ranked these poses by. Every non-map term is unaffected."
            )

    def _score_kwargs(self):
        return {
            "interaction_cutoff": float(self._opt("rescore_interaction_cutoff", 6.0)),
            "rep_max": self._resolve_rep_max(),
            "electro_clamp": float(self._opt("rescore_electro_clamp", 2.0)),
        }

    def _echo_total(self, combined, mol, conf_id=0):
        block = Chem.MolToMolBlock(mol, includeStereo=True, confId=conf_id)
        return float(self._ext.run_echo_score(
            combined, block, confId=0, weights=self._weights, **self._score_kwargs()
        ))

    def _echo_terms(self, combined, mol, conf_id=0):
        block = Chem.MolToMolBlock(mol, includeStereo=True, confId=conf_id)
        return self._ext.run_echo_terms(
            combined, block, confId=0, **self._score_kwargs()
        )

    def _breakdown(self, terms, echo_total):
        """Weighted per-term contributions plus the derived map score."""
        raw = {name: float(terms.get(name, 0.0)) for name in RAW_TERMS}
        weighted = {
            name: raw[name] * float(getattr(self._weights, name, 0.0))
            for name in LINEAR_TERMS
        }
        echo_linear = float(sum(weighted.values()))
        offsets = sum(raw[name] for name in OFFSET_TERMS)
        # total = -(echo_linear + map_score) + offsets  ->  solve for map_score
        map_score = -(echo_total - offsets) - echo_linear
        return raw, weighted, echo_linear, map_score

    # -------------------------------------------------------------------- run

    def run(self):
        self.system.log(Messages.create_centered_box("Rescore Poses (ECHO)"))

        if not getattr(self.system, "ligand", None):
            raise ValueError(
                "No ligands to re-score. Supply poses with `ligand = poses.sdf` "
                "or `ligands_from_dir = <dir>` in the config."
            )

        self._get_output()
        self._dock = self._get_docking()
        self._ext = self._dock._echo_ext()
        self._weights = self._dock._weights()
        self._sites = self._get_sites()

        self.system.log(
            f"[rescore] engine={self._dock.ENGINE_NAME} "
            f"sites={[str(k) for k, _ in self._sites]} "
            f"rep_max={self._resolve_rep_max()}"
        )
        self._warn_unreproducible_map_term()

        relax_h = bool(self._opt("rescore_minimise_hydrogens", False))
        if relax_h:
            self.system.log(
                "[rescore] relaxing polar-hydrogen torsions against ECHO before "
                "scoring; this rewrites the ligand conformers in place, so any "
                "protocol running after --rescore-poses sees the relaxed poses."
            )

        relax_prot_h = bool(self._opt("rescore_protein_h", False))
        if relax_prot_h:
            self.system.log(
                "[rescore] relaxing rotatable protein donor-H orientations "
                f"(source={self._opt('protein_hydrogens', 'rdkit')}). The protein is "
                "restored between poses, so each pose gets its own best receptor "
                "hydrogen arrangement and none inherits the previous one's. "
                "Rotation is unpenalised, so echo_total is an upper bound: read it "
                "against echo_total_prot_h_pre."
            )

        rows = []
        for lig_idx, ligand in enumerate(self.system.ligand):
            rows.extend(self._rescore_ligand(lig_idx, ligand, relax_h, relax_prot_h))

        _number_poses(rows)
        self._write_csv(rows, relax_h, relax_prot_h)
        self._write_weights()
        if relax_prot_h:
            self._write_donor_csv(rows)
        if not self._opt("rescore_no_sdf", False):
            self._write_sdfs(rows)

        n_ok = sum(1 for r in rows if not r.get("error"))
        self.system.log(
            f"[rescore] {n_ok}/{len(rows)} poses scored -> {self.output}"
        )
        return rows

    def _rescore_ligand(self, lig_idx, ligand, relax_h, relax_prot_h=False):
        rows = []
        n_conf = ligand.mol.GetNumConformers()
        if n_conf == 0:
            self.system.log(f"[rescore] ligand {lig_idx} has no conformers; skipping.")
            return rows

        # Both of these are per-ligand, not per-pose: the precompute is a fixed
        # parameterisation and the torsion set is topology-driven.
        precomp_lig = PreCompDataLigand(
            ligand,
            self.system.platform,
            flexible_rings=False,
            resource_owner=self.system,
        )
        h_torsions = donor_h_torsions(ligand.mol) if relax_h else []
        if relax_h:
            self.system.log(
                f"[rescore] ligand {lig_idx} ({ligand.identifier}): "
                f"{len(h_torsions)} donor-H torsion(s), {n_conf} pose(s)"
            )

        # `combined` deep-copies the site precompute, so build one per
        # (site, ligand) pair and reuse it across that ligand's poses.
        combined_by_site = {}
        # Donors are derived from `combined`, not from the site precompute, so the
        # reference hydrogen coordinates they capture are the ones the relaxer will
        # write back into on restore.
        donors_by_site = {}

        for conf in ligand.mol.GetConformers():
            conf_id = conf.GetId()
            row = {
                "ligand": ligand.identifier,
                "ligand_idx": lig_idx,
                "source": str(getattr(ligand, "input", "")),
                "conf_id": conf_id,
                "site_id": "",
                "error": "",
            }
            try:
                site_id, binding_site = self._site_for_pose(conf.GetPositions())
                row["site_id"] = str(site_id)

                if site_id not in combined_by_site:
                    site_precomp = self._precomp_for_site(site_id, binding_site)
                    combined_by_site[site_id] = site_precomp + precomp_lig
                    donors_by_site[site_id] = (
                        rotatable_protein_donors(
                            binding_site.rdkit_lining_mol,
                            combined_by_site[site_id].protein_hydrogens,
                            combined_by_site[site_id].protein_positions,
                        ) if relax_prot_h else []
                    )
                    if relax_prot_h:
                        self.system.log(
                            f"[rescore] site {site_id}: "
                            f"{len(donors_by_site[site_id])} rotatable protein donor(s)"
                        )
                combined = combined_by_site[site_id]

                self._score_pose(row, combined, ligand, conf_id, h_torsions,
                                 donors_by_site.get(site_id, []))
            except Exception as exc:
                # One unscorable pose must not cost the caller the other 99.
                row["error"] = f"{type(exc).__name__}: {exc}"
                self.system.log(
                    f"[rescore] ligand {lig_idx} pose {conf_id} failed: {row['error']}"
                )
            rows.append(row)

        return rows

    def _score_pose(self, row, combined, ligand, conf_id, h_torsions, donors=()):
        if h_torsions:
            row["echo_total_prehmin"] = self._echo_total(combined, ligand.mol, conf_id)

            t0 = time.perf_counter()
            relaxer = HydrogenTorsionRelaxer(
                h_torsions,
                lambda mol: self._echo_total(combined, mol, 0),
                grid_deg=float(self._opt("rescore_h_min_grid", 60.0)),
                passes=int(self._opt("rescore_h_min_passes", 2)),
                maxiter=int(self._opt("rescore_h_min_maxiter", 100)),
            )
            _score, n_evals, max_delta = relaxer.relax(ligand.mol, conf_id)

            # Keep the ParmEd view in step with the conformer we just rotated.
            ligand.set_positions(
                ligand.mol.GetConformer(conf_id).GetPositions(), conf_id=conf_id
            )

            row["n_h_torsions"] = len(h_torsions)
            row["h_nm_evals"] = n_evals
            row["h_delta_deg_max"] = round(max_delta, 3)
            row["h_relax_seconds"] = round(time.perf_counter() - t0, 3)
        elif self._opt("rescore_minimise_hydrogens", False):
            row["echo_total_prehmin"] = ""
            row["n_h_torsions"] = 0
            row["h_nm_evals"] = 0
            row["h_delta_deg_max"] = 0.0
            row["h_relax_seconds"] = 0.0

        restore_protein = None
        if self._opt("rescore_protein_h", False):
            restore_protein = self._relax_protein_hydrogens(
                row, combined, ligand, conf_id, donors
            )

        # Score after relaxation so the reported total matches the written pose.
        echo_total = self._echo_total(combined, ligand.mol, conf_id)
        terms = self._echo_terms(combined, ligand.mol, conf_id)
        raw, weighted, echo_linear, map_score = self._breakdown(terms, echo_total)

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

        if restore_protein is not None:
            # Every pose is scored against the same reference receptor, so the
            # winning orientations must not leak into the next pose.
            restore_protein()

    def _relax_protein_hydrogens(self, row, combined, ligand, conf_id, donors):
        """Rotate the site's rotatable donor hydrogens to suit this pose.

        Returns a callable that puts the protein back, or None if nothing moved.
        The caller invokes it *after* scoring, so echo_total and the term
        breakdown describe the relaxed receptor while the next pose still starts
        from the reference placement.
        """
        # Heavy atoms only: the H-bond loop iterates the ligand's heavy atoms
        # (ScoringFunctions.cpp:246), so counting hydrogens here would report
        # donors as "in range" that the scorer never pairs with anything.
        coords = ligand.mol.GetConformer(conf_id).GetPositions()
        heavy = [a.GetIdx() for a in ligand.mol.GetAtoms() if a.GetAtomicNum() != 1]
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
            lambda: self._echo_total(combined, ligand.mol, conf_id),
            grid_deg=float(self._opt("rescore_protein_h_grid", 30.0)),
            passes=int(self._opt("rescore_protein_h_passes", 2)),
            maxiter=int(self._opt("rescore_protein_h_maxiter", 100)),
        )
        score, start_score, n_evals, max_delta, per_donor, best = relaxer.relax()

        row["prot_h_evals"] = n_evals
        row["prot_h_delta_deg_max"] = round(max_delta, 3)
        row["prot_h_relax_seconds"] = round(time.perf_counter() - t0, 3)
        row["echo_total_prot_h_pre"] = "" if start_score is None else start_score

        for entry in per_donor:
            self._donor_rows.append(dict(
                entry,
                ligand=row["ligand"],
                ligand_idx=row["ligand_idx"],
                conf_id=row["conf_id"],
                site_id=row["site_id"],
            ))

        if best is None or score is None:
            return None

        relaxer.apply(best)
        return relaxer.restore

    # ----------------------------------------------------------------- output

    def _columns(self, relax_h, relax_prot_h=False):
        cols = [
            "ligand", "source", "pose", "ligand_idx", "conf_id", "site_id",
            "echo_total", "echo_linear", "map_score",
        ]
        cols += OFFSET_TERMS
        if relax_h:
            cols += [
                "echo_total_prehmin", "n_h_torsions", "h_delta_deg_max",
                "h_nm_evals", "h_relax_seconds",
            ]
        if relax_prot_h:
            cols += [
                "echo_total_prot_h_pre", "n_prot_h_donors", "n_prot_h_donors_close",
                "prot_h_delta_deg_max", "prot_h_evals", "prot_h_relax_seconds",
            ]
        cols += [f"raw_{name}" for name in RAW_TERMS]
        cols += [f"w_{name}" for name in LINEAR_TERMS]
        cols.append("error")
        return cols

    def _write_csv(self, rows, relax_h, relax_prot_h=False):
        path = os.path.join(self.output, "echo_rescore.csv")
        cols = self._columns(relax_h, relax_prot_h)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in cols})
        self.system.log(f"[rescore] wrote {path}")

    def _write_donor_csv(self, rows):
        """Per-donor attribution, one row per (pose, rotatable donor).

        The pose-level delta says whether relaxation helps; only this says which
        residues are responsible, which is what separates a real mis-oriented
        hydroxyl from the optimiser finding score noise.
        """
        path = os.path.join(self.output, "echo_rescore_protein_h.csv")
        cols = ["ligand", "ligand_idx", "conf_id", "pose", "site_id",
                "donor", "heavy_idx", "planar", "delta_deg", "delta_echo"]

        pose_by_key = {
            (r["ligand_idx"], r["conf_id"]): r.get("pose", "") for r in rows
        }
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for entry in self._donor_rows:
                entry = dict(entry)
                entry["pose"] = pose_by_key.get(
                    (entry["ligand_idx"], entry["conf_id"]), ""
                )
                writer.writerow({c: entry.get(c, "") for c in cols})
        self.system.log(
            f"[rescore] wrote {path} ({len(self._donor_rows)} donor rows)"
        )

    def _write_weights(self):
        path = os.path.join(self.output, "echo_weights.json")
        payload = {
            name: float(getattr(self._weights, name, 0.0))
            for name in LINEAR_TERMS + LUMPED_TERMS
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=4)
        self.system.log(f"[rescore] wrote {path}")

    def _write_sdfs(self, rows):
        """One SDF per input source, poses best-first, terms as SD properties.

        Grouped by source rather than by ligand index on purpose: a multi-record
        poses SDF is loaded as one Ligand *per record*, so grouping by ligand
        would emit a file per pose -- all to the same filename.
        """
        by_source = {}
        for row in rows:
            if row.get("error"):
                continue
            by_source.setdefault(row["source"], []).append(row)

        for source, src_rows in by_source.items():
            # Lower is better, matching the sort docking applies to its poses.
            src_rows.sort(key=lambda r: r["echo_total"])

            name = _source_stem(source, src_rows[0]["ligand_idx"])
            path = os.path.join(self.output, f"{name}_rescored.sdf")
            with Chem.SDWriter(path) as writer:
                for rank, row in enumerate(src_rows):
                    ligand = self.system.ligand[row["ligand_idx"]]
                    mol = Chem.Mol(ligand.mol)
                    mol.SetProp("_Name", f"{name}_pose_{row['pose']}")
                    mol.SetIntProp("rescore_rank", rank)
                    for key, value in row.items():
                        if key in ("ligand_idx", "conf_id", "pose"):
                            mol.SetIntProp(key, int(value))
                        elif isinstance(value, float):
                            mol.SetDoubleProp(key, float(value))
                        else:
                            mol.SetProp(key, str(value))
                    writer.write(mol, confId=row["conf_id"])
            self.system.log(f"[rescore] wrote {path} ({len(src_rows)} poses)")


def _number_poses(rows):
    """Give every pose a running index within its input source.

    A multi-record poses SDF is loaded as one Ligand per record, each with a
    single conformer, so neither `ligand_idx` nor `conf_id` alone reads as a pose
    number. `pose` is that number, in input order.
    """
    counters = {}
    for row in rows:
        source = row.get("source", "")
        row["pose"] = counters.get(source, 0)
        counters[source] = row["pose"] + 1


def _source_stem(source, lig_idx):
    """Filesystem-safe stem for an input source, which may be a SMILES string."""
    source = str(source or "")
    if os.path.exists(source):
        stem = os.path.splitext(os.path.basename(source))[0]
    else:
        stem = ""
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)[:80]
    return safe or f"Ligand_{lig_idx}"
