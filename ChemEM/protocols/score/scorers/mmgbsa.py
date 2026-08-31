#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Single-frame MM-GBSA binding energy of a pose (``--score --score-with mmgbsa``).

Needs the prepared receptor and the ligand's force-field parameters, but no binding
site and no map, so ``score_deps`` returns ``()``.

Two things worth knowing before reading the numbers:

  * MM-GBSA is defined on *refined* geometry. On a raw docked pose a single residual
    clash dominates the van der Waals term and gives an absurd deltaG.
    ``--score-mmgbsa-minimise`` relaxes the ligand inside a pinned pocket first and
    reports how far it moved as ``mmgbsa_min_shift_A``; use it, or read raw-pose
    energies as a ranking at best.
  * this scorer is all-atom, so unlike Q-score and the density terms it *does* see
    the hydrogens ``--score-echo-minimise-hydrogens`` moved. That is why hydrogen
    relaxation is a ``pre_score`` phase running before every scorer rather than
    something ECHO does privately.
"""

from __future__ import annotations

import os

from ChemEM.protocols.orchestrator.scoring import mmgbsa_single_frame

from ..poses import source_stem
from .base import PoseScorer

COMPONENTS = ("EEL", "VDW", "EGB", "ECAV")


class MMGBSAScorer(PoseScorer):
    NAME = "mmgbsa"
    HELP = "Single-frame MM-GBSA binding energy (kcal/mol)"
    DEPS = ()
    HEADLINE = "mmgbsa"
    HIGHER_IS_BETTER = False
    NEEDS_SITE = False

    COLUMNS = ("mmgbsa",) + tuple(f"mmgbsa_{c.lower()}" for c in COMPONENTS) + (
        "mmgbsa_min_shift_A",
    )

    def __init__(self, system, opts):
        super().__init__(system, opts)
        self._minimised = []          # (row-identity, relaxed coords) per pose

    def extra_columns(self):
        return ("mmgbsa_minimised_sdf",) if self._opt("score_mmgbsa_write_minimised") else ()

    def setup_run(self, ctx) -> None:
        if getattr(self.system, "protein", None) is None:
            raise ValueError("MM-GBSA scoring requires a prepared protein.")
        self.minimise = bool(self._opt("score_mmgbsa_minimise", False))
        self.min_iters = int(self._opt("score_mmgbsa_min_iters", 300))
        self.pocket_radius = float(self._opt("score_mmgbsa_pocket_radius", 12.0))
        # Reusing the per-ligand OpenMM systems and Contexts across poses is
        # bit-identical (verified by ChemEM.benchmark.verify_mmgbsa_cache) and much
        # faster; --score-mmgbsa-no-cache exists only to re-run that check.
        self.reuse_cache = not bool(self._opt("score_mmgbsa_no_cache", False))
        self.write_minimised = bool(self._opt("score_mmgbsa_write_minimised", False))
        self.system.log(
            f"[score:mmgbsa] minimise={self.minimise} "
            f"pocket_radius={self.pocket_radius} reuse_cache={self.reuse_cache}"
        )
        if self.write_minimised and not self.minimise:
            raise ValueError(
                "--score-mmgbsa-write-minimised needs --score-mmgbsa-minimise: "
                "without it no minimisation runs, so there is no relaxed pose to write."
            )
        if self.write_minimised:
            self.system.log(
                "[score:mmgbsa] writing the pocket-relaxed poses. Only the LIGAND "
                "moves -- the pocket residues are positionally pinned and their "
                "minimised coordinates are discarded in favour of the reference "
                "ones -- so the receptor written alongside is the prepared input, "
                "unchanged."
            )
        if not self.minimise:
            self.system.log(
                "[score:mmgbsa] NOTE: scoring un-minimised poses. MM-GBSA is defined "
                "on refined geometry; pass --score-mmgbsa-minimise unless these "
                "poses are already refined."
            )

    def score(self, pose, row) -> None:
        result = mmgbsa_single_frame(
            pose.coords,
            pose.ligand,
            self.system.protein,
            pose_idx=pose.pose,
            resource_owner=self.system,
            minimise_ligand=self.minimise,
            minimise_iters=self.min_iters,
            minimise_cutoff_A=self.pocket_radius,
            reuse_cache=self.reuse_cache,
        )
        if result is None:
            # mmgbsa_single_frame swallows its own exceptions, so a None here means
            # the pose could not be parameterised or the energy eval blew up.
            row["mmgbsa_failed"] = 1
            return

        row["mmgbsa"] = float(result.deltaG)
        components = getattr(result, "components", None) or {}
        for name in COMPONENTS:
            value = components.get(name)
            row[f"mmgbsa_{name.lower()}"] = float(value) if value is not None else ""
        if self.minimise:
            shift = getattr(result, "min_shift_A", None)
            row["mmgbsa_min_shift_A"] = float(shift) if shift is not None else ""

        if self.write_minimised:
            coords = getattr(result, "min_coords_A", None)
            if coords is None:
                # No receptor within the pocket radius: _pocket_minimise returns the
                # pose untouched, so there is nothing relaxed to write.
                row["mmgbsa_minimised_sdf"] = ""
            else:
                self._minimised.append((pose, dict(row), coords))

    # ----------------------------------------------------------------- output

    def finish_run(self, ctx, rows) -> dict:
        meta = {
            "minimise": self.minimise,
            "min_iters": self.min_iters,
            "pocket_radius": self.pocket_radius,
            "reuse_cache": self.reuse_cache,
        }
        if self._minimised:
            meta["minimised_sdfs"] = self._write_minimised_sdfs(ctx, rows)
            receptor = self._write_receptor(ctx)
            if receptor:
                meta["receptor_pdb"] = receptor
        return meta

    def _write_minimised_sdfs(self, ctx, rows):
        """One SDF per input source holding the pocket-relaxed ligand poses.

        Written from a copy of the pose molecule with the relaxed coordinates
        substituted, so the atom order matches the input SDF exactly and the file
        can be diffed against it.
        """
        from rdkit import Chem
        from rdkit.Geometry import Point3D

        by_source = {}
        for pose, row, coords in self._minimised:
            by_source.setdefault(pose.source, []).append((pose, row, coords))

        written = []
        for source, entries in by_source.items():
            name = source_stem(source, entries[0][0].ligand_idx)
            path = os.path.join(ctx.output, f"{name}_mmgbsa_minimised.sdf")
            with Chem.SDWriter(path) as writer:
                for pose, row, coords in entries:
                    mol = Chem.Mol(pose.mol)
                    mol.RemoveAllConformers()
                    conf = Chem.Conformer(mol.GetNumAtoms())
                    for i, (x, y, z) in enumerate(coords):
                        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
                    mol.AddConformer(conf, assignId=True)
                    mol.SetProp("_Name", f"{name}_pose_{pose.pose}_mmgbsa_min")
                    mol.SetIntProp("pose", int(pose.pose))
                    mol.SetDoubleProp("mmgbsa", float(row["mmgbsa"]))
                    shift = row.get("mmgbsa_min_shift_A")
                    if shift not in ("", None):
                        mol.SetDoubleProp("mmgbsa_min_shift_A", float(shift))
                    writer.write(mol)
            written.append(path)
            self.system.log(
                f"[score:mmgbsa] wrote {path} ({len(entries)} relaxed pose(s))"
            )

        # Point every row at the file its pose landed in.
        paths = {}
        for source, entries in by_source.items():
            name = source_stem(source, entries[0][0].ligand_idx)
            paths[source] = os.path.join(ctx.output, f"{name}_mmgbsa_minimised.sdf")
        for row in rows:
            if "mmgbsa_minimised_sdf" not in row:
                row["mmgbsa_minimised_sdf"] = paths.get(row.get("source"), "")
        return written

    def _write_receptor(self, ctx):
        """The receptor the poses were relaxed against.

        Written for convenience so the SDF has something to load against; it is the
        prepared input structure, NOT a minimised one. The pocket residues are pinned
        during the relaxation and their minimised coordinates never leave
        ``_pocket_minimise`` -- the returned complex keeps the reference receptor.
        """
        struct = getattr(getattr(self.system, "protein", None), "complex_structure", None)
        if struct is None:
            return None
        path = os.path.join(ctx.output, "mmgbsa_receptor.pdb")
        try:
            struct.save(path, overwrite=True)
        except Exception as exc:
            self.system.log(f"[score:mmgbsa] could not write {path}: {exc}")
            return None
        self.system.log(
            f"[score:mmgbsa] wrote {path} (prepared receptor, unchanged by "
            "minimisation -- the pocket is pinned)"
        )
        return path
