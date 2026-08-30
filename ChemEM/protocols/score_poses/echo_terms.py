#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
echo_terms — dump the RAW (unweighted) ECHO docking-score channels per pose, for
offline weight-fitting on the CrossDocked funnel.

For ONE scoring conf (= one receptor + one binding-site centroid + a native ligand and
its decoy pose SDFs) it:
  * builds the binding site / protein precompute **once** (no density map), writing NO
    binding-site files (`write_site_files=False`);
  * runs every pose of every `ligand=` SDF through ChemEM **protonation** (coordinates
    preserved, Hs added), then scores it with the C++ `run_echo_terms`;
  * writes one CSV row per pose: identity + the ~18 raw channels (incl. the split
    aromatic/nonbond attractive/repulsive/clash sub-terms) — joinable to `rmsds.csv`
    on `sdf_path` + `pose_idx`.

Deliberately no map terms (CrossDocked has no map) and no weighted score — the fit
consumes the raw channels.

Run from the repo root so the local ChemEM package is imported:

    cd .../chemem2-dev
    python -m ChemEM.protocols.score_poses.echo_terms CONF --out CASE.csv \\
        [--rep-max 5.0 --interaction-cutoff 6.0 --electro-clamp 2.0 --proton-ph 7.4]

One CSV per conf → naturally shardable/resumable across the cluster.
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile

from rdkit import Chem

from ChemEM.__main__ import (
    apply_overrides,
    build_parser,
    build_pipeline,
    load_system,
    resolve_protocol_order,
)
from ChemEM.parsers.ligand_parser import _make_ligand_from_rd_mol
from ChemEM.parsers.remodel.protonation import set_mol_protonatation_state
from ChemEM.tools.precomputed_data import PreCompDataLigand, PreCompDataProtein
from ChemEM import docking

# Raw channels returned by run_echo_terms (split + lumped + penalties).
TERM_COLUMNS = [
    "aromatic_attr", "aromatic_clash",
    "nonbond_attr", "nonbond_rep", "clash",
    "saltbridge_raw", "hbond_raw",
    "ligand_intra", "ligand_torsion",
    "electro_attractive", "electro_repulsive_clamp", "desolvation_penalty_scaled",
    "hphobe_raw_hpho", "hphobe_raw_hpil",
    "hphob_enc_gt_7_only_hpho", "hphob_enc_gt_7_only_hpil_unsat",
    "unsat_polar",
    "aromatic", "nonbond",          # lumped (== sum of their split channels)
    "bias", "constraint",
]
# `decoy_file` (basename) + `folder` + `pose_idx` is the ROOT-INDEPENDENT join key to
# rmsds.csv (echo_terms runs on the HPC with different absolute paths than compute_rmsds).
LEAD_COLUMNS = ["case_id", "folder", "rec_id", "lig_token", "decoy_file", "sdf_path",
                "pose_idx", "role"]


# --------------------------------------------------------------------------- #
# Conf parsing / system build
# --------------------------------------------------------------------------- #
def parse_conf(path):
    """Pull protein, centroid line(s), output and the ordered ligand paths from a conf."""
    protein = output = None
    centroids, ligands = [], []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            key, val = s.split("=", 1)
            key, val = key.strip().lower(), val.strip()
            if key == "protein":
                protein = val
            elif key == "output":
                output = val
            elif key == "centroid":
                centroids.append(val)
            elif key == "ligand":
                ligands.append(val)
    return protein, output, centroids, ligands


def build_site_system(args, protein, centroids, out_dir):
    """Build protein + binding site from a LIGAND-FREE temp conf (skips loading/protonating
    the decoy poses up front). Runs only the binding_site protocol with writes suppressed."""
    lines = [f"protein = {protein}", f"output = {out_dir}"]
    lines += [f"centroid = {c}" for c in centroids]
    conf_path = os.path.join(out_dir, "_site.conf")
    with open(conf_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    system = load_system(conf_path)
    system.options = args
    apply_overrides(system, args)
    args.write_site_files = False  # gate in protocols/binding_site.py suppresses all site output

    order = resolve_protocol_order(["binding_site"], args)
    build_pipeline(system, order)
    system.run()
    return system


def protonate(raw_mol, ph, pka_prec):
    """ChemEM protonation of one pose (preserves heavy-atom coords, adds Hs)."""
    m = set_mol_protonatation_state(raw_mol, pH=ph, pka_prec=pka_prec, max_varients=1)
    if m is None:
        m = Chem.AddHs(raw_mol, addCoords=True)
    return m


def make_precomp_lig(prot_mol, sdf_path, system):
    Chem.GetSymmSSSR(prot_mol)  # ring perception for aromatic ring types
    lig_obj = _make_ligand_from_rd_mol(prot_mol, sdf_path, name="LIG")
    return PreCompDataLigand(lig_obj, system.platform,
                             flexible_rings=False, resource_owner=system)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = build_parser()  # config positional + every protocol's args/defaults
    g = parser.add_argument_group("echo_terms")
    g.add_argument("--out", required=True, help="output CSV (one row per pose)")
    g.add_argument("--rep-max", type=float, default=5.0, help="ECHO repulsion cap (default 5.0)")
    g.add_argument("--interaction-cutoff", type=float, default=6.0)
    g.add_argument("--electro-clamp", type=float, default=2.0)
    g.add_argument("--proton-ph", type=float, default=7.4, help="protonation pH (default 7.4)")
    g.add_argument("--proton-pka-prec", type=float, default=0.0)
    g.add_argument("--case-id", default=None)
    args = parser.parse_args(argv)

    conf = args.config
    protein, _output, centroids, ligand_paths = parse_conf(conf)
    if not protein or not centroids or not ligand_paths:
        raise SystemExit(f"conf missing protein/centroid/ligand: {conf}")

    base = os.path.splitext(os.path.basename(conf))[0]
    folder = os.path.basename(os.path.dirname(conf))
    rec_id, lig_token = (base.split("__", 1) + [""])[:2] if "__" in base else (base, "")
    case_id = args.case_id or base

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        system = build_site_system(args, protein, centroids, tmp)

        sites = list(system.binding_sites.items())
        if not sites:
            raise SystemExit(f"no binding site produced for {conf}")
        if len(sites) > 1:
            print(f"[echo_terms] WARNING: {len(sites)} sites; using the first ({sites[0][0]})")
        binding_site = sites[0][1]

        if getattr(binding_site, "is_alpha_feature_site", False):
            bias_radius = float(system.options.feature_site_radius)
        else:
            bias_radius = float(system.options.bias_radius)
        precomp_site = PreCompDataProtein(
            binding_site, system,
            bias_radius=bias_radius,
            split_site=system.options.split_site,
        )

        for lig_ord, sdf_path in enumerate(ligand_paths):
            role = "native" if lig_ord == 0 else "decoy"
            combined = None
            ref_natoms = None
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)  # entry idx == compute_rmsds pose_idx

            for pose_idx, raw in enumerate(suppl):
                base_row = dict(case_id=case_id, folder=folder, rec_id=rec_id,
                                lig_token=lig_token, decoy_file=os.path.basename(sdf_path),
                                sdf_path=sdf_path, pose_idx=pose_idx, role=role)
                if raw is None:
                    rows.append({**base_row, "error": "pose_parse_failed"})
                    continue
                try:
                    prot = protonate(raw, args.proton_ph, args.proton_pka_prec)
                    if combined is None:
                        combined = precomp_site + make_precomp_lig(prot, sdf_path, system)
                        ref_natoms = prot.GetNumAtoms()
                        cb = combined
                    elif prot.GetNumAtoms() != ref_natoms:
                        # atom-order / topology guard: score this pose with its own precomp
                        cb = precomp_site + make_precomp_lig(prot, sdf_path, system)
                    else:
                        cb = combined
                    block = Chem.MolToMolBlock(prot, includeStereo=True, confId=0)
                    terms = docking.run_echo_terms(
                        cb, block, confId=0,
                        interaction_cutoff=args.interaction_cutoff,
                        rep_max=args.rep_max,
                        electro_clamp=args.electro_clamp,
                    )
                    row = {**base_row, "error": ""}
                    for k in TERM_COLUMNS:
                        row[k] = terms.get(k, "")
                    rows.append(row)
                except Exception as exc:  # never abort the whole file on one bad pose
                    rows.append({**base_row, "error": f"score_failed:{type(exc).__name__}"})

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cols = LEAD_COLUMNS + TERM_COLUMNS + ["error"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    n_ok = sum(1 for r in rows if r.get("error", "") == "")
    print(f"[echo_terms] {case_id}: poses={len(rows)} scored={n_ok} "
          f"errors={len(rows) - n_ok} -> {out_path}")


if __name__ == "__main__":
    main()
