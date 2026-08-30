#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_poses — standalone offline pose re-scorer for ChemEM orchestrator development.

Re-scores already-generated docked poses (the orchestrator *audit* SDFs) with the
FULL set of per-pose score terms — Q-score, density coverage/precision/overlap/ccc,
mutual information, SCI (+ components), shape descriptors, skeleton/topology — and,
optionally (``--do-mmgbsa``), single-frame MMGBSA. It writes one CSV row per pose
for external analysis (term selection / scoring-function development).

Design goals (deliberately self-contained + additive):
  * IMPORTS the existing ChemEM scorers — no changes to orchestrator/triage/scoring;
  * NOT registered in ``protocol_spec.REGISTRY``;
  * delete this directory to remove the tool entirely.

It rebuilds the *same* ``system`` a normal run uses — ``load_system`` + the
``binding_site`` / ``alpha_mask`` / ``confidence_map`` segmentation — so the site
maps (and therefore the scores) match the pipeline exactly. **No docking,
minimisation, or smart_refine is run.** Invoke it with the SAME config (and the
same segmentation options) used for the original ``--orchestrate`` run.

Usage
-----
Run from the repository root (the dir that contains the ``ChemEM/`` package) so the
local package is imported rather than any stale install::

    cd .../chemem2-dev
    python -m ChemEM.protocols.score_poses.score_poses CONFIG \\
        [--poses-dir DIR] [--score-out scores_full_terms.csv] \\
        [--do-mmgbsa] [--reference REF.sdf --rmsd-cutoff 2.0]

(or set ``PYTHONPATH`` to that root). ``--poses-dir`` defaults to
``<output>/audit/01_gate1_scored/sdf`` (the full n x m grid). Point it at any audit
stage's ``sdf`` directory to score that stage instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time

import numpy as np
from rdkit import Chem

from ChemEM.__main__ import (
    apply_overrides,
    build_parser,
    build_pipeline,
    load_system,
    resolve_protocol_order,
)
from ChemEM.protocols.orchestrator.scoring import (
    density_fit_metrics,
    mmgbsa_single_frame,
    qscore_pose_metrics,
)

# Segmentation protocols that populate ``system.binding_site_maps`` (== dock's deps
# minus dock). Running exactly these reproduces the site maps the orchestrator
# scored against, with no docking or refinement.
SEGMENTATION_PROTOCOLS = ["binding_site", "alpha_mask", "confidence_map"]

# Per-atom lists / nested dicts that do not belong in a flat numeric CSV.
_DROP_KEYS = (
    "q_per_atom",
    "q_heavy_atom_indices",
    "density_feature_metrics",
    "density_sci_terms",
)

_POSE_RE = re.compile(r"ligand_(\d+)_pose_(\d+)")
# Docking-output layout: .../{docking,refined}/binding_site_<bs>_Ligand_<lig>/Ligand_<sol>.sdf
_CONF_BS_RE = re.compile(r"binding_site_(\d+)_Ligand_(\d+)")
_CONF_SOL_RE = re.compile(r"Ligand_(\d+)\.sdf$")

# Preferred leading column order (only those actually present are emitted).
_LEAD_COLUMNS = [
    "case_id", "site_id", "ligand_idx", "pose_idx", "stage",
    "dock_score", "qscore", "q_mean", "q_low_tail",
    "density_coverage", "density_precision", "density_overlap", "density_ccc",
    "density_mi", "density_normalized_mi", "density_envelope_iou",
    "density_excess_fraction", "density_sci",
    "mmgbsa", "mmgbsa_eel", "mmgbsa_vdw", "mmgbsa_egb", "mmgbsa_ecav", "mmgbsa_min_shift_A",
    "ligand_strain",
    "protein_ligand_clash_penalty", "protein_ligand_clash_count", "protein_ligand_max_overlap_A",
    "ligand_self_clash_penalty", "ligand_self_clash_count",
    "rmsd_to_ref", "is_correct",
]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    # build_parser() gives us the `config` positional + every protocol's args
    # (incl. the segmentation options) with their defaults, so segmentation runs
    # identically to a real invocation.
    parser = build_parser()
    g = parser.add_argument_group("score_poses (offline re-scorer)")
    g.add_argument("--poses-dir", default=None,
                   help="Directory of audit pose SDFs to score "
                        "(default: <output>/audit/01_gate1_scored/sdf).")
    g.add_argument("--score-out", default="scores_full_terms.csv",
                   help="Output CSV path (one row per pose).")
    g.add_argument("--case-id", default=None,
                   help="Case id written into the CSV (default: output dir basename).")
    g.add_argument("--do-mmgbsa", action="store_true",
                   help="Also compute single-frame MMGBSA per pose (slower; needs the receptor). "
                        "MMGBSA is defined on refined geometry, so it is most meaningful on "
                        "post-minimisation survivor poses; on raw docked poses expect noisy energies. "
                        "Writes mmgbsa (deltaG) + mmgbsa_eel/vdw/egb/ecav components.")
    g.add_argument("--mmgbsa-minimise", action="store_true",
                   help="Before the single-frame MMGBSA, relax the ligand inside a pinned pocket of "
                        "surrounding residues (~pocket-radius A), then score the FULL complex with no "
                        "minimise. Removes docking clashes (raw poses can give absurd deltaG) cheaply; "
                        "adds mmgbsa_min_shift_A (how far the ligand moved).")
    g.add_argument("--mmgbsa-min-iters", type=int, default=300,
                   help="Max iterations for the pocket minimise (default 300).")
    g.add_argument("--mmgbsa-pocket-radius", type=float, default=12.0,
                   help="Pocket radius (A) for --mmgbsa-minimise: residues within this distance of the "
                        "ligand are included and pinned; the ligand relaxes among them (default 12).")
    g.add_argument("--do-strain", action="store_true",
                   help="Also compute ligand internal MMFF94 strain per pose "
                        "(ligand_strain = E_pose - E_relaxed, kcal/mol; cheap, no map/protein). "
                        "High strain flags a ligand contorted to fit fake/wrong density.")
    g.add_argument("--do-clash", action="store_true",
                   help="Also compute protein-ligand + ligand self steric-clash penalties per pose "
                        "(cheap; reuses the smart_refine_2 clash utilities).")
    g.add_argument("--reference", default=None,
                   help="Reference/deposited ligand SDF; if given, adds rmsd_to_ref + is_correct "
                        "columns (heavy-atom, symmetry-corrected, NO superposition).")
    g.add_argument("--rmsd-cutoff", type=float, default=2.0,
                   help="RMSD (A) at/below which a pose is labelled is_correct=1.")
    g.add_argument("--score-density-threshold-frac", type=float, default=0.05,
                   help="threshold_frac for density_fit_metrics (match your orch run; default 0.05).")
    g.add_argument("--layout", choices=["audit", "conf"], default="audit",
                   help="audit (default): walk --poses-dir for site_*/ligand_*_pose_*.sdf. "
                        "conf: score the poses listed as `ligand=` lines in the config; the "
                        "binding-site id and stage are parsed from each ligand's path.")
    g.add_argument("--density-region", choices=["full", "box", "site"], default="full",
                   help="Region the density-fit terms are scored against. full (default): the "
                        "whole map (all terms correct; coverage denominator is a per-case "
                        "constant). box: a fixed-size cube around each pose (normalised coverage; "
                        "faster). site: legacy segmented binding-site map (needs segmentation; can "
                        "miss -> no_site_map).")
    g.add_argument("--box-size", type=float, default=24.0,
                   help="Cube edge in A for --density-region box (centred on each pose centroid).")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# System build (segmentation only — no fitting)
# --------------------------------------------------------------------------- #
def build_scored_system(args):
    """load_system; run the segmentation chain only when it is needed (--density-region site).

    full/box regions score against the full map (or a crop of it), so segmentation is skipped
    entirely — faster, and it removes the no_site_map failure mode.
    """
    system = load_system(args.config)
    system.options = args
    apply_overrides(system, args)
    if getattr(args, "density_region", "full") == "site":
        order = resolve_protocol_order(SEGMENTATION_PROTOCOLS, args)
        build_pipeline(system, order)
        system.run()
    return system


def region_map_for(system, ctx, site_id, coords, row):
    """Return the map object to score the density-fit terms against for --density-region.

    full -> the whole map; box -> a fixed-size cube around the pose centroid; site -> the
    legacy segmented binding-site map. Records density_failed / flags on `row` when unavailable.
    """
    region = ctx["density_region"]
    full = getattr(system, "density_map", None)
    if region == "site":
        sm = site_maps_for(system, site_id)
        if sm is None:
            row["density_failed"] = "no_site_map"
        return sm
    if full is None:
        row["density_failed"] = "no_density_map"
        return None
    if region == "full":
        return full
    # box: fixed-size cube (box_size A) cropped from the full map, centred on the pose centroid.
    from ChemEM.tools.density import crop_map_around_point
    from ChemEM.parsers.EMMap import EMMap
    heavy = np.asarray(coords, dtype=float)
    centroid = heavy.mean(axis=0)
    box_size = float(ctx["box_size"])
    apix = np.asarray(full.apix, dtype=float).reshape(-1)
    if apix.size == 1:
        apix = np.array([float(apix), float(apix), float(apix)])
    try:
        sub, new_origin = crop_map_around_point(
            np.asarray(full.density_map, dtype=float),
            np.asarray(full.origin, dtype=float),
            apix,
            centroid,
            box_size,
        )
    except Exception as exc:  # noqa: BLE001 - centre off-map, etc.
        row["density_failed"] = f"box_empty:{type(exc).__name__}"
        return None
    if sub.size == 0:
        row["density_failed"] = "box_empty"
        return None
    row["box_size"] = box_size
    if float(np.max(np.abs(heavy - centroid))) > box_size / 2.0:
        row["ligand_exceeds_box"] = 1  # some atoms fall outside the box -> partial density
    return EMMap(tuple(float(x) for x in new_origin),
                 tuple(float(a) for a in apix), sub, getattr(full, "resolution", 3.0))


def site_maps_for(system, site_id):
    """Mirror SmartOrchestrator._site_maps_for: tolerate str/int key mismatch."""
    maps = getattr(system, "binding_site_maps", None) or {}
    if site_id in maps:
        return maps[site_id]
    try:
        int_id = int(site_id)
    except (TypeError, ValueError):
        int_id = None
    if int_id is not None and int_id in maps:
        return maps[int_id]
    for key, value in maps.items():
        if str(key) == str(site_id):
            return value
    return None


# --------------------------------------------------------------------------- #
# Pose IO
# --------------------------------------------------------------------------- #
def iter_pose_files(poses_dir):
    """Yield (site_id, ligand_idx, pose_idx, sdf_path) for site_*/ligand_*_pose_*.sdf."""
    if not os.path.isdir(poses_dir):
        raise FileNotFoundError(f"poses-dir not found: {poses_dir}")
    for site_dir in sorted(os.listdir(poses_dir)):
        if not site_dir.startswith("site_"):
            continue
        site_path = os.path.join(poses_dir, site_dir)
        if not os.path.isdir(site_path):
            continue
        site_id = site_dir[len("site_"):]
        for fname in sorted(os.listdir(site_path)):
            if not fname.endswith(".sdf"):
                continue
            m = _POSE_RE.search(fname)
            if not m:
                continue
            yield site_id, int(m.group(1)), int(m.group(2)), os.path.join(site_path, fname)


def _parse_conf_pose_path(path):
    """(stage, site_id, ligand_idx, pose_idx) parsed from a docking-output pose path."""
    sep = os.sep
    if f"{sep}refined{sep}" in path:
        stage = "refined"
    elif f"{sep}docking{sep}" in path:
        stage = "docking"
    else:
        stage = "unknown"
    bs = ligand_idx = pose_idx = None
    m = _CONF_BS_RE.search(path)
    if m:
        bs, ligand_idx = m.group(1), int(m.group(2))
    m2 = _CONF_SOL_RE.search(os.path.basename(path))
    if m2:
        pose_idx = int(m2.group(1))
    return stage, bs, ligand_idx, pose_idx


def iter_conf_poses(system):
    """Yield (stage, site_id, ligand_idx, pose_idx, sdf_path, ligand_obj) for each pose
    loaded from the config's `ligand=` lines. Site/stage are parsed from the ligand path;
    the ligand object is passed through for optional MMGBSA (FF params)."""
    for ligand in getattr(system, "ligand", None) or []:
        raw = getattr(ligand, "input", None)
        if not raw:
            continue
        path = os.path.abspath(str(raw))
        if not path.endswith(".sdf"):
            continue
        stage, bs, ligand_idx, pose_idx = _parse_conf_pose_path(path)
        yield stage, bs, ligand_idx, pose_idx, path, ligand


def load_pose(sdf_path):
    """Return (mol, coords, sanitized) for one pose SDF.

    Atom order is the file order, which equals the ligand's atom order (the audit
    SDF was written from system.ligand[idx]), so `coords` aligns with ligand.mol.
    Tries a sanitized read first (usable for symmetry-aware RMSD); falls back to an
    unsanitized read (coords only) if sanitization fails on a strained pose.
    """
    for sanitize in (True, False):
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=sanitize)
        mol = next((m for m in supplier if m is not None), None)
        if mol is not None and mol.GetNumConformers() > 0:
            coords = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
            return mol, coords, sanitize
    return None, None, False


# --------------------------------------------------------------------------- #
# RMSD (no superposition, symmetry-corrected, heavy-atom)
# --------------------------------------------------------------------------- #
def pose_rmsd_to_reference(pose_mol, ref_mol):
    """Heavy-atom RMSD between a pose and a reference in the SAME frame.

    Unlike AllChem.GetBestRMS this does NOT superpose the molecules — the pose and
    the deposited reference already share the map coordinate frame, so alignment
    would wrongly reward a correctly-shaped ligand placed in the wrong location.
    Symmetry is handled by minimising over automorphic atom matches.
    """
    try:
        probe = Chem.RemoveHs(pose_mol)
        ref = Chem.RemoveHs(ref_mol)
    except Exception:
        probe, ref = pose_mol, ref_mol
    try:
        pc = np.asarray(probe.GetConformer().GetPositions(), dtype=float)
        rc = np.asarray(ref.GetConformer().GetPositions(), dtype=float)
    except Exception:
        return None

    matches = probe.GetSubstructMatches(ref, uniquify=False)
    if not matches:
        n = min(len(pc), len(rc))
        if n == 0:
            return None
        d = pc[:n] - rc[:n]
        return float(np.sqrt((d * d).sum() / n))

    best = None
    for match in matches:
        if len(match) != len(rc):
            continue
        d = pc[list(match)] - rc
        rmsd = float(np.sqrt((d * d).sum() / len(match)))
        if best is None or rmsd < best:
            best = rmsd
    return best


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _flatten_metrics(dst, metrics):
    if not metrics:
        return
    for key, value in metrics.items():
        if key in _DROP_KEYS:
            continue
        if isinstance(value, (list, tuple, dict)):
            dst[key] = json.dumps(value, default=float)
        else:
            dst[key] = value


def ligand_mmff_strain(mol):
    """MMFF94 internal strain (kcal/mol) of the pose conformer: E(pose) - E(relaxed).

    Force-field-only (no map, no protein). High strain flags a ligand contorted to fit
    fake or wrong density. Returns None if MMFF params are unavailable for the ligand.
    """
    try:
        from rdkit.Chem import AllChem
        m = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(m)
        except Exception:  # noqa: BLE001 - unsanitizable pose -> no strain
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
        ff_r = AllChem.MMFFGetMoleculeForceField(relaxed, props_r) if props_r is not None else None
        if ff_r is None:
            return None
        return e_pose - float(ff_r.CalcEnergy())
    except Exception:  # noqa: BLE001 - never let strain break scoring
        return None


def _protein_clash_atoms(system):
    """(coords Nx3 float, elements N object) of protein heavy atoms, excluding ligand
    residues. Built once and cached on the system (a per-case constant)."""
    cache = getattr(system, "_score_poses_protein_atoms", None)
    if cache is not None:
        return cache
    result = (None, None)
    struct = getattr(getattr(system, "protein", None), "complex_structure", None)
    if struct is not None:
        lig_resnames = set()
        for lig in getattr(system, "ligand", None) or []:
            ls = getattr(lig, "complex_structure", None)
            for res in getattr(ls, "residues", []) or []:
                nm = str(getattr(res, "name", "") or "").upper().strip()
                if nm:
                    lig_resnames.add(nm)
        coords, elems = [], []
        for atom in getattr(struct, "atoms", []) or []:
            el = getattr(atom, "element_name", None) or getattr(atom, "element", None) or ""
            z = getattr(atom, "atomic_number", None)
            if (isinstance(z, int) and z <= 1) or str(el).upper() == "H":
                continue
            resn = str(getattr(getattr(atom, "residue", None), "name", "") or "").upper().strip()
            if resn.startswith("LIG") or resn in lig_resnames:
                continue
            coords.append([atom.xx, atom.xy, atom.xz])
            elems.append(str(el) or "C")
        if coords:
            result = (np.asarray(coords, dtype=float), np.asarray(elems, dtype=object))
    system._score_poses_protein_atoms = result
    return result


def _add_clash_terms(row, system, mol, coords):
    """Protein-ligand (heavy atoms) + ligand self clash penalties, reusing the
    smart_refine_2 clash utilities. Populates *row* in place; never raises."""
    try:
        from ChemEM.protocols.smart_refine_2.optimisers import (
            protein_ligand_clash, ligand_self_clash,
        )
    except Exception:  # noqa: BLE001 - utils unavailable -> skip clash silently
        return
    try:
        lig_coords = np.asarray(coords, dtype=float)
        atoms = list(mol.GetAtoms())
        lig_elems = np.array([a.GetSymbol() for a in atoms], dtype=object)
        heavy = np.array([a.GetAtomicNum() > 1 for a in atoms])
        lc, le = (lig_coords[heavy], lig_elems[heavy]) if heavy.any() else (lig_coords, lig_elems)
        prot_coords, prot_elems = _protein_clash_atoms(system)
        if prot_coords is not None:
            pc = protein_ligand_clash(lc, le, prot_coords, prot_elems)
            row["protein_ligand_clash_penalty"] = float(pc.penalty)
            row["protein_ligand_clash_count"] = int(pc.count)
            row["protein_ligand_max_overlap_A"] = float(pc.max_overlap_A)
        sc = ligand_self_clash(lig_coords, lig_elems, mol=mol)
        row["ligand_self_clash_penalty"] = float(sc.penalty)
        row["ligand_self_clash_count"] = int(sc.count)
    except Exception:  # noqa: BLE001 - never let clash break scoring
        return


def score_pose(system, ctx, site_id, ligand_idx, pose_idx, sdf_path, stage, ligand_obj=None):
    row = {
        "case_id": ctx["case_id"],
        "site_id": site_id,
        "ligand_idx": ligand_idx,
        "pose_idx": pose_idx,
        "stage": stage,
        "sdf_path": sdf_path,
    }
    # Read the pose molecule + coords straight from the SDF — reliable pose coordinates,
    # independent of the config's ligand library or how any conformer was loaded.
    pose_mol, coords, sanitized = load_pose(sdf_path)
    if coords is None or pose_mol is None:
        row["error"] = "coords_read_failed"
        return row
    mol = pose_mol

    props = pose_mol.GetPropsAsDict()
    if "dock_score" in props:
        row["dock_score"] = props["dock_score"]

    # Q-score against the full map (mirrors SmartOrchestrator._score_qscore).
    qm = qscore_pose_metrics(coords, mol, system.density_map, sigma_ref=ctx["sigma_ref"])
    if qm is None:
        row["qscore_failed"] = 1
    else:
        _flatten_metrics(row, qm)

    # Density fit + SCI + shape + skeleton (mirrors SmartOrchestrator._score_density_fit with
    # all diagnostics on). The region scored against depends on --density-region.
    row["density_region"] = ctx["density_region"]
    region_map = region_map_for(system, ctx, site_id, coords, row)
    if region_map is not None:
        try:
            dm = density_fit_metrics(
                coords, mol, region_map,
                threshold_frac=ctx["threshold_frac"],
                compute_sci=True,
                compute_shape=True,
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            dm = None
            row["density_failed"] = type(exc).__name__
        _flatten_metrics(row, dm)

    # Optional single-frame MMGBSA (mirrors SmartOrchestrator._score_mmgbsa). Needs the
    # ligand object for force-field params: the conf-loaded ligand (conf mode) or the
    # library ligand at ligand_idx (audit mode).
    if ctx["do_mmgbsa"]:
        lig = ligand_obj
        if lig is None:
            try:
                lig = system.ligand[ligand_idx]
            except (IndexError, TypeError, KeyError):
                lig = None
        if lig is None:
            row["mmgbsa"] = None
        else:
            ps = mmgbsa_single_frame(
                coords, lig, system.protein,
                pose_idx=pose_idx if pose_idx is not None else 0, resource_owner=system,
                minimise_ligand=ctx["mmgbsa_minimise"], minimise_iters=ctx["mmgbsa_min_iters"],
                minimise_cutoff_A=ctx["mmgbsa_pocket_radius"],
                reuse_cache=True,   # per-case system/Context reuse (bit-identical, much faster)
            )
            if ps is not None:
                row["mmgbsa"] = float(ps.deltaG)
                comps = getattr(ps, "components", None) or {}
                for comp in ("EEL", "VDW", "EGB", "ECAV"):
                    val = comps.get(comp)
                    row[f"mmgbsa_{comp.lower()}"] = float(val) if val is not None else None
                if ctx["mmgbsa_minimise"]:
                    shift = getattr(ps, "min_shift_A", None)
                    row["mmgbsa_min_shift_A"] = float(shift) if shift is not None else None
            else:
                row["mmgbsa"] = None

    # Optional cheap orthogonal terms: ligand internal strain (RDKit MMFF) and protein/self
    # steric clash (reuses smart_refine_2 utils). Robust on raw docked poses, and directly
    # target the two Q-score blind spots (a ligand forced into fake/wrong density is
    # strained/clashy even when its density fit looks fine).
    if ctx["do_strain"]:
        row["ligand_strain"] = ligand_mmff_strain(pose_mol)
    if ctx["do_clash"]:
        _add_clash_terms(row, system, pose_mol, coords)

    # Optional RMSD-to-reference labelling.
    if ctx["ref_mol"] is not None and sanitized:
        rmsd = pose_rmsd_to_reference(pose_mol, ctx["ref_mol"])
        row["rmsd_to_ref"] = rmsd
        row["is_correct"] = int(rmsd is not None and rmsd <= ctx["rmsd_cutoff"])

    return row


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def write_csv(rows, path):
    seen, fields = set(), []
    for key in _LEAD_COLUMNS:
        if key not in seen and any(key in r for r in rows):
            fields.append(key)
            seen.add(key)
    for r in rows:
        for key in r.keys():
            if key not in seen and key != "sdf_path":
                fields.append(key)
                seen.add(key)
    if "sdf_path" in {k for r in rows for k in r}:
        fields.append("sdf_path")  # keep the long path column last

    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)

    system = build_scored_system(args)

    output = getattr(system, "output", None) or getattr(args, "output", None) or "."
    poses_dir = args.poses_dir or os.path.join(output, "audit", "01_gate1_scored", "sdf")
    stage = os.path.basename(os.path.dirname(os.path.normpath(poses_dir))) or "unknown"

    ref_mol = None
    if args.reference:
        supplier = Chem.SDMolSupplier(args.reference, removeHs=False, sanitize=True)
        ref_mol = next((m for m in supplier if m is not None), None)
        if ref_mol is None:
            print(f"[score_poses] WARNING: could not read reference {args.reference}; "
                  "skipping RMSD labels.")

    ctx = {
        "case_id": args.case_id or os.path.basename(os.path.normpath(output)) or "case",
        "sigma_ref": float(getattr(args, "sigma_ref", 0.6) or 0.6),
        "threshold_frac": float(args.score_density_threshold_frac),
        "do_mmgbsa": bool(args.do_mmgbsa),
        "mmgbsa_minimise": bool(args.mmgbsa_minimise),
        "mmgbsa_min_iters": int(args.mmgbsa_min_iters),
        "mmgbsa_pocket_radius": float(args.mmgbsa_pocket_radius),
        "do_strain": bool(args.do_strain),
        "do_clash": bool(args.do_clash),
        "ref_mol": ref_mol,
        "rmsd_cutoff": float(args.rmsd_cutoff),
        "density_region": getattr(args, "density_region", "full"),
        "box_size": float(getattr(args, "box_size", 24.0)),
    }

    rows = []
    if getattr(args, "layout", "audit") == "conf":
        conf_poses = list(iter_conf_poses(system))
        total = len(conf_poses)
        print(f"[score_poses] scoring {total} poses from {ctx['case_id']} "
              f"(mmgbsa={ctx['do_mmgbsa']}, minimise={ctx['mmgbsa_minimise']})", flush=True)
        t_start = time.time()
        for i, (stage, site_id, ligand_idx, pose_idx, sdf_path, ligand_obj) in enumerate(conf_poses, 1):
            t_pose = time.time()
            row = score_pose(system, ctx, site_id, ligand_idx, pose_idx,
                             sdf_path, stage, ligand_obj=ligand_obj)
            rows.append(row)
            mg = row.get("mmgbsa")
            mg = f"{mg:9.1f}" if isinstance(mg, (int, float)) else str(mg)
            print(f"[score_poses]   {i:3d}/{total} [{str(stage):7s}] {time.time()-t_pose:5.1f}s "
                  f"mmgbsa={mg}", flush=True)
        print(f"[score_poses] scored {total} poses in {time.time()-t_start:.0f}s", flush=True)
        source = "config ligand list"
    else:
        for site_id, ligand_idx, pose_idx, sdf_path in iter_pose_files(poses_dir):
            rows.append(score_pose(system, ctx, site_id, ligand_idx, pose_idx, sdf_path, stage))
            if len(rows) % 50 == 0:
                print(f"[score_poses] scored {len(rows)} poses...", flush=True)
        source = poses_dir

    if not rows:
        print(f"[score_poses] no poses found ({source})")
        return 1

    write_csv(rows, args.score_out)
    print(f"[score_poses] wrote {len(rows)} rows -> {args.score_out}"
          f"{' (with MMGBSA)' if args.do_mmgbsa else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
