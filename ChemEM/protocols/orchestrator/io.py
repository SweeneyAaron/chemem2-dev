# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Output writers for SmartOrchestrator.

Per-gate JSON audit (gate1/gate2/gate3.json) lets a user retrace why each
final assignment was picked. final_complex.pdb is the assembled answer:
receptor + every assigned ligand pose, with distinct chain IDs when the
same ligand is placed in multiple sites.
"""

from __future__ import annotations

import copy
import csv
import json
import os
from typing import Dict, List

import numpy as np
from openmm import unit
from rdkit import Chem
from rdkit.Geometry import Point3D

from ChemEM.parsers.writers import save_structure_parmed, write_to_sdf

from .state import FinalAssignment, PoseCandidate


def _json_default(value):
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def write_gate_json(
    candidates_by_site: Dict[str, List[PoseCandidate]],
    path: str,
) -> None:
    payload = {
        str(site_id): [c.to_dict() for c in cands]
        for site_id, cands in candidates_by_site.items()
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def write_assignments_json(assignments: List[FinalAssignment], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([a.to_dict() for a in assignments], f, indent=2, default=_json_default)


def write_summary_json(
    assignments: List[FinalAssignment],
    gate_counts: Dict[str, Dict[str, int]],
    path: str,
    *,
    rejections=None,
    assignment_eval_summary: dict | None = None,
) -> None:
    payload = {
        "gate_counts": gate_counts,
        "assignments": [a.to_dict() for a in assignments],
    }
    if rejections is not None:
        payload["rejections"] = [
            r.to_dict() if hasattr(r, "to_dict") else dict(r)
            for r in rejections
        ]
    if assignment_eval_summary is not None:
        payload["assignment_eval"] = assignment_eval_summary
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _ligand_with_coords(ligand, coords) -> Chem.Mol:
    """Return a deep-copy of ligand.mol with conformer 0 set to coords."""
    rdmol = Chem.Mol(ligand.mol)
    conf = rdmol.GetConformer(0)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return rdmol


def _set_sdf_props(mol: Chem.Mol, values: dict) -> Chem.Mol:
    for key, value in values.items():
        if key == "coords":
            continue
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            try:
                value = json.dumps(value, default=_json_default)
            except Exception:
                value = str(value)
        mol.SetProp(str(key), str(value))
    return mol


def _flatten_candidate(candidate: PoseCandidate, *, rank: int | None = None, sdf_path: str | None = None) -> dict:
    row = candidate.to_dict()
    row["rank"] = rank
    row["sdf_path"] = sdf_path
    for key, value in candidate.metrics.items():
        if key == "q_per_atom" or key == "density_feature_metrics":
            continue
        row[key] = value
    for key, value in candidate.refine_metrics.items():
        row[f"refine_{key}"] = value
    return row


def _candidate_sort_key(candidate: PoseCandidate) -> tuple:
    score = candidate.rank_score
    try:
        score = float(score)
    except Exception:
        score = float("-inf")
    if not np.isfinite(score):
        score = float("-inf")
    mm = candidate.mmgbsa if candidate.mmgbsa is not None else float("inf")
    return (str(candidate.site_id), -score, mm, int(candidate.ligand_idx), int(candidate.pose_idx))


def _candidate_ranks(candidates_by_site: Dict[str, List[PoseCandidate]]) -> dict[int, int]:
    ranks = {}
    for _site_id, candidates in candidates_by_site.items():
        ordered = sorted(candidates, key=_candidate_sort_key)
        for rank, candidate in enumerate(ordered, start=1):
            ranks[id(candidate)] = rank
    return ranks


def _write_scores_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base = [
        "audit_stage",
        "stage",
        "site_id",
        "ligand_idx",
        "pose_idx",
        "rank",
        "rank_score",
        "assignment_score",
        "assignment_margin",
        "dock_score",
        "qscore",
        "q_low_tail",
        "density_coverage",
        "density_precision",
        "density_ccc",
        "density_overlap",
        "density_envelope_iou",
        "density_excess_fraction",
        "density_mi",
        "density_normalized_mi",
        "density_mi_nbins",
        "density_sci",
        "density_sci_cc0",
        "density_sci_ccx",
        "density_sci_ccy",
        "density_sci_ccz",
        "density_sci_ccxx",
        "density_sci_ccyy",
        "density_sci_cczz",
        "shape_zernike_similarity",
        "shape_zernike_distance",
        "shape_spharm_similarity",
        "shape_spharm_distance",
        "shape_radial_profile_similarity",
        "shape_volume_ratio",
        "shape_exp_component_count",
        "shape_selected_component_fraction",
        "shape_selected_component_mode",
        "shape_zernike_backend",
        "density_skeleton_precision",
        "density_skeleton_recall",
        "density_skeleton_f1",
        "density_skeleton_mean_distance_A",
        "density_skeleton_unmatched_density_length_A",
        "density_skeleton_unmatched_ligand_length_A",
        "density_skeleton_exp_branch_count",
        "density_skeleton_sim_branch_count",
        "density_skeleton_branch_count_diff",
        "density_skeleton_exp_point_count",
        "density_skeleton_sim_point_count",
        "density_skeleton_component_mode",
        "density_skeleton_backend",
        "mmgbsa",
        "rejection_reason",
        "label",
        "sdf_path",
        "notes",
    ]
    extra = sorted({key for row in rows for key in row.keys()} - set(base) - {"coords"})
    fields = [key for key in base if any(key in row for row in rows)] + extra
    if not fields:
        fields = base
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, default=_json_default)
                out[key] = value
            writer.writerow(out)


def write_audit_candidates(
    candidates_by_site: Dict[str, List[PoseCandidate]],
    ligands,
    audit_dir: str,
    stage_name: str,
    *,
    include_sdfs: bool = True,
) -> None:
    stage_dir = os.path.join(audit_dir, stage_name)
    sdf_dir = os.path.join(stage_dir, "sdf")
    os.makedirs(stage_dir, exist_ok=True)
    if include_sdfs:
        os.makedirs(sdf_dir, exist_ok=True)

    ranks = _candidate_ranks(candidates_by_site)
    rows = []
    payload = {
        "stage": stage_name,
        "candidates": [],
    }
    for site_id in sorted(candidates_by_site.keys(), key=str):
        for candidate in sorted(candidates_by_site[site_id], key=_candidate_sort_key):
            sdf_path = None
            row = _flatten_candidate(
                candidate,
                rank=ranks.get(id(candidate)),
                sdf_path=None,
            )
            row["audit_stage"] = stage_name
            row["stage"] = candidate.stage
            if include_sdfs:
                ligand = ligands[candidate.ligand_idx]
                site_dir = os.path.join(sdf_dir, f"site_{candidate.site_id}")
                os.makedirs(site_dir, exist_ok=True)
                sdf_path = os.path.join(
                    site_dir,
                    f"ligand_{candidate.ligand_idx}_pose_{candidate.pose_idx}.sdf",
                )
                row["sdf_path"] = sdf_path
                rdmol = _ligand_with_coords(ligand, candidate.coords)
                _set_sdf_props(rdmol, row)
                write_to_sdf(rdmol, sdf_path)
            rows.append(row)
            payload["candidates"].append(row)

    with open(os.path.join(stage_dir, "scores.json"), "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    _write_scores_csv(rows, os.path.join(stage_dir, "scores.csv"))


def write_audit_assignments(
    assignments: List[FinalAssignment],
    ligands,
    audit_dir: str,
    stage_name: str = "08_assignments",
    *,
    include_sdfs: bool = True,
) -> None:
    stage_dir = os.path.join(audit_dir, stage_name)
    sdf_dir = os.path.join(stage_dir, "sdf")
    os.makedirs(stage_dir, exist_ok=True)
    if include_sdfs:
        os.makedirs(sdf_dir, exist_ok=True)

    rows = []
    for rank, assignment in enumerate(assignments, start=1):
        row = assignment.to_dict()
        row["audit_stage"] = stage_name
        row["rank"] = rank
        row["sdf_path"] = None
        for key, value in assignment.metrics.items():
            if key == "q_per_atom" or key == "density_feature_metrics":
                continue
            row[key] = value
        if include_sdfs:
            ligand = ligands[assignment.ligand_idx]
            site_dir = os.path.join(sdf_dir, f"site_{assignment.site_id}")
            os.makedirs(site_dir, exist_ok=True)
            sdf_path = os.path.join(
                site_dir,
                f"ligand_{assignment.ligand_idx}_pose_{assignment.pose_idx}_chain_{assignment.chain_id}.sdf",
            )
            row["sdf_path"] = sdf_path
            rdmol = _ligand_with_coords(ligand, assignment.coords)
            _set_sdf_props(rdmol, row)
            write_to_sdf(rdmol, sdf_path)
        rows.append(row)

    with open(os.path.join(stage_dir, "scores.json"), "w") as f:
        json.dump({"stage": stage_name, "assignments": rows}, f, indent=2, default=_json_default)
    _write_scores_csv(rows, os.path.join(stage_dir, "scores.csv"))


def write_assignment_eval(
    rows: list[dict],
    summary: dict,
    audit_dir: str,
    stage_name: str = "09_assignment_eval",
) -> None:
    stage_dir = os.path.join(audit_dir, stage_name)
    os.makedirs(stage_dir, exist_ok=True)
    payload = {
        "stage": stage_name,
        "summary": summary,
        "rows": rows,
    }
    with open(os.path.join(stage_dir, "scores.json"), "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    _write_scores_csv(rows, os.path.join(stage_dir, "scores.csv"))


def _iter_site_maps(site_maps):
    if site_maps is None:
        return []
    if hasattr(site_maps, "density_map"):
        return [(0, site_maps)]
    if isinstance(site_maps, dict):
        values = list(site_maps.values())
    elif isinstance(site_maps, (list, tuple)):
        values = list(site_maps)
    else:
        values = [site_maps]

    out = []
    for idx, item in enumerate(values):
        emmap = item
        if isinstance(item, (list, tuple)) and item:
            emmap = item[0]
        if hasattr(emmap, "density_map"):
            out.append((idx, emmap))
    return out


def _safe_path_part(value) -> str:
    text = str(value)
    return "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text
    )


def _array_like_list(value):
    try:
        return [float(v) for v in np.asarray(value, dtype=float).reshape(-1).tolist()]
    except Exception:
        return None


def write_audit_maps(binding_site_maps, audit_dir: str) -> None:
    """Persist the exact feature maps used by orchestrator density scoring."""
    maps_dir = os.path.join(audit_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    manifest = {
        "description": (
            "Feature maps from system.binding_site_maps used for orchestrator "
            "density scoring."
        ),
        "maps": [],
    }

    for site_id in sorted((binding_site_maps or {}).keys(), key=str):
        site_maps = binding_site_maps[site_id]
        site_dir = os.path.join(maps_dir, f"site_{_safe_path_part(site_id)}")
        os.makedirs(site_dir, exist_ok=True)
        for feature_idx, emmap in _iter_site_maps(site_maps):
            rel_path = os.path.join(
                "maps",
                f"site_{_safe_path_part(site_id)}",
                f"feature_{int(feature_idx)}.mrc",
            )
            abs_path = os.path.join(audit_dir, rel_path)
            entry = {
                "site_id": str(site_id),
                "feature_idx": int(feature_idx),
                "path": rel_path,
                "shape": [
                    int(v)
                    for v in np.asarray(getattr(emmap, "density_map")).shape
                ],
                "origin": _array_like_list(getattr(emmap, "origin", None)),
                "apix": _array_like_list(getattr(emmap, "apix", None)),
                "resolution": getattr(emmap, "resolution", None),
                "written": False,
            }
            try:
                if not hasattr(emmap, "write_mrc"):
                    raise AttributeError("map object has no write_mrc method")
                emmap.write_mrc(abs_path)
                entry["written"] = True
            except Exception as exc:
                entry["error"] = type(exc).__name__
            manifest["maps"].append(entry)

    with open(os.path.join(maps_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)


def write_assignment_sdfs(
    assignments: List[FinalAssignment],
    ligands,
    out_dir: str,
) -> None:
    """One SDF per assignment under <out_dir>/Ligand_{i}/site_{s}_chain_{c}.sdf."""
    for a in assignments:
        ligand = ligands[a.ligand_idx]
        sub = os.path.join(out_dir, f"Ligand_{a.ligand_idx}")
        os.makedirs(sub, exist_ok=True)
        rdmol = _ligand_with_coords(ligand, a.coords)
        sdf_path = os.path.join(
            sub, f"site_{a.site_id}_chain_{a.chain_id}.sdf"
        )
        write_to_sdf(rdmol, sdf_path)


def write_final_complex_pdb(
    assignments: List[FinalAssignment],
    system,
    path: str,
) -> None:
    """Write receptor + every assigned ligand pose into one PDB.

    Each ligand copy gets a distinct chain ID (FinalAssignment.chain_id)
    so the same ligand placed in multiple sites is unambiguous in the file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    assembled = copy.deepcopy(system.protein.complex_structure)

    for a in assignments:
        ligand = system.ligand[a.ligand_idx]
        lig_struct = copy.deepcopy(ligand.complex_structure)
        # Plant the assignment coordinates onto the ligand structure copy.
        for atom, (x, y, z) in zip(lig_struct.atoms, a.coords):
            atom.xx, atom.xy, atom.xz = float(x), float(y), float(z)
        # Stamp the chain ID so duplicate ligands are distinguishable.
        for residue in lig_struct.residues:
            residue.chain = a.chain_id
        for atom in lig_struct.atoms:
            atom.residue.chain = a.chain_id
        assembled = assembled + lig_struct

    save_structure_parmed(assembled, path)
