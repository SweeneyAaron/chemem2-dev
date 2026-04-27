# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

import json
import os
from typing import Any, List

import numpy as np

from ChemEM.tools.ligand import mol_with_positions, write_to_sdf


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def write_ligand_outputs(
    out_root: str,
    lig_idx: int,
    ligand,
    scorer_name: str,
    selected_records: List[dict],
    all_records: List[dict],
) -> None:
    outdir = os.path.join(out_root, f"Ligand_{lig_idx}")
    os.makedirs(outdir, exist_ok=True)

    for rank, rec in enumerate(selected_records, start=1):
        pose_mol = mol_with_positions(
            ligand.mol, np.asarray(rec["ligand_coords_A"], dtype=np.float64)
        )
        write_to_sdf(pose_mol, os.path.join(outdir, f"pose_{rank:03d}.sdf"))

    ranked_all = sorted(all_records, key=lambda r: float(r["final_score"]), reverse=True)
    score_rows = []
    for rank, rec in enumerate(ranked_all, start=1):
        row = {
            "rank": int(rank),
            "outer_iter": int(rec["outer_iter"]),
            "proposal_idx": int(rec["proposal_idx"]),
            "scorer": str(scorer_name),
            "final_score": float(rec["final_score"]),
            "score": float(rec["score"]),
            "energy_kcal": float(rec["energy_kcal"]),
            "clash_penalty": float(rec["clash_penalty"]),
            "score_terms": _jsonable(rec.get("score_terms", {})),
        }
        score_rows.append(row)

    with open(os.path.join(outdir, "scores.json"), "w") as f:
        json.dump(score_rows, f, indent=2)


def write_summary(out_root: str, summaries: List[dict]) -> None:
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(_jsonable(summaries), f, indent=2)
