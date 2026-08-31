#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""Row assembly and output writing for the ``--score`` protocol.

One CSV, one row per pose, columns grouped by scorer in ``--score-with`` order. The
column names are deliberately *not* re-prefixed: ``echo_total``, ``qscore``,
``density_*`` and ``mmgbsa*`` keep the exact spellings the old separate tools used,
so the benchmark scripts that read them keep working.
"""

from __future__ import annotations

import csv
import json
import os

from rdkit import Chem

from .poses import source_stem

#: Identity columns, always first and always in this order.
IDENTITY_COLUMNS = (
    "case_id", "ligand", "source", "pose", "ligand_idx", "conf_id", "site_id",
)

#: Keys that never reach the CSV: the nested per-atom / per-feature payloads a flat
#: table cannot hold. They go to pose_scores.json instead.
INTERNAL_KEYS = ("_json",)


def _is_status(key):
    """Status columns are collected at the end regardless of who emitted them.

    They would otherwise be swept into a scorer's detail block by the name prefix --
    ``qscore_failed`` starts with ``qscore_`` -- which buries them in the middle of
    a hundred numeric columns.
    """
    return key == "error" or key.endswith("_error") or key.endswith("_failed")


def build_fieldnames(scorers, rows):
    """Deterministic CSV column order.

    identity, then each scorer's headline, then each scorer's detail block (declared
    columns first, then anything it emitted at runtime), then the status columns.

    Runtime-discovered keys are appended rather than dropped, so a kernel gaining a
    metric shows up without a change here. The status columns are emitted for every
    scorer whether or not anything failed, so the header does not change shape
    between a clean run and a run with one bad pose.
    """
    fields: list[str] = []
    seen: set[str] = set()

    def take(name):
        if name not in seen:
            seen.add(name)
            fields.append(name)

    for name in IDENTITY_COLUMNS:
        take(name)

    # Headline block: the one number per scorer, side by side and easy to eyeball.
    for scorer in scorers:
        if scorer.HEADLINE:
            take(scorer.HEADLINE)

    present = sorted({key for row in rows for key in row})

    for scorer in scorers:
        for name in tuple(scorer.COLUMNS) + tuple(scorer.extra_columns()):
            take(name)
        # Anything this scorer produced that it did not declare. Attributed by name
        # prefix, which is why the column names carry their scorer's name.
        prefixes = tuple({scorer.NAME + "_", (scorer.HEADLINE or scorer.NAME) + "_"})
        for key in present:
            if key in seen or key in INTERNAL_KEYS or _is_status(key):
                continue
            if key.startswith(prefixes):
                take(key)

    # Anything nobody claimed, in first-seen order.
    for row in rows:
        for key in row:
            if key in seen or key in INTERNAL_KEYS or _is_status(key):
                continue
            take(key)

    # Status columns last: the protocol's own, then one per scorer, then whatever
    # the scorers invented (`qscore_failed` and friends).
    take("error")
    for scorer in scorers:
        take(f"{scorer.NAME}_error")
    for key in present:
        if _is_status(key):
            take(key)

    return fields


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "x".join(str(v) for v in value)
    return value


def write_csv(path, scorers, rows, log=None):
    fields = build_fieldnames(scorers, rows)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _csv_value(row.get(c, "")) for c in fields})
    if log:
        log(f"[score] wrote {path} ({len(rows)} poses, {len(fields)} columns)")
    return path


def write_json(path, rows, log=None):
    """Full scores, including the values a flat CSV cannot hold.

    Nested by ligand identifier then conformer, mirroring the shape the old
    ``mapq_scores.json`` used, so a reader keyed that way needs minimal changes.
    """
    payload: dict = {}
    for row in rows:
        entry = {k: v for k, v in row.items() if k not in INTERNAL_KEYS}
        entry.update(row.get("_json", {}))
        payload.setdefault(str(row.get("ligand", "")), {})[
            f"conf_{row.get('conf_id')}"
        ] = entry
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=4, default=str)
    if log:
        log(f"[score] wrote {path}")
    return path


def write_run_json(path, payload, log=None):
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=4, default=str)
    if log:
        log(f"[score] wrote {path}")
    return path


def write_sdfs(output, system, rows, rank_by, higher_is_better, log=None):
    """One SDF per input source, poses best-first, scores as SD properties.

    Grouped by source rather than by ligand index on purpose: a multi-record poses
    SDF is loaded as one ``Ligand`` *per record*, so grouping by ligand would emit a
    file per pose -- all to the same filename.
    """
    by_source: dict = {}
    for row in rows:
        if row.get("error"):
            continue
        by_source.setdefault(row.get("source", ""), []).append(row)

    written = []
    for source, src_rows in by_source.items():
        rankable = [r for r in src_rows if isinstance(r.get(rank_by), (int, float))]
        unrankable = [r for r in src_rows if r not in rankable]
        rankable.sort(key=lambda r: r[rank_by], reverse=bool(higher_is_better))
        ordered = rankable + unrankable

        name = source_stem(source, ordered[0]["ligand_idx"])
        path = os.path.join(output, f"{name}_scored.sdf")
        with Chem.SDWriter(path) as writer:
            for rank, row in enumerate(ordered):
                ligand = system.ligand[row["ligand_idx"]]
                mol = Chem.Mol(ligand.mol)
                mol.SetProp("_Name", f"{name}_pose_{row['pose']}")
                mol.SetIntProp("score_rank", rank)
                for key, value in row.items():
                    if key in INTERNAL_KEYS or value == "" or value is None:
                        continue
                    if key in ("ligand_idx", "conf_id", "pose"):
                        mol.SetIntProp(key, int(value))
                    elif isinstance(value, bool):
                        mol.SetProp(key, str(value))
                    elif isinstance(value, float):
                        mol.SetDoubleProp(key, float(value))
                    else:
                        mol.SetProp(key, str(value))
                writer.write(mol, confId=row["conf_id"])
        written.append(path)
        if log:
            log(f"[score] wrote {path} ({len(ordered)} poses, ranked by {rank_by})")
    return written
