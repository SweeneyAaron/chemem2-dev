#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-0 objective-faithfulness diagnostic for SmartRefine2.

PURPOSE
-------
The SR2 benchmark regresses flexible ligands (ADP/ATP/AGS phosphate tails,
EUJ, CO8, CRN) badly — sometimes catastrophically (e.g. CO8 4.33 -> 9.73 A).
Before we invest in search-side fixes (drift guards, no-regression gate) we
must know *whether the QScore objective itself is misleading*: i.e. does the
drifted pose score HIGHER than the deposited ground truth against the SAME map?

  - QScore(drifted) > QScore(ground_truth)  -> objective is misleading.
        Search guards alone cannot fix it; a strain / start-proximity term in
        the objective is required (plan Phase 3).
  - QScore(drifted) < QScore(ground_truth)  -> objective is faithful.
        The problem is pure search drift; the no-regression gate + tighter
        drift guards (plan Phases 1-2) are sufficient.

This script is STRICTLY READ-ONLY: it loads a map + protein + ligand poses and
prints their QScores. It does not modify any ChemEM state or write outputs.

FAITHFUL BY CONSTRUCTION
------------------------
It reuses the exact production code path: it builds the `system` from a ChemEM
config file (same as a real run), then constructs a real `RefineLigand` per
candidate pose. `RefineLigand.__init__` builds the local protein context and
calls `update_atom_qscores()` -> `compute_qscores_from_emmap(...)`. The scalar
QScore the SR2 acceptance scorer uses is `mean(per_atom_qscores)` (see
`scorers.QScoreScorer` / `map_metrics.score_qscore`). So the numbers printed
here are the same numbers SR2 optimises.

WHERE TO RUN
------------
The 173-case benchmark MAPS and PROTEINS are NOT on the laptop — they live on
the fitting server (the docked-result JSONs reference /mnt/... paths). RUN THIS
ON THE SERVER, pointing --conf at the per-case ChemEM config used for that EMDB
entry (protein + densmap + resolution + centroid).

USAGE
-----
  python -m ChemEM.tests.diagnostics.qscore_faithfulness \
      --conf /path/to/EMD-60407_site.conf \
      --pose ground_truth=/path/to/deposited_CO8.sdf \
      --pose docked=/path/to/server_minimised_ligands/EMD-60407/Ligand_0.sdf \
      --pose drifted=/path/to/slr2_results/EMD-60407/smart_refine/Ligand_001.sdf

Pass --pose LABEL=PATH as many times as you like. Each SDF is scored
independently on its own atoms (QScore is a per-atom mean, so atom ordering
between files does not matter).

Optionally `--candidate-dirs N` to match the run's --sr2-qscore-candidate-dirs
(default 128, as in RefineLigand).
"""
from __future__ import annotations

import argparse
import sys
import types

import numpy as np


def _load_mol_from_sdf(path: str):
    """Load the first molecule from an SDF, keeping coordinates, no H removal.

    RefineLigand._init_atoms filters hydrogens by element symbol, so we keep
    whatever the file has and let the production code do the heavy-atom
    selection exactly as in a real run.
    """
    from rdkit import Chem

    suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise ValueError(f"No valid molecule parsed from {path}")
    if mol.GetNumConformers() == 0:
        raise ValueError(f"Molecule in {path} has no 3D conformer")
    return mol


def _score_pose(sr2, density_map, sdf_path: str, candidate_dirs: int) -> dict:
    """Build a real RefineLigand for one pose SDF and return its QScore stats.

    Reuses sr2._protein_index (already built by get_protein_complex) so the
    occlusion context is identical to a production run.
    """
    from ChemEM.protocols.smart_refine_2.smart_refine import RefineLigand

    mol = _load_mol_from_sdf(sdf_path)
    # Minimal ligand wrapper: RefineLigand only needs `.mol` off the ligand.
    lig = types.SimpleNamespace(mol=mol, name=sdf_path)

    rl = RefineLigand(
        lig,
        sr2._protein_index,
        density_map,
        qscore_candidate_dirs=candidate_dirs,
    )
    q = np.asarray(rl._per_atom_qscores, dtype=float).ravel()
    return {
        "mean_qscore": float(np.mean(q)) if q.size else float("nan"),
        "min_qscore": float(np.min(q)) if q.size else float("nan"),
        "n_heavy_atoms": int(q.size),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", required=True,
                    help="ChemEM config file for this case (protein + densmap + "
                         "resolution + centroid). Built the same way a real run is.")
    ap.add_argument("--pose", action="append", default=[], metavar="LABEL=PATH",
                    help="A labelled pose SDF, e.g. drifted=/path/Ligand_001.sdf. "
                         "Repeat for ground_truth / docked / drifted.")
    ap.add_argument("--candidate-dirs", type=int, default=128,
                    help="QScore candidate directions (match --sr2-qscore-candidate-dirs).")
    args = ap.parse_args(argv)

    if not args.pose:
        ap.error("provide at least one --pose LABEL=PATH (typically ground_truth, "
                 "docked and drifted)")

    poses = []
    for spec in args.pose:
        if "=" not in spec:
            ap.error(f"--pose must be LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        poses.append((label.strip(), path.strip()))

    # Build the production system exactly as `chemem <conf>` does.
    from ChemEM.config import Config
    from ChemEM.protocols.smart_refine_2.smart_refine import SmartRefine2

    system = Config().load_config(args.conf)
    if getattr(system, "density_map", None) is None:
        print("ERROR: system has no density_map — QScore cannot be computed. "
              "Check the conf's densmap/resolution.", file=sys.stderr)
        return 2

    sr2 = SmartRefine2(system)
    sr2.get_protein_complex()  # builds sr2._protein_index (occlusion context)

    print(f"# conf: {args.conf}")
    print(f"# candidate_dirs: {args.candidate_dirs}")
    print(f"{'label':<16}{'mean_Q':>10}{'min_Q':>10}{'n_heavy':>10}  sdf")
    results = {}
    for label, path in poses:
        try:
            stats = _score_pose(sr2, system.density_map, path, args.candidate_dirs)
        except Exception as exc:  # noqa: BLE001 - diagnostic, report and continue
            print(f"{label:<16}{'ERR':>10}{'':>10}{'':>10}  {path}  ({exc})")
            continue
        results[label] = stats["mean_qscore"]
        print(f"{label:<16}{stats['mean_qscore']:>10.4f}{stats['min_qscore']:>10.4f}"
              f"{stats['n_heavy_atoms']:>10}  {path}")

    # Verdict, if we have both reference labels.
    gt = results.get("ground_truth")
    drifted = results.get("drifted")
    if gt is not None and drifted is not None:
        print()
        if drifted > gt:
            print(f"VERDICT: objective MISLEADING — QScore(drifted)={drifted:.4f} > "
                  f"QScore(ground_truth)={gt:.4f}. A strain/start-proximity term is "
                  f"needed (plan Phase 3), not just search guards.")
        else:
            print(f"VERDICT: objective FAITHFUL — QScore(drifted)={drifted:.4f} <= "
                  f"QScore(ground_truth)={gt:.4f}. No-regression gate + tighter drift "
                  f"guards (plan Phases 1-2) should suffice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
