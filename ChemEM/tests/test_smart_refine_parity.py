"""SmartRefine2 parity-test scaffold.

Phase-A deliverable for the smart_refine_2 performance work. Detects
regressions as the Q-score / clash / parallelisation work lands.

Workflow
========

1) Capture a baseline by running the real ChemEM protocol on your config
   with the parity-snapshot env var set:

       CHEMEM_SR2_PARITY_DIR=test/test_output/smart_refine_2/baseline_<sha> \\
       chemem 7jjo_conf.txt -slr2

   This drops one ``ligand_NNN/`` subdir per refined ligand containing
   ``per_atom_qscores.npy``, ``final_coords_A.npy``, and ``scalars.json``.

2) Profile the same run if you want a cProfile trace too:

       CHEMEM_SR2_PARITY_DIR=test/test_output/smart_refine_2/baseline_<sha> \\
       python -m cProfile -o smart_refine_2.prof -m ChemEM 7jjo_conf.txt -slr2
       # view with: snakeviz smart_refine_2.prof   (pip install snakeviz)

3) After a kernel rewrite (Phase B / C / D), capture a candidate run into
   a different directory, then compare:

       python -m ChemEM.tests.test_smart_refine_parity compare \\
           --baseline test/test_output/smart_refine_2/baseline_<sha> \\
           --candidate test/test_output/smart_refine_2/candidate_<sha>

4) The pytest path auto-skips unless both env vars are set:

       CHEMEM_PARITY_BASELINE_DIR=... CHEMEM_PARITY_CANDIDATE_DIR=... \\
       pytest ChemEM/tests/test_smart_refine_parity.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest


DEFAULT_TOLERANCES = {
    # Phase A: defaults preserve current behaviour at 256 dirs, modulo the
    # new 128-dirs Q-score default (mean drift typically < 0.02).
    "qscore_max_abs": 0.02,
    "rmsd_max_A": 0.05,
    "clash_penalty_max_abs": 1e-3,
}


def save_baseline(
    out_dir: str | os.PathLike,
    *,
    per_atom_qscores: np.ndarray,
    final_coords_A: np.ndarray,
    clash_penalty: float,
    clash_count: int,
    meta: dict[str, Any] | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "per_atom_qscores.npy", np.asarray(per_atom_qscores, dtype=np.float64))
    np.save(out / "final_coords_A.npy", np.asarray(final_coords_A, dtype=np.float64))
    with (out / "scalars.json").open("w") as fh:
        json.dump(
            {
                "clash_penalty": float(clash_penalty),
                "clash_count": int(clash_count),
                "meta": meta or {},
            },
            fh,
            indent=2,
        )


def load_baseline(baseline_dir: str | os.PathLike) -> dict[str, Any]:
    b = Path(baseline_dir)
    scalars = json.loads((b / "scalars.json").read_text())
    return {
        "per_atom_qscores": np.load(b / "per_atom_qscores.npy"),
        "final_coords_A": np.load(b / "final_coords_A.npy"),
        "clash_penalty": float(scalars["clash_penalty"]),
        "clash_count": int(scalars["clash_count"]),
        "meta": scalars.get("meta", {}),
    }


def _ligand_subdirs(root: str | os.PathLike) -> list[Path]:
    r = Path(root)
    if (r / "scalars.json").exists():
        return [r]
    return sorted(p for p in r.iterdir() if p.is_dir() and (p / "scalars.json").exists())


def compare_to_baseline(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return a structured diff. Caller asserts on the ``passed`` field."""
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}

    q_b = np.asarray(baseline["per_atom_qscores"], dtype=np.float64).ravel()
    q_c = np.asarray(candidate["per_atom_qscores"], dtype=np.float64).ravel()
    coords_b = np.asarray(baseline["final_coords_A"], dtype=np.float64).reshape(-1, 3)
    coords_c = np.asarray(candidate["final_coords_A"], dtype=np.float64).reshape(-1, 3)

    diff: dict[str, Any] = {}

    if q_b.shape != q_c.shape:
        diff["qscore_shape_mismatch"] = (q_b.shape, q_c.shape)
        q_abs = float("inf")
    else:
        q_abs = float(np.max(np.abs(q_b - q_c)))
    diff["qscore_max_abs"] = q_abs
    diff["qscore_passed"] = q_abs <= tol["qscore_max_abs"]

    if coords_b.shape != coords_c.shape:
        diff["coords_shape_mismatch"] = (coords_b.shape, coords_c.shape)
        rmsd = float("inf")
    else:
        rmsd = float(np.sqrt(np.mean(np.sum((coords_b - coords_c) ** 2, axis=1))))
    diff["rmsd_A"] = rmsd
    diff["rmsd_passed"] = rmsd <= tol["rmsd_max_A"]

    clash_delta = abs(float(baseline["clash_penalty"]) - float(candidate["clash_penalty"]))
    diff["clash_penalty_abs"] = clash_delta
    diff["clash_passed"] = clash_delta <= tol["clash_penalty_max_abs"]

    diff["passed"] = bool(
        diff["qscore_passed"] and diff["rmsd_passed"] and diff["clash_passed"]
    )
    diff["tolerances"] = tol
    return diff


def compare_directories(
    baseline_root: str | os.PathLike,
    candidate_root: str | os.PathLike,
    *,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    base_subs = _ligand_subdirs(baseline_root)
    cand_subs = _ligand_subdirs(candidate_root)
    if len(base_subs) != len(cand_subs):
        return {
            "passed": False,
            "error": f"ligand count mismatch: baseline={len(base_subs)} candidate={len(cand_subs)}",
        }
    per_ligand = []
    overall_pass = True
    for b, c in zip(base_subs, cand_subs):
        diff = compare_to_baseline(load_baseline(b), load_baseline(c), tolerances=tolerances)
        diff["baseline_dir"] = str(b)
        diff["candidate_dir"] = str(c)
        overall_pass = overall_pass and bool(diff["passed"])
        per_ligand.append(diff)
    return {"passed": overall_pass, "per_ligand": per_ligand}


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    cmp = sub.add_parser(
        "compare",
        help="Compare two snapshot directories captured via CHEMEM_SR2_PARITY_DIR.",
    )
    cmp.add_argument("--baseline", required=True, help="Baseline snapshot directory")
    cmp.add_argument("--candidate", required=True, help="Candidate snapshot directory")

    args = p.parse_args()
    if args.cmd == "compare":
        report = compare_directories(args.baseline, args.candidate)
        print(json.dumps(report, indent=2, default=float))
        raise SystemExit(0 if report["passed"] else 1)


@pytest.mark.skipif(
    "CHEMEM_PARITY_BASELINE_DIR" not in os.environ
    or "CHEMEM_PARITY_CANDIDATE_DIR" not in os.environ,
    reason="Set CHEMEM_PARITY_BASELINE_DIR + CHEMEM_PARITY_CANDIDATE_DIR to enable parity test.",
)
def test_smart_refine_2_parity():
    report = compare_directories(
        os.environ["CHEMEM_PARITY_BASELINE_DIR"],
        os.environ["CHEMEM_PARITY_CANDIDATE_DIR"],
    )
    assert report["passed"], json.dumps(report, indent=2, default=float)


if __name__ == "__main__":
    _main()
