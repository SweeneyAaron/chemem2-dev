"""Compare wrong-torsion input, branch-rebuilt output, and ground-truth SDFs.

Run after each branch-rebuilder change:
    conda activate chemem2-run
    python ChemEM/tests/smoke_branch_rebuild.py [--refined PATH]

Pass criterion (after Steps 2+ are enabled):
    - heavy-atom RMSD(refined -> truth) <= heavy-atom RMSD(wrong -> truth)
    - at least one rotatable torsion has |refined - truth| < 30 deg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

DEFAULT_WRONG = (
    "/Users/aaron.sweeney/Documents/chemem2_build/ChemEM2_build_sep_11/test/"
    "test_output/docking/binding_site_0_Ligand_0/Ligand_1.sdf"
)
DEFAULT_TRUTH = (
    "/Users/aaron.sweeney/Documents/chemem2_build/ChemEM2_feb26/chemem2-dev/"
    "test/Ligand_dep.sdf"
)
DEFAULT_REFINED = (
    "/Users/aaron.sweeney/Documents/chemem2_build/ChemEM2_feb26/chemem2-dev/"
    "slr_targeted_branch_test_raw/smart_ligand_refine/Ligand_0/smart_refined.sdf"
)


def load(path: str):
    mol = Chem.MolFromMolFile(path, removeHs=False)
    if mol is None:
        sys.exit(f"could not load: {path}")
    if mol.GetNumConformers() == 0:
        sys.exit(f"no conformer in: {path}")
    return mol


def best_rmsd(probe, ref):
    try:
        return AllChem.GetBestRMS(probe, ref)
    except Exception:
        # GetBestRMS needs identical graphs; fall back to ordered RMSD on heavy atoms
        pc = probe.GetConformer().GetPositions()
        rc = ref.GetConformer().GetPositions()
        n = min(pc.shape[0], rc.shape[0])
        d = pc[:n] - rc[:n]
        return float((d * d).sum() / n) ** 0.5


def rotatable_dihedrals(truth, *probes):
    """Yield (i,j,k,l, [deg_per_probe], deg_truth) for each non-ring rotatable bond."""
    patt = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    seen = set()
    for i, j in truth.GetSubstructMatches(patt):
        bond = truth.GetBondBetweenAtoms(int(i), int(j))
        if bond is None or bond.GetIsAromatic() or bond.IsInRing():
            continue
        nbrs_i = [a.GetIdx() for a in truth.GetAtomWithIdx(int(i)).GetNeighbors() if a.GetIdx() != j]
        nbrs_j = [a.GetIdx() for a in truth.GetAtomWithIdx(int(j)).GetNeighbors() if a.GetIdx() != i]
        if not nbrs_i or not nbrs_j:
            continue
        a, d = int(nbrs_i[0]), int(nbrs_j[0])
        key = (a, int(i), int(j), d)
        if key in seen:
            continue
        seen.add(key)
        try:
            t = rdMolTransforms.GetDihedralDeg(truth.GetConformer(), a, int(i), int(j), d)
        except Exception:
            continue
        probe_vals = []
        for p in probes:
            try:
                probe_vals.append(rdMolTransforms.GetDihedralDeg(p.GetConformer(), a, int(i), int(j), d))
            except Exception:
                probe_vals.append(float("nan"))
        yield (a, int(i), int(j), d, probe_vals, t)


def angle_delta(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrong", default=DEFAULT_WRONG)
    ap.add_argument("--truth", default=DEFAULT_TRUTH)
    ap.add_argument("--refined", default=DEFAULT_REFINED)
    args = ap.parse_args()

    if not Path(args.refined).exists():
        print(f"[warn] refined SDF missing: {args.refined}")
        print("       run smart_ligand_refine first; reporting wrong-vs-truth only.")
        mw = load(args.wrong)
        mt = load(args.truth)
        print(f"RMSD wrong->truth   : {best_rmsd(mw, mt):.4f} A")
        return 0

    mw = load(args.wrong)
    mt = load(args.truth)
    mr = load(args.refined)

    rmsd_w = best_rmsd(mw, mt)
    rmsd_r = best_rmsd(mr, mt)
    print(f"RMSD wrong->truth   : {rmsd_w:.4f} A")
    print(f"RMSD refined->truth : {rmsd_r:.4f} A")
    print(f"RMSD delta           : {rmsd_w - rmsd_r:+.4f} A   (positive = improvement)")

    print()
    print(f"{'a-i-j-d':<18s} {'wrong':>8s} {'refined':>10s} {'truth':>8s} "
          f"{'|w-t|':>7s} {'|r-t|':>7s} {'verdict':>10s}")
    n_wrong = 0
    n_fixed = 0
    for a, i, j, d, (dw, dr), dt in rotatable_dihedrals(mt, mw, mr):
        delta_wt = angle_delta(dw, dt)
        delta_rt = angle_delta(dr, dt)
        verdict = ""
        if delta_wt > 30.0:
            n_wrong += 1
            if delta_rt < 30.0:
                verdict = "FIXED"
                n_fixed += 1
            elif delta_rt + 5.0 < delta_wt:
                verdict = "improved"
            else:
                verdict = "stuck"
        elif delta_rt > delta_wt + 5.0:
            verdict = "regressed"
        print(f"{a:>3d}-{i:>3d}-{j:>3d}-{d:>3d}  {dw:>8.1f} {dr:>10.1f} {dt:>8.1f}  "
              f"{delta_wt:>7.1f} {delta_rt:>7.1f}  {verdict:>10s}")

    print()
    print(f"Wrong torsions in input  : {n_wrong}")
    print(f"  fixed by refinement    : {n_fixed}")
    if n_wrong > 0:
        ok = n_fixed > 0 and rmsd_r < rmsd_w
        print("Pass:", "YES" if ok else "NO")
        return 0 if ok else 1
    print("(no torsions were >30 deg from truth in the input — nothing to fix)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
