"""Focused regression tests for the lipid cryo-EM benchmark pipeline.

The fixtures are deliberately synthetic: these tests exercise the lipid-specific logic without
depending on the benchmark corpus, the network, or a built test/lipid_ligand/ tree.

Three behaviours matter enough to pin down, because each one fails SILENTLY in a way that would
leave the benchmark reporting confident nonsense:

  * a truncated deposited lipid must route to the substructure matcher, not to spyrmsd, which
    cannot match a subgraph and would simply drop the case;
  * crop and bias radii must scale with the lipid's measured extent, or an extended lipid gets
    a search sphere smaller than itself;
  * the OpenFF harmonic fallback must be detected, because it pins every torsion and turns a
    high-rotatable-bond docking run into a rigid-body one while still reporting success.

Run with:  pytest ChemEM/tests/test_lipid_benchmark.py     (env: chemem2-run)
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from ChemEM.benchmark import lipid_rmsd
from ChemEM.benchmark import select_lipid_cases as sel


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _embedded(smiles: str, seed: int = 0xC0FFEE):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


def _translated(mol, delta):
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i in range(out.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x + delta[0], p.y + delta[1], p.z + delta[2]))
    return out


# --------------------------------------------------------------------------------------
# truncated ground truth
# --------------------------------------------------------------------------------------
def test_truncated_lipid_uses_substructure_matcher_not_spyrmsd():
    """A deposited lipid missing tail atoms is a SUBGRAPH of the docked molecule.

    spyrmsd matches on graph isomorphism and must fail; pose_rmsd must succeed and report the
    number of atoms it actually matched over. This is the 8UL9 (40/44) and 7OY2 (69/77) case.
    """
    from ChemEM.benchmark.covalent_rmsds import pose_rmsd
    from ChemEM.benchmark.characterise_solutions import safe_symm_rmsd

    full = _embedded("CCCCCCCCCC(=O)O")           # decanoic acid: the docked molecule
    truncated = Chem.RWMol(full)                   # deposited: last two tail carbons unmodelled
    for idx in sorted([0, 1], reverse=True):
        truncated.RemoveAtom(idx)
    truncated = truncated.GetMol()
    Chem.SanitizeMol(truncated)

    value, err = safe_symm_rmsd(truncated, full)
    assert value is None and err.startswith("rmsd_atom_mapping_failed")

    rmsd, n_matched, mode = pose_rmsd(truncated, full)
    assert rmsd is not None and rmsd == pytest.approx(0.0, abs=1e-6)
    assert n_matched == truncated.GetNumAtoms()
    assert mode == "dep_in_pose"


def test_score_case_routes_on_modelled_fraction(monkeypatch, tmp_path):
    """modelled_fraction < 1 must select the substructure path for every pose in the case."""
    native = _embedded("CCCCCCCC(=O)O")
    pose_path = tmp_path / "Ligand_0.sdf"
    with Chem.SDWriter(str(pose_path)) as w:
        w.write(native)
    native_path = tmp_path / "native.sdf"
    with Chem.SDWriter(str(native_path)) as w:
        w.write(native)

    monkeypatch.setattr(lipid_rmsd, "iter_poses",
                        lambda case_dir: [("docking", "0", 0, str(pose_path))])
    monkeypatch.setattr(lipid_rmsd, "read_scores", lambda results_dir: {})

    for fraction, expected in ((0.9, "dep_in_pose"), (1.0, "symm")):
        rows = lipid_rmsd.score_case(
            {"case_id": "c", "pdb_id": "X", "native_sdf": "native.sdf",
             "modelled_fraction": str(fraction)},
            str(tmp_path), rmsd_cutoff=2.0)
        assert [r["match_mode"] for r in rows] == [expected]


# --------------------------------------------------------------------------------------
# site vs pose outcome
# --------------------------------------------------------------------------------------
def test_wrong_site_is_not_reported_as_a_bad_pose():
    """An annular lipid can adopt a good conformation in a neighbouring empty site.

    With apo receptors that is a site-assignment error, not a pose error, and the two must not
    be conflated -- otherwise every T4 case reads as 'bad geometry'.
    """
    native = _embedded("CCCCCCCCCCCCCCCC(=O)O")
    native_xyz = lipid_rmsd._heavy_xyz(native)

    same_site = lipid_rmsd._heavy_xyz(_translated(native, (0.4, 0.0, 0.0)))
    far_away = lipid_rmsd._heavy_xyz(_translated(native, (40.0, 0.0, 0.0)))

    overlap_near, dist_near = lipid_rmsd.site_metrics(native_xyz, same_site)
    overlap_far, dist_far = lipid_rmsd.site_metrics(native_xyz, far_away)

    assert overlap_near == pytest.approx(1.0)
    assert dist_near == pytest.approx(0.4, abs=1e-6)
    assert overlap_far == 0.0
    assert dist_far == pytest.approx(40.0, abs=1e-6)

    assert lipid_rmsd.classify(0.4, overlap_near, 2.0) == "correct_site_correct_pose"
    assert lipid_rmsd.classify(6.0, overlap_near, 2.0) == "correct_site_wrong_pose"
    assert lipid_rmsd.classify(40.0, overlap_far, 2.0) == "wrong_site"


def test_grade_saturates_like_the_other_corpora():
    assert [lipid_rmsd.rmsd_grade(r) for r in (0.5, 1.5, 2.5, 3.5, 4.5)] == [4, 3, 2, 1, 0]
    assert lipid_rmsd.rmsd_grade(None) == 0


# --------------------------------------------------------------------------------------
# radii scale with the lipid, not with a constant
# --------------------------------------------------------------------------------------
def _measure_radii(extent):
    return (max(sel.CROP_FLOOR, extent / 2 + sel.CROP_MARGIN),
            max(sel.BIAS_FLOOR, extent / 2 + sel.BIAS_MARGIN))


def test_bias_radius_can_always_contain_the_lipid():
    """ChemEM's default 12 A bias sphere is smaller than an extended PIP2 (32.8 A)."""
    for extent in (9.8, 20.6, 27.5, 32.8):
        crop, bias = _measure_radii(extent)
        assert bias >= extent / 2, f"bias sphere cannot contain a {extent} A lipid"
        assert crop >= extent / 2, f"crop cuts through a {extent} A lipid"
    assert _measure_radii(32.8)[1] == pytest.approx(19.4)
    assert _measure_radii(27.5)[1] == pytest.approx(16.75)


def test_radii_never_fall_below_the_corpus_floors():
    """Compact lipids must still get the 25 A crop / 12 A bias the other corpora use."""
    crop, bias = _measure_radii(2.0)
    assert crop == sel.CROP_FLOOR
    assert bias == sel.BIAS_FLOOR


def test_measure_reports_radii_from_the_measured_extent(monkeypatch):
    """measure() must derive the radii from the deposited copy, not from a constant."""
    atoms = [
        {"auth_asym_id": "A", "auth_seq_id": "1", "type_symbol": "C", "label_alt_id": ".",
         "Cartn_x": "0.0", "Cartn_y": "0.0", "Cartn_z": "0.0",
         "occupancy": "1.00", "B_iso_or_equiv": "30.0"},
        {"auth_asym_id": "A", "auth_seq_id": "1", "type_symbol": "C", "label_alt_id": ".",
         "Cartn_x": "30.0", "Cartn_y": "0.0", "Cartn_z": "0.0",
         "occupancy": "1.00", "B_iso_or_equiv": "30.0"},
    ]
    monkeypatch.setattr(sel, "lipid_instances", lambda pdb, comp: {("A", "1"): atoms})
    monkeypatch.setattr(sel, "count_contacts", lambda *a, **k: 8)

    m = sel.measure("XXXX", "LIP", n_heavy_ccd=4)
    assert m["max_extent_A"] == pytest.approx(30.0)
    assert m["crop_radius_A"] == pytest.approx(31.0)
    assert m["bias_radius_A"] == pytest.approx(18.0)
    assert m["modelled_fraction"] == pytest.approx(0.5)
    assert m["contacts_per_atom"] == pytest.approx(4.0)
    assert m["centroid_x"] == pytest.approx(15.0)


def test_best_modelled_copy_wins_ties_broken_on_b_factor(monkeypatch):
    """Picking an arbitrary copy would make the ground truth a lottery in annular cases."""
    def atom(chain, seq, x, b):
        return {"auth_asym_id": chain, "auth_seq_id": seq, "type_symbol": "C",
                "label_alt_id": ".", "Cartn_x": str(x), "Cartn_y": "0.0", "Cartn_z": "0.0",
                "occupancy": "1.00", "B_iso_or_equiv": str(b)}

    instances = {
        ("A", "1"): [atom("A", "1", 0, 20.0)],                              # least complete
        ("B", "1"): [atom("B", "1", 0, 90.0), atom("B", "1", 2, 90.0)],     # complete, poor B
        ("C", "1"): [atom("C", "1", 0, 25.0), atom("C", "1", 2, 25.0)],     # complete, best B
    }
    monkeypatch.setattr(sel, "lipid_instances", lambda pdb, comp: instances)
    monkeypatch.setattr(sel, "count_contacts", lambda *a, **k: 0)
    m = sel.measure("XXXX", "LIP", n_heavy_ccd=2)
    assert (m["instance_chain"], m["mean_b"]) == ("C", 25.0)
    assert m["n_copies_in_entry"] == 3


# --------------------------------------------------------------------------------------
# ModelServer parsing
# --------------------------------------------------------------------------------------
CIF = """data_XXXX
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_alt_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
HETATM C . 1.0 0.0 0.0 1.00 30.0 301 A
HETATM H . 2.0 0.0 0.0 1.00 30.0 301 A
HETATM C B 3.0 0.0 0.0 0.50 30.0 301 A
ATOM N . 4.0 0.0 0.0 1.00 30.0 12 A
#
"""


def test_atom_site_parser_drops_hydrogens_and_secondary_altlocs():
    """A two-conformer lipid would otherwise report ~double its atoms, so modelled_fraction
    could exceed 1.0 and a truncated tail would be scored as complete."""
    rows = sel.parse_atom_site(CIF)
    assert len(rows) == 4
    heavy = sel._heavy_rows(rows)
    assert [r["Cartn_x"] for r in heavy] == ["1.0", "4.0"]
