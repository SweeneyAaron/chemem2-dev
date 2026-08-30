"""Focused regression tests for the offline false-positive analysis package.

The fixtures are deliberately synthetic: these tests exercise data-integrity and
validation behaviour without depending on the benchmark corpus being present.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ChemEM.benchmark import analyze_false_positives as analysis
from ChemEM.benchmark import select_blind_energy_subset as blind_subset


def test_exact_path_join_does_not_fall_back_to_basename(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    exact_path = "/labels/case-a/Ligand_0.sdf"
    payload = {
        "emd_id": "EMD-A",
        "resolution": None,
        "warnings": [],
        "case_flags": [],
        "solutions": [
            {
                "path": exact_path,
                "binding_site": 1,
                "libidx": 2,
                "stage": "docking",
                "solution_idx": 3,
                "is_correct_assignment": 1,
            }
        ],
    }
    (labels_dir / "EMD-A.json").write_text(json.dumps(payload), encoding="utf-8")
    labels, cases = analysis.load_rmsd_labels(labels_dir)
    scores = pd.DataFrame(
        {
            "sdf_path": ["/scores/case-a/Ligand_0.sdf"],
            "case": ["EMD-A"],
            "site_id": [1],
            "ligand_idx": [2],
            "stage": ["docking"],
            "pose_idx": [3],
            "qscore": [0.7],
        }
    )

    with pytest.raises(ValueError, match="lack exact labels"):
        analysis.exact_path_join(scores, labels, strict=True)

    scores.loc[0, "sdf_path"] = exact_path
    joined, audit = analysis.exact_path_join(scores, labels, strict=True)

    assert len(joined) == 1
    assert joined.loc[0, "is_correct_assignment"] == 1
    assert cases.loc[0, "case_warning_free"]
    assert isinstance(audit, dict)


def test_logical_pose_key_is_stage_aware():
    assert analysis.LOGICAL_KEY == (
        "case",
        "site_id",
        "ligand_idx",
        "stage",
        "pose_idx",
    )
    assert blind_subset.LOGICAL_KEY == (
        "case_id",
        "site_id",
        "ligand_idx",
        "stage",
        "pose_idx",
    )

    docking = {
        "case_id": "EMD-A",
        "site_id": 1,
        "ligand_idx": 2,
        "stage": "docking",
        "pose_idx": 3,
    }
    refined = {**docking, "stage": "refined"}
    blind_subset.validate_unique([docking, refined])

    with pytest.raises(ValueError, match="duplicate"):
        blind_subset.validate_unique([docking, dict(docking)])


def test_coherent_selection_prefers_refined_then_maximises_qscore():
    poses = pd.DataFrame(
        [
            ("EMD-A", 1, 0, "docking", 0, 0.99, "/a/d0.sdf"),
            ("EMD-A", 1, 0, "refined", 1, 0.20, "/a/r1.sdf"),
            ("EMD-A", 1, 0, "refined", 2, 0.40, "/a/r2.sdf"),
            ("EMD-A", 1, 1, "docking", 3, 0.30, "/a/d3.sdf"),
            ("EMD-A", 1, 1, "docking", 4, 0.60, "/a/d4.sdf"),
        ],
        columns=[
            "case",
            "site_id",
            "ligand_idx",
            "stage",
            "pose_idx",
            "qscore",
            "sdf_path",
        ],
    )

    selected = analysis.select_coherent_poses(poses, top_k=1)
    keys = set(
        selected[["ligand_idx", "stage", "pose_idx"]]
        .itertuples(index=False, name=None)
    )

    assert keys == {(0, "refined", 2), (1, "docking", 4)}
    assert len(selected) == 2


def test_ambiguous_wrong_site_rows_exclude_the_entire_site():
    labels = pd.DataFrame(
        [
            ("EMD-A", 1, "correct_ligand_wrong_site", 1.99),
            ("EMD-A", 2, "correct_ligand_wrong_site", 2.00),
            ("EMD-A", 3, "correct_site_wrong_pose", 0.50),
            ("EMD-B", 1, "correct_ligand_wrong_site", np.nan),
        ],
        columns=["case", "site_id", "characterisation", "rmsd_best"],
    )
    rows = pd.DataFrame(
        {
            "case": ["EMD-A", "EMD-A", "EMD-A", "EMD-B"],
            "site_id": [1, 1, 2, 1],
            "ligand_idx": [0, 1, 0, 0],
        }
    )

    ambiguous = analysis.ambiguous_site_keys(labels, rmsd_cutoff=2.0)
    retained = analysis.exclude_ambiguous_sites(rows, ambiguous)

    assert ambiguous == {("EMD-A", 1)}
    assert set(retained[["case", "site_id"]].itertuples(index=False, name=None)) == {
        ("EMD-A", 2),
        ("EMD-B", 1),
    }


def test_leave_one_case_out_splits_never_leak_cases():
    rows = pd.DataFrame(
        {"case": ["EMD-A", "EMD-A", "EMD-B", "EMD-C"], "value": range(4)}
    )
    folds = list(analysis.leave_one_case_out_splits(rows))

    assert len(folds) == 3
    seen_test_rows: list[int] = []
    for train_idx, test_idx in folds:
        train_cases = set(rows.iloc[list(train_idx)]["case"])
        test_cases = set(rows.iloc[list(test_idx)]["case"])
        assert len(test_cases) == 1
        assert train_cases.isdisjoint(test_cases)
        seen_test_rows.extend(int(i) for i in test_idx)
    assert sorted(seen_test_rows) == list(range(len(rows)))


def test_pocket_energy_missingness_falls_back_to_qscore():
    sites = pd.DataFrame(
        {
            "case": ["EMD-A"] * 3,
            "site_id": [1, 2, 3],
            "qscore": [0.10, 0.20, 0.30],
            "mmgbsa_vdw": [-10.0, -5.0, np.nan],
        }
    )

    scored = analysis.pocket_scores(sites).set_index("site_id")

    assert scored.loc[3, "combined_score"] == pytest.approx(
        scored.loc[3, "qscore_rank"]
    )
    assert scored.loc[1, "combined_score"] == pytest.approx(
        0.5 * scored.loc[1, "qscore_rank"]
        + 0.5 * scored.loc[1, "energy_vdw_rank"]
    )
    fallback_columns = [name for name in scored if "fallback" in name]
    assert fallback_columns, "pocket scores must make fallback use auditable"
    assert bool(scored.loc[3, fallback_columns[0]])
    assert not bool(scored.loc[1, fallback_columns[0]])


def test_ligand_missing_energy_triggers_site_wide_overlap_fallback():
    candidates = pd.DataFrame(
        {
            "case": ["EMD-A"] * 4,
            "site_id": [1, 1, 2, 2],
            "ligand_idx": [0, 1, 0, 1],
            "density_overlap": [0.8, 0.2, 0.9, 0.1],
            "mmgbsa": [-2.0, np.nan, -1.0, -8.0],
        }
    )

    scored = analysis.ligand_scores(candidates)
    incomplete = scored[scored["site_id"] == 1]
    complete = scored[scored["site_id"] == 2]

    assert np.allclose(incomplete["combined_score"], incomplete["overlap_rank"])
    assert np.allclose(
        complete["combined_score"],
        0.6 * complete["mmgbsa_rank"] + 0.4 * complete["overlap_rank"],
    )
    fallback_columns = [name for name in scored if "fallback" in name]
    assert fallback_columns, "ligand scores must make site fallback use auditable"
    assert incomplete[fallback_columns[0]].astype(bool).all()
    assert not complete[fallback_columns[0]].astype(bool).any()


def test_percentile_ranking_is_bounded_and_outlier_magnitude_safe():
    ordinary = analysis.percentile_rank(
        pd.Series([1.0, 2.0, 3.0]), higher_is_better=True
    )
    extreme = analysis.percentile_rank(
        pd.Series([1.0, 2.0, 1.0e300]), higher_is_better=True
    )
    reversed_rank = analysis.percentile_rank(
        pd.Series([1.0, 2.0, 3.0]), higher_is_better=False
    )

    assert np.allclose(ordinary, extreme)
    assert ordinary.between(0.0, 1.0).all()
    assert ordinary.iloc[0] < ordinary.iloc[1] < ordinary.iloc[2]
    assert reversed_rank.iloc[0] > reversed_rank.iloc[1] > reversed_rank.iloc[2]


def test_case_cluster_bootstrap_is_seed_deterministic():
    rows = pd.DataFrame(
        {
            "case": ["EMD-A", "EMD-A", "EMD-B", "EMD-C"],
            "value": [0.0, 2.0, 10.0, 20.0],
        }
    )

    def mean_value(sample: pd.DataFrame) -> float:
        return float(sample["value"].mean())

    first = analysis.case_cluster_bootstrap(
        rows, mean_value, n_bootstrap=100, seed=123
    )
    second = analysis.case_cluster_bootstrap(
        rows, mean_value, n_bootstrap=100, seed=123
    )

    assert first == second
    assert isinstance(first, dict)
    assert all(np.isfinite(value) for value in first.values())


def test_selector_ignores_json_labels_and_strips_label_columns(tmp_path):
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    score_rows = pd.DataFrame(
        [
            ("EMD-A", 1, 0, 0, "docking", 0.99, "/a/d0.sdf", 0.1, 1, 0),
            ("EMD-A", 1, 0, 1, "refined", 0.30, "/a/r1.sdf", 9.9, 0, 1),
            ("EMD-A", 1, 0, 2, "refined", 0.40, "/a/r2.sdf", 8.8, 1, 0),
        ],
        columns=[
            "case_id",
            "site_id",
            "ligand_idx",
            "pose_idx",
            "stage",
            "qscore",
            "sdf_path",
            "rmsd_best",
            "is_correct_assignment",
            "site_is_decoy",
        ],
    )
    score_rows.to_csv(scores_dir / "EMD-A.csv", index=False)
    # A co-located, deliberately invalid label file proves the loader only scans CSV.
    (scores_dir / "EMD-A.json").write_text("{not valid label JSON", encoding="utf-8")

    loaded = blind_subset.load_score_rows(scores_dir)
    selected = blind_subset.select_blind_poses(loaded, top_k=3)

    assert [(row["stage"], row["pose_idx"]) for row in selected] == [
        ("refined", 2),
        ("refined", 1),
    ]
    assert set(selected[0]) == set(blind_subset.MANIFEST_COLUMNS)
    forbidden = {"rmsd_best", "is_correct_assignment", "site_is_decoy"}
    assert forbidden.isdisjoint(selected[0])
    parser_destinations = {action.dest for action in blind_subset.build_parser()._actions}
    assert {"rmsd_dir", "labels", "rmsd_results"}.isdisjoint(parser_destinations)
