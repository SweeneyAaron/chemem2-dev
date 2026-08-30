from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

try:
    from ChemEM import protein_coevolution as pc
except ModuleNotFoundError:
    import protein_coevolution as pc


def _blast_hit(**overrides) -> pc.BlastHit:
    values = {
        "query_id": "query",
        "subject_id": "subject",
        "query_length": 10,
        "subject_length": 10,
        "query_start": 1,
        "query_end": 10,
        "subject_start": 1,
        "subject_end": 10,
        "evalue": 1e-20,
        "bitscore": 100.0,
        "percent_identity": 90.0,
        "query_coverage_percent": 100.0,
        "query_aligned": "ACDEFGHIKL",
        "subject_aligned": "ACDEFGHIKL",
        "iteration": 2,
    }
    values.update(overrides)
    return pc.BlastHit(**values)


def _tabular_row(subject_id: str, bitscore: float) -> str:
    return "\t".join(
        [
            "query",
            subject_id,
            "10",
            "10",
            "1",
            "10",
            "1",
            "10",
            "1e-20",
            str(bitscore),
            "90.0",
            "100.0",
            "ACDEFGHIKL",
            "ACDEFGHIKL",
        ]
    )


def test_query_normalization_removes_terminal_stop_and_reports_ambiguity() -> None:
    sequence, warnings = pc.normalize_protein_sequence("  acd x*\n")

    assert sequence == "ACDX"
    assert any("terminal stop" in warning for warning in warnings)
    assert any("Ambiguous/noncanonical" in warning and "X" in warning for warning in warnings)
    assert any("shorter than 10" in warning for warning in warnings)


@pytest.mark.parametrize("sequence", ["", "ACD-", "AC*DE"])
def test_query_normalization_rejects_empty_or_invalid_sequences(sequence: str) -> None:
    with pytest.raises(pc.CoevolutionError):
        pc.normalize_protein_sequence(sequence)


def test_psiblast_parser_retains_only_the_final_iteration() -> None:
    text = "\n".join(
        [
            "# PSI-BLAST 2.15",
            "# Iteration: 1",
            _tabular_row("old_hit", 40.0),
            "# Iteration 2",
            _tabular_row("final_hit_1", 80.0),
            _tabular_row("final_hit_2", 70.0),
        ]
    )

    hits = pc.parse_psiblast_tabular(text)

    assert [hit.subject_id for hit in hits] == ["final_hit_1", "final_hit_2"]
    assert [hit.iteration for hit in hits] == [2, 2]
    assert hits[0].bitscore == pytest.approx(80.0)
    assert hits[0].query_aligned == "ACDEFGHIKL"


def test_psiblast_parser_rejects_malformed_rows() -> None:
    with pytest.raises(pc.CoevolutionError, match="expected 14 tab-separated fields"):
        pc.parse_psiblast_tabular("query\ttoo\tfew\tfields")


def test_hsp_projection_discards_insertions_and_preserves_subject_deletions() -> None:
    hit = _blast_hit(
        query_start=2,
        query_end=7,
        subject_start=10,
        subject_end=15,
        query_aligned="CD-EFGH",
        subject_aligned="CDW-FGH",
        query_coverage_percent=60.0,
    )

    projected = pc.project_hit_to_query(hit, "ACDEFGHIKL")

    assert projected == "-CD-FGH---"
    assert len(projected) == 10


def test_hsp_projection_rejects_a_malformed_reported_span() -> None:
    hit = _blast_hit(
        query_start=2,
        query_end=7,
        query_aligned="CDEFG",
        subject_aligned="CDEFG",
    )

    with pytest.raises(pc.CoevolutionError, match="span is inconsistent"):
        pc.project_hit_to_query(hit, "ACDEFGHIKL")


def test_run_psiblast_builds_local_command_and_returns_projected_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = "/mock/bin/psiblast"
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        if command == [executable, "-version"]:
            assert kwargs["timeout"] == 30
            return subprocess.CompletedProcess(
                command, 0, stdout="psiblast: 2.16.0+\n", stderr=""
            )

        assert kwargs == {"check": False, "capture_output": True, "text": True}
        hits_path = Path(command[command.index("-out") + 1])
        pssm_path = Path(command[command.index("-out_ascii_pssm") + 1])
        hits_path.write_text(
            "\n".join(
                [
                    "# PSI-BLAST 2.16.0+",
                    "# Iteration: 2",
                    "\t".join(
                        [
                            "query",
                            "sp|P12345|HIT",
                            "10",
                            "50",
                            "2",
                            "7",
                            "10",
                            "15",
                            "1e-20",
                            "80.0",
                            "83.3",
                            "60.0",
                            "CD-EFGH",
                            "CDW-FGH",
                        ]
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        pssm_path.write_text("mock ASCII PSSM\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(pc, "_resolve_executable", lambda _: executable)
    monkeypatch.setattr(pc.shutil, "which", lambda _: None)
    monkeypatch.setattr(pc.subprocess, "run", fake_run)

    alignment, provenance = pc.run_psiblast(
        executable="psiblast",
        query_id="query",
        query_sequence="ACDEFGHIKL",
        database="/data/uniref90",
        output_dir=tmp_path,
        iterations=4,
        evalue=1e-3,
        inclusion_evalue=1e-4,
        max_hits=123,
        min_query_coverage=0.7,
        threads=3,
    )

    search_command = calls[0]

    def command_value(flag: str) -> str:
        return search_command[search_command.index(flag) + 1]

    assert search_command[0] == executable
    assert command_value("-db") == "/data/uniref90"
    assert command_value("-num_iterations") == "4"
    assert command_value("-inclusion_ethresh") == "0.0001"
    assert command_value("-evalue") == "0.001"
    assert command_value("-max_target_seqs") == "123"
    assert command_value("-qcov_hsp_perc") == "70.0"
    assert command_value("-num_threads") == "3"
    assert command_value("-max_hsps") == "1"
    assert command_value("-outfmt").startswith("7 qseqid sseqid qlen slen")
    assert command_value("-outfmt").endswith("qseq sseq")
    assert "-remote" not in search_command
    assert calls[1] == [executable, "-version"]

    assert (tmp_path / "query.fasta").read_text(encoding="utf-8") == (
        ">query\nACDEFGHIKL\n"
    )
    assert (tmp_path / "psiblast.pssm").is_file()
    assert (tmp_path / "psiblast.stderr.log").read_text(encoding="utf-8") == ""
    assert alignment.ids == ["query", "sp|P12345|HIT"]
    assert alignment.sequences == ["ACDEFGHIKL", "-CD-FGH---"]
    assert alignment.query_positions == list(range(1, 11))
    assert alignment.source == "psiblast"
    assert provenance["backend"] == "psiblast"
    assert provenance["version"] == "psiblast: 2.16.0+"
    assert provenance["database"] == "/data/uniref90"
    assert provenance["reported_hsps"] == 1
    assert provenance["unique_subjects"] == 1
    assert provenance["command"] == search_command
    assert "-remote" not in provenance["command"]


def test_database_provenance_fingerprints_local_index_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "family_db"
    (tmp_path / "family_db.pin").write_bytes(b"index")
    (tmp_path / "family_db.psq").write_bytes(b"sequences")
    monkeypatch.setattr(pc.shutil, "which", lambda _: None)

    provenance = pc.blast_database_provenance(
        "/missing/bin/psiblast", str(prefix)
    )

    assert provenance["prefix"] == str(prefix)
    assert len(provenance["index_files"]) == 2
    assert len(provenance["metadata_sha256"]) == 64


def test_sequence_identity_weights_use_joint_residues_not_shared_gaps() -> None:
    encoded = pc.encode_alignment(["AAAA", "AAAA", "CCCC", "A---"])

    weights = pc.sequence_identity_weights(
        encoded,
        identity_threshold=0.8,
        minimum_overlap=0.5,
        block_size=2,
    )

    np.testing.assert_allclose(weights, [0.5, 0.5, 1.0, 1.0])

    mostly_missing = pc.encode_alignment(["A---", "A---"])
    missing_weights = pc.sequence_identity_weights(
        mostly_missing,
        identity_threshold=0.8,
        minimum_overlap=0.5,
        block_size=1,
    )
    np.testing.assert_allclose(missing_weights, [1.0, 1.0])


def _binary_counts(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = pc.AA_TO_CODE["A"]
    c = pc.AA_TO_CODE["C"]
    codes_i = np.asarray([a] * 50 + [c] * 50, dtype=np.uint8)
    if kind == "independent":
        codes_j = np.asarray([a] * 25 + [c] * 25 + [a] * 25 + [c] * 25)
    elif kind == "perfect":
        codes_j = codes_i.copy()
    elif kind == "anti":
        codes_j = np.asarray([c] * 50 + [a] * 50, dtype=np.uint8)
    else:  # pragma: no cover - protects the test helper itself
        raise ValueError(kind)

    counts = np.zeros((pc.GAP_CODE, pc.GAP_CODE), dtype=float)
    np.add.at(counts, (codes_i, codes_j), 1.0)
    return codes_i, codes_j, counts


@pytest.mark.parametrize(
    ("kind", "expected_mi", "expected_pearson"),
    [
        ("independent", 0.0, 0.0),
        ("perfect", 1.0, 1.0),
        ("anti", 1.0, -1.0),
    ],
)
def test_independent_perfect_and_anticorrelated_statistics(
    kind: str, expected_mi: float, expected_pearson: float
) -> None:
    codes_i, codes_j, counts = _binary_counts(kind)

    raw_mi, stabilized_mi, _, normalized_mi = pc._mutual_information(
        counts, pseudocount=0.0
    )
    chi_square, pearson, minimum_binary_support = pc._binary_query_statistics(
        codes_i,
        codes_j,
        np.ones(codes_i.size),
        pc.AA_TO_CODE["A"],
        pc.AA_TO_CODE["A"],
    )
    (
        categorical_chi,
        degrees_freedom,
        cramers_v,
        _,
        _,
        categorical_retained,
    ) = pc._categorical_statistics(counts, minimum_state_count=1.0)

    assert raw_mi == pytest.approx(expected_mi, abs=1e-12)
    assert stabilized_mi == pytest.approx(expected_mi, abs=1e-12)
    assert normalized_mi == pytest.approx(expected_mi, abs=1e-12)
    assert pearson == pytest.approx(expected_pearson, abs=1e-12)
    assert minimum_binary_support == pytest.approx(50.0)
    assert categorical_retained == pytest.approx(1.0)
    assert degrees_freedom == 1
    if kind == "independent":
        assert chi_square == pytest.approx(0.0, abs=1e-12)
        assert categorical_chi == pytest.approx(0.0, abs=1e-12)
        assert cramers_v == pytest.approx(0.0, abs=1e-12)
    else:
        assert chi_square == pytest.approx(100.0)
        assert categorical_chi == pytest.approx(100.0)
        assert cramers_v == pytest.approx(1.0)


def test_single_rare_substitution_is_not_a_supported_perfect_correlation() -> None:
    a = pc.AA_TO_CODE["A"]
    c = pc.AA_TO_CODE["C"]
    d = pc.AA_TO_CODE["D"]
    codes_i = np.asarray([a] * 99 + [c], dtype=np.uint8)
    codes_j = np.asarray([a] * 99 + [d], dtype=np.uint8)
    weights = np.ones(100)
    counts = np.zeros((pc.GAP_CODE, pc.GAP_CODE), dtype=float)
    np.add.at(counts, (codes_i, codes_j), weights)

    chi_square, pearson, minimum_support = pc._binary_query_statistics(
        codes_i, codes_j, weights, a, a, minimum_state_count=2.0
    )
    categorical = pc._categorical_statistics(counts, minimum_state_count=2.0)
    state_pearson, state_i, state_j = pc._max_residue_pearson(
        counts, minimum_state_count=2.0
    )

    assert minimum_support == pytest.approx(1.0)
    assert math.isnan(chi_square)
    assert math.isnan(pearson)
    assert math.isnan(categorical[0])
    assert categorical[4] is False
    assert math.isnan(state_pearson)
    assert state_i == state_j == ""


def test_average_product_correction_uses_finite_off_diagonal_means() -> None:
    matrix = np.asarray(
        [
            [math.nan, 1.0, 2.0],
            [1.0, math.nan, 3.0],
            [2.0, 3.0, math.nan],
        ]
    )

    corrected = pc._average_product_correction(matrix)

    assert np.isnan(np.diag(corrected)).all()
    assert corrected[0, 1] == pytest.approx(-0.5)
    assert corrected[0, 2] == pytest.approx(0.125)
    assert corrected[1, 2] == pytest.approx(0.5)
    np.testing.assert_allclose(corrected, corrected.T, equal_nan=True)


def _with_insertion(sequence: str, residue: str) -> str:
    return sequence[:3] + residue + sequence[3:]


def test_existing_msa_pipeline_writes_query_anchored_outputs(tmp_path: Path) -> None:
    anchored = [
        "ACDEFGHIKA",
        "ACDEYGHIKA",
        "ACNEFGHIKA",
        "CCDEFGHIKC",
        "CCDEYGHIKC",
        "CCNEFGHIKC",
    ]
    ids = ["query", "a1", "a2", "c1", "c2", "c3"]
    insertion_states = ["-", "T", "S", "T", "S", "V"]
    msa_path = tmp_path / "input.fasta"
    pc.write_fasta(
        msa_path,
        ids,
        [
            _with_insertion(sequence, insertion)
            for sequence, insertion in zip(anchored, insertion_states)
        ],
    )
    output_dir = tmp_path / "results"

    return_code = pc.main(
        [
            "--msa",
            str(msa_path),
            "--output",
            str(output_dir),
            "--identity-threshold",
            "1.0",
            "--min-pair-neff",
            "1",
            "--min-state-count",
            "1",
            "--mi-pseudocount",
            "0",
            "--min-separation",
            "1",
            "--top",
            "0",
        ]
    )

    assert return_code == 0
    expected_outputs = {
        "msa_query_anchored.fasta",
        "msa_filtered.fasta",
        "positions.csv",
        "coevolution_pairs.csv",
        "score_matrices.npz",
        "summary.json",
    }
    assert expected_outputs <= {path.name for path in output_dir.iterdir()}

    anchored_records = pc.parse_fasta_records(
        (output_dir / "msa_query_anchored.fasta").read_text(encoding="utf-8")
    )
    assert [sequence for _, sequence in anchored_records] == anchored
    assert all(len(sequence) == 10 for _, sequence in anchored_records)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["search"]["backend"] == "existing_msa"
    assert summary["query"]["id"] == "query"
    assert summary["query"]["length"] == 10
    assert summary["alignment"]["retained_sequences"] == 6
    assert summary["alignment"]["effective_sequences"] == pytest.approx(6.0)
    assert summary["alignment"]["analyzable_positions"] == 4
    assert summary["alignment"]["reported_pairs"] == 6

    with (output_dir / "coevolution_pairs.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 6
    pair_1_10 = next(
        row for row in pairs if row["position_i"] == "1" and row["position_j"] == "10"
    )
    assert float(pair_1_10["mi_raw_bits"]) == pytest.approx(1.0)
    assert float(pair_1_10["binary_pearson_r"]) == pytest.approx(1.0)

    with np.load(output_dir / "score_matrices.npz") as matrices:
        np.testing.assert_array_equal(matrices["query_positions"], [1, 3, 5, 10])
        np.testing.assert_allclose(matrices["sequence_weights"], np.ones(6))
        assert matrices["mi_apc"].shape == (4, 4)
