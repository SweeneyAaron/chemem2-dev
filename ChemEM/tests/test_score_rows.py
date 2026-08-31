"""Tests for the ``--score`` CSV schema.

The contract downstream depends on is the *header*, not any stored data: the
benchmark scripts (plots_and_cutoffs, analyze_energy, characterise_solutions,
fit_echo_weights) read columns by name. So the column names the old separate tools
emitted must all still be there, and the ordering must be deterministic.
"""
import argparse
import unittest

try:
    from ChemEM.protocols.score import rows as rows_mod
    from ChemEM.protocols.score.scorers import SCORER_NAMES, load_scorer_cls
    from ChemEM.protocols.score.scorers import echo as echo_mod
except ModuleNotFoundError:
    from protocols.score import rows as rows_mod
    from protocols.score.scorers import SCORER_NAMES, load_scorer_cls
    from protocols.score.scorers import echo as echo_mod


# The columns the old --rescore-poses wrote (RescorePoses._columns with both
# relaxation flags off), hard-coded here so a refactor cannot silently drop one.
LEGACY_RESCORE_COLUMNS = (
    ["ligand", "source", "pose", "ligand_idx", "conf_id", "site_id",
     "echo_total", "echo_linear", "map_score"]
    + echo_mod.OFFSET_TERMS
    + [f"raw_{n}" for n in echo_mod.RAW_TERMS]
    + [f"w_{n}" for n in echo_mod.LINEAR_TERMS]
    + ["error"]
)

# The leading columns the old standalone score_poses.py emitted, minus the ones that
# belonged to scorers outside --score's scope (strain, clash, rmsd labelling).
LEGACY_SCORE_POSES_COLUMNS = [
    "case_id", "site_id", "ligand_idx",
    "qscore", "q_mean", "q_low_tail",
    "density_coverage", "density_precision", "density_overlap", "density_ccc",
    "density_mi", "density_normalized_mi", "density_envelope_iou",
    "density_excess_fraction", "density_sci",
    "mmgbsa", "mmgbsa_eel", "mmgbsa_vdw", "mmgbsa_egb", "mmgbsa_ecav",
    "mmgbsa_min_shift_A",
]


def _scorers(names, **opts):
    args = argparse.Namespace(**opts)
    return [load_scorer_cls(n)(system=None, opts=args) for n in names]


class TestFieldnames(unittest.TestCase):
    def test_identity_columns_come_first(self):
        fields = rows_mod.build_fieldnames(_scorers(["echo"]), [{"echo_total": 1.0}])
        self.assertEqual(
            fields[:len(rows_mod.IDENTITY_COLUMNS)], list(rows_mod.IDENTITY_COLUMNS)
        )

    def test_headlines_are_grouped_before_the_detail_blocks(self):
        """One number per scorer, side by side, so a CSV is readable by eye."""
        scorers = _scorers(["echo", "qscore", "mmgbsa"])
        fields = rows_mod.build_fieldnames(scorers, [{}])
        start = len(rows_mod.IDENTITY_COLUMNS)
        self.assertEqual(fields[start:start + 3], ["echo_total", "qscore", "mmgbsa"])

    def test_scorer_order_follows_score_with(self):
        forward = rows_mod.build_fieldnames(_scorers(["echo", "qscore"]), [{}])
        reverse = rows_mod.build_fieldnames(_scorers(["qscore", "echo"]), [{}])
        self.assertLess(forward.index("echo_total"), forward.index("qscore"))
        self.assertLess(reverse.index("qscore"), reverse.index("echo_total"))

    def test_is_deterministic_across_calls(self):
        scorers = _scorers(["echo", "density", "qscore"], score_density_region="full")
        rows = [{"echo_total": 1.0, "qscore": 0.5, "density_ccc": 0.2}]
        self.assertEqual(
            rows_mod.build_fieldnames(scorers, rows),
            rows_mod.build_fieldnames(scorers, rows),
        )

    def test_no_duplicate_columns(self):
        scorers = _scorers(list(SCORER_NAMES), score_density_region="full")
        fields = rows_mod.build_fieldnames(scorers, [{}])
        self.assertEqual(len(fields), len(set(fields)))

    def test_runtime_keys_are_kept_not_dropped(self):
        """A kernel gaining a metric must show up without a code change here."""
        scorers = _scorers(["density"], score_density_region="full")
        fields = rows_mod.build_fieldnames(
            scorers, [{"density_brand_new_metric": 1.0}]
        )
        self.assertIn("density_brand_new_metric", fields)

    def test_status_columns_come_last(self):
        scorers = _scorers(["echo", "qscore"])
        fields = rows_mod.build_fieldnames(
            scorers, [{"error": "", "echo_error": "boom", "qscore_failed": 1}]
        )
        self.assertEqual(
            fields[-4:], ["error", "echo_error", "qscore_error", "qscore_failed"]
        )

    def test_status_columns_are_stable_when_nothing_failed(self):
        """A clean run and a run with one bad pose must produce the same header, or
        two CSVs from the same command cannot be concatenated."""
        scorers = _scorers(["echo", "qscore"])
        clean = rows_mod.build_fieldnames(scorers, [{"echo_total": 1.0}])
        broken = rows_mod.build_fieldnames(
            scorers, [{"echo_total": 1.0}, {"echo_error": "boom"}]
        )
        self.assertEqual(clean, broken)
        self.assertIn("echo_error", clean)
        self.assertIn("error", clean)

    def test_a_scorer_prefix_does_not_swallow_its_status_column(self):
        """`qscore_failed` starts with `qscore_`, so the prefix sweep would bury it
        in the middle of the detail block."""
        fields = rows_mod.build_fieldnames(
            _scorers(["qscore"]), [{"qscore_failed": 1}]
        )
        self.assertGreater(fields.index("qscore_failed"), fields.index("q_low_tail"))

    def test_internal_keys_never_reach_the_csv(self):
        """`_json` carries the per-atom lists a flat table cannot hold."""
        fields = rows_mod.build_fieldnames(
            _scorers(["qscore"]), [{"qscore": 0.5, "_json": {"q_per_atom": [1, 2]}}]
        )
        self.assertNotIn("_json", fields)


class TestLegacySchemaIsASubset(unittest.TestCase):
    def test_old_rescore_poses_columns_all_survive(self):
        scorers = _scorers(["echo"])
        fields = set(rows_mod.build_fieldnames(scorers, [{}]))
        missing = [c for c in LEGACY_RESCORE_COLUMNS if c not in fields]
        self.assertEqual(missing, [], f"--rescore-poses columns lost: {missing}")

    def test_old_score_poses_columns_all_survive(self):
        scorers = _scorers(list(SCORER_NAMES), score_density_region="full")
        fields = set(rows_mod.build_fieldnames(scorers, [{}]))
        missing = [c for c in LEGACY_SCORE_POSES_COLUMNS if c not in fields]
        self.assertEqual(missing, [], f"score_poses.py columns lost: {missing}")

    def test_hydrogen_relaxation_columns_appear_only_when_enabled(self):
        off = set(rows_mod.build_fieldnames(_scorers(["echo"]), [{}]))
        self.assertNotIn("echo_total_prehmin", off)
        self.assertNotIn("echo_total_prot_h_pre", off)

        on = set(rows_mod.build_fieldnames(
            _scorers(["echo"], score_echo_minimise_hydrogens=True,
                     score_echo_protein_h=True),
            [{}],
        ))
        for column in ("echo_total_prehmin", "n_h_torsions", "h_delta_deg_max",
                       "echo_total_prot_h_pre", "n_prot_h_donors",
                       "n_prot_h_donors_close"):
            self.assertIn(column, on)


class TestCsvValues(unittest.TestCase):
    def test_none_becomes_empty(self):
        self.assertEqual(rows_mod._csv_value(None), "")

    def test_tuples_are_flattened_not_repr_d(self):
        """density_subgrid_shape is a shape tuple; a bare repr would carry commas
        into the CSV."""
        self.assertEqual(rows_mod._csv_value((3, 4, 5)), "3x4x5")


if __name__ == "__main__":
    unittest.main()
