"""Tests for the SmartRefine2 no-regression safety net (plan Phase 1).

Covers:
  - _result_best_objective: objective extraction with raw-score fallback.
  - _snapshot_fit_result: aliasing-free snapshots survive later mutation.
  - _apply_no_regression_gate: reverts to the anchor iff the final pose
    regressed by the acceptance objective.
  - _final_map_minimise_ligand: discards an OpenMM polish that worsens the
    objective.
  - --sr2-selection default is 'greedy'.
"""
import argparse
import importlib
import types
import unittest

import numpy as np

try:
    from ChemEM.protocols.smart_refine_2 import smart_refine as sr
except ModuleNotFoundError:
    from protocols.smart_refine_2 import smart_refine as sr


def _result(best_objective, *, best_coords, best_raw=None):
    """Minimal stand-in for FitInMapResult covering the gate's reads."""
    if best_raw is None:
        best_raw = best_objective
    return types.SimpleNamespace(
        best_objective=float(best_objective),
        best_raw_score=float(best_raw),
        initial_raw_score=0.0,
        initial_objective=0.0,
        best_coords_A=np.asarray(best_coords, dtype=float),
    )


class TestResultBestObjective(unittest.TestCase):
    def test_returns_objective_when_finite(self):
        res = _result(0.42, best_coords=[[0, 0, 0]], best_raw=0.99)
        self.assertAlmostEqual(sr._result_best_objective(res), 0.42)

    def test_falls_back_to_raw_when_objective_missing(self):
        res = types.SimpleNamespace(best_raw_score=0.7)  # no best_objective
        self.assertAlmostEqual(sr._result_best_objective(res), 0.7)

    def test_none_is_neg_inf(self):
        self.assertEqual(sr._result_best_objective(None), float("-inf"))


class TestSnapshotFitResult(unittest.TestCase):
    def test_snapshot_is_aliasing_free(self):
        live = _result(0.5, best_coords=[[1.0, 2.0, 3.0]])
        snap = sr._snapshot_fit_result(live)
        # Mutate the live result the way the loop / rescore would.
        live.best_objective = -5.0
        live.best_coords_A[0, 0] = 99.0
        self.assertAlmostEqual(snap.best_objective, 0.5)
        self.assertAlmostEqual(snap.best_coords_A[0, 0], 1.0)

    def test_none_returns_none(self):
        self.assertIsNone(sr._snapshot_fit_result(None))


class TestNoRegressionGate(unittest.TestCase):
    def _fake_rl(self, final_result):
        return types.SimpleNamespace(
            _last_fit_in_map_result=final_result,
            _atom_positions=np.zeros((1, 3)),
        )

    def test_reverts_when_final_worse(self):
        anchor = _result(0.80, best_coords=[[0.0, 0.0, 0.0]])
        final = _result(0.30, best_coords=[[5.0, 0.0, 0.0]])  # drifted, worse
        rl = self._fake_rl(final)

        applied = {}

        def fake_apply(refine_ligand, result):
            applied["result"] = result
            refine_ligand._atom_positions = np.asarray(result.best_coords_A).copy()
            return refine_ligand

        orig = sr.apply_refinement
        sr.apply_refinement = fake_apply
        try:
            out = sr._apply_no_regression_gate(rl, anchor, 0.80)
        finally:
            sr.apply_refinement = orig

        self.assertTrue(out._sr2_reverted_to_anchor)
        self.assertIs(applied["result"], anchor)
        self.assertIs(out._last_fit_in_map_result, anchor)
        np.testing.assert_allclose(out._atom_positions, anchor.best_coords_A)

    def test_keeps_when_final_better_or_equal(self):
        anchor = _result(0.50, best_coords=[[0.0, 0.0, 0.0]])
        final = _result(0.90, best_coords=[[0.1, 0.0, 0.0]])  # improved
        rl = self._fake_rl(final)

        def fail_apply(*a, **k):
            raise AssertionError("apply_refinement must not be called when improved")

        orig = sr.apply_refinement
        sr.apply_refinement = fail_apply
        try:
            out = sr._apply_no_regression_gate(rl, anchor, 0.50)
        finally:
            sr.apply_refinement = orig

        self.assertFalse(out._sr2_reverted_to_anchor)
        self.assertIs(out._last_fit_in_map_result, final)

    def test_no_anchor_is_noop(self):
        final = _result(0.10, best_coords=[[9.0, 0.0, 0.0]])
        rl = self._fake_rl(final)
        out = sr._apply_no_regression_gate(rl, None, float("-inf"))
        self.assertFalse(out._sr2_reverted_to_anchor)


class TestFinalPolishGate(unittest.TestCase):
    """_final_map_minimise_ligand gates the polish on the acceptance objective,
    scoring the actual pre/post poses (not stale result fields). Drive it with a
    fake self whose _pose_objective is scripted to return pre then post."""

    def _fake_self(self, pre_val, post_val):
        vals = iter([pre_val, post_val])
        fake = types.SimpleNamespace(
            _final_minimise_enabled=lambda: True,
            _refresh_protein_context=lambda: None,
            system=types.SimpleNamespace(options=None),
            fit_config=None,
            _acceptance_scorer=lambda: "qscore",
            _pose_objective=lambda rl, scorer, coords: next(vals),
        )
        # _final_map_minimise_ligand routes through self._polish; here it just
        # invokes the (monkeypatched) standard polish.
        fake._polish = lambda rl, result, context_label: sr.local_refine_polish_ligand(
            rl, result, context_label=context_label
        )
        return fake

    def test_reverts_worsening_polish(self):
        pre = _result(0.80, best_coords=[[0.0, 0.0, 0.0]])
        rl = types.SimpleNamespace(
            _atom_positions=np.zeros((1, 3)), _last_fit_in_map_result=pre
        )

        def fake_polish(refine_ligand, result, **kw):
            # Polish walks the pose off-density and mutates result in place.
            result.best_objective = 0.20
            result.best_coords_A = np.asarray([[7.0, 0.0, 0.0]], dtype=float)
            refine_ligand._atom_positions = result.best_coords_A.copy()
            return refine_ligand, result

        def fake_apply(refine_ligand, result):
            refine_ligand._atom_positions = np.asarray(result.best_coords_A).copy()
            return refine_ligand

        orig_polish, orig_apply = sr.local_refine_polish_ligand, sr.apply_refinement
        sr.local_refine_polish_ligand = fake_polish
        sr.apply_refinement = fake_apply
        try:
            out_rl, out_res = sr.SmartRefine2._final_map_minimise_ligand(
                self._fake_self(0.80, 0.20), rl, rl._last_fit_in_map_result
            )
        finally:
            sr.local_refine_polish_ligand = orig_polish
            sr.apply_refinement = orig_apply

        # post (0.20) < pre (0.80) -> reverted to the origin pose.
        np.testing.assert_allclose(out_rl._atom_positions, [[0.0, 0.0, 0.0]])

    def test_keeps_improving_polish(self):
        pre = _result(0.40, best_coords=[[0.0, 0.0, 0.0]])
        rl = types.SimpleNamespace(
            _atom_positions=np.zeros((1, 3)), _last_fit_in_map_result=pre
        )

        def fake_polish(refine_ligand, result, **kw):
            result.best_objective = 0.95
            result.best_coords_A = np.asarray([[0.2, 0.0, 0.0]], dtype=float)
            refine_ligand._atom_positions = result.best_coords_A.copy()
            return refine_ligand, result

        orig_polish = sr.local_refine_polish_ligand
        sr.local_refine_polish_ligand = fake_polish
        try:
            out_rl, out_res = sr.SmartRefine2._final_map_minimise_ligand(
                self._fake_self(0.40, 0.95), rl, rl._last_fit_in_map_result
            )
        finally:
            sr.local_refine_polish_ligand = orig_polish

        # post (0.95) > pre (0.40) -> kept the polished pose.
        np.testing.assert_allclose(out_rl._atom_positions, [[0.2, 0.0, 0.0]])
        self.assertAlmostEqual(sr._result_best_objective(out_res), 0.95)


class _Clash:
    """Minimal stand-in for the ClashResult fields _objective_from_eval reads."""

    def __init__(self, count, penalty):
        self.count = int(count)
        self.penalty = float(penalty)


class TestObjectiveFromEval(unittest.TestCase):
    """The single objective convention. Regression guard for the bug where the
    polish and branch-walker paths hard-coded `-inf` on any clash while
    fit_in_map used the configured (soft) clash mode: on any ligand that cannot
    reach zero clashes the two met at the no-regression gate, `-inf` lost to the
    finite anchor, and the entire search was reverted away."""

    def test_soft_mode_is_finite_for_a_clashing_pose(self):
        cfg = types.SimpleNamespace(clash_mode="soft", clash_weight=1.0)
        obj = sr._objective_from_eval(0.68, _Clash(count=7, penalty=0.5), cfg, 44)
        self.assertTrue(np.isfinite(obj), "a clashing pose must stay rankable")
        # raw - weight * penalty / n_atoms
        self.assertAlmostEqual(obj, 0.68 - 0.5 / 44)

    def test_soft_mode_ranks_two_clashing_poses(self):
        cfg = types.SimpleNamespace(clash_mode="soft", clash_weight=1.0)
        better = sr._objective_from_eval(0.68, _Clash(7, 0.5), cfg, 44)
        worse = sr._objective_from_eval(0.55, _Clash(7, 0.5), cfg, 44)
        self.assertGreater(better, worse)

    def test_hard_mode_still_available_when_asked_for(self):
        cfg = types.SimpleNamespace(clash_mode="hard", clash_weight=1.0)
        self.assertEqual(
            sr._objective_from_eval(0.68, _Clash(1, 0.5), cfg, 44), float("-inf")
        )
        self.assertAlmostEqual(
            sr._objective_from_eval(0.68, _Clash(0, 0.0), cfg, 44), 0.68
        )

    def test_defaults_to_soft_when_config_lacks_the_fields(self):
        obj = sr._objective_from_eval(0.68, _Clash(3, 0.44), types.SimpleNamespace(), 44)
        self.assertTrue(np.isfinite(obj))

    def test_branch_walk_config_carries_the_clash_convention(self):
        cfg = sr.BranchWalkConfig()
        self.assertEqual(cfg.clash_mode, "soft")
        self.assertAlmostEqual(cfg.clash_weight, 1.0)


class TestNoRegressionGateCrossConvention(unittest.TestCase):
    """The gate must never rewind a run because one side carries a sentinel."""

    def _fake_rl(self, final_result):
        return types.SimpleNamespace(
            _last_fit_in_map_result=final_result,
            _atom_positions=np.zeros((1, 3)),
        )

    def _run_gate(self, rl, anchor, anchor_objective):
        applied = {}

        def fake_apply(refine_ligand, result):
            applied["result"] = result
            refine_ligand._atom_positions = np.asarray(result.best_coords_A).copy()
            return refine_ligand

        orig = sr.apply_refinement
        sr.apply_refinement = fake_apply
        try:
            out = sr._apply_no_regression_gate(rl, anchor, anchor_objective)
        finally:
            sr.apply_refinement = orig
        return out, applied

    def test_does_not_revert_when_final_objective_is_inf_but_raw_is_better(self):
        # The exact 8ul9_dga failure: search reached raw 0.68, a polish stamped
        # -inf onto its objective, and the finite anchor (0.55) won the compare.
        anchor = _result(0.55350, best_coords=[[0.0, 0.0, 0.0]], best_raw=0.56644)
        final = _result(float("-inf"), best_coords=[[3.0, 0.0, 0.0]], best_raw=0.68)
        rl = self._fake_rl(final)

        out, applied = self._run_gate(rl, anchor, 0.55350)

        self.assertFalse(out._sr2_reverted_to_anchor)
        self.assertNotIn("result", applied)
        self.assertIs(out._last_fit_in_map_result, final)
        np.testing.assert_allclose(out._atom_positions, np.zeros((1, 3)))

    def test_still_reverts_when_inf_final_is_genuinely_worse_on_raw(self):
        anchor = _result(0.55350, best_coords=[[0.0, 0.0, 0.0]], best_raw=0.56644)
        final = _result(float("-inf"), best_coords=[[9.0, 0.0, 0.0]], best_raw=0.21)
        rl = self._fake_rl(final)

        out, applied = self._run_gate(rl, anchor, 0.55350)

        self.assertTrue(out._sr2_reverted_to_anchor)
        self.assertIs(applied["result"], anchor)

    def test_both_inf_keeps_the_search_result(self):
        anchor = _result(float("-inf"), best_coords=[[0.0, 0.0, 0.0]], best_raw=0.55)
        final = _result(float("-inf"), best_coords=[[2.0, 0.0, 0.0]], best_raw=0.68)
        rl = self._fake_rl(final)

        out, applied = self._run_gate(rl, anchor, float("-inf"))

        self.assertFalse(out._sr2_reverted_to_anchor)
        self.assertNotIn("result", applied)

    def test_unrankable_pair_keeps_the_search_result(self):
        anchor = _result(float("-inf"), best_coords=[[0.0, 0.0, 0.0]], best_raw=float("nan"))
        final = _result(float("-inf"), best_coords=[[2.0, 0.0, 0.0]], best_raw=float("nan"))
        rl = self._fake_rl(final)

        out, applied = self._run_gate(rl, anchor, float("-inf"))

        self.assertFalse(out._sr2_reverted_to_anchor)
        self.assertNotIn("result", applied)


class TestFragmentMinDecision(unittest.TestCase):
    """`-inf >= -inf` made the overall-objective test a tautology, so the
    fragment minimiser accepted on iteration 1 and never pinned a fragment —
    `--sr2-minimiser fragment` silently degraded to a plain minimisation."""

    def test_does_not_blindly_accept_when_both_objectives_are_inf(self):
        # No block dropped, so the ONLY thing standing between this and "accept"
        # is the overall-objective test. `-inf >= -inf` used to pass it.
        blocks = [0.60, 0.55, 0.50]
        decision, _ = sr._fragment_min_decision(
            blocks, [0.61, 0.56, 0.51], float("-inf"), float("-inf"), block_tol=0.1
        )
        self.assertNotEqual(decision, "accept")

    def test_pins_the_collapsed_block_when_objectives_are_inf(self):
        decision, dropped = sr._fragment_min_decision(
            [0.60, 0.55, 0.50], [0.60, 0.20, 0.50],  # block 1 collapsed
            float("-inf"), float("-inf"), block_tol=0.1,
        )
        self.assertEqual(decision, "pin")
        self.assertEqual(dropped, [1])

    def test_accepts_when_objective_improves_and_no_block_drops(self):
        blocks = [0.60, 0.55, 0.50]
        decision, dropped = sr._fragment_min_decision(
            blocks, [0.62, 0.56, 0.51], 0.50, 0.55, block_tol=0.1
        )
        self.assertEqual(decision, "accept")
        self.assertEqual(dropped, [])

    def test_pins_worst_block_when_overall_regresses_within_tol(self):
        decision, dropped = sr._fragment_min_decision(
            [0.60, 0.55], [0.58, 0.52], 0.55, 0.40, block_tol=0.1
        )
        self.assertEqual(decision, "pin")
        self.assertEqual(dropped, [1])  # -0.03 is the larger drop


class TestBestPoseTracking(unittest.TestCase):
    """The loop must revert to the best pose it produced, not to the anchor.

    Previously `best_raw_score_so_far` was a bare float with no pose attached, so
    a run that improved for several iterations and then slipped on the last one
    rewound all the way to its input, discarding every intermediate gain."""

    def _drive(self, iteration_objectives, anchor_objective=0.55):
        """Run refine_ligand with the search stubbed out, one entry per iteration."""
        anchor = _result(anchor_objective, best_coords=[[0.0, 0.0, 0.0]],
                         best_raw=anchor_objective)
        rl = types.SimpleNamespace(
            _atom_positions=np.zeros((1, 3)),
            _last_fit_in_map_result=anchor,
            _rotor_tree=[types.SimpleNamespace(block_id=0)],
            _block_qscores=[0.5],
            get_best_block_by_qscore=lambda: 0,
        )
        pending = list(iteration_objectives)

        def fake_fit_and_accept(refine_ligand, *a, **k):
            return refine_ligand

        def fake_branch_walker(refine_ligand, **k):
            return ["candidate"] if pending else []

        def fake_refit(base_rl, branch_results, *a, **k):
            # Mirrors the real path: the winning clone comes back with the pose
            # already applied AND its fit result attached.
            objective, x = pending.pop(0)
            base_rl._last_fit_in_map_result = _result(
                objective, best_coords=[[x, 0.0, 0.0]], best_raw=objective
            )
            base_rl._atom_positions = np.asarray([[x, 0.0, 0.0]], dtype=float)
            return base_rl

        def fake_apply(refine_ligand, result):
            refine_ligand._atom_positions = np.asarray(result.best_coords_A).copy()
            return refine_ligand

        originals = (
            sr._fit_and_accept, sr.branch_walker,
            sr._refit_branch_candidates, sr.apply_refinement,
        )
        sr._fit_and_accept = fake_fit_and_accept
        sr.branch_walker = fake_branch_walker
        sr._refit_branch_candidates = fake_refit
        sr.apply_refinement = fake_apply
        try:
            return sr.refine_ligand(
                rl, scorer="qscore", max_iters=len(iteration_objectives),
                patience=None, kick_config=None, polish_cb=None,
            )
        finally:
            (sr._fit_and_accept, sr.branch_walker,
             sr._refit_branch_candidates, sr.apply_refinement) = originals

    def test_reverts_to_best_iteration_not_the_anchor(self):
        # Peaks at 0.68 on iteration 2, then slides to 0.50.
        out = self._drive([(0.60, 1.0), (0.68, 2.0), (0.50, 3.0)])
        self.assertTrue(out._sr2_reverted_to_anchor)
        np.testing.assert_allclose(out._atom_positions, [[2.0, 0.0, 0.0]])
        self.assertAlmostEqual(
            sr._result_best_objective(out._last_fit_in_map_result), 0.68
        )

    def test_keeps_final_pose_when_it_is_the_best(self):
        out = self._drive([(0.60, 1.0), (0.68, 2.0), (0.80, 3.0)])
        self.assertFalse(out._sr2_reverted_to_anchor)
        np.testing.assert_allclose(out._atom_positions, [[3.0, 0.0, 0.0]])

    def test_falls_back_to_anchor_when_search_never_improves(self):
        # Nothing beats the 0.55 anchor -> previous behaviour, revert to it.
        out = self._drive([(0.40, 1.0), (0.30, 2.0)], anchor_objective=0.55)
        self.assertTrue(out._sr2_reverted_to_anchor)
        np.testing.assert_allclose(out._atom_positions, [[0.0, 0.0, 0.0]])


class TestBlockFreezing(unittest.TestCase):
    """get_blocks_to_update freezes well-fit blocks out of the torsion search."""

    def _fake_rl(self, block_qscores, freeze):
        rotor = [types.SimpleNamespace(block_id=i) for i in range(len(block_qscores))]
        best_idx = int(np.argmax(block_qscores))
        return types.SimpleNamespace(
            _rotor_tree=rotor,
            _block_qscores=[float(q) for q in block_qscores],
            _freeze_block_qscore=float(freeze),
            get_best_block_by_qscore=lambda: best_idx,
        )

    def test_freezes_well_fit_blocks(self):
        # block 0 best (0.9); block 1 well-fit (0.8 >= 0.7) -> frozen;
        # blocks 2,3 poor -> searched.
        rl = self._fake_rl([0.9, 0.8, 0.5, 0.3], freeze=0.7)
        ids = [b.block_id for b in sr.RefineLigand.get_blocks_to_update(rl)]
        self.assertEqual(ids, [2, 3])

    def test_disabled_includes_all_below_best(self):
        rl = self._fake_rl([0.9, 0.8, 0.5, 0.3], freeze=0.0)
        ids = [b.block_id for b in sr.RefineLigand.get_blocks_to_update(rl)]
        self.assertEqual(ids, [1, 2, 3])


class TestRobustBundle(unittest.TestCase):
    def test_applies_profile_when_enabled(self):
        opts = types.SimpleNamespace(
            sr2_robust=True,
            sr2_pre_minimise=False,
            sr2_centroid_trust=False,  # robust overrides this
        )
        sr._apply_robust_bundle(opts)
        self.assertTrue(opts.sr2_pre_minimise)
        self.assertTrue(opts.sr2_centroid_trust)
        self.assertEqual(opts.sr2_selection, "greedy")
        self.assertAlmostEqual(opts.sr2_centroid_trust_radius, 5.0)
        self.assertAlmostEqual(opts.sr2_centroid_trust_k, 0.4)
        self.assertTrue(opts.sr2_envelope_gate)
        self.assertAlmostEqual(opts.sr2_freeze_block_qscore, 0.7)

    def test_noop_when_disabled(self):
        opts = types.SimpleNamespace(sr2_robust=False, sr2_pre_minimise=False)
        sr._apply_robust_bundle(opts)
        self.assertFalse(opts.sr2_pre_minimise)
        self.assertFalse(hasattr(opts, "sr2_selection"))


class TestSelectionDefault(unittest.TestCase):
    def _parser(self):
        protocol_spec = importlib.import_module("protocol_spec")
        parser = argparse.ArgumentParser()
        protocol_spec.add_smart_ligand_refine2_args(parser)
        return parser

    def test_default_is_greedy(self):
        ns = self._parser().parse_args([])
        self.assertEqual(ns.sr2_selection, "greedy")

    def test_branches_opt_out(self):
        ns = self._parser().parse_args(["--sr2-selection", "branches"])
        self.assertEqual(ns.sr2_selection, "branches")


if __name__ == "__main__":
    unittest.main()
