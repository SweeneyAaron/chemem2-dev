import unittest

import numpy as np

try:
    from ChemEM.protocols.orchestrator.state import FinalAssignment, PoseCandidate
    from ChemEM.protocols.orchestrator.triage import (
        assignment_evaluation_rows,
        composite,
        gate1_select,
        gate2_select,
        gate3_select,
    )
except ModuleNotFoundError:
    from protocols.orchestrator.state import FinalAssignment, PoseCandidate
    from protocols.orchestrator.triage import (
        assignment_evaluation_rows,
        composite,
        gate1_select,
        gate2_select,
        gate3_select,
    )


def _pc(site, lig, pose, *, q=None, m=None, score=0.0, stage="docked", metrics=None):
    candidate = PoseCandidate(
        site_id=site,
        ligand_idx=lig,
        pose_idx=pose,
        coords=np.zeros((3, 3)),
        dock_score=float(score),
        qscore=q,
        mmgbsa=m,
        stage=stage,
    )
    if metrics:
        candidate.metrics.update(metrics)
    return candidate


class TestGate1(unittest.TestCase):
    def test_per_site_top_k_by_qscore(self):
        cands = {
            "S1": [_pc("S1", 0, 0, q=0.4), _pc("S1", 1, 0, q=0.7), _pc("S1", 2, 0, q=0.6)],
            "S2": [_pc("S2", 0, 0, q=0.2), _pc("S2", 0, 1, q=0.5)],
        }
        out = gate1_select(cands, top_k=2)
        self.assertEqual([c.qscore for c in out["S1"]], [0.7, 0.6])
        self.assertEqual([c.qscore for c in out["S2"]], [0.5, 0.2])

    def test_drops_candidates_with_no_qscore(self):
        cands = {"S1": [_pc("S1", 0, 0, q=None), _pc("S1", 1, 0, q=0.3)]}
        out = gate1_select(cands, top_k=5)
        self.assertEqual(len(out["S1"]), 1)
        self.assertEqual(out["S1"][0].ligand_idx, 1)

    def test_keeps_all_when_fewer_than_k(self):
        cands = {"S1": [_pc("S1", 0, 0, q=0.5)]}
        out = gate1_select(cands, top_k=10)
        self.assertEqual(len(out["S1"]), 1)

    def test_empty_site_returns_empty_list(self):
        out = gate1_select({"S1": []}, top_k=3)
        self.assertEqual(out["S1"], [])

    def test_accepts_shared_rank_kwargs_from_orchestrator(self):
        cands = {"S1": [_pc("S1", 0, 0, q=0.5)]}
        out = gate1_select(
            cands,
            top_k=1,
            score_mode="coverage",
            w_qscore=1.0,
            w_mmgbsa=0.0,
            w_qtail=0.5,
            w_density_coverage=2.0,
            w_density_precision=0.5,
            w_density_ccc=0.5,
        )
        self.assertEqual(len(out["S1"]), 1)


class TestGate2Composite(unittest.TestCase):
    def test_higher_qscore_wins_when_mmgbsa_tied(self):
        cands = [_pc("S1", 0, 0, q=0.3, m=-10.0), _pc("S1", 1, 0, q=0.7, m=-10.0)]
        scores = composite(cands, w_qscore=1.0, w_mmgbsa=0.5)
        self.assertGreater(scores[1], scores[0])

    def test_lower_mmgbsa_wins_when_qscore_tied(self):
        cands = [_pc("S1", 0, 0, q=0.5, m=-5.0), _pc("S1", 1, 0, q=0.5, m=-20.0)]
        scores = composite(cands, w_qscore=1.0, w_mmgbsa=0.5)
        self.assertGreater(scores[1], scores[0])

    def test_missing_mmgbsa_zeroes_that_channel(self):
        cands = [_pc("S1", 0, 0, q=0.3, m=None), _pc("S1", 1, 0, q=0.7, m=None)]
        scores = composite(cands, w_qscore=1.0, w_mmgbsa=0.5)
        self.assertGreater(scores[1], scores[0])

    def test_per_site_top_k(self):
        cands = {
            "S1": [
                _pc("S1", 0, 0, q=0.3, m=-5.0),
                _pc("S1", 1, 0, q=0.7, m=-15.0),
                _pc("S1", 2, 0, q=0.5, m=-10.0),
            ]
        }
        out = gate2_select(cands, top_k=2, w_qscore=1.0, w_mmgbsa=0.5)
        ligs = [c.ligand_idx for c in out["S1"]]
        self.assertIn(1, ligs)
        self.assertEqual(len(ligs), 2)

    def test_coverage_mode_can_beat_mean_qscore_bias(self):
        small = _pc("S1", 0, 0, q=0.8)
        small.metrics.update(
            q_low_tail=0.8,
            density_coverage=0.25,
            density_precision=0.95,
            density_ccc=0.7,
        )
        large = _pc("S1", 1, 0, q=0.7)
        large.metrics.update(
            q_low_tail=0.65,
            density_coverage=0.85,
            density_precision=0.8,
            density_ccc=0.75,
        )
        out = gate1_select(
            {"S1": [small, large]},
            top_k=1,
            score_mode="coverage",
            w_qscore=1.0,
            w_qtail=0.5,
            w_density_coverage=2.0,
            w_density_precision=0.5,
            w_density_ccc=0.5,
        )
        self.assertEqual(out["S1"][0].ligand_idx, 1)


class TestGate3FinalPick(unittest.TestCase):
    def test_per_site_winner(self):
        cands = {
            "S1": [_pc("S1", 0, 0, q=0.4, m=-5.0), _pc("S1", 1, 0, q=0.8, m=-12.0)],
            "S2": [_pc("S2", 0, 0, q=0.2, m=-3.0), _pc("S2", 2, 0, q=0.6, m=-7.0)],
        }
        out = gate3_select(cands, w_qscore=1.0, w_mmgbsa=0.5)
        site_to_lig = {a.site_id: a.ligand_idx for a in out}
        self.assertEqual(site_to_lig, {"S1": 1, "S2": 2})

    def test_capacity_aware_distinct_chain_ids_for_repeated_ligand(self):
        # Same ligand wins both sites -> different chain IDs.
        cands = {
            "S1": [_pc("S1", 0, 0, q=0.9, m=-15.0)],
            "S2": [_pc("S2", 0, 0, q=0.9, m=-15.0)],
        }
        out = gate3_select(cands, w_qscore=1.0, w_mmgbsa=0.5)
        self.assertEqual(len(out), 2)
        self.assertEqual([a.ligand_idx for a in out], [0, 0])
        chains = sorted(a.chain_id for a in out)
        self.assertEqual(chains, ["A", "B"])

    def test_skips_empty_sites(self):
        out = gate3_select({"S1": [], "S2": [_pc("S2", 0, 0, q=0.5, m=-1.0)]},
                           w_qscore=1.0, w_mmgbsa=0.5)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].site_id, "S2")

    def test_tie_break_by_tighter_mmgbsa(self):
        # Two candidates with identical composite by construction:
        # both have only one entry per site, so z-scores collapse to 0.
        # Tie-break must use lower mmgbsa to pick a winner.
        cands = {
            "S1": [
                _pc("S1", 0, 0, q=0.5, m=-5.0),
                _pc("S1", 1, 0, q=0.5, m=-15.0),
            ]
        }
        out = gate3_select(cands, w_qscore=1.0, w_mmgbsa=0.5)
        # With identical qscores, MMGBSA channel breaks the tie; lig 1 has tighter mmgbsa.
        self.assertEqual(out[0].ligand_idx, 1)


class TestAbsoluteAssignmentScoring(unittest.TestCase):
    def _metric_candidate(self, site, lig, *, q, tail, cov, precision, ccc, overlap, mm=-200.0):
        return _pc(
            site,
            lig,
            0,
            q=q,
            m=mm,
            metrics={
                "q_low_tail": tail,
                "density_coverage": cov,
                "density_precision": precision,
                "density_ccc": ccc,
                "density_overlap": overlap,
            },
            stage="smart_refine_2",
        )

    def test_absolute_replay_6x3x_gate3_expected_ligands(self):
        rows = {
            "7": [
                self._metric_candidate("7", 0, q=0.7468564169747489, tail=0.7241500417391459, cov=0.3510809918235827, precision=0.5001384285585279, ccc=0.6496617488637334, overlap=0.41903352556497914, mm=-281.3361720804023),
                self._metric_candidate("7", 1, q=0.6814588963985443, tail=0.4757925669352214, cov=0.61358674010545, precision=0.3527234037036876, ccc=0.7083433999768879, overlap=0.465251472238568, mm=-184.25152830411935),
            ],
            "19": [
                self._metric_candidate("19", 0, q=0.7993264453751701, tail=0.7468394239743551, cov=0.3520402844799515, precision=0.43615608799733224, ccc=0.6414030136242751, overlap=0.3918475638513574, mm=-267.87562560944025),
                self._metric_candidate("19", 1, q=0.666180794686079, tail=0.5277638410528501, cov=0.5541762082190533, precision=0.3227782924587211, ccc=0.6696341571471821, overlap=0.4229558786305693, mm=-168.34563554358738),
            ],
            "24": [
                self._metric_candidate("24", 0, q=0.7332574512277331, tail=0.5825980206330618, cov=0.6367430815474785, precision=0.3140950544571376, ccc=0.6381841362741865, overlap=0.4472111949335133, mm=-330.07941873305754),
                self._metric_candidate("24", 1, q=0.09759567081928253, tail=-0.42566970984141034, cov=0.5831929496440819, precision=0.14977061340063125, ccc=0.40641696940310706, overlap=0.29555393064514205, mm=12283.191147032936),
            ],
            "30": [
                self._metric_candidate("30", 0, q=0.6829358211585453, tail=0.5619748334089915, cov=0.3764804912049831, precision=0.38187994233278494, ccc=0.5908429511728683, overlap=0.37917060575785344, mm=-285.8050028092548),
                self._metric_candidate("30", 1, q=0.5479879431426525, tail=0.2843948577841123, cov=0.49679346200274804, precision=0.2519381475901434, ccc=0.5686752577523795, overlap=0.3537765087913742, mm=-132.78725482036043),
            ],
            "54": [
                self._metric_candidate("54", 0, q=0.6631176940032414, tail=0.5288473665714264, cov=0.5664124554321867, precision=0.28516576950349853, ccc=0.6232780235029651, overlap=0.40189730492961206, mm=-316.42570169386454),
                self._metric_candidate("54", 1, q=0.1719059322029352, tail=-0.3022661432623863, cov=0.6330515727356291, precision=0.14159996719707968, ccc=0.4250154157480379, overlap=0.2993656245404813, mm=11518456401.03354),
            ],
            "59": [
                self._metric_candidate("59", 0, q=0.7342880708830697, tail=0.6546895106633505, cov=0.3711498648281906, precision=0.5286481316986568, ccc=0.6901455844018589, overlap=0.44295336393533796, mm=-280.45116176146985),
                self._metric_candidate("59", 1, q=0.7058040529489518, tail=0.6012398600578308, cov=0.5392112771133537, precision=0.37252560200345114, ccc=0.7140415273463995, overlap=0.44829616139395885, mm=-212.76980253478087),
            ],
        }
        out = gate3_select(
            rows,
            w_qscore=0.5,
            w_mmgbsa=0.5,
            score_mode="absolute",
            w_qtail=0.25,
            w_density_coverage=5.0,
            w_density_ccc=1.0,
            w_density_overlap=1.0,
            min_assignment_score=0.0,
            min_density_coverage=0.0,
            min_assignment_margin=0.0,
        )
        self.assertEqual(
            {a.site_id: a.ligand_idx for a in out},
            {"7": 1, "19": 1, "24": 0, "30": 1, "54": 0, "59": 1},
        )

    def test_absolute_score_prefers_coverage_over_tiny_qscore_edge(self):
        small = self._metric_candidate(
            "S1", 0, q=0.81, tail=0.80, cov=0.30, precision=0.95, ccc=0.70, overlap=0.53
        )
        large = self._metric_candidate(
            "S1", 1, q=0.80, tail=0.72, cov=0.62, precision=0.60, ccc=0.72, overlap=0.61
        )
        out = gate3_select(
            {"S1": [small, large]},
            w_qscore=0.5,
            w_mmgbsa=0.5,
            score_mode="absolute",
            w_qtail=0.25,
            w_density_coverage=5.0,
            w_density_ccc=1.0,
            w_density_overlap=1.0,
            min_assignment_score=0.0,
            min_density_coverage=0.0,
            min_assignment_margin=0.0,
        )
        self.assertEqual(out[0].ligand_idx, 1)

    def test_cross_ligand_mmgbsa_cannot_flip_absolute_identity(self):
        map_fit = self._metric_candidate(
            "S1", 1, q=0.70, tail=0.55, cov=0.62, precision=0.40, ccc=0.70, overlap=0.50, mm=1000.0
        )
        energy_trap = self._metric_candidate(
            "S1", 0, q=0.72, tail=0.58, cov=0.35, precision=0.55, ccc=0.62, overlap=0.42, mm=-999999.0
        )
        out = gate3_select(
            {"S1": [energy_trap, map_fit]},
            w_qscore=0.5,
            w_mmgbsa=1000.0,
            score_mode="absolute",
            w_qtail=0.25,
            w_density_coverage=5.0,
            w_density_ccc=1.0,
            w_density_overlap=1.0,
            min_assignment_score=0.0,
            min_density_coverage=0.0,
            min_assignment_margin=0.0,
        )
        self.assertEqual(out[0].ligand_idx, 1)

    def test_same_ligand_mmgbsa_can_choose_pose_inside_mapfit_window(self):
        pose_a = self._metric_candidate(
            "S1", 0, q=0.70, tail=0.60, cov=0.60, precision=0.40, ccc=0.70, overlap=0.50, mm=-10.0
        )
        pose_b = self._metric_candidate(
            "S1", 0, q=0.70, tail=0.60, cov=0.595, precision=0.40, ccc=0.70, overlap=0.50, mm=-50.0
        )
        pose_b.pose_idx = 1
        out = gate3_select(
            {"S1": [pose_a, pose_b]},
            w_qscore=0.5,
            w_mmgbsa=0.5,
            score_mode="absolute",
            w_qtail=0.25,
            w_density_coverage=5.0,
            w_density_ccc=1.0,
            w_density_overlap=1.0,
            same_ligand_mmgbsa_window=0.15,
            min_assignment_score=0.0,
            min_density_coverage=0.0,
            min_assignment_margin=0.0,
        )
        self.assertEqual(out[0].pose_idx, 1)

    def test_assignment_evaluation_labels(self):
        tp = FinalAssignment(
            site_id="7",
            ligand_idx=1,
            pose_idx=0,
            coords=np.zeros((1, 3)),
            qscore=0.7,
            mmgbsa=None,
            composite=4.0,
            stage="smart_refine_2",
            chain_id="A",
            metrics={"assignment_score": 4.0},
            rank_score=4.0,
        )
        wrong = FinalAssignment(
            site_id="19",
            ligand_idx=0,
            pose_idx=0,
            coords=np.zeros((1, 3)),
            qscore=0.7,
            mmgbsa=None,
            composite=4.0,
            stage="smart_refine_2",
            chain_id="B",
            metrics={"assignment_score": 4.0},
            rank_score=4.0,
        )
        fp = FinalAssignment(
            site_id="63",
            ligand_idx=0,
            pose_idx=0,
            coords=np.zeros((1, 3)),
            qscore=0.7,
            mmgbsa=None,
            composite=4.0,
            stage="smart_refine_2",
            chain_id="C",
            metrics={"assignment_score": 4.0},
            rank_score=4.0,
        )
        rejected = gate3_select(
            {
                "54": [
                    self._metric_candidate(
                        "54", 0, q=0.2, tail=0.1, cov=0.1, precision=0.2, ccc=0.2, overlap=0.1
                    )
                ],
                "65": [
                    self._metric_candidate(
                        "65", 0, q=0.2, tail=0.1, cov=0.1, precision=0.2, ccc=0.2, overlap=0.1
                    )
                ],
            },
            w_qscore=0.5,
            w_mmgbsa=0.5,
            score_mode="absolute",
            w_qtail=0.25,
            w_density_coverage=5.0,
            w_density_ccc=1.0,
            w_density_overlap=1.0,
            return_rejections=True,
        )[1]
        rows, summary = assignment_evaluation_rows(
            [tp, wrong, fp],
            rejected,
            {"7": 1, "19": 1, "30": 1, "54": 0},
        )
        labels = {row["site_id"]: row["label"] for row in rows}
        self.assertEqual(labels["7"], "true_positive")
        self.assertEqual(labels["19"], "wrong_ligand")
        self.assertEqual(labels["30"], "false_negative_site")
        self.assertEqual(labels["54"], "false_negative_site")
        self.assertEqual(labels["63"], "false_positive_site")
        self.assertEqual(labels["65"], "true_negative_rejected")
        self.assertEqual(summary["counts"]["true_positive"], 1)


if __name__ == "__main__":
    unittest.main()
