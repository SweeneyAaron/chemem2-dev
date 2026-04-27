import unittest

import numpy as np

try:
    from ChemEM.protocols.orchestrator.state import PoseCandidate
    from ChemEM.protocols.orchestrator.triage import (
        composite,
        gate1_select,
        gate2_select,
        gate3_select,
    )
except ModuleNotFoundError:
    from protocols.orchestrator.state import PoseCandidate
    from protocols.orchestrator.triage import (
        composite,
        gate1_select,
        gate2_select,
        gate3_select,
    )


def _pc(site, lig, pose, *, q=None, m=None, score=0.0, stage="docked"):
    return PoseCandidate(
        site_id=site,
        ligand_idx=lig,
        pose_idx=pose,
        coords=np.zeros((3, 3)),
        dock_score=float(score),
        qscore=q,
        mmgbsa=m,
        stage=stage,
    )


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


if __name__ == "__main__":
    unittest.main()
