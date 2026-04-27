import unittest
from types import SimpleNamespace

try:
    from ChemEM.protocols.refine.search_refine.acceptance import (
        AcceptOutcome,
        BasinHoppingAccept,
        GreedyAccept,
        MetropolisAccept,
        get_acceptance,
    )
except ModuleNotFoundError:
    from protocols.refine.search_refine.acceptance import (
        AcceptOutcome,
        BasinHoppingAccept,
        GreedyAccept,
        MetropolisAccept,
        get_acceptance,
    )


def _opts(**kw):
    defaults = dict(
        sr_min_delta=1e-4,
        sr_max_outer_iter=50,
        sr_accept_temp_start=0.05,
        sr_accept_temp_end=0.005,
        sr_basin_hop_stale=3,
        sr_seed=1,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _rec(score):
    return {"final_score": float(score), "score": float(score)}


class TestGreedyAccept(unittest.TestCase):
    def test_major_accept_when_delta_exceeds_min(self):
        s = GreedyAccept(_opts(sr_min_delta=1e-4))
        picked, outcome = s.decide(_rec(0.5), [_rec(0.6), _rec(0.55)], iter_idx=1)
        self.assertEqual(outcome, AcceptOutcome.MAJOR)
        self.assertAlmostEqual(picked["final_score"], 0.6)

    def test_micro_accept_when_delta_small(self):
        s = GreedyAccept(_opts(sr_min_delta=0.01))
        picked, outcome = s.decide(_rec(0.5), [_rec(0.505)], iter_idx=1)
        self.assertEqual(outcome, AcceptOutcome.MICRO)
        self.assertAlmostEqual(picked["final_score"], 0.505)

    def test_reject_when_no_improvement(self):
        s = GreedyAccept(_opts())
        accepted = _rec(0.5)
        picked, outcome = s.decide(accepted, [_rec(0.4), _rec(0.45)], iter_idx=1)
        self.assertEqual(outcome, AcceptOutcome.REJECT)
        self.assertIs(picked, accepted)

    def test_reject_when_empty(self):
        s = GreedyAccept(_opts())
        accepted = _rec(0.5)
        picked, outcome = s.decide(accepted, [], iter_idx=1)
        self.assertEqual(outcome, AcceptOutcome.REJECT)
        self.assertIs(picked, accepted)


class TestMetropolisAccept(unittest.TestCase):
    def test_improvement_always_accepted(self):
        s = MetropolisAccept(_opts(sr_min_delta=1e-4))
        picked, outcome = s.decide(_rec(0.5), [_rec(0.6)], iter_idx=1)
        self.assertEqual(outcome, AcceptOutcome.MAJOR)
        self.assertAlmostEqual(picked["final_score"], 0.6)

    def test_uses_rng_deterministically(self):
        # Same seed → identical sequence of decisions given worse proposals.
        opts = _opts(sr_seed=42, sr_accept_temp_start=0.1, sr_accept_temp_end=0.1)
        s1 = MetropolisAccept(opts)
        s2 = MetropolisAccept(opts)

        decisions_1 = []
        decisions_2 = []
        for t in range(10):
            d1, _ = s1.decide(_rec(0.5), [_rec(0.4)], iter_idx=t + 1)
            d2, _ = s2.decide(_rec(0.5), [_rec(0.4)], iter_idx=t + 1)
            decisions_1.append(d1["final_score"])
            decisions_2.append(d2["final_score"])

        self.assertEqual(decisions_1, decisions_2)


class TestBasinHoppingAccept(unittest.TestCase):
    def test_emits_perturb_after_stale_rejects(self):
        s = BasinHoppingAccept(_opts(sr_basin_hop_stale=2))
        accepted = _rec(0.5)
        _, o1 = s.decide(accepted, [_rec(0.4)], iter_idx=1)
        _, o2 = s.decide(accepted, [_rec(0.4)], iter_idx=2)
        self.assertEqual(o1, AcceptOutcome.REJECT)
        self.assertEqual(o2, AcceptOutcome.PERTURB)

    def test_resets_stale_on_major(self):
        s = BasinHoppingAccept(_opts(sr_basin_hop_stale=2, sr_min_delta=1e-4))
        _, o_rej = s.decide(_rec(0.5), [_rec(0.4)], iter_idx=1)
        self.assertEqual(o_rej, AcceptOutcome.REJECT)
        _, o_major = s.decide(_rec(0.5), [_rec(0.7)], iter_idx=2)
        self.assertEqual(o_major, AcceptOutcome.MAJOR)
        # Subsequent single reject should not already be perturb — counter reset.
        _, o_rej2 = s.decide(_rec(0.7), [_rec(0.5)], iter_idx=3)
        self.assertEqual(o_rej2, AcceptOutcome.REJECT)


class TestFactory(unittest.TestCase):
    def test_registry(self):
        self.assertIsInstance(get_acceptance("greedy", _opts()), GreedyAccept)
        self.assertIsInstance(get_acceptance("metropolis", _opts()), MetropolisAccept)
        self.assertIsInstance(get_acceptance("basin_hopping", _opts()), BasinHoppingAccept)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_acceptance("nope", _opts())


if __name__ == "__main__":
    unittest.main()
