"""Tests for the ``--score`` protocol loop.

Two guarantees, both easy to break by accident:

  * **no scorer pays for another's setup.** ``--score-with qscore`` must never touch
    ``system.binding_sites``. That is not just a ``score_deps`` question -- if the
    protocol built the site list eagerly, a qscore-only run would still crash on a
    config with no binding site.
  * **failures are contained per scorer.** One scorer blowing up on one pose must
    cost that pose only that scorer's columns.
"""
import argparse
import types
import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from ChemEM.protocols.score.context import ScoreContext
    from ChemEM.protocols.score.score_poses import ScorePoses
    from ChemEM.protocols.score.scorers.base import PoseScorer
except ModuleNotFoundError:
    from protocols.score.context import ScoreContext
    from protocols.score.score_poses import ScorePoses
    from protocols.score.scorers.base import PoseScorer


def _mol():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    return mol


class _Ligand:
    def __init__(self, mol, source="poses.sdf", identifier="LIG"):
        self.mol = mol
        self.input = source
        self.identifier = identifier


class _ExplodingSites(dict):
    """Stands in for a system whose binding sites cannot be built.

    ``__bool__`` is overridden because an empty dict is falsy, and every reader
    writes ``getattr(system, "binding_sites", None) or {}`` -- so a falsy fake would
    be swapped for a plain ``{}`` and never touched, making the negative control
    pass for the wrong reason.
    """

    def __bool__(self):
        return True

    def items(self):
        raise AssertionError("binding_sites must not be touched")


def _system(tmpdir, **options):
    mol = _mol()
    system = types.SimpleNamespace(
        ligand=[_Ligand(mol)],
        output=str(tmpdir),
        density_map=None,
        protein=object(),
        platform="CPU",
        options=argparse.Namespace(**options),
    )
    system.binding_sites = _ExplodingSites()
    system.log = lambda msg: None
    return system


# --------------------------------------------------------------------------- #
# Fake scorers
# --------------------------------------------------------------------------- #
class _Cheap(PoseScorer):
    NAME = "cheap"
    HEADLINE = "cheap_value"
    COLUMNS = ("cheap_value",)
    NEEDS_SITE = False

    def score(self, pose, row):
        row["cheap_value"] = float(len(pose.coords))


class _NeedsSite(PoseScorer):
    NAME = "sited"
    HEADLINE = "sited_value"
    COLUMNS = ("sited_value",)
    NEEDS_SITE = True

    def score(self, pose, row):
        site_id, _bs = pose.site()
        row["sited_value"] = str(site_id)


class _Explodes(PoseScorer):
    NAME = "boom"
    HEADLINE = "boom_value"
    COLUMNS = ("boom_value",)

    def score(self, pose, row):
        raise RuntimeError("scoring blew up")


class _MutatesInPreScore(PoseScorer):
    """Stands in for ECHO's hydrogen relaxation."""
    NAME = "mutator"
    COLUMNS = ()
    restored = False

    def pre_score(self, pose, row):
        conf = pose.mol.GetConformer(pose.conf_id)
        self._before = conf.GetPositions().copy()
        conf.SetAtomPosition(0, (99.0, 99.0, 99.0))
        pose.touch()

    def post_score(self, pose, row):
        conf = pose.mol.GetConformer(pose.conf_id)
        for i, xyz in enumerate(self._before):
            conf.SetAtomPosition(i, tuple(float(v) for v in xyz))
        type(self).restored = True


class _ObservesCoords(PoseScorer):
    NAME = "observer"
    COLUMNS = ("observer_x0",)

    def score(self, pose, row):
        row["observer_x0"] = float(pose.coords[0][0])


# --------------------------------------------------------------------------- #
def _run(system, scorers):
    proto = ScorePoses(system)
    proto._get_output()
    proto.scorers = scorers
    ctx = ScoreContext(system, system.options, proto.output, "case", scorers)
    for scorer in scorers:
        scorer.setup_run(ctx)

    from ChemEM.protocols.score.poses import iter_ligands
    timings = {s.NAME: 0.0 for s in scorers}
    failures = {s.NAME: 0 for s in scorers}
    for lig_ctx, poses in iter_ligands(system, ctx):
        proto._score_ligand(ctx, lig_ctx, poses, timings, failures)
    return proto.rows, failures


class TestSiteLaziness(unittest.TestCase):
    def test_a_site_free_scorer_never_touches_binding_sites(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            system = _system(tmp)
            rows, failures = _run(system, [_Cheap(system, system.options)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(failures["cheap"], 0)
        self.assertGreater(rows[0]["cheap_value"], 0)

    def test_a_site_needing_scorer_does_touch_them(self):
        """The negative control: if this passes trivially, the test above proves
        nothing about laziness."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            system = _system(tmp)
            rows, failures = _run(system, [_NeedsSite(system, system.options)])
        self.assertEqual(failures["sited"], 1)
        self.assertIn("binding_sites must not be touched", rows[0]["sited_error"])

    def test_needs_site_is_reported_on_the_context(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            system = _system(tmp)
            cheap = _Cheap(system, system.options)
            sited = _NeedsSite(system, system.options)
            self.assertFalse(
                ScoreContext(system, system.options, tmp, "c", [cheap]).needs_site
            )
            self.assertTrue(
                ScoreContext(system, system.options, tmp, "c", [cheap, sited]).needs_site
            )


class TestFailureContainment(unittest.TestCase):
    def test_one_failing_scorer_does_not_cost_the_others(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            system = _system(tmp)
            rows, failures = _run(system, [
                _Explodes(system, system.options),
                _Cheap(system, system.options),
            ])
        row = rows[0]
        self.assertIn("RuntimeError: scoring blew up", row["boom_error"])
        self.assertNotIn("boom_value", row)
        self.assertGreater(row["cheap_value"], 0)   # the other scorer still ran
        self.assertEqual(failures, {"boom": 1, "cheap": 0})

    def test_a_failure_is_counted_once_per_pose(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            system = _system(tmp)
            _rows, failures = _run(system, [_Explodes(system, system.options)])
        self.assertEqual(failures["boom"], 1)


class TestPreScoreOrdering(unittest.TestCase):
    def test_every_scorer_sees_the_mutated_geometry(self):
        """ECHO's hydrogen relaxation runs before ANY scorer scores, so MM-GBSA --
        which is all-atom -- must see the coordinates ECHO scored, not the input."""
        import tempfile
        _MutatesInPreScore.restored = False
        with tempfile.TemporaryDirectory() as tmp:
            system = _system(tmp)
            rows, _failures = _run(system, [
                _MutatesInPreScore(system, system.options),
                _ObservesCoords(system, system.options),
            ])
        self.assertAlmostEqual(rows[0]["observer_x0"], 99.0)

    def test_post_score_runs_after_every_scorer(self):
        import tempfile
        _MutatesInPreScore.restored = False
        with tempfile.TemporaryDirectory() as tmp:
            system = _system(tmp)
            mol = system.ligand[0].mol
            before = mol.GetConformer(0).GetPositions().copy()
            _run(system, [
                _MutatesInPreScore(system, system.options),
                _ObservesCoords(system, system.options),
            ])
            after = mol.GetConformer(0).GetPositions()
        self.assertTrue(_MutatesInPreScore.restored)
        np.testing.assert_allclose(before, after, atol=1e-9)

    def test_pre_score_order_is_independent_of_scorer_order(self):
        """The mutator listed second must still run its pre_score before the
        observer's score."""
        import tempfile
        _MutatesInPreScore.restored = False
        with tempfile.TemporaryDirectory() as tmp:
            system = _system(tmp)
            rows, _failures = _run(system, [
                _ObservesCoords(system, system.options),
                _MutatesInPreScore(system, system.options),
            ])
        self.assertAlmostEqual(rows[0]["observer_x0"], 99.0)


if __name__ == "__main__":
    unittest.main()
