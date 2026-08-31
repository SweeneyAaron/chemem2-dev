"""Tests for the ``--score`` CLI: scorer selection, dependencies, deprecations.

The load-bearing property here is that ``score_deps`` is *scorer-dependent*. It is
the only deps() in the registry that reads its args, and it is what makes
``--score --score-with qscore`` skip binding-site detection and segmentation
entirely -- work the old ``--rescore-poses`` always paid for.
"""
import argparse
import types
import unittest

try:
    from ChemEM.protocols.score import cli
    from ChemEM.protocols.score.scorers import SCORER_NAMES, load_scorer_cls
except ModuleNotFoundError:
    from protocols.score import cli
    from protocols.score.scorers import SCORER_NAMES, load_scorer_cls

SEGMENTATION = ("binding_site", "alpha_mask", "confidence_map")


def _args(**kwargs):
    return argparse.Namespace(**kwargs)


class TestResolveScorers(unittest.TestCase):
    def test_bare_score_defaults_to_echo(self):
        self.assertEqual(cli.resolve_scorers(_args()), ("echo",))
        self.assertEqual(cli.resolve_scorers(_args(score_with=None)), ("echo",))
        self.assertEqual(cli.resolve_scorers(_args(score_with=[])), ("echo",))

    def test_comma_list(self):
        self.assertEqual(
            cli.resolve_scorers(_args(score_with=["echo,qscore,mmgbsa"])),
            ("echo", "qscore", "mmgbsa"),
        )

    def test_repeated_flag(self):
        self.assertEqual(
            cli.resolve_scorers(_args(score_with=["echo", "qscore"])),
            ("echo", "qscore"),
        )

    def test_order_is_preserved(self):
        """The order drives CSV column order and the default ranking, so it is not
        allowed to be normalised into registry order."""
        self.assertEqual(
            cli.resolve_scorers(_args(score_with=["density,echo"])),
            ("density", "echo"),
        )

    def test_duplicates_collapse(self):
        self.assertEqual(
            cli.resolve_scorers(_args(score_with=["echo,echo", "echo"])), ("echo",)
        )

    def test_all_expands_to_every_scorer(self):
        self.assertEqual(cli.resolve_scorers(_args(score_with=["all"])), SCORER_NAMES)

    def test_all_after_a_name_keeps_that_name_first(self):
        picked = cli.resolve_scorers(_args(score_with=["mmgbsa,all"]))
        self.assertEqual(picked[0], "mmgbsa")
        self.assertEqual(set(picked), set(SCORER_NAMES))

    def test_whitespace_and_case_are_tolerated(self):
        self.assertEqual(
            cli.resolve_scorers(_args(score_with=[" ECHO , qscore "])),
            ("echo", "qscore"),
        )

    def test_a_bare_string_works(self):
        """Benchmark scripts hand-build namespaces rather than going via argparse."""
        self.assertEqual(cli.resolve_scorers(_args(score_with="echo,qscore")),
                         ("echo", "qscore"))

    def test_unknown_scorer_names_the_valid_set(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.resolve_scorers(_args(score_with=["echo,qscroe"]))
        message = str(ctx.exception)
        self.assertIn("qscroe", message)
        for name in SCORER_NAMES:
            self.assertIn(name, message)


class TestScoreDeps(unittest.TestCase):
    def test_qscore_alone_needs_nothing(self):
        self.assertEqual(cli.score_deps(_args(score_with=["qscore"])), ())

    def test_mmgbsa_alone_needs_nothing(self):
        self.assertEqual(cli.score_deps(_args(score_with=["mmgbsa"])), ())

    def test_the_cheap_scorers_together_need_nothing(self):
        self.assertEqual(
            cli.score_deps(_args(score_with=["qscore,mmgbsa"])), ()
        )

    def test_echo_needs_the_full_segmentation_chain(self):
        self.assertEqual(cli.score_deps(_args(score_with=["echo"])), SEGMENTATION)

    def test_density_full_and_box_need_nothing(self):
        for region in ("full", "box"):
            self.assertEqual(
                cli.score_deps(_args(score_with=["density"],
                                     score_density_region=region)),
                (), f"region {region}",
            )

    def test_density_site_needs_segmentation(self):
        """The segmented site map only exists once alpha_mask has run."""
        self.assertEqual(
            cli.score_deps(_args(score_with=["density"], score_density_region="site")),
            SEGMENTATION,
        )

    def test_deps_are_unioned_without_duplicates(self):
        deps = cli.score_deps(
            _args(score_with=["echo,qscore,density"], score_density_region="site")
        )
        self.assertEqual(deps, SEGMENTATION)
        self.assertEqual(len(deps), len(set(deps)))

    def test_bare_namespace_does_not_raise(self):
        """generate_custom_usage() builds throwaway parsers whose namespaces carry
        none of these attributes."""
        self.assertEqual(cli.score_deps(argparse.Namespace()), SEGMENTATION)

    def test_every_registered_scorer_resolves_and_declares_a_name(self):
        for name in SCORER_NAMES:
            cls = load_scorer_cls(name)
            self.assertEqual(cls.NAME, name)
            self.assertIsInstance(cls.deps_for(argparse.Namespace()), tuple)


class TestProtocolOrdering(unittest.TestCase):
    def test_qscore_only_run_is_a_single_protocol(self):
        from ChemEM.__main__ import resolve_protocol_order

        args = _args(score_with=["qscore"])
        self.assertEqual(resolve_protocol_order(["score"], args), ["score"])

    def test_echo_run_pulls_the_chain_in_a_valid_order(self):
        from ChemEM.__main__ import resolve_protocol_order

        order = resolve_protocol_order(["score"], _args(score_with=["echo"]))
        self.assertEqual(order[-1], "score")
        for dep in SEGMENTATION:
            self.assertIn(dep, order)
        # alpha_mask depends on binding_site and confidence_map.
        self.assertLess(order.index("binding_site"), order.index("alpha_mask"))
        self.assertLess(order.index("confidence_map"), order.index("alpha_mask"))


class TestParserWiring(unittest.TestCase):
    def _parse(self, argv):
        from ChemEM.__main__ import build_parser
        return build_parser().parse_args(["conf.txt"] + argv)

    def test_score_flag_and_short_alias(self):
        self.assertTrue(self._parse(["--score"]).run_score)
        self.assertTrue(self._parse(["-sc"]).run_score)

    def test_rp_short_alias_survived_the_registry_removal(self):
        """-rp used to be generated from SHORT_ALIASES; it is now hand-registered
        on the deprecated flag, which is easy to drop by accident."""
        self.assertTrue(self._parse(["-rp"]).score_alias_rescore_poses)

    def test_sigma_ref_is_shared_not_score_namespaced(self):
        """The orchestrator and smart_refine_2 both read `sigma_ref`."""
        self.assertEqual(self._parse(["--sigma-ref", "0.8"]).sigma_ref, 0.8)

    def test_per_atom_alias_still_parses(self):
        self.assertTrue(self._parse(["--per-atom"]).score_qscore_per_atom)

    def test_dock_mmgbsa_rescore_keeps_its_old_spelling(self):
        """Dock's bare --rescore (post-docking MM-GBSA) is a different thing from
        the old --rescore-poses (ECHO); it was renamed, but must keep working."""
        self.assertTrue(self._parse(["--dock", "--dock-rescore-mmgbsa"]).rescore)
        self.assertTrue(self._parse(["--dock", "--rescore"]).rescore)
        self.assertFalse(self._parse(["--dock"]).rescore)

    def test_rescore_and_rescore_poses_stay_distinct(self):
        """The two flags share a prefix; argparse must not conflate them."""
        args = self._parse(["--rescore"])
        self.assertTrue(args.rescore)
        self.assertFalse(args.score_alias_rescore_poses)

        args = self._parse(["--rescore-poses"])
        self.assertFalse(args.rescore)
        self.assertTrue(args.score_alias_rescore_poses)


class TestBackCompat(unittest.TestCase):
    def _parse_and_translate(self, argv):
        from ChemEM.__main__ import build_parser

        args = build_parser().parse_args(["conf.txt"] + argv)
        cli.apply_score_back_compat(args)
        return args

    def test_rescore_poses_becomes_score_with_echo(self):
        args = self._parse_and_translate(["--rescore-poses"])
        self.assertTrue(args.run_score)
        self.assertEqual(cli.resolve_scorers(args), ("echo",))
        # The old protocol wrote ranked SDFs, so the alias keeps doing that.
        self.assertTrue(args.score_sdf)

    def test_mapq_score_becomes_score_with_qscore(self):
        args = self._parse_and_translate(["--mapq-score"])
        self.assertTrue(args.run_score)
        self.assertEqual(cli.resolve_scorers(args), ("qscore",))
        self.assertEqual(cli.score_deps(args), ())

    def test_both_aliases_select_both_scorers(self):
        args = self._parse_and_translate(["--rescore-poses", "--mapq-score"])
        self.assertEqual(cli.resolve_scorers(args), ("echo", "qscore"))

    def test_rescore_no_sdf_still_suppresses_the_sdfs(self):
        args = self._parse_and_translate(["--rescore-poses", "--rescore-no-sdf"])
        self.assertFalse(args.score_sdf)

    def test_an_explicit_score_with_wins_over_the_alias_default(self):
        args = self._parse_and_translate(
            ["--rescore-poses", "--score-with", "qscore"]
        )
        self.assertEqual(cli.resolve_scorers(args), ("qscore",))

    def test_is_idempotent(self):
        args = self._parse_and_translate(["--rescore-poses"])
        before = dict(vars(args))
        cli.apply_score_back_compat(args)
        self.assertEqual(vars(args), before)

    def test_no_alias_is_a_no_op(self):
        args = self._parse_and_translate(["--score", "--score-with", "density"])
        self.assertEqual(cli.resolve_scorers(args), ("density",))
        self.assertFalse(args.score_sdf)

    def test_old_rescore_option_spellings_reach_the_new_dests(self):
        args = self._parse_and_translate([
            "--rescore-poses",
            "--rescore-engine", "docking_v2",
            "--rescore-interaction-cutoff", "7.0",
            "--rescore-electro-clamp", "3.0",
            "--rescore-out", "rescore",
            "--rescore-site", "2",
        ])
        self.assertEqual(args.score_echo_engine, "docking_v2")
        self.assertEqual(args.score_echo_interaction_cutoff, 7.0)
        self.assertEqual(args.score_echo_electro_clamp, 3.0)
        self.assertEqual(args.score_out, "rescore")
        self.assertEqual(args.score_site, "2")


class TestMMGBSAWriteMinimised(unittest.TestCase):
    """--score-mmgbsa-write-minimised is meaningless without the minimisation that
    produces the relaxed pose, so it fails loudly at setup rather than writing
    nothing (or, worse, writing the unrelaxed input back out)."""

    def _scorer(self, **opts):
        cls = load_scorer_cls("mmgbsa")
        system = types.SimpleNamespace(protein=object(), log=lambda m: None)
        return cls(system, argparse.Namespace(**opts)), system

    def test_requires_minimise(self):
        scorer, _ = self._scorer(score_mmgbsa_write_minimised=True,
                                 score_mmgbsa_minimise=False)
        with self.assertRaises(ValueError) as ctx:
            scorer.setup_run(None)
        self.assertIn("--score-mmgbsa-minimise", str(ctx.exception))

    def test_accepted_with_minimise(self):
        scorer, _ = self._scorer(score_mmgbsa_write_minimised=True,
                                 score_mmgbsa_minimise=True)
        scorer.setup_run(None)
        self.assertTrue(scorer.write_minimised)

    def test_column_only_declared_when_requested(self):
        off, _ = self._scorer(score_mmgbsa_minimise=True)
        on, _ = self._scorer(score_mmgbsa_minimise=True,
                             score_mmgbsa_write_minimised=True)
        self.assertNotIn("mmgbsa_minimised_sdf", off.extra_columns())
        self.assertIn("mmgbsa_minimised_sdf", on.extra_columns())


if __name__ == "__main__":
    unittest.main()
