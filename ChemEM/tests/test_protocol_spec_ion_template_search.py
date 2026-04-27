from pathlib import Path
import sys
import unittest


def _load_main_module():
    try:
        from ChemEM import __main__ as chemem_main
        return chemem_main
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[1]
        parent = str(root.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from ChemEM import __main__ as chemem_main
        return chemem_main


class TestIonTemplateSearchCLI(unittest.TestCase):
    def test_parser_exposes_its_flags_and_defaults(self):
        main = _load_main_module()
        parser = main.build_parser()

        args = parser.parse_args(["dummy.conf", "-its"])
        self.assertTrue(args.run_ion_template_search)
        self.assertFalse(args.its_auto_run_ion_fixer)
        self.assertAlmostEqual(args.its_confidence_thresh, 0.65)
        self.assertEqual(args.its_max_entry_candidates, 200)
        self.assertEqual(args.its_max_templates, 25)
        self.assertAlmostEqual(args.its_seq_identity_min, 0.35)
        self.assertAlmostEqual(args.its_local_chain_radius_a, 12.0)
        self.assertEqual(args.its_ion_elements, "")
        self.assertTrue(args.its_similarity_enabled)

    def test_protocol_order_its_before_ion_fixer(self):
        main = _load_main_module()
        parser = main.build_parser()

        args = parser.parse_args(["dummy.conf", "-its", "--ion-fixer"])
        picked = main.selected_protocols(args)
        ordered = main.resolve_protocol_order(picked, args)

        self.assertIn("ion_template_search", ordered)
        self.assertIn("ion_fixer", ordered)
        self.assertLess(ordered.index("ion_template_search"), ordered.index("ion_fixer"))


if __name__ == "__main__":
    unittest.main()
