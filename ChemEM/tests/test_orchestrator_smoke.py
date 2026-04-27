"""Integration smoke test for the SmartOrchestrator end-to-end.

Skipped unless CHEMEM_TEST_DATA points to a valid ChemEM config (e.g. the
project's 7jjo_conf.txt). To run:

    CHEMEM_TEST_DATA=/path/to/7jjo_conf.txt \
        python -m unittest ChemEM.tests.test_orchestrator_smoke -v
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

CONFIG_ENV = "CHEMEM_TEST_DATA"


@unittest.skipUnless(
    os.environ.get(CONFIG_ENV) and os.path.exists(os.environ[CONFIG_ENV]),
    f"set {CONFIG_ENV} to a ChemEM config to run this smoke test",
)
class SmartOrchestratorSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = os.environ[CONFIG_ENV]
        self.tmpdir = tempfile.mkdtemp(prefix="orch_smoke_")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_runs_to_assembly(self) -> None:
        from ChemEM.__main__ import (
            apply_overrides,
            build_parser,
            build_pipeline,
            load_system,
            resolve_protocol_order,
            selected_protocols,
        )

        argv = [
            self.config_path,
            "-o",
            "--orch-skip-mmgbsa",
            "--orch-skip-search-refine",
            "--output",
            self.tmpdir,
        ]
        args = build_parser().parse_args(argv)
        system = load_system(self.config_path)
        system.options = args
        apply_overrides(system, args)

        order = resolve_protocol_order(selected_protocols(args), args)
        build_pipeline(system, order)
        system.run()

        out = os.path.join(self.tmpdir, "orchestrator")
        self.assertTrue(os.path.isdir(out), out)
        self.assertTrue(os.path.exists(os.path.join(out, "final_complex.pdb")))
        with open(os.path.join(out, "summary.json")) as f:
            summary = json.load(f)
        self.assertIn("assignments", summary)


if __name__ == "__main__":
    unittest.main()
