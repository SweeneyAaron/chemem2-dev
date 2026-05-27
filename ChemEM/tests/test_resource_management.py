from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from ChemEM.__main__ import apply_overrides, build_parser
from ChemEM.data.system import System
from ChemEM.tools.resources import (
    apply_cpu_budget,
    apply_cpus_per_site,
    openmm_platform_properties,
    resolve_cpu_budget,
    resolve_cpus_per_site,
    thread_limit_env,
)


class ResourceManagementTests(unittest.TestCase):
    def test_resolve_cpu_budget_accepts_canonical_and_legacy_names(self):
        self.assertEqual(resolve_cpu_budget(SimpleNamespace(ncpu=2)), 2)
        self.assertEqual(resolve_cpu_budget(SimpleNamespace(n_cpu=3)), 3)
        self.assertEqual(resolve_cpu_budget(SimpleNamespace(n_cpus=4)), 4)
        self.assertEqual(resolve_cpu_budget({"ncpu": "5"}), 5)

    def test_apply_cpu_budget_sets_system_and_options_aliases(self):
        system = System()
        system.options = SimpleNamespace()

        budget = apply_cpu_budget(system, 2)

        self.assertEqual(budget, 2)
        for obj in (system, system.options):
            self.assertEqual(obj.ncpu, 2)
            self.assertEqual(obj.n_cpu, 2)
            self.assertEqual(obj.n_cpus, 2)

    def test_cli_ncpu_override_reaches_system_and_options(self):
        args = build_parser().parse_args(["config.txt", "--ncpu", "2"])
        system = System()
        system.platform = "CPU"
        system.options = args

        apply_overrides(system, args)

        self.assertEqual(system.ncpu, 2)
        self.assertEqual(system.options.ncpu, 2)
        self.assertEqual(system.n_cpu, 2)
        self.assertEqual(system.n_cpus, 2)

    def test_openmm_platform_properties_respect_cpu_platform_threads(self):
        self.assertEqual(
            openmm_platform_properties("CPU", ncpu=2),
            {"Threads": "2"},
        )

    def test_openmm_platform_properties_prefers_explicit_ncpu(self):
        source = SimpleNamespace(ncpu=8)

        props = openmm_platform_properties("CPU", source=source, ncpu=2)

        self.assertEqual(props["Threads"], "2")

    def test_openmm_platform_properties_do_not_add_threads_to_opencl(self):
        props = openmm_platform_properties("OpenCL", ncpu=2)

        self.assertNotIn("Threads", props)
        self.assertEqual(props["Precision"], "single")
        self.assertEqual(props["UseCpuPme"], "false")

    def test_openmm_platform_properties_parses_false_cpu_pme_string(self):
        props = openmm_platform_properties("CUDA", ncpu=2, use_cpu_pme="false")

        self.assertEqual(props["UseCpuPme"], "false")

    def test_thread_limit_env_sets_and_restores_common_thread_env_vars(self):
        old_omp = os.environ.get("OMP_NUM_THREADS")
        old_mkl = os.environ.get("MKL_NUM_THREADS")

        with thread_limit_env(2):
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "2")

        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), old_omp)
        self.assertEqual(os.environ.get("MKL_NUM_THREADS"), old_mkl)

    def test_precomputed_data_uses_requested_cpu_budget(self):
        chemem_dir = Path(__file__).resolve().parents[1]
        text = (chemem_dir / "tools" / "precomputed_data.py").read_text()

        self.assertIn("self.ncpu = resolve_cpu_budget(system)", text)

    def test_resolve_cpus_per_site_uses_heuristic_default(self):
        # No source / no explicit override: max(2, total // 4), clamped to total.
        self.assertEqual(resolve_cpus_per_site(total_cpus=16), 4)
        self.assertEqual(resolve_cpus_per_site(total_cpus=4), 2)
        self.assertEqual(resolve_cpus_per_site(total_cpus=1), 1)

    def test_resolve_cpus_per_site_honours_explicit_attribute(self):
        source = SimpleNamespace(cpus_per_site=8)
        self.assertEqual(resolve_cpus_per_site(source, total_cpus=16), 8)

        legacy = SimpleNamespace(CPUS_PER_SITE=6)
        self.assertEqual(resolve_cpus_per_site(legacy, total_cpus=16), 6)

    def test_resolve_cpus_per_site_clamps_to_total(self):
        source = SimpleNamespace(cpus_per_site=64)
        self.assertEqual(resolve_cpus_per_site(source, total_cpus=4), 4)

    def test_resolve_cpus_per_site_reads_options_alias(self):
        source = SimpleNamespace(options=SimpleNamespace(cpus_per_site=3))
        self.assertEqual(resolve_cpus_per_site(source, total_cpus=16), 3)

    def test_apply_cpus_per_site_writes_aliases(self):
        system = System()
        system.options = SimpleNamespace()
        apply_cpu_budget(system, 16)

        value = apply_cpus_per_site(system, 5)

        self.assertEqual(value, 5)
        self.assertEqual(system.cpus_per_site, 5)
        self.assertEqual(system.CPUS_PER_SITE, 5)
        self.assertEqual(system.options.cpus_per_site, 5)
        self.assertEqual(system.options.CPUS_PER_SITE, 5)

    def test_cli_cpus_per_site_override_reaches_system_and_options(self):
        args = build_parser().parse_args(
            ["config.txt", "--ncpu", "16", "--cpus-per-site", "5"]
        )
        system = System()
        system.platform = "CPU"
        system.options = args

        apply_overrides(system, args)

        self.assertEqual(system.cpus_per_site, 5)
        self.assertEqual(system.CPUS_PER_SITE, 5)

    def test_grid_maps_exports_openmp_thread_setter(self):
        chemem_dir = Path(__file__).resolve().parents[1]
        text = (chemem_dir / "cpp" / "grid_maps.cpp").read_text()

        self.assertIn("set_omp_num_threads", text)
        self.assertIn("omp_set_num_threads", text)


if __name__ == "__main__":
    unittest.main()
