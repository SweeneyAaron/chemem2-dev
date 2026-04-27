import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

try:
    from rdkit import Chem
    _HAS_RDKIT = True
except ModuleNotFoundError:
    Chem = None
    _HAS_RDKIT = False

if _HAS_RDKIT:
    try:
        from ChemEM.protocols.refine.ion_template_search import IonTemplateSearch
    except ModuleNotFoundError:
        from protocols.refine.ion_template_search import IonTemplateSearch
else:
    IonTemplateSearch = None


class _DummySystem:
    def __init__(self, output, options):
        self.output = output
        self.options = options
        self._logs = []

    def log(self, msg):
        self._logs.append(str(msg))


class _Atom:
    def __init__(self, xx, xy, xz, element_name="C", name="C"):
        self.xx = float(xx)
        self.xy = float(xy)
        self.xz = float(xz)
        self.element_name = str(element_name)
        self.name = str(name)


class _Residue:
    def __init__(self, chain_id, name, number, atoms):
        self.chain = SimpleNamespace(id=str(chain_id))
        self.name = str(name)
        self.number = int(number)
        self.atoms = list(atoms)


class _Structure:
    def __init__(self, residues):
        self.residues = list(residues)
        self.atoms = [a for r in self.residues for a in r.atoms]


if _HAS_RDKIT:
    class _StubSearch(IonTemplateSearch):
        def __init__(self, system, ranked_templates):
            self._ranked_templates = ranked_templates
            self.ion_fixer_called = False
            super().__init__(system)

        def _prepare_context(self):
            return {
                "target_chains": {},
                "query_smiles": "",
                "query_comp_id_hint": None,
                "metal_elements": list(self.DEFAULT_METAL_ELEMENTS),
            }

        def _collect_template_candidates(self, context):
            return []

        def _rank_templates(self, candidates, context):
            return list(self._ranked_templates)

        def _run_ion_fixer(self):
            self.ion_fixer_called = True
            return {"n_cycles": 10}

    class _MappingSearch(_StubSearch):
        def __init__(self, system, ranked_templates, template_smiles):
            self._template_smiles = template_smiles
            super().__init__(system, ranked_templates)

        def _fetch_chemcomp_smiles(self, comp_id):
            return self._template_smiles

    class _EvalSearch(_StubSearch):
        def __init__(self, system, neighborhoods):
            self._neighborhoods = neighborhoods
            super().__init__(system, ranked_templates=[])

        def _fetch_nonpolymer_entity(self, entry_id, entity_id):
            return {"rcsb_nonpolymer_entity_container_identifiers": {"nonpolymer_comp_id": "LIG"}}

        def _fetch_residue_interaction_cif(self, entry_id, ligand_comp_id):
            return "mock-cif"

        def _extract_coordination_neighborhoods(self, interaction_text, ligand_comp_id, metal_elements):
            return list(self._neighborhoods)

        def _fetch_template_chain_sequence(self, entry_id, asym_id):
            return "DE"

        def _fetch_entry_data(self, entry_id):
            return {"rcsb_entry_info": {"resolution_combined": [2.0]}}

        def _fetch_chemcomp_smiles(self, comp_id):
            return "NCCO"
else:
    class _StubSearch:
        pass


@unittest.skipUnless(_HAS_RDKIT, "rdkit is required for ion_template_search tests")
class TestSearchHitCombiner(unittest.TestCase):
    def test_exact_and_similarity_fallback_merge(self):
        exact = [
            {
                "identifier": "1ABC_1",
                "entry_id": "1ABC",
                "entity_id": "1",
                "search_score": 0.9,
                "match_mode": "exact_smiles",
            }
        ]
        similar = [
            {
                "identifier": "1ABC_1",
                "entry_id": "1ABC",
                "entity_id": "1",
                "search_score": 0.95,
                "match_mode": "similar_smiles",
            },
            {
                "identifier": "2XYZ_3",
                "entry_id": "2XYZ",
                "entity_id": "3",
                "search_score": 0.8,
                "match_mode": "similar_smiles",
            },
        ]

        out = IonTemplateSearch._combine_search_hits(exact, similar, max_hits=10)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["identifier"], "1ABC_1")
        self.assertEqual(out[0]["match_mode"], "exact_smiles")
        self.assertEqual(out[1]["identifier"], "2XYZ_3")


@unittest.skipUnless(_HAS_RDKIT, "rdkit is required for ion_template_search tests")
class TestSequenceAlignment(unittest.TestCase):
    def test_alignment_mapping_positions(self):
        aln = IonTemplateSearch._global_align_with_mapping("ACDEFG", "ACXDEFG")
        self.assertGreaterEqual(aln["identity"], 0.99)
        self.assertEqual(aln["template_to_target"][1], 1)
        self.assertEqual(aln["template_to_target"][2], 2)
        self.assertEqual(aln["template_to_target"][3], 4)
        self.assertEqual(aln["template_to_target"][6], 7)

    def test_alignment_respects_local_chain_restriction(self):
        protocol = _StubSearch(_DummySystem(".", SimpleNamespace()), ranked_templates=[])
        target_chains = {
            "A": SimpleNamespace(sequence="AAAA"),
            "B": SimpleNamespace(sequence="CCCC"),
        }
        best = protocol._best_target_chain_alignment(
            template_seq="AAAA",
            target_chains=target_chains,
            eligible_chain_ids={"B"},
        )
        self.assertIsNotNone(best)
        self.assertEqual(best["target_chain_id"], "B")


@unittest.skipUnless(_HAS_RDKIT, "rdkit is required for ion_template_search tests")
class TestLocalConsistencyGate(unittest.TestCase):
    def test_protein_contact_mapping_tracks_local_consistency(self):
        protocol = _StubSearch(_DummySystem(".", SimpleNamespace()), ranked_templates=[])
        res_local = _Residue("A", "GLU", 10, [])
        res_far = _Residue("A", "ASP", 20, [])
        target_chains = {
            "A": SimpleNamespace(
                sequence="",
                residues_by_seq={10: res_local, 20: res_far},
            )
        }
        chain_alignments = {
            "X": {
                "target_chain_id": "A",
                "template_to_target": {5: 10, 6: 20},
            }
        }
        neighborhoods = [
            {
                "protein_contacts": [
                    {"asym_id": "X", "seq_id": 5, "atom_id": "OE2", "comp_id": "GLU"},
                    {"asym_id": "X", "seq_id": 6, "atom_id": "OD1", "comp_id": "ASP"},
                ],
                "ligand_contacts": [],
            }
        ]
        result = protocol._protein_specs_from_neighborhoods(
            neighborhoods=neighborhoods,
            chain_alignments=chain_alignments,
            target_chains=target_chains,
            local_residue_keys={protocol._residue_key(res_local)},
        )
        self.assertEqual(result["stats"]["total_contacts"], 2)
        self.assertEqual(result["stats"]["mapped_contacts"], 2)
        self.assertEqual(result["stats"]["local_mapped_contacts"], 1)
        self.assertEqual(result["specs"], ["A:GLU:10:OE2"])

    def test_high_global_identity_but_nonlocal_contact_fails(self):
        opts = SimpleNamespace(
            its_seq_identity_min=0.35,
            its_local_chain_radius_a=12.0,
            its_max_templates=25,
            atom_specs=[],
            ion_type=None,
            coordination_geometry="Octahedral",
        )
        protocol = _EvalSearch(_DummySystem(".", opts), neighborhoods=[
            {
                "metal_element": "MG",
                "protein_contacts": [{"asym_id": "X", "seq_id": 2, "atom_id": "OD1", "comp_id": "ASP"}],
                "ligand_contacts": [{"comp_id": "LIG", "atom_id": "O1", "type_symbol": "O", "atom_index": 3}],
            }
        ])

        res_local = _Residue("A", "GLU", 101, [_Atom(0, 0, 0, "O", "OE2")])
        res_far = _Residue("A", "ASP", 202, [_Atom(0, 0, 0, "O", "OD1")])
        target_chains = {
            "A": SimpleNamespace(sequence="DE", residues_by_seq={1: res_local, 2: res_far}),
        }
        context = {
            "target_chains": target_chains,
            "local_chain_ids": {"A"},
            "local_residue_keys": {protocol._residue_key(res_local)},
            "ligand": SimpleNamespace(atom_names=["N1", "C1", "C2", "O1"]),
            "query_mol": Chem.MolFromSmiles("NCCO"),
            "metal_elements": list(protocol.DEFAULT_METAL_ELEMENTS),
        }

        tpl = protocol._evaluate_template(
            {"entry_id": "1ABC", "entity_id": "1", "identifier": "1ABC_1", "search_score": 1.0, "match_mode": "exact"},
            context,
        )
        self.assertIsNone(tpl)

    def test_valid_local_contact_mapping_passes(self):
        opts = SimpleNamespace(
            its_seq_identity_min=0.35,
            its_local_chain_radius_a=12.0,
            its_max_templates=25,
            atom_specs=[],
            ion_type=None,
            coordination_geometry="Octahedral",
        )
        protocol = _EvalSearch(_DummySystem(".", opts), neighborhoods=[
            {
                "metal_element": "MG",
                "protein_contacts": [{"asym_id": "X", "seq_id": 1, "atom_id": "OE2", "comp_id": "GLU"}],
                "ligand_contacts": [{"comp_id": "LIG", "atom_id": "O1", "type_symbol": "O", "atom_index": 3}],
            }
        ])

        res_local = _Residue("A", "GLU", 101, [_Atom(0, 0, 0, "O", "OE2")])
        res_far = _Residue("A", "ASP", 202, [_Atom(0, 0, 0, "O", "OD1")])
        target_chains = {
            "A": SimpleNamespace(sequence="DE", residues_by_seq={1: res_local, 2: res_far}),
        }
        context = {
            "target_chains": target_chains,
            "local_chain_ids": {"A"},
            "local_residue_keys": {protocol._residue_key(res_local)},
            "ligand": SimpleNamespace(atom_names=["N1", "C1", "C2", "O1"]),
            "query_mol": Chem.MolFromSmiles("NCCO"),
            "metal_elements": list(protocol.DEFAULT_METAL_ELEMENTS),
        }

        tpl = protocol._evaluate_template(
            {"entry_id": "1ABC", "entity_id": "1", "identifier": "1ABC_1", "search_score": 1.0, "match_mode": "exact"},
            context,
        )
        self.assertIsNotNone(tpl)
        self.assertEqual(tpl["protein_atom_specs"], ["A:GLU:101:OE2"])
        self.assertEqual(tpl["ligand_atom_specs"], ["LIG:0:O1"])
        self.assertGreater(tpl["scores"]["local_consistency"], 0.9)


@unittest.skipUnless(_HAS_RDKIT, "rdkit is required for ion_template_search tests")
class TestLigandMCSMapping(unittest.TestCase):
    def test_mcs_maps_template_indices_to_query(self):
        query = Chem.MolFromSmiles("NCCO")
        template = Chem.MolFromSmiles("NCCO")
        mapped = IonTemplateSearch._map_template_ligand_atoms_by_mcs(
            query_mol=query,
            template_mol=template,
            template_atom_indices=[0, 3],
        )
        self.assertEqual(set(mapped), {0, 3})

    def test_contact_anchored_ligand_mapping_skips_untraceable_contacts(self):
        opts = SimpleNamespace()
        dummy = _DummySystem(".", opts)
        protocol = _MappingSearch(dummy, ranked_templates=[], template_smiles=None)
        ligand = SimpleNamespace(atom_names=["N1", "N2", "N3", "N4"])
        query = Chem.MolFromSmiles("N1CCCC1")
        neighborhoods = [
            {
                "ligand_contacts": [
                    {"comp_id": "ATP", "atom_id": "O2A", "type_symbol": "O", "atom_index": None},
                    {"comp_id": "ATP", "atom_id": "O3B", "type_symbol": "O", "atom_index": None},
                ],
                "protein_contacts": [],
            }
        ]
        specs = protocol._ligand_specs_from_neighborhoods(
            neighborhoods=neighborhoods,
            query_mol=query,
            ligand=ligand,
            template_comp_id="ATP",
        )
        self.assertEqual(specs, [])

    def test_contact_anchored_ligand_mapping_uses_traceable_index(self):
        opts = SimpleNamespace()
        dummy = _DummySystem(".", opts)
        protocol = _MappingSearch(dummy, ranked_templates=[], template_smiles="NCCO")
        ligand = SimpleNamespace(atom_names=["N1", "C1", "C2", "O1"])
        query = Chem.MolFromSmiles("NCCO")
        neighborhoods = [
            {
                "ligand_contacts": [
                    {"comp_id": "LIG", "atom_id": "O1", "type_symbol": "O", "atom_index": 3},
                ],
                "protein_contacts": [],
            }
        ]
        specs = protocol._ligand_specs_from_neighborhoods(
            neighborhoods=neighborhoods,
            query_mol=query,
            ligand=ligand,
            template_comp_id="LIG",
        )
        self.assertEqual(specs, ["LIG:0:O1"])


@unittest.skipUnless(_HAS_RDKIT, "rdkit is required for ion_template_search tests")
class TestConfidenceGateAndHandoff(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="its_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _opts(self, **kw):
        base = dict(
            its_confidence_thresh=0.65,
            its_auto_run_ion_fixer=False,
            its_local_chain_radius_a=12.0,
            atom_specs=[],
            ion_type=None,
            coordination_geometry="Octahedral",
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def _candidate(self, confidence):
        return {
            "identifier": "1ABC_1",
            "entry_id": "1ABC",
            "entity_id": "1",
            "ligand_comp_id": "ABC",
            "ion_type": "ZN",
            "coordination_geometry": "Tetrahedral",
            "coordination_inference": {
                "geometry": "Tetrahedral",
                "target_cn": 4,
                "observed_cn": 3,
                "missing_sites": 1,
                "water_completion_likely": True,
                "ion_type": "ZN",
            },
            "protein_atom_specs": ["A:HIS:10:NE2"],
            "ligand_atom_specs": ["LIG:0:O1"],
            "confidence": float(confidence),
        }

    def test_confidence_gate_pass_and_fail(self):
        # pass
        sys_pass = _DummySystem(self.tmpdir, self._opts(its_auto_run_ion_fixer=False))
        prot_pass = _StubSearch(sys_pass, ranked_templates=[self._candidate(0.90)])
        out_pass = prot_pass.run()
        self.assertEqual(out_pass["status"], "ok")
        self.assertTrue(out_pass["applied"])
        self.assertIn("A:HIS:10:NE2", sys_pass.options.atom_specs)

        # fail
        sys_fail = _DummySystem(self.tmpdir, self._opts(its_auto_run_ion_fixer=False))
        prot_fail = _StubSearch(sys_fail, ranked_templates=[self._candidate(0.20)])
        out_fail = prot_fail.run()
        self.assertEqual(out_fail["status"], "ok")
        self.assertFalse(out_fail["applied"])
        self.assertEqual(sys_fail.options.atom_specs, [])

    def test_auto_apply_requires_both_protein_and_ligand_specs(self):
        system1 = _DummySystem(self.tmpdir, self._opts(its_auto_run_ion_fixer=False))
        prot_only = dict(self._candidate(0.95))
        prot_only["ligand_atom_specs"] = []
        out1 = _StubSearch(system1, ranked_templates=[prot_only]).run()
        self.assertFalse(out1["applied"])

        system2 = _DummySystem(self.tmpdir, self._opts(its_auto_run_ion_fixer=False))
        lig_only = dict(self._candidate(0.95))
        lig_only["protein_atom_specs"] = []
        out2 = _StubSearch(system2, ranked_templates=[lig_only]).run()
        self.assertFalse(out2["applied"])

    def test_no_hit_skips_auto_handoff(self):
        system = _DummySystem(self.tmpdir, self._opts(its_auto_run_ion_fixer=True))
        protocol = _StubSearch(system, ranked_templates=[])
        out = protocol.run()

        self.assertEqual(out["status"], "no_hit")
        self.assertFalse(protocol.ion_fixer_called)

        report_path = os.path.join(protocol.output, "report.json")
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(report["auto_run"]["status"], "skipped")

    def test_high_confidence_auto_handoff_runs_ion_fixer(self):
        system = _DummySystem(self.tmpdir, self._opts(its_auto_run_ion_fixer=True))
        protocol = _StubSearch(system, ranked_templates=[self._candidate(0.95)])
        out = protocol.run()

        self.assertEqual(out["status"], "ok")
        self.assertTrue(out["applied"])
        self.assertTrue(protocol.ion_fixer_called)

    def test_geometry_inference_metadata_written_to_outputs(self):
        system = _DummySystem(self.tmpdir, self._opts(its_auto_run_ion_fixer=False))
        protocol = _StubSearch(system, ranked_templates=[self._candidate(0.95)])
        protocol.run()

        proposal_path = os.path.join(protocol.output, "proposed_ion_fixer_args.json")
        with open(proposal_path, "r", encoding="utf-8") as f:
            proposal = json.load(f)

        self.assertIn("coordination_inference", proposal)
        self.assertEqual(proposal["coordination_inference"]["missing_sites"], 1)
        self.assertTrue(proposal["coordination_inference"]["water_completion_likely"])


@unittest.skipUnless(_HAS_RDKIT, "rdkit is required for ion_template_search tests")
class TestGeometryInference(unittest.TestCase):
    def test_undercoordinated_mg_reports_missing_sites(self):
        protocol = _StubSearch(_DummySystem(".", SimpleNamespace()), ranked_templates=[])
        info = protocol.infer_coordination_geometry(
            ion_type="MG",
            protein_spec_count=2,
            ligand_spec_count=1,
            neighborhoods=[],
        )
        self.assertEqual(info["geometry"], "Octahedral")
        self.assertEqual(info["target_cn"], 6)
        self.assertEqual(info["observed_cn"], 3)
        self.assertEqual(info["missing_sites"], 3)
        self.assertTrue(info["water_completion_likely"])


if __name__ == "__main__":
    unittest.main()
