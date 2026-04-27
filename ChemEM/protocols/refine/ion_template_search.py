#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

import json
import os
import re
import shlex
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from rdkit import Chem
from rdkit.Chem import rdFMCS


@dataclass
class ChainSequence:
    chain_id: str
    sequence: str
    residues_by_seq: Dict[int, Any]


class IonTemplateSearch:
    """
    Standalone pre-ion-fixer protocol that mines PDB templates for
    protein/ligand metal-coordination patterns and proposes IonFixer atom specs.

    Behavior highlights
    -------------------
    - Uses exact ligand search first; can expand to similarity search.
    - Applies confidence gating before mutating ion_fixer options.
    - Can optionally auto-run IonFixer in the same protocol run.
    - Network/API failures are handled non-fatally and reported in JSON outputs.
    """

    SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
    DATA_CORE_URL = "https://data.rcsb.org/rest/v1/core"
    MODELSERVER_BASE = "https://models.rcsb.org/v1"

    DEFAULT_METAL_ELEMENTS = (
        "LI", "NA", "K", "MG", "CA", "MN", "FE", "CO", "NI", "CU", "ZN", "CD",
    )

    # Geometry labels compatible with coord_geom_to_int() alias handling.
    GEOMETRY_BY_CN = {
        2: "linear",
        3: "Trigonal Planar",
        4: "Tetrahedral",
        5: "Trigonal Bipyramidal",
        6: "Octahedral",
        7: "Pentagonal Bipyramidal",
    }

    AA3_TO_1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
        "GLU": "E", "GLY": "G", "HIS": "H", "HID": "H", "HIE": "H", "HIP": "H",
        "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "SEC": "U",
        "PYL": "O",
    }

    # Common donor atoms for residue side-chains that coordinate metal ions.
    PROTEIN_DONOR_ATOMS = {
        ("ASP", "OD1"), ("ASP", "OD2"),
        ("GLU", "OE1"), ("GLU", "OE2"),
        ("HIS", "ND1"), ("HIS", "NE2"),
        ("HID", "ND1"), ("HIE", "NE2"), ("HIP", "ND1"), ("HIP", "NE2"),
        ("CYS", "SG"), ("CYM", "SG"), ("CYX", "SG"),
        ("MET", "SD"),
        ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"),
        ("ASN", "OD1"), ("GLN", "OE1"),
        ("LYS", "NZ"), ("ARG", "NE"),
    }

    DONOR_ELEMENTS = {"N", "O", "S", "F", "CL", "BR", "I"}
    LIGAND_DONOR_ELEMENTS = {"O", "N", "S"}
    MG_LIKE_IONS = {"MG", "CA"}
    ELEMENT_SYMBOLS = {
        "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE",
        "NA", "MG", "AL", "SI", "P", "S", "CL", "AR", "K", "CA",
        "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
        "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR",
        "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
        "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND",
        "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
        "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
        "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
        "PA", "U", "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM",
        "MD", "NO", "LR",
    }

    def __init__(self, system):
        self.system = system
        self.output = os.path.join(getattr(self.system, "output", "."), "ion_template_search")
        os.makedirs(self.output, exist_ok=True)

        self._chain_seq_cache: Dict[Tuple[str, str], Optional[str]] = {}
        self._entry_data_cache: Dict[str, Optional[dict]] = {}

    # -----------------------
    # Public entry point
    # -----------------------

    def run(self):
        report: Dict[str, Any] = {
            "protocol": "ion_template_search",
            "status": "started",
            "confidence_threshold": float(self._opt("its_confidence_thresh", 0.65)),
            "errors": [],
            "warnings": [],
            "search": {},
            "auto_run": {
                "requested": bool(self._opt("its_auto_run_ion_fixer", False)),
                "attempted": False,
                "ran": False,
                "status": "not_requested",
            },
        }

        selected_template: Dict[str, Any] = {}
        proposed_args: Dict[str, Any] = {
            "atom_specs": [],
            "ion_type": None,
            "coordination_geometry": None,
            "confidence": 0.0,
            "applied": False,
            "reason": "not_evaluated",
        }

        try:
            context = self._prepare_context()
            report["target"] = self._context_summary(context)

            candidates = self._collect_template_candidates(context)
            report["search"]["n_candidates"] = len(candidates)
            report["search"]["n_homolog_entries"] = int(context.get("n_homolog_entries", 0))
            report["search"]["n_candidates_pre_homolog_filter"] = int(
                context.get("n_candidates_pre_homolog_filter", 0)
            )
            report["search"]["n_candidates_after_homolog_filter"] = int(
                context.get("n_candidates_after_homolog_filter", 0)
            )

            ranked = self._rank_templates(candidates, context)
            report["search"]["n_ranked_templates"] = len(ranked)
            report["ranked_templates"] = [self._compact_template_summary(x) for x in ranked]
            report["search"]["template_rejections"] = list(context.get("template_rejections", []))

            if ranked:
                selected_template = ranked[0]
                proposed_args = self._build_proposed_ion_fixer_args(selected_template)
            else:
                proposed_args["reason"] = "no_templates"

            confidence = float(proposed_args.get("confidence", 0.0))
            threshold = float(report["confidence_threshold"])
            has_protein = bool(list(proposed_args.get("protein_atom_specs", []) or []))
            has_ligand = bool(list(proposed_args.get("ligand_atom_specs", []) or []))
            if confidence >= threshold and has_protein and has_ligand and proposed_args.get("atom_specs"):
                applied = self._apply_proposal_to_options(proposed_args)
                proposed_args["applied"] = bool(applied)
                proposed_args["reason"] = "applied" if applied else "apply_failed"
            else:
                if confidence < threshold:
                    proposed_args["reason"] = "below_threshold"
                elif not has_protein or not has_ligand:
                    proposed_args["reason"] = "insufficient_mixed_specs"
                else:
                    proposed_args["reason"] = "no_specs"

            # Optional same-run handoff to IonFixer.
            if bool(self._opt("its_auto_run_ion_fixer", False)):
                report["auto_run"]["attempted"] = True

                if bool(proposed_args.get("applied", False)):
                    try:
                        ion_result = self._run_ion_fixer()
                        report["auto_run"].update(
                            {
                                "ran": True,
                                "status": "ok",
                                "ion_fixer_summary": self._summarize_ion_fixer_result(ion_result),
                            }
                        )
                    except Exception as exc:
                        report["auto_run"].update(
                            {
                                "ran": False,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                else:
                    report["auto_run"].update(
                        {
                            "ran": False,
                            "status": "skipped",
                            "reason": "confidence_gate_failed",
                        }
                    )

            report["status"] = "ok" if ranked else "no_hit"

        except Exception as exc:
            report["status"] = "error"
            report["errors"].append(f"{type(exc).__name__}: {exc}")

        # Always persist outputs, even on failures.
        self._write_outputs(report, selected_template, proposed_args)

        if report["status"] == "ok":
            self._log(
                "[ion_template_search] selected template "
                f"{selected_template.get('entry_id', 'n/a')} "
                f"(confidence={float(proposed_args.get('confidence', 0.0)):.3f})"
            )
        elif report["status"] == "no_hit":
            self._log("[ion_template_search] no confident template found; wrote diagnostics only")
        else:
            self._log("[ion_template_search] failed; wrote diagnostics")

        return {
            "status": report.get("status"),
            "confidence": float(proposed_args.get("confidence", 0.0)),
            "applied": bool(proposed_args.get("applied", False)),
            "output": self.output,
        }

    # -----------------------
    # Context preparation
    # -----------------------

    def _prepare_context(self) -> Dict[str, Any]:
        ligands = list(getattr(self.system, "ligand", []) or [])
        if not ligands:
            raise RuntimeError("IonTemplateSearch requires at least one ligand")

        ligand = ligands[0]  # v1: target first active ligand only
        query_mol = Chem.RemoveHs(Chem.Mol(ligand.mol))
        query_smiles = Chem.MolToSmiles(query_mol, canonical=True)

        comp_hint = None
        try:
            resname = str(ligand.complex_structure.residues[0].name).upper().strip()
            if 1 <= len(resname) <= 3 and resname.isalnum() and not resname.startswith("LIG"):
                comp_hint = resname
        except Exception:
            comp_hint = None

        target_chains = self._build_target_chain_sequences()
        if not target_chains:
            raise RuntimeError("Could not derive target protein chain sequences")

        local_radius_A = float(self._opt("its_local_chain_radius_a", 12.0))
        local_ctx = self._build_ligand_local_context(ligand=ligand, radius_A=local_radius_A)

        return {
            "ligand": ligand,
            "query_mol": query_mol,
            "query_smiles": query_smiles,
            "query_comp_id_hint": comp_hint,
            "target_chains": target_chains,
            "metal_elements": self._metal_elements(),
            "local_chain_ids": local_ctx["local_chain_ids"],
            "local_residue_keys": local_ctx["local_residue_keys"],
            "local_radius_A": float(local_radius_A),
        }

    def _build_target_chain_sequences(self) -> Dict[str, ChainSequence]:
        out: Dict[str, Dict[str, Any]] = {}
        structure = self.system.protein.complex_structure

        for res in structure.residues:
            resname = str(getattr(res, "name", "")).upper()
            aa = self.AA3_TO_1.get(resname)
            if aa is None:
                continue

            chain_id = self._chain_id_from_residue(res)
            if chain_id not in out:
                out[chain_id] = {
                    "seq": [],
                    "residue_by_seq": {},
                }

            seq_pos = len(out[chain_id]["seq"]) + 1
            out[chain_id]["seq"].append(aa)
            out[chain_id]["residue_by_seq"][seq_pos] = res

        chains: Dict[str, ChainSequence] = {}
        for chain_id, data in out.items():
            seq = "".join(data["seq"])
            if not seq:
                continue
            chains[chain_id] = ChainSequence(
                chain_id=chain_id,
                sequence=seq,
                residues_by_seq=data["residue_by_seq"],
            )
        return chains

    def _build_ligand_local_context(self, ligand, radius_A: float) -> Dict[str, Any]:
        protein_structure = self.system.protein.complex_structure
        ligand_atoms = list(getattr(ligand.complex_structure, "atoms", []) or [])
        lig_xyz = [self._atom_xyz(a) for a in ligand_atoms if self._is_heavy_atom(a)]
        if not lig_xyz:
            return {"local_chain_ids": set(), "local_residue_keys": set()}

        cutoff2 = float(radius_A) * float(radius_A)
        local_chain_ids = set()
        local_residue_keys = set()

        for res in protein_structure.residues:
            heavy_xyz = [self._atom_xyz(a) for a in res.atoms if self._is_heavy_atom(a)]
            if not heavy_xyz:
                continue
            near = False
            for px, py, pz in heavy_xyz:
                for lx, ly, lz in lig_xyz:
                    dx = px - lx
                    dy = py - ly
                    dz = pz - lz
                    if (dx * dx + dy * dy + dz * dz) <= cutoff2:
                        near = True
                        break
                if near:
                    break
            if not near:
                continue

            chain_id = self._chain_id_from_residue(res)
            local_chain_ids.add(chain_id)
            local_residue_keys.add(self._residue_key(res))

        return {
            "local_chain_ids": local_chain_ids,
            "local_residue_keys": local_residue_keys,
        }

    # -----------------------
    # Search + ranking
    # -----------------------

    def _collect_template_candidates(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        max_hits = int(self._opt("its_max_entry_candidates", 1000))

        homolog_entries = self._search_homolog_entries(context)
        context["homolog_entries"] = homolog_entries
        context["n_homolog_entries"] = len(homolog_entries)

        if homolog_entries:
            # Direct enumeration: any non-metal non-polymer entity in a homolog
            # entry is a candidate. Whether its ligand graph matches the query is
            # decided downstream by MCS at evaluation time, which tolerates
            # element substitutions that RCSB's chemical service does not
            # (e.g. ATP <-> ANP across the bridging oxygen/nitrogen).
            candidates = self._enumerate_nonpoly_candidates(homolog_entries)
            context["n_candidates_pre_homolog_filter"] = 0
            context["n_candidates_after_homolog_filter"] = len(candidates)
            return candidates

        # Fallback: ligand-driven chemical search when homolog set is empty
        # (e.g. RCSB sequence service unreachable).
        exact_hits = self._search_templates_exact(
            query_smiles=context["query_smiles"],
            comp_id_hint=context["query_comp_id_hint"],
            max_rows=max_hits,
        )

        similar_hits: List[Dict[str, Any]] = []
        if bool(self._opt("its_similarity_enabled", True)):
            similar_hits = self._search_templates_similar(
                query_smiles=context["query_smiles"],
                max_rows=max_hits,
            )

        combined = self._combine_search_hits(exact_hits, similar_hits, max_hits=max_hits)
        context["n_candidates_pre_homolog_filter"] = len(combined)
        context["n_candidates_after_homolog_filter"] = len(combined)
        return combined

    def _enumerate_nonpoly_candidates(self, entries: Set[str]) -> List[Dict[str, Any]]:
        metal_set = {x.upper() for x in self._metal_elements()}
        out: List[Dict[str, Any]] = []
        for entry_id in sorted(entries):
            entry_data = self._fetch_entry_data(entry_id)
            if not entry_data:
                continue
            cids = entry_data.get("rcsb_entry_container_identifiers", {}) or {}
            for entity_id in list(cids.get("non_polymer_entity_ids", []) or []):
                nonpoly = self._fetch_nonpolymer_entity(entry_id, str(entity_id))
                comp_id = self._extract_nonpoly_comp_id(nonpoly)
                if not comp_id:
                    continue
                if comp_id.upper() in metal_set:
                    continue
                out.append(
                    {
                        "identifier": f"{entry_id}_{entity_id}",
                        "entry_id": str(entry_id).upper(),
                        "entity_id": str(entity_id),
                        "search_score": 1.0,
                        "match_mode": "homolog_enum",
                    }
                )
        return out

    @staticmethod
    def _combine_search_hits(
        exact_hits: Sequence[Dict[str, Any]],
        similar_hits: Sequence[Dict[str, Any]],
        max_hits: int,
    ) -> List[Dict[str, Any]]:
        by_identifier: Dict[str, Dict[str, Any]] = {}

        for hit in exact_hits:
            key = str(hit.get("identifier", ""))
            if not key:
                continue
            by_identifier[key] = dict(hit)

        for hit in similar_hits:
            key = str(hit.get("identifier", ""))
            if not key:
                continue

            if key in by_identifier:
                # Keep exact-match provenance but preserve best score.
                if float(hit.get("search_score", 0.0)) > float(by_identifier[key].get("search_score", 0.0)):
                    by_identifier[key]["search_score"] = float(hit.get("search_score", 0.0))
                continue

            by_identifier[key] = dict(hit)

        out = list(by_identifier.values())
        out.sort(key=lambda h: float(h.get("search_score", 0.0)), reverse=True)
        return out[: max(0, int(max_hits))]

    def _search_templates_exact(
        self,
        query_smiles: str,
        comp_id_hint: Optional[str],
        max_rows: int,
    ) -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []

        # 1) Exact by ligand CCD code when available.
        if comp_id_hint:
            payload = {
                "query": {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id",
                        "operator": "exact_match",
                        "value": comp_id_hint,
                    },
                },
                "return_type": "non_polymer_entity",
                "request_options": {
                    "paginate": {"start": 0, "rows": int(max_rows)},
                    "results_verbosity": "minimal",
                },
            }
            response = self._post_json(self.SEARCH_URL, payload, timeout_s=20)
            hits.extend(self._normalise_search_hits(response, match_mode="exact_comp_id"))

        # 2) Exact by chemical descriptor as fallback/expansion.
        if query_smiles:
            payload = {
                "query": {
                    "type": "terminal",
                    "service": "chemical",
                    "parameters": {
                        "value": query_smiles,
                        "type": "descriptor",
                        "descriptor_type": "SMILES",
                        "match_type": "graph-exact",
                    },
                },
                "return_type": "non_polymer_entity",
                "request_options": {
                    "paginate": {"start": 0, "rows": int(max_rows)},
                    "results_verbosity": "minimal",
                },
            }
            response = self._post_json(self.SEARCH_URL, payload, timeout_s=20)
            hits.extend(self._normalise_search_hits(response, match_mode="exact_smiles"))

        return hits

    def _search_homolog_entries(self, context: Dict[str, Any]) -> Set[str]:
        """Return the set of PDB entry IDs that contain a chain homologous to the
        target near the ligand. Used to pre-filter chemical-search hits.
        """
        target_chains: Dict[str, ChainSequence] = context.get("target_chains", {}) or {}
        local_chain_ids = set(context.get("local_chain_ids", set()) or set())

        # Restrict to chains actually near the ligand, then dedupe by sequence.
        seen_seqs: Set[str] = set()
        sequences: List[str] = []
        for chain_id, chain in target_chains.items():
            if local_chain_ids and str(chain_id) not in local_chain_ids:
                continue
            seq = str(getattr(chain, "sequence", "") or "")
            if not seq or seq in seen_seqs:
                continue
            seen_seqs.add(seq)
            sequences.append(seq)

        if not sequences:
            return set()

        identity_min = float(self._opt("its_homolog_identity_min", 0.35))
        max_rows = int(self._opt("its_max_homolog_entries", 1000))

        nodes = [
            {
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "sequence_type": "protein",
                    "evalue_cutoff": 1.0,
                    "identity_cutoff": float(identity_min),
                    "value": seq,
                },
            }
            for seq in sequences
        ]

        if len(nodes) == 1:
            query = nodes[0]
        else:
            query = {"type": "group", "logical_operator": "or", "nodes": nodes}

        payload = {
            "query": query,
            "return_type": "polymer_entity",
            "request_options": {
                "paginate": {"start": 0, "rows": int(max_rows)},
                "sort": [{"sort_by": "score", "direction": "desc"}],
                "results_verbosity": "minimal",
            },
        }

        response = self._post_json(self.SEARCH_URL, payload, timeout_s=25)
        if not response:
            return set()

        entries: Set[str] = set()
        for item in list(response.get("result_set", []) or []):
            identifier = str(item.get("identifier", "")).strip()
            if not identifier or "_" not in identifier:
                continue
            entry_id = identifier.split("_", 1)[0].upper()
            if entry_id:
                entries.add(entry_id)
        return entries

    def _search_templates_similar(self, query_smiles: str, max_rows: int) -> List[Dict[str, Any]]:
        if not query_smiles:
            return []

        payload = {
            "query": {
                "type": "terminal",
                "service": "chemical",
                "parameters": {
                    "value": query_smiles,
                    "type": "descriptor",
                    "descriptor_type": "SMILES",
                    "match_type": "graph-relaxed-stereo",
                },
            },
            "return_type": "non_polymer_entity",
            "request_options": {
                "paginate": {"start": 0, "rows": int(max_rows)},
                "results_verbosity": "minimal",
            },
        }
        response = self._post_json(self.SEARCH_URL, payload, timeout_s=20)
        return self._normalise_search_hits(response, match_mode="similar_smiles")

    def _normalise_search_hits(self, response: Optional[dict], match_mode: str) -> List[Dict[str, Any]]:
        if not response:
            return []

        out: List[Dict[str, Any]] = []
        for item in list(response.get("result_set", []) or []):
            identifier = str(item.get("identifier", "")).strip()
            if not identifier or "_" not in identifier:
                continue
            entry_id, entity_id = identifier.split("_", 1)
            out.append(
                {
                    "identifier": identifier,
                    "entry_id": entry_id.upper(),
                    "entity_id": entity_id,
                    "search_score": float(item.get("score", 0.0)),
                    "match_mode": match_mode,
                }
            )
        return out

    def _rank_templates(
        self,
        candidates: Sequence[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        max_templates = max(0, int(self._opt("its_max_templates", 25)))
        ranked: List[Dict[str, Any]] = []
        rejections: List[Dict[str, Any]] = []

        for hit in list(candidates)[:max_templates]:
            try:
                evaluated, reason = self._evaluate_template(hit, context)
            except Exception as exc:
                tb = traceback.format_exc(limit=8)
                rejections.append(
                    {
                        "identifier": hit.get("identifier"),
                        "reason": f"exception:{type(exc).__name__}:{exc}",
                        "traceback": tb,
                    }
                )
                self._log(
                    f"[ion_template_search] template {hit.get('identifier')} failed: {type(exc).__name__}: {exc}\n{tb}"
                )
                continue

            if evaluated is not None:
                ranked.append(evaluated)
            else:
                rejections.append(
                    {"identifier": hit.get("identifier"), "reason": reason or "unknown"}
                )

        context["template_rejections"] = rejections
        ranked.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
        return ranked

    def _evaluate_template(self, hit: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        entry_id = str(hit["entry_id"]).upper()
        entity_id = str(hit["entity_id"])

        nonpoly = self._fetch_nonpolymer_entity(entry_id, entity_id)
        comp_id = self._extract_nonpoly_comp_id(nonpoly)
        if not comp_id:
            return None, "no_comp_id"

        interaction_text = self._fetch_residue_interaction_cif(entry_id, comp_id)
        if not interaction_text:
            return None, "modelserver_no_response"

        neighborhoods = self._extract_coordination_neighborhoods(
            interaction_text=interaction_text,
            ligand_comp_id=comp_id,
            metal_elements=context["metal_elements"],
        )
        if not neighborhoods:
            return None, "no_metal_neighborhoods"

        # Build chain mappings for template chains observed in protein contacts.
        chain_alignments: Dict[str, List[Dict[str, Any]]] = {}
        seq_identity_min = float(self._opt("its_seq_identity_min", 0.35))
        local_chain_ids = set(context.get("local_chain_ids", set()) or set())
        local_residue_keys = set(context.get("local_residue_keys", set()) or set())
        if not local_chain_ids:
            return None, "no_local_chain_ids"

        template_asym_ids = sorted(
            {
                str(contact.get("asym_id", ""))
                for ng in neighborhoods
                for contact in ng.get("protein_contacts", [])
                if str(contact.get("asym_id", ""))
            }
        )
        if not template_asym_ids:
            return None, "no_template_protein_contacts"

        n_seq_fetch_failed = 0
        n_below_identity = 0
        for asym_id in template_asym_ids:
            template_seq = self._fetch_template_chain_sequence(entry_id, asym_id)
            if not template_seq:
                n_seq_fetch_failed += 1
                continue

            alignments = self._eligible_target_chain_alignments(
                template_seq=template_seq,
                target_chains=context["target_chains"],
                identity_min=seq_identity_min,
                eligible_chain_ids=local_chain_ids,
            )
            if not alignments:
                n_below_identity += 1
                continue

            chain_alignments[asym_id] = alignments

        if not chain_alignments:
            return None, (
                f"no_chain_alignments(asyms={len(template_asym_ids)},"
                f"seq_fetch_failed={n_seq_fetch_failed},"
                f"below_identity={n_below_identity})"
            )

        protein_map = self._protein_specs_from_neighborhoods(
            neighborhoods=neighborhoods,
            chain_alignments=chain_alignments,
            target_chains=context["target_chains"],
            local_residue_keys=local_residue_keys,
        )
        protein_specs = list(protein_map["specs"])
        protein_stats = dict(protein_map["stats"])
        sample_keys = list(protein_map.get("sample_keys", []) or [])
        if not protein_specs:
            return None, (
                f"no_protein_specs(stats={protein_stats},"
                f"sample_target_keys={sample_keys},"
                f"local_chain_ids={sorted(set(context.get('local_chain_ids', []) or []))},"
                f"n_local_keys={len(local_residue_keys)})"
            )

        total_contacts = int(protein_stats.get("total_contacts", 0))
        mapped_contacts = int(protein_stats.get("mapped_contacts", 0))
        local_mapped = int(protein_stats.get("local_mapped_contacts", 0))
        resname_consistent = int(protein_stats.get("resname_consistent_contacts", 0))

        mapping_ratio = float(mapped_contacts) / float(max(1, total_contacts))
        local_ratio = float(local_mapped) / float(max(1, mapped_contacts))
        resname_ratio = float(resname_consistent) / float(max(1, mapped_contacts))
        local_consistency = min(mapping_ratio, local_ratio, resname_ratio)

        # Global + local gate: mapped contacts must stay local and chemically consistent.
        if mapping_ratio < 0.5 or local_ratio < 0.8 or resname_ratio < 0.5:
            return None, (
                f"gate_failed(mapping_ratio={mapping_ratio:.2f},"
                f"local_ratio={local_ratio:.2f},resname_ratio={resname_ratio:.2f},"
                f"stats={protein_stats})"
            )

        ligand_specs = self._ligand_specs_from_neighborhoods(
            neighborhoods=neighborhoods,
            query_mol=context["query_mol"],
            ligand=context["ligand"],
            template_comp_id=comp_id,
        )

        if not protein_specs and not ligand_specs:
            return None, "no_specs"

        ion_type = self._select_ion_type_from_neighborhoods(neighborhoods)
        geom_info = self.infer_coordination_geometry(
            ion_type=ion_type,
            protein_spec_count=len(protein_specs),
            ligand_spec_count=len(ligand_specs),
            neighborhoods=neighborhoods,
        )
        coord_geometry = geom_info.get("geometry")

        entry_data = self._fetch_entry_data(entry_id)
        resolution = self._entry_resolution(entry_data)

        seq_score = self._mean(
            [
                float(aln["identity"])
                for alignments in chain_alignments.values()
                for aln in alignments
            ]
        )
        ligand_cov = self._ligand_coverage_score(ligand_specs, neighborhoods)
        coord_score = min(1.0, float(len(protein_specs) + len(ligand_specs)) / float(max(1, geom_info.get("target_cn", 6))))
        quality_score = self._resolution_quality_score(resolution)

        confidence = (
            0.30 * seq_score
            + 0.25 * local_consistency
            + 0.20 * ligand_cov
            + 0.15 * coord_score
            + 0.10 * quality_score
        )

        return (
            {
                "identifier": hit.get("identifier"),
                "entry_id": entry_id,
                "entity_id": entity_id,
                "search_score": float(hit.get("search_score", 0.0)),
                "match_mode": hit.get("match_mode"),
                "ligand_comp_id": comp_id,
                "ion_type": ion_type,
                "coordination_geometry": coord_geometry,
                "coordination_inference": geom_info,
                "protein_atom_specs": sorted(protein_specs),
                "ligand_atom_specs": sorted(ligand_specs),
                "confidence": float(confidence),
                "scores": {
                    "sequence_agreement": float(seq_score),
                    "local_consistency": float(local_consistency),
                    "ligand_mapping_coverage": float(ligand_cov),
                    "coordination_consistency": float(coord_score),
                    "structure_quality": float(quality_score),
                },
                "protein_contact_stats": protein_stats,
                "resolution": resolution,
                "n_neighborhoods": len(neighborhoods),
                "chain_alignments": chain_alignments,
            },
            None,
        )

    # -----------------------
    # Mapping helpers
    # -----------------------

    def _eligible_target_chain_alignments(
        self,
        template_seq: str,
        target_chains: Dict[str, ChainSequence],
        identity_min: float,
        eligible_chain_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return one alignment per eligible target chain whose identity passes the
        threshold. Necessary for homo-oligomeric targets where two or more chains
        tie on identity but only one of them is actually near the bound ligand.
        """
        allowed = None
        if eligible_chain_ids is not None:
            allowed = {str(x) for x in eligible_chain_ids}

        out: List[Dict[str, Any]] = []
        for chain_id, chain in target_chains.items():
            if allowed is not None and str(chain_id) not in allowed:
                continue
            aln = self._global_align_with_mapping(template_seq, chain.sequence)
            if float(aln["identity"]) < float(identity_min):
                continue
            out.append(
                {
                    "target_chain_id": chain_id,
                    "identity": float(aln["identity"]),
                    "template_to_target": aln["template_to_target"],
                }
            )
        out.sort(key=lambda d: float(d["identity"]), reverse=True)
        return out

    @staticmethod
    def _global_align_with_mapping(seq_a: str, seq_b: str) -> Dict[str, Any]:
        """Needleman-Wunsch alignment with residue-position mapping.

        Returns mapping from 1-based positions in seq_a -> 1-based positions in seq_b
        where aligned residues are identical.
        """
        a = str(seq_a or "")
        b = str(seq_b or "")

        if not a or not b:
            return {"identity": 0.0, "template_to_target": {}}

        match_score = 2
        mismatch_penalty = -1
        gap_penalty = -1

        n = len(a)
        m = len(b)

        score = [[0] * (m + 1) for _ in range(n + 1)]
        trace = [[0] * (m + 1) for _ in range(n + 1)]  # 0 diag, 1 up, 2 left

        for i in range(1, n + 1):
            score[i][0] = i * gap_penalty
            trace[i][0] = 1
        for j in range(1, m + 1):
            score[0][j] = j * gap_penalty
            trace[0][j] = 2

        for i in range(1, n + 1):
            ai = a[i - 1]
            for j in range(1, m + 1):
                bj = b[j - 1]
                diag = score[i - 1][j - 1] + (match_score if ai == bj else mismatch_penalty)
                up = score[i - 1][j] + gap_penalty
                left = score[i][j - 1] + gap_penalty

                if diag >= up and diag >= left:
                    score[i][j] = diag
                    trace[i][j] = 0
                elif up >= left:
                    score[i][j] = up
                    trace[i][j] = 1
                else:
                    score[i][j] = left
                    trace[i][j] = 2

        # Traceback.
        i = n
        j = m
        a_pos = []
        b_pos = []
        while i > 0 or j > 0:
            t = trace[i][j]
            if i > 0 and j > 0 and t == 0:
                a_pos.append(i)
                b_pos.append(j)
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or t == 1):
                a_pos.append(i)
                b_pos.append(None)
                i -= 1
            else:
                a_pos.append(None)
                b_pos.append(j)
                j -= 1

        a_pos.reverse()
        b_pos.reverse()

        mapping: Dict[int, int] = {}
        matches = 0
        for pa, pb in zip(a_pos, b_pos):
            if pa is None or pb is None:
                continue
            if a[pa - 1] == b[pb - 1]:
                mapping[int(pa)] = int(pb)
                matches += 1

        identity = float(matches) / float(max(1, len(a)))
        return {
            "identity": identity,
            "template_to_target": mapping,
        }

    def _protein_specs_from_neighborhoods(
        self,
        neighborhoods: Sequence[Dict[str, Any]],
        chain_alignments: Dict[str, List[Dict[str, Any]]],
        target_chains: Dict[str, ChainSequence],
        local_residue_keys: Optional[Sequence[Tuple[str, int, str]]] = None,
    ) -> Dict[str, Any]:
        local_keys = set(local_residue_keys or [])
        specs = set()
        stats = {
            "total_contacts": 0,
            "mapped_contacts": 0,
            "local_mapped_contacts": 0,
            "resname_consistent_contacts": 0,
        }
        sample_keys: List[Tuple[str, int, str]] = []

        for ng in neighborhoods:
            for c in ng.get("protein_contacts", []):
                asym_id = str(c.get("asym_id", ""))
                atom_name = str(c.get("atom_id", "")).upper().strip()
                seq_id = self._safe_int(c.get("seq_id"))
                tmpl_resname = str(c.get("comp_id", "")).upper().strip()

                if not asym_id or not atom_name or seq_id is None:
                    continue
                alignments = chain_alignments.get(asym_id) or []
                if not alignments:
                    continue
                stats["total_contacts"] += 1

                contact_mapped = False
                contact_local = False
                contact_resname_ok = False

                for aln in alignments:
                    t_to_q = aln.get("template_to_target", {})
                    q_pos = t_to_q.get(int(seq_id))
                    if q_pos is None:
                        continue
                    contact_mapped = True

                    target_chain_id = str(aln.get("target_chain_id"))
                    chain = target_chains.get(target_chain_id)
                    if chain is None:
                        continue

                    residue = chain.residues_by_seq.get(int(q_pos))
                    if residue is None:
                        continue
                    rkey = self._residue_key(residue)
                    if len(sample_keys) < 6:
                        sample_keys.append(rkey)

                    is_local = (not local_keys) or (rkey in local_keys)
                    if is_local:
                        contact_local = True

                    resname = str(getattr(residue, "name", "")).upper()
                    if tmpl_resname and resname == tmpl_resname:
                        contact_resname_ok = True

                    if local_keys and not is_local:
                        continue
                    if (resname, atom_name) not in self.PROTEIN_DONOR_ATOMS:
                        # Keep only canonical donor sidechain atoms by default.
                        continue

                    resnum = self._resnum_from_residue(residue)
                    if resnum is None:
                        continue

                    specs.add(f"{target_chain_id}:{resname}:{resnum}:{atom_name}")

                if contact_mapped:
                    stats["mapped_contacts"] += 1
                if contact_local:
                    stats["local_mapped_contacts"] += 1
                if contact_resname_ok:
                    stats["resname_consistent_contacts"] += 1

        return {"specs": sorted(specs), "stats": stats, "sample_keys": sample_keys}

    def _ligand_specs_from_neighborhoods(
        self,
        neighborhoods: Sequence[Dict[str, Any]],
        query_mol,
        ligand,
        template_comp_id: str,
    ) -> List[str]:
        # Contact-anchored mapping only: no broad element fallback.
        template_contact_atoms: List[Dict[str, Any]] = []

        for ng in neighborhoods:
            for c in ng.get("ligand_contacts", []):
                if str(c.get("comp_id", "")).upper() != str(template_comp_id).upper():
                    continue
                elem = str(c.get("type_symbol", "")).upper().strip()
                if elem and elem not in self.LIGAND_DONOR_ELEMENTS:
                    continue
                template_contact_atoms.append(
                    {
                        "atom_name": str(c.get("atom_id", "")).upper().strip(),
                        "atom_index": self._safe_int(c.get("atom_index")),
                        "element": elem,
                    }
                )

        query_indices = self._map_query_ligand_indices_from_template_contacts(
            query_mol=query_mol,
            template_comp_id=template_comp_id,
            template_contact_atoms=template_contact_atoms,
            ligand=ligand,
        )

        atom_names = list(getattr(ligand, "atom_names", []) or [])
        specs = []
        for idx in query_indices:
            if 0 <= int(idx) < len(atom_names):
                specs.append(f"LIG:0:{atom_names[int(idx)]}")

        return sorted(set(specs))

    def _map_query_ligand_indices_from_template_contacts(
        self,
        query_mol,
        template_comp_id: str,
        template_contact_atoms: Sequence[Dict[str, Any]],
        ligand,
    ) -> List[int]:
        if not template_contact_atoms:
            return []

        template_smiles = self._fetch_chemcomp_smiles(template_comp_id)
        t_to_q_idx = {}
        if template_smiles:
            template_mol = Chem.MolFromSmiles(template_smiles)
            if template_mol is not None:
                t_to_q_idx = self._template_to_query_atom_map_by_mcs(query_mol, template_mol)

        mapped = set()

        # 1) index-based mapping when indices are available and traceable.
        for c in template_contact_atoms:
            tidx = self._safe_int(c.get("atom_index"))
            if tidx is None:
                continue
            if int(tidx) in t_to_q_idx:
                qidx = int(t_to_q_idx[int(tidx)])
                if self._query_atom_is_valid_donor(query_mol, qidx, c.get("element")):
                    mapped.add(qidx)

        if mapped:
            return sorted(mapped)

        # 2) name-anchored fallback (still contact-anchored, no generic element fallback).
        query_atom_names = list(getattr(ligand, "atom_names", []) or [])
        name_to_idx = {}
        for i, n in enumerate(query_atom_names):
            norm = self._canonical_contact_atom_token(n)
            if norm:
                name_to_idx.setdefault(norm, []).append(i)

        for c in template_contact_atoms:
            norm = self._canonical_contact_atom_token(c.get("atom_name"))
            if not norm or norm not in name_to_idx:
                continue
            for qidx in name_to_idx[norm]:
                if self._query_atom_is_valid_donor(query_mol, int(qidx), c.get("element")):
                    mapped.add(int(qidx))
                    break

        return sorted(mapped)

    @staticmethod
    def _template_to_query_atom_map_by_mcs(
        query_mol,
        template_mol,
    ) -> Dict[int, int]:
        query = Chem.RemoveHs(Chem.Mol(query_mol))
        template = Chem.RemoveHs(Chem.Mol(template_mol))

        if query.GetNumAtoms() == 0 or template.GetNumAtoms() == 0:
            return {}

        mcs = rdFMCS.FindMCS(
            [query, template],
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
            matchValences=False,
            timeout=5,
        )
        if not mcs.smartsString:
            return {}

        pattern = Chem.MolFromSmarts(mcs.smartsString)
        if pattern is None:
            return {}

        q_match = query.GetSubstructMatch(pattern)
        t_match = template.GetSubstructMatch(pattern)
        if not q_match or not t_match:
            return {}

        return {int(ti): int(qi) for qi, ti in zip(q_match, t_match)}

    @staticmethod
    def _map_template_ligand_atoms_by_mcs(
        query_mol,
        template_mol,
        template_atom_indices: Sequence[int],
    ) -> List[int]:
        t_to_q = IonTemplateSearch._template_to_query_atom_map_by_mcs(query_mol, template_mol)
        mapped = [t_to_q[int(ti)] for ti in template_atom_indices if int(ti) in t_to_q]
        return sorted(set(mapped))

    # -----------------------
    # Candidate -> proposal
    # -----------------------

    def _build_proposed_ion_fixer_args(self, selected_template: Dict[str, Any]) -> Dict[str, Any]:
        protein_specs = list(selected_template.get("protein_atom_specs", []) or [])
        ligand_specs = list(selected_template.get("ligand_atom_specs", []) or [])

        atom_specs = []
        atom_specs.extend(protein_specs)
        atom_specs.extend(ligand_specs)

        # Keep insertion order and uniqueness.
        seen = set()
        uniq = []
        for spec in atom_specs:
            if spec not in seen:
                seen.add(spec)
                uniq.append(spec)

        return {
            "atom_specs": uniq,
            "protein_atom_specs": sorted(set(protein_specs)),
            "ligand_atom_specs": sorted(set(ligand_specs)),
            "ion_type": selected_template.get("ion_type"),
            "coordination_geometry": selected_template.get("coordination_geometry"),
            "coordination_inference": selected_template.get("coordination_inference"),
            "confidence": float(selected_template.get("confidence", 0.0)),
            "source_template": {
                "entry_id": selected_template.get("entry_id"),
                "entity_id": selected_template.get("entity_id"),
                "ligand_comp_id": selected_template.get("ligand_comp_id"),
            },
        }

    def _apply_proposal_to_options(self, proposal: Dict[str, Any]) -> bool:
        opts = self.system.options

        new_specs = list(proposal.get("atom_specs", []) or [])
        if not new_specs:
            return False

        existing_specs = list(getattr(opts, "atom_specs", []) or [])
        merged_specs = []
        seen = set()
        for spec in existing_specs + new_specs:
            if spec not in seen:
                seen.add(spec)
                merged_specs.append(spec)

        opts.atom_specs = merged_specs

        # Populate ion type only when it is not explicitly provided.
        ion_type = str(proposal.get("ion_type", "") or "").upper().strip()
        if ion_type and not str(getattr(opts, "ion_type", "") or "").strip():
            opts.ion_type = ion_type

        # Keep explicit non-default user geometry; otherwise accept inferred.
        inferred_geom = str(proposal.get("coordination_geometry", "") or "").strip()
        current_geom = str(getattr(opts, "coordination_geometry", "") or "").strip()
        if inferred_geom and (not current_geom or current_geom.lower() == "octahedral"):
            opts.coordination_geometry = inferred_geom

        return True

    def _run_ion_fixer(self):
        from .ion_fixer import IonFixer

        return IonFixer(self.system).run()

    # -----------------------
    # API + parsing
    # -----------------------

    def _fetch_nonpolymer_entity(self, entry_id: str, entity_id: str) -> Optional[dict]:
        url = f"{self.DATA_CORE_URL}/nonpolymer_entity/{entry_id}/{entity_id}"
        return self._get_json(url, timeout_s=15)

    def _fetch_entry_data(self, entry_id: str) -> Optional[dict]:
        key = str(entry_id).upper()
        if key in self._entry_data_cache:
            return self._entry_data_cache[key]

        url = f"{self.DATA_CORE_URL}/entry/{key}"
        data = self._get_json(url, timeout_s=15)
        self._entry_data_cache[key] = data
        return data

    def _fetch_template_chain_sequence(self, entry_id: str, asym_id: str) -> Optional[str]:
        key = (str(entry_id).upper(), str(asym_id))
        if key in self._chain_seq_cache:
            return self._chain_seq_cache[key]

        sequence = None
        try:
            inst_url = f"{self.DATA_CORE_URL}/polymer_entity_instance/{key[0]}/{key[1]}"
            inst = self._get_json(inst_url, timeout_s=15)
            entity_id = self._extract_polymer_entity_id_from_instance(inst)

            if entity_id is not None:
                pe_url = f"{self.DATA_CORE_URL}/polymer_entity/{key[0]}/{entity_id}"
                pe = self._get_json(pe_url, timeout_s=15)
                sequence = self._extract_polymer_sequence(pe)
        except Exception:
            sequence = None

        self._chain_seq_cache[key] = sequence
        return sequence

    def _fetch_chemcomp_smiles(self, comp_id: str) -> Optional[str]:
        cid = str(comp_id).upper().strip()
        if not cid:
            return None

        url = f"{self.DATA_CORE_URL}/chemcomp/{cid}"
        data = self._get_json(url, timeout_s=15)
        if not data:
            return None

        # rcsb_chem_comp_descriptor: dict with keys like SMILES / SMILES_stereo (new schema)
        rcsb_desc = data.get("rcsb_chem_comp_descriptor")
        if isinstance(rcsb_desc, dict):
            for key in ("SMILES_stereo", "SMILES"):
                val = str(rcsb_desc.get(key, "") or "").strip()
                if val:
                    return val

        # pdbx_chem_comp_descriptor: list of dicts with type/program/descriptor (legacy)
        pdbx_desc = data.get("pdbx_chem_comp_descriptor", []) or []
        if isinstance(pdbx_desc, list):
            for d in pdbx_desc:
                if not isinstance(d, dict):
                    continue
                typ = str(d.get("type", "")).upper()
                prog = str(d.get("program", "")).upper()
                val = str(d.get("descriptor", "") or "").strip()
                if not val:
                    continue
                if "SMILES" in typ or "SMILES" in prog:
                    return val

        # Fallback key used by some snapshots.
        alt = str(data.get("smiles", "") or "").strip()
        return alt or None

    def _fetch_residue_interaction_cif(self, entry_id: str, ligand_comp_id: str) -> Optional[str]:
        params = {
            "label_comp_id": str(ligand_comp_id).upper(),
            "radius": "3.2",
            "encoding": "cif",
        }

        # ModelServer path naming has varied; try a few resilient variants.
        # Only the camelCase form is currently live; the others return HTML 404
        # bodies on some deployments, so accept a response only if it parses
        # as a CIF data block.
        for endpoint in ("residueInteraction", "residue_interaction", "residue-interaction"):
            query = urlparse.urlencode(params)
            url = f"{self.MODELSERVER_BASE}/{entry_id}/{endpoint}?{query}"
            txt = self._get_text(url, timeout_s=25)
            if txt and txt.lstrip().startswith("data_"):
                return txt

        return None

    def _extract_coordination_neighborhoods(
        self,
        interaction_text: Optional[str],
        ligand_comp_id: str,
        metal_elements: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if not interaction_text:
            return []

        tables = self._parse_cif_loops(interaction_text)
        rows = list(tables.get("struct_conn", []) or [])
        if not rows:
            return []

        metal_set = {str(x).upper().strip() for x in metal_elements}
        ligand_comp = str(ligand_comp_id).upper().strip()

        grouped: Dict[Tuple[str, Any, str, str], Dict[str, Any]] = {}

        for row in rows:
            conn_type = str(row.get("conn_type_id", "") or "").lower()
            if "metal" not in conn_type:
                continue

            p1 = self._extract_struct_conn_partner(row, "ptnr1")
            p2 = self._extract_struct_conn_partner(row, "ptnr2")

            is_p1_metal = str(p1.get("type_symbol", "")).upper() in metal_set
            is_p2_metal = str(p2.get("type_symbol", "")).upper() in metal_set

            if is_p1_metal == is_p2_metal:
                continue

            metal = p1 if is_p1_metal else p2
            donor = p2 if is_p1_metal else p1

            key = (
                str(metal.get("asym_id", "")),
                metal.get("seq_id"),
                str(metal.get("atom_id", "")),
                str(metal.get("type_symbol", "")).upper(),
            )
            if key not in grouped:
                grouped[key] = {
                    "metal_element": str(metal.get("type_symbol", "")).upper(),
                    "metal_comp_id": str(metal.get("comp_id", "")).upper(),
                    "metal_asym_id": str(metal.get("asym_id", "")),
                    "metal_seq_id": metal.get("seq_id"),
                    "metal_atom_id": str(metal.get("atom_id", "")),
                    "protein_contacts": [],
                    "ligand_contacts": [],
                }

            donor_comp = str(donor.get("comp_id", "")).upper()
            donor_atom = str(donor.get("atom_id", "")).upper()
            donor_elem = str(donor.get("type_symbol", "")).upper()

            contact = {
                "asym_id": str(donor.get("asym_id", "")),
                "seq_id": donor.get("seq_id"),
                "comp_id": donor_comp,
                "atom_id": donor_atom,
                "type_symbol": donor_elem,
                "atom_index": donor.get("atom_index"),
            }

            if donor_comp in self.AA3_TO_1 and donor.get("seq_id") is not None:
                grouped[key]["protein_contacts"].append(contact)
            elif donor_comp == ligand_comp or (
                donor.get("seq_id") is None and donor_comp not in {"HOH", "WAT"} and donor_elem not in metal_set
            ):
                grouped[key]["ligand_contacts"].append(contact)

        out = []
        for ng in grouped.values():
            if not ng["protein_contacts"]:
                continue
            out.append(ng)

        return out

    @staticmethod
    def _extract_struct_conn_partner(row: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        # RCSB uses fields like ptnr1_label_asym_id, ptnr2_label_seq_id, etc.
        asym = row.get(f"{prefix}_label_asym_id") or row.get(f"{prefix}_auth_asym_id")
        seq = row.get(f"{prefix}_label_seq_id") or row.get(f"{prefix}_auth_seq_id")
        comp = row.get(f"{prefix}_label_comp_id") or row.get(f"{prefix}_auth_comp_id")
        atom = row.get(f"{prefix}_label_atom_id") or row.get(f"{prefix}_auth_atom_id")
        elem = row.get(f"{prefix}_type_symbol")

        out = {
            "asym_id": str(asym or "").strip(),
            "seq_id": IonTemplateSearch._safe_int(seq),
            "comp_id": str(comp or "").strip(),
            "atom_id": str(atom or "").strip(),
            "type_symbol": str(elem or "").strip().upper(),
            "atom_index": IonTemplateSearch._safe_int(
                row.get(f"{prefix}_label_atom_id_id")
                or row.get(f"{prefix}_label_atom_id_ordinal")
                or row.get(f"{prefix}_atom_id")
            ),
        }

        # Some outputs omit type_symbol; infer from atom/comp when possible.
        if not out["type_symbol"]:
            if out["comp_id"]:
                maybe = str(out["comp_id"]).strip().upper()
                if len(maybe) <= 2:
                    out["type_symbol"] = maybe
            if not out["type_symbol"] and out["atom_id"]:
                out["type_symbol"] = IonTemplateSearch._infer_element_from_atom_id(out["atom_id"])

        return out

    @staticmethod
    def _parse_cif_loops(cif_text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Very small CIF loop parser sufficient for struct_conn extraction.

        This parser handles standard single-line loop rows and quoted tokens.
        It intentionally keeps scope narrow and returns best-effort rows.
        """
        if not cif_text:
            return {}

        lines = []
        for raw in str(cif_text).splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)

        tables: Dict[str, List[Dict[str, Any]]] = {}
        i = 0
        n = len(lines)

        while i < n:
            if lines[i] != "loop_":
                i += 1
                continue

            i += 1
            tags: List[str] = []
            while i < n and lines[i].startswith("_"):
                tags.append(lines[i])
                i += 1

            if not tags:
                continue

            rows_raw: List[str] = []
            while i < n and lines[i] != "loop_" and not lines[i].startswith("_"):
                rows_raw.append(lines[i])
                i += 1

            first = tags[0]
            if "." not in first:
                continue
            category = first.split(".", 1)[0].lstrip("_")
            columns = [t.split(".", 1)[1] if "." in t else t.lstrip("_") for t in tags]
            ncol = len(columns)

            data: List[Dict[str, Any]] = []
            token_buffer: List[str] = []

            for row in rows_raw:
                try:
                    toks = shlex.split(row, posix=True)
                except Exception:
                    toks = row.split()
                token_buffer.extend(toks)

                while len(token_buffer) >= ncol:
                    vals = token_buffer[:ncol]
                    token_buffer = token_buffer[ncol:]
                    rec = {}
                    for c, v in zip(columns, vals):
                        if v in {".", "?"}:
                            rec[c] = None
                        else:
                            rec[c] = v
                    data.append(rec)

            if data:
                tables.setdefault(category, []).extend(data)

        return tables

    # -----------------------
    # Scoring helpers
    # -----------------------

    def _select_ion_type_from_neighborhoods(self, neighborhoods: Sequence[Dict[str, Any]]) -> Optional[str]:
        metals = [str(n.get("metal_element", "")).upper() for n in neighborhoods]
        metals = [m for m in metals if m]
        if not metals:
            return None
        return Counter(metals).most_common(1)[0][0]

    def infer_coordination_geometry(
        self,
        ion_type: Optional[str],
        protein_spec_count: int,
        ligand_spec_count: int,
        neighborhoods: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        ion = str(ion_type or "").upper().strip()
        observed = int(max(0, protein_spec_count) + max(0, ligand_spec_count))

        # Also estimate observed contacts directly from neighborhoods.
        observed_from_contacts = 0
        for ng in neighborhoods:
            observed_from_contacts = max(
                observed_from_contacts,
                len(list(ng.get("protein_contacts", []) or []))
                + len(list(ng.get("ligand_contacts", []) or [])),
            )
        observed = max(observed, observed_from_contacts)

        if ion in self.MG_LIKE_IONS:
            target_cn = 6
        elif ion in {"ZN", "CU", "NI", "CO", "CD"}:
            target_cn = 4 if observed <= 4 else 6
        elif ion in {"MN", "FE"}:
            target_cn = 6 if observed >= 5 else 4
        else:
            target_cn = max(2, min(7, observed if observed > 0 else 6))

        missing_sites = max(0, int(target_cn) - int(observed))
        geom = self.GEOMETRY_BY_CN.get(int(target_cn))
        if geom is None:
            geom = self.GEOMETRY_BY_CN.get(max(2, min(7, int(observed))), "Octahedral")

        return {
            "geometry": geom,
            "target_cn": int(target_cn),
            "observed_cn": int(observed),
            "missing_sites": int(missing_sites),
            "water_completion_likely": bool(missing_sites > 0),
            "ion_type": ion or None,
        }

    def _infer_coordination_geometry(self, coordination_count: int) -> Optional[str]:
        # Backward-compatible helper retained for tests and callers.
        return self.GEOMETRY_BY_CN.get(int(coordination_count))

    @staticmethod
    def _ligand_coverage_score(
        ligand_specs: Sequence[str],
        neighborhoods: Sequence[Dict[str, Any]],
    ) -> float:
        expected = 0
        for ng in neighborhoods:
            expected += len(list(ng.get("ligand_contacts", []) or []))

        if expected <= 0:
            return 0.5 if ligand_specs else 0.0

        if not ligand_specs:
            # Template has confirmed metal-coordinating ligand atoms, but the
            # atom-level mapping to the query couldn't be made (e.g., generic
            # SDF atom names + no atom_index in struct_conn). Still give partial
            # credit since the structural evidence exists.
            return 0.5

        return min(1.0, float(len(ligand_specs)) / float(max(1, expected)))

    @staticmethod
    def _resolution_quality_score(resolution: Optional[float]) -> float:
        if resolution is None:
            return 0.5
        r = float(resolution)
        # 1.5 A -> high, 3.5+ A -> low
        return max(0.0, min(1.0, (3.5 - r) / 2.0))

    @staticmethod
    def _entry_resolution(entry_data: Optional[dict]) -> Optional[float]:
        if not entry_data:
            return None
        rinfo = entry_data.get("rcsb_entry_info", {})
        vals = list(rinfo.get("resolution_combined", []) or [])
        nums = []
        for v in vals:
            try:
                nums.append(float(v))
            except Exception:
                pass
        if not nums:
            return None
        return min(nums)

    @staticmethod
    def _mean(vals: Iterable[float]) -> float:
        arr = [float(v) for v in vals]
        if not arr:
            return 0.0
        return float(sum(arr) / float(len(arr)))

    # -----------------------
    # REST helpers
    # -----------------------

    def _post_json(self, url: str, payload: dict, timeout_s: int = 20) -> Optional[dict]:
        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urlrequest.urlopen(req, timeout=timeout_s) as resp:
                txt = resp.read().decode("utf-8")
            if not txt.strip():
                return None
            return json.loads(txt)
        except (urlerror.URLError, json.JSONDecodeError, TimeoutError) as exc:
            self._log(f"[ion_template_search] POST failed ({url}): {type(exc).__name__}: {exc}")
            return None

    def _get_json(self, url: str, timeout_s: int = 20) -> Optional[dict]:
        req = urlrequest.Request(url, method="GET", headers={"Accept": "application/json"})
        try:
            with urlrequest.urlopen(req, timeout=timeout_s) as resp:
                txt = resp.read().decode("utf-8")
            if not txt.strip():
                return None
            return json.loads(txt)
        except (urlerror.URLError, json.JSONDecodeError, TimeoutError):
            return None

    def _get_text(self, url: str, timeout_s: int = 20) -> Optional[str]:
        req = urlrequest.Request(url, method="GET")
        try:
            with urlrequest.urlopen(req, timeout=timeout_s) as resp:
                status = getattr(resp, "status", 200)
                if not (200 <= int(status) < 300):
                    return None
                txt = resp.read().decode("utf-8")
            return txt or None
        except urlerror.HTTPError:
            return None
        except (urlerror.URLError, TimeoutError):
            return None

    # -----------------------
    # Data extraction helpers
    # -----------------------

    @staticmethod
    def _extract_nonpoly_comp_id(nonpoly: Optional[dict]) -> Optional[str]:
        if not nonpoly:
            return None
        ids = nonpoly.get("rcsb_nonpolymer_entity_container_identifiers", {})
        comp_id = ids.get("nonpolymer_comp_id")
        if comp_id:
            return str(comp_id).upper().strip()
        return None

    @staticmethod
    def _extract_polymer_entity_id_from_instance(instance_data: Optional[dict]) -> Optional[str]:
        if not instance_data:
            return None

        cids = instance_data.get("rcsb_polymer_entity_instance_container_identifiers", {})
        ent_id = cids.get("entity_id")
        if ent_id is not None:
            return str(ent_id)

        # fallback field observed in some snapshots
        rcsb_id = cids.get("rcsb_id")
        if rcsb_id and "_" in str(rcsb_id):
            return str(rcsb_id).split("_", 1)[1]

        return None

    @staticmethod
    def _extract_polymer_sequence(poly_entity: Optional[dict]) -> Optional[str]:
        if not poly_entity:
            return None

        entity_poly = poly_entity.get("entity_poly", {})
        raw = str(entity_poly.get("pdbx_seq_one_letter_code_can", "") or "")
        if not raw:
            return None

        # Remove whitespace/newlines and non-residue characters.
        seq = re.sub(r"[^A-Za-z]", "", raw).upper()
        return seq or None

    # -----------------------
    # Output
    # -----------------------

    def _write_outputs(self, report: Dict[str, Any], selected_template: Dict[str, Any], proposed_args: Dict[str, Any]) -> None:
        os.makedirs(self.output, exist_ok=True)

        report_path = os.path.join(self.output, "report.json")
        selected_path = os.path.join(self.output, "selected_template.json")
        proposal_path = os.path.join(self.output, "proposed_ion_fixer_args.json")

        self._write_json(report_path, report)
        self._write_json(selected_path, selected_template or {})
        self._write_json(proposal_path, proposed_args or {})

    def _write_json(self, path: str, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(payload), f, indent=2, sort_keys=True)

    def _json_safe(self, obj: Any):
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._json_safe(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._json_safe(v) for v in obj]
        if isinstance(obj, set):
            return [self._json_safe(v) for v in sorted(obj)]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    # -----------------------
    # Formatting/helpers
    # -----------------------

    def _context_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        chains = context.get("target_chains", {})
        local_keys = list(context.get("local_residue_keys", []) or [])
        return {
            "n_target_chains": len(chains),
            "target_chain_lengths": {cid: len(ch.sequence) for cid, ch in chains.items()},
            "local_chain_ids": sorted(list(context.get("local_chain_ids", []) or [])),
            "n_local_residues": len(local_keys),
            "local_residue_keys_sample": [list(k) for k in sorted(local_keys)[:30]],
            "local_radius_A": context.get("local_radius_A"),
            "query_comp_id_hint": context.get("query_comp_id_hint"),
            "query_smiles": context.get("query_smiles"),
        }

    @staticmethod
    def _compact_template_summary(tpl: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "identifier": tpl.get("identifier"),
            "entry_id": tpl.get("entry_id"),
            "entity_id": tpl.get("entity_id"),
            "match_mode": tpl.get("match_mode"),
            "search_score": float(tpl.get("search_score", 0.0)),
            "confidence": float(tpl.get("confidence", 0.0)),
            "n_protein_specs": len(list(tpl.get("protein_atom_specs", []) or [])),
            "n_ligand_specs": len(list(tpl.get("ligand_atom_specs", []) or [])),
            "local_consistency": float((tpl.get("scores", {}) or {}).get("local_consistency", 0.0)),
            "missing_coordination_sites": int(((tpl.get("coordination_inference", {}) or {}).get("missing_sites", 0))),
        }

    @staticmethod
    def _summarize_ion_fixer_result(result: Any) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return {"type": type(result).__name__}

        out = {
            "n_cycles": result.get("n_cycles"),
            "n_specs": len(list(result.get("fixed_target_dist_A", []) or [])),
        }
        if "refinement_outputs" in result and isinstance(result["refinement_outputs"], dict):
            out["refinement_dir"] = result["refinement_outputs"].get("refinement_dir")
        return out

    def _metal_elements(self) -> List[str]:
        raw = str(self._opt("its_ion_elements", "") or "").strip()
        if not raw:
            return list(self.DEFAULT_METAL_ELEMENTS)

        out = []
        for tok in raw.split(","):
            val = tok.strip().upper()
            if val:
                out.append(val)
        return out or list(self.DEFAULT_METAL_ELEMENTS)

    def _opt(self, name: str, default: Any) -> Any:
        return getattr(self.system.options, name, default)

    def _log(self, msg: str) -> None:
        if hasattr(self.system, "log"):
            self.system.log(msg)
        else:
            print(msg)

    @staticmethod
    def _chain_id_from_residue(res) -> str:
        chain = getattr(res, "chain", "")
        chain_id = getattr(chain, "id", chain)
        return str(chain_id)

    @classmethod
    def _residue_key(cls, residue) -> Tuple[str, int, str]:
        chain_id = cls._chain_id_from_residue(residue)
        resnum = cls._resnum_from_residue(residue)
        resname = str(getattr(residue, "name", "")).upper().strip()
        return (chain_id, int(resnum) if resnum is not None else -1, resname)

    @staticmethod
    def _atom_xyz(atom) -> Tuple[float, float, float]:
        return (float(getattr(atom, "xx", 0.0)), float(getattr(atom, "xy", 0.0)), float(getattr(atom, "xz", 0.0)))

    @staticmethod
    def _is_heavy_atom(atom) -> bool:
        elem = str(getattr(atom, "element_name", "") or "").upper().strip()
        if elem:
            return elem != "H"
        name = str(getattr(atom, "name", "") or "").upper().strip()
        return not name.startswith("H")

    @classmethod
    def _canonical_contact_atom_token(cls, name: Any) -> str:
        raw = re.sub(r"[^A-Za-z0-9]", "", str(name or "").upper())
        if not raw:
            return ""
        m = re.match(r"^([A-Z]+)(\d+)", raw)
        if m:
            return f"{m.group(1)}{m.group(2)}"
        return raw

    @classmethod
    def _infer_element_from_atom_id(cls, atom_id: Any) -> str:
        token = re.sub(r"[^A-Za-z]", "", str(atom_id or "").upper())
        if not token:
            return ""
        if len(token) >= 2 and token[:2] in cls.ELEMENT_SYMBOLS:
            return token[:2]
        if token[0] in cls.ELEMENT_SYMBOLS:
            return token[0]
        return token[:2]

    @classmethod
    def _query_atom_is_valid_donor(cls, query_mol, qidx: int, expected_element: Optional[str]) -> bool:
        try:
            mol = Chem.RemoveHs(Chem.Mol(query_mol))
            if qidx < 0 or qidx >= mol.GetNumAtoms():
                return False
            elem = str(mol.GetAtomWithIdx(int(qidx)).GetSymbol()).upper()
            if elem not in cls.LIGAND_DONOR_ELEMENTS:
                return False
            exp = str(expected_element or "").upper().strip()
            if exp and exp in cls.LIGAND_DONOR_ELEMENTS and elem != exp:
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def _resnum_from_residue(res) -> Optional[int]:
        num = getattr(res, "number", None)
        if num is None:
            num = getattr(res, "idx", None)
        return IonTemplateSearch._safe_int(num)

    @staticmethod
    def _safe_int(x) -> Optional[int]:
        try:
            if x is None:
                return None
            s = str(x).strip()
            if not s or s in {".", "?"}:
                return None
            return int(float(s))
        except Exception:
            return None


__all__ = ["IonTemplateSearch"]
