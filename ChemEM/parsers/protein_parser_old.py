# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

import sys
import math
import tempfile
from collections import Counter
from typing import List

import numpy as np
import parmed

from openmm import app, unit
from openmm.app import PDBFile, Modeller, Topology
from openmm.app import element as omm_element
from pdbfixer import PDBFixer

from .models import Protein

from ChemEM.parsers.remodel.topology_ops import ensure_water_geometry_types
from ChemEM.parsers.mapping import build_residue_map_by_positions
#from ChemEM.tools.biomolecule import (
#    ensure_water_geometry_types,
#    build_residue_map_by_positions,
#)


class ProteinParser:
    # -------------------------
    # Residue name inventories
    # -------------------------
    _AA3 = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "MSE", "SEC"
    }
    _WATER = {"HOH", "WAT", "H2O", "SOL", "TIP", "TIP3", "T3P"}

    _ION_RESNAMES = {
        "NA", "K", "CL", "CA", "MG", "ZN", "MN", "FE", "FE2", "CU", "CO", "NI", "CD", "SR", "CS", "BR", "IOD", "I", "F",
        "SO4", "PO4", "NO3", "CO3",
    }
    _COFACTORS_COMMON = {
        "HEM", "HEC", "HEA", "HEME", "HEB", "FAD", "FMN", "NAD", "NADP", "ATP", "ADP", "AMP", "GDP", "GTP",
        "SAM", "SAH", "PLP", "PHA"
    }
    _GLYCAN_CODES = {"NAG", "NDG", "BMA", "MAN", "GAL", "GLC", "FUC", "BGC", "SIA", "NGA", "A2G"}

    # Nucleic acids
    _RNA_RESNAMES = {"A", "C", "G", "U", "I"}
    _DNA_RESNAMES = {"DA", "DC", "DG", "DT", "DU", "DI"}
    _NA_CANON = _RNA_RESNAMES | _DNA_RESNAMES

    _NA_SUGAR_HINTS = {
        "C1'", "C2'", "C3'", "C4'", "C5'",
        "O2'", "O3'", "O4'", "O5'",
    }
    _NA_PHOS_HINTS = {"P", "OP1", "OP2", "OP3", "O1P", "O2P"}

    # -------------------------
    # Component classification
    # -------------------------
    @staticmethod
    def _has_backbone(res) -> bool:
        names = {a.name for a in res.atoms()}
        return {"N", "CA", "C"}.issubset(names)

    @classmethod
    def _is_protein_res(cls, res) -> bool:
        return cls._has_backbone(res)

    @classmethod
    def _is_water(cls, res) -> bool:
        if res.name in cls._WATER:
            return True
        atoms = list(res.atoms())
        elems = {a.element.symbol if a.element else None for a in atoms}
        return elems.issubset({"O", "H", None}) and len(atoms) <= 4

    @classmethod
    def _is_ion(cls, res) -> bool:
        if res.name in cls._ION_RESNAMES:
            return True
        atoms = list(res.atoms())
        if len(atoms) == 1:
            el = atoms[0].element.symbol if atoms[0].element else ""
            return el in {"Na", "K", "Cl", "Ca", "Mg", "Zn", "Mn", "Fe", "Cu", "Co", "Ni", "Cd", "Sr", "Cs", "Br", "I", "F"}
        return False

    @classmethod
    def _is_cofactor(cls, res) -> bool:
        return res.name in cls._COFACTORS_COMMON

    @classmethod
    def _is_glycan(cls, res) -> bool:
        return res.name in cls._GLYCAN_CODES

    # -------------------------
    # Nucleic-acid detection
    # -------------------------
    @staticmethod
    def _norm_atom_name(name: str) -> str:
        # normalize star/prime variants (e.g. O3* vs O3')
        return name.replace("*", "'").strip()

    @classmethod
    def _has_nucleic_backbone(cls, res) -> bool:
        names = {cls._norm_atom_name(a.name) for a in res.atoms()}
        sugar_count = sum(1 for x in cls._NA_SUGAR_HINTS if x in names)
        has_sugar = sugar_count >= 4
        has_p = "P" in names
        has_po = (("OP1" in names or "O1P" in names) and ("OP2" in names or "O2P" in names)) or ("OP3" in names)
        return has_sugar and has_p and has_po

    @classmethod
    def _is_nucleic_acid_res(cls, res) -> bool:
        if res.name in cls._NA_CANON:
            return True
        return cls._has_nucleic_backbone(res)

    # -------------------------
    # Scan + report
    # -------------------------
    @classmethod
    def scan_components(cls, top):
        rep = {
            "protein_res": 0,
            "nonstd_prot": Counter(),
            "na_res": 0,
            "dna": Counter(),
            "rna": Counter(),
            "na_modified": Counter(),
            "waters": Counter(),
            "ions": Counter(),
            "ligands": Counter(),
            "cofactors": Counter(),
            "glycans": Counter(),
            "chains": Counter(),
            "na_chains": Counter(),
        }

        for ch in top.chains():
            prot_in_chain = 0
            na_in_chain = 0

            for res in ch.residues():
                if cls._is_water(res):
                    rep["waters"][res.name] += 1
                    continue

                if cls._has_backbone(res):
                    rep["protein_res"] += 1
                    prot_in_chain += 1
                    if res.name not in cls._AA3:
                        rep["nonstd_prot"][res.name] += 1
                    continue

                if cls._is_nucleic_acid_res(res):
                    rep["na_res"] += 1
                    na_in_chain += 1
                    if res.name in cls._DNA_RESNAMES:
                        rep["dna"][res.name] += 1
                    elif res.name in cls._RNA_RESNAMES:
                        rep["rna"][res.name] += 1
                    else:
                        rep["na_modified"][res.name] += 1
                    continue

                if cls._is_ion(res):
                    rep["ions"][res.name] += 1
                    continue
                if cls._is_glycan(res):
                    rep["glycans"][res.name] += 1
                    continue
                if cls._is_cofactor(res):
                    rep["cofactors"][res.name] += 1
                    continue

                rep["ligands"][res.name] += 1

            if prot_in_chain:
                rep["chains"][ch.id or "?"] += prot_in_chain
            if na_in_chain:
                rep["na_chains"][ch.id or "?"] += na_in_chain

        rep["has_waters"] = sum(rep["waters"].values()) > 0
        rep["has_ions"] = sum(rep["ions"].values()) > 0
        rep["has_ligs"] = sum(rep["ligands"].values()) + sum(rep["cofactors"].values()) > 0
        rep["has_nonstd"] = sum(rep["nonstd_prot"].values()) > 0
        rep["has_na"] = rep["na_res"] > 0
        return rep

    @staticmethod
    def print_component_report(rep) -> None:
        def show(lbl, x):
            print(f"{lbl:<24} {x}")

        print("=== Structure contents ===")
        show("Protein residues", rep["protein_res"])
        if rep["nonstd_prot"]:
            show("Non-standard AAs", ", ".join(f"{k}:{v}" for k, v in rep["nonstd_prot"].most_common()))

        show("NA residues", rep.get("na_res", 0))
        if rep.get("dna"):
            show("  DNA names", ", ".join(f"{k}:{v}" for k, v in rep["dna"].most_common()))
        if rep.get("rna"):
            show("  RNA names", ", ".join(f"{k}:{v}" for k, v in rep["rna"].most_common()))
        if rep.get("na_modified"):
            show("  NA modified", ", ".join(f"{k}:{v}" for k, v in rep["na_modified"].most_common()))
        if rep.get("na_chains"):
            show("NA chains", ", ".join(f"{cid}:{n}" for cid, n in rep["na_chains"].items()))

        show("Waters (res)", sum(rep["waters"].values()))
        if rep["waters"]:
            show("  Water names", ", ".join(f"{k}:{v}" for k, v in rep["waters"].most_common()))
        show("Ions (res)", sum(rep["ions"].values()))
        if rep["ions"]:
            show("  Ion names", ", ".join(f"{k}:{v}" for k, v in rep["ions"].most_common()))
        show("Cofactors (res)", sum(rep["cofactors"].values()))
        if rep["cofactors"]:
            show("  Cofactor names", ", ".join(f"{k}:{v}" for k, v in rep["cofactors"].most_common()))
        show("Glycans (res)", sum(rep["glycans"].values()))
        if rep["glycans"]:
            show("  Glycan names", ", ".join(f"{k}:{v}" for k, v in rep["glycans"].most_common()))
        show("Ligands (res)", sum(rep["ligands"].values()))
        if rep["ligands"]:
            show("  Ligand names", ", ".join(f"{k}:{v}" for k, v in rep["ligands"].most_common()))
        if rep["chains"]:
            show("Protein chains", ", ".join(f"{cid}:{n}" for cid, n in rep["chains"].items()))
        print("==========================")

    # -------------------------
    # Forcefield selection
    # -------------------------
    @staticmethod
    def _ff_loads(files: List[str]):
        try:
            ff = app.ForceField(*files)
            return ff, []
        except Exception as e:
            return None, [f"Failed to load {files}: {e}"]

    @classmethod
    def pick_forcefields_from_inventory(
        cls,
        rep,
        prefer_protein: str = "amber14/protein.ff14SB.xml",
        prefer_water: str = "amber14/tip3p.xml",
        try_implicit: bool = True,
    ):
        """
        Returns (ff, notes, mode).
        Mode is one of: explicit, implicit, vacuum.
        """
        notes: List[str] = []

        # If NA present, prefer amber14-all.xml (includes DNA/RNA templates)
        protein_ff = "amber14-all.xml" if rep.get("has_na", False) else prefer_protein
        if rep.get("has_na", False):
            notes.append("Detected nucleic acids: preferring amber14-all.xml (includes DNA/RNA templates).")

        implicit_xml = "implicit/gbn2.xml"
        solvent_mode = "explicit" if rep["has_waters"] else "implicit"

        if solvent_mode == "explicit":
            files = [protein_ff, prefer_water]
            ff, errs = cls._ff_loads(files)
            if ff is None:
                for alt in ["amber14/tip3pfb.xml", "amber14/spce.xml", "amber14/tip4pew.xml", "amber14/tip4pfb.xml"]:
                    if alt == prefer_water:
                        continue
                    ff, errs = cls._ff_loads([protein_ff, alt])
                    if ff:
                        notes.append(f"Preferred water {prefer_water} failed; using {alt}.")
                        return ff, notes, "explicit"

                ff, errs = cls._ff_loads([protein_ff])
                if ff:
                    notes.append("No water XML loaded; running in vacuum with protein-only parameters.")
                    return ff, notes, "vacuum"
                raise RuntimeError("\n".join(errs))

            return ff, notes, "explicit"

        # implicit requested
        if try_implicit:
            ff, errs = cls._ff_loads([protein_ff, implicit_xml])
            if ff:
                if rep["has_ions"]:
                    ff2, errs2 = cls._ff_loads([protein_ff, implicit_xml, prefer_water])
                    if ff2:
                        notes.append("Implicit solvent + structural ions: added water XML to supply ion parameters.")
                        return ff2, notes, "implicit"
                    notes.append("Could not add water XML for ion types; continuing with implicit only.")
                return ff, notes, "implicit"
            notes.append("implicit/gbn2.xml not found or failed to load; falling back.")

        # fallback vacuum
        if rep["has_ions"]:
            ff, errs = cls._ff_loads([protein_ff, prefer_water])
            if ff:
                notes.append("GBn2 unavailable; using protein+water XML for ion parameters (no bulk waters added).")
                return ff, notes, "vacuum"

        ff, errs = cls._ff_loads([protein_ff])
        if ff:
            notes.append("GBn2 unavailable; running in vacuum.")
            return ff, notes, "vacuum"

        raise RuntimeError("\n".join(errs))

    @classmethod
    def analyze_and_autopick_ff(
        cls,
        protein_file: str,
        prefer_protein: str = "amber14/protein.ff14SB.xml",
        prefer_water: str = "amber14/tip3p.xml",
    ):
        pdb = app.PDBFile(protein_file)
        rep = cls.scan_components(pdb.topology)
        cls.print_component_report(rep)
        ff, notes, mode = cls.pick_forcefields_from_inventory(
            rep, prefer_protein=prefer_protein, prefer_water=prefer_water, try_implicit=True
        )
        return ff, rep, notes, mode

    # -------------------------
    # Main entry point
    # -------------------------
    @staticmethod
    def load_protein_structure(protein_file, forcefield, prefer_water="amber14/tip3p.xml"):
        warnings = []

        comp_report = None
        if not forcefield:
            omm_forcefield, comp_report, notes, solvent_mode = ProteinParser.analyze_and_autopick_ff(
                protein_file,
                prefer_protein="amber14/protein.ff14SB.xml",
                prefer_water=prefer_water,
            )
            for n in notes:
                print("[FF note]", n)
        else:
            omm_forcefield = ProteinParser.get_forcefield(forcefield)

        # If we didn't compute a component report, do a quick one now
        if comp_report is None:
            try:
                pdb_probe = app.PDBFile(protein_file)
                comp_report = ProteinParser.scan_components(pdb_probe.topology)
            except Exception:
                comp_report = {"has_na": False}

        has_na = bool(comp_report.get("has_na", False))

        # Attempt 1: direct load (keep original behaviour for protein-only)
        if not has_na:
            try:
                receptor_pdbfile = PDBFile(protein_file)
                receptor_system = omm_forcefield.createSystem(receptor_pdbfile.topology)

                receptor_structure = parmed.openmm.load_topology(
                    receptor_pdbfile.topology, receptor_system, xyz=receptor_pdbfile.positions
                )
                patched = ensure_water_geometry_types(receptor_structure, water_model=prefer_water)
                print(f"[ParmEd] Patched water geometry on {patched} residues.")
                return Protein(protein_file, receptor_system, receptor_structure, omm_forcefield)

            except Exception as e:
                warnings.append(
                    f"ChemEM-Error: receptor failed building direct from file {protein_file}.\n Full Error {e}\n"
                )

        # Attempt 2: PDBFixer + chain splitting + NA-safe rebuilds
        try:
            fixer = PDBFixer(protein_file)

            receptor_openmm = PDBFile(protein_file)
            ori_top = receptor_openmm.topology
            ori_pos = receptor_openmm.positions

            receptor_pdbfile, receptor_system = ProteinParser.build_openmm_model_from_pdbfixer(
                fixer, omm_forcefield, receptor_openmm
            )
            
            receptor_structure = parmed.openmm.load_topology(
                receptor_pdbfile.topology, receptor_system, xyz=receptor_pdbfile.positions
            )
            
            last_label_map = build_residue_map_by_positions(
                ori_top,
                ori_pos,
                receptor_structure.topology,
                receptor_structure.positions,
                tol_ang=1e-3,
            )

            patched = ensure_water_geometry_types(receptor_structure, water_model=prefer_water)
            print(f"[ParmEd] Patched water geometry on {patched} residues.")

            return Protein(
                protein_file,
                receptor_system,
                receptor_structure,
                omm_forcefield,
                residue_map=last_label_map,
            )

        except Exception as e:
            warnings.append(
                f"ChemEM-Error: receptor failed building with chain truncation {protein_file}.\n Full Error {e}\n"
            )

        for error in warnings:
            print(error)
        sys.exit()

    # -------------------------
    # OpenMM ForceField builder (keep original API)
    # -------------------------
    @staticmethod
    def get_forcefield(forcefield):
        if isinstance(forcefield, str):
            forcefield = [forcefield]

        inc_forcefields = []
        if "amber14" in forcefield:
            inc_forcefields.append("amber14-all.xml")
        elif "charmm36" in forcefield:
            inc_forcefields.append("charmm36.xml")
            inc_forcefields.append("implicit/gbn2.xml")
        else:
            inc_forcefields += ["amber14-all.xml", "implicit/gbn2.xml", "amber14/tip3pfb.xml"]

        for f in forcefield:
            if f not in ["amber14", "charmm36"] and f not in inc_forcefields:
                inc_forcefields.append(f)

        return app.ForceField(*inc_forcefields)

    # -------------------------
    # PDBFixer workflow helpers
    # -------------------------
    @staticmethod
    def model_to_fixer_interchange(modeller):
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp:
            PDBFile.writeFile(modeller.topology, modeller.positions, temp)
            temp.flush()
            receptor_pdbfile = PDBFixer(temp.name)
        return receptor_pdbfile

    @staticmethod
    def fixer_to_model_interchange(fixer):
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp:
            PDBFile.writeFile(fixer.topology, fixer.positions, temp)
            temp.flush()
            receptor_pdbfile = PDBFile(temp.name)
        return receptor_pdbfile

    # -------------------------
    # NA-safe remodel helpers (TEMPy-ReFF style)
    # -------------------------
    @staticmethod
    def _delete_all_hydrogens(modeller: Modeller) -> int:
        hs = [a for a in modeller.topology.atoms()
              if a.element is not None and a.element == omm_element.hydrogen]
        if hs:
            modeller.delete(hs)
        return len(hs)

    @classmethod
    def _trim_na_5prime_phosphates(cls, modeller: Modeller) -> int:
        """
        Remove P/OP1/OP2/OP3 (and O1P/O2P) from the FIRST residue of each NA chain.
        This mirrors the TEMPy-ReFF phosphate trimming step and helps avoid NA terminal template mismatches.
        """
        to_delete = []
        for ch in modeller.topology.chains():
            first = None
            for r in ch.residues():
                first = r
                break
            if first is None:
                continue
            if first.name not in cls._NA_CANON:
                continue
            for a in first.atoms():
                n = cls._norm_atom_name(a.name)
                if n in ("P", "OP1", "OP2", "OP3", "O1P", "O2P"):
                    to_delete.append(a)
        if to_delete:
            modeller.delete(to_delete)
        return len(to_delete)

    @staticmethod
    def _rebuild_standard_bonds(top: Topology) -> None:
        # Standard bonds (peptide + NA O3'-P) and disulfides if available
        top.createStandardBonds()
        try:
            top.createDisulfideBonds()
        except Exception:
            pass

    @staticmethod
    def build_openmm_model_from_pdbfixer(fixer, forcefield, openmm_receptor):
        """
        Robust builder for protein + nucleic acids:
          - split chains on breaks (protein and NA)
          - rebuild standard bonds
          - fixer: replace nonstandard, add missing atoms (NO hydrogens here)
          - rebuild bonds again
          - trim NA 5' phosphates (optional but default)
          - delete H and add H LAST using Modeller.addHydrogens(forcefield)
          - create system (with fallback ignoreExternalBonds=True)
        """
        # 1) split chains (protein + NA break logic)
        new_topology, new_positions = ProteinParser.split_chains_on_breaks(fixer)

        # 2) rebuild standard bonds on split topology
        ProteinParser._rebuild_standard_bonds(new_topology)

        modeller = Modeller(new_topology, new_positions)

        # 3) fixer roundtrip (missing atoms, nonstandard residues)
        new_fixer = ProteinParser.model_to_fixer_interchange(modeller)
        new_fixer.findNonstandardResidues()
        new_fixer.replaceNonstandardResidues()
        new_fixer.findMissingResidues()
        new_fixer.findMissingAtoms()
        new_fixer.addMissingAtoms()

        # IMPORTANT: don't add hydrogens in PDBFixer if you later write/read or change bonds
        # new_fixer.addMissingHydrogens(7.4)

        # 4) final modeller from fixer's topology/positions
        final_modeller = Modeller(new_fixer.topology, new_fixer.positions)
        
        # 5) rebuild bonds again (fixer IO/bond inference can vary)
        ProteinParser._rebuild_standard_bonds(final_modeller.topology)

        # 6) trim 5' phosphate atoms at NA chain starts (helps template matching)
        trimmed = ProteinParser._trim_na_5prime_phosphates(final_modeller)
        if trimmed:
            print(f"[NA] Trimmed {trimmed} 5' phosphate atoms from NA chain starts.")
            ProteinParser._rebuild_standard_bonds(final_modeller.topology)

        # 7) delete any H and re-add LAST based on final bond graph
        deleted_h = ProteinParser._delete_all_hydrogens(final_modeller)
        if deleted_h:
            print(f"[H] Deleted {deleted_h} existing hydrogens before re-adding.")
        final_modeller.addHydrogens(forcefield, pH=7.4)

        # 8) build system (fallback for messy termini/external bonds)
        try:
            system = forcefield.createSystem(final_modeller.topology)
        except Exception as e:
            system = forcefield.createSystem(final_modeller.topology, ignoreExternalBonds=True)
            print(f"[FF] createSystem() failed once ({e}); retried with ignoreExternalBonds=True.")

        return final_modeller, system

    # -------------------------
    # Serial-based mapping helpers (kept as-is)
    # -------------------------
    @staticmethod
    def _pdb_residue_serials(pdb_path):
        res_to_serials = {}
        serial_to_res = {}

        with open(pdb_path, "r") as f:
            for line in f:
                rec = line[:6]
                if rec.strip() not in ("ATOM", "HETATM"):
                    continue
                serial = int(line[6:11])
                chain = line[21].strip() or "?"
                resseq = line[22:26].strip()
                icode = line[26].strip()
                resid = f"{resseq}{icode}".strip()

                key = (chain, resid)
                res_to_serials.setdefault(key, []).append(serial)
                serial_to_res[serial] = key

        for k in res_to_serials:
            res_to_serials[k].sort()
        return res_to_serials, serial_to_res

    @staticmethod
    def build_residue_map_via_serials(orig_topology, orig_positions, final_topology, final_positions):
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp_orig:
            PDBFile.writeFile(orig_topology, orig_positions, tmp_orig, keepIds=True)
            tmp_orig.flush()
            orig_res_to_serials, _ = ProteinParser._pdb_residue_serials(tmp_orig.name)

        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp_fin:
            PDBFile.writeFile(final_topology, final_positions, tmp_fin, keepIds=True)
            tmp_fin.flush()
            _, final_serial_to_res = ProteinParser._pdb_residue_serials(tmp_fin.name)

        res_map = {}
        for orig_res, serials in orig_res_to_serials.items():
            rep_serial = serials[0]
            if rep_serial in final_serial_to_res:
                res_map[orig_res] = final_serial_to_res[rep_serial]
            else:
                res_map[orig_res] = orig_res
        return res_map

    @staticmethod
    def map_residue_label_via_serials(orig_chain, orig_resid, res_map, default_to_same=True):
        key = (str(orig_chain).strip(), str(orig_resid).strip())
        if key in res_map:
            return res_map[key]
        if default_to_same:
            return key
        raise KeyError(f"No mapping for {key}")

    @staticmethod
    def build_label_mapping(orig_top: Topology, new_top: Topology):
        """
        Returns:
          atom_map: list of dicts (len = n_atoms)
          res_map : (orig_chain, orig_resid) -> (new_chain, new_resid)
        Assumes atoms are copied in the same order.
        """
        def explode(top):
            rows = []
            for ch in top.chains():
                cid = ch.id or "?"
                for ri, res in enumerate(ch.residues()):
                    resid = res.id
                    rname = res.name
                    for atom in res.atoms():
                        rows.append(
                            {"chain": cid, "resid": resid, "resname": rname, "res_index": ri, "atom": atom.name}
                        )
            return rows

        A = explode(orig_top)
        B = explode(new_top)
        if len(A) != len(B):
            raise ValueError("Atom count changed during split; cannot build 1:1 map.")

        atom_map = []
        res_map = {}
        for a, b in zip(A, B):
            atom_map.append({"orig": a, "new": b})
            res_key = (a["chain"], a["resid"])
            res_val = (b["chain"], b["resid"])
            res_map.setdefault(res_key, res_val)

        return atom_map, res_map

    @staticmethod
    def remap_cov_specs_to_new_labels(cov_specs, res_map):
        remapped = []
        for s in cov_specs:
            key = (s["prot_chain"], s["prot_resnum"])
            if key in res_map:
                new_chain, new_resid = res_map[key]
                s2 = dict(s)
                s2["prot_chain"] = new_chain
                s2["prot_resnum"] = new_resid
                remapped.append(s2)
            else:
                remapped.append(s)
        return remapped

    # -------------------------
    # Chain splitting (protein + nucleic acids)
    # -------------------------
    @staticmethod
    def split_chains_on_breaks(fixer, acceptable_c_n=1.6, acceptable_o3_p=2.2):
        """
        Splits chains on breaks detected by:
          - non-sequential residue numbering
          - protein C(prev)-N(curr) and N(prev)-C(curr) distance heuristic
          - nucleic-acid O3'(prev)-P(curr) distance heuristic
    
        Returns:
          new_topology: openmm.app.Topology
          new_positions: unit.Quantity(list_of_Vec3, unit.nanometer)
        """
    
        def residues_sequential(res1, res2):
            try:
                return int(res2.id) == int(res1.id) + 1
            except Exception:
                return False
    
        def _as_quantity_pos(p):
            # Ensure p is a Quantity(Vec3, length)
            return p if isinstance(p, unit.Quantity) else (p * unit.nanometer)
    
        def get_atom_position(res, atom_name):
            want = atom_name
            alts = {want}
            if "*" in want:
                alts.add(want.replace("*", "'"))
            if "'" in want:
                alts.add(want.replace("'", "*"))
    
            for atom in res.atoms():
                if atom.name in alts:
                    p = original_positions[atom.index]
                    return _as_quantity_pos(p)
            return None
    
        def distance(p1_xyz, p2_xyz):
            # p1_xyz/p2_xyz are plain float arrays (Å)
            dx = p1_xyz[0] - p2_xyz[0]
            dy = p1_xyz[1] - p2_xyz[1]
            dz = p1_xyz[2] - p2_xyz[2]
            return math.sqrt(dx * dx + dy * dy + dz * dz)
    
        new_topology = Topology()
        new_positions_vals = []  # store unitless Vec3 values in *nanometers*
    
        original_topology = fixer.topology
        original_positions = fixer.positions  # typically unit.Quantity(list_of_Vec3, nm)
    
        for chain in original_topology.chains():
            orig_cid = chain.id or "?"
            seg_idx = 1
            new_chain = new_topology.addChain(id=orig_cid)
    
            prev_residue = None
            for res in chain.residues():
                if prev_residue and prev_residue.name not in ProteinParser._WATER:
                    do_split = False
    
                    # (1) numbering break
                    if not residues_sequential(prev_residue, res):
                        do_split = True
                    else:
                        # (2) NA break heuristic: O3'(prev) - P(curr)
                        prev_is_na = ProteinParser._is_nucleic_acid_res(prev_residue)
                        curr_is_na = ProteinParser._is_nucleic_acid_res(res)
    
                        if prev_is_na and curr_is_na:
                            pos_prev_o3_q = get_atom_position(prev_residue, "O3'")
                            pos_curr_p_q = get_atom_position(res, "P")
    
                            if pos_prev_o3_q is not None and pos_curr_p_q is not None:
                                pos_prev_o3 = np.array(pos_prev_o3_q.value_in_unit(unit.angstrom))
                                pos_curr_p = np.array(pos_curr_p_q.value_in_unit(unit.angstrom))
                                if distance(pos_prev_o3, pos_curr_p) > acceptable_o3_p:
                                    do_split = True
                        else:
                            # (3) protein break heuristic: C(prev)-N(curr) and N(prev)-C(curr)
                            pos_prev_C_q = get_atom_position(prev_residue, "C")
                            pos_prev_N_q = get_atom_position(prev_residue, "N")
                            pos_curr_N_q = get_atom_position(res, "N")
                            pos_curr_C_q = get_atom_position(res, "C")
    
                            if (pos_prev_C_q is not None and pos_prev_N_q is not None and
                                    pos_curr_N_q is not None and pos_curr_C_q is not None):
    
                                pos_prev_C = np.array(pos_prev_C_q.value_in_unit(unit.angstrom))
                                pos_prev_N = np.array(pos_prev_N_q.value_in_unit(unit.angstrom))
                                pos_curr_N = np.array(pos_curr_N_q.value_in_unit(unit.angstrom))
                                pos_curr_C = np.array(pos_curr_C_q.value_in_unit(unit.angstrom))
    
                                d_curr_n_prev_c = distance(pos_curr_N, pos_prev_C)
                                d_curr_c_prev_n = distance(pos_curr_C, pos_prev_N)
    
                                if d_curr_n_prev_c > acceptable_c_n and d_curr_c_prev_n > acceptable_c_n:
                                    do_split = True
    
                    if do_split:
                        seg_idx += 1
                        new_chain = new_topology.addChain(id=f"{orig_cid}#{seg_idx}")
    
                new_residue = new_topology.addResidue(res.name, new_chain, id=res.id)
                for atom in res.atoms():
                    new_topology.addAtom(atom.name, atom.element, new_residue)
    
                    p = original_positions[atom.index]
                    p_q = p if isinstance(p, unit.Quantity) else (p * unit.nanometer)
                    # store unitless Vec3 values (in nanometers) so we can wrap once at the end
                    new_positions_vals.append(p_q.value_in_unit(unit.nanometer))
    
                prev_residue = res
    
        new_positions = unit.Quantity(new_positions_vals, unit.nanometer)
        return new_topology, new_positions

    @staticmethod
    def _split_chains_on_breaks(fixer, acceptable_c_n=1.6, acceptable_o3_p=2.2):
        """
        Splits chains on breaks detected by:
          - non-sequential residue numbering
          - protein C(prev)-N(curr) and N(prev)-C(curr) distance heuristic
          - nucleic-acid O3'(prev)-P(curr) distance heuristic

        Keeps original chain IDs:
          - first segment keeps original chain id (e.g. 'A')
          - subsequent segments get suffixed ids (e.g. 'A#2', 'A#3', ...)
        """
        def residues_sequential(res1, res2):
            try:
                return int(res2.id) == int(res1.id) + 1
            except Exception:
                return False

        def get_atom_position(res, atom_name):
            want = atom_name
            alts = {want}
            if "*" in want:
                alts.add(want.replace("*", "'"))
            if "'" in want:
                alts.add(want.replace("'", "*"))
            for atom in res.atoms():
                if atom.name in alts:
                    return original_positions[atom.index]
            return None

        def distance(pos1, pos2):
            dx = pos1[0] - pos2[0]
            dy = pos1[1] - pos2[1]
            dz = pos1[2] - pos2[2]
            return math.sqrt(dx * dx + dy * dy + dz * dz)

        new_topology = Topology()
        new_positions = []

        original_topology = fixer.topology
        original_positions = fixer.positions  # nm

        for chain in original_topology.chains():
            orig_cid = chain.id or "?"
            seg_idx = 1
            new_chain = new_topology.addChain(id=orig_cid)

            prev_residue = None
            for res in chain.residues():
                if prev_residue and prev_residue.name not in ProteinParser._WATER:
                    do_split = False

                    # (1) numbering break
                    if not residues_sequential(prev_residue, res):
                        do_split = True
                    else:
                        # (2) NA break heuristic: O3'(prev) - P(curr)
                        prev_is_na = ProteinParser._is_nucleic_acid_res(prev_residue)
                        curr_is_na = ProteinParser._is_nucleic_acid_res(res)
                        if prev_is_na and curr_is_na:
                            pos_prev_o3_raw = get_atom_position(prev_residue, "O3'")
                            pos_curr_p_raw = get_atom_position(res, "P")
                            if pos_prev_o3_raw is not None and pos_curr_p_raw is not None:
                                pos_prev_o3 = np.array(pos_prev_o3_raw.value_in_unit(unit.angstrom))
                                pos_curr_p = np.array(pos_curr_p_raw.value_in_unit(unit.angstrom))
                                if distance(pos_prev_o3, pos_curr_p) > acceptable_o3_p:
                                    do_split = True
                        else:
                            # (3) protein break heuristic: C(prev)-N(curr) and N(prev)-C(curr)
                            pos_prev_C_raw = get_atom_position(prev_residue, "C")
                            pos_prev_N_raw = get_atom_position(prev_residue, "N")
                            pos_curr_N_raw = get_atom_position(res, "N")
                            pos_curr_C_raw = get_atom_position(res, "C")

                            if (pos_prev_C_raw is not None and pos_prev_N_raw is not None and
                                    pos_curr_N_raw is not None and pos_curr_C_raw is not None):

                                pos_prev_C = np.array(pos_prev_C_raw.value_in_unit(unit.angstrom))
                                pos_prev_N = np.array(pos_prev_N_raw.value_in_unit(unit.angstrom))
                                pos_curr_N = np.array(pos_curr_N_raw.value_in_unit(unit.angstrom))
                                pos_curr_C = np.array(pos_curr_C_raw.value_in_unit(unit.angstrom))

                                d_curr_n_prev_c = distance(pos_curr_N, pos_prev_C)
                                d_curr_c_prev_n = distance(pos_curr_C, pos_prev_N)

                                if d_curr_n_prev_c > acceptable_c_n and d_curr_c_prev_n > acceptable_c_n:
                                    do_split = True

                    if do_split:
                        seg_idx += 1
                        new_chain = new_topology.addChain(id=f"{orig_cid}#{seg_idx}")

                new_residue = new_topology.addResidue(res.name, new_chain, id=res.id)
                for atom in res.atoms():
                    new_topology.addAtom(atom.name, atom.element, new_residue)
                    new_positions.append(original_positions[atom.index])

                prev_residue = res

        return new_topology, new_positions
