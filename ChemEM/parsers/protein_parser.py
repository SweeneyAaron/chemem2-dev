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
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import parmed

from openmm import app, unit
from openmm.app import PDBFile, Modeller, Topology
from pdbfixer import PDBFixer

from .models import Protein
from ChemEM.tools.biomolecule import (
    ensure_water_geometry_types,
    build_residue_map_by_positions,
)


class ProteinParser:
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
        elems = {a.element.symbol if a.element else None for a in res.atoms()}
        return elems.issubset({"O", "H", None}) and sum(1 for _ in res.atoms()) <= 4

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

    @classmethod
    def scan_components(cls, top):
        rep = {
            "protein_res": 0,
            "nonstd_prot": Counter(),
            "waters": Counter(),
            "ions": Counter(),
            "ligands": Counter(),
            "cofactors": Counter(),
            "glycans": Counter(),
            "chains": Counter(),
        }

        for ch in top.chains():
            prot_in_chain = 0
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
                if cls._is_ion(res):
                    rep["ions"][res.name] += 1
                    continue
                if res.name in cls._GLYCAN_CODES:
                    rep["glycans"][res.name] += 1
                    continue
                if res.name in cls._COFACTORS_COMMON:
                    rep["cofactors"][res.name] += 1
                    continue
                rep["ligands"][res.name] += 1
            if prot_in_chain:
                rep["chains"][ch.id or "?"] += prot_in_chain

        rep["has_waters"] = sum(rep["waters"].values()) > 0
        rep["has_ions"] = sum(rep["ions"].values()) > 0
        rep["has_ligs"] = sum(rep["ligands"].values()) + sum(rep["cofactors"].values()) > 0
        rep["has_nonstd"] = sum(rep["nonstd_prot"].values()) > 0
        return rep

    @staticmethod
    def print_component_report(rep) -> None:
        def show(lbl, x):
            print(f"{lbl:<24} {x}")
        print("=== Structure contents ===")
        show("Protein residues", rep["protein_res"])
        if rep["nonstd_prot"]:
            show("Non-standard AAs", ", ".join(f"{k}:{v}" for k, v in rep["nonstd_prot"].most_common()))
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
        notes: List[str] = []
        solvent_mode = "explicit" if rep["has_waters"] else "implicit"

        protein_ff = prefer_protein
        implicit_xml = "implicit/gbn2.xml"

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

        # fallback
        if rep["has_ions"]:
            ff, errs = cls._ff_loads([protein_ff, prefer_water])
            if ff:
                notes.append("GBn2 unavailable; using protein+water XML for ion parameters (no bulk waters added).")
                return ff, notes, "vacuum"

        ff, errs = cls._ff_loads([protein_ff])
        if ff:
            notes.append("GBn2 unavailable; running in vacuum (no implicit solvent).")
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
            rep, prefer_protein, prefer_water, try_implicit=True
        )
        return ff, rep, notes, mode

    # -------------------------
    # Main protein loader
    # -------------------------
    @staticmethod
    def load_protein_structure(protein_file, forcefield, prefer_water="amber14/tip3p.xml"):
        warnings = []

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

        # Attempt 1: direct load
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

        # Attempt 2: PDBFixer + chain truncation/splitting
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
    # OpenMM ForceField builder
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
    def build_openmm_model_from_pdbfixer(fixer, forcefield, openmm_receptor):
        
        new_topology, new_positions = ProteinParser.split_chains_on_breaks(fixer)
        modeller = Modeller(new_topology, fixer.positions)

        new_fixer = ProteinParser.model_to_fixer_interchange(modeller)

        new_fixer.findNonstandardResidues()
        new_fixer.replaceNonstandardResidues()
        new_fixer.findMissingResidues()
        new_fixer.findMissingAtoms()
        new_fixer.addMissingAtoms()
        new_fixer.addMissingHydrogens(7.4)

        new_modeller = ProteinParser.fixer_to_model_interchange(new_fixer)

        system = forcefield.createSystem(new_modeller.topology)
        return new_modeller, system

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
    # Serial-based mapping helpers
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

    # These were accidentally nested in your original snippet; kept here as proper methods.
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
    # Chain splitting
    # -------------------------
    @staticmethod
    def split_chains_on_breaks(fixer, acceptable_c_n=1.6):
        """
        Splits chains on breaks detected by non-sequential residue numbering
        or when both the C(prev)-N(curr) and N(prev)-C(curr) distances exceed acceptable limits.

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
            for atom in res.atoms():
                if atom.name == atom_name:
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
                    if not residues_sequential(prev_residue, res):
                        seg_idx += 1
                        new_chain = new_topology.addChain(id=f"{orig_cid}#{seg_idx}")
                    else:
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
                                seg_idx += 1
                                new_chain = new_topology.addChain(id=f"{orig_cid}#{seg_idx}")

                new_residue = new_topology.addResidue(res.name, new_chain, id=res.id)
                for atom in res.atoms():
                    new_topology.addAtom(atom.name, atom.element, new_residue)
                    new_positions.append(original_positions[atom.index])

                prev_residue = res

        return new_topology, new_positions

