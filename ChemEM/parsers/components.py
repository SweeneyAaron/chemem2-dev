# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>
from collections import Counter
from dataclasses import dataclass, field 

@dataclass
class RepCounter:
    protein_res : int = 0
    na_res: int = 0
    nonstd_prot : Counter = field(default_factory = Counter)
    dna : Counter = field(default_factory = Counter)
    rna : Counter = field(default_factory = Counter) 
    na_modified : Counter = field(default_factory = Counter)
    waters : Counter = field(default_factory = Counter) 
    ions : Counter = field(default_factory = Counter) 
    ligands : Counter = field(default_factory = Counter) 
    cofactors : Counter = field(default_factory = Counter) 
    glycans : Counter = field(default_factory = Counter) 
    chains : Counter = field(default_factory = Counter) 
    na_chains : Counter = field(default_factory = Counter)
    
    @property
    def has_waters(self) -> bool:
        return sum(self.waters.values()) > 0

    @property
    def has_ions(self) -> bool:
        return sum(self.ions.values()) > 0

    @property
    def has_ligs(self) -> bool:
        return (sum(self.ligands.values()) + sum(self.cofactors.values())) > 0

    @property
    def has_nonstd(self) -> bool:
        return sum(self.nonstd_prot.values()) > 0

    @property
    def has_na(self) -> bool:
        return self.na_res > 0
    
    
    def print_component_report(self) -> None:
        def show(lbl, x):
            print(f"{lbl:<24} {x}")

        print("=== Structure contents ===")
        show("Protein residues", self.protein_res)
        if self.nonstd_prot:
            show("Non-standard AAs", ", ".join(f"{k}:{v}" for k, v in self.nonstd_prot.most_common()))

        show("NA residues", self.na_res)
        if self.dna:
            show("  DNA names", ", ".join(f"{k}:{v}" for k, v in self.dna.most_common()))
        if self.rna:
            show("  RNA names", ", ".join(f"{k}:{v}" for k, v in self.rna.most_common()))
        if self.na_modified:
            show("  NA modified", ", ".join(f"{k}:{v}" for k, v in self.na_modified.most_common()))
        if self.na_chains:
            show("NA chains", ", ".join(f"{cid}:{n}" for cid, n in self.na_chains.items()))

        show("Waters (res)", sum(self.waters.values()))
        if self.waters:
            show("  Water names", ", ".join(f"{k}:{v}" for k, v in self.waters.most_common()))
        show("Ions (res)", sum(self.ions.values()))
        if self.ions:
            show("  Ion names", ", ".join(f"{k}:{v}" for k, v in self.ions.most_common()))
        show("Cofactors (res)", sum(self.cofactors.values()))
        if self.cofactors:
            show("  Cofactor names", ", ".join(f"{k}:{v}" for k, v in self.cofactors.most_common()))
        show("Glycans (res)", sum(self.glycans.values()))
        if self.glycans:
            show("  Glycan names", ", ".join(f"{k}:{v}" for k, v in self.glycans.most_common()))
        show("Ligands (res)", sum(self.ligands.values()))
        if self.ligands:
            show("  Ligand names", ", ".join(f"{k}:{v}" for k, v in self.ligands.most_common()))
        if self.chains:
            show("Protein chains", ", ".join(f"{cid}:{n}" for cid, n in self.chains.items()))
        print("==========================")



class Components:
    
    
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
    
    
    
    
    @classmethod
    def scan_components(cls, top) -> RepCounter: #make data class
        
        rep = RepCounter()
    
        for ch in top.chains():
            prot_in_chain = 0
            na_in_chain = 0
    
            for res in ch.residues():
                if cls._is_water(res):
                    rep.waters[res.name] += 1
                    continue
    
                if cls._has_backbone(res):
                    rep.protein_res += 1
                    prot_in_chain += 1
                    if res.name not in cls._AA3:
                        rep.nonstd_prot[res.name] += 1
                    continue
    
                if cls._is_nucleic_acid_res(res):
                    rep.na_res += 1
                    na_in_chain += 1
                    if res.name in cls._DNA_RESNAMES:
                        rep.dna[res.name] += 1
                    elif res.name in cls._RNA_RESNAMES:
                        rep.rna[res.name] += 1
                    else:
                        rep.na_modified[res.name] += 1
                    continue
    
                if cls._is_ion(res):
                    rep.ions[res.name] += 1
                    continue
                if cls._is_glycan(res):
                    rep.glycans[res.name] += 1
                    continue
                if cls._is_cofactor(res):
                    rep.cofactors[res.name] += 1
                    continue
    
                rep.ligands[res.name] += 1
    
            if prot_in_chain:
                rep.chains[ch.id or "?"] += prot_in_chain
            if na_in_chain:
                rep.na_chains[ch.id or "?"] += na_in_chain
                
        return rep
    
    
    #-----solvent detection
    @classmethod
    def _is_water(cls, res) -> bool:
        if res.name in cls._WATER:
            return True
        atoms = list(res.atoms())
        elems = {a.element.symbol if a.element else None for a in atoms}
        return elems.issubset({"O", "H", None}) and len(atoms) <= 4
    
    #-----protein detection
    @staticmethod
    def _has_backbone(res) -> bool:
        names = {a.name for a in res.atoms()}
        return {"N", "CA", "C"}.issubset(names)
    
    #-----nuclic acid detection
    
    @classmethod
    def _is_nucleic_acid_res(cls, res) -> bool:
        if res.name in cls._NA_CANON:
            return True
        return cls._has_nucleic_backbone(res)
    
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
    
    #----het groups
    
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
    
    
    
    