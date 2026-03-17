# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

from openmm import unit
from rdkit.Geometry import Point3D
import numpy as np
import os
from pathlib import PurePath
from ChemEM.tools.biomolecule import sse_groups_from_parmed
from .writers import write_to_sdf
from collections import Counter
from dataclasses import dataclass
from collections import UserList
from typing import Optional, Iterable


def _safe_cast(obj, new_type):
    try:
        new_obj = new_type(obj)
        return new_obj 
    except ValueError:
        return None

@dataclass
class AtomSpec:
    structure: object 
    atom_idx: int 
    
    
    def _get_atom(self):
        return self.structure.complex_structure.atoms[self.atom_idx]
    def get_point(self):
        atom =  self._get_atom()
        return np.array([atom.xx, atom.xy, atom.xz], dtype=float) 
    def get_element(self):
        atom =  self._get_atom()
        return atom.element_name.upper()

class Protein:
    def __init__(self,
                 file,
                 complex_system,
                 complex_structure,
                 forcefield_params=None,
                 residue_map=None):
        self.filename = file
        self.included_forcefields = forcefield_params
        self.complex_system = complex_system
        self.complex_structure = complex_structure
        self.sse_groups, self.sse_codes = sse_groups_from_parmed(complex_structure)
        self.res_map = residue_map or {}
        self.delaunay = None

    def get_residue_mapping(self, chain, res_num):
        key = (str(chain).strip(), str(res_num).strip())
        mapped = self.res_map.get(key, None)
        if mapped is None:
            return None

        for res in self.complex_structure.residues:
            if str(res.chain) == mapped[0] and str(res.number) == mapped[1]:
                return res
        return None
    
    def get_atom_idx_from_spec(self, spec):
        spec = spec.split(':')
        
        if len(spec) != 4:
            raise RuntimeError(f"[ERROR] IonFixer protein atom spec format unknown {spec}")
        
        residue_idx = _safe_cast(spec[2], int)
        
        if residue_idx is None:
            raise ValueError(f"[ERROR] IonFixer Can't convert {spec[1]} to type int()")
        
        chain, res_name, _ , atom_name = spec 
        
        for res in self.complex_structure.topology.residues():
            if not all([res.chain.id == chain,  res.name == res_name, int(res.id) == residue_idx]):
                continue
            
            atoms = [atom for atom in res.atoms() if atom.name == spec[3]]
            
            if len(atoms) != 1:
                raise RuntimeError(f"[Error] Protein multiple atoms with same spec {spec}")
            
            return AtomSpec(structure=self, atom_idx=atoms[0].index)
            
            
        return None


class Ligand:
    def __init__(self,
                 ligand_input,
                 rd_mol,
                 complex_system,
                 complex_structure,
                 atom_types,
                 ring_types,
                 ring_indices,
                 ligand_charges=None):
        self.input = ligand_input
        self.mol = rd_mol
        self.complex_system = complex_system
        self.complex_structure = complex_structure
        self.atom_types = atom_types
        self.ring_types = ring_types
        self.ring_indices = ring_indices
        
        self.docked = []
        self.mmgbsa_scores = []

        self.ligand_charge = []
        self.ligand_charge_idx = []
        ligand_charges = ligand_charges or []
        for idx, charge in ligand_charges:
            self.ligand_charge_idx.append(idx)
            self.ligand_charge.append(charge)
        
        self.get_atom_names()
    
    @property 
    def ligand_id(self):
        return self.complex_structure.residues[0].name 
    @property 
    def ligand_int(self):
        _id = self.complex_structure.residues[0].name 
        if _id.startswith("LIG"):
            return int(_id.strip("LIG"))
    
    @property
    def identifier(self):
        
        if os.path.exists(self.input):
            #is a path
            path = PurePath(str(self.input))
            return path.name or path.parts[-1] if path.parts else str(self.input)
        else:
            #is a smiles string
            return str(self.input)
    
    def set_positions(self, coords: np.ndarray, conf_id: int = 0) -> None:
        """
        coords shape (n_atoms,3) in Å
        """
        if coords.shape != (self.mol.GetNumAtoms(), 3):
            raise ValueError(
                f"coords shape {coords.shape} does not match {self.mol.GetNumAtoms()} atoms in ligand."
            )

        conf = self.mol.GetConformer(conf_id)
        for idx, (x, y, z) in enumerate(coords.astype(float)):
            conf.SetAtomPosition(idx, Point3D(x, y, z))

        self.complex_structure.positions = (coords * unit.angstrom)
    
    def write_sdf(self, file):
        write_to_sdf(self.mol, file)
    
    def get_atom_positons(self):
        positions = []
        #check it has conformers 
        if not self.mol.GetNumConformers():
            return np.array(positions)
        
        for conf in self.mol.GetConformers():
            positions.append(conf.GetPositions())
        
        return np.array(positions)
    
    def get_positions(self):
        if not self.mol.GetNumConformers():
            return np.array([])
        
        conf = self.mol.GetConformer()
        return conf.GetPositions()
    
    def get_atom_names(self):
        atom_counts = Counter()
        self.atom_names = []
        for atom in self.mol.GetAtoms():
            atom_counts[atom.GetSymbol()] += 1
            self.atom_names.append( f'{atom.GetSymbol().upper()}{str(atom_counts[atom.GetSymbol()])}' )
    
    def get_atom_idx_from_name(self, name):
        if name in self.atom_names:
            return self.atom_names.index(name)
        return None
    
    
    def get_heavy_atom_positions(self):
        positions = []
        #check it has conformers 
        n_heavy = self.mol.GetNumHeavyAtoms()
        if not self.mol.GetNumConformers():
            return np.array(positions)
        
        for conf in self.mol.GetConformers():
            positions.append(conf.GetPositions()[:n_heavy])
        
        return np.array(positions)



class LigandList(UserList):
    def __init__(self, ligands: Optional[Iterable[Ligand]] = None):
        super().__init__(list(ligands) if ligands is not None else [])

    def get_atom_idx_from_spec(self, spec):
        #move this into Ligand need a new style ligand list to hold the ligands that has various methods
        
        
        spec = spec.split(":")
        if len(spec) != 3:
            raise RuntimeError(f"[ERROR] LigandList Ligand atom spec format unknown {spec}")
        
        ligand_idx = _safe_cast(spec[1], int)
        if ligand_idx is None:
            raise ValueError(f"[ERROR] LigandList Can't convert {spec[1]} to type int()")
        
        if len(self) <= ligand_idx:
            raise RuntimeError(f'[ERROR] LigandList no ligand with index {ligand_idx}')
        
        ligand = self[ligand_idx]
        
        atom_name = spec[2]
        atom_idx = ligand.get_atom_idx_from_name(spec[2])
        if atom_idx is None:
            raise RuntimeError(f'[ERROR] LigandList no ligand atom with name {atom_name}')
        
        
        return AtomSpec(structure = ligand,
                        atom_idx = atom_idx)

    def append(self, item):
        self._validate(item)
        super().append(item)

    def extend(self, other):
        other = list(other)
        for item in other:
            self._validate(item)
        super().extend(other)

    def insert(self, i, item):
        self._validate(item)
        super().insert(i, item)

    @staticmethod
    def _validate(item):
        if not isinstance(item, Ligand):
            raise TypeError(f"LigandList only accepts Ligand objects, got {type(item).__name__}")
    
