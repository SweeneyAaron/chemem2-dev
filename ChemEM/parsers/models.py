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

from ChemEM.tools.biomolecule import sse_groups_from_parmed


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

    def get_residue_mapping(self, chain, res_num):
        key = (str(chain).strip(), str(res_num).strip())
        mapped = self.res_map.get(key, None)
        if mapped is None:
            return None

        for res in self.complex_structure.residues:
            if str(res.chain) == mapped[0] and str(res.number) == mapped[1]:
                return res
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
