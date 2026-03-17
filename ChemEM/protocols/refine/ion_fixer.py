# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from itertools import permutations
import numpy as np

from .refine_utils import create_structure_subset, generate_initial_ion_position
from ChemEM.protocols.core.forces import ForceBuilder
from ChemEM.protocols.core.core_utils import all_pairwise_distances_leq
from ChemEM.parsers.parametised_ions import (
    create_parameterized_ion_structure,
    propose_dummy_water_oxygen_positions,
    coord_geom_to_int,
    create_parametrised_tip3p_water,
)
from ChemEM.parsers.parse_forcefield import ff_load
from openmm import app, unit
from openmm import app, unit, LangevinMiddleIntegrator, Platform

import os 
import copy
import uuid
from datetime import datetime
from rdkit import Chem
from rdkit.Geometry import Point3D

from collections import deque
from rdkit.Chem import rdMolTransforms
from scipy.ndimage import gaussian_filter


class IonFixer:
    def __init__(self, system):
        self.system = system
        self.spec_atoms = None
        self.positions = None
        self.warning_distance = 12.0
        self.select_residues_within = 12.0

        self.openmm_system = None
        self.ion_atom = None
        self.coord_atoms_unordered = None
        self.coord_atoms_ordered = None
        self.coord_atom_indices = None
        self.coord_assignment = None
        self.ion_restraint_forces = None

        # Track what we add at merge time
        self.added_ion_resnum = None
        self.added_ion_atom_idx = None
        self.added_water_resnums = []
        self.added_water_oxygen_indices = []
        
        self.pin_atom_indices = None
        self.pin_atoms = None
        self.pin_force = None
        
        self.local_density_map = None
        self.map_atoms = None
        self.map_atom_indices = None
        self.map_force = None
        
        self.preopt_local_density_map = None
        self.preopt_use_map_score = False
        self.preopt_map_resolution = None
        self.preopt_map_sigma_coeff = 0.356
        self.preopt_map_weight_mode = "mass"
        self.preopt_map_n_bins = 20
        self.preopt_map_normalized = False
        self.preopt_map_zscore = False
        self.preopt_map_normalise_model = True
                
    def get_spec_atoms(self):
        self.spec_atoms = []
        self.exclude_atom_specs = []
        if not len(self.system.options.atom_specs):
            raise RuntimeError("[ERROR] IonFixer requires at least one atom-spec argument")

        for spec in self.system.options.atom_specs:
            if spec.startswith("LIG"):
                self.spec_atoms.append(self.system.ligand.get_atom_idx_from_spec(spec))
            else:
                self.spec_atoms.append(self.system.protein.get_atom_idx_from_spec(spec))
        
        for spec in self.system.options.exclude_specs:
            if spec.startswith("LIG"):
                self.exclude_atom_specs.append(self.system.ligand.get_atom_idx_from_spec(spec))
            else:
                self.exclude_atom_specs.append(self.system.protein.get_atom_idx_from_spec(spec))
            

    def create_complex_structure(self):
        self._spec_atoms_exist()
        self.validate_spec_atom_distances()

        complex_structure = self.system.protein.complex_structure
        ligand_structures = [lig.complex_structure for lig in self.system.ligand]

        if ligand_structures:
            complex_structure = self.system.protein.complex_structure + ligand_structures[0]
            for lig_struct in ligand_structures[1:]:
                complex_structure += lig_struct

        self.complex_structure = complex_structure
        self.curr_resnum = len(complex_structure.residues)

    def get_residue_selection(self):
        self._spec_atoms_exist()
        self._complex_structure_exists()

        cutoff = float(self.select_residues_within)
        spec_points = np.asarray([atom.get_point() for atom in self.spec_atoms], dtype=float)

        selected_residues = []
        selected_keys = set()

        for res in self.complex_structure.residues:
            atoms = list(res.atoms)
            if not atoms:
                continue

            res_xyz = np.array([[atom.xx, atom.xy, atom.xz] for atom in atoms], dtype=float)
            diffs = res_xyz[:, None, :] - spec_points[None, :, :]
            d2 = np.sum(diffs * diffs, axis=-1)

            if np.any(d2 <= cutoff * cutoff):
                key = (str(res.chain), int(res.number), res.name)
                if key not in selected_keys:
                    selected_keys.add(key)
                    selected_residues.append(res)

        self.selected_residues = selected_residues
        self.selected_structure = create_structure_subset(
            self.complex_structure,
            selected_residues,
        )

    def get_initial_position(self):
        spec_xyz = np.asarray([atom.get_point() for atom in self.spec_atoms], dtype=float)

        obstacle_xyz = []
        obstacle_elements = []

        for atom in self.selected_structure.atoms:
            elem = atom.element_name.upper()
            if elem == "H":
                continue
            obstacle_xyz.append([atom.xx, atom.xy, atom.xz])
            obstacle_elements.append(elem)

        obstacle_xyz = np.asarray(obstacle_xyz, dtype=float)

        initial_pos = generate_initial_ion_position(
            spec_xyz=spec_xyz,
            obstacle_xyz=obstacle_xyz,
            obstacle_elements=obstacle_elements,
            ion_name=self.system.options.ion_type,
            target_dists=None,
        )

        self.initial_pos = initial_pos

    def get_paramitised_ion(self):
        forcefield = ff_load(self.system.options.ion_forcefield)
        if forcefield is None:
            raise RuntimeError(f"[ERROR] Can't load Forcefield {self.system.options.ion_forcefield}")

        ion_structure = create_parameterized_ion_structure(
            ion_name=self.system.options.ion_type,
            coordination_number=self.coordination_number,
            forcefield=forcefield,
            position=self.initial_pos,
            residue_id=str(self.curr_resnum),
            chain_id="Z",
        )
        self.ion_resnum = self.curr_resnum
        self.curr_resnum += 1
        self.ion_structure = ion_structure

    def get_paramitised_waters(self):
        ion_to_water_O_dist = 2.1
        donor_xyz = np.asarray([a.get_point() for a in self.spec_atoms], dtype=float)

        obstacle_xyz = []
        for a in self.selected_structure.atoms:
            if a.element_name.upper() == "H":
                continue
            obstacle_xyz.append([a.xx, a.xy, a.xz])

        solutions = propose_dummy_water_oxygen_positions(
            ion_xyz=self.initial_pos,
            donor_xyz=donor_xyz,
            coordination_geometry=self.system.options.coordination_geometry,
            ion_to_water_O_dist=ion_to_water_O_dist,
            obstacle_xyz=obstacle_xyz,
            min_obstacle_dist=1.5,
            max_solutions=5,
        )

        if not solutions:
            raise RuntimeError("[ERROR] Could not place dummy waters without clashes.")

        best = solutions[0]
        water_O_xyz = best["water_O_xyz"]
        ion_xyz = np.asarray(self.initial_pos, float)

        waters = []
        self.water_res_nums = []

        for o_xyz in water_O_xyz:
            w = create_parametrised_tip3p_water(
                o_xyz=o_xyz,
                ion_xyz=ion_xyz,
                residue_name="HOH",
                chain_id="W",
                resnum=self.curr_resnum,
                water_ff_xml=self.system.options.ion_forcefield,
            )
            waters.append(w)
            self.water_res_nums.append(self.curr_resnum)
            self.curr_resnum += 1

        self.waters = waters

    def merge_system(self):
        self._selected_structure_exists()
        self._ion_and_waters_exist()

        self.added_ion_resnum = None
        self.added_ion_atom_idx = None
        self.added_water_resnums = []
        self.added_water_oxygen_indices = []

        # Add ion and record exactly what got appended
        self.selected_structure += self.ion_structure
        ion_res = self.selected_structure.residues[-1]
        self.added_ion_resnum = ion_res.number

        ion_atoms = list(ion_res.atoms)
        if len(ion_atoms) != 1:
            raise RuntimeError(
                f"[ERROR] Expected exactly one ion atom in merged ion residue, found {len(ion_atoms)}"
            )
        self.added_ion_atom_idx = int(ion_atoms[0].idx)

        # Add dummy waters and record each oxygen directly
        for w in self.waters:
            self.selected_structure += w
            water_res = self.selected_structure.residues[-1]
            self.added_water_resnums.append(water_res.number)

            atoms = list(water_res.atoms)
            oxygens = [a for a in atoms if str(a.element_name).upper() == "O"]
            if not oxygens:
                oxygens = [a for a in atoms if str(a.name).upper().startswith("O")]

            if len(oxygens) != 1:
                raise RuntimeError(
                    f"[ERROR] Expected exactly one water oxygen in merged water residue "
                    f"{water_res.number}, found {len(oxygens)}"
                )

            self.added_water_oxygen_indices.append(int(oxygens[0].idx))

    def build_system(self):
        bad_bonds = debug_missing_bond_params(self.selected_structure)
        if bad_bonds:
            print("Bad bonds found:")
            for row in bad_bonds:
                print(row)
            raise RuntimeError("selected_structure contains bonds with missing parameters")

        self.openmm_system = self.selected_structure.createSystem(
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            rigidWater=True,
        )

        return self.openmm_system

    def identify_ion_angle_restraint_atoms(self, tol=1e-3):
        """
        Identify and order the coordinating atoms so the ordering matches the
        canonical site ordering used by ForceBuilder._coordination_template_dirs().

        Returns
        -------
        dict
            {
                "ion_atom": ParmEd atom,
                "coord_atoms_unordered": list[ParmEd atom],
                "coord_atoms_ordered": list[ParmEd atom],
                "coord_atom_indices": list[int],
                "assignment": list[int],
            }
        """
        self._spec_atoms_exist()
        self._selected_structure_exists()
        self._ion_and_waters_exist()

        ion_atom = self._find_ion_atom_in_selected_structure()

        donor_atoms = [
            find_atom_from_spec_by_coord_and_element(spec_atom, self.selected_structure, tol=tol)
            for spec_atom in self.spec_atoms
        ]
        water_o_atoms = self._find_dummy_water_oxygen_atoms()

        coord_atoms = donor_atoms + water_o_atoms
        if len(coord_atoms) != self.coordination_number:
            raise RuntimeError(
                f"[ERROR] Expected {self.coordination_number} coordinating atoms "
                f"({len(donor_atoms)} donors + {len(water_o_atoms)} waters) but found {len(coord_atoms)}."
            )

        ion_xyz = np.asarray([ion_atom.xx, ion_atom.xy, ion_atom.xz], dtype=float)
        coord_xyz = np.asarray([[a.xx, a.xy, a.xz] for a in coord_atoms], dtype=float)
        vecs = coord_xyz - ion_xyz[None, :]
        norms = np.linalg.norm(vecs, axis=1)
        if np.any(norms < 1e-8):
            raise RuntimeError("[ERROR] Found a zero-length ion-to-coordinator vector.")
        actual_dirs = vecs / norms[:, None]

        template_dirs = ForceBuilder._coordination_template_dirs(
            self.system.options.coordination_geometry
        )
        if template_dirs.shape[0] != len(coord_atoms):
            raise RuntimeError(
                f"[ERROR] Coordination geometry {self.system.options.coordination_geometry!r} "
                f"expects {template_dirs.shape[0]} sites but found {len(coord_atoms)} atoms."
            )

        # CN is small (2-7), so exhaustive assignment is fine.
        best_perm = None
        best_score = -np.inf
        idxs = range(len(coord_atoms))
        for perm in permutations(idxs):
            score = 0.0
            for site_idx, atom_idx in enumerate(perm):
                score += float(np.dot(actual_dirs[atom_idx], template_dirs[site_idx]))
            if score > best_score:
                best_score = score
                best_perm = perm

        ordered_atoms = [coord_atoms[i] for i in best_perm]
        ordered_indices = [int(a.idx) for a in ordered_atoms]

        self.ion_atom = ion_atom
        self.coord_atoms_unordered = coord_atoms
        self.coord_atoms_ordered = ordered_atoms
        self.coord_atom_indices = ordered_indices
        self.coord_assignment = list(best_perm)

        return {
            "ion_atom": ion_atom,
            "coord_atoms_unordered": coord_atoms,
            "coord_atoms_ordered": ordered_atoms,
            "coord_atom_indices": ordered_indices,
            "assignment": list(best_perm),
        }

    def apply_ion_coordination_restraints(
        self,
        target_dist_A=None,
        flat_bottom_A=0.2,
        include_angles=True,
        k_dist=1000.0,
        k_ang=200.0,
        k_dist_name="k_ion_dist",
        k_ang_name="k_ion_ang",
        force_group_dist=20,
        force_group_ang=21,
    ):
        """
        Add ion distance and optional angle restraints directly to the built
        OpenMM System.
        """
        self._openmm_system_exists()

        if self.coord_atom_indices is None or self.ion_atom is None:
            self.identify_ion_angle_restraint_atoms()

        if target_dist_A is None:
            target_dist_A = self._current_coordination_distances_A()

        restraints = ForceBuilder.create_ion_coordination_restraints(
            ion_idx=int(self.ion_atom.idx),
            coord_atom_indices=self.coord_atom_indices,
            target_dist_A=target_dist_A,
            flat_bottom_A=float(flat_bottom_A),
            include_angles=bool(include_angles),
            geometry=self.system.options.coordination_geometry if include_angles else None,
            k_dist_name=k_dist_name,
            k_ang_name=k_ang_name,
            force_group_dist=int(force_group_dist),
            force_group_ang=int(force_group_ang),
        )

        dist_force = restraints["distance_force"]
        self._set_global_parameter_default(
            dist_force,
            k_dist_name,
            float(k_dist) * unit.kilojoule_per_mole / unit.nanometer**2,
        )
        self.openmm_system.addForce(dist_force)

        ang_force = restraints.get("angle_force")
        if ang_force is not None:
            self._set_global_parameter_default(
                ang_force,
                k_ang_name,
                float(k_ang) * unit.kilojoule_per_mole / unit.radian**2,
            )
            self.openmm_system.addForce(ang_force)

        self.ion_restraint_forces = restraints
        return restraints

    def add_atom_restraints(self, *args, **kwargs):
        return self.apply_ion_coordination_restraints(*args, **kwargs)

    def get_coordination_number(self):
        self.coordination_number = coord_geom_to_int(self.system.options.coordination_geometry)

    def validate_spec_atom_distances(self):
        self._spec_atoms_exist()
        positions = []
        for atom in self.spec_atoms:
            positions.append(atom.get_point())
        self.positions = positions
        if not all_pairwise_distances_leq(positions, self.warning_distance):
            print(
                "[WARNING] atom-spec distances are greater then 12.0 Å this may causse issues fixing ions"
            )
    
    
    def identify_atoms_to_pin(self, tol=1e-3):
        """
        Pin:
          - all other protein and nucleotide heavy atoms
          - only backbone heavy atoms for donor residues
          - any excluded_spec atoms explicitly, including ligand atoms
        """
        self._spec_atoms_exist()
        self._selected_structure_exists()
    
        donor_atoms = [
            find_atom_from_spec_by_coord_and_element(spec_atom, self.selected_structure, tol=tol)
            for spec_atom in self.spec_atoms
        ]
        donor_residue_keys = {self._residue_key(atom.residue) for atom in donor_atoms}
    
        excluded_spec_atoms = self._find_excluded_spec_atoms_in_selected_structure(tol=tol)
        excluded_spec_idx_set = {int(a.idx) for a in excluded_spec_atoms}
    
        added_water_resnums = {int(x) for x in self.added_water_resnums}
    
        pin_atoms = []
        seen = set()
    
        for res in self.selected_structure.residues:
            resnum = int(res.number)
    
            # Skip added ion and dummy waters
            if self.added_ion_resnum is not None and resnum == int(self.added_ion_resnum):
                continue
            if resnum in added_water_resnums:
                continue
    
            resname = str(res.name).upper()
    
            is_supported_polymer = (
                self._is_protein_residue_name(resname)
                or self._is_nucleotide_residue_name(resname)
            )
            is_donor_residue = self._residue_key(res) in donor_residue_keys
    
            for atom in res.atoms:
                if not self._is_heavy_atom(atom):
                    continue
    
                idx = int(atom.idx)
    
                # Always pin explicitly excluded spec atoms, even if they are ligand atoms
                if idx in excluded_spec_idx_set:
                    if idx not in seen:
                        seen.add(idx)
                        pin_atoms.append(atom)
                    continue
    
                # Otherwise keep the original polymer-only logic
                if not is_supported_polymer:
                    continue
    
                if is_donor_residue and not self._is_backbone_atom(res, atom):
                    continue
    
                if idx not in seen:
                    seen.add(idx)
                    pin_atoms.append(atom)
    
        self.pin_atoms = pin_atoms
        self.pin_atom_indices = [int(a.idx) for a in pin_atoms]
    
        return {
            "pin_atoms": pin_atoms,
            "pin_atom_indices": self.pin_atom_indices,
            "excluded_spec_pin_atoms": excluded_spec_atoms,
            "excluded_spec_pin_atom_indices": [int(a.idx) for a in excluded_spec_atoms],
        }
    
    def __identify_atoms_to_pin(self, tol=1e-3):
        """
        Pin:
          - all other protein and nucleotide heavy atoms
          - only backbone heavy atoms for donor residues
    
        Donor residues are residues containing any atom-spec donor atom.
        Added ion and dummy waters are excluded.
        """
        self._spec_atoms_exist()
        self._selected_structure_exists()
    
        donor_atoms = [
            find_atom_from_spec_by_coord_and_element(spec_atom, self.selected_structure, tol=tol)
            for spec_atom in self.spec_atoms
        ]
        donor_residue_keys = {self._residue_key(atom.residue) for atom in donor_atoms}
    
        added_water_resnums = {int(x) for x in self.added_water_resnums}
    
        pin_atoms = []
        seen = set()
    
        for res in self.selected_structure.residues:
            resnum = int(res.number)
    
            # Skip added ion and dummy waters
            if self.added_ion_resnum is not None and resnum == int(self.added_ion_resnum):
                continue
            if resnum in added_water_resnums:
                continue
    
            resname = str(res.name).upper()
    
            # Only protein / nucleotide residues
            if not (self._is_protein_residue_name(resname) or self._is_nucleotide_residue_name(resname)):
                continue
    
            is_donor_residue = self._residue_key(res) in donor_residue_keys
    
            for atom in res.atoms:
                if not self._is_heavy_atom(atom):
                    continue
    
                if is_donor_residue:
                    # donor residues: only backbone heavy atoms
                    if not self._is_backbone_atom(res, atom):
                        continue
    
                idx = int(atom.idx)
                if idx not in seen:
                    seen.add(idx)
                    pin_atoms.append(atom)
    
        self.pin_atoms = pin_atoms
        self.pin_atom_indices = [int(a.idx) for a in pin_atoms]
    
        return {
            "pin_atoms": pin_atoms,
            "pin_atom_indices": self.pin_atom_indices,
        }
    
    def _find_excluded_spec_atoms_in_selected_structure(self, tol=1e-3):
        """
        Resolve self.exclude_atom_specs onto atoms in self.selected_structure.
        These atom-spec objects are the same type as self.spec_atoms.
        """
        self._selected_structure_exists()
    
        excluded_atoms = []
        seen = set()
    
        for spec_atom in getattr(self, "exclude_atom_specs", []):
            atom = find_atom_from_spec_by_coord_and_element(
                spec_atom,
                self.selected_structure,
                tol=tol,
            )
            idx = int(atom.idx)
            if idx not in seen:
                seen.add(idx)
                excluded_atoms.append(atom)
    
        return excluded_atoms
    
    def apply_pin_restraints(
        self,
        k_pin=1000.0,
        k_pin_name="k_pin",
        force_group=22,
    ):
        """
        Apply positional pin restraints using ForceBuilder.create_positional_pin().
        """
        self._openmm_system_exists()
    
        if self.pin_atom_indices is None:
            self.identify_atoms_to_pin()
    
        if not self.pin_atom_indices:
            raise RuntimeError("[ERROR] No atoms identified for pin restraints.")
    
        # ForceBuilder.create_positional_pin expects ref_positions_nm to be indexed
        # by the GLOBAL atom indices it receives. So build a full atom-indexed array.
        max_idx = max(int(a.idx) for a in self.selected_structure.atoms)
        ref_positions_nm = np.zeros((max_idx + 1, 3), dtype=float)
    
        for atom in self.pin_atoms:
            ref_positions_nm[int(atom.idx)] = np.array([atom.xx, atom.xy, atom.xz], dtype=float) * 0.1
    
        pin_force = ForceBuilder.create_positional_pin(
            atom_indices=self.pin_atom_indices,
            ref_positions_nm=ref_positions_nm,
            k_name=k_pin_name,
        )
        pin_force.setForceGroup(int(force_group))
    
        self._maybe_set_global_parameter_default(
            pin_force,
            k_pin_name,
            float(k_pin) * unit.kilojoule_per_mole / unit.nanometer**2,
        )
    
        self.openmm_system.addForce(pin_force)
        self.pin_force = pin_force
        return pin_force
    
    
    @staticmethod
    def _residue_key(res):
        return (str(getattr(res, "chain", "")), int(res.number), str(res.name).upper())
    
    
    @staticmethod
    def _is_heavy_atom(atom):
        elem = str(getattr(atom, "element_name", "")).upper()
        if elem:
            return elem != "H"
        return not str(atom.name).upper().startswith("H")
    
    
    @staticmethod
    def _is_protein_residue_name(resname):
        protein_names = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
            "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "MET",
            "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
            "ASH", "GLH", "CYX", "CYM", "LYN",
            "NALA", "NARG", "NASN", "NASP", "NCYS", "NGLN", "NGLU", "NGLY",
            "NHIS", "NILE", "NLEU", "NLYS", "NMET", "NPHE", "NPRO", "NSER",
            "NTHR", "NTRP", "NTYR", "NVAL",
            "CALA", "CARG", "CASN", "CASP", "CCYS", "CGLN", "CGLU", "CGLY",
            "CHIS", "CILE", "CLEU", "CLYS", "CMET", "CPHE", "CPRO", "CSER",
            "CTHR", "CTRP", "CTYR", "CVAL",
        }
        return resname in protein_names
    
    
    @staticmethod
    def _is_nucleotide_residue_name(resname):
        nucleotide_names = {
            "A", "C", "G", "U", "I",
            "DA", "DC", "DG", "DT", "DI",
            "ADE", "CYT", "GUA", "URA", "THY",
            "RA", "RC", "RG", "RU",
        }
        return resname in nucleotide_names
    
    
    @staticmethod
    def _is_backbone_atom(res, atom):
        """
        Protein backbone:
          N, CA, C, O, OXT
    
        Nucleotide backbone / sugar-phosphate:
          P, OP1, OP2, OP3, O5', C5', C4', O4', C3', O3', C2', O2', C1'
        """
        resname = str(res.name).upper()
        aname = str(atom.name).upper()
    
        protein_backbone = {"N", "CA", "C", "O", "OXT"}
        nucleotide_backbone = {
            "P", "OP1", "OP2", "OP3",
            "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
            "O5*", "C5*", "C4*", "O4*", "C3*", "O3*", "C2*", "O2*", "C1*",
        }
    
        if IonFixer._is_protein_residue_name(resname):
            return aname in protein_backbone
    
        if IonFixer._is_nucleotide_residue_name(resname):
            return aname in nucleotide_backbone
    
        return False
    
    
    @staticmethod
    def _maybe_set_global_parameter_default(force, param_name, value):
        for i in range(force.getNumGlobalParameters()):
            if force.getGlobalParameterName(i) == param_name:
                force.setGlobalParameterDefaultValue(i, value)
                return True
        return False
    
    
    def _get_atom_by_idx(self, atom_idx):
        for atom in self.selected_structure.atoms:
            if int(atom.idx) == int(atom_idx):
                return atom
        raise RuntimeError(f"[ERROR] Could not find atom with idx={atom_idx} in selected_structure")

    def _find_ion_atom_in_selected_structure(self):
        if self.added_ion_atom_idx is None:
            raise RuntimeError(
                "[ERROR] No added ion atom index recorded. Run merge_system() first."
            )
        return self._get_atom_by_idx(self.added_ion_atom_idx)

    def _find_dummy_water_oxygen_atoms(self):
        if not self.added_water_oxygen_indices:
            return []

        oxygen_atoms = []
        for atom_idx in self.added_water_oxygen_indices:
            oxygen_atoms.append(self._get_atom_by_idx(atom_idx))
        return oxygen_atoms

    def _current_coordination_distances_A(self):
        ion_xyz = np.asarray([self.ion_atom.xx, self.ion_atom.xy, self.ion_atom.xz], dtype=float)
        return [
            float(np.linalg.norm(np.asarray([a.xx, a.xy, a.xz], dtype=float) - ion_xyz))
            for a in self.coord_atoms_ordered
        ]

    @staticmethod
    def _set_global_parameter_default(force, param_name, value):
        for i in range(force.getNumGlobalParameters()):
            if force.getGlobalParameterName(i) == param_name:
                force.setGlobalParameterDefaultValue(i, value)
                return
        raise RuntimeError(f"[ERROR] Force does not contain global parameter {param_name!r}")

    def _spec_atoms_exist(self):
        if not self.spec_atoms:
            raise RuntimeError(
                "[ERROR] IonFixer no runtime atoms set consider running get_spec_atoms() first."
            )

    def _complex_structure_exists(self):
        if not hasattr(self, "complex_structure") or self.complex_structure is None:
            raise RuntimeError(
                "[ERROR] IonFixer complex structure not built. Run create_complex_structure() first."
            )

    def _selected_structure_exists(self):
        if not hasattr(self, "selected_structure") or self.selected_structure is None:
            raise RuntimeError(
                "[ERROR] IonFixer selected structure not built. Run get_residue_selection() first."
            )

    def _ion_and_waters_exist(self):
        if not hasattr(self, "ion_structure") or self.ion_structure is None:
            raise RuntimeError("[ERROR] Ion structure not built. Run get_paramitised_ion() first.")
        if not hasattr(self, "waters") or self.waters is None:
            raise RuntimeError("[ERROR] Dummy waters not built. Run get_paramitised_waters() first.")

    def _openmm_system_exists(self):
        if self.openmm_system is None:
            raise RuntimeError("[ERROR] OpenMM System not built. Run build_system() first.")
    
    def should_apply_map_restraint(self):
        return (
            getattr(self.system, "density", None) is not None
            and not bool(getattr(self.system.options, "no_map", False))
        )
    
    
    def cut_density_map_around_structure(
        self,
        pad_A=4.0,
        heavy_only=True,
        pad_value=0.0,
    ):
        """
        Cut a local EM submap around the current selected_structure.
    
        The returned map keeps the original world-space coordinate frame,
        just cropped to a local box around the structure.
        """
        self._selected_structure_exists()
    
        density = getattr(self.system, "density", None)
        if density is None:
            raise RuntimeError("[ERROR] self.system.density is None, cannot cut local map.")
    
        coords = []
        for atom in self.selected_structure.atoms:
            if heavy_only and not self._is_heavy_atom(atom):
                continue
            coords.append([atom.xx, atom.xy, atom.xz])
    
        if not coords:
            raise RuntimeError("[ERROR] No atoms available to define local map box.")
    
        coords = np.asarray(coords, dtype=float)
    
        mins = coords.min(axis=0) - float(pad_A)
        maxs = coords.max(axis=0) + float(pad_A)
    
        apix = np.asarray(density.apix, dtype=float)  # (ax, ay, az) in Å/px
        extents = np.maximum(maxs - mins, apix)
    
        nxyz = np.ceil(extents / apix).astype(int) + 1
        nx, ny, nz = int(nxyz[0]), int(nxyz[1]), int(nxyz[2])
    
        submap = density.submap(
            origin=(float(mins[0]), float(mins[1]), float(mins[2])),
            box_size=(nz, ny, nx),   # EMMap.submap expects (z, y, x)
            pad_value=float(pad_value),
        )
    
        # Keep the parent contour level if present
        if hasattr(density, "map_contour"):
            submap.map_contour = density.map_contour
    
        self.local_density_map = submap
        return submap
    
    def identify_map_restrained_atoms(self):
        """
        Use all heavy atoms in selected_structure except:
          - the added dummy waters
    
        The added ion is included.
        """
        self._selected_structure_exists()
    
        added_water_resnums = {int(x) for x in self.added_water_resnums}
        map_atoms = []
        seen = set()
    
        for atom in self.selected_structure.atoms:
            if not self._is_heavy_atom(atom):
                continue
    
            resnum = int(atom.residue.number)
    
            if resnum in added_water_resnums:
                continue
    
            idx = int(atom.idx)
            if idx not in seen:
                seen.add(idx)
                map_atoms.append(atom)
    
        self.map_atoms = map_atoms
        self.map_atom_indices = [int(a.idx) for a in map_atoms]
    
        return {
            "map_atoms": map_atoms,
            "map_atom_indices": self.map_atom_indices,
        }
    
    
    def apply_map_restraint(
        self,
        k_map=1.0,
        map_pad_A=4.0,
        smooth_sigma_A=0.0,
        smooth_sigma_vox=0.0,
        normalise=True,
        force_group=7,
    ):
        """
        Cut a local density map around the structure and add a map potential
        restraint using ForceBuilder.create_map_potential().
    
        The force is applied to the selected map atoms, excluding dummy waters
        but keeping the added ion.
        """
        self._openmm_system_exists()
    
        if not self.should_apply_map_restraint():
            return None
    
        if self.local_density_map is None:
            self.cut_density_map_around_structure(pad_A=map_pad_A, heavy_only=True)
    
        if self.map_atom_indices is None:
            self.identify_map_restrained_atoms()
    
        if not self.map_atom_indices:
            raise RuntimeError("[ERROR] No atoms identified for map restraint.")
    
        added_water_resnums = {int(x) for x in self.added_water_resnums}
    
        filtered_map_atom_indices = []
        for atom_idx in self.map_atom_indices:
            atom = self._get_atom_by_idx(int(atom_idx))
            resnum = int(atom.residue.number)
    
            # exclude dummy waters only
            if resnum in added_water_resnums:
                continue
    
            filtered_map_atom_indices.append(int(atom_idx))
    
        if not filtered_map_atom_indices:
            raise RuntimeError("[ERROR] No non-dummy-water atoms identified for map restraint.")
    
        map_force = ForceBuilder.create_map_potential(
            density_map_obj=self.local_density_map,
            global_k=float(k_map),
            smooth_sigma_vox=float(smooth_sigma_vox),
            smooth_sigma_A=float(smooth_sigma_A),
            normalise=bool(normalise),
            force_group=int(force_group),
        )
    
        for atom_idx in filtered_map_atom_indices:
            map_force.addBond([int(atom_idx)], [])
    
        self.openmm_system.addForce(map_force)
        self.map_force = map_force
        self.map_atom_indices = filtered_map_atom_indices
        self.map_atoms = [self._get_atom_by_idx(i) for i in filtered_map_atom_indices]
    
        return map_force
    
    
    def _get_openmm_topology(self):
        if hasattr(self.selected_structure, "topology") and self.selected_structure.topology is not None:
            return self.selected_structure.topology
        raise RuntimeError(
            "[ERROR] selected_structure has no OpenMM topology. "
            "Expected a ParmEd Structure with .topology available."
        )
    
    
    def _get_openmm_positions(self):
        """
        Return positions as an OpenMM Quantity in Å.
        Prefer ParmEd positions if present, otherwise build from atom coordinates.
        """
        if hasattr(self.selected_structure, "positions") and self.selected_structure.positions is not None:
            return self.selected_structure.positions
    
        coords_A = np.array(
            [[atom.xx, atom.xy, atom.xz] for atom in self.selected_structure.atoms],
            dtype=float,
        )
        return coords_A * unit.angstrom
    
    
    def create_simulation(
        self,
        temperature_K=300.0,
        friction_per_ps=1.0,
        step_size_ps=0.002,
        platform_name=None,
        platform_properties=None,
        initialize_velocities=True,
        random_seed=None,
    ):
        """
        Create an OpenMM Simulation from self.openmm_system and self.selected_structure.
    
        Returns
        -------
        simulation : openmm.app.Simulation
        """
        self._selected_structure_exists()
        self._openmm_system_exists()
    
        topology = self._get_openmm_topology()
        positions = self._get_openmm_positions()
    
        integrator = LangevinMiddleIntegrator(
            float(temperature_K) * unit.kelvin,
            float(friction_per_ps) / unit.picosecond,
            float(step_size_ps) * unit.picoseconds,
        )
    
        if random_seed is not None:
            integrator.setRandomNumberSeed(int(random_seed))
    
        if platform_name is not None:
            platform = Platform.getPlatformByName(str(platform_name))
            simulation = app.Simulation(
                topology,
                self.openmm_system,
                integrator,
                platform,
                platform_properties or {},
            )
        else:
            simulation = app.Simulation(
                topology,
                self.openmm_system,
                integrator,
            )
    
        simulation.context.setPositions(positions)
    
        if initialize_velocities:
            simulation.context.setVelocitiesToTemperature(float(temperature_K) * unit.kelvin)
    
        self.integrator = integrator
        self.simulation = simulation
        return simulation
    
    
    def _sync_selected_structure_from_context(self):
        """
        Pull the current coordinates from the OpenMM Context back into
        self.selected_structure.
        """
        if not hasattr(self, "simulation") or self.simulation is None:
            raise RuntimeError("[ERROR] No simulation present. Run create_simulation() first.")
    
        state = self.simulation.context.getState(getPositions=True, getEnergy=True)
        coords_A = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    
        for atom, xyz in zip(self.selected_structure.atoms, coords_A):
            atom.xx = float(xyz[0])
            atom.xy = float(xyz[1])
            atom.xz = float(xyz[2])
    
        if hasattr(self.selected_structure, "coordinates"):
            self.selected_structure.coordinates = np.asarray(coords_A, dtype=float)
        if hasattr(self.selected_structure, "positions"):
            self.selected_structure.positions = np.asarray(coords_A, dtype=float) * unit.angstrom
    
        return state
    
    
    def _get_ion_geometry_summary(self):
        """
        Return current ion-coordination distances and pair angles from the simulation.
        """
        if self.ion_atom is None or self.coord_atoms_ordered is None:
            self.identify_ion_angle_restraint_atoms()
    
        ion_xyz = np.asarray([self.ion_atom.xx, self.ion_atom.xy, self.ion_atom.xz], dtype=float)
        coord_xyz = np.asarray([[a.xx, a.xy, a.xz] for a in self.coord_atoms_ordered], dtype=float)
    
        distances_A = [
            float(np.linalg.norm(xyz - ion_xyz))
            for xyz in coord_xyz
        ]
    
        angles_deg = []
        for i in range(len(coord_xyz)):
            vi = coord_xyz[i] - ion_xyz
            ni = np.linalg.norm(vi)
            if ni < 1e-12:
                continue
            vi = vi / ni
    
            for j in range(i + 1, len(coord_xyz)):
                vj = coord_xyz[j] - ion_xyz
                nj = np.linalg.norm(vj)
                if nj < 1e-12:
                    continue
                vj = vj / nj
    
                c = float(np.clip(np.dot(vi, vj), -1.0, 1.0))
                angles_deg.append(float(np.degrees(np.arccos(c))))
    
        return {
            "distances_A": distances_A,
            "angles_deg": angles_deg,
        }
    
    
    def minimise_and_relax_ion_geometry(
        self,
        temperature_K=300.0,
        friction_per_ps=1.0,
        step_size_ps=0.002,
        platform_name=None,
        platform_properties=None,
        random_seed=None,
        minimization_tolerance=5.0,
        minimization_max_iterations=500,
        md_steps=2000,
        reinitialize_velocities=True,
        final_minimization=True,
    ):
        """
        Minimise, run restrained dynamics, then optionally minimise again.
    
        Returns
        -------
        dict with energies, final positions, and ion-geometry summary
        """
        if not hasattr(self, "simulation") or self.simulation is None:
            self.create_simulation(
                temperature_K=temperature_K,
                friction_per_ps=friction_per_ps,
                step_size_ps=step_size_ps,
                platform_name=platform_name,
                platform_properties=platform_properties,
                initialize_velocities=False,
                random_seed=random_seed,
            )
    
        # Initial energy before any optimisation
        state0 = self.simulation.context.getState(getEnergy=True)
        e0 = state0.getPotentialEnergy()
    
        # First minimisation
        self.simulation.minimizeEnergy(
            tolerance=float(minimization_tolerance) * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=int(minimization_max_iterations),
        )
    
        state1 = self.simulation.context.getState(getEnergy=True, getPositions=True)
        e1 = state1.getPotentialEnergy()
    
        # Short restrained MD to let the ion geometry relax
        if int(md_steps) > 0:
            if reinitialize_velocities:
                self.simulation.context.setVelocitiesToTemperature(float(temperature_K) * unit.kelvin)
            self.simulation.step(int(md_steps))
    
        state2 = self.simulation.context.getState(getEnergy=True, getPositions=True)
        e2 = state2.getPotentialEnergy()
    
        # Optional final minimisation
        if final_minimization:
            self.simulation.minimizeEnergy(
                tolerance=float(minimization_tolerance) * unit.kilojoule_per_mole / unit.nanometer,
                maxIterations=int(minimization_max_iterations),
            )
    
        statef = self._sync_selected_structure_from_context()
        ef = statef.getPotentialEnergy()
    
        # Refresh mapped atoms from final coordinates
        self.ion_atom = self._find_ion_atom_in_selected_structure()
        if self.coord_atom_indices is not None:
            self.coord_atoms_ordered = [self._get_atom_by_idx(i) for i in self.coord_atom_indices]
    
        geom = self._get_ion_geometry_summary()
    
        result = {
            "potential_energy_initial_kj_mol": float(e0.value_in_unit(unit.kilojoule_per_mole)),
            "potential_energy_after_first_min_kj_mol": float(e1.value_in_unit(unit.kilojoule_per_mole)),
            "potential_energy_after_md_kj_mol": float(e2.value_in_unit(unit.kilojoule_per_mole)),
            "potential_energy_final_kj_mol": float(ef.value_in_unit(unit.kilojoule_per_mole)),
            "ion_distances_A": geom["distances_A"],
            "ion_pair_angles_deg": geom["angles_deg"],
            "final_positions_A": statef.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
        }
    
        self.optimisation_result = result
        return result

    def write_output_pdb(self, filename=None, use_context_positions=True, keep_ids=True):
        """
        Write the current selected_structure to a PDB file.
        """
        self._selected_structure_exists()
    
        if filename is None:
            filename = os.path.join(self.system.output, "ion_refinment.pdb")
    
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    
        topology = self._get_openmm_topology()
    
        if (
            use_context_positions
            and hasattr(self, "simulation")
            and self.simulation is not None
        ):
            state = self.simulation.context.getState(getPositions=True)
            positions = state.getPositions()
        else:
            positions = self._get_openmm_positions()
    
        with open(filename, "w") as f:
            app.PDBFile.writeFile(topology, positions, f, keepIds=keep_ids)
    
        self.output_pdb_path = filename
        return filename
    
    
    def _structure_positions_from_atoms(self, structure):
        coords_A = np.asarray(
            [[atom.xx, atom.xy, atom.xz] for atom in structure.atoms],
            dtype=float,
        )
        return coords_A * unit.angstrom
    
    
    def _sync_structure_coordinate_arrays(self, structure):
        coords_A = np.asarray(
            [[atom.xx, atom.xy, atom.xz] for atom in structure.atoms],
            dtype=float,
        )
        if hasattr(structure, "coordinates"):
            structure.coordinates = coords_A
        if hasattr(structure, "positions"):
            structure.positions = coords_A * unit.angstrom
        return coords_A
    
    
    @staticmethod
    def _safe_name(text, fallback="obj"):
        text = str(text) if text is not None else fallback
        cleaned = "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in text)
        cleaned = cleaned.strip("._")
        return cleaned or fallback
    
    
    @staticmethod
    def _atom_identity_map(structure):
        """
        Build an atom-identity map:
          (chain, resnum, resname, atom_name, occurrence_index_within_residue)
        """
        amap = {}
    
        for res in structure.residues:
            chain = str(getattr(res, "chain", ""))
            try:
                resnum = int(res.number)
            except Exception:
                resnum = str(res.number)
            resname = str(res.name).upper()
    
            local_counts = {}
            for atom in res.atoms:
                aname = str(atom.name)
                occ = local_counts.get(aname, 0)
                local_counts[aname] = occ + 1
    
                key = (chain, resnum, resname, aname, occ)
                amap[key] = atom
    
        return amap
    
    
    def _copy_matching_atom_positions(self, source_structure, target_structure):
        """
        Copy matching atom coordinates from source_structure -> target_structure
        using residue/atom identity, ignoring atoms that do not exist in target.
        """
        src_map = self._atom_identity_map(source_structure)
        tgt_map = self._atom_identity_map(target_structure)
    
        n_updated = 0
        for key, src_atom in src_map.items():
            tgt_atom = tgt_map.get(key, None)
            if tgt_atom is None:
                continue
    
            tgt_atom.xx = float(src_atom.xx)
            tgt_atom.xy = float(src_atom.xy)
            tgt_atom.xz = float(src_atom.xz)
            n_updated += 1
    
        self._sync_structure_coordinate_arrays(target_structure)
        return n_updated
    
    
    def _get_refined_ion_structure_copy(self):
        """
        Return a copy of self.ion_structure with coordinates updated to the
        refined ion position from self.selected_structure.
        """
        self._ion_and_waters_exist()
    
        ion_atom = self._find_ion_atom_in_selected_structure()
        ion_copy = copy.deepcopy(self.ion_structure)
    
        ion_copy_atom = list(ion_copy.atoms)[0]
        ion_copy_atom.xx = float(ion_atom.xx)
        ion_copy_atom.xy = float(ion_atom.xy)
        ion_copy_atom.xz = float(ion_atom.xz)
    
        self._sync_structure_coordinate_arrays(ion_copy)
        return ion_copy
    
    
    def _update_or_append_refined_ion(self, structure):
        """
        Add the refined ion to a structure if not present, otherwise update its coords.
        No dummy waters are ever added here.
        """
        refined_ion = self._get_refined_ion_structure_copy()
        ion_res = list(refined_ion.residues)[0]
        ion_atom = list(refined_ion.atoms)[0]
    
        ion_chain = str(getattr(ion_res, "chain", ""))
        try:
            ion_resnum = int(ion_res.number)
        except Exception:
            ion_resnum = str(ion_res.number)
        ion_resname = str(ion_res.name).upper()
    
        for res in structure.residues:
            chain = str(getattr(res, "chain", ""))
            try:
                resnum = int(res.number)
            except Exception:
                resnum = str(res.number)
            resname = str(res.name).upper()
    
            if chain == ion_chain and resnum == ion_resnum and resname == ion_resname:
                atoms = list(res.atoms)
                if len(atoms) != 1:
                    raise RuntimeError(
                        f"[ERROR] Expected ion residue ({ion_chain}, {ion_resnum}, {ion_resname}) "
                        f"to contain 1 atom, found {len(atoms)}."
                    )
                atoms[0].xx = float(ion_atom.xx)
                atoms[0].xy = float(ion_atom.xy)
                atoms[0].xz = float(ion_atom.xz)
                self._sync_structure_coordinate_arrays(structure)
                return structure
    
        structure += refined_ion
        self._sync_structure_coordinate_arrays(structure)
        return structure
    
    
    def update_original_complex_with_refined_positions(self, add_ion=True):
        """
        Update self.complex_structure using refined coordinates from self.selected_structure,
        and optionally add the refined ion (no dummy waters).
        """
        self._complex_structure_exists()
        self._selected_structure_exists()
    
        n_updated = self._copy_matching_atom_positions(
            source_structure=self.selected_structure,
            target_structure=self.complex_structure,
        )
    
        if add_ion:
            self._update_or_append_refined_ion(self.complex_structure)
    
        self._sync_structure_coordinate_arrays(self.complex_structure)
    
        return {
            "n_atoms_updated": n_updated,
            "ion_added": bool(add_ion),
        }
    
    
    def _update_ligand_complex_structures_from_refined_complex(self):
        """
        Update each ligand.complex_structure from the corresponding contiguous
        ligand atom block in self.complex_structure.
    
        Assumes self.complex_structure was built as:
            protein + lig1 + lig2 + ...
        """
        updates = []
    
        complex_atoms = list(self.complex_structure.atoms)
        start = len(self.system.protein.complex_structure.atoms)
    
        for i, lig in enumerate(self.system.ligand, start=1):
            lig_struct = getattr(lig, "complex_structure", None)
    
            if lig_struct is None:
                updates.append(
                    {
                        "ligand_index": i,
                        "updated": False,
                        "updated_atoms": 0,
                        "reason": "missing complex_structure",
                    }
                )
                continue
    
            lig_atoms = list(lig_struct.atoms)
            end = start + len(lig_atoms)
            refined_atoms = complex_atoms[start:end]
    
            if len(refined_atoms) != len(lig_atoms):
                updates.append(
                    {
                        "ligand_index": i,
                        "updated": False,
                        "updated_atoms": 0,
                        "reason": (
                            f"atom-count mismatch refined_block={len(refined_atoms)} "
                            f"lig_struct={len(lig_atoms)}"
                        ),
                    }
                )
                start = end
                continue
    
            for lig_atom, ref_atom in zip(lig_atoms, refined_atoms):
                lig_atom.xx = float(ref_atom.xx)
                lig_atom.xy = float(ref_atom.xy)
                lig_atom.xz = float(ref_atom.xz)
    
            self._sync_structure_coordinate_arrays(lig_struct)
    
            updates.append(
                {
                    "ligand_index": i,
                    "updated": True,
                    "updated_atoms": len(lig_atoms),
                    "atom_range": (start, end),
                    "source": "self.complex_structure_block",
                }
            )
    
            start = end
    
        return updates
    
    
    def _update_ligand_mols_from_complex_structures(self):
        """
        Update each ligand.mol conformer coordinates from the corresponding
        contiguous ligand atom block in self.complex_structure.
    
        Assumes ligand.mol atom order matches ligand.complex_structure atom order.
        """
        updates = []
    
        complex_atoms = list(self.complex_structure.atoms)
        start = len(self.system.protein.complex_structure.atoms)
    
        for i, lig in enumerate(self.system.ligand, start=1):
            mol = getattr(lig, "mol", None)
            lig_struct = getattr(lig, "complex_structure", None)
    
            if mol is None or lig_struct is None:
                lig_n_atoms = len(lig_struct.atoms) if lig_struct is not None else 0
                start += lig_n_atoms
                updates.append(
                    {
                        "ligand_index": i,
                        "updated": False,
                        "reason": "missing mol or complex_structure",
                    }
                )
                continue
    
            n_struct_atoms = len(lig_struct.atoms)
            end = start + n_struct_atoms
            refined_atoms = complex_atoms[start:end]
            n_mol_atoms = int(mol.GetNumAtoms())
    
            if len(refined_atoms) != n_struct_atoms:
                updates.append(
                    {
                        "ligand_index": i,
                        "updated": False,
                        "reason": (
                            f"complex block / ligand structure mismatch "
                            f"refined_block={len(refined_atoms)} lig_struct={n_struct_atoms}"
                        ),
                    }
                )
                start = end
                continue
    
            if n_struct_atoms != n_mol_atoms:
                updates.append(
                    {
                        "ligand_index": i,
                        "updated": False,
                        "reason": f"atom-count mismatch refined_block={n_struct_atoms} mol={n_mol_atoms}",
                    }
                )
                start = end
                continue
    
            if mol.GetNumConformers() == 0:
                conf = Chem.Conformer(n_mol_atoms)
                mol.AddConformer(conf, assignId=True)
    
            conf = mol.GetConformer()
    
            for j, atom in enumerate(refined_atoms):
                conf.SetAtomPosition(
                    j,
                    Point3D(float(atom.xx), float(atom.xy), float(atom.xz)),
                )
    
            updates.append(
                {
                    "ligand_index": i,
                    "updated": True,
                    "n_atoms": n_mol_atoms,
                    "atom_range": (start, end),
                    "source": "self.complex_structure_block",
                }
            )
    
            start = end
    
        return updates
    
    
    def export_refinement_outputs(self, refinement_dir=None, tag=None, keep_ids=True):
        """
        Update original structures with refined coordinates, add the refined ion
        to the complex (no dummy waters), update ligand mol coordinates, and write:
    
          - protein+ion PDB
          - one ligand SDF per ligand
        """
        self._complex_structure_exists()
        self._selected_structure_exists()
    
        if refinement_dir is None:
            refinement_dir = os.path.join(self.system.output, "refinement")
        os.makedirs(refinement_dir, exist_ok=True)
    
        if tag is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = f"ionfixer_{stamp}_{uuid.uuid4().hex[:8]}"
    
        # 1. Update the original refined complex and add the refined ion
        complex_update_info = self.update_original_complex_with_refined_positions(add_ion=True)
    
        # 2. Update ligand complex structures and ligand mol coords from contiguous blocks
        ligand_complex_updates = self._update_ligand_complex_structures_from_refined_complex()
        ligand_mol_updates = self._update_ligand_mols_from_complex_structures()
    
        # 3. Build a protein-only export structure and add the refined ion
        protein_export = copy.deepcopy(self.system.protein.complex_structure)
        self._copy_matching_atom_positions(
            source_structure=self.complex_structure,
            target_structure=protein_export,
        )
        self._update_or_append_refined_ion(protein_export)
    
        protein_pdb_name = f"protein_{tag}.pdb"
        protein_pdb_path = os.path.join(refinement_dir, protein_pdb_name)
    
        with open(protein_pdb_path, "w") as f:
            app.PDBFile.writeFile(
                protein_export.topology,
                self._structure_positions_from_atoms(protein_export),
                f,
                keepIds=keep_ids,
            )
    
        # 4. Write ligand SDFs
        ligand_sdf_paths = []
        for i, lig in enumerate(self.system.ligand, start=1):
            mol = getattr(lig, "mol", None)
            if mol is None:
                continue
    
            lig_name = None
            for attr in ("name", "resname", "res_name", "id"):
                if hasattr(lig, attr):
                    lig_name = getattr(lig, attr)
                    if lig_name:
                        break
    
            lig_label = self._safe_name(lig_name, fallback=f"ligand_{i:02d}")
            sdf_name = f"{lig_label}_{tag}.sdf"
            sdf_path = os.path.join(refinement_dir, sdf_name)
    
            writer = Chem.SDWriter(sdf_path)
            writer.write(mol)
            writer.close()
    
            ligand_sdf_paths.append(sdf_path)
    
        outputs = {
            "refinement_dir": refinement_dir,
            "tag": tag,
            "protein_pdb": protein_pdb_path,
            "ligand_sdfs": ligand_sdf_paths,
            "complex_update": complex_update_info,
            "ligand_complex_updates": ligand_complex_updates,
            "ligand_mol_updates": ligand_mol_updates,
        }
    
        self.refinement_outputs = outputs
        return outputs
    
    
    #initial fixer 
        # -------------------------------------------------------------------------
    # Optional quick ligand torsion pre-optimiser for better ion start placement
    # -------------------------------------------------------------------------
    
    def _classify_spec_atoms(self, tol=1e-3):
        """
        Split self.spec_atoms into:
          - ligand specs mapped to (ligand_index, ligand_atom_index)
          - protein specs as original atom-spec objects
        """
        ligand_spec_matches = []
        protein_spec_atoms = []
    
        for spec_atom in self.spec_atoms:
            matches = []
    
            for lig_idx, lig in enumerate(self.system.ligand):
                lig_struct = getattr(lig, "complex_structure", None)
                if lig_struct is None:
                    continue
    
                try:
                    lig_atom = find_atom_from_spec_by_coord_and_element(spec_atom, lig_struct, tol=tol)
                    matches.append((lig_idx, int(lig_atom.idx)))
                except RuntimeError:
                    pass
    
            if len(matches) == 0:
                protein_spec_atoms.append(spec_atom)
            elif len(matches) == 1:
                ligand_spec_matches.append((spec_atom, matches[0][0], matches[0][1]))
            else:
                raise RuntimeError(
                    f"[ERROR] Atom-spec {spec_atom} matched multiple ligands; "
                    "cannot determine target ligand."
                )
    
        return ligand_spec_matches, protein_spec_atoms
    
    def _identify_target_ligand_for_quick_minimiser(self, tol=1e-3):
        ligand_spec_matches, protein_spec_atoms = self._classify_spec_atoms(tol=tol)
    
        if not ligand_spec_matches:
            raise RuntimeError(
                "[ERROR] Quick ion-start minimiser needs at least one ligand atom-spec."
            )
        if not protein_spec_atoms:
            raise RuntimeError(
                "[ERROR] Quick ion-start minimiser needs at least one protein atom-spec."
            )
    
        ligand_indices = sorted({lig_idx for _, lig_idx, _ in ligand_spec_matches})
        if len(ligand_indices) != 1:
            raise RuntimeError(
                "[ERROR] Quick ion-start minimiser currently supports one ligand at a time; "
                f"found ligand specs on ligand indices {ligand_indices}."
            )
    
        ligand_index = ligand_indices[0]
        ligand_atom_indices = [
            lig_atom_idx
            for _, lig_idx, lig_atom_idx in ligand_spec_matches
            if lig_idx == ligand_index
        ]
    
        protein_spec_xyz = np.asarray([a.get_point() for a in protein_spec_atoms], dtype=float)
        protein_spec_elements = [str(a.get_element()).upper() for a in protein_spec_atoms]
    
        return ligand_index, ligand_atom_indices, protein_spec_xyz, protein_spec_elements
    
    def _map_spec_list_to_target_ligand_atom_indices(self, spec_atom_list, ligand_index, tol=1e-3):
        """
        Map a list of spec atoms to atom indices in the target ligand only.
        Non-ligand / other-ligand specs are ignored.
        """
        lig_struct = self.system.ligand[int(ligand_index)].complex_structure
        out = []
    
        for spec_atom in spec_atom_list:
            try:
                lig_atom = find_atom_from_spec_by_coord_and_element(spec_atom, lig_struct, tol=tol)
                out.append(int(lig_atom.idx))
            except RuntimeError:
                pass
    
        return sorted(set(out))
    
    def _find_matching_atom_by_coord_and_element(self, source_atom, target_structure, tol=1e-3):
        """
        Match a ParmEd atom into another structure by xyz + element.
        """
        target_xyz = np.asarray([source_atom.xx, source_atom.xy, source_atom.xz], dtype=float)
        target_elem = str(source_atom.element_name).upper()
    
        for atom in target_structure.atoms:
            atom_elem = str(atom.element_name).upper()
            if atom_elem != target_elem:
                continue
    
            atom_xyz = np.asarray([atom.xx, atom.xy, atom.xz], dtype=float)
            if np.allclose(atom_xyz, target_xyz, atol=tol, rtol=0.0):
                return atom
    
        raise RuntimeError(
            f"[ERROR] Could not match target-ligand atom {source_atom.name} "
            f"({target_elem}) into selected_structure."
        )
    
    def _get_target_ligand_atoms_in_selected_structure(self, ligand_index, tol=1e-3):
        """
        Return the atoms of the target ligand as they appear in selected_structure,
        preserving ligand atom order.
        """
        self._selected_structure_exists()
    
        lig_struct = self.system.ligand[int(ligand_index)].complex_structure
        selected_atoms = []
    
        for atom in lig_struct.atoms:
            selected_atoms.append(
                self._find_matching_atom_by_coord_and_element(atom, self.selected_structure, tol=tol)
            )
    
        return selected_atoms
    
    @staticmethod
    def _bond_distance_matrix_int(mol):
        return np.asarray(Chem.GetDistanceMatrix(mol), dtype=int)
    
    @staticmethod
    def _vdw_radius_A(element_symbol):
        radii = {
            "H": 1.20,
            "C": 1.70,
            "N": 1.55,
            "O": 1.52,
            "F": 1.47,
            "P": 1.80,
            "S": 1.80,
            "CL": 1.75,
            "BR": 1.85,
            "I": 1.98,
            "MG": 1.73,
            "ZN": 1.39,
            "CA": 1.94,
            "MN": 1.61,
            "FE": 1.56,
            "CU": 1.40,
            "CO": 1.52,
            "NI": 1.63,
        }
        return radii.get(str(element_symbol).upper(), 1.70)
    
    @staticmethod
    def _coord_target_distance_A(element_symbol, ion_type=None):
        """
        Simple preferred ion-contact distance in Å.
        This is deliberately conservative for the quick pre-optimiser.
        """
        elem = str(element_symbol).upper()
        if elem in ("O", "N"):
            return 2.10
        if elem == "S":
            return 2.35
        if elem in ("F", "CL", "BR", "I"):
            return 2.60
        return 2.15
    
    @staticmethod
    def _bfs_side_atoms(mol, start_idx, blocked_idx):
        """
        Return the connected component reached from start_idx when the edge to blocked_idx
        is removed.
        """
        out = set()
        q = deque([start_idx])
        seen = {blocked_idx}
    
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            out.add(u)
    
            atom = mol.GetAtomWithIdx(int(u))
            for nbr in atom.GetNeighbors():
                v = nbr.GetIdx()
                if v not in seen:
                    q.append(v)
    
        return out
    
    @staticmethod
    def _choose_torsion_terminal_neighbor(mol, center_idx, exclude_idx, prefer_atoms=None):
        nbrs = [
            n.GetIdx()
            for n in mol.GetAtomWithIdx(int(center_idx)).GetNeighbors()
            if n.GetIdx() != int(exclude_idx)
        ]
        if not nbrs:
            return None
    
        if prefer_atoms is not None:
            heavy_pref = [i for i in nbrs if i in prefer_atoms and mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
            if heavy_pref:
                return heavy_pref[0]
            any_pref = [i for i in nbrs if i in prefer_atoms]
            if any_pref:
                return any_pref[0]
    
        heavy = [i for i in nbrs if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
        if heavy:
            return heavy[0]
    
        return nbrs[0]
    
    def _resolve_allowed_ligand_atoms(self, mol, include_atoms=None, exclude_atoms=None):
        n_atoms = int(mol.GetNumAtoms())
        all_atoms = set(range(n_atoms))
    
        if include_atoms is None:
            allowed_atoms = set(all_atoms)
        else:
            allowed_atoms = {int(i) for i in include_atoms}
    
        if exclude_atoms is not None:
            allowed_atoms -= {int(i) for i in exclude_atoms}
    
        return allowed_atoms
    
    def _find_allowed_ligand_torsions(
        self,
        mol,
        include_atoms=None,
        exclude_atoms=None,
    ):
        """
        Identify rotatable torsions whose moved side is fully contained in the allowed set.
        """
        allowed_atoms = self._resolve_allowed_ligand_atoms(
            mol,
            include_atoms=include_atoms,
            exclude_atoms=exclude_atoms,
        )
    
        torsions = []
    
        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                continue
            if bond.IsInRing():
                continue
    
            a = int(bond.GetBeginAtomIdx())
            b = int(bond.GetEndAtomIdx())
    
            if mol.GetAtomWithIdx(a).GetDegree() < 2 or mol.GetAtomWithIdx(b).GetDegree() < 2:
                continue
    
            side_b = self._bfs_side_atoms(mol, b, a)
            side_a = self._bfs_side_atoms(mol, a, b)
    
            orient = None
    
            if side_b.issubset(allowed_atoms) and not side_a.issubset(allowed_atoms):
                orient = (a, b, side_b)
            elif side_a.issubset(allowed_atoms) and not side_b.issubset(allowed_atoms):
                orient = (b, a, side_a)
            elif side_a.issubset(allowed_atoms) and side_b.issubset(allowed_atoms):
                if len(side_a) <= len(side_b):
                    orient = (b, a, side_a)
                else:
                    orient = (a, b, side_b)
    
            if orient is None:
                continue
    
            atom_a, atom_b, moved_side = orient
    
            i = self._choose_torsion_terminal_neighbor(mol, atom_a, atom_b, prefer_atoms=None)
            l = self._choose_torsion_terminal_neighbor(mol, atom_b, atom_a, prefer_atoms=moved_side)
    
            if i is None or l is None:
                continue
    
            torsions.append(
                {
                    "atoms": (int(i), int(atom_a), int(atom_b), int(l)),
                    "moved_atoms": sorted(int(x) for x in moved_side),
                    "bond": (int(atom_a), int(atom_b)),
                }
            )
    
        return torsions
    
    def _ligand_complex_atom_block(self, ligand_index):
        """
        Return (start, end) atom indices in self.complex_structure corresponding
        to ligand_index.
        """
        start = len(self.system.protein.complex_structure.atoms)
        for i, lig in enumerate(self.system.ligand):
            n = len(lig.complex_structure.atoms)
            end = start + n
            if i == int(ligand_index):
                return start, end
            start = end
    
        raise IndexError(f"[ERROR] Invalid ligand_index={ligand_index}")
    
    def _extract_ligand_xyz_from_mol(self, mol):
        conf = mol.GetConformer()
        xyz = np.zeros((mol.GetNumAtoms(), 3), dtype=float)
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            xyz[i] = [float(p.x), float(p.y), float(p.z)]
        return xyz
    
    def _set_mol_xyz(self, mol, xyz):
        if mol.GetNumConformers() == 0:
            mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()), assignId=True)
    
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])))
    
    def _compute_clash_penalty(
        self,
        ligand_xyz,
        ligand_elements,
        env_xyz,
        env_elements,
        intra_pairs,
        overlap_scale=0.80,
        clash_weight_inter=1.0,
        clash_weight_intra=1.0,
    ):
        penalty = 0.0
    
        if len(env_xyz) > 0:
            for i in range(len(ligand_xyz)):
                ri = self._vdw_radius_A(ligand_elements[i])
                xi = ligand_xyz[i]
                for j in range(len(env_xyz)):
                    rj = self._vdw_radius_A(env_elements[j])
                    cutoff = overlap_scale * (ri + rj)
                    d = float(np.linalg.norm(xi - env_xyz[j]))
                    if d < cutoff:
                        penalty += clash_weight_inter * (cutoff - d) ** 2
    
        for i, j in intra_pairs:
            ri = self._vdw_radius_A(ligand_elements[i])
            rj = self._vdw_radius_A(ligand_elements[j])
            cutoff = overlap_scale * (ri + rj)
            d = float(np.linalg.norm(ligand_xyz[i] - ligand_xyz[j]))
            if d < cutoff:
                penalty += clash_weight_intra * (cutoff - d) ** 2
    
        return penalty
    
    def _score_quick_ion_start_pose(
        self,
        ligand_xyz,
        ligand_elements,
        ligand_spec_atom_indices,
        protein_spec_xyz,
        protein_spec_elements,
        env_xyz,
        env_elements,
        intra_pairs,
        start_xyz,
        movable_atom_indices,
        distance_weight=10.0,
        clash_weight=1.0,
        pose_weight=3.0,
        donor_anchor_weight=6.0,
        map_weight=1.0,
        mi_score_fn=None,
    ):
        """
        Lower is better.
        """
        ligand_spec_atom_indices = np.asarray(ligand_spec_atom_indices, dtype=int)
        ligand_spec_xyz = ligand_xyz[ligand_spec_atom_indices]
        spec_xyz = np.vstack([ligand_spec_xyz, protein_spec_xyz])
    
        ligand_spec_idx_set = set(int(i) for i in ligand_spec_atom_indices)
        ligand_obstacle_xyz = []
        ligand_obstacle_elements = []
    
        for i in range(len(ligand_xyz)):
            if i in ligand_spec_idx_set:
                continue
            elem = str(ligand_elements[i]).upper()
            if elem == "H":
                continue
            ligand_obstacle_xyz.append(ligand_xyz[i])
            ligand_obstacle_elements.append(elem)
    
        if len(ligand_obstacle_xyz):
            obstacle_xyz = np.vstack([env_xyz, np.asarray(ligand_obstacle_xyz, dtype=float)])
            obstacle_elements = list(env_elements) + list(ligand_obstacle_elements)
        else:
            obstacle_xyz = np.asarray(env_xyz, dtype=float)
            obstacle_elements = list(env_elements)
    
        ion_xyz = generate_initial_ion_position(
            spec_xyz=spec_xyz,
            obstacle_xyz=obstacle_xyz,
            obstacle_elements=obstacle_elements,
            ion_name=self.system.options.ion_type,
            target_dists=None,
        )
    
        d = np.linalg.norm(spec_xyz - ion_xyz[None, :], axis=1)
    
        donor_elements = [ligand_elements[int(i)] for i in ligand_spec_atom_indices] + list(protein_spec_elements)
        target_d = np.asarray(
            [self._coord_target_distance_A(elem, getattr(self.system.options, "ion_type", None)) for elem in donor_elements],
            dtype=float,
        )
    
        distance_score = float(np.mean((d - target_d) ** 2))
    
        clash_score = self._compute_clash_penalty(
            ligand_xyz=ligand_xyz,
            ligand_elements=ligand_elements,
            env_xyz=env_xyz,
            env_elements=env_elements,
            intra_pairs=intra_pairs,
        )
    
        movable_atom_indices = np.asarray(sorted(set(int(i) for i in movable_atom_indices)), dtype=int)
        if movable_atom_indices.size > 0:
            pose_penalty = float(
                np.mean(
                    np.sum((ligand_xyz[movable_atom_indices] - start_xyz[movable_atom_indices]) ** 2, axis=1)
                )
            )
        else:
            pose_penalty = 0.0
    
        donor_anchor_penalty = float(
            np.mean(
                np.sum((ligand_xyz[ligand_spec_atom_indices] - start_xyz[ligand_spec_atom_indices]) ** 2, axis=1)
            )
        )
    
        map_score = 0.0
        if mi_score_fn is not None:
            try:
                map_score = float(
                    mi_score_fn(
                        ligand_xyz=ligand_xyz,
                        ion_xyz=np.asarray(ion_xyz, dtype=float),
                        fixer=self,
                    )
                )
            except Exception as e:
                raise RuntimeError(f"[ERROR] quick minimiser MI callback failed: {e}")
    
        total = (
            float(distance_weight) * distance_score
            + float(clash_weight) * clash_score
            + float(pose_weight) * pose_penalty
            + float(donor_anchor_weight) * donor_anchor_penalty
            - float(map_weight) * map_score
        )
    
        return total, {
            "ion_xyz": np.asarray(ion_xyz, dtype=float),
            "distance_score": float(distance_score),
            "clash_score": float(clash_score),
            "pose_penalty": float(pose_penalty),
            "donor_anchor_penalty": float(donor_anchor_penalty),
            "map_score": float(map_score),
            "total_score": float(total),
        }
    
    def _apply_optimised_ligand_pose(self, ligand_index, mol, selected_target_atoms=None):
        """
        Write optimised ligand coordinates back into:
          - ligand.mol
          - ligand.complex_structure
          - self.complex_structure ligand block
          - self.selected_structure target ligand atoms (if provided)
        """
        lig = self.system.ligand[int(ligand_index)]
        xyz = self._extract_ligand_xyz_from_mol(mol)
    
        # 1. ligand.mol
        self._set_mol_xyz(lig.mol, xyz)
    
        # 2. ligand.complex_structure
        lig_struct = lig.complex_structure
        if len(lig_struct.atoms) != len(xyz):
            raise RuntimeError(
                f"[ERROR] ligand.complex_structure atom count {len(lig_struct.atoms)} "
                f"!= mol atom count {len(xyz)}"
            )
        for atom, pos in zip(lig_struct.atoms, xyz):
            atom.xx = float(pos[0])
            atom.xy = float(pos[1])
            atom.xz = float(pos[2])
        self._sync_structure_coordinate_arrays(lig_struct)
    
        # 3. self.complex_structure contiguous ligand block
        start, end = self._ligand_complex_atom_block(int(ligand_index))
        block_atoms = list(self.complex_structure.atoms)[start:end]
        if len(block_atoms) != len(xyz):
            raise RuntimeError(
                f"[ERROR] complex ligand block atom count {len(block_atoms)} != mol atom count {len(xyz)}"
            )
        for atom, pos in zip(block_atoms, xyz):
            atom.xx = float(pos[0])
            atom.xy = float(pos[1])
            atom.xz = float(pos[2])
        self._sync_structure_coordinate_arrays(self.complex_structure)
    
        # 4. self.selected_structure target ligand atoms
        if selected_target_atoms is not None:
            if len(selected_target_atoms) != len(xyz):
                raise RuntimeError(
                    f"[ERROR] selected_structure target ligand atom count {len(selected_target_atoms)} "
                    f"!= mol atom count {len(xyz)}"
                )
            for atom, pos in zip(selected_target_atoms, xyz):
                atom.xx = float(pos[0])
                atom.xy = float(pos[1])
                atom.xz = float(pos[2])
            self._sync_structure_coordinate_arrays(self.selected_structure)
    
    
    def quick_preoptimize_ligand_for_ion_start(
        self,
        include_atoms=None,
        exclude_atoms=None,
        torsion_step_deg=20.0,
        min_torsion_step_deg=5.0,
        step_decay=0.5,
        max_outer_loops=4,
        random_seed=1,
        distance_weight=10.0,
        clash_weight=1.0,
        pose_weight=3.0,
        donor_anchor_weight=6.0,
        map_weight=1.0,
        mi_score_fn=None,
        map_pad_A=4.0,
        mi_resolution=None,
        mi_sigma_coeff=0.356,
        mi_weight_mode="mass",
        mi_n_bins=64,
        mi_normalized=False,
        mi_zscore=False,
        mi_normalise_model=True,
        tol=1e-3,
    ):
        """
        Optional torsion-only pre-optimiser for better ion-start geometry.
    
        This is a local, conservative pre-optimiser designed to open just enough
        space for the ion while keeping the ligand close to its starting pose.
    
        If density is available and mi_score_fn is None, a local cropped map
        around selected_structure is used for internal MI scoring.
        """
        self._spec_atoms_exist()
        self._complex_structure_exists()
        self._selected_structure_exists()
    
        # local preopt map only; do not reuse this later for map restraint
        self.preopt_local_density_map = None
        self.preopt_use_map_score = False
    
        if mi_score_fn is None and self.should_apply_map_restraint() and float(map_weight) != 0.0:
            preopt_map = self.cut_density_map_around_structure(
                pad_A=float(map_pad_A),
                heavy_only=True,
            )
            self.preopt_local_density_map = preopt_map
            self.local_density_map = None  # force a fresh cut later for the actual map restraint
            self.preopt_use_map_score = True
            self.preopt_map_resolution = float(
                mi_resolution
                if mi_resolution is not None
                else getattr(preopt_map, "resolution", getattr(self.system.density, "resolution", 3.0))
            )
            self.preopt_map_sigma_coeff = float(mi_sigma_coeff)
            self.preopt_map_weight_mode = str(mi_weight_mode)
            self.preopt_map_n_bins = int(mi_n_bins)
            self.preopt_map_normalized = bool(mi_normalized)
            self.preopt_map_zscore = bool(mi_zscore)
            self.preopt_map_normalise_model = bool(mi_normalise_model)
    
        ligand_index, ligand_spec_atom_indices, protein_spec_xyz, protein_spec_elements = (
            self._identify_target_ligand_for_quick_minimiser(tol=tol)
        )
    
        lig = self.system.ligand[int(ligand_index)]
        mol0 = Chem.Mol(lig.mol)
        if mol0.GetNumConformers() == 0:
            raise RuntimeError(
                "[ERROR] Target ligand.mol has no conformer for torsion pre-optimisation."
            )
    
        spec_exclude_atoms = self._map_spec_list_to_target_ligand_atom_indices(
            getattr(self, "exclude_atom_specs", []),
            ligand_index=int(ligand_index),
            tol=tol,
        )
    
        all_exclude_atoms = set(spec_exclude_atoms)
        if exclude_atoms is not None:
            all_exclude_atoms |= {int(i) for i in exclude_atoms}
    
        allowed_atoms = self._resolve_allowed_ligand_atoms(
            mol0,
            include_atoms=include_atoms,
            exclude_atoms=all_exclude_atoms,
        )
    
        torsions = self._find_allowed_ligand_torsions(
            mol0,
            include_atoms=include_atoms,
            exclude_atoms=all_exclude_atoms,
        )
        if not torsions:
            return {
                "performed": False,
                "reason": "no allowed torsions found",
                "ligand_index": int(ligand_index),
                "exclude_atoms_used": sorted(all_exclude_atoms),
            }
    
        selected_target_atoms = self._get_target_ligand_atoms_in_selected_structure(
            ligand_index=int(ligand_index),
            tol=tol,
        )
        selected_target_atom_ids = {id(a) for a in selected_target_atoms}
    
        env_xyz = []
        env_elements = []
        for atom in self.selected_structure.atoms:
            if id(atom) in selected_target_atom_ids:
                continue
            elem = str(atom.element_name).upper()
            if elem == "H":
                continue
            env_xyz.append([atom.xx, atom.xy, atom.xz])
            env_elements.append(elem)
    
        env_xyz = np.asarray(env_xyz, dtype=float) if env_xyz else np.zeros((0, 3), dtype=float)
    
        ligand_elements = [str(atom.element_name).upper() for atom in lig.complex_structure.atoms]
    
        topod = self._bond_distance_matrix_int(mol0)
        intra_pairs = []
        for i in range(mol0.GetNumAtoms()):
            ai = mol0.GetAtomWithIdx(i)
            if ai.GetAtomicNum() == 1:
                continue
            for j in range(i + 1, mol0.GetNumAtoms()):
                aj = mol0.GetAtomWithIdx(j)
                if aj.GetAtomicNum() == 1:
                    continue
                if int(topod[i, j]) >= 3:
                    intra_pairs.append((i, j))
    
        rng = np.random.default_rng(int(random_seed))
        best_mol = Chem.Mol(mol0)
        start_xyz = self._extract_ligand_xyz_from_mol(best_mol)
        best_xyz = start_xyz.copy()
    
        best_score, best_meta = self._score_quick_ion_start_pose(
            ligand_xyz=best_xyz,
            ligand_elements=ligand_elements,
            ligand_spec_atom_indices=ligand_spec_atom_indices,
            protein_spec_xyz=protein_spec_xyz,
            protein_spec_elements=protein_spec_elements,
            env_xyz=env_xyz,
            env_elements=env_elements,
            intra_pairs=intra_pairs,
            start_xyz=start_xyz,
            movable_atom_indices=sorted(allowed_atoms),
            distance_weight=distance_weight,
            clash_weight=clash_weight,
            pose_weight=pose_weight,
            donor_anchor_weight=donor_anchor_weight,
            map_weight=map_weight,
            mi_score_fn=mi_score_fn,
        )
    
        step = float(torsion_step_deg)
    
        for _outer in range(int(max_outer_loops)):
            improved_any = False
            order = rng.permutation(len(torsions))
    
            for tidx in order:
                tors = torsions[int(tidx)]
                a0, a1, a2, a3 = tors["atoms"]
    
                conf = best_mol.GetConformer()
                current = float(
                    rdMolTransforms.GetDihedralDeg(conf, int(a0), int(a1), int(a2), int(a3))
                )
    
                local_best_score = best_score
                local_best_mol = None
                local_best_meta = None
    
                for trial_angle in (current - step, current, current + step):
                    trial = Chem.Mol(best_mol)
                    tconf = trial.GetConformer()
                    rdMolTransforms.SetDihedralDeg(
                        tconf,
                        int(a0), int(a1), int(a2), int(a3),
                        float(trial_angle),
                    )
    
                    trial_xyz = self._extract_ligand_xyz_from_mol(trial)
                    trial_score, trial_meta = self._score_quick_ion_start_pose(
                        ligand_xyz=trial_xyz,
                        ligand_elements=ligand_elements,
                        ligand_spec_atom_indices=ligand_spec_atom_indices,
                        protein_spec_xyz=protein_spec_xyz,
                        protein_spec_elements=protein_spec_elements,
                        env_xyz=env_xyz,
                        env_elements=env_elements,
                        intra_pairs=intra_pairs,
                        start_xyz=start_xyz,
                        movable_atom_indices=sorted(allowed_atoms),
                        distance_weight=distance_weight,
                        clash_weight=clash_weight,
                        pose_weight=pose_weight,
                        donor_anchor_weight=donor_anchor_weight,
                        map_weight=map_weight,
                        mi_score_fn=mi_score_fn,
                    )
    
                    if trial_score < local_best_score:
                        local_best_score = trial_score
                        local_best_mol = trial
                        local_best_meta = trial_meta
    
                if local_best_mol is not None:
                    best_mol = local_best_mol
                    best_score = local_best_score
                    best_meta = local_best_meta
                    improved_any = True
    
            if not improved_any:
                step *= float(step_decay)
                if step < float(min_torsion_step_deg):
                    break
    
        self._apply_optimised_ligand_pose(
            int(ligand_index),
            best_mol,
            selected_target_atoms=selected_target_atoms,
        )
    
        self.get_spec_atoms()
    
        result = {
            "performed": True,
            "ligand_index": int(ligand_index),
            "n_torsions": len(torsions),
            "best_score": float(best_score),
            "best_meta": best_meta,
            "final_torsion_step_deg": float(step),
            "exclude_atoms_used": sorted(all_exclude_atoms),
            "movable_atoms_used": sorted(allowed_atoms),
            "used_internal_mi": bool(self.preopt_use_map_score and mi_score_fn is None),
        }
    
        self.quick_preopt_result = result
    
        writer = Chem.SDWriter(os.path.join(self.system.output, "ionfixer_preopt_result.sdf"))
        writer.write(best_mol)
        writer.close()
    
        return result
    
    def __quick_preoptimize_ligand_for_ion_start(
        self,
        include_atoms=None,
        exclude_atoms=None,
        torsion_step_deg=20.0,
        min_torsion_step_deg=5.0,
        step_decay=0.5,
        max_outer_loops=4,
        random_seed=1,
        distance_weight=10.0,
        clash_weight=1.0,
        pose_weight=3.0,
        donor_anchor_weight=6.0,
        map_weight=1.0,
        mi_score_fn=None,
        tol=1e-3,
    ):
        """
        Optional torsion-only pre-optimiser for better ion-start geometry.
    
        This is a local, conservative pre-optimiser designed to open just enough
        space for the ion while keeping the ligand close to its starting pose.
        """
        self._spec_atoms_exist()
        self._complex_structure_exists()
        self._selected_structure_exists()
    
        ligand_index, ligand_spec_atom_indices, protein_spec_xyz, protein_spec_elements = (
            self._identify_target_ligand_for_quick_minimiser(tol=tol)
        )
    
        lig = self.system.ligand[int(ligand_index)]
        mol0 = Chem.Mol(lig.mol)
        if mol0.GetNumConformers() == 0:
            raise RuntimeError(
                "[ERROR] Target ligand.mol has no conformer for torsion pre-optimisation."
            )
    
        # Map exclude_specs automatically onto the target ligand
        spec_exclude_atoms = self._map_spec_list_to_target_ligand_atom_indices(
            getattr(self, "exclude_atom_specs", []),
            ligand_index=int(ligand_index),
            tol=tol,
        )
    
        all_exclude_atoms = set(spec_exclude_atoms)
        if exclude_atoms is not None:
            all_exclude_atoms |= {int(i) for i in exclude_atoms}
    
        allowed_atoms = self._resolve_allowed_ligand_atoms(
            mol0,
            include_atoms=include_atoms,
            exclude_atoms=all_exclude_atoms,
        )
    
        torsions = self._find_allowed_ligand_torsions(
            mol0,
            include_atoms=include_atoms,
            exclude_atoms=all_exclude_atoms,
        )
        if not torsions:
            return {
                "performed": False,
                "reason": "no allowed torsions found",
                "ligand_index": int(ligand_index),
                "exclude_atoms_used": sorted(all_exclude_atoms),
            }
    
        # Match target ligand atoms inside selected_structure before moving anything
        selected_target_atoms = self._get_target_ligand_atoms_in_selected_structure(
            ligand_index=int(ligand_index),
            tol=tol,
        )
        selected_target_atom_ids = {id(a) for a in selected_target_atoms}
    
        # Local environment from selected_structure only, excluding the target ligand atoms
        env_xyz = []
        env_elements = []
        for atom in self.selected_structure.atoms:
            if id(atom) in selected_target_atom_ids:
                continue
            elem = str(atom.element_name).upper()
            if elem == "H":
                continue
            env_xyz.append([atom.xx, atom.xy, atom.xz])
            env_elements.append(elem)
    
        env_xyz = np.asarray(env_xyz, dtype=float) if env_xyz else np.zeros((0, 3), dtype=float)
    
        ligand_elements = [str(atom.element_name).upper() for atom in lig.complex_structure.atoms]
    
        topod = self._bond_distance_matrix_int(mol0)
        intra_pairs = []
        for i in range(mol0.GetNumAtoms()):
            ai = mol0.GetAtomWithIdx(i)
            if ai.GetAtomicNum() == 1:
                continue
            for j in range(i + 1, mol0.GetNumAtoms()):
                aj = mol0.GetAtomWithIdx(j)
                if aj.GetAtomicNum() == 1:
                    continue
                if int(topod[i, j]) >= 3:
                    intra_pairs.append((i, j))
    
        rng = np.random.default_rng(int(random_seed))
        best_mol = Chem.Mol(mol0)
        start_xyz = self._extract_ligand_xyz_from_mol(best_mol)
        best_xyz = start_xyz.copy()
    
        best_score, best_meta = self._score_quick_ion_start_pose(
            ligand_xyz=best_xyz,
            ligand_elements=ligand_elements,
            ligand_spec_atom_indices=ligand_spec_atom_indices,
            protein_spec_xyz=protein_spec_xyz,
            protein_spec_elements=protein_spec_elements,
            env_xyz=env_xyz,
            env_elements=env_elements,
            intra_pairs=intra_pairs,
            start_xyz=start_xyz,
            movable_atom_indices=sorted(allowed_atoms),
            distance_weight=distance_weight,
            clash_weight=clash_weight,
            pose_weight=pose_weight,
            donor_anchor_weight=donor_anchor_weight,
            map_weight=map_weight,
            mi_score_fn=mi_score_fn,
        )
    
        step = float(torsion_step_deg)
    
        for _outer in range(int(max_outer_loops)):
            improved_any = False
            order = rng.permutation(len(torsions))
    
            for tidx in order:
                tors = torsions[int(tidx)]
                a0, a1, a2, a3 = tors["atoms"]
    
                conf = best_mol.GetConformer()
                current = float(
                    rdMolTransforms.GetDihedralDeg(conf, int(a0), int(a1), int(a2), int(a3))
                )
    
                local_best_score = best_score
                local_best_mol = None
                local_best_meta = None
    
                for trial_angle in (current - step, current, current + step):
                    trial = Chem.Mol(best_mol)
                    tconf = trial.GetConformer()
                    rdMolTransforms.SetDihedralDeg(
                        tconf,
                        int(a0), int(a1), int(a2), int(a3),
                        float(trial_angle),
                    )
    
                    trial_xyz = self._extract_ligand_xyz_from_mol(trial)
                    trial_score, trial_meta = self._score_quick_ion_start_pose(
                        ligand_xyz=trial_xyz,
                        ligand_elements=ligand_elements,
                        ligand_spec_atom_indices=ligand_spec_atom_indices,
                        protein_spec_xyz=protein_spec_xyz,
                        protein_spec_elements=protein_spec_elements,
                        env_xyz=env_xyz,
                        env_elements=env_elements,
                        intra_pairs=intra_pairs,
                        start_xyz=start_xyz,
                        movable_atom_indices=sorted(allowed_atoms),
                        distance_weight=distance_weight,
                        clash_weight=clash_weight,
                        pose_weight=pose_weight,
                        donor_anchor_weight=donor_anchor_weight,
                        map_weight=map_weight,
                        mi_score_fn=mi_score_fn,
                    )
    
                    if trial_score < local_best_score:
                        local_best_score = trial_score
                        local_best_mol = trial
                        local_best_meta = trial_meta
    
                if local_best_mol is not None:
                    best_mol = local_best_mol
                    best_score = local_best_score
                    best_meta = local_best_meta
                    improved_any = True
    
            if not improved_any:
                step *= float(step_decay)
                if step < float(min_torsion_step_deg):
                    break
    
        self._apply_optimised_ligand_pose(
            int(ligand_index),
            best_mol,
            selected_target_atoms=selected_target_atoms,
        )
    
        # refresh atom-spec runtime atoms so downstream steps see moved ligand coordinates
        self.get_spec_atoms()
    
        result = {
            "performed": True,
            "ligand_index": int(ligand_index),
            "n_torsions": len(torsions),
            "best_score": float(best_score),
            "best_meta": best_meta,
            "final_torsion_step_deg": float(step),
            "exclude_atoms_used": sorted(all_exclude_atoms),
            "movable_atoms_used": sorted(allowed_atoms),
        }
    
        self.quick_preopt_result = result
    
        # debug output
        writer = Chem.SDWriter(os.path.join(self.system.output, "ionfixer_preopt_result.sdf"))
        writer.write(best_mol)
        writer.close()
    
        return result
        
    
    @staticmethod
    def _ensure_xyz_apix(apix):
        arr = np.asarray(apix, dtype=float).reshape(-1)
        if arr.size == 1:
            return np.array([arr[0], arr[0], arr[0]], dtype=float)
        if arr.size != 3:
            raise ValueError(f"[ERROR] apix must be scalar or length-3, got {arr}")
        return arr
    
    @staticmethod
    def _zscore_array(arr, eps=1e-12):
        arr = np.asarray(arr, dtype=float)
        mu = float(np.mean(arr))
        sd = float(np.std(arr))
        if sd < eps:
            return np.zeros_like(arr, dtype=float)
        return (arr - mu) / sd
    
    @staticmethod
    def _clean_element_symbol(symbol):
        s = "".join(ch for ch in str(symbol).strip() if ch.isalpha())
        if not s:
            return "C"
        if len(s) == 1:
            return s.upper()
        return s[0].upper() + s[1:].lower()
    
    def _element_weight(self, element_symbol, weight_mode="mass"):
        sym = self._clean_element_symbol(element_symbol)
        if weight_mode == "ones":
            return 1.0
    
        pt = Chem.GetPeriodicTable()
        try:
            anum = int(pt.GetAtomicNumber(sym))
        except Exception:
            anum = 6
    
        if weight_mode == "atomic_number":
            return float(anum)
    
        if weight_mode == "mass":
            try:
                return float(pt.GetAtomicWeight(anum))
            except Exception:
                return 12.0
    
        raise ValueError(f"[ERROR] Unknown weight_mode={weight_mode!r}")
    
    def _blur_points_to_map_array(
        self,
        points_xyz,
        point_elements,
        emmap,
        resolution,
        sigma_coeff=0.356,
        weight_mode="mass",
        normalise=True,
    ):
        """
        Rasterise points to the EMMap grid and Gaussian blur them.
        Returns a numpy array on the same grid as emmap.density_map.
        """
        origin = np.asarray(emmap.origin, dtype=float)
        apix = self._ensure_xyz_apix(emmap.apix)
        nz, ny, nx = np.asarray(emmap.density_map).shape
    
        grid = np.zeros((nz, ny, nx), dtype=float)
        points_xyz = np.asarray(points_xyz, dtype=float)
    
        for xyz, elem in zip(points_xyz, point_elements):
            ix = int(round((xyz[0] - origin[0]) / apix[0]))
            iy = int(round((xyz[1] - origin[1]) / apix[1]))
            iz = int(round((xyz[2] - origin[2]) / apix[2]))
    
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                grid[iz, iy, ix] += self._element_weight(elem, weight_mode=weight_mode)
    
        sigma_A = float(sigma_coeff) * float(resolution)
        sigma_zyx = np.array(
            [
                sigma_A / apix[2],
                sigma_A / apix[1],
                sigma_A / apix[0],
            ],
            dtype=float,
        )
    
        blurred = gaussian_filter(grid, sigma=sigma_zyx, mode="constant", cval=0.0)
    
        if normalise:
            blurred = self._zscore_array(blurred)
    
        return blurred
    
    def _mutual_information_score_from_arrays(
        self,
        arr_a,
        arr_b,
        n_bins=64,
        mask=None,
        nonzero_union=True,
        zscore=False,
        normalized=False,
        eps=1e-12,
    ):
        """
        Histogram MI between two same-shaped arrays.
        Higher is better.
        """
        a = np.asarray(arr_a, dtype=float)
        b = np.asarray(arr_b, dtype=float)
    
        if a.shape != b.shape:
            raise ValueError(f"[ERROR] MI array shape mismatch: {a.shape} vs {b.shape}")
    
        if mask is None:
            use_mask = np.ones(a.shape, dtype=bool)
            if nonzero_union:
                use_mask &= ((a != 0.0) | (b != 0.0))
        else:
            use_mask = np.asarray(mask, dtype=bool)
            if use_mask.shape != a.shape:
                raise ValueError(
                    f"[ERROR] MI mask shape mismatch: {use_mask.shape} vs {a.shape}"
                )
    
        av = a[use_mask].ravel()
        bv = b[use_mask].ravel()
    
        if av.size == 0:
            return 0.0
    
        if zscore:
            av = self._zscore_array(av, eps=eps)
            bv = self._zscore_array(bv, eps=eps)
    
        joint_hist, _, _ = np.histogram2d(av, bv, bins=int(n_bins))
        total = float(joint_hist.sum())
        if total <= 0.0:
            return 0.0
    
        pxy = joint_hist / total
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
    
        nz_mask = pxy > 0.0
        mi = float(np.sum(pxy[nz_mask] * np.log((pxy[nz_mask] + eps) / (px_py[nz_mask] + eps))))
    
        if not normalized:
            return mi
    
        px_nz = px > 0.0
        py_nz = py > 0.0
        hx = -float(np.sum(px[px_nz] * np.log(px[px_nz] + eps)))
        hy = -float(np.sum(py[py_nz] * np.log(py[py_nz] + eps)))
        denom = np.sqrt(max(hx * hy, eps))
        return float(mi / denom)
    
    def _score_local_preopt_map_mi(self, ligand_xyz, ligand_elements, ion_xyz):
        """
        Score MI between the cropped local experimental map and a blurred
        ligand+ion model map on the same grid.
        """
        if self.preopt_local_density_map is None:
            return 0.0
    
        exp_map = np.asarray(self.preopt_local_density_map.density_map, dtype=float)
    
        points_xyz = np.vstack(
            [
                np.asarray(ligand_xyz, dtype=float),
                np.asarray(ion_xyz, dtype=float).reshape(1, 3),
            ]
        )
        point_elements = list(ligand_elements) + [str(self.system.options.ion_type)]
    
        model_map = self._blur_points_to_map_array(
            points_xyz=points_xyz,
            point_elements=point_elements,
            emmap=self.preopt_local_density_map,
            resolution=float(self.preopt_map_resolution),
            sigma_coeff=float(self.preopt_map_sigma_coeff),
            weight_mode=self.preopt_map_weight_mode,
            normalise=bool(self.preopt_map_normalise_model),
        )
    
        return self._mutual_information_score_from_arrays(
            exp_map,
            model_map,
            n_bins=int(self.preopt_map_n_bins),
            mask=None,
            nonzero_union=True,
            zscore=bool(self.preopt_map_zscore),
            normalized=bool(self.preopt_map_normalized),
        )
    
    def run(
        self,
        *,
        apply_restraints=True,
        include_angles=True,
        target_dist_A=None,
        flat_bottom_A=0.2,
        k_dist=1000.0,
        k_ang=200.0,
        k_pin=5000,
        k_map=150.0,
        map_pad_A=4.0,
        map_smooth_sigma_A=0.0,
        map_smooth_sigma_vox=0.0,
        map_normalise=True,
        #start here
        preoptimize_ligand_start=False,
        preopt_include_atoms=None,
        preopt_exclude_atoms=None,
        preopt_torsion_step_deg=30.0,
        preopt_min_torsion_step_deg=5.0,
        preopt_step_decay=0.5,
        preopt_max_outer_loops=4,
        preopt_random_seed=1,
        preopt_distance_weight=0.0,
        preopt_clash_weight=5.0,
        preopt_map_weight=1.0,
        preopt_pose_weight=3.0,
        preopt_donor_anchor_weight=6.0,
        preopt_mi_score_fn=None,
    ):
        self.get_spec_atoms()
        self.get_coordination_number()
        self.create_complex_structure()
        self.get_residue_selection()
        
        if preoptimize_ligand_start:
            '''
            preopt_result = self.quick_preoptimize_ligand_for_ion_start(
                include_atoms=preopt_include_atoms,
                exclude_atoms=[a.atom_idx for a in self.exclude_atom_specs],
                torsion_step_deg=preopt_torsion_step_deg,
                min_torsion_step_deg=preopt_min_torsion_step_deg,
                step_decay=preopt_step_decay,
                max_outer_loops=preopt_max_outer_loops,
                random_seed=preopt_random_seed,
                distance_weight=preopt_distance_weight,
                clash_weight=preopt_clash_weight,
                pose_weight=preopt_pose_weight,
                donor_anchor_weight=preopt_donor_anchor_weight,
                map_weight=preopt_map_weight,
                mi_score_fn=preopt_mi_score_fn,
            )
            print(f"Quick ion-start preoptimiser: {preopt_result}")
            '''
            
            preopt_result = self.quick_preoptimize_ligand_for_ion_start(
                include_atoms=preopt_include_atoms,
                exclude_atoms=[a.atom_idx for a in self.exclude_atom_specs],
                torsion_step_deg=preopt_torsion_step_deg,
                min_torsion_step_deg=preopt_min_torsion_step_deg,
                step_decay=preopt_step_decay,
                max_outer_loops=preopt_max_outer_loops,
                random_seed=preopt_random_seed,
                distance_weight=preopt_distance_weight,
                clash_weight=preopt_clash_weight,
                pose_weight=preopt_pose_weight,
                donor_anchor_weight=preopt_donor_anchor_weight,
                map_weight=preopt_map_weight,
                mi_score_fn=preopt_mi_score_fn,
                map_pad_A=map_pad_A,
                mi_resolution=getattr(self.system.density, "resolution", 3.0) if getattr(self.system, "density", None) is not None else 3.0,
                mi_sigma_coeff=0.356,
                mi_weight_mode="mass",
                mi_n_bins=64,
                mi_normalized=False,
                mi_zscore=False,
                mi_normalise_model=True,
            )
            print(f"Quick ion-start preoptimiser: {preopt_result}")
            #import pdb 
            #pdb.set_trace()
        
        self.get_initial_position()
        self.get_paramitised_ion()
        self.get_paramitised_waters()
        self.merge_system()
        self.build_system()

       
        self.identify_ion_angle_restraint_atoms()
        self.apply_ion_coordination_restraints(
            target_dist_A=target_dist_A,
            flat_bottom_A=flat_bottom_A,
            include_angles=include_angles,
            k_dist=k_dist,
            k_ang=k_ang,
        )
        
        self.identify_atoms_to_pin()
        self.apply_pin_restraints(k_pin=k_pin)
        
        
        if self.should_apply_map_restraint():
            self.cut_density_map_around_structure(pad_A=map_pad_A, heavy_only=True)
            self.identify_map_restrained_atoms()
            self.apply_map_restraint(
                k_map=k_map,
                map_pad_A=map_pad_A,
                smooth_sigma_A=map_smooth_sigma_A,
                smooth_sigma_vox=map_smooth_sigma_vox,
                normalise=map_normalise,
            )
        
        self.create_simulation(
            temperature_K=300.0,
            friction_per_ps=1.0,
            step_size_ps=0.002,
            platform_name=self.system.platform,   # or "OpenCL", "CPU", or None
            platform_properties=None,
            initialize_velocities=False,
        )
        
        result = self.minimise_and_relax_ion_geometry(
            temperature_K=300.0,
            friction_per_ps=1.0,
            step_size_ps=0.002,
            platform_name=self.system.platform,
            platform_properties=None,
            minimization_tolerance=5.0,
            minimization_max_iterations=500,
            md_steps=2000,
            reinitialize_velocities=True,
            final_minimization=True,
        )
        
        out_pdb = self.write_output_pdb(
            filename=os.path.join(self.system.output, "ion_refinment.pdb"),
            use_context_positions=True,
        )
        print(f"Wrote refined ion structure to: {out_pdb}")
        
        refinement_outputs = self.export_refinement_outputs()
        result["refinement_outputs"] = refinement_outputs
        print(f"Wrote ion-fixer refinement outputs to: {refinement_outputs['refinement_dir']}")
        
        import pdb 
        pdb.set_trace()
        
    


# ------------------------------------------------------------------------------- DEBUG
def debug_missing_bond_params(structure):
    bad = []

    for i, bond in enumerate(structure.bonds):
        btype = getattr(bond, "type", None)
        if btype is None:
            bad.append(
                (
                    i,
                    "bond.type is None",
                    bond.atom1.idx, bond.atom1.name, bond.atom1.residue.name, bond.atom1.residue.number,
                    bond.atom2.idx, bond.atom2.name, bond.atom2.residue.name, bond.atom2.residue.number,
                )
            )
            continue

        req = getattr(btype, "req", None)
        k = getattr(btype, "k", None)
        if req is None or k is None:
            bad.append(
                (
                    i,
                    f"bond.type has req={req}, k={k}",
                    bond.atom1.idx, bond.atom1.name, bond.atom1.residue.name, bond.atom1.residue.number,
                    bond.atom2.idx, bond.atom2.name, bond.atom2.residue.name, bond.atom2.residue.number,
                )
            )

    return bad


def find_atom_from_spec_by_coord_and_element(atom_spec, complex_structure, tol=1e-3):
    """
    Find the matching atom in a ParmEd complex_structure by comparing
    x, y, z coordinates and element.
    """
    target_xyz = np.asarray(atom_spec.get_point(), dtype=float)
    target_elem = atom_spec.get_element()

    for atom in complex_structure.atoms:
        atom_elem = atom.element_name.upper()
        atom_xyz = np.array([atom.xx, atom.xy, atom.xz], dtype=float)

        if atom_elem != target_elem:
            continue

        if np.allclose(atom_xyz, target_xyz, atol=tol, rtol=0.0):
            return atom

    raise RuntimeError(
        f"[ERROR] Could not find matching atom for element={target_elem} "
        f"at xyz={target_xyz.tolist()} within tol={tol}"
    )

