#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


from itertools import permutations
import numpy as np

from .refine_utils import (create_structure_subset,
                           generate_initial_ion_position,
                           find_atom_from_spec_by_coord_and_element,
                           _default_k_ang_for_cn,
                           _default_target_distances)

from ChemEM.protocols.core.forces import ForceBuilder
from ChemEM.protocols.core.core_utils import all_pairwise_distances_leq
from ChemEM.parsers.parametised_ions import (
    create_parameterized_ion_structure,
    propose_dummy_water_oxygen_positions,
    coord_geom_to_int,
    create_parametrised_tip3p_water,
)

from ChemEM.parsers.parse_forcefield import ff_load
from openmm import app, unit, LangevinMiddleIntegrator
from ChemEM.tools.resources import make_openmm_simulation

import os
import copy
from rdkit import Chem
from rdkit.Geometry import Point3D




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

class IonFixer:
    
    def __init__(self, system):
        self.system = system 
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
        self.excluded_pin_force = None

        self.fixed_target_dist_A = None
        self.spec_atom_indices_selected = None
        self.exclude_atom_indices_selected = None
        
        self.local_density_map = None
        self.map_atoms = None
        self.map_atom_indices = None
        self.map_force = None
    
    def get_spec_atoms(self):
        self.spec_atoms = []
        self.spec_atom_is_ligand = []
        self.exclude_atom_specs = []

        atom_specs = list(getattr(self.system.options, "atom_specs", []) or [])
        exclude_specs = list(getattr(self.system.options, "exclude_specs", []) or [])

        if not atom_specs:
            raise RuntimeError("[ERROR] IonFixer requires at least one atom-spec argument")

        for spec in atom_specs:
            if spec.startswith("LIG"):
                atom = self.system.ligand.get_atom_idx_from_spec(spec)
                self.spec_atoms.append(atom)
                self.spec_atom_is_ligand.append(True)
                print(f"[DEBUG spec_atom] {spec} -> LIGAND, elem={atom.get_element()}, xyz={atom.get_point()}")
            else:
                atom = self.system.protein.get_atom_idx_from_spec(spec)
                self.spec_atoms.append(atom)
                self.spec_atom_is_ligand.append(False)
                print(f"[DEBUG spec_atom] {spec} -> PROTEIN, elem={atom.get_element()}, xyz={atom.get_point()}")

        for spec in exclude_specs:
            if spec.startswith("LIG"):
                self.exclude_atom_specs.append(self.system.ligand.get_atom_idx_from_spec(spec))
            else:
                self.exclude_atom_specs.append(self.system.protein.get_atom_idx_from_spec(spec))

        spec_signatures = {
            (
                str(a.get_element()).upper(),
                tuple(np.round(np.asarray(a.get_point(), dtype=float), 3)),
            )
            for a in self.spec_atoms
        }
        
        exclude_signatures = {
            (
                str(a.get_element()).upper(),
                tuple(np.round(np.asarray(a.get_point(), dtype=float), 3)),
            )
            for a in self.exclude_atom_specs
        }
        
        overlap = spec_signatures & exclude_signatures
        if overlap:
            raise RuntimeError(
                "[ERROR] atom_specs and exclude_specs overlap. "
                "A coordinating atom cannot also be excluded/pinned."
            )
        
        
    def get_coordination_number(self):
        self.coordination_number = coord_geom_to_int(self.system.options.coordination_geometry)

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
        spec_signatures = {
            (
                str(atom.get_element()).upper(),
                tuple(np.round(np.asarray(atom.get_point(), dtype=float), 3)),
            )
            for atom in self.spec_atoms
        }

        obstacle_xyz = []
        obstacle_elements = []

        for atom in self.selected_structure.atoms:
            elem = atom.element_name.upper()
            if elem == "H":
                continue

            signature = (
                elem,
                tuple(np.round(np.array([atom.xx, atom.xy, atom.xz], dtype=float), 3)),
            )
            if signature in spec_signatures:
                continue

            obstacle_xyz.append([atom.xx, atom.xy, atom.xz])
            obstacle_elements.append(elem)

        obstacle_xyz = np.asarray(obstacle_xyz, dtype=float)

        spec_weights = [0.1 if is_lig else 1.0 for is_lig in self.spec_atom_is_ligand]

        initial_pos = generate_initial_ion_position(
            spec_xyz=spec_xyz,
            obstacle_xyz=obstacle_xyz,
            obstacle_elements=obstacle_elements,
            ion_name=self.system.options.ion_type,
            target_dists=self.fixed_target_dist_A,
            spec_weights=spec_weights,
        )

        init_dists = [float(np.linalg.norm(spec_xyz[i] - initial_pos)) for i in range(len(spec_xyz))]
        print(f"[DEBUG initial_pos] spec_weights={spec_weights}")
        print(f"[DEBUG initial_pos] ion_position={initial_pos.tolist()}")
        print(f"[DEBUG initial_pos] distances_to_spec_atoms={[f'{d:.2f}' for d in init_dists]}")
        print(f"[DEBUG initial_pos] is_ligand={self.spec_atom_is_ligand}")

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

    
    def cache_spec_atom_indices_in_selected_structure(self, tol=1e-3):
        """
        Resolve atom-spec objects onto the current selected_structure once and
        store stable selected_structure atom indices for reuse after the atoms
        move during refinement.
        """
        self._spec_atoms_exist()
        self._selected_structure_exists()

        spec_indices = []
        for spec_atom in self.spec_atoms:
            atom = find_atom_from_spec_by_coord_and_element(
                spec_atom,
                self.selected_structure,
                tol=tol,
            )
            spec_indices.append(int(atom.idx))
        self.spec_atom_indices_selected = spec_indices

        exclude_indices = []
        for spec_atom in getattr(self, "exclude_atom_specs", []):
            atom = find_atom_from_spec_by_coord_and_element(
                spec_atom,
                self.selected_structure,
                tol=tol,
            )
            exclude_indices.append(int(atom.idx))
        self.exclude_atom_indices_selected = exclude_indices

        return {
            "spec_atom_indices_selected": list(self.spec_atom_indices_selected),
            "exclude_atom_indices_selected": list(self.exclude_atom_indices_selected),
        }
    
    
    def setup_constraints(self):
        
        
        self.k_ang = self.system.options.k_ang
        
        if self.k_ang is None:
            self.k_ang = _default_k_ang_for_cn(self.coordination_number)
        
        self.distance_only_fraction = min(self.system.options.distance_fraction, 1.0)
        
        self.total_cycles = max(self.system.options.n_cycles, 1)
        self.use_staged = self.total_cycles > 1
        self.early_cycles = 0
        self.late_cycles = self.total_cycles
        if self.use_staged:
            self.early_cycles = int(round(float(self.distance_only_fraction) * self.total_cycles))
            self.early_cycles = max(1, min(self.total_cycles - 1, self.early_cycles))
            self.late_cycles = self.total_cycles - self.early_cycles
        
    
    def _prepare_refinement_system_from_current_structure(
        self,
        *,
        include_angles=True,
        target_dist_A=None,
        flat_bottom_A=0.0,
        k_dist=1000.0,
        k_ang=None,
        k_pin=5000.0,
        k_pin_excluded=None,
        k_map=0.0,
        map_pad_A=4.0,
        map_smooth_sigma_A=0.0,
        map_smooth_sigma_vox=0.0,
        map_normalise=True,
        temperature_K=50.0,
        friction_per_ps=1.0,
        step_size_ps=0.002,
        platform_name=None,
        platform_properties=None,
        random_seed=None,
    ):
        """
        Rebuild the OpenMM System from the current selected_structure coordinates
        and apply the requested restraint set for the next refinement phase.
        """
        self._selected_structure_exists()

        if hasattr(self, "simulation") and self.simulation is not None:
            self._sync_selected_structure_from_context()

        if k_ang is None:
            k_ang = _default_k_ang_for_cn(self.coordination_number)

        if target_dist_A is None:
            target_dist_A = list(self.fixed_target_dist_A or [])
        else:
            target_dist_A = list(target_dist_A)

        if not target_dist_A:
            if self.coord_atom_indices is None or self.ion_atom is None:
                self.identify_ion_angle_restraint_atoms()
            n_coord = len(self.coord_atom_indices)
            target_dist_A = list(_default_target_distances(
                self.system.options.ion_type, n_coord
            ))
            print(f"[DEBUG _prepare] target_dist_A from defaults: {target_dist_A}")

        self._reset_runtime_forces_and_simulation()

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
        if k_pin_excluded is None:
            k_pin_excluded = max(float(k_pin) * 5.0, float(k_pin))
        self.apply_pin_restraints(
            k_pin=k_pin,
            k_pin_excluded=k_pin_excluded,
        )

        if self.should_apply_map_restraint() and float(k_map) > 0.0:
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
            temperature_K=temperature_K,
            friction_per_ps=friction_per_ps,
            step_size_ps=step_size_ps,
            platform_name=platform_name,
            platform_properties=platform_properties,
            initialize_velocities=False,
            random_seed=random_seed,
        )

        return {
            "target_dist_A": list(target_dist_A),
            "map_applied": bool(self.map_force is not None),
            "k_dist": float(k_dist),
            "k_ang": float(k_ang),
            "k_pin": float(k_pin),
            "k_map": float(k_map),
        }
    
    
    
    def iterative_minimize_ion_geometry(
        self,
        n_cycles=50,
        k_dist_start=500.0,
        k_dist_end=2000.0,
        k_ang=None,
        k_ang_start=None,
        k_ang_end=None,
        k_pin=5000.0,
        minimization_tolerance=1.0,
        minimization_max_iterations=1000,
        final_md_steps=0,
        final_md_temperature_K=50.0,
        friction_per_ps=1.0,
        step_size_ps=0.002,
        platform_name=None,
        platform_properties=None,
        random_seed=None,
        k_pin_name="k_pin",
        k_dist_name="k_ion_dist",
        k_ang_name="k_ion_ang",
        ion_reposition_fraction=1.0,
        stage_label=None,
    ):
        """
        Iterative minimization protocol for ion geometry correction.

        Distance and angle restraint strengths can be ramped independently.
        Ion repositioning can be limited to an early fraction of the cycles so
        the donors can first find the ion, after which the local geometry is
        allowed to settle without the ion continually chasing the donors.
        """
        if k_ang is None:
            k_ang = _default_k_ang_for_cn(self.coordination_number)
        if k_ang_end is None:
            k_ang_end = float(k_ang)
        if k_ang_start is None:
            k_ang_start = float(k_ang_end)

        if not hasattr(self, "simulation") or self.simulation is None:
            self.create_simulation(
                temperature_K=final_md_temperature_K,
                friction_per_ps=friction_per_ps,
                step_size_ps=step_size_ps,
                platform_name=platform_name,
                platform_properties=platform_properties,
                initialize_velocities=False,
                random_seed=random_seed,
            )

        state0 = self.simulation.context.getState(getEnergy=True)
        e0 = state0.getPotentialEnergy()

        if self.pin_force is not None:
            self.simulation.context.setParameter(
                k_pin_name,
                float(k_pin) * unit.kilojoule_per_mole / unit.nanometer**2,
            )

        has_angle_force = (
            self.ion_restraint_forces is not None
            and self.ion_restraint_forces.get("angle_force") is not None
        )

        ion_idx = int(self.added_ion_atom_idx)
        cycle_log = []
        n_cycles = max(int(n_cycles), 0)
        reposition_fraction = min(max(float(ion_reposition_fraction), 0.0), 1.0)
        reposition_cycles = int(np.ceil(reposition_fraction * n_cycles)) if n_cycles > 0 else 0

        for cycle in range(1, n_cycles + 1):
            frac = float(cycle) / float(n_cycles) if n_cycles > 0 else 1.0
            k_dist_current = float(k_dist_start) + frac * (float(k_dist_end) - float(k_dist_start))
            k_ang_current = float(k_ang_start) + frac * (float(k_ang_end) - float(k_ang_start))

            self.simulation.context.setParameter(
                k_dist_name,
                float(k_dist_current) * unit.kilojoule_per_mole / unit.nanometer**2,
            )
            if has_angle_force:
                self.simulation.context.setParameter(
                    k_ang_name,
                    float(k_ang_current) * unit.kilojoule_per_mole / unit.radian**2,
                )

            self.simulation.minimizeEnergy(
                tolerance=float(minimization_tolerance)
                * unit.kilojoule_per_mole / unit.nanometer,
                maxIterations=int(minimization_max_iterations),
            )

            state = self.simulation.context.getState(getPositions=True, getEnergy=True)
            all_pos_A = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            energy_kj = float(state.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole
            ))

            spec_pos_A = np.array(
                [all_pos_A[i] for i in self.coord_atom_indices], dtype=float
            )
            ion_pos_A = np.array(all_pos_A[ion_idx], dtype=float)

            dists_before = [
                float(np.linalg.norm(sp - ion_pos_A)) for sp in spec_pos_A
            ]

            did_reposition = bool(cycle <= reposition_cycles)
            if did_reposition:
                new_ion_pos_A = self._recompute_ion_position(spec_pos_A)
                positions_nm = state.getPositions(asNumpy=True)
                positions_nm[ion_idx] = new_ion_pos_A * 0.1 * unit.nanometer
                self.simulation.context.setPositions(positions_nm)
                dists_after = [
                    float(np.linalg.norm(sp - new_ion_pos_A)) for sp in spec_pos_A
                ]
                ion_displacement = float(np.linalg.norm(new_ion_pos_A - ion_pos_A))
            else:
                new_ion_pos_A = ion_pos_A
                dists_after = list(dists_before)
                ion_displacement = 0.0

            cycle_log.append({
                "stage": stage_label,
                "cycle": cycle,
                "k_dist": k_dist_current,
                "k_ang": k_ang_current if has_angle_force else None,
                "energy_kj_mol": energy_kj,
                "ion_spec_dists_before_A": dists_before,
                "ion_spec_dists_after_A": dists_after,
                "ion_displacement_A": ion_displacement,
                "did_reposition_ion": did_reposition,
            })

            print(
                f"  [{stage_label or 'iter'}] Cycle {cycle}/{n_cycles}: E={energy_kj:.1f} kJ/mol, "
                f"ion moved {ion_displacement:.3f} A, "
                f"dists={[f'{d:.2f}' for d in dists_after]}"
            )

        self.simulation.context.setParameter(
            k_dist_name,
            float(k_dist_end) * unit.kilojoule_per_mole / unit.nanometer**2,
        )
        if has_angle_force:
            self.simulation.context.setParameter(
                k_ang_name,
                float(k_ang_end) * unit.kilojoule_per_mole / unit.radian**2,
            )
        self.simulation.minimizeEnergy(
            tolerance=float(minimization_tolerance)
            * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=int(minimization_max_iterations),
        )

        if final_md_steps > 0:
            self.integrator.setTemperature(
                float(final_md_temperature_K) * unit.kelvin
            )
            self.simulation.context.setVelocitiesToTemperature(
                float(final_md_temperature_K) * unit.kelvin
            )
            self.simulation.step(int(final_md_steps))

            self.simulation.minimizeEnergy(
                tolerance=float(minimization_tolerance)
                * unit.kilojoule_per_mole / unit.nanometer,
                maxIterations=int(minimization_max_iterations),
            )

        statef = self._sync_selected_structure_from_context()
        ef = statef.getPotentialEnergy()

        self.ion_atom = self._find_ion_atom_in_selected_structure()
        if self.coord_atom_indices is not None:
            self.coord_atoms_ordered = [
                self._get_atom_by_idx(i) for i in self.coord_atom_indices
            ]

        geom = self._get_ion_geometry_summary()

        result = {
            "stage": stage_label,
            "potential_energy_initial_kj_mol": float(
                e0.value_in_unit(unit.kilojoule_per_mole)
            ),
            "cycle_log": cycle_log,
            "potential_energy_final_kj_mol": float(
                ef.value_in_unit(unit.kilojoule_per_mole)
            ),
            "ion_distances_A": geom["distances_A"],
            "ion_pair_angles_deg": geom["angles_deg"],
            "final_positions_A": statef.getPositions(asNumpy=True).value_in_unit(
                unit.angstrom
            ),
            "n_cycles": int(n_cycles),
            "fixed_target_dist_A": list(self.fixed_target_dist_A or []),
            "final_md_steps": int(final_md_steps),
            "ion_reposition_fraction": float(reposition_fraction),
            "k_ang_start": float(k_ang_start),
            "k_ang_end": float(k_ang_end),
        }

        self.optimisation_result = result
        return result
    
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

        donor_atoms = self._get_current_spec_atoms_in_selected_structure()
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
    
    
    def final_map_recovery(
        self,
        include_angles=True,
        flat_bottom_A=0.0,
        k_dist=500.0,
        k_ang=None,
        k_pin=5000.0,
        k_map=450.0,
        map_pad_A=6.0,
        map_smooth_sigma_A=0.75,
        map_smooth_sigma_vox=0.0,
        map_normalise=True,
        minimization_tolerance=0.5,
        minimization_max_iterations=1500,
        temperature_K=50.0,
        friction_per_ps=1.0,
        step_size_ps=0.002,
        platform_name=None,
        platform_properties=None,
        random_seed=None,
    ):
        """
        Rebuild the refinement system from the current coordinates and perform a
        final density-guided minimization with reduced ion-restraint strength.

        This stage is intended for already-refined protein/ligand models where
        the coordination shell has largely been corrected, but the ligand still
        needs to be pulled back into the density.
        """
        self._selected_structure_exists()

        if k_ang is None:
            k_ang = _default_k_ang_for_cn(self.coordination_number)

        target_dist_A = list(self.fixed_target_dist_A or [])
        if not target_dist_A:
            if self.coord_atom_indices is None or self.ion_atom is None:
                self.identify_ion_angle_restraint_atoms()
            n_coord = len(self.coord_atom_indices)
            target_dist_A = list(_default_target_distances(
                self.system.options.ion_type, n_coord
            ))
            print(f"[DEBUG map_recovery] target_dist_A from defaults: {target_dist_A}")

        prep = self._prepare_refinement_system_from_current_structure(
            include_angles=include_angles,
            target_dist_A=target_dist_A,
            flat_bottom_A=flat_bottom_A,
            k_dist=k_dist,
            k_ang=k_ang,
            k_pin=k_pin,
            k_pin_excluded=max(float(k_pin) * 5.0, float(k_pin)),
            k_map=k_map,
            map_pad_A=map_pad_A,
            map_smooth_sigma_A=map_smooth_sigma_A,
            map_smooth_sigma_vox=map_smooth_sigma_vox,
            map_normalise=map_normalise,
            temperature_K=temperature_K,
            friction_per_ps=friction_per_ps,
            step_size_ps=step_size_ps,
            platform_name=platform_name,
            platform_properties=platform_properties,
            random_seed=random_seed,
        )

        state0 = self.simulation.context.getState(getEnergy=True)
        e0 = state0.getPotentialEnergy()

        self.simulation.minimizeEnergy(
            tolerance=float(minimization_tolerance)
            * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=int(minimization_max_iterations),
        )

        statef = self._sync_selected_structure_from_context()
        ef = statef.getPotentialEnergy()

        self.ion_atom = self._find_ion_atom_in_selected_structure()
        if self.coord_atom_indices is not None:
            self.coord_atoms_ordered = [
                self._get_atom_by_idx(i) for i in self.coord_atom_indices
            ]

        geom = self._get_ion_geometry_summary()

        result = {
            "applied": True,
            "map_applied": bool(prep["map_applied"]),
            "target_dist_A": list(target_dist_A),
            "k_dist": float(k_dist),
            "k_ang": float(k_ang),
            "k_pin": float(k_pin),
            "k_map": float(k_map),
            "map_pad_A": float(map_pad_A),
            "map_smooth_sigma_A": float(map_smooth_sigma_A),
            "map_smooth_sigma_vox": float(map_smooth_sigma_vox),
            "potential_energy_initial_kj_mol": float(
                e0.value_in_unit(unit.kilojoule_per_mole)
            ),
            "potential_energy_final_kj_mol": float(
                ef.value_in_unit(unit.kilojoule_per_mole)
            ),
            "ion_distances_A": geom["distances_A"],
            "ion_pair_angles_deg": geom["angles_deg"],
            "final_positions_A": statef.getPositions(asNumpy=True).value_in_unit(
                unit.angstrom
            ),
        }
        self.final_map_recovery_result = result
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

    def export_refinement_outputs(self, refinement_dir=None, keep_ids=True):
        """
        Write ion-fixer refinement outputs to a stable directory structure:
          - refined_protein_no_ligands.pdb  (protein + refined ion)
          - refined_protein_with_ligands.pdb (protein + ligands + refined ion)
          - ligand_{index:02d}_{name}.sdf    (one per ligand)
        """
        self._complex_structure_exists()
        self._selected_structure_exists()

        if refinement_dir is None:
            refinement_dir = os.path.join(self.system.output, "ion_fixer")
        os.makedirs(refinement_dir, exist_ok=True)

        # 1) Sync selected refined coords into the full complex and include refined ion.
        complex_update_info = self.update_original_complex_with_refined_positions(add_ion=True)

        # 2) Propagate full-complex refined coords back into ligand structures/mols.
        ligand_complex_updates = self._update_ligand_complex_structures_from_refined_complex()
        ligand_mol_updates = self._update_ligand_mols_from_complex_structures()

        # 3) Build and write the "with ligands" full refined protein model.
        protein_with_ligands = copy.deepcopy(self.complex_structure)
        self._sync_structure_coordinate_arrays(protein_with_ligands)
        with_ligands_pdb = os.path.join(refinement_dir, "refined_protein_with_ligands.pdb")
        with open(with_ligands_pdb, "w") as f:
            app.PDBFile.writeFile(
                protein_with_ligands.topology,
                self._structure_positions_from_atoms(protein_with_ligands),
                f,
                keepIds=keep_ids,
            )

        # 4) Build and write the "no ligands" full refined protein model (still with ion).
        protein_no_ligands = copy.deepcopy(self.system.protein.complex_structure)
        self._copy_matching_atom_positions(
            source_structure=self.complex_structure,
            target_structure=protein_no_ligands,
        )
        self._update_or_append_refined_ion(protein_no_ligands)
        no_ligands_pdb = os.path.join(refinement_dir, "refined_protein_no_ligands.pdb")
        with open(no_ligands_pdb, "w") as f:
            app.PDBFile.writeFile(
                protein_no_ligands.topology,
                self._structure_positions_from_atoms(protein_no_ligands),
                f,
                keepIds=keep_ids,
            )

        # 5) Write one refined SDF per ligand.
        ligand_sdf_paths = []
        for i, lig in enumerate(self.system.ligand, start=1):
            mol = getattr(lig, "mol", None)
            if mol is None:
                continue

            lig_name = getattr(lig, "ligand_id", None)
            lig_label = self._safe_name(lig_name, fallback=f"ligand_{i:02d}")
            sdf_name = f"ligand_{i:02d}_{lig_label}.sdf"
            sdf_path = os.path.join(refinement_dir, sdf_name)

            writer = Chem.SDWriter(sdf_path)
            writer.write(mol)
            writer.close()

            ligand_sdf_paths.append(sdf_path)

        outputs = {
            "refinement_dir": refinement_dir,
            "protein_no_ligands_pdb": no_ligands_pdb,
            "protein_with_ligands_pdb": with_ligands_pdb,
            "ligand_sdfs": ligand_sdf_paths,
            "complex_update": complex_update_info,
            "ligand_complex_updates": ligand_complex_updates,
            "ligand_mol_updates": ligand_mol_updates,
        }

        self.refinement_outputs = outputs
        return outputs
    
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
    
    def should_apply_map_restraint(self):
        return (
            getattr(self.system, "density", None) is not None
            and not bool(getattr(self.system.options, "no_map", False))
        )
    
    
    
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

        Distance targets are fixed for the whole run unless the caller
        explicitly rebuilds the restraint force.
        """
        self._openmm_system_exists()

        if self.coord_atom_indices is None or self.ion_atom is None:
            self.identify_ion_angle_restraint_atoms()

        n_coord = len(self.coord_atom_indices)
        if target_dist_A is None:
            target_dist_A = list(_default_target_distances(
                self.system.options.ion_type, n_coord
            ))
            print(f"[DEBUG apply_restraints] target_dist_A from defaults: {target_dist_A}")
        elif np.isscalar(target_dist_A):
            target_dist_A = [float(target_dist_A)] * n_coord
        else:
            target_dist_A = [float(x) for x in target_dist_A]

        if len(target_dist_A) != n_coord:
            raise RuntimeError(
                f"[ERROR] target_dist_A must contain {n_coord} values for "
                f"coordination geometry {self.system.options.coordination_geometry!r}, "
                f"got {len(target_dist_A)}"
            )

        self.fixed_target_dist_A = list(target_dist_A)

        restraints = ForceBuilder.create_ion_coordination_restraints(
            ion_idx=int(self.ion_atom.idx),
            coord_atom_indices=self.coord_atom_indices,
            target_dist_A=self.fixed_target_dist_A,
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

    def identify_atoms_to_pin(self, tol=1e-3):
        """
        Pin all non-coordinating protein/nucleotide heavy atoms.

        This keeps the already-refined macromolecule fixed while allowing only
        the exact coordinating atoms to adjust locally. Explicitly excluded
        atoms are always pinned, including ligand atoms.
        """
        self._spec_atoms_exist()
        self._selected_structure_exists()

        coord_atoms = self._get_current_spec_atoms_in_selected_structure()
        coord_idx_set = {int(a.idx) for a in coord_atoms}

        excluded_spec_atoms = self._find_excluded_spec_atoms_in_selected_structure(tol=tol)
        excluded_spec_idx_set = {int(a.idx) for a in excluded_spec_atoms}

        added_water_resnums = {int(x) for x in self.added_water_resnums}

        pin_atoms = []
        seen = set()

        for res in self.selected_structure.residues:
            resnum = int(res.number)

            if self.added_ion_resnum is not None and resnum == int(self.added_ion_resnum):
                continue
            if resnum in added_water_resnums:
                continue

            resname = str(res.name).upper()
            is_supported_polymer = (
                self._is_protein_residue_name(resname)
                or self._is_nucleotide_residue_name(resname)
            )

            for atom in res.atoms:
                if not self._is_heavy_atom(atom):
                    continue

                idx = int(atom.idx)

                if idx in excluded_spec_idx_set:
                    if idx not in seen:
                        seen.add(idx)
                        pin_atoms.append(atom)
                    continue

                if not is_supported_polymer:
                    continue

                if idx in coord_idx_set:
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
            "coordination_atoms_left_mobile": [int(a.idx) for a in coord_atoms],
        }

    def apply_pin_restraints(
        self,
        k_pin=1000.0,
        k_pin_excluded=None,
        k_pin_name="k_pin",
        k_pin_excluded_name="k_pin_excluded",
        force_group=22,
    ):
        """
        Apply positional pin restraints using ForceBuilder.create_positional_pin().

        Excluded-spec atoms get their own stronger force with a separate global
        parameter (k_pin_excluded) so they remain effectively fixed throughout
        refinement.
        """
        self._openmm_system_exists()

        if self.pin_atom_indices is None:
            self.identify_atoms_to_pin()

        if not self.pin_atom_indices:
            raise RuntimeError("[ERROR] No atoms identified for pin restraints.")

        if k_pin_excluded is None:
            k_pin_excluded = max(float(k_pin) * 5.0, float(k_pin))

        excluded_spec_idx_set = set()
        if hasattr(self, "exclude_atom_specs") and self.exclude_atom_specs:
            excluded_atoms = self._find_excluded_spec_atoms_in_selected_structure()
            excluded_spec_idx_set = {int(a.idx) for a in excluded_atoms}

        regular_pin_indices = [i for i in self.pin_atom_indices if i not in excluded_spec_idx_set]
        excluded_pin_indices = [i for i in self.pin_atom_indices if i in excluded_spec_idx_set]

        max_idx = max(int(a.idx) for a in self.selected_structure.atoms)
        ref_positions_nm = np.zeros((max_idx + 1, 3), dtype=float)

        for atom in self.pin_atoms:
            ref_positions_nm[int(atom.idx)] = np.array([atom.xx, atom.xy, atom.xz], dtype=float) * 0.1

        if regular_pin_indices:
            pin_force = ForceBuilder.create_positional_pin(
                atom_indices=regular_pin_indices,
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
        else:
            self.pin_force = None

        self.excluded_pin_force = None
        if excluded_pin_indices:
            excl_pin_force = ForceBuilder.create_positional_pin(
                atom_indices=excluded_pin_indices,
                ref_positions_nm=ref_positions_nm,
                k_name=k_pin_excluded_name,
            )
            excl_pin_force.setForceGroup(int(force_group))
            self._maybe_set_global_parameter_default(
                excl_pin_force,
                k_pin_excluded_name,
                float(k_pin_excluded) * unit.kilojoule_per_mole / unit.nanometer**2,
            )
            self.openmm_system.addForce(excl_pin_force)
            self.excluded_pin_force = excl_pin_force

        return self.pin_force

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

    def identify_map_restrained_atoms(
        self,
        include_protein_donors=True,
        include_all_ligand_heavy_atoms=True,
    ):
        """
        Use a focused map restraint selection:
          - the added ion
          - exact coordinating atoms
          - ligand / non-polymer heavy atoms

        Dummy waters are always excluded. This prevents the map term from
        fighting the coordination correction across the whole local protein.
        """
        self._selected_structure_exists()

        added_water_resnums = {int(x) for x in self.added_water_resnums}
        ion_idx = int(self.added_ion_atom_idx) if self.added_ion_atom_idx is not None else None
        coord_idx_set = set(int(i) for i in (self.coord_atom_indices or []))

        map_atoms = []
        seen = set()

        for atom in self.selected_structure.atoms:
            if not self._is_heavy_atom(atom):
                continue

            resnum = int(atom.residue.number)
            if resnum in added_water_resnums:
                continue

            idx = int(atom.idx)
            resname = str(atom.residue.name).upper()
            is_supported_polymer = (
                self._is_protein_residue_name(resname)
                or self._is_nucleotide_residue_name(resname)
            )

            keep = False
            if ion_idx is not None and idx == ion_idx:
                keep = True
            elif idx in coord_idx_set and (include_protein_donors or not is_supported_polymer):
                keep = True
            elif include_all_ligand_heavy_atoms and not is_supported_polymer:
                keep = True

            if keep and idx not in seen:
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
    
        simulation = make_openmm_simulation(
            topology,
            self.openmm_system,
            integrator,
            platform_name=platform_name,
            source=getattr(self, "system", None),
            platform_properties=platform_properties,
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

    def _recompute_ion_position(self, spec_positions_A):
        """
        Recompute optimal ion position given updated spec atom coordinates.

        The current ion, exact coordinating atoms, and dummy waters are not
        treated as obstacles. Fixed target distances are reused if available.

        Returns
        -------
        np.ndarray, shape (3,)
            New ion position in Angstroms.
        """
        state = self.simulation.context.getState(getPositions=True)
        all_pos_A = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

        coord_idx_set = set(int(i) for i in (self.coord_atom_indices or []))
        ion_idx = int(self.added_ion_atom_idx) if self.added_ion_atom_idx is not None else None
        added_water_resnums = {int(x) for x in self.added_water_resnums}

        obstacle_xyz = []
        obstacle_elements = []
        for atom in self.selected_structure.atoms:
            elem = atom.element_name.upper()
            if elem == "H":
                continue

            idx = int(atom.idx)
            if ion_idx is not None and idx == ion_idx:
                continue
            if idx in coord_idx_set:
                continue
            if int(atom.residue.number) in added_water_resnums:
                continue

            obstacle_xyz.append(all_pos_A[idx])
            obstacle_elements.append(elem)

        obstacle_xyz = np.asarray(obstacle_xyz, dtype=float) if obstacle_xyz else np.zeros((0, 3))

        # spec_weights covers the original spec atoms; pad with 1.0 for
        # any additional coordinating atoms (dummy waters).
        n_coord = len(spec_positions_A)
        base_weights = [0.1 if is_lig else 1.0 for is_lig in self.spec_atom_is_ligand]
        spec_weights = base_weights + [1.0] * (n_coord - len(base_weights))

        return generate_initial_ion_position(
            spec_xyz=spec_positions_A,
            obstacle_xyz=obstacle_xyz,
            obstacle_elements=obstacle_elements,
            ion_name=self.system.options.ion_type,
            target_dists=self.fixed_target_dist_A,
            spec_weights=spec_weights,
        )

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

    def _get_current_spec_atoms_in_selected_structure(self):
        if self.spec_atom_indices_selected is None:
            self.cache_spec_atom_indices_in_selected_structure()
        return [self._get_atom_by_idx(i) for i in self.spec_atom_indices_selected]

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

    def _reset_runtime_forces_and_simulation(self):
        """
        Clear OpenMM runtime objects so the current selected_structure coordinates
        can be used to rebuild a fresh restraint system for the next phase.
        """
        self.openmm_system = None
        self.ion_restraint_forces = None
        self.pin_force = None
        self.excluded_pin_force = None
        self.local_density_map = None
        self.map_force = None
        self.map_atoms = None
        self.map_atom_indices = None
        self.simulation = None
        self.integrator = None

    def _find_excluded_spec_atoms_in_selected_structure(self, tol=1e-3):
        """
        Return the excluded atoms resolved once onto selected_structure and then
        tracked by stable selected_structure atom indices even after movement.
        """
        self._selected_structure_exists()
        return self._get_current_excluded_atoms_in_selected_structure()

    def _get_current_excluded_atoms_in_selected_structure(self):
        if self.exclude_atom_indices_selected is None:
            self.cache_spec_atom_indices_in_selected_structure()
        return [self._get_atom_by_idx(i) for i in self.exclude_atom_indices_selected]

    def _is_heavy_atom(self, atom):
        elem = str(getattr(atom, "element_name", "")).upper()
        if elem:
            return elem != "H"
        return not str(atom.name).upper().startswith("H")

    def _is_protein_residue_name(self,resname):
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

    def _is_nucleotide_residue_name(self, resname):
        nucleotide_names = {
            "A", "C", "G", "U", "I",
            "DA", "DC", "DG", "DT", "DI",
            "ADE", "CYT", "GUA", "URA", "THY",
            "RA", "RC", "RG", "RU",
        }
        return resname in nucleotide_names

    def _maybe_set_global_parameter_default(self, force, param_name, value):
        for i in range(force.getNumGlobalParameters()):
            if force.getGlobalParameterName(i) == param_name:
                force.setGlobalParameterDefaultValue(i, value)
                return True
        return False

    def _set_global_parameter_default(self, force, param_name, value):
        for i in range(force.getNumGlobalParameters()):
            if force.getGlobalParameterName(i) == param_name:
                force.setGlobalParameterDefaultValue(i, value)
                return
        raise RuntimeError(f"[ERROR] Force does not contain global parameter {param_name!r}")

    def run(self):
        self.get_spec_atoms()
        self.get_coordination_number()
        self.create_complex_structure()
        self.get_residue_selection()
        
        self.get_initial_position()
        self.get_paramitised_ion()
        self.get_paramitised_waters()
        self.merge_system()
        self.cache_spec_atom_indices_in_selected_structure()
        
        self.setup_constraints()
        #refactor once you know the surface area
        result = None
        distance_only_result = None
        angle_map_result = None
        all_cycle_log = []
        
        distance_only_k_dist_end_fraction=1.0
        k_dist_start=500.0
        k_dist_end=5000.0
        k_map=150.0
        distance_only_k_map_scale=0.5
        step_size_ps=0.002
        friction_per_ps = 1.0
        final_md_temperature_K=50.0
        map_normalise=True
        map_smooth_sigma_A=0.0,
        map_smooth_sigma_vox=0.0
        map_pad_A=4.0
        k_pin=5000.0
        k_dist = 1000.0
        target_dist_A=None
        include_angles=True
        angle_ramp_k_map_scale=1.0
        final_md_steps=0
        do_final_map_recovery=True
        final_map_recovery_map_pad_A=None
        final_map_recovery_map_smooth_sigma_A=None
        final_map_recovery_map_smooth_sigma_vox =None
        final_map_recovery_k_dist_scale=0.5
        final_map_recovery_k_ang_scale=0.5
        final_map_recovery_k_pin_scale=1.0
        final_map_recovery_k_map_scale=3.0
        final_map_recovery_map_pad_A=None
        final_map_recovery_map_smooth_sigma_A=None
        final_map_recovery_map_smooth_sigma_vox=None
        final_map_recovery_minimization_tolerance=0.5
        final_map_recovery_minimization_max_iterations=1500
    
   
        stage1_k_dist_end = float(k_dist_start) + float(distance_only_k_dist_end_fraction) * (
            float(k_dist_end) - float(k_dist_start)
        )
        
        stage1_k_dist_end = min(stage1_k_dist_end, float(k_dist_end))
        
        stage1_k_map = max(float(k_map) * float(distance_only_k_map_scale), 0.0)
        
        self._prepare_refinement_system_from_current_structure(
            include_angles=include_angles,
            target_dist_A=target_dist_A,
            k_dist=k_dist,
            k_ang=0.0,
            k_pin=k_pin,
            k_pin_excluded=max(float(k_pin) * 5.0, float(k_pin)),
            k_map=stage1_k_map,
            map_pad_A=map_pad_A,
            map_smooth_sigma_A=map_smooth_sigma_A,
            map_smooth_sigma_vox=map_smooth_sigma_vox,
            map_normalise=map_normalise,
            temperature_K=final_md_temperature_K,
            friction_per_ps=friction_per_ps,
            step_size_ps=step_size_ps,
            platform_name=self.system.platform,
            
        )
        
        
        print('[DEBUG] fixed_target_dist_A', self.fixed_target_dist_A)
        distance_only_result = self.iterative_minimize_ion_geometry(
            n_cycles=self.early_cycles,
            k_dist_start=k_dist_start,
            k_dist_end=stage1_k_dist_end,
            k_ang=self.k_ang,
            k_ang_start=0.0,
            k_ang_end=0.0,
            k_pin=k_pin,
            final_md_steps=0,
            final_md_temperature_K=final_md_temperature_K,
            platform_name=self.system.platform,
            ion_reposition_fraction=1.0,
            stage_label="distance_only",
        )
        
        all_cycle_log.extend(distance_only_result["cycle_log"])
        print(f"[DEBUG distance_only DONE] final dists={distance_only_result['ion_distances_A']}")
        print(f"[DEBUG distance_only DONE] targets={self.fixed_target_dist_A}")

        if self.late_cycles > 0:
            stage2_k_map = max(float(k_map) * float(angle_ramp_k_map_scale), 0.0)
            
            self._prepare_refinement_system_from_current_structure(
                include_angles=include_angles,
                target_dist_A=target_dist_A,
                k_dist=stage1_k_dist_end,
                k_ang=self.k_ang,
                k_pin=k_pin,
                k_pin_excluded=max(float(k_pin) * 5.0, float(k_pin)),
                k_map=stage2_k_map,
                map_pad_A=map_pad_A,
                map_smooth_sigma_A=map_smooth_sigma_A,
                map_smooth_sigma_vox=map_smooth_sigma_vox,
                map_normalise=map_normalise,
                temperature_K=final_md_temperature_K,
                friction_per_ps=1.0,
                step_size_ps=0.002,
                platform_name=self.system.platform
            )
            
            angle_map_result = self.iterative_minimize_ion_geometry(
                n_cycles=self.late_cycles,
                k_dist_start=stage1_k_dist_end,
                k_dist_end=k_dist_end,
                k_ang=self.k_ang,
                k_ang_start=0.0,
                k_ang_end=self.k_ang,
                k_pin=k_pin,
                final_md_steps=final_md_steps,
                final_md_temperature_K=final_md_temperature_K,
                platform_name=self.system.platform,
                ion_reposition_fraction=0.0,
                stage_label="angle_map_ramp",
            )
            all_cycle_log.extend(angle_map_result["cycle_log"])
            print(f"[DEBUG angle_map DONE] final dists={angle_map_result['ion_distances_A']}")
            print(f"[DEBUG angle_map DONE] targets={self.fixed_target_dist_A}")
            terminal_result = angle_map_result
        else:
            terminal_result = distance_only_result
        
        result = {
            "staged_refinement": True,
            "distance_only_stage": distance_only_result,
            "angle_map_stage": angle_map_result,
            "cycle_log": all_cycle_log,
            "potential_energy_initial_kj_mol": float(distance_only_result["potential_energy_initial_kj_mol"]),
            "potential_energy_final_kj_mol": float(terminal_result["potential_energy_final_kj_mol"]),
            "ion_distances_A": terminal_result["ion_distances_A"],
            "ion_pair_angles_deg": terminal_result["ion_pair_angles_deg"],
            "final_positions_A": terminal_result["final_positions_A"],
            "n_cycles": int(self.total_cycles),
            "fixed_target_dist_A": list(self.fixed_target_dist_A or []),
            "final_md_steps": int(final_md_steps),
            "distance_only_fraction": float(self.distance_only_fraction),
            "distance_only_k_map_scale": float(distance_only_k_map_scale),
            "angle_ramp_k_map_scale": float(angle_ramp_k_map_scale),
        }
        
        #final thing
        if do_final_map_recovery and self.should_apply_map_restraint():
            recovery_map_pad_A = (
                max(float(map_pad_A), 6.0)
                if final_map_recovery_map_pad_A is None
                else float(final_map_recovery_map_pad_A)
            )
            recovery_sigma_A = (
                max(float(map_smooth_sigma_A), 0.75)
                if final_map_recovery_map_smooth_sigma_A is None
                else float(final_map_recovery_map_smooth_sigma_A)
            )
            recovery_sigma_vox = (
                float(map_smooth_sigma_vox)
                if final_map_recovery_map_smooth_sigma_vox is None
                else float(final_map_recovery_map_smooth_sigma_vox)
            )
            
            
            recovery_result = self.final_map_recovery(
                include_angles=include_angles,
                k_dist=max(float(k_dist_end) * float(final_map_recovery_k_dist_scale), 1.0e-6),
                k_ang=max(float(self.k_ang) * float(final_map_recovery_k_ang_scale), 1.0e-6),
                k_pin=max(float(k_pin) * float(final_map_recovery_k_pin_scale), 1.0e-6),
                k_map=max(float(k_map) * float(final_map_recovery_k_map_scale), 1.0e-6),
                map_pad_A=recovery_map_pad_A,
                map_smooth_sigma_A=recovery_sigma_A,
                map_smooth_sigma_vox=recovery_sigma_vox,
                map_normalise=map_normalise,
                minimization_tolerance=final_map_recovery_minimization_tolerance,
                minimization_max_iterations=final_map_recovery_minimization_max_iterations,
                temperature_K=final_md_temperature_K,
                platform_name=self.system.platform,
            )
            result["final_map_recovery"] = recovery_result
        
        else:
            result["final_map_recovery"] = {
                "applied": False,
                "reason": "disabled" if not do_final_map_recovery else "no_map_restraint",
            }
        
        
        refinement_outputs = self.export_refinement_outputs()
        result["refinement_outputs"] = refinement_outputs
        print(f"Wrote ion-fixer refinement outputs to: {refinement_outputs['refinement_dir']}")

        return result
