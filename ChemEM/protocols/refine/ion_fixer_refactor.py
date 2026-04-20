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
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .refine_utils import (create_structure_subset,
                           generate_initial_ion_position,
                           find_atom_from_spec_by_coord_and_element,
                           _default_k_ang_for_cn)

from ChemEM.protocols.core.forces import ForceBuilder
from ChemEM.protocols.core.core_utils import all_pairwise_distances_leq
from ChemEM.parsers.parametised_ions import (
    create_parameterized_ion_structure,
    propose_dummy_water_oxygen_positions,
    coord_geom_to_int,
    create_parametrised_tip3p_water,
)

from ChemEM.parsers.parse_forcefield import ff_load
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
        self.exclude_atom_specs = []

        atom_specs = list(getattr(self.system.options, "atom_specs", []) or [])
        exclude_specs = list(getattr(self.system.options, "exclude_specs", []) or [])

        if not atom_specs:
            raise RuntimeError("[ERROR] IonFixer requires at least one atom-spec argument")

        for spec in atom_specs:
            if spec.startswith("LIG"):
                self.spec_atoms.append(self.system.ligand.get_atom_idx_from_spec(spec))
            else:
                self.spec_atoms.append(self.system.protein.get_atom_idx_from_spec(spec))

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

        initial_pos = generate_initial_ion_position(
            spec_xyz=spec_xyz,
            obstacle_xyz=obstacle_xyz,
            obstacle_elements=obstacle_elements,
            ion_name=self.system.options.ion_type,
            target_dists=self.fixed_target_dist_A,
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
            target_dist_A = list(self._current_coordination_distances_A())

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
            target_dist_A = list(self._current_coordination_distances_A())

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
    
    def _get_atom_by_idx(self, atom_idx):
        for atom in self.selected_structure.atoms:
            if int(atom.idx) == int(atom_idx):
                return atom
        raise RuntimeError(f"[ERROR] Could not find atom with idx={atom_idx} in selected_structure")

    def _get_current_spec_atoms_in_selected_structure(self):
        if self.spec_atom_indices_selected is None:
            self.cache_spec_atom_indices_in_selected_structure()
        return [self._get_atom_by_idx(i) for i in self.spec_atom_indices_selected]

    def _get_current_excluded_atoms_in_selected_structure(self):
        if self.exclude_atom_indices_selected is None:
            self.cache_spec_atom_indices_in_selected_structure()
        return [self._get_atom_by_idx(i) for i in self.exclude_atom_indices_selected]
    

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
        k_dist_end=2000.0
        k_map=150.0
        distance_only_k_map_scale=0.35
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
        final_map_recovery_k_dist_scale=0.35
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
        
        
        self.write_output_pdb(
            filename=os.path.join(self.system.output, "ion_refinment.pdb"),
            use_context_positions=True,
        )



        
        
        
        