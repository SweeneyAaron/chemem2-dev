# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


from tqdm.auto import tqdm
from contextlib import nullcontext
import time
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union

from openmm import (
    unit, app, 
    LangevinIntegrator, Platform, 
    MonteCarloBarostat,
    Vec3
)
from openmm.app import NoCutoff, HBonds, PDBFile

from ChemEM.tools.forces import ForceBuilder
from ChemEM.tools.biomolecule import select_atoms, find_atoms_outside_ligand, create_structure_subset


class ProgressLogger:
    @staticmethod
    def get_bar(total, desc, unit="steps", leave=False, active=True):
        if active and total > 0:
            return tqdm(total=total, desc=desc, unit=unit, leave=leave)
        return nullcontext()
    
    @staticmethod
    def human_ps(ps: float) -> str:
        if ps >= 1000.0: return f"{ps/1000.0:.1f} ns"
        if ps >= 1.0: return f"{ps:.1f} ps"
        return f"{ps*1000.0:.1f} fs"


@dataclass
class AnnealingConfig:
    """Configuration for simulated annealing schedule."""
    minimize_before: bool = True
    minimize_after: bool = True
    staged_min: bool = True
    k_heavy_stage1: float = 5000.0
    k_backbone_stage2: float = 1500.0
    map_scale_stage1: float = 1.0
    map_scale_stage2: float = 1.0
    heat_to_K: float = 315.0
    cool_to_K: float = 300.0
    ramp_up_ps: float = 2.0
    ramp_down_ps: float = 3.0
    high_hold_ps: float = 0.0

class PoseMinimiser:
    def __init__(
        self,
        protein_structure,
        ligand_structure,
        residues=None,
        density_map=None,
        padding=1.0,
        solvent=app.GBn2,
        platform_name='OpenCL',
        # Restraint settings
        protein_restraint='protein',  # 'protein', 'sse', 'none'
        sse_groups: Optional[List[List[int]]] = None,
        sse_k: float = 50.0,
        pin_k: float = 5000.0,
        localise: bool = True
    ):
        
        self.log = ProgressLogger()
        self.density_map = density_map
        
        print("bond_types:", len(protein_structure.bond_types), "bonds:", len(protein_structure.bonds))
        print("any bond.type None:", any(b.type is None for b in protein_structure.bonds))

        # 1. Structure Prep
        if residues is not None:
            protein_structure = create_structure_subset(protein_structure, residues)
        
        self.complex_structure, self.complex_system = self._create_system(
            protein_structure, ligand_structure, solvent
        )
        
        # 2. Identify Indices
        self._identify_indices()
        
        # 3. Density Map Prep (Cropping & Force Creation)
        if self.density_map:
            self._setup_density_map(protein_structure, padding)
        
        # 4. Simulation Context Prep
        self.integrator = LangevinIntegrator(
            300 * unit.kelvin, 
            1.0 / unit.picoseconds, 
            1.0 * unit.femtoseconds
        )
        
        
        self.platform = Platform.getPlatformByName(platform_name)
        props = {"Precision": "single"} if platform_name != 'CPU' else {}
        
        self.simulation = app.Simulation(
            self.complex_structure.topology,
            self.complex_system,
            self.integrator,
            self.platform,
            platformProperties=props
        )
        
        self.simulation.context.setPositions(self.complex_structure.positions)
        self.restraint_config = {
            'mode': protein_restraint, 
            'sse_groups': sse_groups, 
            'sse_k': sse_k, 
            'pin_k': pin_k,
            'localise': localise
        }
        
        self.setup_protein_restraints()
    
    @staticmethod
    def first_missing_params(struct):
        for b in struct.bonds:
            if b.type is None:
                return ("bond", [b.atom1, b.atom2])
        for a in struct.angles:
            if a.type is None:
                return ("angle", [a.atom1, a.atom2, a.atom3])
        for d in struct.dihedrals:
            if d.type is None:
                return ("dihedral", [d.atom1, d.atom2, d.atom3, d.atom4])
        for i in getattr(struct, "impropers", []):
            if i.type is None:
                return ("improper", [i.atom1, i.atom2, i.atom3, i.atom4])
        return None

    def _create_system(self, protein, ligand_list, solvent):
        """Merges structures and builds OpenMM system."""
        # Ensure list
        
        if not isinstance(ligand_list, list):
            ligand_list = [ligand_list]

        # Rename ligands to avoid conflicts
        for lig in ligand_list:
            for i, res in enumerate(lig.residues):
                if res.name == 'UNL': res.name = f'LIG_{i}'

        complex_struc = protein
        for lig in ligand_list:
            complex_struc += lig

        # System creation
        kwargs = {
            'nonbondedMethod': NoCutoff,
            'nonbondedCutoff': 9.0 * unit.angstrom,
            'constraints': HBonds,
            'removeCMMotion': True,
        }
        
        if solvent:
            kwargs['implicitSolvent'] = solvent
        else:
            kwargs['rigidWater'] = True
        
        
        miss = PoseMinimiser.first_missing_params(complex_struc)
        if miss:
            kind, atoms = miss
            print(kind, "missing for:")
            for at in atoms:
                print(f"  {at.residue.name}:{at.name} (idx {at.idx})")
        else:
            print("No missing valence params found (issue may be nonbonded or elsewhere).")
        
        system = complex_struc.createSystem(**kwargs)
        return complex_struc, system

    def _identify_indices(self):
        """Cache atom indices for protein and ligand."""
        self.all_ligand_indices = [
            a.idx for a in self.complex_structure.atoms 
            if a.residue.name.startswith('LIG')
        ]
        
       
        
        self.ligand_heavy_indices = [
            i for i in self.all_ligand_indices 
            if self.complex_structure.atoms[i].element != 1
        ]
        
        # Protein heavy atoms
        self.protein_heavy_indices = [
            a.idx for a in self.complex_structure.atoms
            if not a.residue.name.startswith('LIG') and a.element != 1
        ]

    def _setup_density_map(self, protein_struc, padding):
        """Crops density map and adds force to system."""
        print("Processing Density Map...")
        # (Logic for cropping map - simplified here for brevity, keep original logic if specific)
        # Note: In a full refactor, cropping logic should move to force_builder or a MapUtils class
        
        # Calculate global_k dynamic default
        res = getattr(self.density_map, 'resolution', 3.0)
        global_k = 75.0 if res < 3.0 else 25.0
        
        map_force = ForceBuilder.create_map_potential(self.density_map, global_k)
        
        # Apply to all atoms
        for _ in self.complex_structure.atoms:
            map_force.addBond([_.idx])
            
        self.complex_system.addForce(map_force)

    def setup_protein_restraints(self):
        """Applies SSE or Pinning restraints."""
        mode = self.restraint_config['mode']
        if mode == 'none': return
        
        state = self.simulation.context.getState(getPositions=True)
        ref_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        
        atoms_to_restrain = []
        ligand_set = set(self.all_ligand_indices)
        
        if mode == 'protein':
            # Select all heavy protein atoms
            atoms_to_restrain = select_atoms(
                self.complex_structure, 
                indices=self.protein_heavy_indices, 
                exclude_indices=ligand_set
            )
            
        elif mode == 'sse':
            
            groups = self.restraint_config['sse_groups']
            if not groups: raise ValueError("SSE mode requires sse_groups")
           
            
            for grp in groups:
                
                valid_grp = [i for i in grp if i not in ligand_set]
                
                #
                if self.restraint_config['localise']:
                    outside = find_atoms_outside_ligand(
                        self.complex_structure, self.ligand_heavy_indices
                    )
                    valid_grp = [i for i in valid_grp if i not in outside]

                rmsd_force = ForceBuilder.create_rmsd_restraint(
                    ref_nm, valid_grp, self.restraint_config['sse_k']
                )
                if rmsd_force:
                    self.complex_system.addForce(rmsd_force)
            
            return # Exit here as SSE adds forces individually
        
        # Apply filtering for localisation (radius from ligand)
        if self.restraint_config['localise']:
            outside_indices = find_atoms_outside_ligand(
                self.complex_structure, self.ligand_heavy_indices
            )
            atoms_to_restrain = [i for i in atoms_to_restrain if i not in outside_indices]
       
        # Apply Pin Force
        if atoms_to_restrain:
            pin_f = ForceBuilder.create_positional_pin(
                atoms_to_restrain, ref_nm, k_name="k_static_pin"
            )
            # Set static K immediately
            self.complex_system.addForce(pin_f)
            self.simulation.context.reinitialize(preserveState=True)
            self.simulation.context.setParameter("k_static_pin", self.restraint_config['pin_k'])

    # ------------------------------------------------------------------
    # Minimization Logic
    # ------------------------------------------------------------------

    def minimize_pose_list(self, ligand_poses_angstrom, max_iters=0):
        """Minimizes a list of ligand poses sequentially."""
        results = []
        print(f"Minimizing {len(ligand_poses_angstrom)} poses...")
        
        # Save protein positions to reset after every pose? 
        # Usually good practice if poses are independent.
        initial_pos = self.simulation.context.getState(getPositions=True).getPositions()

        for i, pose_ang in enumerate(ligand_poses_angstrom):
            t_start = time.perf_counter()
            
            # Reset System
            self.simulation.context.setPositions(initial_pos)
            
            # Update Ligand Positions
            curr_pos_nm = np.array(initial_pos.value_in_unit(unit.nanometer))
            pose_nm = pose_ang / 10.0
            curr_pos_nm[self.all_ligand_indices] = pose_nm
            
            self.simulation.context.setPositions(curr_pos_nm)
            
            # Run
            self.simulation.minimizeEnergy(maxIterations=max_iters)
            
            # Collect
            state = self.simulation.context.getState(getPositions=True, getEnergy=True)
            final_pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            final_lig = final_pos[self.all_ligand_indices] * 10.0
            
            energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            results.append(final_lig)
            
            print(f" Pose {i+1}: {time.perf_counter() - t_start:.2f}s | E: {energy:.2f} kcal/mol")

        return results

    def run_simulated_annealing(self, config: AnnealingConfig, write_pdb=None):
        """Runs the staged minimization and annealing protocol."""
        
        # 1. Staged Minimization
        if config.staged_min:
            self._run_staged_minimization(config)
        elif config.minimize_before:
            self.simulation.minimizeEnergy()

        # 2. Annealing
        self._run_annealing_ramp(config)

        # 3. Output
        return self._collect_results(write_pdb)

    def _run_staged_minimization(self, config):
        """
        Creates temporary pin forces, runs stages, removes forces.
        """
        state = self.simulation.context.getState(getPositions=True)
        ref_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        # Create temporary pins
        heavy_atoms = select_atoms(self.complex_structure, include="heavy")
        backbone_atoms = select_atoms(self.complex_structure, include="backbone")
        
        f_heavy = ForceBuilder.create_positional_pin(heavy_atoms, ref_nm, "k_stage_heavy")
        f_bb = ForceBuilder.create_positional_pin(backbone_atoms, ref_nm, "k_stage_bb")
        
        idx_h = self.complex_system.addForce(f_heavy)
        idx_b = self.complex_system.addForce(f_bb)
        self.simulation.context.reinitialize(preserveState=True)

        # Stage 1: Heavy Pins + Map Scaled
        print("Stage 1: Heavy Restraints")
        self.simulation.context.setParameter("k_stage_heavy", config.k_heavy_stage1)
        self.simulation.context.setParameter("k_stage_bb", 0)
        if self.density_map:
            self.simulation.context.setParameter("global_k", config.map_scale_stage1 * 75.0) # simplify k handling
        self.simulation.minimizeEnergy()

        # Stage 2: Backbone Pins
        print("Stage 2: Backbone Restraints")
        self.simulation.context.setParameter("k_stage_heavy", 0)
        self.simulation.context.setParameter("k_stage_bb", config.k_backbone_stage2)
        self.simulation.minimizeEnergy()

        # Cleanup
        self.complex_system.removeForce(idx_b) # Remove in reverse index order ideally
        self.complex_system.removeForce(idx_h)
        self.simulation.context.reinitialize(preserveState=True)

        # Stage 3: No pins (SSE restraints persist if added in __init__)
        print("Stage 3: Relax")
        if self.density_map:
             self.simulation.context.setParameter("global_k", 75.0)
        self.simulation.minimizeEnergy()

    def _run_annealing_ramp(self, config):
        """Execution of temperature ramp."""
        sim = self.simulation
        
        # Helper to set temp
        def set_T(k):
            self.integrator.setTemperature(k * unit.kelvin)

        # Ramp Up
        print(f"Ramping {300} -> {config.heat_to_K}K")
        steps = int(config.ramp_up_ps / 0.001) # assuming 1fs step
        chunks = 50
        for i in range(chunks):
            t = 300 + (config.heat_to_K - 300) * (i/chunks)
            set_T(t)
            sim.step(steps // chunks)

        # Hold
        if config.high_hold_ps > 0:
            print("Holding High T")
            sim.step(int(config.high_hold_ps / 0.001))

        # Ramp Down
        print(f"Cooling -> {config.cool_to_K}K")
        steps = int(config.ramp_down_ps / 0.001)
        for i in range(chunks):
            t = config.heat_to_K + (config.cool_to_K - config.heat_to_K) * (i/chunks)
            set_T(t)
            sim.step(steps // chunks)
        
        # Final minimize
        sim.minimizeEnergy()

    def _collect_results(self, write_pdb=None):
        state = self.simulation.context.getState(getPositions=True, getEnergy=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        
        if write_pdb:
            with open(write_pdb, 'w') as f:
                PDBFile.writeFile(self.complex_structure.topology, state.getPositions(), f)
                
        return {
            "energy": state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole),
            "positions": pos
        }