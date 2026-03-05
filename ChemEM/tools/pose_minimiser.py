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

from openmm import CustomExternalForce


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
    staged_min: bool = False
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
        localise: bool = True,
        global_k : float = 75.0,
        smooth_sigma_A : float = 1.0,
        do_biased_md : bool = True,
    ):
        
        self.log = ProgressLogger()
        
        self.density_map = density_map
        self.global_k = global_k
        self.smooth_sigma_A = smooth_sigma_A
        self.do_biased_md = do_biased_md
        
        self.md_pre_min_iters = 0
        self.md_ps = 5.0
        self.dt_ps = 0.001
        self.md_temp_K = 150.0

        self.md_report_ps = 5.0

        self.md_seed = 1
        
        
        if residues is not None:
            protein_structure = create_structure_subset(protein_structure, residues)
        
        self.complex_structure, self.complex_system = self._create_system(
            protein_structure, ligand_structure, solvent
        )
        
        self._identify_indices()
        
        if self.density_map:
            self._setup_density_map(protein_structure, padding)
        
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
       
        print('-- Global k:', self.global_k)
        map_force = ForceBuilder.create_map_potential(self.density_map, 
                                                      self.global_k,
                                                      smooth_sigma_vox=0.0) #smaller blur
                                                      #smooth_sigma_A = 0.5)
        
        # Apply to all atoms 
        #not hydrogens
        for atom in self.complex_structure.atoms:
            if atom.element != 1:
                map_force.addBond([atom.idx])
            
        self.complex_system.addForce(map_force)
    
    def _set_ligand_flatbottom_tether(self, ref_pos_nm, *, r0_A=0.1, k_kcal_per_mol_A2=0.2):
        """
        ref_pos_nm: full system positions in nm (N_atoms x 3)
        r0_A: flat bottom radius in Å
        k_kcal_per_mol_A2: spring outside r0, in kcal/mol/Å^2
        """
        self._ensure_ligand_flatbottom_tether()
    
        # Unit conversions:
        # r0: Å -> nm
        r0_nm = float(r0_A) * 0.1
    
        # k: kcal/mol/Å^2 -> kJ/mol/nm^2  (× 418.4)
        k_kj_per_mol_nm2 = float(k_kcal_per_mol_A2) * 418.4
    
        f = self._lig_fb_tether
        for p, atom_idx in enumerate(self._lig_fb_atoms):
            x0, y0, z0 = map(float, ref_pos_nm[int(atom_idx)])
            f.setParticleParameters(p, int(atom_idx), [x0, y0, z0])
    
        # Push updated per-particle params into the existing Context.
        # (Docs: changes have “no effect… unless you call updateParametersInContext().”) :contentReference[oaicite:1]{index=1}
        f.updateParametersInContext(self.simulation.context)
    
        self.simulation.context.setParameter("r0", r0_nm)
        self.simulation.context.setParameter("k_tether", k_kj_per_mol_nm2)
    
    def _clear_ligand_flatbottom_tether(self):
        if hasattr(self, "_lig_fb_tether"):
            self.simulation.context.setParameter("k_tether", 0.0)


    def _ensure_ligand_flatbottom_tether(self):
        """Per-atom flat-bottom tether for ligand heavy atoms. Initially OFF."""
        if hasattr(self, "_lig_fb_tether"):
            return
    
        # r = distance from reference position (x0,y0,z0)
        # Energy = 0 inside r0, harmonic outside r0
        dist = "sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0))"
        expr = f"0.5*k_tether*step({dist}-r0)*({dist}-r0)^2"
        f = CustomExternalForce(expr)

        f.addGlobalParameter("k_tether", 0.0)  # kJ/mol/nm^2
        f.addGlobalParameter("r0", 0.0)        # nm
        f.addPerParticleParameter("x0")
        f.addPerParticleParameter("y0")
        f.addPerParticleParameter("z0")
    
        self._lig_fb_atoms = list(self.ligand_heavy_indices)
        for atom_idx in self._lig_fb_atoms:
            f.addParticle(int(atom_idx), [0.0, 0.0, 0.0])
    
        self.complex_system.addForce(f)
        self._lig_fb_tether = f
        self.simulation.context.reinitialize(preserveState=True)
        print("FLATBOTTOM THETHER ACTIVE")

    
    def _set_ligand_tether(self, ref_pos_nm: np.ndarray, k_kcal_per_mol_A2: float):
        """
        ref_pos_nm: full system positions in nm (N_atoms x 3)
        k_kcal_per_mol_A2: spring constant in kcal/mol/Å^2 (more intuitive)
        """
        self._ensure_ligand_tether_force()
    
        # Convert kcal/mol/Å^2 -> kJ/mol/nm^2
        # 1 kcal = 4.184 kJ, 1 Å^2 = 0.01 nm^2 => multiply by 4.184*100 = 418.4
        k_kj_per_mol_nm2 = float(k_kcal_per_mol_A2) * 418.4
    
        f = self._lig_tether_force
        for p, atom_idx in enumerate(self._lig_tether_atoms):
            x0, y0, z0 = map(float, ref_pos_nm[int(atom_idx)])
            f.setParticleParameters(p, int(atom_idx), [x0, y0, z0])
    
        # Push updated per-particle params into the live Context
        f.updateParametersInContext(self.simulation.context)  # required :contentReference[oaicite:2]{index=2}
        self.simulation.context.setParameter("k_tether", k_kj_per_mol_nm2)
    
    def _clear_ligand_tether(self):
        """Turn off tether without removing it."""
        if hasattr(self, "_lig_tether_force"):
            self.simulation.context.setParameter("k_tether", 0.0)


    def _ensure_ligand_tether_force(self):
        """Create a per-atom harmonic tether for ligand heavy atoms (initially off)."""
        if hasattr(self, "_lig_tether_force"):
            return
    
        f = CustomExternalForce("0.5*k_tether*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        f.addGlobalParameter("k_tether", 0.0)  # kJ/mol/nm^2
        f.addPerParticleParameter("x0")
        f.addPerParticleParameter("y0")
        f.addPerParticleParameter("z0")
    
        # Store the order so we can update per-particle params later
        self._lig_tether_atoms = list(self.ligand_heavy_indices)
    
        for atom_idx in self._lig_tether_atoms:
            f.addParticle(int(atom_idx), [0.0, 0.0, 0.0])
    
        self.complex_system.addForce(f)
        self._lig_tether_force = f
    
        # new Force added -> reinitialize Context to include it
        self.simulation.context.reinitialize(preserveState=True)
    
        
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
    
    def _debug_min_state(self, tag: str, intended_lig_nm=None):
        st = self.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        E = st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        frc = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    
        lig_idx = np.array(self.all_ligand_indices, dtype=int)
        lig_f = frc[lig_idx]
        lig_f_norm = np.linalg.norm(lig_f, axis=1)
        max_f = float(lig_f_norm.max()) if lig_f_norm.size else 0.0
        rms_f = float(np.sqrt(np.mean(lig_f_norm**2))) if lig_f_norm.size else 0.0
    
        msg = f"[{tag}] E={E:.2f} kcal/mol | ligand max|F|={max_f:.3g} kJ/mol/nm | ligand rms|F|={rms_f:.3g}"
        if intended_lig_nm is not None:
            d = pos[lig_idx] - intended_lig_nm
            dA = np.linalg.norm(d, axis=1) * 10.0
            msg += f" | ligand rms disp vs intended={float(np.sqrt(np.mean(dA**2))):.3f} Å | max={float(dA.max()):.3f} Å"
    
        # Print key parameters if they exist
        for pname in ("k_static_pin", "k_stage_heavy", "k_stage_bb", "global_k"):
            try:
                msg += f" | {pname}={self.simulation.context.getParameter(pname)}"
            except Exception:
                pass
    
        print(msg)

    
    
    def minimize_pose_list_(self, ligand_poses_angstrom, max_iters=0):
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
            try:
                self.simulation.context.applyConstraints(1e-6)  # tolerance
            except Exception:
                pass
            
            # Update Ligand Positions
            curr_pos_nm = np.array(initial_pos.value_in_unit(unit.nanometer))
            pose_nm = pose_ang / 10.0
            curr_pos_nm[self.all_ligand_indices] = pose_nm
            
            #zdebug
            self.simulation.context.setPositions(curr_pos_nm)
            intended_lig_nm = np.array(curr_pos_nm)[self.all_ligand_indices]
            self._debug_min_state(f"pose {i+1} PRE", intended_lig_nm=intended_lig_nm)
            # Run
            self._set_ligand_flatbottom_tether(curr_pos_nm, r0_A=0.4, k_kcal_per_mol_A2=0.2)
            if self.do_biased_md and self.md_ps and self.md_ps > 0:
                
                self.run_biased_md(pose_index=i + 1,
                                intended_lig_nm=intended_lig_nm,
                                md_ps=self.md_ps,
                                dt_ps=self.dt_ps,
                                md_temp_K=self.md_temp_K,
                                md_seed=self.md_seed,
                                md_pre_min_iters=self.md_pre_min_iters,
                                md_report_ps=self.md_report_ps,
                            )
            
            self.simulation.minimizeEnergy(maxIterations=max_iters)
            self._clear_ligand_flatbottom_tether()
            
            
            #debug
            self._debug_min_state(f"pose {i+1} POST", intended_lig_nm=intended_lig_nm)
            # Collect
            
            state = self.simulation.context.getState(getPositions=True, getEnergy=True)
            
            from openmm.app import PDBFile

            # Get positions specifically
            positions = state.getPositions()
            
            # Write to a PDB file
            with open(f'/Users/aaron.sweeney/Documents/chemem2_build/ChemEM2_feb26/chemem2-dev/test/fragment_screen/debug_struct_{i}.pdb', 'w') as f:
                PDBFile.writeFile(self.simulation.topology, positions, f)
            
            final_pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            
            #debug 
            dA = np.linalg.norm(final_pos[self.all_ligand_indices] - intended_lig_nm, axis=1) * 10.0
            print(f" pose {i+1} ligand move: rms={np.sqrt(np.mean(dA**2)):.3f} Å max={dA.max():.3f} Å")
            
            final_lig = final_pos[self.all_ligand_indices] * 10.0
            
            energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            results.append(final_lig)
            #write the protein file !!!
            print(f" Pose {i+1}: {time.perf_counter() - t_start:.2f}s | E: {energy:.2f} kcal/mol")

        return results
    
    def minimize_pose_list(self, ligand_poses_angstrom, max_iters=0):
        results = []
        print(f"Minimizing {len(ligand_poses_angstrom)} poses...")
    
        initial_pos = self.simulation.context.getState(getPositions=True).getPositions()
    
        for i, pose_ang in enumerate(ligand_poses_angstrom):
            t_start = time.perf_counter()
            print(f"\n--- Pose {i+1} ---")
    
            try:
                # Reset
                self.simulation.context.setPositions(initial_pos)
    
                # Update ligand
                curr_pos_nm = np.array(initial_pos.value_in_unit(unit.nanometer), copy=True)
                curr_pos_nm[self.all_ligand_indices] = pose_ang / 10.0
    
                # IMPORTANT: set with units
                self.simulation.context.setPositions(curr_pos_nm * unit.nanometer)
    
                # Enforce constraints after setPositions (OpenMM recommends this)
                try:
                    self.simulation.context.applyConstraints(1e-6)
                except Exception:
                    pass
    
                intended_lig_nm = np.array(curr_pos_nm)[self.all_ligand_indices]
                self._debug_min_state(f"pose {i+1} PRE", intended_lig_nm=intended_lig_nm)
                self._debug_ligand_vs_map_bounds(f"pose {i+1} PRE")
                self._debug_map_vs_other_forces(f"pose {i+1} PRE")
                # Pre-flight reject if forces are crazy
                bad, reason = self._pose_is_bad(
                    max_force_kj_per_mol_nm=2e5,
                    rms_force_kj_per_mol_nm=5e4,
                )
                if bad:
                    print(f"[pose {i+1}] SKIP (pre-check): {reason}")
                    continue
    
                # Local tether (keep it local)
                self._set_ligand_flatbottom_tether(curr_pos_nm, r0_A=0.4, k_kcal_per_mol_A2=0.2)
    
                # Optional biased MD
                if self.do_biased_md and self.md_ps and self.md_ps > 0:
                    self.run_biased_md(
                        pose_index=i + 1,
                        intended_lig_nm=intended_lig_nm,
                        md_ps=self.md_ps,
                        dt_ps=self.dt_ps,
                        md_temp_K=self.md_temp_K,
                        md_seed=self.md_seed,
                        md_pre_min_iters=self.md_pre_min_iters,
                        md_report_ps=self.md_report_ps,
                    )
    
                    # Post-MD sanity check
                    bad, reason = self._pose_is_bad(
                        max_force_kj_per_mol_nm=3e5,
                        rms_force_kj_per_mol_nm=8e4,
                    )
                    if bad:
                        print(f"[pose {i+1}] SKIP (post-MD): {reason}")
                        continue
    
                # Final minimize (this is where it can get stuck if NaNs appear)
                self.simulation.minimizeEnergy(maxIterations=max_iters)
    
                # Post-min sanity check
                bad, reason = self._pose_is_bad(
                    max_force_kj_per_mol_nm=3e5,
                    rms_force_kj_per_mol_nm=8e4,
                )
                if bad:
                    print(f"[pose {i+1}] SKIP (post-min): {reason}")
                    continue
    
                self._debug_min_state(f"pose {i+1} POST", intended_lig_nm=intended_lig_nm)
                self._debug_ligand_vs_map_bounds(f"pose {i+1} POST")
                self._debug_map_vs_other_forces(f"pose {i+1} POST")
                # Collect
                state = self.simulation.context.getState(getPositions=True, getEnergy=True)
                final_pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    
                dA = np.linalg.norm(final_pos[self.all_ligand_indices] - intended_lig_nm, axis=1) * 10.0
                print(f" pose {i+1} ligand move: rms={np.sqrt(np.mean(dA**2)):.3f} Å max={dA.max():.3f} Å")
    
                final_lig = final_pos[self.all_ligand_indices] * 10.0
                energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
                results.append(final_lig)
    
                print(f" Pose {i+1}: {time.perf_counter() - t_start:.2f}s | E: {energy:.2f} kcal/mol")
    
            except FloatingPointError as e:
                print(f"[pose {i+1}] SKIP (non-finite): {e}")
    
            except Exception as e:
                print(f"[pose {i+1}] SKIP (exception): {type(e).__name__}: {e}")
    
            finally:
                # Always clear tether so one bad pose doesn't poison the next
                try:
                    self._clear_ligand_flatbottom_tether()
                except Exception:
                    pass
    
        return results


    def _pose_is_bad(
        self,
        *,
        max_force_kj_per_mol_nm: float = 2e5,
        rms_force_kj_per_mol_nm: float = 5e4,
        abs_energy_kcal: float = 1e7,
    ):
        """
        Return (is_bad, reason).
        Uses current Context state (so call it after setting positions).
        """
        st = self.simulation.context.getState(getEnergy=True, getForces=True)
        E = st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    
        #if not np.isfinite(E) or abs(E) > abs_energy_kcal:
        #    return True, f"bad energy: {E}"
    
        frc = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
        lig_idx = np.array(self.all_ligand_indices, dtype=int)
        lig_f = frc[lig_idx]
        lig_norm = np.linalg.norm(lig_f, axis=1)
        fmax = float(lig_norm.max()) if lig_norm.size else 0.0
        frms = float(np.sqrt(np.mean(lig_norm**2))) if lig_norm.size else 0.0
    
        if (not np.isfinite(fmax)) or (not np.isfinite(frms)):
            return True, f"non-finite force: max={fmax}, rms={frms}"
    
        #if fmax > max_force_kj_per_mol_nm or frms > rms_force_kj_per_mol_nm:
        #    return True, f"too-large force: max={fmax:.3g}, rms={frms:.3g}"
    
        return False, "ok"

    
    def run_biased_md(self,
                      pose_index,
                      intended_lig_nm,
                      md_ps: float,
                      dt_ps: float,
                      md_pre_min_iters : int = 0,
                      md_temp_K :float = 150,
                      md_seed: int = 1,
                      md_report_ps: float = 0.0):
        
        """
        Run a short biased MD burst (bias comes from whatever forces are in the System,
        e.g. your map potential with a fixed global_k).

        Parameters
        ----------
        pose_index : int
            1-based pose index for logging.
        intended_lig_nm : np.ndarray
            Ligand coordinates (N_lig x 3) in nm used as reference for displacement logs.
        md_ps : float
            Total MD time in ps.
        dt_ps : float
            Integrator timestep in ps.
        md_temp_K : float
            Temperature for velocity initialization.
        md_seed : int
            Base seed; actual seed is md_seed + (pose_index-1).
        md_pre_min_iters : int
            Small pre-minimization iterations to tame huge forces before dynamics.
        md_report_ps : float
            If >0, print a progress line every md_report_ps ps.
        """
        
        
        
        self.simulation.minimizeEnergy(maxIterations=int(md_pre_min_iters))
        
        try:
            self.simulation.context.setVelocitiesToTemperature(
                md_temp_K * unit.kelvin,
                int(md_seed + (pose_index - 1)),
            )
        except TypeError:
            # Older signatures differ slightly; fall back without seed
            self.simulation.context.setVelocitiesToTemperature(md_temp_K * unit.kelvin)
        
        nsteps = int(round(float(md_ps) / float(dt_ps)))
        if nsteps < 1:
            nsteps = 1
        
        if md_report_ps and md_report_ps > 0:
            chunk = max(1, int(round(float(md_report_ps) / float(dt_ps))))
            done = 0
            while done < nsteps:
                step_now = min(chunk, nsteps - done)
                self.simulation.step(step_now)
                done += step_now

                # light debug (forces are expensive; reuse your helper if you want full force info)
                st = self.simulation.context.getState(getEnergy=True, getPositions=True)
                E = st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
                pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

                dA = np.linalg.norm(pos[self.all_ligand_indices] - intended_lig_nm, axis=1) * 10.0
                print(
                    f"  [pose {pose_index} MD {done*dt_ps:.1f} ps] E={E:.2f} kcal/mol | "
                    f"lig rmsΔ={np.sqrt(np.mean(dA**2)):.3f} Å maxΔ={dA.max():.3f} Å"
                )
        else:
            self.simulation.step(nsteps)
                    
   

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
    #--debuging 
    def _map_bounds_nm(self):
        mp = self.density_map
        vol = mp.density_map
        nz, ny, nx = vol.shape  # z,y,x
        ox, oy, oz = map(float, mp.origin)
        ax, ay, az = map(float, mp.apix)
        A2NM = 0.1
        xmin = (ox - 0.5 * ax) * A2NM; xmax = (ox + nx * ax - 0.5 * ax) * A2NM
        ymin = (oy - 0.5 * ay) * A2NM; ymax = (oy + ny * ay - 0.5 * ay) * A2NM
        zmin = (oz - 0.5 * az) * A2NM; zmax = (oz + nz * az - 0.5 * az) * A2NM
        return xmin, xmax, ymin, ymax, zmin, zmax
    
    def _debug_ligand_vs_map_bounds(self, tag: str):
        if not self.density_map:
            return
        xmin, xmax, ymin, ymax, zmin, zmax = self._map_bounds_nm()
    
        st = self.simulation.context.getState(getPositions=True)
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    
        lig = pos[np.array(self.all_ligand_indices, dtype=int)]
        mn = lig.min(axis=0); mx = lig.max(axis=0)
    
        outside = (
            (mn[0] < xmin) or (mx[0] > xmax) or
            (mn[1] < ymin) or (mx[1] > ymax) or
            (mn[2] < zmin) or (mx[2] > zmax)
        )
    
        print(
            f"[{tag}] map box nm: x[{xmin:.3f},{xmax:.3f}] y[{ymin:.3f},{ymax:.3f}] z[{zmin:.3f},{zmax:.3f}] | "
            f"lig box nm: x[{mn[0]:.3f},{mx[0]:.3f}] y[{mn[1]:.3f},{mx[1]:.3f}] z[{mn[2]:.3f},{mx[2]:.3f}] | "
            f"outside={outside}"
        )
    
    def _force_energy_by_group(self, group: int):
        mask = 1 << int(group)
        stE = self.simulation.context.getState(getEnergy=True, groups=mask)
        stF = self.simulation.context.getState(getForces=True, groups=mask)
        E = stE.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        F = stF.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
        return E, F
    
    def _debug_map_vs_other_forces(self, tag: str):
        # pick phosphate-ish atoms: P (15) and O (8) in ligand
        lig_set = set(self.all_ligand_indices)
        phos = [a.idx for a in self.complex_structure.atoms
                if a.idx in lig_set and a.element in (15, 8)]  # coarse, but good enough to start
    
        # energies + forces
        E_map, F_map = self._force_energy_by_group(7)
    
        # everything else: build mask of all groups, then subtract map
        allmask = 0
        for f in self.complex_system.getForces():
            allmask |= (1 << f.getForceGroup())
        othermask = allmask & ~(1 << 7)
    
        stE = self.simulation.context.getState(getEnergy=True, groups=othermask)
        stF = self.simulation.context.getState(getForces=True, groups=othermask)
        E_other = stE.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        F_other = stF.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    
        def stats(F, idxs):
            if not idxs:
                return 0.0, 0.0
            v = np.linalg.norm(F[np.array(idxs, dtype=int)], axis=1)
            return float(v.max()), float(np.sqrt(np.mean(v*v)))
    
        lig = self.all_ligand_indices
        maxFm_lig, rmsFm_lig = stats(F_map, lig)
        maxFo_lig, rmsFo_lig = stats(F_other, lig)
    
        maxFm_ph, rmsFm_ph = stats(F_map, phos)
        maxFo_ph, rmsFo_ph = stats(F_other, phos)
    
        print(
            f"[{tag}] E_map(g7)={E_map:.2f} kcal | E_other={E_other:.2f} kcal | "
            f"lig |F_map|max={maxFm_lig:.3g} rms={rmsFm_lig:.3g} | "
            f"lig |F_other|max={maxFo_lig:.3g} rms={rmsFo_lig:.3g} | "
            f"PO |F_map|max={maxFm_ph:.3g} rms={rmsFm_ph:.3g} | "
            f"PO |F_other|max={maxFo_ph:.3g} rms={rmsFo_ph:.3g}"
        )