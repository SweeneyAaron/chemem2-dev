# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>



from .pose_minimiser import ChemEMSimulationSetup, MinimiseInPlace, SimulatedAnnealingInPlace, AnnealingConfig
from .refine_utils import get_residue_positions, get_residue_subset_from_points
from ChemEM.protocols.core.density import submap_from_structure
from ChemEM.protocols.core.simulation import update_global_positions, update_ligand_positions
from ChemEM.parsers.writers import save_structure_parmed
from openmm import unit
import numpy as np
from openmm.app import PDBFile
import os 

class Refine:
    def __init__(self, system):
        self.system = system 
        self._split_maps = None 
        self._split_sites = None
        self._ligand_residues = []
        self._ligand_local_structures = []
        self._minimiser = None
    
    
    def get_output(self):
        self.output = os.path.join(self.system.output, 'refine')
        os.makedirs(self.output, exist_ok=True)
    
    def create_complex_structure(self):
        
        self.complex_structure = self.system.protein.complex_structure
        
    
    def get_sites(self):
        
        if not self.system.options.local_refine:
            submap = None 
            
            if not self.system.options.no_map and self.densmap is not None:
                submap = submap_from_structure(self.complex_structure, self.system.density_map)
                self._ligand_local_structures.append((self.complex_structure, submap, self.system.ligand))
                
            return
        
        for ligand in self.system.ligand:
            
            points = get_residue_positions(ligand.complex_structure.residues[0])
            selected_residues = get_residue_subset_from_points(points, 
                                                               self.complex_structure, 
                                                               distance_cutoff = self.system.options.local_radius)
            
            submap = None
            #cut the map by structure
            if not self.system.options.no_map and self.densmap is not None:
                submap = submap_from_structure(selected_residues, self.system.density_map)
            
            self._ligand_local_structures.append((selected_residues, submap, [ligand]))
            
        

    def get_map(self):
        self.densmap = getattr(self.system, "density_map", None)
    
    def get_minimiser(self):
        if self.system.options.annealing:
            self._minimiser = self.annealing
        else:
            self._minimiser = self.minimise
    
    def annealing(self, env):
        minimiser = SimulatedAnnealingInPlace(env)
        
    
        config = AnnealingConfig(
            minimise_before=self.system.options.pre_minimise,
            minimise_after=self.system.options.post_minimise,
            base_T=self.system.options.base_temp,
            heat_to_K=self.system.options.heat_to_k,
            temp_step_K=self.system.options.temp_step_k,
            md_steps_per_T=self.system.options.steps_per_temp,
            high_hold_ps = self.system.options.high_hold_ps,
            cycles=self.system.options.cycles,
            seed=self.system.options.seed,
            use_com_tether=self.system.options.com_restraint,
            com_tether_r0_A=self.system.options.com_restraint_dist,
            com_tether_k_kcal_per_mol_A2=self.system.options.com_restrain_kcal_per_mol,
            com_tether_alpha_per_nm=self.system.options.com_restrain_alpha_per_nm,
        )


        #config.cycles = self.system.options.annealing_cycles 
        energy = minimiser.run(config)
        return energy
        
    
    def minimise(self, env):
        minimizer = MinimiseInPlace(env)
        final_energy = minimizer.run(
            do_biased_md=self.system.options.do_biased_md,
            md_ps=5.0,     # Or getattr(self.system.options, 'md_ps', 5.0)
            max_iters=200
        )
        return final_energy
    
    
    
    def refine(self):

        def _positions_to_numpy_angstrom(pos):
            """
            Convert a ParmEd/OpenMM positions object into an (N, 3) float numpy array in Å.
            Works for lists of Vec3, tuples, or Quantity-wrapped coordinates.
            """
            try:
                pos = pos.value_in_unit(unit.angstrom)
            except Exception:
                pass
    
            arr = []
            for p in pos:
                try:
                    arr.append([float(p[0]), float(p[1]), float(p[2])])
                except Exception:
                    arr.append([float(p.x), float(p.y), float(p.z)])
            return np.asarray(arr, dtype=float)

        for structure, sub_map, ligand_structures in self._ligand_local_structures:
            
            '''
            env = ChemEMSimulationSetup(
                protein_structure=structure,
                ligand_structure=[lig.complex_structure for lig in ligand_structures],
                density_map=sub_map,
                platform_name=getattr(self.system, 'platform', 'CPU'),
                protein_restraint='protein',
                pin_k=5000.0,
                localise=False,
            )
            '''
            
            
            
            # Pass Ligand objects directly (not just .complex_structure). This
            # lets create_system see any attached CovalentLinkSpec and inject
            # junction bonded terms + updated charges before createSystem().
            env = ChemEMSimulationSetup(
                protein_structure=structure,
                ligand_structure=list(ligand_structures),
                density_map=sub_map,
                platform_name=getattr(self.system, 'platform', 'CPU'),
                protein_restraint='protein',
                pin_k=5000.0,
                localise=False,
                pin_specs=getattr(self.system.options, 'pin_specs', []),
                distance_specs=getattr(self.system.options, 'distance_specs', []),
                resource_owner=self.system,
            )
    
            final_energy = self._minimiser(env)
    
            if final_energy is None:
                print("Skipping structure update due to bad pose/forces.")
                continue
    
            # ------------------------------------------------------------
            # DEBUG 1: inspect ligand directly in the OpenMM-refined env
            # ------------------------------------------------------------
            env_pos_A = _positions_to_numpy_angstrom(env.complex_structure.positions)
            lig_idx = np.asarray(env.all_ligand_indices, dtype=int)
            env_lig_pos_A = env_pos_A[lig_idx]
            env_lig_com_A = env_lig_pos_A.mean(axis=0)
    
            print(
                f"[debug refine] env ligand COM before transfer = "
                f"({env_lig_com_A[0]:.3f}, {env_lig_com_A[1]:.3f}, {env_lig_com_A[2]:.3f})"
            )
    
            # ------------------------------------------------------------
            # Copy protein/local structure back
            # ------------------------------------------------------------
            update_global_positions(
                full_structure=self.complex_structure,
                local_structure=env.complex_structure
            )
    
            # ------------------------------------------------------------
            # DEBUG 2: compare ligand object coords before ligand update
            # ------------------------------------------------------------
            for lig_num, lig in enumerate(ligand_structures):
                lig_pos_before_A = _positions_to_numpy_angstrom(lig.complex_structure.positions)
                lig_com_before_A = lig_pos_before_A.mean(axis=0)
    
                print(
                    f"[debug refine] ligand[{lig_num}] object COM before ligand update = "
                    f"({lig_com_before_A[0]:.3f}, {lig_com_before_A[1]:.3f}, {lig_com_before_A[2]:.3f})"
                )
    
            # ------------------------------------------------------------
            # Copy refined ligand coords back into ligand objects
            # ------------------------------------------------------------
            update_ligand_positions(
                local_structure=env.complex_structure,
                ligand_objects=ligand_structures
            )
    
            # ------------------------------------------------------------
            # DEBUG 3: compare ligand object coords after ligand update
            # ------------------------------------------------------------
            for lig_num, lig in enumerate(ligand_structures):
                lig_pos_after_A = _positions_to_numpy_angstrom(lig.complex_structure.positions)
                lig_com_after_A = lig_pos_after_A.mean(axis=0)
    
                print(
                    f"[debug refine] ligand[{lig_num}] object COM after ligand update = "
                    f"({lig_com_after_A[0]:.3f}, {lig_com_after_A[1]:.3f}, {lig_com_after_A[2]:.3f})"
                )
    
                if len(lig_pos_after_A) == len(env_lig_pos_A):
                    rms = np.sqrt(np.mean(np.sum((lig_pos_after_A - env_lig_pos_A) ** 2, axis=1)))
                    dcom = np.linalg.norm(lig_com_after_A - env_lig_com_A)
                    print(
                        f"[debug refine] ligand[{lig_num}] vs env ligand: "
                        f"RMS={rms:.3f} Å | COM shift={dcom:.3f} Å"
                    )
    
    def write_output(self):
        pdb_out = os.path.join(self.output, 'minimised_receptor.pdb')
        save_structure_parmed(self.complex_structure, pdb_out)
        
        for num, ligand in enumerate(self.system.ligand):
            sdf_out = os.path.join(self.output, f'Ligand_{num}.sdf')
            ligand.write_sdf(sdf_out)
        
        
        
    def run(self):
        
        self.get_output()
        self.create_complex_structure()
        self.get_map()
        self.get_sites()
        self.get_minimiser()
        self.refine()   
        
        self.write_output()


