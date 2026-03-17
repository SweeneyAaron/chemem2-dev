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
        if self.system.options.anneling:
            self._minimiser = self.anneling
        else:
            self._minimiser = self.minimise
    
    def anneling(self, env):
        minimiser = SimulatedAnnealingInPlace(env)
        config = AnnealingConfig()
        energy = minimiser.run(config)
        return energy
        
    
    def minimise(self, env):
        minimizer = MinimiseInPlace(env)
        final_energy = minimizer.run(
            do_biased_md=self.system.options,
            md_ps=5.0,     # Or getattr(self.system.options, 'md_ps', 5.0)
            max_iters=200
        )
        return final_energy
    
    
        
    
    def refine(self):
        
        for structure, sub_map, ligand_structures in self._ligand_local_structures:
            
            env = ChemEMSimulationSetup(
                protein_structure=structure,
                ligand_structure=[lig.complex_structure for lig in ligand_structures],
                density_map=sub_map,
                platform_name=getattr(self.system, 'platform', 'CPU'),
                protein_restraint='protein',
                pin_k=5000.0,
                localise=False, # Free side chains
            )
           
            final_energy = self._minimiser(env)
            
            if final_energy is None:
                print("Skipping structure update due to bad pose/forces.")
                continue
            
            #final_pos_angstrom = env.complex_structure.positions.value_in_unit(unit.angstrom)
            
            #need to alter the protein residues from this part 
            #need to alter the ligand sdf positions
            updated_atoms = update_global_positions(
                full_structure=self.complex_structure, 
                local_structure=env.complex_structure
            )
            
            
            updated_ligs = update_ligand_positions(
                local_structure=env.complex_structure,
                ligand_objects=ligand_structures 
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



