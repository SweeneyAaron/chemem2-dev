# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

import os
from ChemEM.protocols.core.density import submap_from_structure
from .refine_utils import get_residue_positions, get_residue_subset_from_points

class Anneling:
    
    def __init__(self, system):
        self.system = system 
    
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
    
    
    def get_anneling(self):
        pass
    
    def run(self):
        
        self.get_output()
        self.create_complex_structure()
        self.get_map()
        self.get_sites()
        import pdb 
        pdb.set_trace()
        
        
        
        


