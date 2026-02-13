# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


import numpy as np 
import os
from ChemEM.messages import Messages
from ChemEM.tools.ligand import get_van_der_waals_radius
from ChemEM.tools.density import (compute_p_values,
                                  benjamini_yekutieli,
                                  estimate_background_distribution,
                                  coords_to_grid,
                                  create_mask_from_model,
                                  refine_mask,
                                  estimate_background_stats
                                  )
class ConfidenceMap:
    def __init__(self, system):
        self.system = system 
        self.radius_in_angstrom = 6.0
        self.use_model = False
        
    def get_position_input(self):
        
        
        self.atoms = [i for i in self.protein_openff_structure.atoms if i.element > 1]
        self.positions = np.array([[i.xx, i.xy, i.xz] for i in self.atoms ])
        
        self.atom_radii = np.array( [get_van_der_waals_radius(i.element_name) for i in self.atoms] )
        #pdb.set_trace()
    
    
    def get_protein_atom_radius(self):
        #improve this to include the resolution and atomm element
        self.voxel_radius = int(np.ceil(self.radius_in_angstrom / self.densmap.apix[0]))
        
    def get_grid_coords(self):
        self.grid_coords = coords_to_grid(self.positions,
                         self.densmap.origin,
                         self.densmap.apix) #x,y,z
    
    def get_protein_structure(self):
        #TODO! system ligands!!
        
        self.protein_openff_structure =  self.system.protein.complex_structure
        self.densmap = self.system.density_map   
    
    def get_model_mask(self):
        
        
        model_mask = create_mask_from_model(self.densmap.density_map.shape, #z,y,x
                                                 self.grid_coords,
                                                 self.voxel_radius)
        
        self.model_mask = refine_mask(model_mask)
    
    def get_background_by_mask(self):
        self.bg_mean, self.bg_std = estimate_background_stats(self.densmap.density_map, self.model_mask)
    
    def estimate_background_from_cubes(self):
        self.bg_mean, self.bg_std  = estimate_background_distribution(self.densmap.density_map, cube_size=10, num_cubes=4)
    
    def compute_confidence_map(self, cube_size=10, num_cubes=4, desired_ppv=0.99):
        """
        Compute a confidence map from the given density map following the described procedure.
        """
       
        
        # 2. Compute p-values for each voxel
        pvals = compute_p_values(self.densmap.density_map, self.bg_mean, self.bg_std)
        
        # 3. Apply Benjamini-Yekutieli to get q-values
        qvals = benjamini_yekutieli(pvals)
        
        # 4. Convert q-values to PPVs: PPV = 1 - q
        ppvs = 1 - qvals
        
        # 5. Threshold the PPV map based on desired PPV (e.g., 0.99)
        # This thresholded version can be visualized. For general visualization,
        # you might store the PPV map to a file format supported by your visualization software.
        thresholded_map = (ppvs >= desired_ppv).astype(np.float32)
        
        
        outpath = os.path.join(self.system.output, 'fdr_maps')
        try:
            os.mkdir(outpath)
        except FileExistsError:
            pass
            
        confidence_map = self.densmap.copy() 
        confidence_map.density_map = ppvs 
        out_file = os.path.join(outpath, 'confidence_map.mrc')
        confidence_map.write_mrc(out_file)
        
        masked_map = self.densmap.copy() 
        masked_map.density_map = masked_map.density_map * thresholded_map
        out_file = os.path.join(outpath, 'masked_map.mrc')
        masked_map.write_mrc(out_file)
        self.system.confidence_map = masked_map
        #return ppvs, thresholded_map
    
    def run(self):
        if self.system.options.no_map or self.system.density_map is None:
            self.system.log(Messages.chemem_warning(self.__class__, 'run()', '[Warning] no density map requested or specified skipping confidence_map'))
            return
        
        self.system.log(Messages.create_centered_box('Confidence Mapping'))
        self.get_protein_structure() 
        
        
        if self.use_model:
            self.get_position_input()
            self.get_grid_coords()
            self.get_protein_atom_radius()
            self.get_model_mask() #replace this with a blurred map!!
            self.get_background_by_mask()
        else:
            self.estimate_background_from_cubes()
        
        self.compute_confidence_map()

