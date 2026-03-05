#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from ChemEM.messages import Messages
import numpy as np
from .mapq_utils import compute_qscores_from_emmap
import json
import os

class ScoreMapQ:
    
    def __init__(self, system):
        self.system = system 
        self.density_map = None
        self.outfile = None
    
    def _check_density_map(self):
        if self.system.density_map is not None:
            self.density_map = self.system.density_map
        else:
            raise ValueError("No density map found in System. MapQ scoring requires a density map.")
            
    def _get_outfile(self):
        
        if getattr(self.system, 'output', None) is None:
            self.system.output = "."
            
        os.makedirs(self.system.output, exist_ok=True)
        self.outfile = os.path.join(self.system.output, 'mapq_scores.json')
        return self.outfile

    def _is_within_map_bounds(self, positions):
        """
        Checks if ALL heavy atoms in the conformer are within the bounds 
        of the EMMap grid.
        """
        try:
           
            origin = np.array(self.density_map.origin)
            apix = self.density_map.apix
            shape = np.array(self.density_map.density_map.shape)
            
           
            max_bounds = origin + (shape - 1) * apix
            
            return np.all(positions >= origin) and np.all(positions <= max_bounds)
            
        except AttributeError as e:
            self.system.log(f"Warning: Could not verify map bounds due to missing EMMap attributes. Error: {e}")
            return True # Fail open if we can't verify

    def run(self):
        self.system.log(Messages.create_centered_box("MapQ Score"))
        
        self._check_density_map()
        outfile_path = self._get_outfile()
        
        results = {}
        
        
        sigma_ref = getattr(self.system.options, 'sigma_ref', 0.6)
        per_atom = getattr(self.system.options, 'per_atom', False)

        for lig_id, ligand in enumerate(self.system.ligand):
            lig_key = f"ligand_{lig_id}"
            results[lig_key] = {}
            
            
            heavy_idxs = [a.GetIdx() for a in ligand.mol.GetAtoms() if a.GetSymbol() != 'H']
            
            if not heavy_idxs:
                self.system.log(f"Warning: Ligand {lig_id} has no heavy atoms. Skipping.")
                continue

            
            for conf in ligand.mol.GetConformers():
                conf_id = conf.GetId()
                
               
                positions = conf.GetPositions()[heavy_idxs]
                
                if not self._is_within_map_bounds(positions):
                    self.system.log(f"  -> Ligand {lig_id} (Conf {conf_id}) is outside map bounds. Skipping.")
                    results[lig_key][f"conf_{conf_id}"] = None
                    continue
                
                qs = compute_qscores_from_emmap(
                    atoms_xyz=positions, 
                    emmap=self.density_map, 
                    sigma_ref=sigma_ref
                )
                
                
                if not per_atom:
                    # Calculate mean and convert numpy.float32 to standard python float
                    final_score = float(np.mean(qs))
                else:
                    # Convert the numpy array of individual scores to a list of standard floats
                    final_score = [float(q) for q in qs]
                
                results[lig_key][f"conf_{conf_id}"] = final_score
        
        
        self.system.log(f"Writing MapQ scores to {outfile_path}")
        with open(outfile_path, 'w') as f:
            json.dump(results, f, indent=4)
            
