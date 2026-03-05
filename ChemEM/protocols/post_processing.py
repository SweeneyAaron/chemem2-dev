#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 00:15:24 2025

@author: aaron.sweeney
"""
import os
from ChemEM.messages import Messages
from openmm import  unit, app, CustomCompoundBondForce
from openmm import MonteCarloBarostat, XmlSerializer, app, unit, CustomCompoundBondForce, Continuous3DFunction, vec3, Vec3, CustomNonbondedForce
from openmm.app import HBonds, NoCutoff, PDBFile, Modeller, Topology, PME, StateDataReporter
from openmm.unit.quantity import Quantity
from openmm import LangevinIntegrator, Platform
from ChemEM.tools.pose_minimiser import PoseMinimiser, AnnealingConfig




class PostProcess:
    def __init__(self, system):
        """
        system must expose:
          - system.protein : parmed.Structure
          - system.ligands: List[parmed.Structure]
          - optional system.density_map with .density_map, .origin, .apix, .resolution, .map_contour
        """
        self.system = system
        self.solvent = None
    
    
    def get_output_dir(self):
        
        self.output = os.path.join(self.system.output, 'post_processing')
        try:
            os.mkdir(self.output)
        except FileExistsError:
            pass
    def get_protein_structure(self):
        #TODO! system ligands!!
        self.protein_openff_structure =  self.system.protein.complex_structure
        self.sse_groups = self.system.protein.sse_groups 
        self.sse_codes = self.system.protein.sse_codes
        
    def get_ligand_structures(self):
        self.ligand_openff_structures = [i.complex_structure for i in self.system.ligand]
    def get_density_map(self):
        self.density_map = getattr(self.system, 'confidence_map', None)
        
    def add_protein_restraints(self, complex_structure, complex_system):
        
        
        #add protein restraints options, protein, sse, none 
        if self.system.options.md_restratins == 'protein':
            pass
        
        elif self.system.options.md_restratins == 'sse':
            import pdb 
            pdb.set_trace()
        
        else:
            pass
        
    
    def run(self):
        self.system.log(Messages.create_centered_box('MD-density fit'))
        self.get_output_dir()
        self.get_protein_structure()
        self.get_ligand_structures()
        self.get_density_map()
        
        pm = PoseMinimiser(self.protein_openff_structure,
                           self.ligand_openff_structures,
                           density_map=self.density_map,
                           global_k=self.system.options.global_k, #add option
                           solvent=None,#add option
                           pin_k = 5000,
                           platform_name = self.system.platform,
                           protein_restraint = self.system.options.md_restraints,#'protein', 'sse', 'none'
                           localise = self.system.options.restrain_sidechains,
                           sse_groups = self.sse_groups,
                           #ss_codes = self.sse_codes,
                           #sse_mode = 'protein',
                           sse_k = 5000.0,
                           smooth_sigma_A = 1.0 #?!
                           )
        

        anneling_config = AnnealingConfig()
        #TODO! add options
        #results = pm.simulated_anneling(write_pdb = os.path.join(self.output, 'refined_complex.pdb'))
        result  = pm.run_simulated_annealing(anneling_config, write_pdb = os.path.join(self.output, 'refined_complex.pdb'))
        


def get_openmm_combined_complex_system(protein, 
                                       ligand,
                                       nonbondedCutoff=1.0 * unit.nanometer,
                                       constraints=HBonds,
                                       padding=1.0 * unit.nanometer,  # Padding for the water box
                                       ionicStrength=0.15 * unit.molar,
                                       platform = 'OpenCL',
                                       solvent=app.GBn2):
    
   
    for lig in ligand:
        for num, residue in enumerate(lig.residues):
            if residue.name == 'UNL':  # Replace 'UNL' with the correct residue name
                residue.name = f'LIG_{num}'
    
    complex_structure =  protein + ligand[0]
    for structure in ligand[1:]:
        complex_structure += structure
    
    if solvent is not None:
        complex_system = complex_structure.createSystem(
            nonbondedMethod=NoCutoff,
            nonbondedCutoff=9.0 * unit.angstrom,
            constraints=HBonds,
            removeCMMotion=True,
            implicitSolvent=app.GBn2
        )
    else:
        complex_system = complex_structure.createSystem(nonbondedMethod=NoCutoff,
                                                        nonbondedCutoff=9.0 * unit.angstrom,
                                                        constraints=HBonds, 
                                                        removeCMMotion=True, 
                                                        rigidWater=True)
    
    return complex_structure, complex_system





        