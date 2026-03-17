#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 00:03:03 2025

@author: aaron.sweeney
"""

import os 
from openmm.app import HBonds, NoCutoff
from openmm import unit, app, XmlSerializer

class ExportSimulation:
    
    def __init__(self, system):
        self.system = system 
    
    def get_structures(self):
        
        self.protein_openff_structure = self.system.protein.complex_structure
        self.ligand_openff_structures = [i.complex_structure for i in self.system.ligand]
        
    def export_simulation(self):
        
        export_complex_system(self.protein_openff_structure, self.ligand_openff_structures, output = self.system.output)
        
    def run(self):
        self.get_structures()
        self.export_simulation()
        
    

def export_complex_system(protein, 
                          ligands,
                          flexible_side_chains = True,
                          solvent = False,
                          platform = 'OpenCL',
                          output = './'
                          ):
    
    if len(ligands):
        
        complex_structure =  protein + ligands[0]
        
        if len(ligands) > 1:
            for structure in ligands[1:]:
                complex_structure += structure
    
    else:
        complex_structure = protein
    
    if solvent == True:
        complex_system = complex_structure.createSystem(
            nonbondedMethod=NoCutoff,
            nonbondedCutoff=9.0 * unit.angstrom,
            constraints=HBonds,
            removeCMMotion=False,
            implicitSolvent=app.GBn2
        )
        
    else:
        complex_system = complex_structure.createSystem(nonbondedMethod=NoCutoff,
                                                        nonbondedCutoff=9.0 * unit.angstrom,
                                                        constraints=HBonds, 
                                                        removeCMMotion=False, 
                                                        rigidWater=True)
        
    
    
    os.makedirs(output, exist_ok=True)
    
    system_xml = XmlSerializer.serialize(complex_system)
    with open(os.path.join(output, 'complex_system.xml'), 'w') as f:
        f.write(system_xml)
    
    
    prmtop = os.path.join(output, 'complex_structure.prmtop')
    inpcrd = os.path.join(output, 'complex_structure.inpcrd')

    for f in (prmtop, inpcrd):
        if os.path.exists(f):
            os.remove(f)

    complex_structure.save(prmtop)
    complex_structure.save(inpcrd)    

    
    
