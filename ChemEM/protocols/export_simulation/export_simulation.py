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

    def export_torsion_profiles(self):
        '''
        Write clean intrinsic (PeriodicTorsionForce-only) torsion profiles for every
        ligand to torsion_profiles.json next to the exported system files. These are the
        same profiles ChemEM computes during docking preprocessing; persisting them lets
        the ChimeraX plugin display the correct curves instead of a noisy live scan.

        Profiles are keyed by atom names (the robust join with the plugin) and also carry
        full-system indices. The exported complex concatenates protein + ligands in order
        (see export_complex_system), so a ligand's full-system index offset is the protein
        atom count plus the atom counts of all preceding ligands. The offset is advanced
        even for skipped ligands so later offsets stay correct.
        '''
        import json
        from ChemEM.tools.precomputed_data import export_torsion_profile, get_torsion_lists

        output = self.system.output
        platform = getattr(self.system, 'platform', 'CPU')

        protein_natoms = len(self.protein_openff_structure.atoms)
        offset = protein_natoms

        entries = []
        for lig_pos, ligand in enumerate(self.system.ligand):
            cs = getattr(ligand, 'complex_structure', None)
            mol = getattr(ligand, 'mol', None)
            if cs is None or mol is None:
                continue
            n_atoms = len(cs.atoms)
            try:
                torsion_lists = get_torsion_lists(mol)
                profiles = export_torsion_profile(ligand, torsion_lists, platform,
                                                  normalise=True)
            except Exception:
                import traceback
                traceback.print_exc()
                offset += n_atoms
                continue

            residue_names = sorted({a.residue.name for a in cs.atoms})
            for prof in profiles:
                atom_names, local_idx, angle_energy = prof
                entries.append({
                    'ligand_position': lig_pos,
                    'ligand_residue_names': residue_names,
                    'atom_names': [str(x) for x in atom_names],
                    'local_indices': [int(x) for x in local_idx],
                    'global_offset': int(offset),
                    'global_indices': [int(x) + int(offset) for x in local_idx],
                    'angle_convention': 'deg_-180_180',
                    'energy_convention': 'normalized_0_1_relative',
                    'profile': [[int(a), float(e)] for a, e in angle_energy],
                })
            offset += n_atoms

        doc = {
            'version': 1,
            'protein_natoms': protein_natoms,
            'torsions': entries,
        }
        os.makedirs(output, exist_ok=True)
        with open(os.path.join(output, 'torsion_profiles.json'), 'w') as f:
            json.dump(doc, f)

    def run(self):
        self.get_structures()
        self.export_simulation()
        try:
            self.export_torsion_profiles()
        except Exception:
            import traceback
            traceback.print_exc()
        
    

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

    
    
