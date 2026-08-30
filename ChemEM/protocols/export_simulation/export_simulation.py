#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 00:03:03 2025

@author: aaron.sweeney
"""

import os
from openmm.app import HBonds, NoCutoff
from openmm import unit, app, XmlSerializer

from ChemEM.protocols.core.simulation import resolve_implicit_solvent

class ExportSimulation:
    
    def __init__(self, system):
        self.system = system 
    
    def get_structures(self):

        self.protein_openff_structure = self.system.protein.complex_structure
        self.ligand_openff_structures = [i.complex_structure for i in self.system.ligand]
        #Keep the Ligand objects too: the covalent link spec lives on them, and
        #export_complex_system needs it to inject the junction bond before
        #createSystem() (the bare complex_structures carry no spec).
        self.ligand_objects = list(self.system.ligand)

    def export_simulation(self):

        export_complex_system(self.protein_openff_structure, self.ligand_openff_structures,
                              output = self.system.output,
                              solvent = resolve_implicit_solvent(self.system.options, None),
                              ligand_objects = self.ligand_objects)

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

    def export_covalent_links(self):
        '''
        Write covalent_links.json next to the exported system files, describing every
        covalent link in RESOLVED terms: the ligand/protein atoms that were actually
        bonded and the atoms that were actually deleted (user-specified plus the ones
        auto-detected by valence).

        The ChimeraX plugin consumes this to reconcile its 3D model with the exported
        system. Its ChimeraX-atom -> OpenMM-index mapping is by (residue name, atom
        name) + position, so leaving-group atoms that no longer exist in the export
        would otherwise map to nothing and sit frozen in the display while the rest of
        the structure moves. The plugin cannot derive the deleted-atom list itself
        without reimplementing _auto_h_to_remove_rdkit / _auto_h_to_remove_parmed.

        Writes nothing when no ligand carries a covalent link.
        '''
        import json

        links = []
        for lig_pos, ligand in enumerate(self.ligand_objects):
            specs = getattr(ligand, 'covalent_links', None) or []
            if not specs:
                continue

            cs = getattr(ligand, 'complex_structure', None)
            residue_names = sorted({a.residue.name for a in cs.atoms}) if cs is not None else []

            # One entry per bond. A crosslinker therefore contributes several
            # entries sharing the same ligand_position — the schema is unchanged.
            for spec in specs:
                links.append({
                    'ligand_position': lig_pos,
                    'ligand_residue_names': residue_names,
                    'ligand_atom_name': spec.resolved_ligand_atom_name,
                    'protein': {
                        'chain': spec.resolved_protein_chain,
                        'resname': spec.resolved_protein_resname,
                        'resnum': spec.resolved_protein_resnum,
                        'atom_name': spec.resolved_protein_atom_name,
                    },
                    'bond_order': spec.bond_order,
                    'deleted_ligand_atoms': list(spec.delete_ligand_atoms) + list(spec.auto_deleted_ligand_atoms),
                    'deleted_protein_atoms': list(spec.delete_protein_atoms) + list(spec.auto_deleted_protein_atoms),
                })

        if not links:
            return

        output = self.system.output
        os.makedirs(output, exist_ok=True)
        with open(os.path.join(output, 'covalent_links.json'), 'w') as f:
            json.dump({'version': 1, 'links': links}, f)

    def run(self):
        self.get_structures()
        self.export_simulation()
        try:
            self.export_torsion_profiles()
        except Exception:
            import traceback
            traceback.print_exc()
        try:
            self.export_covalent_links()
        except Exception:
            import traceback
            traceback.print_exc()



def export_complex_system(protein,
                          ligands,
                          flexible_side_chains = True,
                          solvent = False,
                          platform = 'OpenCL',
                          output = './',
                          ligand_objects = None
                          ):

    if len(ligands):

        complex_structure =  protein + ligands[0]

        if len(ligands) > 1:
            for structure in ligands[1:]:
                complex_structure += structure

    else:
        complex_structure = protein

    #Covalent ligands: strip the leaving atoms and inject the junction
    #bond/angle/dihedral terms into the merged structure BEFORE createSystem(),
    #so OpenMM generates the 1-2/1-3/1-4 exclusions from the real bond. Mirrors
    #ChemEM.protocols.core.simulation.create_system. Without this the exported
    #prmtop/xml describe a non-covalent complex and the ligand drifts off its
    #anchor residue in the ChimeraX plugin's simulation.
    if ligand_objects and any(getattr(l, "covalent_link", None) for l in ligand_objects):
        from ChemEM.parsers.covalent_fragment import inject_covalent_bonds
        inject_covalent_bonds(complex_structure, ligand_objects)

    # `solvent` is an openmm.app GB constant, or None/False for vacuum (the historical
    # default). implicitSolvent and rigidWater are mutually exclusive, matching
    # ChemEM.protocols.core.simulation.finalize_system_from_structure.
    kwargs = {
        "nonbondedMethod": NoCutoff,
        "nonbondedCutoff": 9.0 * unit.angstrom,
        "constraints": HBonds,
        "removeCMMotion": False,
    }
    if solvent:
        kwargs["implicitSolvent"] = solvent
    else:
        kwargs["rigidWater"] = True

    complex_system = complex_structure.createSystem(**kwargs)


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

    
    
