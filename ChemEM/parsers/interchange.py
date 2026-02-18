# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

#conversion from openmm to other things 
import parmed 

from openmm import app
from ChemEM.messages import Messages



def deduplicate_topology_bonds(top):
    new = app.Topology()
    atom_map = {}

    for chain in top.chains():
        new_chain = new.addChain(chain.id)
        for res in chain.residues():
            
            try:
                new_res = new.addResidue(res.name, new_chain, id=res.id)
            except TypeError:
                new_res = new.addResidue(res.name, new_chain)

            for atom in res.atoms():
                new_atom = new.addAtom(atom.name, atom.element, new_res, id=atom.id)
                atom_map[atom] = new_atom

    # Add only unique bonds
    seen = set()
    for a1, a2 in top.bonds():
        key = tuple(sorted((a1.index, a2.index)))
        if key in seen:
            continue
        seen.add(key)
        new.addBond(atom_map[a1], atom_map[a2])

    
    vecs = top.getPeriodicBoxVectors()
    if vecs is not None:
        new.setPeriodicBoxVectors(vecs)

    return new


def modeller_to_parmed(modeller, forcefield):
    #need to depupe the bonds as we call createStandard bonds 
    #in some versions this is alreday called and can result in duplicate bonds
    topo = deduplicate_topology_bonds(modeller.topology)
    modeller = app.Modeller(topo, modeller.positions)
    system = forcefield.createSystem(modeller.topology)
    receptor_structure = parmed.openmm.load_topology(
        modeller.topology, system, xyz=modeller.positions
    )

    n_missing = sum(b.type is None for b in receptor_structure.bonds)
    
    if n_missing:
        raise RuntimeError(
            f"ParmEd conversion left {n_missing} bonds untyped. "
            "This usually indicates duplicate or mismatched bonds in the OpenMM topology."
        )
         
        
    return receptor_structure, system
