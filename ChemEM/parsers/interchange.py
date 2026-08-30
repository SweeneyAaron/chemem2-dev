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
from .remodel.topology_ops import _WATER



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

    # Water O-H bonds are legitimately untyped here and must NOT trip the guard.
    # createSystem defaults to rigidWater=True, which replaces them with constraints rather
    # than HarmonicBondForce terms, so ParmEd finds no type to attach. That is precisely what
    # ensure_water_geometry_types() patches -- but it runs on the structure this function
    # returns, so raising here made it unreachable for every input containing water, i.e. for
    # exactly the inputs it exists to fix. The guard's real job is catching duplicate or
    # mismatched bonds from a double createStandardBonds(), and it still does that for
    # everything else.
    untyped = [b for b in receptor_structure.bonds
               if b.type is None
               and not (b.atom1.residue.name.upper() in _WATER
                        and b.atom2.residue.name.upper() in _WATER)]

    if untyped:
        offenders = sorted({f"{b.atom1.residue.name}{b.atom1.residue.number}"
                            for b in untyped})
        raise RuntimeError(
            f"ParmEd conversion left {len(untyped)} non-water bonds untyped "
            f"(residues: {', '.join(offenders[:10])}). "
            "This usually indicates duplicate or mismatched bonds in the OpenMM topology."
        )


    return receptor_structure, system
