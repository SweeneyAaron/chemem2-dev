# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from pdbfixer import PDBFixer
import os
from .topology_ops import split_chains_on_breaks, rebuild_standard_bonds
from .fixer import (FixerOptions, load_fixer, model_to_fixer_interchange,
                    fixer_to_model_interchange, run_standard_repairs)
from .nucleic_acid import trim_na_5prime_phosphates
from .protonation import delete_all_hydrogens, add_hydrogens
from .determinism import PrepOptions, prep_platform
from ChemEM.parsers.parse_forcefield import forcefield_without_implicit
from ChemEM.messages import Messages
from openmm import app


def remodel_from_fixer(pdb_file, forcefield, split_chains = True, prep = None):
    """Repair and protonate a protein.

    `prep` (a PrepOptions) pins the two OpenMM minimisations in here -- PDBFixer's
    rebuilt-atom relaxation and Modeller's hydrogen placement -- to a fixed
    platform and seed, so repeated runs produce the same coordinates. See
    remodel/determinism.py for why that was not the case before.
    """
    if prep is None:
        prep = PrepOptions()

    #handel optionals

    if not os.path.exists(pdb_file):
        raise FileNotFoundError(Messages.fatal_exception(__file__, f"[ERROR] File not found {pdb_file}"))

    # One platform object for both minimisations, with its properties restored on
    # exit so a pinned thread count cannot leak into later docking Contexts.
    with prep_platform(prep) as platform:
        return _remodel(pdb_file, forcefield, split_chains, prep, platform)


def _remodel(pdb_file, forcefield, split_chains, prep, platform):
    notes = []

    fixer = load_fixer(pdb_file, platform=platform)



    if split_chains:
        #always split chains in ChemEM but including the option
        #so that this can be moved into its own package
        new_topology, new_positions = split_chains_on_breaks(fixer, 
                                                             acceptable_c_n=1.6,
                                                             acceptable_o3_p=2.2,
                                                             water_ids = None)
       
        rebuild_standard_bonds(new_topology)
        modeller = app.Modeller(new_topology, new_positions)
    else:
        modeller = app.Modeller(fixer.topology, fixer.positions)
    
    
    
    
    
    #fix non standard with pdbfixer
    new_fixer = model_to_fixer_interchange(modeller, platform=platform)


    run_standard_repairs(new_fixer, FixerOptions(
        pH=prep.pH,
        seed=prep.seed,
        clash_relief_steps=prep.clash_relief_steps,
    ))

    modeller = app.Modeller(new_fixer.topology, new_fixer.positions)
    rebuild_standard_bonds(modeller.topology)
    
    #modeling ops
   # if opts.trim_na_5prime:
    if True:
        
        ntrim = trim_na_5prime_phosphates(modeller)
        if ntrim:
            notes.append(f"Trimmed {ntrim} NA 5' phosphate group(s).")
            rebuild_standard_bonds(modeller.topology)
    
    if True:
    #if opts.add_hydrogens_last:
        ndeleted = delete_all_hydrogens(modeller)
        if ndeleted:
            notes.append(f"Deleted {ndeleted} existing hydrogen atoms before re-adding.")
        # Hydrogen placement does not need implicit solvent, and GBn2 makes its
        # minimisation several times more expensive -- see
        # forcefield_without_implicit.
        h_ff = forcefield if prep.h_placement_implicit else forcefield_without_implicit(forcefield)
        add_hydrogens(modeller, h_ff, pH=prep.pH, platform=platform,
                      seed=prep.seed if prep.deterministic else None)
    
    for n in notes:
        print(n)
    

    
    return modeller
        
        
    
    