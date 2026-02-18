# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import tempfile
from pdbfixer import PDBFixer
from openmm.app import PDBFile

@dataclass(frozen=True, slots=True)
class FixerOptions:
    pH: float = 7.4
    add_missing_residues: bool = True   # conservative default
    add_hydrogens: bool = False          # prefer add-H-last in hydrogenation.py


def load_fixer(pdb_path: str) -> PDBFixer:
    return PDBFixer(pdb_path)


def run_standard_repairs(fixer: PDBFixer, opts: Optional[FixerOptions] = None) -> PDBFixer:
    """
    Run PDBFixer repairs:
      - replace nonstandard residues
      - identify missing residues/atoms
      - add missing atoms
      - (optionally) add missing hydrogens (usually False; do it later in hydrogenation.py)
    """
    if opts is None:
        opts = FixerOptions()

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    
    if not opts.add_missing_residues:
        # good idea: be explicit about policy (default is conservative)
        # PDBFixer stores missingResidues in fixer.missingResidues; setting it empty skips rebuilding them
        fixer.missingResidues = {}

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    if opts.add_hydrogens:
        fixer.addMissingHydrogens(opts.pH)

    return fixer


 
def model_to_fixer_interchange(modeller):
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp:
        PDBFile.writeFile(modeller.topology, modeller.positions, temp)
        temp.flush()
        receptor_pdbfile = PDBFixer(temp.name)
    return receptor_pdbfile


def fixer_to_model_interchange(fixer):
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp:
        PDBFile.writeFile(fixer.topology, fixer.positions, temp)
        temp.flush()
        receptor_pdbfile = PDBFile(temp.name)
    return receptor_pdbfile
