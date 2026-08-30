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

from .determinism import (DEFAULT_CLASH_RELIEF_STEPS, DEFAULT_PREP_SEED,
                          bounded_clash_relief)


@dataclass(frozen=True, slots=True)
class FixerOptions:
    pH: float = 7.4
    add_hydrogens: bool = False          # prefer add-H-last in hydrogenation.py
    # Seed for the Langevin dynamics PDBFixer runs on newly-added atoms. Left
    # unset it defaults to 0, which OpenMM reads as "pick a fresh seed per
    # Context" -- the reason prep coordinates used to differ on every run.
    seed: int = DEFAULT_PREP_SEED
    # Step budget for PDBFixer's clash-relief dynamics; see bounded_clash_relief.
    clash_relief_steps: int = DEFAULT_CLASH_RELIEF_STEPS


def load_fixer(pdb_path: str, platform=None) -> PDBFixer:
    return PDBFixer(pdb_path, platform=platform)


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

    # Must be called even though it always returns {} here: findMissingAtoms reads
    # self.missingResidues to decide about 5' phosphates and OXT. It is always
    # empty because model_to_fixer_interchange round-trips through a temp PDB and
    # PDBFile writes no SEQRES, so there is no sequence to diff against -- i.e.
    # ChemEM never rebuilds missing loops, only incomplete side chains.
    fixer.findMissingResidues()

    fixer.findMissingAtoms()
    # An explicit seed is what makes the rebuilt-atom coordinates reproducible;
    # the platform (set on the PDBFixer in model_to_fixer_interchange) is the
    # other half. Both are needed -- see remodel/determinism.py. The step budget
    # caps the clash-relief dynamics, which dominates preparation cost.
    with bounded_clash_relief(opts.clash_relief_steps):
        fixer.addMissingAtoms(seed=opts.seed)

    if opts.add_hydrogens:
        fixer.addMissingHydrogens(opts.pH)

    return fixer



def model_to_fixer_interchange(modeller, platform=None):
    """Round-trip a Modeller into a PDBFixer via a temp PDB.

    This is the PDBFixer instance `run_standard_repairs` actually operates on, so
    it is the one whose platform decides whether the repair is reproducible.
    """
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp:
        PDBFile.writeFile(modeller.topology, modeller.positions, temp, keepIds=True)
        temp.flush()
        receptor_pdbfile = PDBFixer(temp.name, platform=platform)
    return receptor_pdbfile


def fixer_to_model_interchange(fixer):
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp:
        PDBFile.writeFile(fixer.topology, fixer.positions, temp, keepIds=True)
        temp.flush()
        receptor_pdbfile = PDBFile(temp.name)
    return receptor_pdbfile
