# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from openmm.app import element as omm_element

def delete_all_hydrogens(modeller) -> int:
    hs = [a for a in modeller.topology.atoms()
          if a.element is not None and a.element == omm_element.hydrogen]
    if hs:
        modeller.delete(hs)
    return len(hs)

def add_hydrogens(modeller, forcefield, pH=7.4):
    modeller.addHydrogens(forcefield, pH=pH)
    