# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

import random

from openmm.app import element as omm_element
from dimorphite_dl import protonate_smiles
from rdkit import Chem
from typing import List
from .ligand_ops import transfer_mol_coords


def delete_all_hydrogens(modeller) -> int:
    hs = [a for a in modeller.topology.atoms()
          if a.element is not None and a.element == omm_element.hydrogen]
    if hs:
        modeller.delete(hs)
    return len(hs)

def add_hydrogens(modeller, forcefield, pH=7.4, platform=None, seed=None):
    """Add hydrogens reproducibly.

    Two independent sources of run-to-run variation, both handled here:

    * `Modeller.addHydrogens` seeds each new hydrogen at a **random** offset from
      its parent before relaxing it -- ``delta = Vec3(random.random(), ...)`` in
      openmm/app/modeller.py, whose own comment reads "The hydrogens were added at
      random positions". That draws from Python's global `random`, so seeding it
      makes the starting geometry reproducible. The global RNG state is saved and
      restored so seeding cannot perturb anything else in the process.
    * It then minimises in an OpenMM Context, which is only reproducible on a
      fixed platform.

    Both matter: the ECHO H-bond term reads protein hydrogen positions.
    """
    if seed is None:
        modeller.addHydrogens(forcefield, pH=pH, platform=platform)
        return

    state = random.getstate()
    try:
        random.seed(int(seed))
        modeller.addHydrogens(forcefield, pH=pH, platform=platform)
    finally:
        random.setstate(state)


def set_smiles_protonation_state(smi, pH=7.4, pka_prec=1.0, max_varients=128) -> List[str]:
    
    
    if isinstance(pH, (list, tuple)):
        if len(pH) != 2:
            raise ValueError("pH as list must be of lenght 2 [min_ph, max_ph]")
        pH = sorted(pH)
        ph_min = pH[0]
        ph_max = pH[1]
    
    elif isinstance(pH, (float, int)):
        ph_min = pH 
        ph_max = pH 
    else:
        raise ValueError(f"pH must be list[float | int ] | tuple[float | int] | int | float not  {type(pH)}")
    
    dimorphite_dl = protonate_smiles(
        smi,
        ph_min=ph_min,
        ph_max=ph_max,
        max_variants=max_varients,
        label_states=False,
        precision=pka_prec,
    )
    return dimorphite_dl

def set_mol_protonatation_state(mol, pH=7.0, pka_prec=1.0, max_varients=128):
    
    
    smiles = set_smiles_protonation_state(Chem.MolToSmiles(mol), pH=pH, pka_prec=pka_prec, max_varients=max_varients)
    
    if smiles is None:
        print(f"Can't protonate smiles {smiles} from rdkit mol")
        return Chem.AddHs(mol, addCoords=True)
    
    if  isinstance(smiles, list) or isinstance(smiles, tuple):
        smiles = smiles[0]

    mol_protonated =  Chem.MolFromSmiles(smiles)
    
    if mol_protonated is None:
        print(f"Can't protonate smiles {smiles} from rdkit mol")
        return Chem.AddHs(mol, addCoords=True)
    
    
    mol_protonated_noH = Chem.RemoveHs(mol_protonated)
    mol_noH = Chem.RemoveHs(mol)

    # Nothing to do: dimorphite returned the state the input is already in.
    # Short-circuiting keeps the input atom order and any existing hydrogen
    # positions, which matters when the input is an already-prepared pose
    # (e.g. a docking output being re-loaded for refinement).
    if Chem.MolToSmiles(mol_noH) == Chem.MolToSmiles(mol_protonated_noH):
        return Chem.AddHs(mol, addCoords=True)

    # No embedding here: transfer_mol_coords allocates the conformer it needs
    # and overwrites every position, so ETKDG would only add a failure mode
    # (it returns -1 on large flexible ligands stripped of their hydrogens).
    mol_protonated_noH = transfer_mol_coords(mol_noH, mol_protonated_noH)

    if mol_protonated_noH is None:
        print(f"Can't transfer coordinates to protonated smiles {smiles} from rdkit mol")
        return Chem.AddHs(mol, addCoords=True)

    # Add Hs back with coords
    protonated_mol = Chem.AddHs(mol_protonated_noH, addCoords=True)

    return protonated_mol

    