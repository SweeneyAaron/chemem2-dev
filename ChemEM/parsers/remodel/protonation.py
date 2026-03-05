# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from openmm.app import element as omm_element
from dimorphite_dl import protonate_smiles
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List
from .ligand_ops import transfer_mol_coords


def delete_all_hydrogens(modeller) -> int:
    hs = [a for a in modeller.topology.atoms()
          if a.element is not None and a.element == omm_element.hydrogen]
    if hs:
        modeller.delete(hs)
    return len(hs)

def add_hydrogens(modeller, forcefield, pH=7.4):
    modeller.addHydrogens(forcefield, pH=pH)


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
    AllChem.EmbedMolecule(mol_protonated_noH, randomSeed=0xF00D)
    mol_noH = Chem.RemoveHs(mol)
    
    mol_protonated_noH = transfer_mol_coords(mol_noH, mol_protonated_noH)
    protonated_mol = Chem.AddHs(mol_protonated_noH, addCoords=True)
    
    # Add Hs back with coords
    return protonated_mol

    