#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:37:55 2026

@author: aaron.sweeney
"""
import os
from typing import List, Optional
from .models import Ligand, LigandList, CovalentLinkSpec
from .remodel.protonation import set_smiles_protonation_state, set_mol_protonatation_state
from .remodel.ligand_ops import (check_unassigned_chirality,
                                 set_ligand_rings,
                                 get_charged_atoms,
                                 get_aromatic_rings)

from .openff_ligand import load_ligand_structure
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import AddHs
from ChemEM.data.data import AtomType, RingType


class LigandParser:
    @staticmethod
    def load_ligands(
                     ligand_input : str, #can be sdf file or smiles string
                     protonation=True,
                     chirality=True,
                     rings=True,
                     pH=7.4,
                     pka_prec=0.0,
                     max_varients = 1,
                     name="LIG",
                     covalent_spec: Optional[CovalentLinkSpec] = None,
                     ) -> List[Ligand]:



        #load ligand from either sdf or smiles
        try:

            if os.path.exists(ligand_input):
                ligand = ligand_from_sdf(ligand_input,
                                            protonation=protonation,
                                            chirality=chirality,
                                            rings=rings,
                                            pH=pH,
                                            max_varients = 1,
                                            name= name,
                                            covalent_spec=covalent_spec)

            else:
                ligand = [ligand_from_smiles(ligand_input,
                                            protonation=protonation,
                                            chirality=chirality,
                                            rings=rings,
                                            pH=pH,
                                            max_varients = 1,
                                            name= name,
                                            covalent_spec=covalent_spec,
                                            )]

        except Exception as e:
            raise RuntimeError(f"Failed to load ligands from input {ligand_input}")
            
        
        if any([i is None for i in ligand ]):
            raise RuntimeError(f"Failed to load ligands from input {ligand_input}")
        
        print(f'[DEBUG] loaded {len(ligand)} ligands')
        print(f'[DEBUG] {ligand}')
        
        return LigandList(ligand)
    

def ligand_from_smiles(
    smiles,
    protonation=True,
    chirality=True,
    rings=True,
    pH=7.4,
    pka_prec=0.0,
    max_varients = 1,
    name="LIG",
    covalent_spec: Optional[CovalentLinkSpec] = None,
):
    smiles = Chem.CanonSmiles(smiles)
    if protonation:
        smiles = set_smiles_protonation_state(smiles, pH=pH, pka_prec=pka_prec, max_varients=max_varients)[0]
    mol_from_smi = Chem.MolFromSmiles(smiles)


    if mol_from_smi is None:
        return None

    mol = AddHs(mol_from_smi, addCoords=True)
    AllChem.EmbedMolecule(mol)


    if chirality:
        check_unassigned_chirality(mol)

    if rings:
        set_ligand_rings(mol)


    return _make_ligand_from_rd_mol(mol, smiles, name=name, covalent_spec=covalent_spec)
    
    
    
def ligand_from_sdf(sdf_file,
                    protonation=True,
                    chirality=True,
                    rings=True,
                    pH=7.4,
                    pka_prec=0.0,
                    max_varients = 1,
                    name="LIG",
                    covalent_spec: Optional[CovalentLinkSpec] = None):

    ligands = []

    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    for idx, raw_mol in enumerate(suppl):

        if raw_mol is None:
            continue

        protonated_mol = None


        if protonation:
            #protonate
            protonated_mol = set_mol_protonatation_state(raw_mol,
                                                          pH=pH,
                                                          pka_prec=pka_prec,
                                                          max_varients=max_varients)

        if protonated_mol is None:

            protonated_mol = Chem.AddHs(raw_mol, addCoords=True)

        if chirality:
            check_unassigned_chirality( protonated_mol)

        if rings:
            _ = Chem.GetSymmSSSR( protonated_mol)



        ligands.append(_make_ligand_from_rd_mol(protonated_mol, sdf_file, name,
                                                covalent_spec=covalent_spec))

    return ligands
        
            

def _make_ligand_from_rd_mol(rd_mol, source_id, name="LIG",
                             covalent_spec: Optional[CovalentLinkSpec] = None):
    # NOTE: for covalent ligands we intentionally parameterize the FULL
    # (untruncated) molecule here. OpenFF/SMIRNOFF can't parameterize an
    # open-valence radical, so leaving-group atoms are stripped from the
    # MERGED complex later (inject_covalent_bonds), not from the standalone
    # ligand. Junction-polarized charges are applied at the same stage using
    # params harvested from a capped fragment.
    ligand_charges = get_charged_atoms(rd_mol)
    openff_structure, openff_system = load_ligand_structure(rd_mol)
    openff_structure.residues[0].name = name

    atom_types = [AtomType.from_atom(a) for a in rd_mol.GetAtoms() if a.GetSymbol() != "H"]
    aromatic_rings, ring_indices = get_aromatic_rings(rd_mol)
    ring_types = [RingType.from_ring(ring) for ring in aromatic_rings]

    return Ligand(
        source_id,
        rd_mol,
        openff_system,
        openff_structure,
        atom_types,
        ring_types,
        ring_indices,
        ligand_charges,
    )








