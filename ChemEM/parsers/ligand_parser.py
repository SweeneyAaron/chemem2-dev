# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

# ChemEM/parsers/ligand_parser.py
from __future__ import annotations

import sys
import io
import os
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import AddHs

from openmm.app import PDBFile
import parmed

from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.interchange import Interchange

from dimorphite_dl import protonate_smiles

from ChemEM.data.data import AtomType, RingType
from .models import Ligand
from ChemEM.tools.ligand import get_aromatic_rings


class LigandParser:
    # ----------------- PUBLIC API -----------------

    @staticmethod
    def load_ligands(
        ligand_input,
        protonation=True,
        chirality=True,
        rings=True,
        pH=[6.4, 8.4],
        pka_prec=1.0,
        name="LIG",
        _no_exit=False,
    ):
        """
        Always returns a list of Ligand objects.

        - If ligand_input is a .sdf path -> one Ligand per SDF record.
        - Otherwise treat ligand_input as SMILES -> single Ligand in a list.
        """
        ligands = []

        try:
            if os.path.isfile(ligand_input) and ligand_input.lower().endswith(".sdf"):
                ligands = LigandParser._load_ligands_from_sdf_file(
                    ligand_input,
                    protonation=protonation,
                    chirality=chirality,
                    rings=rings,
                    pH=pH,
                    pka_prec=pka_prec,
                    name=name,
                    _no_exit=_no_exit,
                )
            else:
                lig = LigandParser._load_single_from_smi(
                    ligand_input,
                    protonation=protonation,
                    chirality=chirality,
                    rings=rings,
                    pH=pH,
                    pka_prec=pka_prec,
                    name=name,
                )
                if lig is not None:
                    ligands = [lig]
        except Exception as e:
            print(f"\nChemEM Fatal Error: Could not read ligand input: {ligand_input}")
            print(f"\tError: {e}")
            if not _no_exit:
                sys.exit()
            return []

        if not ligands:
            print(f"\nChemEM Fatal Error: No valid ligands loaded from {ligand_input}")
            if not _no_exit:
                sys.exit()
        #import pdb 
        #pdb.set_trace()
        return ligands

    @staticmethod
    def load_ligand(
        ligand_input,
        protonation=True,
        chirality=True,
        rings=True,
        pH=[6.4, 8.4],
        pka_prec=1.0,
        name="LIG",
        _no_exit=False,
    ):
        """
        Backwards-compatible wrapper: returns a single Ligand.
        If multiple ligands found (multi-record SDF), returns the first with a warning.
        """
        ligands = LigandParser.load_ligands(
            ligand_input,
            protonation=protonation,
            chirality=chirality,
            rings=rings,
            pH=pH,
            pka_prec=pka_prec,
            name=name,
            _no_exit=_no_exit,
        )

        if not ligands:
            return None

        if len(ligands) > 1:
            print(
                f"ChemEM - Warning: {len(ligands)} ligands found in input "
                f"{ligand_input}. Returning the first one only."
            )
        return ligands[0]

    # ----------------- INTERNAL HELPERS -----------------

    @staticmethod
    def _make_ligand_from_rd_mol(rd_mol, source_id, name="LIG"):
        ligand_charges = LigandParser.get_charged_atoms(rd_mol)
        openff_structure, openff_system = LigandParser.load_ligand_structure(rd_mol)
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

    @staticmethod
    def _load_single_from_smi(
        smiles,
        protonation=True,
        chirality=True,
        rings=True,
        pH=[6.4, 8.4],
        pka_prec=1.0,
        name="LIG",
    ):
        rd_mol = LigandParser.mol_from_smiles(
            smiles,
            protonation=protonation,
            pH=pH,
            n=pka_prec,
        )
        if rd_mol is None:
            return None

        if chirality:
            LigandParser.check_unassigned_chirality(rd_mol)

        if rings:
            try:
                _ = Chem.GetSymmSSSR(rd_mol)
            except Exception as e:
                print(
                    "ChemEM- Non-Fatal warning ring info assignment failed "
                    f"with GetSymmSSSR. Full Error: {e}"
                )

        return LigandParser._make_ligand_from_rd_mol(rd_mol, source_id=smiles, name=name)

    @staticmethod
    def _load_ligands_from_sdf_file(
        sdf_file,
        protonation=True,
        chirality=True,
        rings=True,
        pH=[6.4, 8.4],
        pka_prec=1.0,
        name="LIG",
        _no_exit=False,
    ):
        ligands = []
        exceptions = []

        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)

        for idx, raw_mol in enumerate(suppl):
            if raw_mol is None:
                continue

            try:
                rd_mol = LigandParser.mol_from_sdf(
                    sdf_file,
                    conf_num=idx,
                    protonation=protonation,
                    pH=pH,
                    n=pka_prec,
                )
                if rd_mol is None:
                    continue

                if chirality:
                    LigandParser.check_unassigned_chirality(rd_mol)

                if rings:
                    try:
                        _ = Chem.GetSymmSSSR(rd_mol)
                    except Exception as e:
                        print(
                            "ChemEM- Non-Fatal warning ring info assignment "
                            f"failed with GetSymmSSSR for record {idx}. Full Error: {e}"
                        )

                source_id = f"{sdf_file}#{idx}"
                ligands.append(LigandParser._make_ligand_from_rd_mol(rd_mol, source_id=source_id, name=name))

            except Exception as e:
                exceptions.append((idx, e))
                if not _no_exit:
                    print(f"\nChemEM Error: Failed to load record {idx} from {sdf_file}")
                    print(f"\tError: {e}")
                    raise

        if not ligands and exceptions:
            print(f"\nChemEM Fatal Error: No valid ligands loaded from {sdf_file}")
            for idx, e in exceptions:
                print(f"  Record {idx} failed with Error:\n\t{e}")
            if not _no_exit:
                sys.exit()

        return ligands

    # ----------------- RDKit helpers -----------------

    @staticmethod
    def get_charged_atoms(rd_mol):
        return [(atom.GetIdx(), atom.GetFormalCharge()) for atom in rd_mol.GetAtoms()]

    @staticmethod
    def load_ligand_from_mol(mol, name="LIG", chirality=True, rings=True):
        if chirality:
            LigandParser.check_unassigned_chirality(mol)
        if rings:
            _ = Chem.GetSymmSSSR(mol)

        openff_structure, openff_system = LigandParser.load_ligand_structure(mol)
        openff_structure.residues[0].name = name

        atom_types = [AtomType.from_atom(i) for i in mol.GetAtoms() if i.GetSymbol() != "H"]
        aromatic_rings, ring_indices = get_aromatic_rings(mol)
        ring_types = [RingType.from_ring(ring) for ring in aromatic_rings]

        return Ligand(
            Chem.MolToSmiles(mol),
            mol,
            openff_system,
            openff_structure,
            atom_types,
            ring_types,
            ring_indices,
            ligand_charges=[],
        )

    @staticmethod
    def get_rd_mol(
        mol_input,
        input_type,
        protonation=True,
        chirality=True,
        rings=True,
        pH=[6.4, 8.4],
        pka_prec=1.0,
    ):
        if input_type == "smi":
            mol = LigandParser.mol_from_smiles(mol_input, protonation=protonation, pH=pH, n=pka_prec)
        elif input_type == "sdf":
            mol = LigandParser.mol_from_sdf(mol_input)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        if mol is None:
            return None

        if chirality:
            LigandParser.check_unassigned_chirality(mol)

        if rings:
            try:
                _ = Chem.GetSymmSSSR(mol)
            except Exception as e:
                print(
                    "ChemEM- Non-Fatal warning ring info assignment failed with GetSymmSSSR.\n"
                    f"Full Error: {e}"
                )

        return mol

    @staticmethod
    def mol_from_smiles(smiles, protonation=True, pH=[6.4, 8.4], n=1.0):
        smiles = Chem.CanonSmiles(smiles)

        if protonation:
            try:
                smiles = LigandParser.protonate(smiles, pH=pH, n=n)[0]
            except Exception as e:
                print(f"ChemEM - Warning: Failed to protonate mol {smiles}.\n Full Error: {e}")

        mol_from_smi = Chem.MolFromSmiles(smiles)
        if mol_from_smi is None:
            return None

        mol = AddHs(mol_from_smi, addCoords=True)
        AllChem.EmbedMolecule(mol)
        return mol

    @staticmethod
    def mol_from_sdf(sdf_file, conf_num=0, protonation=True, pH=[6.4, 8.4], n=1.0):
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
        mol = suppl[conf_num]
        if mol is None:
            return None

        if not protonation:
            return Chem.AddHs(mol, addCoords=True)

        # Protonate via SMILES and then transfer heavy-atom coords from original
        smiles = Chem.MolToSmiles(mol)
        try:
            prot_smiles = LigandParser.protonate(smiles, pH=pH, n=n)[0]
        except Exception as e:
            print(f"ChemEM - Warning: Failed to protonate mol {smiles}.\n Full Error: {e}")
            return mol

        prot_base = Chem.MolFromSmiles(prot_smiles)
        if prot_base is None:
            print(f"Warning: Could not parse protonated SMILES '{prot_smiles}'.")
            return mol

        # Embed protonated molecule (no H first), then map coords
        prot_noH = Chem.RemoveHs(prot_base)
        AllChem.EmbedMolecule(prot_noH, randomSeed=0xF00D)

        original_noH = Chem.RemoveHs(mol)
        match = prot_noH.GetSubstructMatch(original_noH)
        if not match:
            print("Warning: Could not match the heavy-atom skeletons. Returning unprotonated mol.")
            return mol

        conf_original = mol.GetConformer()
        conf_prot = prot_noH.GetConformer()

        # Map heavy atoms: original_noH atom i corresponds to prot_noH atom match[i]
        orig_heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
        prot_heavy = [a.GetIdx() for a in prot_noH.GetAtoms() if a.GetSymbol() != "H"]

        for i, prot_noH_idx in enumerate(match):
            orig_atom_idx = orig_heavy[i]
            prot_atom_idx = prot_heavy[prot_noH_idx]
            conf_prot.SetAtomPosition(prot_atom_idx, conf_original.GetAtomPosition(orig_atom_idx))

        # Add Hs back with coords
        prot_mol = Chem.AddHs(prot_noH, addCoords=True)
        return prot_mol

    @staticmethod
    def check_unassigned_chirality(mol):
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        unassigned = [i for i in chiral_centers if i[1] == "?"]

        if unassigned:
            print("ChemEM - Warning: Unassigned chirality found; assigning chirality from structure")
            print(f'\tMolecule Chirality (atom_idx, "S"|"R"|"?") : {chiral_centers}')
            try:
                Chem.AssignAtomChiralTagsFromStructure(mol)
                chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
                print(f'\tNew Molecule Chirality (atom_idx, "S"|"R"|"?") : {chiral_centers}')
            except Exception as e:
                print(f"Unable to assign chirality from structure failed with error:\n\t{e}")
                print("Attempting to load ligand with unassigned chirality.")

    @staticmethod
    def protonate(smi, pH=[6.4, 8.4], n=1.0):
        dimorphite_dl = protonate_smiles(
            smi,
            ph_min=pH[0],
            ph_max=pH[1],
            max_variants=128,
            label_states=False,
            precision=n,
        )

        middle_ph = sum(pH) / 2
        most_probable = protonate_smiles(
            smi,
            ph_min=middle_ph,
            ph_max=middle_ph,
            max_variants=1,
            label_states=False,
            precision=0.0,
        )

        display_message = False
        if most_probable and dimorphite_dl:
            return_smi = [most_probable[0]] + [i for i in dimorphite_dl if i != most_probable[0]]
            display_message = True
        elif dimorphite_dl:
            return_smi = dimorphite_dl
            display_message = True
        else:
            return_smi = [smi]

        if display_message:
            if len(return_smi) == 1:
                print(f"Continuing with ligand smiles: {return_smi[0]}")
            else:
                print(f"Multiple Protonation states found for molecule {smi} :")
                for s in return_smi:
                    print(f"\t{s}\n")
                print(f"Continuing calculations with first entry:\n\t{return_smi[0]}")
                print('\nTo specify a protonation state apply chemem.Protonation on your conf file.')
                print('To disable protonation set "protonation = 0" in your configuration file.')
                print("See documentation for further details (https://chemem.topf-group.com/)")

        return return_smi

    # ----------------- OpenFF -> OpenMM conversion -----------------

    @staticmethod
    def load_ligand_structure(molecule):
        ligand_off_molecule = Molecule.from_rdkit(molecule)
        ligand_off_molecule.assign_partial_charges("mmff94")

        force_field = ForceField("openff_unconstrained-2.0.0.offxml")
        interchange = Interchange.from_smirnoff(
            topology=[ligand_off_molecule],
            force_field=force_field,
            charge_from_molecules=[ligand_off_molecule],
        )

        ligand_system = interchange.to_openmm()

        virtual_file = io.StringIO(Chem.MolToPDBBlock(molecule))
        ligand_pdbfile = PDBFile(virtual_file)
        ligand_structure = parmed.openmm.load_topology(
            ligand_pdbfile.topology, ligand_system, xyz=ligand_pdbfile.positions
        )

        return ligand_structure, ligand_system
