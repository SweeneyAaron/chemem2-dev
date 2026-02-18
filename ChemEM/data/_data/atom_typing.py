# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>



from enum import Enum

from rdkit.Chem.rdchem import BondType
from rdkit import Chem
from .system_constants import PROTEIN_RINGS


class RingType(Enum):

    def __init__(self, idx, ring_name, atom_composition):
        #maybe discount hydrogens due to charge states
        self.idx = idx
        self.ring_name = ring_name
        self.atom_composition = atom_composition

    BENZENE = (0, "Benzene", [("C",6)])
    FURAN = (1, "Furan", [("C", 4),("O", 1)])
    THIOPHENE = (2, "Thiophene", [("C", 4), ("S",1)])
    IMIDAZOLE = (3, "Imidazole", [("C", 3), ("N", 2)])
    PYRIDINE = (4, "Pyridine", [("C",5), ("N",1)])
    PYRROLE = (5, "Pyrrole", [("C",4), ("N",1)])
    DEFAULT = (0, "Default", {})

    @classmethod
    def from_ring(cls, ring_atoms):
        """
        Determine the ring type based on the atoms that form the ring.
        """
        composition = {}
        for atom in ring_atoms:
            symbol = atom.GetSymbol()  # Assuming an RDKit-like API.
            composition[symbol] = composition.get(symbol, 0) + 1

        composition = sorted([(k,v) for k,v in composition.items()], key = lambda x: x[0])

        for ring_type in cls:
            if composition == ring_type.atom_composition:
                return ring_type

        return cls.DEFAULT

    @classmethod
    def from_residue_name(cls, residue_name):
        protein_ring_data = PROTEIN_RINGS[residue_name]
        protein_rings = []
        for ring_name, _ in protein_ring_data:
            for ring_type in cls:
                if ring_name == ring_type.ring_name:
                    protein_rings.append(ring_type)
        return protein_rings


#43 -- current max
class AtomType(Enum):

    def __init__(self, idx, symbol, bonds, hydrogens):
        self.idx = idx
        self.symbol = symbol
        self.bonds = bonds
        self.hydrogens = hydrogens

    BROMINE = (0,'Br',[BondType.SINGLE], 0)
    CHLORINE = (1, 'Cl', [BondType.SINGLE], 0)
    FLUORINE = (2,'F', [BondType.SINGLE], 0)
    IODINE = (3, 'I', [BondType.SINGLE], 0)
    CARBON_CH3 = (4, 'C', [BondType.SINGLE] * 4, 3)
    CARBON_AROMATIC = (5, 'C', [BondType.AROMATIC, BondType.AROMATIC, BondType.SINGLE], 1) #need one withour hydrogens
    #CARBON_AROMATIC = (8, 'C', [BondType.AROMATIC, BondType.AROMATIC, BondType.SINGLE], 1) #need one withour hydrogens

    CARBON_CH2 = (6, 'C', [BondType.DOUBLE, BondType.SINGLE, BondType.SINGLE], 2)
    CARBON_BONDED_1 = (7,'C', [BondType.SINGLE] * 4, 2)
    CARBON_BONDED_1_DOUBLE_BOND = (8, 'C', [BondType.DOUBLE, BondType.SINGLE, BondType.SINGLE], 1)
    CARBON_BONDED_2 = (9,'C', [BondType.SINGLE] * 4, 1)
    CARBON_BONDED_3 = (10, 'C', [BondType.SINGLE] * 4, 0)
    CARBON_TRIPLE_BOND = (11, 'C', [BondType.TRIPLE, BondType.SINGLE], 1)
    CARBON_TRIPLE_BOND_BONDED = (12, 'C', [BondType.TRIPLE, BondType.SINGLE], 0)#new same as 10
    NITROGEN = (13, 'N', [BondType.SINGLE] * 3, 2)
    NITROGEN_AROMATIC = (14, 'N', [BondType.AROMATIC, BondType.AROMATIC], 0)
    NITROGEN_BONDED_1 = (15, 'N', [BondType.SINGLE] * 3, 1)

    #NITROGEN_DOUBLE_BOND = (16, 'N', [BondType.DOUBLE, BondType.SINGLE], 1) #new same as 14
    NITROGEN_DOUBLE_BOND = (13, 'N', [BondType.DOUBLE, BondType.SINGLE], 1)

    NITROGEN_TRPNE1 = (43, 'N', [BondType.DOUBLE, BondType.SINGLE], 1)
    NITROGEN_ASNGLN = (43, 'N', [BondType.SINGLE] * 3, 2)

    NITROGEN_TRIPLE_BOND = (17, 'N', [BondType.TRIPLE], 0)
    NITROGEN_BONDED_3 = (18, 'N', [BondType.SINGLE] * 3, 0)
    OXYGEN = (19, 'O', [BondType.SINGLE] * 2, 1)
    OXYGEN_AROMATIC = (20, 'O', [BondType.AROMATIC] * 2, 0)
    OXYGEN_DOUBLE_BOND = (21, 'O', [BondType.DOUBLE], 0)
    OXYGEN_BONDED = (22, 'O', [BondType.SINGLE] * 2, 0)
    PHOSPHORUS = (23, 'P', [None], None) # specil means just assign it to atom
    SULPHUR = (24, 'S', [BondType.SINGLE] * 2, 1)
    SULPHUR_AROMATIC = (25, 'S', [BondType.AROMATIC] * 2, 0)
    SULPHUR_DOUBLE_BOND = (26, 'S', [BondType.DOUBLE], 0)
    SULPHUR_BONDED = (27, 'S', [BondType.SINGLE] * 2, 0)
    WATER = (28, 'O', [BondType.SINGLE] * 2, 2)
    NITROGEN_DOUBLE_BOND_BONDED = (29, 'N', [BondType.SINGLE, BondType.DOUBLE], 0)
    CARBON_BONDED_4_DOUBLE_BOND = (30, 'C', [BondType.DOUBLE, BondType.DOUBLE], 0) #should be same as 8
    CARBON_BONDED_5_DOUBLE_BOND = (31, 'C', [BondType.DOUBLE, BondType.SINGLE, BondType.SINGLE], 0) #same as 8

    CHARGED_NITROGEN_1 = (15, 'N', [BondType.SINGLE] * 4, 1) #same as 13 for now.
    CHARGED_NITROGEN_8 = (15, 'N', [BondType.SINGLE] * 4, 0) #same as 13 for now.
    CHARGED_NITROGEN_2 = (15, 'N', [BondType.SINGLE] * 4, 2) #same as 13 for now.
    CHARGED_NITROGEN_3 = (13, 'N', [BondType.SINGLE] * 4, 3) #same as 13 for now.

    #CHARGED_NITROGEN_4 = (35, 'N', [BondType.SINGLE, BondType.SINGLE, BondType.DOUBLE], 3) #same as 13 for now.
    CHARGED_NITROGEN_4 = (13, 'N', [BondType.SINGLE, BondType.SINGLE, BondType.DOUBLE], 3)
    CHARGE_OXYGEN = (19, 'O', [BondType.SINGLE], 0) #same as 19, changed from 36 due to issues

    #new types that go over other types!!
    AMIDE_N = (37, 'N', [BondType.SINGLE] * 3, 1)
    AMIDE_O = (38, 'O',[BondType.DOUBLE], 0)
    PEPTIDE_O = (39, 'O',[BondType.DOUBLE], 0)
    PEPTIDE_N = (40, 'N', [BondType.SINGLE] * 3, 1)
    PO4_OXYGEN_DOUBLE_BOND = (41, 'O',[BondType.DOUBLE], 0)
    PO4_OXYGEN_DONOR = (42, 'O', [BondType.SINGLE] * 2, 1)

    CHARGED_NITROGEN_5 = (13, 'N', [BondType.SINGLE, BondType.SINGLE, BondType.DOUBLE], 2) #same as 13 for now.
    CHARGED_NITROGEN_6 = (15, 'N', [BondType.SINGLE, BondType.SINGLE, BondType.DOUBLE], 1) #same as 13 for now.
    CHARGED_NITROGEN_7 = (15, 'N', [BondType.SINGLE, BondType.SINGLE, BondType.DOUBLE], 0) #same as 13 for now.
    CHARGED_NITROGEN_AROMATIC = (43, 'N', [BondType.AROMATIC, BondType.AROMATIC,BondType.SINGLE], 1)
    #No matching AtomType for symbol=S, bonds=[rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE], H=0
    SULPHUR_SULPONATE_GENERAL = ( 27, 'S', [BondType.SINGLE, BondType.SINGLE, BondType.DOUBLE, BondType.DOUBLE], 0 )

    @staticmethod
    def has_single_bonded_carbon(atom, skip_atom=None):
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() == 'C' and nbr != skip_atom:
                bond = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
                if bond and bond.GetBondType() == BondType.SINGLE:
                    return True
        return False

    @staticmethod
    def is_peptide_o(atom):
        if atom.GetSymbol() != 'O':
            return False
        for c in atom.GetNeighbors():
            if c.GetSymbol() == 'C':
                bondOC = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), c.GetIdx())
                if bondOC and bondOC.GetBondType() == BondType.DOUBLE:
                    for n in c.GetNeighbors():
                        if n.GetSymbol() == 'N':
                            bondCN = c.GetOwningMol().GetBondBetweenAtoms(c.GetIdx(), n.GetIdx())
                            if bondCN and bondCN.GetBondType() == BondType.SINGLE:
                                if AtomType.has_single_bonded_carbon(c, skip_atom=atom) and \
                                   AtomType.has_single_bonded_carbon(n, skip_atom=c):
                                    return True
        return False

    @staticmethod
    def is_peptide_n(atom):
        if atom.GetSymbol() != 'N':
            return False
        for c in atom.GetNeighbors():
            if c.GetSymbol() == 'C':
                bondCN = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), c.GetIdx())
                if bondCN and bondCN.GetBondType() == BondType.SINGLE:
                    for o in c.GetNeighbors():
                        if o.GetSymbol() == 'O':
                            bondCO = c.GetOwningMol().GetBondBetweenAtoms(c.GetIdx(), o.GetIdx())
                            if bondCO and bondCO.GetBondType() == BondType.DOUBLE:
                                if AtomType.has_single_bonded_carbon(c, skip_atom=atom) and \
                                   AtomType.has_single_bonded_carbon(atom, skip_atom=c):
                                    return True
        return False

    @staticmethod
    def is_amide_o(atom):
        if atom.GetSymbol() != 'O':
            return False
        for c in atom.GetNeighbors():
            if c.GetSymbol() == 'C':
                bondOC = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), c.GetIdx())
                if bondOC and bondOC.GetBondType() == BondType.DOUBLE:
                    for n in c.GetNeighbors():
                        if n.GetSymbol() == 'N':
                            bondCN = c.GetOwningMol().GetBondBetweenAtoms(c.GetIdx(), n.GetIdx())
                            if bondCN and bondCN.GetBondType() == BondType.SINGLE:
                                return True
        return False

    @staticmethod
    def is_amide_n(atom):
        if atom.GetSymbol() != 'N':
            return False
        for c in atom.GetNeighbors():
            if c.GetSymbol() == 'C':
                bondCN = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), c.GetIdx())
                if bondCN and bondCN.GetBondType() == BondType.SINGLE:
                    for o in c.GetNeighbors():
                        if o.GetSymbol() == 'O':
                            bondCO = c.GetOwningMol().GetBondBetweenAtoms(c.GetIdx(), o.GetIdx())
                            if bondCO and bondCO.GetBondType() == BondType.DOUBLE:
                                return True
        return False

    @staticmethod
    def is_po4_oxygen_double_bond(atom):
        if atom.GetSymbol() != 'O':
            return False
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() == 'P':
                bond = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
                if bond and bond.GetBondType() == BondType.DOUBLE:
                    return True
        return False

    @staticmethod
    def is_po4_oxygen_donor(atom):
        if atom.GetSymbol() != 'O':
            return False
        single_bonded_p = False
        single_bonded_h = False
        for nbr in atom.GetNeighbors():
            bond = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
            if bond and bond.GetBondType() == BondType.SINGLE:
                if nbr.GetSymbol() == 'P':
                    single_bonded_p = True
                elif nbr.GetSymbol() == 'H':
                    single_bonded_h = True
        return (single_bonded_p and single_bonded_h)

    @staticmethod
    def from_atom(atom):
        PROTEIN_TYPES = [AtomType.NITROGEN_TRPNE1, AtomType.NITROGEN_ASNGLN]
        if AtomType.is_peptide_o(atom):
            return AtomType.PEPTIDE_O
        if AtomType.is_peptide_n(atom):
            return AtomType.PEPTIDE_N
        if AtomType.is_amide_o(atom):
            return AtomType.AMIDE_O
        if AtomType.is_amide_n(atom):
            return AtomType.AMIDE_N
        if AtomType.is_po4_oxygen_double_bond(atom):
            return AtomType.PO4_OXYGEN_DOUBLE_BOND
        if AtomType.is_po4_oxygen_donor(atom):
            return AtomType.PO4_OXYGEN_DONOR

        element = atom.GetSymbol()
        bonds = sorted([b.GetBondType() for b in atom.GetBonds()])
        num_hydrogens = sum(1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'H')
        atom_is_aromatic = atom.GetIsAromatic()

        for atom_type in AtomType:
            if atom_type in PROTEIN_TYPES:
                continue
            if (atom_type.symbol == element and
                len(atom_type.bonds) == len(bonds) and
                all(b1 == b2 for b1, b2 in zip(sorted(atom_type.bonds), bonds)) and
                atom_type.hydrogens == num_hydrogens):
                return atom_type

            elif atom_type.symbol == element and atom_is_aromatic:
                if BondType.AROMATIC in atom_type.bonds:
                    return atom_type

            elif atom_type.symbol == 'P' and element == 'P':
                return AtomType.PHOSPHORUS

        print(f"No matching AtomType for symbol={element}, bonds={bonds}, H={num_hydrogens}")
        return None

    @staticmethod
    def from_id(res_name, atom_name):
        try:
            atom_type = atom_data[res_name][atom_name]
            return atom_type
        except KeyError:
            import pdb 
            pdb.set_trace()
            print(f"Warning: Could not find type for atom '{atom_name}' in residue '{res_name}'.")
            return None


# Nucleic acid residue -> atom name -> AtomType
# Notes:
#  - Includes both prime (') and legacy asterisk (*) sugar atom aliases (e.g. "C1'" and "C1*")
#  - Includes both OPn and OnP phosphate aliases (e.g. OP1 and O1P)
#  - Phosphate non-bridging oxygens (OP1/OP2) are mapped to CHARGE_OXYGEN (common in polymer context)
#  - O3'/O5' are mapped as OXYGEN_BONDED (bridging/ether-like); termini may differ if you care about OH vs ester


nucleic_atom_data = {
    # ----------------
    # RNA linking: A,C,G,U
    # ----------------
    "A": {
        # phosphate / backbone
        "P": AtomType.PHOSPHORUS,
        "OP1": AtomType.CHARGE_OXYGEN, "O1P": AtomType.CHARGE_OXYGEN,
        "OP2": AtomType.CHARGE_OXYGEN, "O2P": AtomType.CHARGE_OXYGEN,
        "OP3": AtomType.CHARGE_OXYGEN, "O3P": AtomType.CHARGE_OXYGEN,  # seen in monophosphate defs / termini

        "O5'": AtomType.OXYGEN_BONDED, "O5*": AtomType.OXYGEN_BONDED,
        "C5'": AtomType.CARBON_BONDED_1, "C5*": AtomType.CARBON_BONDED_1,
        "C4'": AtomType.CARBON_BONDED_2, "C4*": AtomType.CARBON_BONDED_2,
        "O4'": AtomType.OXYGEN_BONDED,  "O4*": AtomType.OXYGEN_BONDED,
        "C3'": AtomType.CARBON_BONDED_2, "C3*": AtomType.CARBON_BONDED_2,
        "O3'": AtomType.OXYGEN_BONDED,  "O3*": AtomType.OXYGEN_BONDED,
        "C2'": AtomType.CARBON_BONDED_2, "C2*": AtomType.CARBON_BONDED_2,
        "O2'": AtomType.OXYGEN,         "O2*": AtomType.OXYGEN,
        "C1'": AtomType.CARBON_BONDED_2, "C1*": AtomType.CARBON_BONDED_2,

        # base (adenine)
        "N9": AtomType.NITROGEN_AROMATIC,
        "C8": AtomType.CARBON_AROMATIC,
        "N7": AtomType.NITROGEN_AROMATIC,
        "C5": AtomType.CARBON_AROMATIC,
        "C6": AtomType.CARBON_AROMATIC,
        "N6": AtomType.NITROGEN,  # exocyclic amino
        "N1": AtomType.NITROGEN_AROMATIC,
        "C2": AtomType.CARBON_AROMATIC,
        "N3": AtomType.NITROGEN_AROMATIC,
        "C4": AtomType.CARBON_AROMATIC,
    },

    "G": {
        # phosphate / backbone
        "P": AtomType.PHOSPHORUS,
        "OP1": AtomType.CHARGE_OXYGEN, "O1P": AtomType.CHARGE_OXYGEN,
        "OP2": AtomType.CHARGE_OXYGEN, "O2P": AtomType.CHARGE_OXYGEN,
        "OP3": AtomType.CHARGE_OXYGEN, "O3P": AtomType.CHARGE_OXYGEN,

        "O5'": AtomType.OXYGEN_BONDED, "O5*": AtomType.OXYGEN_BONDED,
        "C5'": AtomType.CARBON_BONDED_1, "C5*": AtomType.CARBON_BONDED_1,
        "C4'": AtomType.CARBON_BONDED_2, "C4*": AtomType.CARBON_BONDED_2,
        "O4'": AtomType.OXYGEN_BONDED,  "O4*": AtomType.OXYGEN_BONDED,
        "C3'": AtomType.CARBON_BONDED_2, "C3*": AtomType.CARBON_BONDED_2,
        "O3'": AtomType.OXYGEN_BONDED,  "O3*": AtomType.OXYGEN_BONDED,
        "C2'": AtomType.CARBON_BONDED_2, "C2*": AtomType.CARBON_BONDED_2,
        "O2'": AtomType.OXYGEN,         "O2*": AtomType.OXYGEN,
        "C1'": AtomType.CARBON_BONDED_2, "C1*": AtomType.CARBON_BONDED_2,

        # base (guanine)
        "N9": AtomType.NITROGEN_AROMATIC,
        "C8": AtomType.CARBON_AROMATIC,
        "N7": AtomType.NITROGEN_AROMATIC,
        "C5": AtomType.CARBON_AROMATIC,

        "C6": AtomType.CARBON_BONDED_5_DOUBLE_BOND,  # carbonyl carbon
        "O6": AtomType.OXYGEN_DOUBLE_BOND,            # carbonyl oxygen

        "N1": AtomType.NITROGEN_AROMATIC,
        "C2": AtomType.CARBON_AROMATIC,
        "N2": AtomType.NITROGEN,  # exocyclic amino
        "N3": AtomType.NITROGEN_AROMATIC,
        "C4": AtomType.CARBON_AROMATIC,
    },

    "C": {
        # phosphate / backbone
        "P": AtomType.PHOSPHORUS,
        "OP1": AtomType.CHARGE_OXYGEN, "O1P": AtomType.CHARGE_OXYGEN,
        "OP2": AtomType.CHARGE_OXYGEN, "O2P": AtomType.CHARGE_OXYGEN,
        "OP3": AtomType.CHARGE_OXYGEN, "O3P": AtomType.CHARGE_OXYGEN,

        "O5'": AtomType.OXYGEN_BONDED, "O5*": AtomType.OXYGEN_BONDED,
        "C5'": AtomType.CARBON_BONDED_1, "C5*": AtomType.CARBON_BONDED_1,
        "C4'": AtomType.CARBON_BONDED_2, "C4*": AtomType.CARBON_BONDED_2,
        "O4'": AtomType.OXYGEN_BONDED,  "O4*": AtomType.OXYGEN_BONDED,
        "C3'": AtomType.CARBON_BONDED_2, "C3*": AtomType.CARBON_BONDED_2,
        "O3'": AtomType.OXYGEN_BONDED,  "O3*": AtomType.OXYGEN_BONDED,
        "C2'": AtomType.CARBON_BONDED_2, "C2*": AtomType.CARBON_BONDED_2,
        "O2'": AtomType.OXYGEN,         "O2*": AtomType.OXYGEN,
        "C1'": AtomType.CARBON_BONDED_2, "C1*": AtomType.CARBON_BONDED_2,

        # base (cytosine)
        "N1": AtomType.NITROGEN_AROMATIC,
        "C2": AtomType.CARBON_BONDED_5_DOUBLE_BOND,  # carbonyl carbon
        "O2": AtomType.OXYGEN_DOUBLE_BOND,
        "N3": AtomType.NITROGEN_AROMATIC,
        "C4": AtomType.CARBON_AROMATIC,
        "N4": AtomType.NITROGEN,  # exocyclic amino
        "C5": AtomType.CARBON_AROMATIC,
        "C6": AtomType.CARBON_AROMATIC,
    },

    "U": {
        # phosphate / backbone
        "P": AtomType.PHOSPHORUS,
        "OP1": AtomType.CHARGE_OXYGEN, "O1P": AtomType.CHARGE_OXYGEN,
        "OP2": AtomType.CHARGE_OXYGEN, "O2P": AtomType.CHARGE_OXYGEN,
        "OP3": AtomType.CHARGE_OXYGEN, "O3P": AtomType.CHARGE_OXYGEN,

        "O5'": AtomType.OXYGEN_BONDED, "O5*": AtomType.OXYGEN_BONDED,
        "C5'": AtomType.CARBON_BONDED_1, "C5*": AtomType.CARBON_BONDED_1,
        "C4'": AtomType.CARBON_BONDED_2, "C4*": AtomType.CARBON_BONDED_2,
        "O4'": AtomType.OXYGEN_BONDED,  "O4*": AtomType.OXYGEN_BONDED,
        "C3'": AtomType.CARBON_BONDED_2, "C3*": AtomType.CARBON_BONDED_2,
        "O3'": AtomType.OXYGEN_BONDED,  "O3*": AtomType.OXYGEN_BONDED,
        "C2'": AtomType.CARBON_BONDED_2, "C2*": AtomType.CARBON_BONDED_2,
        "O2'": AtomType.OXYGEN,         "O2*": AtomType.OXYGEN,
        "C1'": AtomType.CARBON_BONDED_2, "C1*": AtomType.CARBON_BONDED_2,

        # base (uracil)
        "N1": AtomType.NITROGEN_AROMATIC,
        "C2": AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        "O2": AtomType.OXYGEN_DOUBLE_BOND,
        "N3": AtomType.NITROGEN_AROMATIC,
        "C4": AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        "O4": AtomType.OXYGEN_DOUBLE_BOND,
        "C5": AtomType.CARBON_AROMATIC,
        "C6": AtomType.CARBON_AROMATIC,
    },

    # ----------------
    # DNA linking: DA,DC,DG,DT (+DI optional)
    # ----------------
    "DA": {
        "P": AtomType.PHOSPHORUS,
        "OP1": AtomType.CHARGE_OXYGEN, "O1P": AtomType.CHARGE_OXYGEN,
        "OP2": AtomType.CHARGE_OXYGEN, "O2P": AtomType.CHARGE_OXYGEN,
        "OP3": AtomType.CHARGE_OXYGEN, "O3P": AtomType.CHARGE_OXYGEN,

        "O5'": AtomType.OXYGEN_BONDED, "O5*": AtomType.OXYGEN_BONDED,
        "C5'": AtomType.CARBON_BONDED_1, "C5*": AtomType.CARBON_BONDED_1,
        "C4'": AtomType.CARBON_BONDED_2, "C4*": AtomType.CARBON_BONDED_2,
        "O4'": AtomType.OXYGEN_BONDED,  "O4*": AtomType.OXYGEN_BONDED,
        "C3'": AtomType.CARBON_BONDED_2, "C3*": AtomType.CARBON_BONDED_2,
        "O3'": AtomType.OXYGEN_BONDED,  "O3*": AtomType.OXYGEN_BONDED,

        # deoxy: no O2'
        "C2'": AtomType.CARBON_BONDED_1, "C2*": AtomType.CARBON_BONDED_1,
        "C1'": AtomType.CARBON_BONDED_2, "C1*": AtomType.CARBON_BONDED_2,

        # base (adenine)
        "N9": AtomType.NITROGEN_AROMATIC,
        "C8": AtomType.CARBON_AROMATIC,
        "N7": AtomType.NITROGEN_AROMATIC,
        "C5": AtomType.CARBON_AROMATIC,
        "C6": AtomType.CARBON_AROMATIC,
        "N6": AtomType.NITROGEN,
        "N1": AtomType.NITROGEN_AROMATIC,
        "C2": AtomType.CARBON_AROMATIC,
        "N3": AtomType.NITROGEN_AROMATIC,
        "C4": AtomType.CARBON_AROMATIC,
    },

    "DC": {
        "P": AtomType.PHOSPHORUS,
        "OP1": AtomType.CHARGE_OXYGEN, "O1P": AtomType.CHARGE_OXYGEN,
        "OP2": AtomType.CHARGE_OXYGEN, "O2P": AtomType.CHARGE_OXYGEN,
        "OP3": AtomType.CHARGE_OXYGEN, "O3P": AtomType.CHARGE_OXYGEN,

        "O5'": AtomType.OXYGEN_BONDED, "O5*": AtomType.OXYGEN_BONDED,
        "C5'": AtomType.CARBON_BONDED_1, "C5*": AtomType.CARBON_BONDED_1,
        "C4'": AtomType.CARBON_BONDED_2, "C4*": AtomType.CARBON_BONDED_2,
        "O4'": AtomType.OXYGEN_BONDED,  "O4*": AtomType.OXYGEN_BONDED,
        "C3'": AtomType.CARBON_BONDED_2, "C3*": AtomType.CARBON_BONDED_2,
        "O3'": AtomType.OXYGEN_BONDED,  "O3*": AtomType.OXYGEN_BONDED,
        "C2'": AtomType.CARBON_BONDED_1, "C2*": AtomType.CARBON_BONDED_1,
        "C1'": AtomType.CARBON_BONDED_2, "C1*": AtomType.CARBON_BONDED_2,

        # base (cytosine)
        "N1": AtomType.NITROGEN_AROMATIC,
        "C2": AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        "O2": AtomType.OXYGEN_DOUBLE_BOND,
        "N3": AtomType.NITROGEN_AROMATIC,
        "C4": AtomType.CARBON_AROMATIC,
        "N4": AtomType.NITROGEN,
        "C5": AtomType.CARBON_AROMATIC,
        "C6": AtomType.CARBON_AROMATIC,
    },

    "DG": {
        "P": AtomType.PHOSPHORUS,
        "OP1": AtomType.CHARGE_OXYGEN, "O1P": AtomType.CHARGE_OXYGEN,
        "OP2": AtomType.CHARGE_OXYGEN, "O2P": AtomType.CHARGE_OXYGEN,
        "OP3": AtomType.CHARGE_OXYGEN, "O3P": AtomType.CHARGE_OXYGEN,

        "O5'": AtomType.OXYGEN_BONDED, "O5*": AtomType.OXYGEN_BONDED,
        "C5'": AtomType.CARBON_BONDED_1, "C5*": AtomType.CARBON_BONDED_1,
        "C4'": AtomType.CARBON_BONDED_2, "C4*": AtomType.CARBON_BONDED_2,
        "O4'": AtomType.OXYGEN_BONDED,  "O4*": AtomType.OXYGEN_BONDED,
        "C3'": AtomType.CARBON_BONDED_2, "C3*": AtomType.CARBON_BONDED_2,
        "O3'": AtomType.OXYGEN_BONDED,  "O3*": AtomType.OXYGEN_BONDED,
        "C2'": AtomType.CARBON_BONDED_1, "C2*": AtomType.CARBON_BONDED_1,
        "C1'": AtomType.CARBON_BONDED_2, "C1*": AtomType.CARBON_BONDED_2,

        # base (guanine)
        "N9": AtomType.NITROGEN_AROMATIC,
        "C8": AtomType.CARBON_AROMATIC,
        "N7": AtomType.NITROGEN_AROMATIC,
        "C5": AtomType.CARBON_AROMATIC,
        "C6": AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        "O6": AtomType.OXYGEN_DOUBLE_BOND,
        "N1": AtomType.NITROGEN_AROMATIC,
        "C2": AtomType.CARBON_AROMATIC,
        "N2": AtomType.NITROGEN,
        "N3": AtomType.NITROGEN_AROMATIC,
        "C4": AtomType.CARBON_AROMATIC,
    },

    "DT": {
        "P": AtomType.PHOSPHORUS,
        "OP1": AtomType.CHARGE_OXYGEN, "O1P": AtomType.CHARGE_OXYGEN,
        "OP2": AtomType.CHARGE_OXYGEN, "O2P": AtomType.CHARGE_OXYGEN,
        "OP3": AtomType.CHARGE_OXYGEN, "O3P": AtomType.CHARGE_OXYGEN,

        "O5'": AtomType.OXYGEN_BONDED, "O5*": AtomType.OXYGEN_BONDED,
        "C5'": AtomType.CARBON_BONDED_1, "C5*": AtomType.CARBON_BONDED_1,
        "C4'": AtomType.CARBON_BONDED_2, "C4*": AtomType.CARBON_BONDED_2,
        "O4'": AtomType.OXYGEN_BONDED,  "O4*": AtomType.OXYGEN_BONDED,
        "C3'": AtomType.CARBON_BONDED_2, "C3*": AtomType.CARBON_BONDED_2,
        "O3'": AtomType.OXYGEN_BONDED,  "O3*": AtomType.OXYGEN_BONDED,
        "C2'": AtomType.CARBON_BONDED_1, "C2*": AtomType.CARBON_BONDED_1,
        "C1'": AtomType.CARBON_BONDED_2, "C1*": AtomType.CARBON_BONDED_2,

        # base (thymine)
        "N1": AtomType.NITROGEN_AROMATIC,
        "C2": AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        "O2": AtomType.OXYGEN_DOUBLE_BOND,
        "N3": AtomType.NITROGEN_AROMATIC,
        "C4": AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        "O4": AtomType.OXYGEN_DOUBLE_BOND,
        "C5": AtomType.CARBON_AROMATIC,
        "C6": AtomType.CARBON_AROMATIC,
        "C7": AtomType.CARBON_CH3,  # methyl (C5M in some alt naming)
        "C5M": AtomType.CARBON_CH3,  # alt atom id seen in CCD
    },

    # optional: deoxyinosine (DNA linking)
    "DI": {
        "P": AtomType.PHOSPHORUS,
        "OP1": AtomType.CHARGE_OXYGEN, "O1P": AtomType.CHARGE_OXYGEN,
        "OP2": AtomType.CHARGE_OXYGEN, "O2P": AtomType.CHARGE_OXYGEN,
        "OP3": AtomType.CHARGE_OXYGEN, "O3P": AtomType.CHARGE_OXYGEN,

        "O5'": AtomType.OXYGEN_BONDED, "O5*": AtomType.OXYGEN_BONDED,
        "C5'": AtomType.CARBON_BONDED_1, "C5*": AtomType.CARBON_BONDED_1,
        "C4'": AtomType.CARBON_BONDED_2, "C4*": AtomType.CARBON_BONDED_2,
        "O4'": AtomType.OXYGEN_BONDED,  "O4*": AtomType.OXYGEN_BONDED,
        "C3'": AtomType.CARBON_BONDED_2, "C3*": AtomType.CARBON_BONDED_2,
        "O3'": AtomType.OXYGEN_BONDED,  "O3*": AtomType.OXYGEN_BONDED,
        "C2'": AtomType.CARBON_BONDED_1, "C2*": AtomType.CARBON_BONDED_1,
        "C1'": AtomType.CARBON_BONDED_2, "C1*": AtomType.CARBON_BONDED_2,

        # base (inosine ~= hypoxanthine)
        "N9": AtomType.NITROGEN_AROMATIC,
        "C8": AtomType.CARBON_AROMATIC,
        "N7": AtomType.NITROGEN_AROMATIC,
        "C5": AtomType.CARBON_AROMATIC,
        "C6": AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        "O6": AtomType.OXYGEN_DOUBLE_BOND,
        "N1": AtomType.NITROGEN_AROMATIC,
        "C2": AtomType.CARBON_AROMATIC,
        "N3": AtomType.NITROGEN_AROMATIC,
        "C4": AtomType.CARBON_AROMATIC,
    },
}

# If you want a single lookup dict:
# atom_data = {**protein_atom_data, **nucleic_atom_data}

# And in AtomType.from_id you can use atom_data instead of protein_atom_data.


protein_atom_data = { 
    
    'HOH':{'O': AtomType.WATER},
                     'GLY': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
                             'O': AtomType.PEPTIDE_O,
                             'OXT':AtomType.PEPTIDE_O,
                             'N':AtomType.PEPTIDE_N,
                             'CA':AtomType.CARBON_BONDED_2},
                     
'ALA': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O': AtomType.PEPTIDE_O,
        'OXT': AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_CH3},

'VAL': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_2,
        'CG1':AtomType.CARBON_CH3,
        'CG2':AtomType.CARBON_CH3},

'ILE': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_2,
        'CG2':AtomType.CARBON_CH3,
        'CG1':AtomType.CARBON_BONDED_1,
        'CD1':AtomType.CARBON_CH3},

'LEU': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_1,
        'CG':AtomType.CARBON_BONDED_2,
        'CD1':AtomType.CARBON_CH3,
        'CD2':AtomType.CARBON_CH3},

'MET': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_1,
        'CG':AtomType.CARBON_BONDED_1,
        'SD':AtomType.SULPHUR_BONDED,
        'CE':AtomType.CARBON_CH3},

'PHE': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_1,
        'CG':AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'CD1':AtomType.CARBON_AROMATIC,
        'CD2':AtomType.CARBON_AROMATIC,
        'CE1':AtomType.CARBON_AROMATIC,
        'CE2':AtomType.CARBON_AROMATIC,
        'CZ':AtomType.CARBON_AROMATIC},

'TYR': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_1,
        'CG':AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'CD1':AtomType.CARBON_AROMATIC,
        'CD2':AtomType.CARBON_AROMATIC,
        'CE1':AtomType.CARBON_AROMATIC,
        'CE2':AtomType.CARBON_AROMATIC,
        'CZ':AtomType.CARBON_AROMATIC,
        'OH':AtomType.OXYGEN},

'TRP': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_1,
        'CG':AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'CD1':AtomType.CARBON_AROMATIC,
        'CD2':AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'NE1': AtomType.NITROGEN_TRPNE1,
        'CE2':AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'CE3':AtomType.CARBON_AROMATIC,
        'CZ2':AtomType.CARBON_AROMATIC,
        'CZ3':AtomType.CARBON_AROMATIC,
        'CH2':AtomType.CARBON_AROMATIC},

'SER': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O': AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_1,
        'OG':AtomType.OXYGEN},

'THR': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_2,
        'CG2':AtomType.CARBON_CH3,
        'OG1':AtomType.OXYGEN},

'ASN': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND, 
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_1,
        'CG':AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'OD1':AtomType.OXYGEN_DOUBLE_BOND,
        'ND2':AtomType.NITROGEN_ASNGLN},

'GLN': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_1,
        'CG':AtomType.CARBON_BONDED_1,
        'CD':AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'OE1':AtomType.OXYGEN_DOUBLE_BOND,
        'NE2':AtomType.NITROGEN_ASNGLN},

'ASP': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND, 
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_1,
        'CG':AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'OD1':AtomType.OXYGEN_DOUBLE_BOND,
        'OD2':AtomType.OXYGEN_DOUBLE_BOND},

'GLU': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND, 
        'O': AtomType.PEPTIDE_O,
        'OXT': AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_1,
        'CG':AtomType.CARBON_BONDED_1,
        'CD':AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'OE1':AtomType.OXYGEN_DOUBLE_BOND,
        'OE2':AtomType.OXYGEN_DOUBLE_BOND},

'ARG': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_1,
        'CG':AtomType.CARBON_BONDED_1,
        'CD':AtomType.CARBON_BONDED_1,
        'NE':AtomType.NITROGEN_ASNGLN,
        'CZ':AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'NH1':AtomType.NITROGEN_ASNGLN,
        'NH2':AtomType.NITROGEN_ASNGLN},

'LYS': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_1,
        'CG':AtomType.CARBON_BONDED_1,
        'CD':AtomType.CARBON_BONDED_1,
        'CE':AtomType.CARBON_BONDED_1,
        'NZ':AtomType.NITROGEN_ASNGLN},

'HIS': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_1,
        'CG':AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'ND1':AtomType.NITROGEN_ASNGLN,
        'CD2':AtomType.CARBON_AROMATIC,
        'CE1':AtomType.CARBON_AROMATIC,
        'NE2':AtomType.NITROGEN_AROMATIC},

'CYS': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N':AtomType.PEPTIDE_N,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_1,
        'SG': AtomType.SULPHUR},

'PRO': {'C': AtomType.CARBON_BONDED_5_DOUBLE_BOND,
        'O':AtomType.PEPTIDE_O,
        'OXT':AtomType.PEPTIDE_O,
        'N': AtomType.NITROGEN_BONDED_3,
        'CA':AtomType.CARBON_BONDED_2,
        'CB':AtomType.CARBON_BONDED_1,
        'CG':AtomType.CARBON_BONDED_1,
        'CD':AtomType.CARBON_BONDED_1}
}

atom_data = {**protein_atom_data, **nucleic_atom_data}

class AtomType_ori(Enum):
    '''
    Atom type identifiers for ChemDock2
    '''
    def __init__(self, idx, symbol, bonds, hydrogens):
        self.idx = idx
        self.symbol = symbol
        self.bonds = bonds
        self.hydrogens = hydrogens

    BROMINE = (0,'Br',[BondType.SINGLE], 0)
    CHLORINE = (1, 'Cl', [BondType.SINGLE], 0)
    FLUORINE = (2,'F', [BondType.SINGLE], 0)
    IODINE = (3, 'I', [BondType.SINGLE], 0)
    CARBON_CH3 = (4, 'C', [BondType.SINGLE] * 4, 3)
    #CARBON_AROMATIC = (5, 'C', [BondType.AROMATIC, BondType.AROMATIC, BondType.SINGLE], 1) #need one withour hydrogens
    CARBON_AROMATIC = (8, 'C', [BondType.AROMATIC, BondType.AROMATIC, BondType.SINGLE], 1) #need one withour hydrogens
    CARBON_CH2 = (6, 'C', [BondType.DOUBLE, BondType.SINGLE, BondType.SINGLE], 2)
    CARBON_BONDED_1 = (7,'C', [BondType.SINGLE] * 4, 2)
    CARBON_BONDED_1_DOUBLE_BOND = (8, 'C', [BondType.DOUBLE, BondType.SINGLE, BondType.SINGLE], 1)
    CARBON_BONDED_2 = (9,'C', [BondType.SINGLE] * 4, 1)
    CARBON_BONDED_3 = (10, 'C', [BondType.SINGLE] * 4, 0)
    CARBON_TRIPLE_BOND = (11, 'C', [BondType.TRIPLE, BondType.SINGLE], 1)
    CARBON_TRIPLE_BOND_BONDED = (12, 'C', [BondType.TRIPLE, BondType.SINGLE], 0)#new same as 10
    NITROGEN = (13, 'N', [BondType.SINGLE] * 3, 2)
    NITROGEN_AROMATIC = (14, 'N', [BondType.AROMATIC, BondType.AROMATIC], 0)
    NITROGEN_BONDED_1 = (15, 'N', [BondType.SINGLE] * 3, 1)
    NITROGEN_DOUBLE_BOND = (16, 'N', [BondType.DOUBLE, BondType.SINGLE], 1) #new same as 14
    NITROGEN_TRIPLE_BOND = (17, 'N', [BondType.TRIPLE], 0)
    NITROGEN_BONDED_3 = (18, 'N', [BondType.SINGLE] * 3, 0)
    OXYGEN = (19, 'O', [BondType.SINGLE] * 2, 1)
    OXYGEN_AROMATIC = (20, 'O', [BondType.AROMATIC] * 2, 0)
    OXYGEN_DOUBLE_BOND = (21, 'O', [BondType.DOUBLE], 0)
    OXYGEN_BONDED = (22, 'O', [BondType.SINGLE] * 2, 0)
    PHOSPHORUS = (23, 'P', [None], None) # specil means just assign it to atom
    SULPHUR = (24, 'S', [BondType.SINGLE] * 2, 1)
    SULPHUR_AROMATIC = (25, 'S', [BondType.AROMATIC] * 2, 0)
    SULPHUR_DOUBLE_BOND = (26, 'S', [BondType.DOUBLE], 0)
    SULPHUR_BONDED = (27, 'S', [BondType.SINGLE] * 2, 0)
    WATER = (28, 'O', [BondType.SINGLE] * 2, 2)
    NITROGEN_DOUBLE_BOND_BONDED = (29, 'N', [BondType.SINGLE, BondType.DOUBLE], 0)
    CARBON_BONDED_4_DOUBLE_BOND = (30, 'C', [BondType.DOUBLE, BondType.DOUBLE], 0) #should be same as 8
    CARBON_BONDED_5_DOUBLE_BOND = (31, 'C', [BondType.DOUBLE, BondType.SINGLE, BondType.SINGLE], 0) #same as 8
    CHARGED_NITROGEN_1 = (32, 'N', [BondType.SINGLE] * 4, 1) #same as 13 for now.
    CHARGED_NITROGEN_2 = (33, 'N', [BondType.SINGLE] * 4, 2) #same as 13 for now.
    CHARGED_NITROGEN_3 = (34, 'N', [BondType.SINGLE] * 4, 3) #same as 13 for now.
    CHARGED_NITROGEN_4 = (35, 'N', [BondType.SINGLE, BondType.SINGLE, BondType.DOUBLE], 3) #same as 13 for now.
    CHARGE_OXYGEN = (36, 'O', [BondType.SINGLE], 0) #same as 19

    #new types that go over other types!!
    AMIDE_N = (37, 'N', [BondType.SINGLE] * 3, 1)
    AMIDE_O = (38, 'O',[BondType.DOUBLE], 0)
    PEPTIDE_O = (39, 'O',[BondType.DOUBLE], 0)
    PEPTIDE_N = (40, 'N', [BondType.SINGLE] * 3, 1)
    PO4_OXYGEN_DOUBLE_BOND = (41, 'O',[BondType.DOUBLE], 0)
    PO4_OXYGEN_DONOR = (42, 'O', [BondType.SINGLE] * 2, 1)

    @staticmethod
    def from_atom(atom):
        element = atom.GetSymbol()
        bonds = sorted([i.GetBondType() for i in atom.GetBonds()])
        num_hydrogens = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H')
        atom_is_aromatic = atom.GetIsAromatic()

        for atom_type in AtomType:
            if (atom_type.symbol == element and
                len(atom_type.bonds) == len(bonds) and
                all(b1 == b2 for b1, b2 in zip(sorted(atom_type.bonds), bonds)) and
                atom_type.hydrogens == num_hydrogens):
                return atom_type

            elif atom_type.symbol == element and atom_is_aromatic:
                if BondType.AROMATIC in atom_type.bonds:
                    return atom_type
            elif atom_type.symbol == 'P':
                return AtomType.PHOSPHORUS

       
        
        raise RuntimeError(f"[Error] No matching AtomType for symbol={element}, bonds={bonds}, hydrogens={num_hydrogens}")
        


# ---- RD_PROTEIN_SMILES  ----
RD_PROTEIN_SMILES = {'CYS': ('N[C@H](C=O)CS', {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'SG': 5}),
 'MET': ('CSCC[C@H](N)C=O',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'SD': 6, 'CE': 7}),
 'GLY': ('NCC=O', {'N': 0, 'CA': 1, 'C': 2, 'O': 3}),
 'ASP': ('N[C@H](C=O)CC(=O)O',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'OD1': 6, 'OD2': 7}),
 'ALA': ('C[C@H](N)C=O', {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4}),
 'VAL': ('CC(C)[C@H](N)C=O',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG1': 5, 'CG2': 6}),
 'PRO': ('O=C[C@@H]1CCCN1',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6}),
 'PHE': ('N[C@H](C=O)Cc1ccccc1',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'CD1': 6,
   'CD2': 7,
   'CE1': 8,
   'CE2': 9,
   'CZ': 10}),
 'ASN': ('NC(=O)C[C@H](N)C=O',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'OD1': 6, 'ND2': 7}),
 'THR': ('C[C@@H](O)[C@H](N)C=O',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'OG1': 5, 'CG2': 6}),
 'HIS': ('N[C@H](C=O)Cc1c[nH]c[nH+]1',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'ND1': 6,
   'CD2': 7,
   'CE1': 8,
   'NE2': 9}),
 'GLN': ('NC(=O)CC[C@H](N)C=O',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'CD': 6,
   'OE1': 7,
   'NE2': 8}),
 'ARG': ('NC(=[NH2+])NCCC[C@H](N)C=O',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'CD': 6,
   'NE': 7,
   'CZ': 8,
   'NH1': 9,
   'NH2': 10}),
 'TRP': ('N[C@H](C=O)Cc1c[nH]c2ccccc12',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'CD1': 6,
   'CD2': 7,
   'NE1': 8,
   'CE2': 9,
   'CE3': 10,
   'CZ2': 11,
   'CZ3': 12,
   'CH2': 13}),
 'ILE': ('CC[C@H](C)[C@H](N)C=O',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG1': 5, 'CG2': 6, 'CD1': 7}),
 'SER': ('N[C@@H](C=O)CO',
  {'N': 0, 'CA': 1, 'CB': 2, 'OG': 3, 'C': 4, 'O': 5}),
 'LYS': ('N[C@H](C=O)CCCC[NH3+]',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'CD': 6,
   'CE': 7,
   'NZ': 8}),
 'LEU': ('CC(C)C[C@H](N)C=O',
  {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD1': 6, 'CD2': 7}),
 'GLU': ('N[C@H](C=O)CCC(=O)O',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'CD': 6,
   'OE1': 7,
   'OE2': 8}),
 'TYR': ('N[C@H](C=O)Cc1ccc(O)cc1',
  {'N': 0,
   'CA': 1,
   'C': 2,
   'O': 3,
   'CB': 4,
   'CG': 5,
   'CD1': 6,
   'CD2': 7,
   'CE1': 8,
   'CE2': 9,
   'CZ': 10,
   'OH': 11})}


def INTRA_RESIDUE_BOND_DATA(a_name, b_name, resname):
    
    
    bond_type = Chem.BondType.SINGLE
    
    if {a_name, b_name} == {"C", "O"}:
        bond_type = Chem.BondType.DOUBLE
    # Terminal carboxylate OXT (will be single; main O already handled above if present)
    if {a_name, b_name} == {"C", "OXT"}:
        bond_type = Chem.BondType.SINGLE
    # Side-chain specific templates:
    if resname == "ASP":
        # Aspartate: OD1 and OD2, one double, one single
        if {a_name, b_name} == {"CG", "OD1"}:
            bond_type = Chem.BondType.DOUBLE
        if {a_name, b_name} == {"CG", "OD2"}:
            bond_type = Chem.BondType.SINGLE
    elif resname == "GLU":
        # Glutamate: OE1 and OE2
        if {a_name, b_name} == {"CD", "OE1"}:
            bond_type = Chem.BondType.DOUBLE
        if {a_name, b_name} == {"CD", "OE2"}:
            bond_type = Chem.BondType.SINGLE
    elif resname == "ASN":
        # Asparagine: OD1 (double), ND2 (single)
        if {a_name, b_name} == {"CG", "OD1"}:
            bond_type = Chem.BondType.DOUBLE
        if {a_name, b_name} == {"CG", "ND2"}:
            bond_type = Chem.BondType.SINGLE
    elif resname == "GLN":
        # Glutamine: OE1 (double), NE2 (single)
        if {a_name, b_name} == {"CD", "OE1"}:
            bond_type = Chem.BondType.DOUBLE
        if {a_name, b_name} == {"CD", "NE2"}:
            bond_type = Chem.BondType.SINGLE
    elif resname == "ARG":
        # Arginine: NE–CZ (single), one of CZ–NH1 / CZ–NH2 is double, the other single
        if {a_name, b_name} == {"NE", "CZ"}:
            bond_type = Chem.BondType.SINGLE
        # Assign CZ=NH1 as double, CZ–NH2 as single (arbitrary choice for resonance)
        if "CZ" in {a_name, b_name} and "NH1" in {a_name, b_name}:
            bond_type = Chem.BondType.DOUBLE
        if "CZ" in {a_name, b_name} and "NH2" in {a_name, b_name}:
            bond_type = Chem.BondType.SINGLE
    elif resname == "HIS":
        # Histidine imidazole ring (5-membered aromatic ring)
        his_ring = {"CG", "ND1", "CD2", "CE1", "NE2"}
        if a_name in his_ring and b_name in his_ring:
            bond_type = Chem.BondType.AROMATIC
    elif resname == "PHE":
        # Phenyl ring in phenylalanine (6-membered aromatic ring)
        phe_ring = {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}
        if a_name in phe_ring and b_name in phe_ring:
            bond_type = Chem.BondType.AROMATIC
    elif resname == "TYR":
        # Phenol ring in tyrosine (same ring atoms as phenylalanine)
        tyr_ring = {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}
        if a_name in tyr_ring and b_name in tyr_ring:
            bond_type = Chem.BondType.AROMATIC
    elif resname == "TRP":
        # Indole ring system in tryptophan (fused 5- and 6-member rings)
        trp_ring = {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"}
        if a_name in trp_ring and b_name in trp_ring:
            bond_type = Chem.BondType.AROMATIC
    
    return bond_type

def INTER_RESIDUE_BOND_DATA(a_name, b_name):
    
    #default bond 
    bond_type = Chem.BondType.SINGLE
    
    if (a_name == "C" and b_name == "N") or (a_name == "N" and b_name == "C"):
        bond_type = Chem.BondType.SINGLE  # peptide C–N is a single bond
        # (The C=O of the carbonyl was handled as double in the intra-residue section)
    # Disulfide bond between cysteine residues (SG–SG)
    if a_name == "SG" and b_name == "SG":
        bond_type = Chem.BondType.SINGLE
    
    return bond_type

