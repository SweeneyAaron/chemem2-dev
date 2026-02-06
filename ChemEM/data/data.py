# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>
"""
Public data facade.

This file re-exports names from ChemEM..data._data.* 
as a way to make scaling and maintining data more manageable
    from ChemEM.data import SYSTEM_ATTRS, AtomType, TABLE_A, ...
"""

from ._data.kernels import kernel_dx, kernel_dy, kernel_dz
from ._data.system_constants import (
    SYSTEM_ATTRS,
    RESIDUE_NAMES,
    PROTEIN_RINGS,
    PROTEIN_DONOR_ATOM_IDS,
    PROTEIN_ACCEPTOR_ATOM_IDS,
    is_protein_atom_donor,
    is_protein_atom_acceptor,
)
from ._data.physchem import XlogP3, IonData
from ._data.tables import (
    load_json,
    TABLE_A,
    TABLE_B,
    TABLE_C,
    HBOND_POLYA,
    HBOND_POLYB,
    HBOND_POLYC,
    HBOND_DONOR_ATOM_IDXS,
    HBOND_ACCEPTOR_ATOM_IDXS,
    HALOGEN_DONOR_ATOM_IDXS,
    HALOGEN_ACCEPTOR_ATOM_IDXS,
)
from ._data.atom_typing import (
    RingType,
    AtomType,
    AtomType_ori,
    protein_atom_data,
    RD_PROTEIN_SMILES,
    INTRA_RESIDUE_BOND_DATA,
    INTER_RESIDUE_BOND_DATA,
)

__all__ = [
    # kernels
    "kernel_dx", "kernel_dy", "kernel_dz",

    # system constants
    "SYSTEM_ATTRS", "RESIDUE_NAMES", "PROTEIN_RINGS",
    "PROTEIN_DONOR_ATOM_IDS", "PROTEIN_ACCEPTOR_ATOM_IDS",
    "is_protein_atom_donor", "is_protein_atom_acceptor",

    # physchem
    "XlogP3", "IonData",

    # tables
    "load_json",
    "TABLE_A", "TABLE_B", "TABLE_C",
    "HBOND_POLYA", "HBOND_POLYB", "HBOND_POLYC",
    "HBOND_DONOR_ATOM_IDXS", "HBOND_ACCEPTOR_ATOM_IDXS",
    "HALOGEN_DONOR_ATOM_IDXS", "HALOGEN_ACCEPTOR_ATOM_IDXS",

    # typing / mapping
    "RingType", "AtomType", "AtomType_ori",
    "protein_atom_data", "RD_PROTEIN_SMILES","INTRA_RESIDUE_BOND_DATA",
    "INTER_RESIDUE_BOND_DATA",
    
]
