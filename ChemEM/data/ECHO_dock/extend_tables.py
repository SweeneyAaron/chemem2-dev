#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 00:17:38 2026

@author: aaron.sweeney
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple, Any, Optional
import json
import numpy as np


# -----------------------------
# Buckingham parameter container
# -----------------------------
@dataclass(frozen=True)
class BuckinghamABC:
    A: float
    B: float
    C: float


# -----------------------------
# Helpers
# -----------------------------
def _to_array2d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2D table, got shape={arr.shape}")
    return arr


def _normalize_atom_type_defs(
    atom_type_defs: Mapping[str, tuple] | Iterable[Tuple[str, tuple]]
) -> Dict[str, tuple]:
    """
    Accepts either:
      - dict: {"ATOM_NAME": (idx, symbol, bonds, hcount), ...}
      - iterable of (name, tuple)
    Returns a normal dict.
    """
    if isinstance(atom_type_defs, Mapping):
        return dict(atom_type_defs)
    return dict(atom_type_defs)


def infer_coordinating_type_ids(
    atom_type_defs: Mapping[str, tuple] | Iterable[Tuple[str, tuple]],
    coordinating_elements: Tuple[str, ...] = ("O", "N", "S"),
    include_names: Optional[set[str]] = None,
    exclude_names: Optional[set[str]] = None,
    include_ids: Optional[set[int]] = None,
    exclude_ids: Optional[set[int]] = None,
) -> set[int]:
    """
    Infer coordinating atom-type indices from your atom type definitions.

    Default behavior:
      - any atom type whose element symbol is O/N/S is considered coordinating

    You can refine with include/exclude names or IDs.
    """
    atom_type_defs = _normalize_atom_type_defs(atom_type_defs)
    include_names = include_names or set()
    exclude_names = exclude_names or set()
    include_ids = include_ids or set()
    exclude_ids = exclude_ids or set()

    coord_ids: set[int] = set()

    for name, spec in atom_type_defs.items():
        idx, symbol = int(spec[0]), str(spec[1]).upper()

        if name in exclude_names or idx in exclude_ids:
            continue

        if name in include_names or idx in include_ids:
            coord_ids.add(idx)
            continue

        if symbol in {e.upper() for e in coordinating_elements}:
            coord_ids.add(idx)

    return coord_ids


# -----------------------------
# Main extension function
# -----------------------------
def extend_buckingham_tables_with_generic_metal(
    table_a: Any,
    table_b: Any,
    table_c: Any,
    *,
    metal_type_index: int = 44,
    atom_type_defs: Mapping[str, tuple] | Iterable[Tuple[str, tuple]],
    bonded_abc: BuckinghamABC,
    nonbonded_abc: BuckinghamABC,
    coordinating_type_ids: Optional[set[int]] = None,
    coordinating_elements: Tuple[str, ...] = ("O", "N", "S"),
    exclude_coord_names: Optional[set[str]] = None,
    exclude_coord_ids: Optional[set[int]] = None,
    metal_self_uses_nonbonded: bool = True,
    verify_symmetric: bool = True,
):
    """
    Extend TABLE_A/B/C by adding a generic metal atom type row/column.

    - Existing tables are NxN
    - metal_type_index is expected to be N (append one new type), but this function
      also supports overwriting an existing index if desired.

    Fills metal interactions:
      metal <-> atom_type_i  => bonded or nonbonded ABC depending on coordination classification

    Returns:
      new_table_a, new_table_b, new_table_c, info_dict
    """
    A = _to_array2d(table_a, "TABLE_A")
    B = _to_array2d(table_b, "TABLE_B")
    C = _to_array2d(table_c, "TABLE_C")

    if not (A.shape == B.shape == C.shape):
        raise ValueError(f"TABLE_A/B/C must have same shape, got {A.shape}, {B.shape}, {C.shape}")

    n = A.shape[0]

    if verify_symmetric:
        for name, T in (("A", A), ("B", B), ("C", C)):
            if not np.allclose(T, T.T, atol=1e-10, rtol=1e-10):
                raise ValueError(f"TABLE_{name} is not symmetric")

    # Decide final size
    if metal_type_index < 0:
        raise ValueError("metal_type_index must be >= 0")

    new_n = max(n, metal_type_index + 1)

    # Create extended tables (preserve old values)
    A_ext = np.zeros((new_n, new_n), dtype=float)
    B_ext = np.zeros((new_n, new_n), dtype=float)
    C_ext = np.zeros((new_n, new_n), dtype=float)

    A_ext[:n, :n] = A
    B_ext[:n, :n] = B
    C_ext[:n, :n] = C

    # Build coordination set if not supplied
    if coordinating_type_ids is None:
        coordinating_type_ids = infer_coordinating_type_ids(
            atom_type_defs=atom_type_defs,
            coordinating_elements=coordinating_elements,
            exclude_names=exclude_coord_names,
            exclude_ids=exclude_coord_ids,
        )

    # Fill metal row/column against all existing atom-type indices [0..n-1]
    for i in range(n):
        use_bonded = i in coordinating_type_ids
        abc = bonded_abc if use_bonded else nonbonded_abc

        A_ext[metal_type_index, i] = abc.A
        A_ext[i, metal_type_index] = abc.A

        B_ext[metal_type_index, i] = abc.B
        B_ext[i, metal_type_index] = abc.B

        C_ext[metal_type_index, i] = abc.C
        C_ext[i, metal_type_index] = abc.C

    # metal-metal self interaction
    if metal_self_uses_nonbonded:
        abc_mm = nonbonded_abc
    else:
        abc_mm = bonded_abc  # usually not what you want, but supported

    A_ext[metal_type_index, metal_type_index] = abc_mm.A
    B_ext[metal_type_index, metal_type_index] = abc_mm.B
    C_ext[metal_type_index, metal_type_index] = abc_mm.C

    info = {
        "old_size": n,
        "new_size": new_n,
        "metal_type_index": metal_type_index,
        "coordinating_type_ids": sorted(int(x) for x in coordinating_type_ids),
        "n_coordinating_ids": len(coordinating_type_ids),
    }

    return A_ext, B_ext, C_ext, info


# -----------------------------
# JSON I/O convenience wrappers
# -----------------------------
def load_json_table(path: str) -> np.ndarray:
    with open(path, "r") as f:
        return np.asarray(json.load(f), dtype=float)


def save_json_table(path: str, table: np.ndarray) -> None:
    with open(path, "w") as f:
        json.dump(table.tolist(), f)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Your fitted values
    bonded = BuckinghamABC(A=92583.3, B=4.2216, C=1649.13)
    nonbonded = BuckinghamABC(A=548.1033308589573, B=1.0346458859947836, C=26.06592074458319)

    # Minimal atom type defs example (replace with your real source / enum extraction)
    # Format: name -> (index, element, bonds, Hcount)
    atom_type_defs = {
        "BROMINE": (0, "Br", None, 0),
        "CHLORINE": (1, "Cl", None, 0),
        "FLUORINE": (2, "F", None, 0),
        "IODINE": (3, "I", None, 0),
        "CARBON_CH3": (4, "C", None, 3),
        "CARBON_AROMATIC": (5, "C", None, 1),
        "CARBON_CH2": (6, "C", None, 2),
        "CARBON_BONDED_1": (7, "C", None, 2),
        "CARBON_BONDED_1_DOUBLE_BOND": (8, "C", None, 1),
        "CARBON_BONDED_2": (9, "C", None, 1),
        "CARBON_BONDED_3": (10, "C", None, 0),
        "CARBON_TRIPLE_BOND": (11, "C", None, 1),
        "CARBON_TRIPLE_BOND_BONDED": (12, "C", None, 0),
        "NITROGEN": (13, "N", None, 2),
        "NITROGEN_AROMATIC": (14, "N", None, 0),
        "NITROGEN_BONDED_1": (15, "N", None, 1),
        "NITROGEN_TRIPLE_BOND": (17, "N", None, 0),
        "NITROGEN_BONDED_3": (18, "N", None, 0),
        "OXYGEN": (19, "O", None, 1),
        "OXYGEN_AROMATIC": (20, "O", None, 0),
        "OXYGEN_DOUBLE_BOND": (21, "O", None, 0),
        "OXYGEN_BONDED": (22, "O", None, 0),
        "PHOSPHORUS": (23, "P", None, None),
        "SULPHUR": (24, "S", None, 1),
        "SULPHUR_AROMATIC": (25, "S", None, 0),
        "SULPHUR_DOUBLE_BOND": (26, "S", None, 0),
        "SULPHUR_BONDED": (27, "S", None, 0),
        "WATER": (28, "O", None, 2),
        "AMIDE_N": (37, "N", None, 1),
        "AMIDE_O": (38, "O", None, 0),
        "PEPTIDE_O": (39, "O", None, 0),
        "PEPTIDE_N": (40, "N", None, 1),
        "PO4_OXYGEN_DOUBLE_BOND": (41, "O", None, 0),
        "PO4_OXYGEN_DONOR": (42, "O", None, 1),
        "CHARGED_NITROGEN_AROMATIC": (43, "N", None, 1),
        # metals share generic index 44 (new row/col to be added)
        "MG": (44, "Mg", [], 0),
        "CA": (44, "Ca", [], 0),
        "MN": (44, "Mn", [], 0),
        "FE": (44, "Fe", [], 0),
        "CU": (44, "Cu", [], 0),
        "ZN": (44, "Zn", [], 0),
    }

    # Optional refinement: exclude atom types you do NOT want treated as coordinating
    # (depends on your chemistry assumptions)
    exclude_coord_names = {
        # "PEPTIDE_N",   # often poor metal donor
        # "AMIDE_N",     # often poor metal donor
        # "NITROGEN_BONDED_3",  # tertiary amine can still coordinate in many cases though
    }

    # Load your existing tables
    A = load_json_table("TableA.json")
    B = load_json_table("TableB.json")
    C = load_json_table("TableC.json")

    # Extend with generic metal type index 44
    A2, B2, C2, info = extend_buckingham_tables_with_generic_metal(
        A, B, C,
        metal_type_index=44,
        atom_type_defs=atom_type_defs,
        bonded_abc=bonded,
        nonbonded_abc=nonbonded,
        exclude_coord_names=exclude_coord_names,
        metal_self_uses_nonbonded=True,
    )

    print("Extension info:", info)
    print("New shapes:", A2.shape, B2.shape, C2.shape)

    # Save
    save_json_table("TableA_extended.json", A2)
    save_json_table("TableB_extended.json", B2)
    save_json_table("TableC_extended.json", C2)