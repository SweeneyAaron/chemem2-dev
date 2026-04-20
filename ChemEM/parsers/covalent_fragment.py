# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>
"""
Covalent ligand support.

Pipeline (refine path):
  1. apply_ligand_deletions(rd_mol, spec)            -- RDKit level, BEFORE OpenFF
  2. apply_protein_deletions(protein_structure, spec)-- parmed level, BEFORE merge
  3. build_and_parameterize_fragment(ligand, spec)   -- OpenFF on capped junction
  4. inject_covalent_bonds(complex_structure, ligs)  -- parmed level, BEFORE createSystem

The fragment is a small capped molecule consisting of the truncated ligand
joined to a methyl-capped sidechain. SMIRNOFF parameterizes it as a single
molecule; junction bond/angle/dihedral terms are lifted out of the fragment
structure and injected into the real protein+ligand merged structure. Partial
charges for ligand atoms near the junction are harvested from the same
fragment and copied onto the ligand, then the ligand charge total is
renormalized to the nearest integer to preserve net charge.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .models import CovalentLinkSpec


# Standard valence used to decide whether forming the new covalent bond
# requires removal of an H from the attachment atom.
_STD_VALENCE = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "P": 3,
    "S": 2,
    "Cl": 1,
    "Br": 1,
    "I": 1,
}

# Minimal residue sidechain templates used to build capped junction fragments.
# Each entry describes how to construct a neutral methyl-capped sidechain for
# the listed residue, keyed by AMBER residue name. The attachment atom name
# is the one the user points at via covalent_protein_atom.
#
# Structure per entry: (smiles_pattern, attachment_atom_smarts, h_to_remove_on_attachment)
#   - smiles_pattern:   SMILES for the neutral capped sidechain INCLUDING any H
#                       that will be auto-removed; used to build a clean RDKit mol.
#   - attachment_atom_smarts: atom that will bond to the ligand in the fragment.
#
# We keep this table small; unknown residues fall back to a generic methyl cap
# attached at the user-specified element.
_SIDECHAIN_TEMPLATES: Dict[str, Dict[str, str]] = {
    # Cysteine — Cα capped with CH3, side chain CH2-SH
    "CYS": {"smiles": "CCS", "attach_atom": "S"},
    "CYX": {"smiles": "CCS", "attach_atom": "S"},
    # Serine
    "SER": {"smiles": "CCO", "attach_atom": "O"},
    # Threonine
    "THR": {"smiles": "CC(C)O", "attach_atom": "O"},
    # Tyrosine — phenol, cap benzene by leaving as-is
    "TYR": {"smiles": "Cc1ccc(O)cc1", "attach_atom": "O"},
    # Lysine — cap ε-amine
    "LYS": {"smiles": "CCCCN", "attach_atom": "N"},
    # Histidine — neutral imidazole (HIE tautomer), attach at NE2
    "HIS": {"smiles": "Cc1c[nH]cn1", "attach_atom": "N"},
    "HIE": {"smiles": "Cc1c[nH]cn1", "attach_atom": "N"},
    "HID": {"smiles": "Cc1cnc[nH]1", "attach_atom": "N"},
    # Aspartate / glutamate (carboxylate, neutral for fragment parameterization)
    "ASP": {"smiles": "CCC(=O)O", "attach_atom": "O"},
    "GLU": {"smiles": "CCCC(=O)O", "attach_atom": "O"},
}


# ---------------------------------------------------------------------------
# Spec parsing helpers
# ---------------------------------------------------------------------------

def parse_protein_atom_spec(spec: str) -> Tuple[str, str, int, str]:
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid protein atom spec '{spec}'. Expected 'CHAIN:RESNAME:RESNUM:ATOMNAME'."
        )
    chain, resname, resnum_str, atom_name = parts
    try:
        resnum = int(resnum_str)
    except ValueError as exc:
        raise ValueError(f"Invalid resnum in '{spec}': {exc}")
    return chain, resname, resnum, atom_name


def parse_ligand_atom_spec(spec: str) -> Tuple[int, str]:
    parts = spec.split(":")
    if len(parts) != 3 or parts[0] != "LIG":
        raise ValueError(
            f"Invalid ligand atom spec '{spec}'. Expected 'LIG:index:ATOMNAME'."
        )
    try:
        lig_idx = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid ligand index in '{spec}': {exc}")
    return lig_idx, parts[2]


# ---------------------------------------------------------------------------
# Auto-detection of H atoms to remove
# ---------------------------------------------------------------------------

def _ligand_atom_name_from_index(rd_mol: Chem.Mol, atom_idx: int) -> str:
    """Mirror Ligand.get_atom_names(): element-upper + 1-based count-within-element."""
    count = 0
    target_symbol = rd_mol.GetAtomWithIdx(atom_idx).GetSymbol()
    for i in range(atom_idx + 1):
        if rd_mol.GetAtomWithIdx(i).GetSymbol() == target_symbol:
            count += 1
    return f"{target_symbol.upper()}{count}"


def _rdkit_find_atom_by_name(rd_mol: Chem.Mol, atom_name: str) -> int:
    """Reproduce Ligand.get_atom_names mapping without building a Ligand."""
    from collections import Counter
    atom_counts: Counter = Counter()
    for i, atom in enumerate(rd_mol.GetAtoms()):
        atom_counts[atom.GetSymbol()] += 1
        if f"{atom.GetSymbol().upper()}{atom_counts[atom.GetSymbol()]}" == atom_name:
            return i
    raise ValueError(f"Ligand atom '{atom_name}' not found in RDKit mol.")


def _auto_h_to_remove_rdkit(rd_mol: Chem.Mol, attach_idx: int, new_bond_order: int) -> Optional[int]:
    """Return the RDKit atom index of an H on attach_idx that should be removed
    to accommodate the new covalent bond, or None if no removal is needed."""
    atom = rd_mol.GetAtomWithIdx(attach_idx)
    symbol = atom.GetSymbol()
    std_val = _STD_VALENCE.get(symbol)
    if std_val is None:
        return None

    # RDKit total valence already accounts for implicit/explicit H.
    current_val = atom.GetTotalValence()
    if current_val + new_bond_order <= std_val:
        return None

    # Need to remove (current_val + new_bond_order - std_val) bonds worth.
    # We only handle the common case: remove exactly one explicit H.
    for nb in atom.GetNeighbors():
        if nb.GetSymbol() == "H":
            return nb.GetIdx()
    return None  # No explicit H; caller will log a warning.


def resolve_ligand_deletions(
    rd_mol: Chem.Mol, spec: CovalentLinkSpec
) -> Tuple[List[int], List[str]]:
    """Return (atom_indices_to_delete, auto_detected_atom_names_log)."""
    _, lig_atom_name = parse_ligand_atom_spec(spec.ligand_atom_spec)
    attach_idx = _rdkit_find_atom_by_name(rd_mol, lig_atom_name)
    auto_log: List[str] = []

    # User-provided deletions override auto-detection entirely.
    if spec.delete_ligand_atoms:
        indices: List[int] = []
        for name in spec.delete_ligand_atoms:
            try:
                indices.append(_rdkit_find_atom_by_name(rd_mol, name))
            except ValueError as exc:
                raise ValueError(f"covalent_delete_ligand_atoms: {exc}")
        return indices, []

    order_map = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}
    bond_order = order_map.get(spec.bond_order.upper(), 1)
    h_idx = _auto_h_to_remove_rdkit(rd_mol, attach_idx, bond_order)
    if h_idx is not None:
        auto_log.append(_ligand_atom_name_from_index(rd_mol, h_idx))
        return [h_idx], auto_log
    return [], auto_log


# ---------------------------------------------------------------------------
# Ligand-side deletion on the RDKit mol
# ---------------------------------------------------------------------------

def _tag_atoms_with_ligand_names(rd_mol: Chem.Mol) -> None:
    """Mark each atom with a 'chemem_name' property matching Ligand.get_atom_names()
    (element-upper + 1-based count-within-element). This is how the merged
    parmed structure names ligand atoms, so we tag once before truncation and
    read back after to produce name strings that align with the real complex.
    """
    from collections import Counter
    counts: Counter = Counter()
    for atom in rd_mol.GetAtoms():
        sym = atom.GetSymbol()
        counts[sym] += 1
        atom.SetProp("chemem_name", f"{sym.upper()}{counts[sym]}")


def _truncate_ligand_for_fragment(
    rd_mol: Chem.Mol, spec: CovalentLinkSpec
) -> Chem.Mol:
    """Return a truncated copy of the ligand with leaving atoms removed and
    the attachment atom marked as non-implicit-H (so that Chem.SanitizeMol
    doesn't re-grow the H). Populates spec.resolved_ligand_atom_name and
    spec.auto_deleted_ligand_atoms.

    Atoms in the returned mol carry a 'chemem_name' property equal to the
    name that atom has on the original Ligand (and in the merged parmed
    structure). This keeps fragment term extraction aligned with the real
    complex's atom naming after truncation.

    The real Ligand's mol is not mutated.
    """
    _, lig_atom_name = parse_ligand_atom_spec(spec.ligand_atom_spec)
    attach_idx = _rdkit_find_atom_by_name(rd_mol, lig_atom_name)

    indices_to_delete, auto_log = resolve_ligand_deletions(rd_mol, spec)
    spec.auto_deleted_ligand_atoms = list(auto_log)
    spec.resolved_ligand_atom_name = lig_atom_name

    # Work on a copy so tagging doesn't mutate the caller's mol.
    work = Chem.RWMol(rd_mol)
    _tag_atoms_with_ligand_names(work)

    if indices_to_delete:
        # Prevent RDKit from re-adding an implicit H on the attachment atom.
        work.GetAtomWithIdx(attach_idx).SetNoImplicit(True)
        for idx in sorted(set(indices_to_delete), reverse=True):
            work.RemoveAtom(idx)

    new_mol = work.GetMol()
    Chem.SanitizeMol(new_mol)
    return new_mol


# ---------------------------------------------------------------------------
# Protein-side deletion on the parmed structure
# ---------------------------------------------------------------------------

def _find_parmed_residue(structure, chain: str, resname: str, resnum: int):
    for res in structure.residues:
        if (str(res.chain) == str(chain)
                and res.name == resname
                and int(res.number) == int(resnum)):
            return res
    return None


def _auto_h_to_remove_parmed(residue, attach_atom_name: str,
                             new_bond_order: int) -> Optional[str]:
    """Return the atom name of an H to remove from attach_atom, or None."""
    attach = next((a for a in residue.atoms if a.name == attach_atom_name), None)
    if attach is None:
        return None
    # Count current heavy+H bond orders
    current_val = 0
    for b in attach.bonds:
        current_val += int(getattr(b, "order", 1) or 1)
    std_val = _STD_VALENCE.get(attach.element_name.capitalize(), None)
    if std_val is None:
        return None
    if current_val + new_bond_order <= std_val:
        return None
    for b in attach.bonds:
        other = b.atom2 if b.atom1 is attach else b.atom1
        if other.element == 1:  # hydrogen
            return other.name
    return None


def apply_protein_deletions(protein_structure, spec: CovalentLinkSpec) -> None:
    """Remove leaving atoms from the parmed protein structure in-place and
    absorb any deleted H's partial charge into its bonded heavy neighbor so
    that residue net charge remains integer.
    """
    chain, resname, resnum, atom_name = parse_protein_atom_spec(spec.protein_atom_spec)
    spec.resolved_protein_chain = chain
    spec.resolved_protein_resname = resname
    spec.resolved_protein_resnum = resnum
    spec.resolved_protein_atom_name = atom_name

    residue = _find_parmed_residue(protein_structure, chain, resname, resnum)
    if residue is None:
        raise RuntimeError(
            f"[covalent] protein residue not found for spec '{spec.protein_atom_spec}'."
        )

    # Resolve which atom names to delete
    if spec.delete_protein_atoms:
        to_delete_names = list(spec.delete_protein_atoms)
        auto_log: List[str] = []
    else:
        order_map = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}
        bond_order = order_map.get(spec.bond_order.upper(), 1)
        auto_h = _auto_h_to_remove_parmed(residue, atom_name, bond_order)
        to_delete_names = [auto_h] if auto_h else []
        auto_log = list(to_delete_names)

    spec.auto_deleted_protein_atoms = auto_log

    if not to_delete_names:
        return

    # Collect Atom objects to delete, and absorb their charges into neighbors
    atoms_to_delete = []
    for name in to_delete_names:
        atom = next((a for a in residue.atoms if a.name == name), None)
        if atom is None:
            raise RuntimeError(
                f"[covalent] protein atom '{name}' not found in "
                f"{chain}:{resname}:{resnum}."
            )
        atoms_to_delete.append(atom)

    # Charge absorption: for each deleted atom, add its partial charge to its
    # bonded heavy-atom neighbor (preferring atoms not also being deleted).
    delete_set = set(atoms_to_delete)
    for atom in atoms_to_delete:
        if abs(float(getattr(atom, "charge", 0.0) or 0.0)) < 1e-12:
            continue
        neighbors = [
            (b.atom2 if b.atom1 is atom else b.atom1)
            for b in atom.bonds
        ]
        target = next(
            (nb for nb in neighbors if nb.element != 1 and nb not in delete_set),
            None,
        )
        if target is None:
            target = next((nb for nb in neighbors if nb not in delete_set), None)
        if target is not None:
            target.charge = float(target.charge) + float(atom.charge)

    # Delete atoms from the structure. parmed's recommended method is to mark
    # with strip(); we use a boolean mask over all atoms.
    keep_mask = np.ones(len(protein_structure.atoms), dtype=bool)
    delete_idx_set = {a.idx for a in atoms_to_delete}
    for i in delete_idx_set:
        keep_mask[i] = False
    # strip inverts a mask: True means "strip this atom"
    protein_structure.strip(~keep_mask)

    print(
        f"[covalent] protein deletions: removed "
        f"{sorted(to_delete_names)} from {chain}:{resname}:{resnum} "
        f"{'(auto)' if auto_log else '(user)'}"
    )


# ---------------------------------------------------------------------------
# Fragment construction & parameterization
# ---------------------------------------------------------------------------

def _build_capped_sidechain(resname: str, attachment_atom_symbol: str) -> Tuple[Chem.Mol, int]:
    """Build a small RDKit mol for a methyl-capped residue sidechain and return
    (mol_with_explicit_Hs, attachment_atom_index).

    Falls back to a generic methyl-X-H fragment if the residue is unknown.
    """
    tmpl = _SIDECHAIN_TEMPLATES.get(resname.upper())
    if tmpl is not None:
        mol = Chem.MolFromSmiles(tmpl["smiles"])
        if mol is None:
            raise RuntimeError(f"[covalent] bad sidechain template for {resname}")
        mol = Chem.AddHs(mol)
        # Pick the first atom matching the attachment element — robust for
        # tiny templates with exactly one such heavy atom.
        attach_sym = tmpl["attach_atom"]
    else:
        # Generic fallback: CH3 - X where X is the attachment element
        smi = f"C{attachment_atom_symbol}"
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError(
                f"[covalent] cannot build generic cap for element {attachment_atom_symbol}"
            )
        mol = Chem.AddHs(mol)
        attach_sym = attachment_atom_symbol

    # Find the attachment atom index (first heavy atom of the target element)
    attach_idx = next(
        (a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == attach_sym),
        None,
    )
    if attach_idx is None:
        raise RuntimeError(
            f"[covalent] sidechain template for {resname} missing '{attach_sym}'"
        )
    return mol, attach_idx


def _form_junction_mol(
    truncated_ligand: Chem.Mol,
    ligand_attach_idx: int,
    capped_sidechain: Chem.Mol,
    sidechain_attach_idx: int,
    bond_order_str: str,
) -> Tuple[Chem.Mol, int, int]:
    """Combine the two RDKit mols into one, form the covalent bond, return
    (combined_mol, ligand_attach_idx_in_combined, sidechain_attach_idx_in_combined).

    Uses atom map numbers on copies so indices can be recovered after atoms
    are removed (we peel one H off the sidechain attachment atom to make room
    for the new bond). Inputs are not mutated.
    """
    # Work on copies so we don't mutate the caller's mols. Drop any existing
    # conformers — the combined fragment will be re-embedded from scratch so
    # it has coherent 3D coords for OpenFF / PDB export.
    lig_copy = Chem.Mol(truncated_ligand)
    cap_copy = Chem.Mol(capped_sidechain)
    lig_copy.RemoveAllConformers()
    cap_copy.RemoveAllConformers()
    for atom in lig_copy.GetAtoms():
        atom.SetAtomMapNum(0)
    for atom in cap_copy.GetAtoms():
        atom.SetAtomMapNum(0)
    lig_copy.GetAtomWithIdx(ligand_attach_idx).SetAtomMapNum(101)
    cap_copy.GetAtomWithIdx(sidechain_attach_idx).SetAtomMapNum(202)

    combined = Chem.CombineMols(lig_copy, cap_copy)
    rw = Chem.RWMol(combined)

    def _find_mapped(mol: Chem.RWMol, map_num: int) -> int:
        for a in mol.GetAtoms():
            if a.GetAtomMapNum() == map_num:
                return a.GetIdx()
        raise RuntimeError(f"[covalent] lost attachment atom map {map_num}")

    # Peel one H off sidechain attachment (valence room for the new bond)
    side_idx = _find_mapped(rw, 202)
    side_atom = rw.GetAtomWithIdx(side_idx)
    for nb in list(side_atom.GetNeighbors()):
        if nb.GetSymbol() == "H":
            rw.RemoveAtom(nb.GetIdx())
            break

    lig_idx = _find_mapped(rw, 101)
    side_idx = _find_mapped(rw, 202)

    order_map = {
        "SINGLE": Chem.BondType.SINGLE,
        "DOUBLE": Chem.BondType.DOUBLE,
        "TRIPLE": Chem.BondType.TRIPLE,
    }
    rw.AddBond(lig_idx, side_idx, order_map.get(bond_order_str.upper(), Chem.BondType.SINGLE))

    # Clear map numbers to keep OpenFF happy
    for a in rw.GetAtoms():
        a.SetAtomMapNum(0)

    final = rw.GetMol()
    Chem.SanitizeMol(final)
    # Embed 3D coords — OpenFF's PDB-based parmed conversion requires them.
    status = AllChem.EmbedMolecule(final, randomSeed=0xF00D)
    if status != 0:
        status = AllChem.EmbedMolecule(
            final, randomSeed=0xF00D, useRandomCoords=True
        )
    if status == 0:
        try:
            AllChem.MMFFOptimizeMolecule(final)
        except Exception:
            pass
    return final, lig_idx, side_idx


def build_and_parameterize_fragment(
    ligand_mol: Chem.Mol,
    spec: CovalentLinkSpec,
) -> Dict:
    """Build a capped junction fragment from the FULL ligand mol + a capped
    residue sidechain, parameterize the fragment with OpenFF, and return a
    dict of extracted junction terms + a ligand-side charge map.

    Internally builds a truncated copy of the ligand (leaving-group atoms
    removed) only for the fragment — the caller's ligand_mol is not mutated.
    """
    from .openff_ligand import load_ligand_structure

    assert spec.resolved_protein_resname is not None

    truncated_ligand_mol = _truncate_ligand_for_fragment(ligand_mol, spec)
    _, lig_atom_name = parse_ligand_atom_spec(spec.ligand_atom_spec)
    ligand_attach_idx = _rdkit_find_atom_by_name(truncated_ligand_mol, lig_atom_name)

    # Determine attachment element of the protein side from the atom name
    # (e.g. 'SG' -> 'S'). First char is element in standard PDB naming.
    attach_elem = spec.resolved_protein_atom_name[0]
    cap_mol, cap_attach_idx = _build_capped_sidechain(
        spec.resolved_protein_resname, attach_elem
    )

    fragment_mol, lig_idx_frag, prot_idx_frag = _form_junction_mol(
        truncated_ligand_mol,
        ligand_attach_idx,
        cap_mol,
        cap_attach_idx,
        spec.bond_order,
    )

    fragment_structure, _ = load_ligand_structure(fragment_mol)
    spec.fragment_structure = fragment_structure

    # Map each ligand-side fragment atom (indices 0..n_lig-1) to its name on
    # the ORIGINAL Ligand / merged parmed structure. `_truncate_ligand_for_fragment`
    # tagged every atom with a 'chemem_name' property before removing leaving
    # atoms, so each surviving atom still carries its real name.
    n_lig = truncated_ligand_mol.GetNumAtoms()
    lig_frag_names: List[str] = []
    for i in range(n_lig):
        src_atom = truncated_ligand_mol.GetAtomWithIdx(i)
        if src_atom.HasProp("chemem_name"):
            lig_frag_names.append(src_atom.GetProp("chemem_name"))
        else:
            # Fallback: reconstruct name by element count from the truncated mol.
            from collections import Counter as _C
            counts: _C = _C()
            names = []
            for a in truncated_ligand_mol.GetAtoms():
                sym = a.GetSymbol()
                counts[sym] += 1
                names.append(f"{sym.upper()}{counts[sym]}")
            lig_frag_names = names
            break

    # Harvest partial charges of ligand-side fragment atoms
    frag_ligand_charges: Dict[str, float] = {}
    for i, name in enumerate(lig_frag_names):
        atom = fragment_structure.atoms[i]
        frag_ligand_charges[name] = float(atom.charge)

    # Extract junction bonded terms
    lig_attach_name = lig_frag_names[lig_idx_frag]  # ligand-side name
    # Protein-side atom in the merged complex will be identified later
    # via (chain, resname, resnum, atom_name) — we only need the fragment
    # atom here.

    def _fragment_atom_role(atom_idx: int) -> str:
        """'ligand', 'protein_attach', or 'cap' (cap-only fragment atoms
        that have no real counterpart)."""
        if atom_idx < n_lig:
            return "ligand"
        if atom_idx == prot_idx_frag:
            return "protein_attach"
        return "cap"

    def _atom_name_for(atom_idx: int) -> Optional[str]:
        role = _fragment_atom_role(atom_idx)
        if role == "ligand":
            return lig_frag_names[atom_idx]
        if role == "protein_attach":
            return spec.resolved_protein_atom_name
        return None  # cap atom

    # Bond across the junction
    junction_bond_params = None
    for b in fragment_structure.bonds:
        pair = {b.atom1.idx, b.atom2.idx}
        if pair == {lig_idx_frag, prot_idx_frag}:
            if b.type is not None:
                junction_bond_params = (float(b.type.k), float(b.type.req))
            break
    spec.junction_bond_params = junction_bond_params

    # Helper: does a term contain the new covalent bond as one of its
    # consecutive edges? For angles a1-a2-a3 the edges are (a1,a2) and
    # (a2,a3); for dihedrals a1-a2-a3-a4 they are (a1,a2), (a2,a3), (a3,a4).
    # We also enforce that EVERY consecutive edge is a real bond in the
    # fragment — this filters out any parmed-expanded impropers / 1-4 pair
    # interactions that aren't true chain terms.
    junction_pair = frozenset({lig_idx_frag, prot_idx_frag})
    fragment_bond_set = {
        frozenset({b.atom1.idx, b.atom2.idx}) for b in fragment_structure.bonds
    }

    def _is_chain(ids: List[int]) -> bool:
        for i in range(len(ids) - 1):
            if frozenset({ids[i], ids[i + 1]}) not in fragment_bond_set:
                return False
        return True

    def _contains_junction_edge(ids: List[int]) -> bool:
        for i in range(len(ids) - 1):
            if frozenset({ids[i], ids[i + 1]}) == junction_pair:
                return True
        return False

    junction_angles: List[Tuple[Tuple[str, str, str], float, float]] = []
    for ang in fragment_structure.angles:
        ids = [ang.atom1.idx, ang.atom2.idx, ang.atom3.idx]
        if "cap" in [_fragment_atom_role(i) for i in ids]:
            continue
        if not _contains_junction_edge(ids):
            continue
        if not _is_chain(ids):
            continue
        names = tuple(_atom_name_for(i) for i in ids)  # type: ignore[misc]
        if None in names or ang.type is None:
            continue
        junction_angles.append(
            (names, float(ang.type.k), float(ang.type.theteq))  # type: ignore[arg-type]
        )
    spec.junction_angles = junction_angles

    junction_dihedrals: List[Tuple[Tuple[str, str, str, str], float, float, int]] = []
    for d in fragment_structure.dihedrals:
        # parmed flags non-proper torsions (impropers / 1-4 pair dihedrals)
        # via d.improper or d.ignore_end. Skip anything that is not a proper
        # chain dihedral containing the junction bond.
        if getattr(d, "improper", False):
            continue
        ids = [d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx]
        if "cap" in [_fragment_atom_role(i) for i in ids]:
            continue
        if not _contains_junction_edge(ids):
            continue
        # Reject SMIRNOFF trefoil impropers that parmed stores in .dihedrals
        # with improper=False — these are not chain torsions (consecutive
        # atoms are not all bonded in the fragment).
        if not _is_chain(ids):
            continue
        names = tuple(_atom_name_for(i) for i in ids)  # type: ignore[misc]
        if None in names or d.type is None:
            continue
        junction_dihedrals.append(
            (
                names,
                float(d.type.phi_k),
                float(d.type.phase),
                int(d.type.per),
            )
        )
    spec.junction_dihedrals = junction_dihedrals
    spec.fragment_ligand_charges = frag_ligand_charges

    print(
        f"[covalent] fragment parameterized: "
        f"bond={'ok' if junction_bond_params else 'missing'}, "
        f"angles={len(junction_angles)}, "
        f"dihedrals={len(junction_dihedrals)}"
    )
    return {
        "bond_params": junction_bond_params,
        "angles": junction_angles,
        "dihedrals": junction_dihedrals,
        "ligand_charges": frag_ligand_charges,
    }


# ---------------------------------------------------------------------------
# Injection into the merged complex
# ---------------------------------------------------------------------------

def _strip_ligand_atoms_by_name(complex_structure, lig_res_name: str,
                                 atom_names: List[str]) -> None:
    """Remove atoms from a ligand residue in the merged parmed structure by
    atom name. Used to mirror covalent leaving-group deletions after the
    ligand has already been merged with the protein."""
    target_indices = set()
    for res in complex_structure.residues:
        if res.name != lig_res_name:
            continue
        for atom in res.atoms:
            if atom.name in atom_names:
                target_indices.add(atom.idx)
    if not target_indices:
        return
    keep_mask = np.ones(len(complex_structure.atoms), dtype=bool)
    for i in target_indices:
        keep_mask[i] = False
    complex_structure.strip(~keep_mask)
    print(
        f"[covalent] stripped leaving atoms from {lig_res_name}: "
        f"{sorted(atom_names)}"
    )


def _find_complex_atom(
    complex_structure,
    residue_name: str,
    atom_name: str,
    resnum: Optional[int] = None,
    chain: Optional[str] = None,
):
    for res in complex_structure.residues:
        if res.name != residue_name:
            continue
        if resnum is not None and int(res.number) != int(resnum):
            continue
        if chain is not None and str(res.chain) != str(chain):
            continue
        for a in res.atoms:
            if a.name == atom_name:
                return a
    return None


def _bond_exists(structure, i: int, j: int) -> bool:
    key = tuple(sorted((i, j)))
    for b in structure.bonds:
        if tuple(sorted((b.atom1.idx, b.atom2.idx))) == key:
            return True
    return False


def _angle_exists(structure, i: int, j: int, k: int) -> bool:
    key = (j, tuple(sorted((i, k))))  # central atom + unordered terminals
    for a in structure.angles:
        if a.atom2.idx == j and tuple(sorted((a.atom1.idx, a.atom3.idx))) == key[1]:
            return True
    return False


def _dihedral_exists(structure, i: int, j: int, k: int, l: int) -> bool:
    target = tuple(sorted((tuple(sorted((i, j, k, l))),)))
    for d in structure.dihedrals:
        ids = tuple(sorted((d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx)))
        if ids == tuple(sorted((i, j, k, l))):
            return True
    return False


def inject_covalent_bonds(complex_structure, ligand_objects) -> None:
    """Inject covalent bond/angle/dihedral terms into the merged parmed
    structure for each ligand that carries a CovalentLinkSpec. Also update
    ligand partial charges with fragment-derived values (junction polarization)
    and renormalize to integer net charge.

    Must be called AFTER protein+ligand merge but BEFORE createSystem().
    """
    from parmed.topologyobjects import (
        Bond, BondType,
        Angle, AngleType,
        Dihedral, DihedralType,
    )

    covalent_ligands = [lig for lig in ligand_objects if getattr(lig, "covalent_link", None)]
    if not covalent_ligands:
        return

    for lig_obj in covalent_ligands:
        spec: CovalentLinkSpec = lig_obj.covalent_link
        if spec.junction_bond_params is None:
            print(f"[covalent] WARNING: no junction bond params for {lig_obj.ligand_id}; skipping")
            continue

        lig_res_name = lig_obj.ligand_id  # e.g. 'LIG_0' (set by create_system)

        # Strip leaving ligand atoms from the merged parmed structure. This
        # mirrors the truncation that the fragment parameterization saw, so
        # that junction atom-name mappings still resolve correctly.
        leaving_names = list(spec.delete_ligand_atoms or []) + list(spec.auto_deleted_ligand_atoms or [])
        if leaving_names:
            _strip_ligand_atoms_by_name(complex_structure, lig_res_name, leaving_names)

        # Resolve ligand attachment atom in the (now truncated) merged structure.
        lig_atom_name = spec.resolved_ligand_atom_name
        lig_atom = _find_complex_atom(complex_structure, lig_res_name, lig_atom_name)
        if lig_atom is None:
            raise RuntimeError(
                f"[covalent] merged structure missing ligand atom {lig_res_name}:{lig_atom_name}"
            )

        # Resolve protein attachment atom
        prot_atom = _find_complex_atom(
            complex_structure,
            spec.resolved_protein_resname,
            spec.resolved_protein_atom_name,
            resnum=spec.resolved_protein_resnum,
            chain=spec.resolved_protein_chain,
        )
        if prot_atom is None:
            raise RuntimeError(
                f"[covalent] merged structure missing protein atom "
                f"{spec.resolved_protein_chain}:{spec.resolved_protein_resname}:"
                f"{spec.resolved_protein_resnum}:{spec.resolved_protein_atom_name}"
            )

        def _resolve_lig(atom_name: str):
            return _find_complex_atom(complex_structure, lig_res_name, atom_name)

        def _resolve_any(name: str):
            if name == spec.resolved_protein_atom_name:
                return prot_atom
            return _resolve_lig(name)

        # --- Inject the junction bond ---
        k_bond, r0 = spec.junction_bond_params
        if not _bond_exists(complex_structure, lig_atom.idx, prot_atom.idx):
            bt = BondType(k=k_bond, req=r0, list=complex_structure.bond_types)
            complex_structure.bond_types.append(bt)
            complex_structure.bonds.append(Bond(lig_atom, prot_atom, type=bt))

        # --- Inject junction angles ---
        for names, k_ang, theta0 in spec.junction_angles:
            a1 = _resolve_any(names[0])
            a2 = _resolve_any(names[1])
            a3 = _resolve_any(names[2])
            if None in (a1, a2, a3):
                continue
            if _angle_exists(complex_structure, a1.idx, a2.idx, a3.idx):
                continue
            at = AngleType(k=k_ang, theteq=theta0, list=complex_structure.angle_types)
            complex_structure.angle_types.append(at)
            complex_structure.angles.append(Angle(a1, a2, a3, type=at))

        # --- Inject junction dihedrals ---
        for names, phi_k, phase, per in spec.junction_dihedrals:
            a1 = _resolve_any(names[0])
            a2 = _resolve_any(names[1])
            a3 = _resolve_any(names[2])
            a4 = _resolve_any(names[3])
            if None in (a1, a2, a3, a4):
                continue
            if _dihedral_exists(complex_structure, a1.idx, a2.idx, a3.idx, a4.idx):
                continue
            dt = DihedralType(
                phi_k=phi_k, per=per, phase=phase,
                list=complex_structure.dihedral_types,
            )
            complex_structure.dihedral_types.append(dt)
            complex_structure.dihedrals.append(
                Dihedral(a1, a2, a3, a4, type=dt)
            )

        # --- Junction polarization: copy fragment ligand charges & renormalize ---
        _redistribute_ligand_charges(
            complex_structure, lig_res_name, spec.fragment_ligand_charges
        )

        print(
            f"[covalent] injected bond {spec.resolved_protein_chain}:"
            f"{spec.resolved_protein_resname}:{spec.resolved_protein_resnum}:"
            f"{spec.resolved_protein_atom_name} — {lig_res_name}:{lig_atom_name} "
            f"(k={k_bond:.1f}, r0={r0:.3f})"
        )


def _redistribute_ligand_charges(
    complex_structure,
    lig_res_name: str,
    fragment_charges: Dict[str, float],
) -> None:
    """Copy fragment-derived charges onto the ligand atoms in the merged
    structure (by atom name), then renormalize ligand total charge to the
    nearest integer by spreading the residual evenly across all ligand atoms.
    """
    ligand_atoms = [
        a for a in complex_structure.atoms if a.residue.name == lig_res_name
    ]
    if not ligand_atoms:
        return

    before_total = float(sum(a.charge for a in ligand_atoms))

    for atom in ligand_atoms:
        if atom.name in fragment_charges:
            atom.charge = float(fragment_charges[atom.name])

    after_total = float(sum(a.charge for a in ligand_atoms))
    target = float(round(after_total))
    residual = target - after_total
    if abs(residual) > 1e-9 and len(ligand_atoms) > 0:
        per_atom = residual / len(ligand_atoms)
        for atom in ligand_atoms:
            atom.charge = float(atom.charge) + per_atom
    final_total = float(sum(a.charge for a in ligand_atoms))

    print(
        f"[covalent] {lig_res_name} net charge: "
        f"{before_total:+.4f} (pre) → {after_total:+.4f} (polarized) → "
        f"{final_total:+.4f} (renormalized)"
    )
