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


# ---------------------------------------------------------------------------
# Addition-warhead saturation (nitrile / Michael acceptor)
# ---------------------------------------------------------------------------
# An "addition" warhead reacts by ADDING across an existing multiple bond rather
# than displacing a leaving group: a nitrile C#N becomes a thioimidate C=N, or a
# Michael-acceptor C=C becomes a saturated C-C, with the freed valences taking up
# H. ChemEM's covalent path is deletion/auto-H based and would over-valence such a
# warhead. We fix it BEFORE parameterization by "saturating" the warhead: drop its
# highest-order incident pi bond by one and let RDKit fill the freed valences with
# H. That yields exactly the already-supported "bound-form" case — the standard
# auto-H path then removes one warhead H to make room for the protein bond.

def warhead_needs_saturation(rd_mol: Chem.Mol, warhead_name: str,
                             new_bond_order: int) -> bool:
    """True iff forming the covalent bond would over-valence the warhead, there is
    no removable H (so auto-H can't handle it), but it has a reducible pi bond."""
    try:
        idx = _rdkit_find_atom_by_name(rd_mol, warhead_name)
    except ValueError:
        return False
    atom = rd_mol.GetAtomWithIdx(idx)
    std = _STD_VALENCE.get(atom.GetSymbol())
    if std is None:
        return False
    if atom.GetTotalValence() + new_bond_order <= std:
        return False  # fits without any change
    if _auto_h_to_remove_rdkit(rd_mol, idx, new_bond_order) is not None:
        return False  # a removable H exists → the normal auto-H path handles it
    return any(b.GetBondTypeAsDouble() >= 2.0 for b in atom.GetBonds())


def saturate_addition_warhead(rd_mol: Chem.Mol, warhead_name: str) -> Chem.Mol:
    """Return a NEW mol with the warhead's highest-order incident pi bond reduced by
    one and the freed valences filled with H (heavy-atom order/coords preserved, so
    the warhead's ChemEM name is unchanged)."""
    heavy = Chem.RemoveHs(rd_mol)  # keeps heavy-atom conformer coords
    idx = _rdkit_find_atom_by_name(heavy, warhead_name)
    atom = heavy.GetAtomWithIdx(idx)
    target = None
    for b in atom.GetBonds():
        if b.GetBondTypeAsDouble() >= 2.0 and (
            target is None or b.GetBondTypeAsDouble() > target.GetBondTypeAsDouble()
        ):
            target = b
    if target is None:
        return rd_mol
    lower = {Chem.BondType.TRIPLE: Chem.BondType.DOUBLE,
             Chem.BondType.DOUBLE: Chem.BondType.SINGLE}.get(target.GetBondType())
    if lower is None:
        return rd_mol
    rw = Chem.RWMol(heavy)
    rw.GetBondBetweenAtoms(target.GetBeginAtomIdx(),
                           target.GetEndAtomIdx()).SetBondType(lower)
    new = rw.GetMol()
    Chem.SanitizeMol(new)                 # fills the freed valences with implicit H
    return Chem.AddHs(new, addCoords=True) # make them explicit with 3D coords


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
    rd_mol: Chem.Mol, specs: List[CovalentLinkSpec]
) -> Chem.Mol:
    """Return a truncated copy of the ligand with the leaving atoms of EVERY
    covalent bond removed, and each attachment atom marked non-implicit-H (so
    Chem.SanitizeMol doesn't re-grow the H). Populates each spec's
    resolved_ligand_atom_name and auto_deleted_ligand_atoms.

    Atoms in the returned mol carry a 'chemem_name' property equal to the
    name that atom has on the original Ligand (and in the merged parmed
    structure). This keeps fragment term extraction aligned with the real
    complex's atom naming after truncation.

    The real Ligand's mol is not mutated.
    """
    if isinstance(specs, CovalentLinkSpec):        # tolerate a bare spec
        specs = [specs]

    attach_indices = []
    all_to_delete: set = set()
    for spec in specs:
        _, lig_atom_name = parse_ligand_atom_spec(spec.ligand_atom_spec)
        attach_indices.append(_rdkit_find_atom_by_name(rd_mol, lig_atom_name))
        spec.resolved_ligand_atom_name = lig_atom_name

        indices_to_delete, auto_log = resolve_ligand_deletions(rd_mol, spec)
        spec.auto_deleted_ligand_atoms = list(auto_log)
        all_to_delete.update(indices_to_delete)

    # Work on a copy so tagging doesn't mutate the caller's mol.
    work = Chem.RWMol(rd_mol)
    _tag_atoms_with_ligand_names(work)

    if all_to_delete:
        # Prevent RDKit from re-adding implicit H's on any attachment atom.
        for attach_idx in attach_indices:
            work.GetAtomWithIdx(attach_idx).SetNoImplicit(True)
        for idx in sorted(all_to_delete, reverse=True):
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
                             new_bond_order: int) -> List[str]:
    """Return the names of the H's to remove from attach_atom so the new bond
    fits its standard valence, or [].

    Removes as many H's as the new bond needs (a DOUBLE Schiff base to a charged
    Lys Nζ needs all 3 ammonium H's gone; a SINGLE bond to Ser Oγ needs 1). The
    count is valence-based (current bonds + new order − standard valence), so it
    is correct for both charged and neutral attachment atoms — unlike a fixed
    one-H removal, which left DOUBLE-bond nitrogens over-valent.
    """
    attach = next((a for a in residue.atoms if a.name == attach_atom_name), None)
    if attach is None:
        return []
    # Count current heavy+H bond orders
    current_val = 0
    for b in attach.bonds:
        current_val += int(getattr(b, "order", 1) or 1)
    std_val = _STD_VALENCE.get(attach.element_name.capitalize(), None)
    if std_val is None:
        return []
    n_remove = current_val + new_bond_order - std_val
    if n_remove <= 0:
        return []
    h_names: List[str] = []
    for b in attach.bonds:
        other = b.atom2 if b.atom1 is attach else b.atom1
        if other.element == 1:  # hydrogen
            h_names.append(other.name)
            if len(h_names) >= n_remove:
                break
    if len(h_names) < n_remove:
        print(
            f"[covalent] WARNING: attachment atom '{attach_atom_name}' needs "
            f"{n_remove} bond(s) freed for the new covalent bond but only "
            f"{len(h_names)} hydrogen(s) are available to remove. Specify "
            "covalent_delete_protein_atoms explicitly, or the atom will be "
            "over-valent."
        )
    return h_names


def _warn_if_acyl_attachment(residue, attach_atom_name: str,
                             user_specified: bool) -> None:
    """Warn when bonding to a carboxyl/carbonyl carbon with nothing set to leave.

    A force-field-built structure carries no bond orders, so a backbone carbonyl
    carbon looks trivalent to the valence check above and no leaving atom is
    detected — yet forming an amide there really does displace the hydroxyl/OXT.
    Detected by the attachment carbon carrying two terminal oxygens (a carboxylate
    or C-terminus). Advisory only; the user decides what leaves.
    """
    if user_specified:
        return
    attach = next((a for a in residue.atoms if a.name == attach_atom_name), None)
    if attach is None or attach.element_name.capitalize() != "C":
        return

    terminal_oxygens = []
    for b in attach.bonds:
        other = b.atom2 if b.atom1 is attach else b.atom1
        if other.element == 8 and len(other.bonds) == 1:
            terminal_oxygens.append(other.name)

    if len(terminal_oxygens) >= 2:
        print(
            f"[covalent] WARNING: bonding to carboxyl carbon "
            f"{residue.name}:{residue.number}:{attach_atom_name}, which carries "
            f"terminal oxygens {terminal_oxygens}. Forming an amide/ester here "
            "normally displaces one of them, but force-field structures carry no "
            "bond orders so this cannot be detected automatically. Set "
            f"covalent_delete_protein_atoms = ['{terminal_oxygens[-1]}'] if that "
            "is the intended chemistry."
        )


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
    _warn_if_acyl_attachment(residue, atom_name, bool(spec.delete_protein_atoms))

    if spec.delete_protein_atoms:
        to_delete_names = list(spec.delete_protein_atoms)
        auto_log: List[str] = []
    else:
        order_map = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}
        bond_order = order_map.get(spec.bond_order.upper(), 1)
        to_delete_names = _auto_h_to_remove_parmed(residue, atom_name, bond_order)
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
    # Only use the residue template when the requested attachment atom is the one
    # the template is built around (its designated sidechain atom). When the user
    # points at a different atom — a backbone N (N-terminal Schiff base), a Cβ, etc.
    # — the sidechain template would attach at the WRONG atom, so fall back to a
    # generic methyl-X cap built on the requested element instead.
    if tmpl is not None and tmpl["attach_atom"] == attachment_atom_symbol:
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
    junctions: List[Tuple[int, Chem.Mol, int, str]],
) -> Tuple[Chem.Mol, List[int], List[int]]:
    """Combine the ligand with one capped sidechain per covalent bond, form all
    the bonds, and return
    (combined_mol, [ligand_attach_idx_in_combined], [cap_attach_idx_in_combined]).

    ``junctions`` is one (ligand_attach_idx, capped_sidechain, cap_attach_idx,
    bond_order_str) per bond, in spec order; the returned index lists are parallel
    to it.

    Atom map numbers on copies let indices survive the H-peeling and the repeated
    CombineMols: the k-th ligand attachment is tagged 101+k and its cap attachment
    202+k. Inputs are not mutated.
    """
    # Work on copies so we don't mutate the caller's mols. Drop any existing
    # conformers — the combined fragment will be re-embedded from scratch so
    # it has coherent 3D coords for OpenFF / PDB export.
    lig_copy = Chem.Mol(truncated_ligand)
    lig_copy.RemoveAllConformers()
    for atom in lig_copy.GetAtoms():
        atom.SetAtomMapNum(0)
    # Tag EVERY ligand attachment before the first CombineMols: that call copies
    # the mol as it stands, so later edits to lig_copy would not be reflected.
    for k, (lig_attach_idx, _cap_mol, _cap_attach_idx, _order) in enumerate(junctions):
        lig_copy.GetAtomWithIdx(lig_attach_idx).SetAtomMapNum(101 + k)

    combined = lig_copy
    for k, (lig_attach_idx, cap_mol, cap_attach_idx, _order) in enumerate(junctions):
        cap_copy = Chem.Mol(cap_mol)
        cap_copy.RemoveAllConformers()
        for atom in cap_copy.GetAtoms():
            atom.SetAtomMapNum(0)
        cap_copy.GetAtomWithIdx(cap_attach_idx).SetAtomMapNum(202 + k)
        combined = Chem.CombineMols(combined, cap_copy)

    rw = Chem.RWMol(combined)

    def _find_mapped(mol: Chem.RWMol, map_num: int) -> int:
        for a in mol.GetAtoms():
            if a.GetAtomMapNum() == map_num:
                return a.GetIdx()
        raise RuntimeError(f"[covalent] lost attachment atom map {map_num}")

    # Peel H's off each cap attachment to make valence room for its new bond.
    # A SINGLE bond needs one H removed; a DOUBLE/TRIPLE bond to a multi-H cap atom
    # (e.g. a LYS/backbone amine forming a Schiff base) needs bond-order H's removed,
    # or the cap atom ends up over-valent. Done in one pass, collecting indices
    # first, because every removal shifts the indices after it.
    h_to_remove: List[int] = []
    for k, (_lig_attach_idx, _cap_mol, _cap_attach_idx, order) in enumerate(junctions):
        n_h = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}.get(order.upper(), 1)
        side_atom = rw.GetAtomWithIdx(_find_mapped(rw, 202 + k))
        hs = [nb.GetIdx() for nb in side_atom.GetNeighbors() if nb.GetSymbol() == "H"]
        h_to_remove.extend(hs[:n_h])
    for h_idx in sorted(set(h_to_remove), reverse=True):
        rw.RemoveAtom(h_idx)

    order_map = {
        "SINGLE": Chem.BondType.SINGLE,
        "DOUBLE": Chem.BondType.DOUBLE,
        "TRIPLE": Chem.BondType.TRIPLE,
    }
    lig_idxs, cap_idxs = [], []
    for k, (_lig_attach_idx, _cap_mol, _cap_attach_idx, order) in enumerate(junctions):
        lig_idx = _find_mapped(rw, 101 + k)
        cap_idx = _find_mapped(rw, 202 + k)
        rw.AddBond(lig_idx, cap_idx, order_map.get(order.upper(), Chem.BondType.SINGLE))
        lig_idxs.append(lig_idx)
        cap_idxs.append(cap_idx)

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
    return final, lig_idxs, cap_idxs


def build_and_parameterize_fragment(
    ligand_mol: Chem.Mol,
    specs,
) -> Dict:
    """Build ONE capped junction fragment from the FULL ligand mol plus a capped
    residue sidechain per covalent bond, parameterize it with OpenFF, and return a
    dict of extracted junction terms + a ligand-side charge map.

    ``specs`` is the ligand's list of CovalentLinkSpec (a bare spec is accepted for
    back-compat). Per-bond results are written onto each spec
    (``junction_bond_params``); the angle/dihedral/charge sets, which are properties
    of the whole fragment, are returned and stored by the caller on the Ligand — a
    term bridging two junctions is then collected once, not once per bond.

    Internally builds a truncated copy of the ligand (leaving-group atoms
    removed) only for the fragment — the caller's ligand_mol is not mutated.
    """
    from .openff_ligand import load_ligand_structure

    if isinstance(specs, CovalentLinkSpec):
        specs = [specs]
    if not specs:
        return {"bond_params": None, "angles": [], "dihedrals": [], "ligand_charges": {}}
    for spec in specs:
        assert spec.resolved_protein_resname is not None

    truncated_ligand_mol = _truncate_ligand_for_fragment(ligand_mol, specs)

    junctions = []
    for spec in specs:
        _, lig_atom_name = parse_ligand_atom_spec(spec.ligand_atom_spec)
        ligand_attach_idx = _rdkit_find_atom_by_name(truncated_ligand_mol, lig_atom_name)
        # Determine attachment element of the protein side from the atom name
        # (e.g. 'SG' -> 'S'). First char is element in standard PDB naming.
        attach_elem = spec.resolved_protein_atom_name[0]
        cap_mol, cap_attach_idx = _build_capped_sidechain(
            spec.resolved_protein_resname, attach_elem
        )
        junctions.append((ligand_attach_idx, cap_mol, cap_attach_idx, spec.bond_order))

    fragment_mol, lig_idxs_frag, prot_idxs_frag = _form_junction_mol(
        truncated_ligand_mol, junctions
    )

    fragment_structure, _ = load_ligand_structure(fragment_mol)

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

    # Extract junction bonded terms.
    # Protein-side atoms in the merged complex are identified later via
    # (chain, resname, resnum, atom_name) — here we only need fragment indices.
    prot_idx_to_spec = {int(idx): spec for idx, spec in zip(prot_idxs_frag, specs)}

    def _fragment_atom_role(atom_idx: int) -> str:
        """'ligand', 'protein_attach', or 'cap' (cap-only fragment atoms
        that have no real counterpart)."""
        if atom_idx < n_lig:
            return "ligand"
        if atom_idx in prot_idx_to_spec:
            return "protein_attach"
        return "cap"

    def _atom_name_for(atom_idx: int) -> Optional[str]:
        role = _fragment_atom_role(atom_idx)
        if role == "ligand":
            return lig_frag_names[atom_idx]
        if role == "protein_attach":
            return prot_idx_to_spec[atom_idx].resolved_protein_atom_name
        return None  # cap atom

    # Bond across each junction, stored on that bond's own spec.
    junction_pairs = {
        frozenset({int(l), int(p)}) for l, p in zip(lig_idxs_frag, prot_idxs_frag)
    }
    bond_params_by_pair = {}
    for b in fragment_structure.bonds:
        pair = frozenset({b.atom1.idx, b.atom2.idx})
        if pair in junction_pairs and b.type is not None:
            bond_params_by_pair[pair] = (float(b.type.k), float(b.type.req))
    for lig_i, prot_i, spec in zip(lig_idxs_frag, prot_idxs_frag, specs):
        spec.junction_bond_params = bond_params_by_pair.get(
            frozenset({int(lig_i), int(prot_i)})
        )
    junction_bond_params = specs[0].junction_bond_params

    # Helper: does a term contain the new covalent bond as one of its
    # consecutive edges? For angles a1-a2-a3 the edges are (a1,a2) and
    # (a2,a3); for dihedrals a1-a2-a3-a4 they are (a1,a2), (a2,a3), (a3,a4).
    # We also enforce that EVERY consecutive edge is a real bond in the
    # fragment — this filters out any parmed-expanded impropers / 1-4 pair
    # interactions that aren't true chain terms.
    fragment_bond_set = {
        frozenset({b.atom1.idx, b.atom2.idx}) for b in fragment_structure.bonds
    }

    def _is_chain(ids: List[int]) -> bool:
        for i in range(len(ids) - 1):
            if frozenset({ids[i], ids[i + 1]}) not in fragment_bond_set:
                return False
        return True

    def _contains_junction_edge(ids: List[int]) -> bool:
        """True if any consecutive edge is one of the new covalent bonds.

        With several bonds a short bridge can produce a term spanning two
        junctions; it is collected once here rather than once per bond.
        """
        for i in range(len(ids) - 1):
            if frozenset({ids[i], ids[i + 1]}) in junction_pairs:
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
    n_bonds_ok = sum(1 for s in specs if s.junction_bond_params is not None)
    bond_state = (
        "ok" if n_bonds_ok == len(specs)
        else f"{n_bonds_ok}/{len(specs)}"
    )
    print(
        f"[covalent] fragment parameterized: "
        f"bonds={bond_state}, "
        f"angles={len(junction_angles)}, "
        f"dihedrals={len(junction_dihedrals)}"
    )
    return {
        "fragment_structure": fragment_structure,
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

    covalent_ligands = [lig for lig in ligand_objects if getattr(lig, "covalent_links", None)]
    if not covalent_ligands:
        return

    for lig_obj in covalent_ligands:
        specs: List[CovalentLinkSpec] = list(lig_obj.covalent_links)
        fragment = getattr(lig_obj, "covalent_fragment", None)

        lig_res_name = lig_obj.ligand_id  # e.g. 'LIG_0' (set by create_system)

        # Strip the leaving atoms of EVERY bond from the merged parmed structure,
        # in one pass. This mirrors the truncation that the fragment
        # parameterization saw, so junction atom-name mappings still resolve.
        leaving_names = []
        for spec in specs:
            leaving_names += list(spec.delete_ligand_atoms or [])
            leaving_names += list(spec.auto_deleted_ligand_atoms or [])
        if leaving_names:
            _strip_ligand_atoms_by_name(
                complex_structure, lig_res_name, sorted(set(leaving_names))
            )

        # Resolve both ends of every bond up front, so a name lookup below can
        # tell a protein attachment atom from a ligand atom.
        resolved = []          # (spec, lig_atom, prot_atom)
        prot_atom_by_name = {}
        for spec in specs:
            lig_atom_name = spec.resolved_ligand_atom_name
            lig_atom = _find_complex_atom(complex_structure, lig_res_name, lig_atom_name)
            if lig_atom is None:
                raise RuntimeError(
                    f"[covalent] merged structure missing ligand atom {lig_res_name}:{lig_atom_name}"
                )
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
            resolved.append((spec, lig_atom, prot_atom))
            prot_atom_by_name[spec.resolved_protein_atom_name] = prot_atom

        def _resolve_lig(atom_name: str):
            return _find_complex_atom(complex_structure, lig_res_name, atom_name)

        def _resolve_any(name: str):
            # Protein attachment atoms are matched by name across all bonds; any
            # other name belongs to the ligand residue.
            if name in prot_atom_by_name:
                return prot_atom_by_name[name]
            return _resolve_lig(name)

        # --- Inject one junction bond per covalent link ---
        for spec, lig_atom, prot_atom in resolved:
            if spec.junction_bond_params is None:
                print(
                    f"[covalent] WARNING: no junction bond params for {lig_res_name}:"
                    f"{spec.resolved_ligand_atom_name}; skipping this bond"
                )
                continue
            k_bond, r0 = spec.junction_bond_params
            if not _bond_exists(complex_structure, lig_atom.idx, prot_atom.idx):
                bt = BondType(k=k_bond, req=r0, list=complex_structure.bond_types)
                complex_structure.bond_types.append(bt)
                complex_structure.bonds.append(Bond(lig_atom, prot_atom, type=bt))
            print(
                f"[covalent] injected bond {spec.resolved_protein_chain}:"
                f"{spec.resolved_protein_resname}:{spec.resolved_protein_resnum}:"
                f"{spec.resolved_protein_atom_name} — {lig_res_name}:"
                f"{spec.resolved_ligand_atom_name} (k={k_bond:.1f}, r0={r0:.3f})"
            )

        if fragment is None:
            continue

        # --- Inject junction angles (one fragment per ligand, so injected once) ---
        for names, k_ang, theta0 in fragment.junction_angles:
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
        for names, phi_k, phase, per in fragment.junction_dihedrals:
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
        # Once per ligand: the charges come from the single fragment spanning all
        # of its junctions, and renormalizing twice would drift the net charge.
        # The ligand's formal charge is unchanged by the reaction, so use it as the
        # renormalization target rather than rounding the drifted partial sum.
        formal_charge = None
        lig_mol = getattr(lig_obj, "mol", None)
        if lig_mol is not None:
            try:
                formal_charge = float(Chem.GetFormalCharge(lig_mol))
            except Exception:
                formal_charge = None

        _redistribute_ligand_charges(
            complex_structure, lig_res_name, fragment.fragment_ligand_charges,
            target_charge=formal_charge,
        )


def _redistribute_ligand_charges(
    complex_structure,
    lig_res_name: str,
    fragment_charges: Dict[str, float],
    target_charge: Optional[float] = None,
) -> None:
    """Copy fragment-derived charges onto the ligand atoms in the merged
    structure (by atom name), then renormalize the ligand's total charge by
    spreading the residual evenly across all ligand atoms.

    ``target_charge`` is the ligand's FORMAL charge, which the reaction does not
    change: each covalent bond swaps a bond-to-leaving-atom for a bond-to-protein,
    and the leaving atoms depart as neutral species (the halide as HCl, taking the
    protein H). Falling back on ``round(after_total)`` guesses that formal charge
    from a partial-charge sum that the stripped leaving atoms have already pulled
    away from it: each halide leaves roughly +0.29 behind, so a ligand with three
    of them (9I93's tris(chloromethyl)triazine) drifts past 0.5 and snaps to +1,
    silently adding a unit of charge to the whole system.
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
    if target_charge is None:
        target = float(round(after_total))
    else:
        target = float(target_charge)
        if abs(target - round(after_total)) > 0.5:
            print(
                f"[covalent] {lig_res_name}: renormalizing to formal charge "
                f"{target:+.0f}; the polarized partial-charge sum was "
                f"{after_total:+.4f}, which would have rounded to "
                f"{round(after_total):+.0f}."
            )
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
