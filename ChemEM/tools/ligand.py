# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>
from rdkit import Chem 
import numpy as np 
from collections import deque
from rdkit.Geometry import Point3D
from rdkit.Chem import TorsionFingerprints as tfp
from rdkit.Chem import rdMolTransforms, rdMolDescriptors, Descriptors, Crippen

# -------------------------
# Ligand misc helpers
# -------------------------
def write_to_sdf(mol: Chem.Mol, file_name: str, conf_id: int = 0) -> None:
    """Write one conformer to SDF."""
    w = Chem.SDWriter(file_name)
    w.write(mol, confId=conf_id)
    w.close()

def mol_with_positions(template_mol: Chem.Mol, coords: np.ndarray, conf_id: int = 0) -> Chem.Mol:
    """Return a copy of template_mol with a single conformer built from coords."""
    newmol = Chem.Mol(template_mol)
    newmol.RemoveAllConformers()

    n_atoms = newmol.GetNumAtoms()
    if coords.shape[0] != n_atoms or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N,3) with N={n_atoms}; got {coords.shape}")

    conf = Chem.Conformer(n_atoms)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    conf.SetId(conf_id)
    newmol.AddConformer(conf, assignId=True)
    return newmol

def get_aromatic_rings(mol):
    """
    Returns:
      aromatic_rings: list[list[Atom]]
      aromatic_indices: list[tuple[int,...]]
    """
    aromatic_rings = []
    aromatic_indices = []
    ring_info = mol.GetRingInfo()

    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic_rings.append([mol.GetAtomWithIdx(idx) for idx in ring])
            aromatic_indices.append(ring)

    return aromatic_rings, aromatic_indices


def get_torsion_lists(mol: Chem.Mol):
    """Generates list of torsion tuples (atom indices)."""
    torsion_list = tfp.CalculateTorsionLists(mol)[0]
    return_torsions = [t for sublist in torsion_list for t in sublist[0]]
    
    # Add explicit hydrogen torsions on donors
    donor_torsions = _get_donor_h_torsions(mol)
    existing_sets = {frozenset(t) for t in return_torsions}
    
    for t in donor_torsions:
        if frozenset(t) not in existing_sets and _only_h_moves_on_rotation(mol, t):
            return_torsions.append(t)
            
    return return_torsions

def _get_donor_h_torsions(mol):
    """Helper: Enumerate torsions around rotatable N/O–heavy bonds with explicit H."""
    rot_smarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
    rot_pat = Chem.MolFromSmarts(rot_smarts)
    torsions = []
    
    for i, j in mol.GetSubstructMatches(rot_pat):
        a1, a2 = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
        
        # Identify donor vs pivot
        if a1.GetAtomicNum() in (7, 8, 16) and any(n.GetAtomicNum() == 1 for n in a1.GetNeighbors()):
            donor, pivot = a1, a2
        elif a2.GetAtomicNum() in (7, 8, 16) and any(n.GetAtomicNum() == 1 for n in a2.GetNeighbors()):
            donor, pivot = a2, a1
        else:
            continue

        for nbr in pivot.GetNeighbors():
            if nbr.GetIdx() == donor.GetIdx() or nbr.GetAtomicNum() == 1: continue
            
            # Form torsion: Neighbor-Pivot-Donor-H
            for hb in donor.GetNeighbors():
                if hb.GetAtomicNum() == 1:
                    torsions.append((nbr.GetIdx(), pivot.GetIdx(), donor.GetIdx(), hb.GetIdx()))
    return torsions

def _only_h_moves_on_rotation(mol, torsion, angle_deg=30.0, tol=1e-4):
    """Check if rotating a torsion only moves the Hydrogen atom."""
    mol_copy = Chem.Mol(mol)
    conf = mol_copy.GetConformer()
    n = conf.GetNumAtoms()
    before = conf.GetPositions()
    
    rdMolTransforms.SetDihedralDeg(conf, *torsion, rdMolTransforms.GetDihedralDeg(conf, *torsion) + angle_deg)
    after = conf.GetPositions()
    
    diff = np.linalg.norm(before - after, axis=1)
    moved_indices = np.where(diff > tol)[0]
    
    return len(moved_indices) == 1 and moved_indices[0] == torsion[3]


def get_hydrophobic_groups(mol):
    """Fragments ligand and calculates logP for fragments."""
    # Fragment on BRICS bonds
    frag_mol = Chem.FragmentOnBRICSBonds(mol)
    fragments = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
    atom_mapping = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=True)
    
    fragment_data = {}
    for idx, (frag, atom_map) in enumerate(zip(fragments, atom_mapping)):
        logp = Crippen.MolLogP(frag)
        if logp > 0.0:
            fragment_data[idx] = (atom_map, logp)
    return fragment_data

def edit_dative_covelent(mol):
    editable_mol = Chem.RWMol(mol)

    for bond in editable_mol.GetBonds():
        if hasattr(Chem.BondType, "DATIVE") and bond.GetBondType() == Chem.BondType.DATIVE:
            bond.SetBondType(Chem.BondType.SINGLE)
        elif bond.HasProp("dative") and bond.GetProp("dative") == "True":
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.ClearProp("dative")

    editable_mol.UpdatePropertyCache()
    Chem.SanitizeMol(editable_mol)
    return editable_mol.GetMol()

def get_van_der_waals_radius(element_symbol):
    """
    Get the van der Waals radius of an element given its symbol using RDKit's PeriodicTable.

    Args:
    - element_symbol (str): The symbol of the element (e.g., 'C', 'O', 'H').

    Returns:
    - float: The van der Waals radius in Angstroms.
    - None: If the element symbol is not found or van der Waals radius is not available.
    """
    periodic_table = Chem.GetPeriodicTable()
    try:
        radius = periodic_table.GetRvdw(element_symbol)
        return radius
    except ValueError:
        print(f"Van der Waals radius for element symbol '{element_symbol}' is not available.")
        return 1.8



def write_xyz(points, filename="points.xyz"):
    """
    Writes a list of points as carbon atoms in an XYZ file format.

    Parameters:
        points (numpy.ndarray): A 2D numpy array of shape (n, 3) where each row is [x, y, z].
        filename (str): Name of the output file. Default is "points.xyz".
    """
    num_atoms = points.shape[0]  # Number of atoms (rows in the array)
    
    with open(filename, "w") as f:
        # Write the number of atoms as the first line (required in XYZ format)
        f.write(f"{num_atoms}\n")
        # Comment line (optional in XYZ format, can be left blank or used for metadata)
        f.write("Generated by write_xyz function\n")
        
        # Write each point as a carbon atom line in the format: "C x y z"
        for point in points:
            x, y, z = point
            f.write(f"Ne {x:.3f} {y:.3f} {z:.3f}\n")
    
def get_ligand_heavy_atom_indexes(mol):
    return np.array([atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != 'H'])

def compute_bond_distances(mol, heavy_atom_indices):
    """BFS to find shortest path between heavy atoms."""
    n_heavy = len(heavy_atom_indices)
    idx_map = {rd_idx: i for i, rd_idx in enumerate(heavy_atom_indices)}
    bond_distances = np.full((n_heavy, n_heavy), 999, dtype=np.int32)
    np.fill_diagonal(bond_distances, 0)

    adjacency = {i: [] for i in heavy_atom_indices}
    for bond in mol.GetBonds():
        b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if b in idx_map and e in idx_map:
            adjacency[b].append(e)
            adjacency[e].append(b)

    for start_node in heavy_atom_indices:
        start_matrix_idx = idx_map[start_node]
        visited = {start_node: 0}
        queue = deque([start_node])
        
        while queue:
            curr = queue.popleft()
            dist = visited[curr]
            for nbr in adjacency[curr]:
                if nbr not in visited:
                    visited[nbr] = dist + 1
                    queue.append(nbr)
                    bond_distances[start_matrix_idx, idx_map[nbr]] = dist + 1
                    
    return bond_distances

def per_atom_logp(mol):
    """Wraps RDKit Crippen contribs."""
    return [x[0] for x in rdMolDescriptors._CalcCrippenContribs(mol)]


def get_ligand_hydrogen_reference(mol):
    """Map heavy atoms to their bonded hydrogens."""
    molH = Chem.AddHs(mol, addCoords=True)
    refs = []
    # RDKit atom indexing changes after AddHs, this mapping needs care
    # Assuming the input `mol` already had Hs or we map by index if consistent.
    # The original code re-generated coords, implying simple index alignment relies on explicit H presence.
    for atom in molH.GetAtoms():
        if atom.GetSymbol() != 'H':
            hs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == 'H']
            refs.append(np.array(hs, dtype=np.int32))
    return refs


def find_best_ring_bond_to_break(mol, ring_info=None):
    """
    Finds the best bond to break in each non-aromatic ring of a molecule.

    The selection is based on a hierarchy of rules to preserve the molecule's
    chemical identity.

    Args:
        mol (rdkit.Mol): The input molecule.
        ring_info (rdkit.RingInfo, optional): Pre-computed ring information.
                                             If None, it will be calculated.

    Returns:
        list: A list of bond indices. Each index corresponds to the best
              breakable bond for a non-aromatic ring. Returns an empty
              list if no suitable bonds are found.
    """
    if ring_info is None:
        ring_info = mol.GetRingInfo()

    
    all_ring_bonds_idx = {bond_idx for ring in ring_info.BondRings() for bond_idx in ring}
    breakable_bonds_by_ring = []
    for ring_bonds, ring_atoms in zip(ring_info.BondRings(), ring_info.AtomRings()):
        #ignor rings with less tan 4 atoms.
        if len(ring_atoms) < 4:
            continue
    
        # We only want to break non-aromatic rings.
        is_aromatic = all(mol.GetBondWithIdx(idx).GetIsAromatic() for idx in ring_bonds)
        if is_aromatic:
            continue
        
        is_kekulized_aromatic = any(mol.GetBondWithIdx(idx).GetBondType() == Chem.BondType.DOUBLE for idx in ring_bonds)
        if is_kekulized_aromatic:
            continue 
        
        candidate_bonds = []
        for bond_idx in ring_bonds:
            bond = mol.GetBondWithIdx(bond_idx)

            # --- Rule 1: Must be a single bond ---
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue

            # --- Rule 2: Must not be in a fused system ---
            # The bond must belong to exactly one ring.
            if ring_info.NumBondRings(bond_idx) != 1:
                continue

          
            candidate_bonds.append(bond_idx)

        if not candidate_bonds:
            continue 

        # --- Rule 4: Prefer C-C bonds---
        cc_bonds = []
        for bond_idx in candidate_bonds:
            bond = mol.GetBondWithIdx(bond_idx)
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            if atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 6:
                cc_bonds.append(bond_idx)

        # If we found any C-C bonds that fit the criteria, use them.
        # Otherwise, fall back to the list of any single bonds.
        final_candidates = cc_bonds if cc_bonds else candidate_bonds
        
        if not final_candidates:
            continue
        
        # --- Final Selection: Choose deterministically ---
        # Pick the bond with the lowest index for reproducibility.
        best_bond_idx = min(final_candidates)
        breakable_bonds_by_ring.append(best_bond_idx)

    
    return sorted(list(set(breakable_bonds_by_ring)))



def remove_bonds_from_mol(mol, bonds_to_break):
    
    rw_mol = Chem.RWMol(mol)
    atoms_to_constrain = []
    constraint_distances = []
    conformer = mol.GetConformer()
    for bond_idx in bonds_to_break:
        bond = rw_mol.GetBondWithIdx(bond_idx)
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        #get distances!!
        pos1 = conformer.GetAtomPosition(idx1)
        pos2 = conformer.GetAtomPosition(idx2)
        distance = round(pos1.Distance(pos2),3)
        constraint_distances.append(distance)
        
        atoms_to_constrain.append((idx1, idx2))
        rw_mol.RemoveBond(idx1, idx2)
    
    sanitized_fragmented_mol = rw_mol.GetMol()
    #checks to enure the mol was not split into multiple pieces 
    fragmented_smiles = Chem.MolToSmiles(sanitized_fragmented_mol)
    fragments = Chem.GetMolFrags(sanitized_fragmented_mol, asMols=True)
    num_fragments = len(fragments)
    if num_fragments == 1:
        return sanitized_fragmented_mol, atoms_to_constrain, constraint_distances
    else:
        return None, None, None
    

def get_imporper_torsion_restraints(mol, atoms_to_constrain):
    improper_torsion_constraints = []
    for idx1, idx2 in atoms_to_constrain:
        # Check atom1
        improper1 = get_improper_torsion_for_atom(mol, idx1, idx2)
        if improper1:
            improper_torsion_constraints.append(improper1)
        # Check atom2
        improper2 = get_improper_torsion_for_atom(mol, idx2, idx1)
        if improper2:
            improper_torsion_constraints.append(improper2)
    
    return improper_torsion_constraints

def get_improper_torsion_for_atom(mol, atom_idx, broken_bond_partner_idx):
    """
    For a given stereocenter, finds the atoms and angle for an improper torsion constraint.
    The central atom of the torsion will be the stereocenter itself.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
        return None

    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
    
    
    if len(neighbors) < 3:
        return None # Not a standard tetrahedral center

    # The improper torsion is defined by the central atom and 3 of its neighbors.
    # We will use the broken bond partner as one of the plane-defining atoms.
    other_neighbors = [n_idx for n_idx in neighbors if n_idx != broken_bond_partner_idx]
    
    if len(other_neighbors) < 2:
        return None # Cannot define the plane

    # Define the 4 atoms for the improper torsion:
    # Atom 1, Atom 2 (define a plane with the central atom)
    # Atom 3 (the central atom itself)
    # Atom 4 (the out-of-plane atom, which is the broken bond partner)
    p1 = other_neighbors[0]
    p2 = other_neighbors[1]
    p3_center = atom_idx
    p4_outofplane = broken_bond_partner_idx
    angle = rdMolTransforms.GetDihedralDeg(mol.GetConformer(), p1, p2, p3_center, p4_outofplane)
    
    return (p1, p2, p3_center, p4_outofplane, angle)

