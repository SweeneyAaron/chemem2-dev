# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

import math
import tempfile
from typing import List, Tuple, Dict, Any, Optional
from ChemEM.data.data import INTRA_RESIDUE_BOND_DATA, INTER_RESIDUE_BOND_DATA
import numpy as np
from openmm import unit, NonbondedForce
from openmm.app import PDBFile
from rdkit import Chem 
from rdkit.Chem.rdchem import MolSanitizeException
from scipy.spatial import cKDTree
from ChemEM.data.data import (
    HBOND_DONOR_ATOM_IDXS, HBOND_ACCEPTOR_ATOM_IDXS, 
    HALOGEN_ACCEPTOR_ATOM_IDXS, PROTEIN_RINGS, 
    is_protein_atom_donor, is_protein_atom_acceptor, AtomType, RingType
)




# -------------------------
# SSE (DSSP) grouping helper
# -------------------------
HELIX_CODES = set("HGI")   # alpha/3_10/pi
STRAND_CODES = set("EB")   # sheet/bridge

def sse_groups_from_parmed(
    pmd_struct,
    min_len: int = 5,
    scheme: str = "eight",    # "eight" or "simplified"
    group_types: Tuple[str, ...] = ("helix", "strand"),
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Returns:
      groups: list of lists of ParmEd atom indices per SSE segment
      ss_codes: (n_res,) array of single-letter DSSP codes for all residues
    """
    try:
        import mdtraj as md
    except Exception as e:
        raise ImportError(
            "mdtraj is required for sse_groups_from_parmed(). "
            "Install mdtraj or disable SSE grouping."
        ) from e

    # ParmEd -> MDTraj trajectory (single frame)
    top_omm = pmd_struct.topology
    top_md = md.Topology.from_openmm(top_omm)
    xyz_nm = (np.asarray(pmd_struct.positions._value, dtype=float) / 10.0)[None, :, :]  # (1,N,3)
    traj = md.Trajectory(xyz_nm, top_md)

    simplified = (scheme == "simplified")
    ss = md.compute_dssp(traj, simplified=simplified)[0]  # (n_res,)
    ss_codes = np.array(ss, dtype="U1")

    pmd_res_list = list(pmd_struct.residues)
    if len(pmd_res_list) != len(ss_codes):
        raise ValueError("Residue count mismatch between ParmEd and MDTraj.")

    def segment_indices(want: set) -> List[Tuple[int,int]]:
        segs = []
        start = None
        for i, code in enumerate(ss_codes):
            in_seg = code in want
            if in_seg and start is None:
                start = i
            if (not in_seg or i == len(ss_codes) - 1) and start is not None:
                end = i if not in_seg else i + 1
                if (end - start) >= min_len:
                    segs.append((start, end))
                start = None
        return segs

    groups: List[List[int]] = []
    want_sets = []
    if "helix" in group_types:
        want_sets.append(HELIX_CODES if scheme == "eight" else set("H"))
    if "strand" in group_types:
        want_sets.append(STRAND_CODES if scheme == "eight" else set("E"))

    for want in want_sets:
        for (a, b) in segment_indices(want):
            atom_ids = []
            for r in pmd_res_list[a:b]:
                atom_ids.extend([a_.idx for a_ in r.atoms])
            if atom_ids:
                groups.append(atom_ids)

    return groups, ss_codes





# Re-exported from ChemEM.tools.biomolecule rather than duplicated. This module
# used to carry its own byte-identical copy of both functions; keeping two
# implementations of the mol the ECHO precompute is built from is how they drift
# apart, and the hydrogen-reference contract below depends on the two staying in
# step. Nothing in-tree calls this copy -- only get_role_int and
# find_atoms_outside_ligand are imported from here -- so this is purely to stop a
# future caller picking up a stale variant.
from ChemEM.tools.biomolecule import (  # noqa: E402  (re-export)
    write_residues_to_pdb,
    get_protein_hydrogen_reference,
)

def select_atoms(structure, indices=None, residues=None, include="heavy", exclude_indices=None):
    """
    General purpose atom selector for an parmed structure obejct
    """
    if exclude_indices is None: exclude_indices = set()
    else: exclude_indices = set(exclude_indices)

    selected = []
    
    
    def process_atom_list(atom_iterable):
        for atom in atom_iterable:
            if atom.idx in exclude_indices: continue
            if include == "heavy" and atom.element == 1: continue
            if include == "backbone" and atom.name not in ("N", "CA", "C"): continue
            
        
            selected.append(atom.idx)

    if indices:
        # If specific indices provided
        atoms = [structure.atoms[i] for i in indices]
        process_atom_list(atoms)
    elif residues:
        for r in residues:
            process_atom_list(r.atoms)
            
    return sorted(list(set(selected)))



def find_atoms_outside_ligand(complex_structure, ligand_indices, cutoff_A=9.0, exclude_backbone=False):
    """
    Identifies heavy atoms outside a radius of the ligand.
    
    Parameters
    ----------
    complex_structure : Structure
        The molecular structure object.
    ligand_indices : list or set
        Indices of the ligand atoms.
    cutoff_A : float
        Distance cutoff in Angstroms.
    exclude_backbone : bool
        If True, protein backbone atoms (N, CA, C, O, OXT) are not returned 
        even if they fall outside the cutoff radius.
    """
    atoms = list(complex_structure.atoms)
    pos_A = np.array([[a.xx, a.xy, a.xz] for a in atoms])
    
    lig_indices_arr = np.array(list(ligand_indices))
    if lig_indices_arr.size == 0: 
        return []
    
    lig_pos = pos_A[lig_indices_arr]
    tree = cKDTree(lig_pos)

    backbone_names = {'N', 'CA', 'C', 'O', 'OXT'}
    
    prot_atoms_idx = []
    

    for a in atoms:
        
        if a.idx in ligand_indices:
            continue
            
        if a.element == 1:
            continue
            
        if exclude_backbone and a.name in backbone_names:
            continue
            
        prot_atoms_idx.append(a.idx)
    
    
    if not prot_atoms_idx:
        return []
        
    prot_pos = pos_A[prot_atoms_idx]
    
    
    dists, _ = tree.query(prot_pos, k=1)
    
   
    mask = dists > float(cutoff_A)
    far_atom_indices = np.array(prot_atoms_idx)[mask]
    
    return far_atom_indices.tolist()

def create_structure_subset(full_structure, residue_list):
    """Creates a new parmed structure from a residue subset."""
    all_atom_indices = set()
    for res in residue_list:
        all_atom_indices.update(a.idx for a in res.atoms)
    
    selection = f"@{','.join(str(i + 1) for i in sorted(all_atom_indices))}"
    return full_structure[selection]

#-------------------------------
#precomputed data tools
#------------------------------
def get_role_int(res_name, atom_name):
    """Maps PDB atom to role integer (0=None, 1=Donor, 2=Acceptor, 3=Both)."""
    role = 0
    if is_protein_atom_donor(res_name, atom_name):
        role += 1
    if is_protein_atom_acceptor(res_name, atom_name):
        role += 2
    return role

def residue_atom_charge(resname, atom_name):
    """Returns standard AMBER-like charges for key residues."""
    charges = {
        "ASP": {"OD1": -0.5, "OD2": -0.5},
        "GLU": {"OE1": -0.5, "OE2": -0.5},
        "ARG": {"NH1": 0.3333, "NH2": 0.3333, "NE": 0.3333},
        "LYS": {"NZ": 1.0}
    }
    return charges.get(resname, {}).get(atom_name, 0.0)

def get_hbond_direction(atom, role_int):
    """Calculates unit vector for H-bond direction based on geometry."""
    if role_int == 0: return np.zeros(3)
    
    pos = np.array([atom.xx, atom.xy, atom.xz])
    neighbors = atom.bond_partners
    hydrogens = [n for n in neighbors if n.element_name == "H"]
    heavy = [n for n in neighbors if n.element_name != "H"]
    
    vec = np.zeros(3)
    
    # Case 1: Donor (point towards H)
    if (role_int & 1) and hydrogens:
        for h in hydrogens:
            h_pos = np.array([h.xx, h.xy, h.xz])
            vec += (h_pos - pos)
        vec /= len(hydrogens)
        
    # Case 2: Acceptor (point away from heavy neighbors)
    elif (role_int & 2) and heavy:
        for n in heavy:
            n_pos = np.array([n.xx, n.xy, n.xz])
            vec -= (n_pos - pos) # Negative vector
            
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else np.zeros(3)

def compute_protein_ring_types(residues, protein_positions, tol=1e-8):
    """Identifies ring centers and types in protein residues."""
    rings = []
    coords = []
    indices = []
    
    for res in residues:
        if res.name not in PROTEIN_RINGS: continue
        
        # Get atom definitions for this ring type
        ring_defs = PROTEIN_RINGS[res.name] # Dictionary or list of tuples
        types = RingType.from_residue_name(res.name)
        
        # Iterate over defined rings in residue (e.g. TRP has 2)
        # Note: Adapting original logic which zipped names
        for (r_name, atom_names), r_type in zip(ring_defs, types):
            ring_pos = []
            heavy_idxs = []
            
            for atom in res.atoms:
                if atom.name in atom_names:
                    pos = np.array([atom.xx, atom.xy, atom.xz])
                    ring_pos.append(pos)
                    # Find matching index in protein_positions array
                    # (Optimized: assumes aligned order or use KDTree in prod)
                    # For strict compatibility with original:
                    dist = np.linalg.norm(protein_positions - pos, axis=1)
                    match = np.where(dist < tol)[0]
                    if match.size > 0: heavy_idxs.append(match[0])

            if ring_pos:
                rings.append(r_type)
                coords.append(np.array(ring_pos))
                indices.append(np.array(heavy_idxs, dtype=np.int32))
                
    return rings, coords, indices


def compute_charges(protein_structure, protein_system, collapse_hydrogens=True):
    """
    Extract positions (Å) and partial charges (e) from an OpenMM structure/system.

    If collapse_hydrogens is True, return only heavy atoms, with:
        q_eff(heavy) = q_heavy + sum(q_H attached)

    Parameters
    ----------
    protein_structure : object with .positions and .topology
        Typically an OpenMM Modeller / Simulation.context.getState(...).
    protein_system : openmm.System
        System containing a NonbondedForce with particle charges.
    collapse_hydrogens : bool
        If True, collapse H charges onto bonded heavy atoms and return
        heavy-atom-only positions/charges.

    Returns
    -------
    positions : (N,3) ndarray
        Positions in Å (heavy atoms only if collapse_hydrogens=True).
    charges : (N,) ndarray
        Charges in elementary charge units (e), collapsed if requested.
    """
    # Positions in Å
    positions = protein_structure.positions.value_in_unit(unit.angstrom)
    positions = np.array(positions, dtype=np.float64)

    # Find NonbondedForce
    nonbonded_force = None
    for force in protein_system.getForces():
        if isinstance(force, NonbondedForce):
            nonbonded_force = force
            break
    if nonbonded_force is None:
        raise ValueError("No NonbondedForce found in the system.")

    # Raw charges for all atoms
    n_atoms = len(positions)
    charges = np.zeros(n_atoms, dtype=np.float64)
    for i in range(n_atoms):
        q, sig, eps = nonbonded_force.getParticleParameters(i)
        charges[i] = q.value_in_unit(unit.elementary_charge)

    if not collapse_hydrogens:
        # Return all atoms as-is
        return positions, charges

    # -------- Collapse hydrogen charges onto heavy atoms --------
    topo = protein_structure.topology

    # Get element symbols in topology order (must match System atom order)
    elements = []
    for atom in topo.atoms():
        # atom.element may be None for some topologies, so fall back to atom.name
        if atom.element is not None:
            elements.append(atom.element)
        else:
            # crude fallback, last char of atom.name (" H1 ", " CA " etc.)
            elements.append(atom.name.strip()[0])

    elements = np.array(elements, dtype=object)
    if len(elements) != n_atoms:
        raise ValueError("Topology atom count and position/charge count differ.")

    # Build neighbor list from bonds
    neighbors = [[] for _ in range(n_atoms)]
    for bond in topo.bonds():
        i = bond[0].index
        j = bond[1].index
        neighbors[i].append(j)
        neighbors[j].append(i)

    # Identify heavy atoms
    heavy_indices = [i for i, e in enumerate(elements) if e.upper() != 1]
    heavy_indices = np.array(heavy_indices, dtype=int)
    n_heavy = heavy_indices.size

    # Map original atom index -> heavy index or -1
    atom_to_heavy = np.full(n_atoms, -1, dtype=int)
    for heavy_idx, atom_idx in enumerate(heavy_indices):
        atom_to_heavy[atom_idx] = heavy_idx

    # Initialize heavy positions & charges from heavy atoms
    heavy_positions = positions[heavy_indices].copy()
    heavy_charges   = charges[heavy_indices].copy()

    # Add hydrogen charges to their bonded heavy partner
    for i_atom in range(n_atoms):
        if elements[i_atom].upper() != "H":
            continue

        # Find a heavy neighbor to receive the H charge
        heavy_partner = None
        for j in neighbors[i_atom]:
            if elements[j].upper() != "H":
                heavy_partner = j
                break

        if heavy_partner is None:
            # Isolated H (shouldn't really happen); skip
            continue

        j_heavy = atom_to_heavy[heavy_partner]
        if j_heavy >= 0:
            heavy_charges[j_heavy] += charges[i_atom]

    return heavy_positions, heavy_charges
