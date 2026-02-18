# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from openmm import unit 
from openmm.app import  Topology
import math 
import numpy as np 


#don't want these to be here!
_WATER = {"HOH", "WAT", "H2O", "SOL", "TIP", "TIP3", "T3P"}
_RNA_RESNAMES = {"A", "C", "G", "U", "I"}
_DNA_RESNAMES = {"DA", "DC", "DG", "DT", "DU", "DI"}
_NA_CANON = _RNA_RESNAMES | _DNA_RESNAMES
_NA_SUGAR_HINTS = {
    "C1'", "C2'", "C3'", "C4'", "C5'",
    "O2'", "O3'", "O4'", "O5'",
}
_NA_PHOS_HINTS = {"P", "OP1", "OP2", "OP3", "O1P", "O2P"}
_AA3 = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "MSE","SEC","PYL"
}


def _norm_atom_name(name: str) -> str:
    # normalize star/prime variants (e.g. O3* vs O3')
    return name.replace("*", "'").strip()

def _has_nucleic_backbone(res) -> bool:
    names = {_norm_atom_name(a.name) for a in res.atoms()}
    sugar_count = sum(1 for x in _NA_SUGAR_HINTS if x in names)
    has_sugar = sugar_count >= 4
    has_p = "P" in names
    has_po = (("OP1" in names or "O1P" in names) and ("OP2" in names or "O2P" in names)) or ("OP3" in names)
    return has_sugar and has_p and has_po

def is_nucleic_acid_res(res) -> bool:
    if res.name in _NA_CANON:
        return True
    return _has_nucleic_backbone(res)

def _has_protein_backbone(res) -> bool:
    """
    Heuristic: protein residues have the backbone atoms N, CA, C.
    Works for standard AAs and most modified residues that keep a protein backbone.
    """
    names = {a.name for a in res.atoms()}
    return {"N", "CA", "C"}.issubset(names)

def is_protein_res(res) -> bool:
    """
    Protein residue classifier.
    - Fast path: known 3-letter amino acid codes
    - Fallback: backbone atom heuristic (N, CA, C present)
    """
    if res.name in _AA3:
        return True
    return _has_protein_backbone(res)


def residues_sequential(res1, res2):
    try:
        return int(res2.id) == int(res1.id) + 1
    except Exception:
        return False
def as_quantity_pos(p):
    # Ensure p is a Quantity(Vec3, length)
    return p if isinstance(p, unit.Quantity) else (p * unit.nanometer)

def get_atom_position(original_positions, res, atom_name):
    want = atom_name
    alts = {want}
    if "*" in want:
        alts.add(want.replace("*", "'"))
    if "'" in want:
        alts.add(want.replace("'", "*"))

    for atom in res.atoms():
        if atom.name in alts:
            p = original_positions[atom.index]
            return as_quantity_pos(p)
    return None

def distance(p1_xyz, p2_xyz):
    # p1_xyz/p2_xyz are float arrays in Å
    dx = p1_xyz[0] - p2_xyz[0]
    dy = p1_xyz[1] - p2_xyz[1]
    dz = p1_xyz[2] - p2_xyz[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)



def rebuild_standard_bonds(top: Topology) -> None:
    # Standard bonds (peptide + NA O3'-P) and disulfides if available
    top.createStandardBonds()
    try:
        top.createDisulfideBonds()
    except Exception:
        pass
    
def split_chains_on_breaks(fixer, 
                           acceptable_c_n=1.6,
                           acceptable_o3_p=2.2,
                           water_ids = None):
    
    WATER_IDS = water_ids if water_ids is not None else _WATER
    new_topology = Topology()
    new_positions_vals = []  # store Vec3 values in nanometers
    original_topology = fixer.topology
    original_positions = fixer.positions  #nm
    
    for chain in original_topology.chains():
        orig_cid = chain.id or "?"
        seg_idx = 1
        new_chain = new_topology.addChain(id=orig_cid)
        prev_residue = None
        
        for res in chain.residues():
            if prev_residue and prev_residue.name not in WATER_IDS:
                
                do_split = False
                
                prev_is_na = is_nucleic_acid_res(prev_residue)
                curr_is_na = is_nucleic_acid_res(res)
                
                prev_is_prot = is_protein_res(prev_residue)
                curr_is_prot = is_protein_res(res)
                if prev_is_na and curr_is_na:
                    do_split = get_split_na(res, prev_residue,original_positions, acceptable_o3_p)
                elif prev_is_prot and curr_is_prot:
                    
                    do_split = get_split_prot(res, prev_residue, original_positions ,acceptable_c_n)
                else:
                    do_split = True
                
                if do_split:
                    seg_idx += 1
                    new_chain = new_topology.addChain(id=f"{orig_cid}#{seg_idx}")
            new_residue = new_topology.addResidue(res.name, new_chain, id=res.id)
            for atom in res.atoms():
                new_topology.addAtom(atom.name, atom.element, new_residue)

                p = original_positions[atom.index]
                p_q = p if isinstance(p, unit.Quantity) else (p * unit.nanometer)
                # store unitless Vec3 values (in nanometers) so we can wrap once at the end
                new_positions_vals.append(p_q.value_in_unit(unit.nanometer))

            prev_residue = res
    new_positions = unit.Quantity(new_positions_vals, unit.nanometer)
    return new_topology, new_positions

            
def get_split_na(res, prev_residue,positions, acceptable_o3_p):     
    pos_prev_o3_q = get_atom_position(positions, prev_residue, "O3'")
    pos_curr_p_q = get_atom_position(positions, res, "P")

    if pos_prev_o3_q is not None and pos_curr_p_q is not None:
        pos_prev_o3 = np.array(pos_prev_o3_q.value_in_unit(unit.angstrom))
        pos_curr_p = np.array(pos_curr_p_q.value_in_unit(unit.angstrom))
        if distance(pos_prev_o3, pos_curr_p) > acceptable_o3_p:
            return True 
    return False

def get_split_prot(res, prev_residue,positions, acceptable_c_n):
    pos_prev_C_q = get_atom_position(positions, prev_residue, "C")
    pos_prev_N_q = get_atom_position(positions, prev_residue, "N")
    pos_curr_N_q = get_atom_position(positions, res, "N")
    pos_curr_C_q = get_atom_position(positions, res, "C")

    if (pos_prev_C_q is not None and pos_prev_N_q is not None and
            pos_curr_N_q is not None and pos_curr_C_q is not None):

        pos_prev_C = np.array(pos_prev_C_q.value_in_unit(unit.angstrom))
        pos_prev_N = np.array(pos_prev_N_q.value_in_unit(unit.angstrom))
        pos_curr_N = np.array(pos_curr_N_q.value_in_unit(unit.angstrom))
        pos_curr_C = np.array(pos_curr_C_q.value_in_unit(unit.angstrom))
    
        d_curr_n_prev_c = distance(pos_curr_N, pos_prev_C)
        d_curr_c_prev_n = distance(pos_curr_C, pos_prev_N)
    
        if d_curr_n_prev_c > acceptable_c_n and d_curr_c_prev_n > acceptable_c_n:
            return True 
    return False

def ensure_water_geometry_types(pmd_struct, water_model: str = "tip3p") -> int:
    """
    Assign BondType/AngleType to water O–H bonds and H–O–H angles so ParmEd can
    build constraints (HBonds/rigidWater) without crashing.

    Returns
    -------
    int : number of water residues patched
    """
    from parmed import unit as u
    from parmed.topologyobjects import BondType, AngleType

    wm = water_model.lower()

    # Geometry (Å, degrees). Force constants are placeholders; irrelevant if constrained.
    if "spce" in wm or "spc/e" in wm or "spc" in wm:
        req    = 1.0000 * u.angstrom
        theteq = 109.47 * u.degree
    else:
        # TIP3P defaults
        req    = 0.9572 * u.angstrom
        theteq = 104.52 * u.degree

    bt = BondType(k=450.0, req=req)
    at = AngleType(k=55.0,  theteq=theteq)

    # Helper: fast bond lookup by (min_idx, max_idx)
    bond_map = {}
    for b in pmd_struct.bonds:
        i = min(b.atom1.idx, b.atom2.idx)
        j = max(b.atom1.idx, b.atom2.idx)
        bond_map[(i, j)] = b

    def get_bond(a, b):
        i = min(a.idx, b.idx)
        j = max(a.idx, b.idx)
        return bond_map.get((i, j), None)

    def get_hoh_angle(H1, O, H2):
        # center must be O; scan angles list
        for ang in pmd_struct.angles:
            if ang.atom2.idx != O.idx:
                continue
            a1, a3 = ang.atom1.idx, ang.atom3.idx
            if {a1, a3} == {H1.idx, H2.idx}:
                return ang
        return None

    water_names = ("HOH","WAT","H2O","SOL","TIP3","T3P","TIP","SPCE","SPC")
    atom_O_names = ("O","OW")
    atom_H_names = ("H","H1","H2","HW1","HW2")

    patched = 0
    for res in pmd_struct.residues:
        if res.name not in water_names:
            continue

        atoms = list(res.atoms)
        O = next((a for a in atoms if a.name in atom_O_names), None)
        Hs = [a for a in atoms if a.name in atom_H_names]

        if O is None or len(Hs) < 2:
            continue
        H1, H2 = Hs[0], Hs[1]

        # Patch O–H bonds
        b1 = get_bond(O, H1)
        b2 = get_bond(O, H2)
        if b1 is not None and b1.type is None:
            b1.type = bt
        if b2 is not None and b2.type is None:
            b2.type = bt

        # Patch H–O–H angle (needed if you ever set rigidWater=True)
        ang = get_hoh_angle(H1, O, H2)
        if ang is not None and ang.type is None:
            ang.type = at

        patched += 1

    return patched

            