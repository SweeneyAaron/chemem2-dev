#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 10:54:01 2025

@author: aaron.sweeney
"""

from ChemEM.messages import Messages
from ChemEM.data.data import (
    # — core type classes —
    AtomType,
    RingType,

    # — lookup tables —
    TABLE_A,
    TABLE_B,
    TABLE_C,
    kernel_dx,
    kernel_dy,
    kernel_dz,

    # — hydrogen-bond constants —
    HBOND_DONOR_ATOM_IDXS,
    HBOND_ACCEPTOR_ATOM_IDXS,
    HBOND_POLYA,
    HBOND_POLYB,
    HBOND_POLYC,

    # — halogen-bond constants —
    HALOGEN_DONOR_ATOM_IDXS,
    HALOGEN_ACCEPTOR_ATOM_IDXS,

    # — protein ring definitions —
    PROTEIN_RINGS,

    # — physico-chemical property helpers —
    XlogP3,

    # — protein donor/acceptor helpers —
    
    is_protein_atom_donor,
    is_protein_atom_acceptor,
)
from rdkit import Chem
from rdkit.Chem import rdmolops, Crippen, Descriptors, AllChem
import numpy as np
from collections import deque
import os 
import copy
import json 
from scipy.spatial.transform import Rotation 
from rdkit.Geometry import Point3D
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import TorsionFingerprints as tfp
#from ChemEM.scoring_functions import PreComputedData
from tqdm import tqdm
from openmm import MonteCarloBarostat, XmlSerializer, app, unit, CustomCompoundBondForce, Continuous3DFunction, vec3, Vec3, CustomNonbondedForce
from openmm.app import HBonds, NoCutoff, PDBFile, Modeller, Topology, PME, StateDataReporter, PDBReporter
from openmm.unit.quantity import Quantity
from openmm.unit import norm
from openmm import LangevinIntegrator, Platform
from openmm.app import ForceField as OpenMMForceField
from openmm import NonbondedForce
from pdbfixer import PDBFixer
import time
import math
from functools import partial
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import pickle
import pkg_resources
import scipy.ndimage as ndimage
from ChemEM.parsers.EMMap import EMMap 
from ChemEM.tools.aromatic_score import AromaticScore
from ChemEM.tools.halogen_bond_score import HalogenBondScore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path  
from kneed import KneeLocator
from scipy.spatial import cKDTree
#cpp test package imports 
from ChemEM import grid_maps
from rdkit.Chem import rdMolDescriptors
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy import signal


def _attach_attributes(dst, src, *, prefix: str = "") -> None:
    """
    Copy every attribute in *src* to *dst* (deepcopy) under the name
    '<prefix><attr>'.  Raises if that name already exists on *dst*.
    """
    for name, value in vars(src).items():
        new_name = f"{prefix}{name}"
        
        setattr(dst, new_name, copy.deepcopy(value))

def per_atom_logp(mol):
    """
    Return per-atom Crippen logP contributions as a list (one value per atom).

    mol : RDKit Mol
    """
    # contribs is a list of (logP_i, MR_i) tuples
    contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    return [c[0] for c in contribs]


def select_ring_flip_torsions(mol, torsion_quads):
    mol = Chem.RemoveHs(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol,breakTies=False))  # symmetry ranks
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    flip_torsions = []
    for (i, j, k, l) in torsion_quads:
        # Sanity: indices must be valid for this mol
        if max(i, j, k, l) >= mol.GetNumAtoms():
            continue
        
        if k in ring_atoms and l in ring_atoms and j not in ring_atoms and i not in ring_atoms:
            a,b,c,d = l,k,j,i
        elif i in ring_atoms and j in ring_atoms and k not in ring_atoms and l not in ring_atoms:
            a,b,c,d = i,j,k,l 
        else:
            continue 
        
        
        neis = mol.GetAtomWithIdx(b).GetNeighbors()
        neis = [i.GetIdx() for i in neis if i.GetIdx() not in [a,c]]
        
        if not neis:
            continue
        
        rank_a = ranks[a]
        
        if any(ranks[n] == rank_a for n in neis):
            continue
        
        
        flip_torsions.append((a,b,c,d))
    
    
    flip_torsions = list(set(flip_torsions))
    
    return flip_torsions
        
        
        


class PreCompDataLigand:
    '''
    class for holding ligand docking paramters to pass to c++ docking score
    '''
    
    
    def __init__(self, ligand, platform, flexible_rings = False):
        # ------------------------------------------------------------------ #
        # basic per-atom arrays                                              #
        # ------------------------------------------------------------------ #
        
        self.atom_masses = []
        for idx in range(ligand.mol.GetNumHeavyAtoms()):
            atom = ligand.mol.GetAtomWithIdx(idx)
            self.atom_masses.append(atom.GetMass())
        
        self.ligand_heavy_atom_indexes =  get_ligand_heavy_atom_indexes(ligand.mol)
        self.ligand_heavy_end_index = np.max(self.ligand_heavy_atom_indexes)
        self.ligand_atom_types = np.array([i.idx for i in ligand.atom_types], dtype=np.int32)
        self.ligand_bond_distances = compute_bond_distances(ligand.mol,
                                                            self.ligand_heavy_atom_indexes)
        
        self.exc_atoms = [1 for i in self.ligand_heavy_atom_indexes]
        new_torsions = self.set_flexible_rings(ligand, flexible_rings)
        self.torsion_lists = get_torsion_lists(ligand.mol)
        self.end_torsions = len(self.torsion_lists)
        
        self.ligand_torsion_profile = export_torsion_profile(ligand, self.torsion_lists, platform)
        self.ligand_torsion_idxs = [i[1] for i in self.ligand_torsion_profile]
        self.ligand_torsion_scores = [i[2] for i in self.ligand_torsion_profile]
        new_torsions = [list(i) for i in new_torsions if list(i) not in self.ligand_torsion_idxs ]
        self.ligand_torsion_idxs += new_torsions
        self.n_torsions = len(self.ligand_torsion_idxs)
        
        
        # ------------------------------------------------------------------ #
        # chemistry fingerprints                                             #
        # ------------------------------------------------------------------ #
        
        self.ligand_hydrophobic = get_hydrophobic_groups(ligand.mol)
        self.ligand_hydrophobic_cpp = list(self.ligand_hydrophobic.values())
        self.ligand_radii= np.array([get_vdw_radius(i.GetSymbol()) for i in ligand.mol.GetAtoms()])
        self.ligand_hydrogen_idx = get_ligand_hydrogen_reference(ligand.mol)
        self.ligand_ring_types = ligand.ring_types 
        self.ligand_ring_type_ints = [i.idx for i in self.ligand_ring_types]
        self.ligand_ring_indices = ligand.ring_indices
        self.halogen_bond_donor_indices, self.halogen_bond_donor_root_indices = compute_halogen_bond_data(ligand.mol, self.ligand_atom_types)
        self.MW = Descriptors.MolWt(ligand.mol)
        _, self.ligand_charges = compute_charges(ligand.complex_structure, ligand.complex_system)
        
        self.ligand_charges = list(self.ligand_charges)
        self.n_ligand_atoms = len(self.ligand_atom_types)
        
        self.per_atom_logp = np.array(per_atom_logp(ligand.mol))
        
        self.ligand_hbond_atom = [1 if i in HBOND_DONOR_ATOM_IDXS + HBOND_ACCEPTOR_ATOM_IDXS else 0 for i in self.ligand_atom_types]
        
        # ------------------------------------------------------------------ #
        # intra-ligand pairwise score tables                                 #
        # ------------------------------------------------------------------ #
        
        #shape (n_ligand_atoms, n_ligand_atoms), type = float
        self.LIGAND_INTRA_A_VALUES =  np.zeros((self.n_ligand_atoms, self.n_ligand_atoms), dtype=np.float64)
        #shape (n_ligand_atoms, n_ligand_atoms), type = float
        self.LIGAND_INTRA_B_VALUES =  np.zeros((self.n_ligand_atoms, self.n_ligand_atoms), dtype=np.float64)
        #shape (n_ligand_atoms, n_ligand_atoms), type = float
        self.LIGAND_INTRA_C_VALUES =  np.zeros((self.n_ligand_atoms, self.n_ligand_atoms), dtype=np.float64)
        
        for i, l_idx_1 in enumerate(self.ligand_atom_types):
            for j, l_idx_2 in enumerate(self.ligand_atom_types):
                # Use p_idx and l_idx as row/col in TABLE_A, etc.
                self.LIGAND_INTRA_A_VALUES[i, j] = TABLE_A[l_idx_1, l_idx_2]
                self.LIGAND_INTRA_B_VALUES[i, j] = TABLE_B[l_idx_1, l_idx_2]
                self.LIGAND_INTRA_C_VALUES[i, j] = TABLE_C[l_idx_1, l_idx_2]
        # ------------------------------------------------------------------ #
        # ring flips                               #
        # ------------------------------------------------------------------ #
        
        torsions = get_torsion_lists_dup(ligand.mol)
        self.ring_flip_torsions = select_ring_flip_torsions(ligand.mol, torsions)
        
        self.ligand_formal_charge = np.asarray(ligand.ligand_charge)
        
        
        
    def __radd__(self, other):
        if isinstance(other, PreCompDataProtein):
            return other + self     # delegate to protein.__add__
            
        return NotImplemented
        
    def set_flexible_rings(self, ligand, flexible_rings):
        """
        If *flexible_rings* is on, identify the best ring bonds to break,
        fragment the ligand, and populate constraint-related attributes.
    
        Returns
        -------
        new_torsions : list[tuple[int, int, int, int]]
            Torsion definitions that survive in the fragmented ligand.
            Empty list when no suitable fragmentation is possible.
        """
        if not flexible_rings:
            self._clear_fragment_state()
            return []
        
        bonds_to_break = find_best_ring_bond_to_break(ligand.mol)
        
        if not bonds_to_break:
            self._clear_fragment_state()
            return []
        
        frag_mol, atoms_to_constrain, dist_constraints = remove_bonds_from_mol(
                                                ligand.mol, bonds_to_break
                                                    )
        if frag_mol is None:
            self._clear_fragment_state()
            return []
        
        candidate_torsions = get_torsion_lists(frag_mol)
        new_torsions = []
        for i in candidate_torsions:
            if all(t < frag_mol.GetNumHeavyAtoms() for t in i):
                new_torsions.append(i)
        
        self.break_bonds = frag_mol
        self.constrain_atoms = atoms_to_constrain
        self.constrain_atoms_dist = dist_constraints
        self.improper_torsion_constraints = get_imporper_torsion_restraints(
            ligand.mol, atoms_to_constrain
            )
        
        return new_torsions
        
    def _clear_fragment_state(self) :
        """Zero-out all fragmentation/constraint attributes on *obj*."""
        self.break_bonds = False
        self.constrain_atoms = []
        self.constrain_atoms_dist = []
        self.improper_torsion_constraints = []


protein_charged_centers = {
    "LYS": {"NZ": +1},
    "ARG" :{"NH1": +1},  # or NH2, pick one and be consistent
    "ASP": {"OD1": -1},
    "GLU": {"OE1": -1},
    # ligand charge centers etc…
}
def residue_atom_charge(resname, atom_name):
    if resname == "ASP" and atom_name in ("OD1", "OD2"):
        return -0.5
    if resname == "GLU" and atom_name in ("OE1", "OE2"):
        return -0.5
    if resname == "ARG" and atom_name in ("NH1", "NH2", "NE"):
        return +1.0/3.0
    if resname == "LYS" and atom_name == "NZ":
        return +1.0
    return 0.0

import numpy as np

def _atom_pos(atom):
    # ParmEd stores coordinates in xx, xy, xz (Å)
    return np.array([atom.xx, atom.xy, atom.xz], dtype=float)

def get_hbond_direction_for_atom(atom, role_int):
    """
    Return a unit vector (np.array shape (3,)) giving the H-bond
    direction for this heavy atom, or [0,0,0] if no good direction.
    
    role_int: 0 = none, 1 = donor, 2 = acceptor, 3 = both
    """
    pos = _atom_pos(atom)

    # Split neighbors into attached H's and heavy-atom neighbors
    
    
    attached_H = [a for a in atom.bond_partners if a.element_name == "H"]
    heavy_neigh = [a for a in atom.bond_partners if a.element_name != "H"]

    v = np.zeros(3, dtype=float)

    # Donor: use average vector from heavy atom to its hydrogens
    if role_int in (1, 3) and attached_H:
        for H in attached_H:
            v += _atom_pos(H) - pos
        # average if multiple Hs
        v /= float(len(attached_H))

    # Acceptor: point away from heavy neighbors
    # (e.g. carbonyl O points away from C; ring N points out of ring)
    if role_int in (2, 3) and not attached_H:
        if heavy_neigh:
            if len(heavy_neigh) == 1:
                # Single neighbor: point away from that neighbor
                v = pos - _atom_pos(heavy_neigh[0])
            else:
                # Multiple neighbors: point opposite the average bond vector
                sum_vec = np.zeros(3, dtype=float)
                for nb in heavy_neigh:
                    sum_vec += _atom_pos(nb) - pos
                v = -sum_vec  # away from neighbors

    # Normalize
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.zeros(3, dtype=float)
    return v / norm


class PreCompDataProtein:
    def __init__(self,
                 binding_site,
                 system,
                 site_maps=None,
                 bias_radius= 12.0, 
                 split_site = False,
                 probe_radius = 1.4,
                 grid_apix = (0.375, 0.375, 0.375),
                 grid_spacing = 0.375,
                 pad_box = 10.0
                 ):
        
        #-----docking setup-----
        self.n_global_search = system.options.n_global_search 
        self.n_local_search = system.options.n_local_search 
        self.ncpu = os.cpu_count() - 2 
        self.repCap0 = system.options.repulsion_cap_0
        self.repCap1 = system.options.repulsion_cap_1
        
        self.repCap_inner_nm = system.options.repulsion_cap_nm
        self.repCap_final_nm = system.options.repulsion_cap_polish
        self.rms_cutoff = 2.0
        self.topN = system.options.return_n
        self.a_lo = 0.25
        self.a_mid = 0.45
        self.a_hi = 0.70
        self.iterations = system.options.max_iterations
        
        
        

        
        #-----protein data------
        self.residues = binding_site.lining_residues
        self.protein_positions = binding_site.rdkit_lining_mol.GetConformer().GetPositions()
        self.protein_pdb_atom_ids = []
        self.protein_atom_types = []
        self.protein_atom_roles = []
        
        self.protein_formal_charge = []
        self.protein_hbond_dirs = []
        
        
        for res in binding_site.lining_residues:
            
            for atom in res.atoms:
                
                if atom.element_name != "H":
                    
                    self.protein_pdb_atom_ids.append((res.name, atom.name))
                    
                    
                    role_int = get_role_int(res.name, atom.name)
                    self.protein_atom_roles.append(role_int)
                    self.protein_atom_types.append(AtomType.from_id(res.name, atom.name).idx)
                    
                    #if res.name in protein_charged_centers:
                    
                    self.protein_formal_charge.append(residue_atom_charge(res.name, atom.name))
                    hdir = get_hbond_direction_for_atom(atom, role_int)
                    self.protein_hbond_dirs.append(hdir)
                    
                        #cc = protein_charged_centers[res.name]
                        
                        #if atom.name in cc:
                            
                        #    self.protein_formal_charge.append(cc[atom.name])
                        #else:
                        #    self.protein_formal_charge.append(0)
                   # else:
                        #self.protein_formal_charge.append(0)
        
        self.protein_formal_charge = np.asarray(self.protein_formal_charge)
        self.protein_hbond_dirs   = np.asarray(self.protein_hbond_dirs, dtype=float)
        
        #import pdb 
        #pdb.set_trace()
                    
        self.protein_atom_types  = np.array(self.protein_atom_types, dtype=np.int32)
        self.protein_atom_roles = np.array(self.protein_atom_roles, dtype=np.int32)
        self.protein_radii = np.array([get_vdw_radius(i.GetSymbol()) for i in binding_site.rdkit_lining_mol.GetAtoms()])
        self.protein_hydrogens = get_protein_hydrogen_reference(binding_site.rdkit_lining_mol)
        
        #Don't need charges any more because the grid is computed already 
        self.protein_ring_types, self.protein_ring_coords, self.protein_ring_idx = compute_protein_ring_types(binding_site.lining_residues, self.protein_positions)
        self.protein_ring_type_ints= np.array([i.idx for i in self.protein_ring_types], dtype=np.int32)
        #has lining mol change
        self.halogen_bond_acceptor_indices, self.halogen_bond_acceptor_root_indices = compute_halogen_bond_data(binding_site.rdkit_lining_mol, self.protein_atom_types, HALOGEN_ACCEPTOR_ATOM_IDXS)
        
        #-----binding site data------
        
        self.translation_points, self.adjacency = get_valid_points_and_adjacency(binding_site.distance_map > probe_radius, 
                                                                      connectivity=26.0)
        
        
        
        self.binding_site_grid_origin = binding_site.origin
        self.binding_site_grid_apix = binding_site.apix
        self.binding_site_grid_nz,self.binding_site_grid_ny,self.binding_site_grid_nx = binding_site.distance_map.shape 
        
        
        self.apix = grid_apix #no longer needed
        
        
        key = int(binding_site.key)
        if system.centroid is None or len(system.centroid) <= key:
            translation_coords = covert_idx_to_coords(np.array(self.translation_points),
                                                      binding_site.origin,
                                                      binding_site.apix
                                                      )
            centroid, radius = centroid_and_radius(translation_coords)
            self.binding_site_centroid = centroid 
            self.bias_radius = radius
        else :
            self.binding_site_centroid = system.centroid[key]
            self.bias_radius = bias_radius
            
        
        
        #split large sites into smaller regions
        if split_site:
            self.set_multi_sites(self.translation_points,
                                 binding_site.origin,
                                 binding_site.apix,
                                 binding_site.distance_map,
                                 system.output)
        
        #----binding site grids 
        if site_maps is None:
            
            
            atoms, positions, atom_radii = get_position_input(system.protein.complex_structure)
            zero_grid_origin, zero_grid = make_zero_grid(positions,
                                               spacing = grid_spacing,
                                               padding = pad_box
                                               )
                
                
                
            
            site_maps = build_site_maps_standalone(
               
                # geometry / atoms
                positions=positions,
                atom_radii=atom_radii,
                atoms=atoms,

                # grid
                grid_origin=zero_grid_origin,
                grid_spacing=grid_spacing,
                grid=zero_grid,  # used for shape

                # things that used to be from self.system
                system_centroid=self.binding_site_centroid,
                protein_complex_structure=system.protein.complex_structure,
                protein_complex_system=system.protein.complex_system,

                # optional tuning knobs to keep your old behavior adjustable
                crop_box_size=(30, 30, 30),
                electro_cutoff=12.0)
        
        

        
        
        self.env_scaled_origin = site_maps['env_scaled_map'].origin
        self.env_scaled_apix   = np.array(site_maps['env_scaled_map'].apix)
        self.env_scaled_grid   = site_maps['env_scaled_map'].density_map
        self.envz, self.envy, self.envx = site_maps['env_scaled_map'].density_map.shape
        
        # electro_scaled
        self.electro_scaled_origin = site_maps['electro_scaled_map'].origin
        self.electro_scaled_apix   = np.array(site_maps['electro_scaled_map'].apix)
        self.electro_scaled_grid   = site_maps['electro_scaled_map'].density_map
        self.electroz, self.electroy, self.electrox = site_maps['electro_scaled_map'].density_map.shape
        
        self.electro_raw_origin = site_maps['electro_raw_map'].origin
        self.electro_raw_apix   = np.array(site_maps['electro_raw_map'].apix)
        self.electro_raw_grid   = site_maps['electro_raw_map'].density_map
        self.electrorawz, self.electrrawoy, self.electrrawox = site_maps['electro_raw_map'].density_map.shape
    
        # hydrophob_raw
        self.hydrophob_raw_origin = site_maps['hydrophob_raw_map'].origin
        self.hydrophob_raw_apix   = np.array(site_maps['hydrophob_raw_map'].apix)
        self.hydrophob_raw_grid   = site_maps['hydrophob_raw_map'].density_map
        self.hydrophob_raw_z, self.hydrophob_raw_y, self.hydrophob_raw_x = site_maps['hydrophob_raw_map'].density_map.shape
        
        # hydrophob_enc
        self.hydrophob_enc_origin = site_maps['hydrophob_enc_map'].origin
        self.hydrophob_enc_apix   = np.array(site_maps['hydrophob_enc_map'].apix)
        self.hydrophob_enc_grid   = site_maps['hydrophob_enc_map'].density_map
        self.hydrophob_enc_z, self.hydrophob_enc_y, self.hydrophob_enc_x = site_maps['hydrophob_enc_map'].density_map.shape
        
        
        #------scores-----
        #salt bridge Buckingham boost 
        self.SB_A = -3.85504
        self.SB_B = 0.345362
        self.SB_C = -144.293
        #aromatic score
        self.aromatic_score = AromaticScore()
        _stack_map = {'p': 0, 't': 1}

        
        keys      = []
        kxA_list  = []; kyA_list  = []; dimA_list  = []; coeffsA = []
        kxB_list  = []; kyB_list  = []; dimB_list  = []; coeffsB = []
        kxC_list  = []; kyC_list  = []; dimC_list  = []; coeffsC = []
        knots_x_list_A = []
        knots_y_list_A = []
        
        knots_x_list_B = []
        knots_y_list_B = []
        
        knots_x_list_C = []
        knots_y_list_C = []
        
        
        for (i1, i2, stack) in self.aromatic_score.spline_models:
            splineA, splineB, splineC = self.aromatic_score.spline_models[(i1, i2, stack)]
            keys.append((i1, i2, _stack_map[stack]))
        
            # — pull knots & degrees for A
            txA, tyA = splineA.get_knots()
            knots_x_list_A.append(np.array(txA, dtype=np.float64))
            knots_y_list_A.append(np.array(tyA, dtype=np.float64))
            kxA, kyA = splineA.degrees
            nxA = len(txA) - kxA - 1
            nyA = len(tyA) - kyA - 1
            cA  = splineA.get_coeffs()
            assert cA.size == nxA*nyA, (
                f"A[{i1},{i2},{stack}]: expected {nxA*nyA} coeffs, got {cA.size}"
            )
            
            # — pull knots & degrees for B
            txB, tyB = splineB.get_knots()
            knots_x_list_B.append(np.array(txB, dtype=np.float64))
            knots_y_list_B.append(np.array(tyB, dtype=np.float64))
            kxB, kyB = splineB.degrees
            nxB = len(txB) - kxB - 1
            nyB = len(tyB) - kyB - 1
            cB  = splineB.get_coeffs()
            assert cB.size == nxB*nyB, (
                f"B[{i1},{i2},{stack}]: expected {nxB*nyB} coeffs, got {cB.size}"
            )
        
            # — pull knots & degrees for C
            txC, tyC = splineC.get_knots()
            knots_x_list_C.append(np.array(txC, dtype=np.float64))
            knots_y_list_C.append(np.array(tyC, dtype=np.float64))
            kxC, kyC = splineC.degrees
            nxC = len(txC) - kxC - 1
            nyC = len(tyC) - kyC - 1
            cC  = splineC.get_coeffs()
            assert cC.size == nxC*nyC, (
                f"C[{i1},{i2},{stack}]: expected {nxC*nyC} coeffs, got {cC.size}"
            )
        
            
        
            # — reshape
            coeffsA.append(cA.reshape((nxA, nyA)))
            kxA_list.append(kxA); kyA_list.append(kyA); dimA_list.append((nxA, nyA))
        
            coeffsB.append(cB.reshape((nxB, nyB)))
            kxB_list.append(kxB); kyB_list.append(kyB); dimB_list.append((nxB, nyB))
        
            coeffsC.append(cC.reshape((nxC, nyC)))
            kxC_list.append(kxC); kyC_list.append(kyC); dimC_list.append((nxC, nyC))
            
            
        # 
        self.arom_keys   = np.array(keys,    dtype=np.int32)     # (M,3)
        
        self.arom_kxA    = np.array(kxA_list, dtype=np.int32)    # (M,)
        self.arom_kyA    = np.array(kyA_list, dtype=np.int32)
        #I only need this if the if else is true!
        self.arom_kx = np.array(kxA_list, dtype=np.int32)    # shape (M,)
        self.arom_ky = np.array(kyA_list, dtype=np.int32)
        
        self.arom_dimsA  = np.array(dimA_list,dtype=np.int32)    # (M,2)
        self.arom_dims = np.array(dimA_list, dtype=np.int32)
        self.arom_coefA  = coeffsA                                    # list of (nxA,nyA) arrays
        
        self.arom_kxB    = np.array(kxB_list, dtype=np.int32)
        self.arom_kyB    = np.array(kyB_list, dtype=np.int32)
        self.arom_dimsB  = np.array(dimB_list,dtype=np.int32)
        self.arom_coefB  = coeffsB
        
        self.arom_kxC    = np.array(kxC_list, dtype=np.int32)
        self.arom_kyC    = np.array(kyC_list, dtype=np.int32)
        self.arom_dimsC  = np.array(dimC_list,dtype=np.int32)
        self.arom_coefC  = coeffsC

        #self.arom_knots_x = knots_x_list         # Python list of 1D numpy arrays
        #self.arom_knots_y = knots_y_list
        self.arom_knots_xA = knots_x_list_A
        self.arom_knots_yA = knots_y_list_A
        self.arom_knots_xB = knots_x_list_B
        self.arom_knots_yB = knots_y_list_B
        self.arom_knots_xC = knots_x_list_C
        self.arom_knots_yC = knots_y_list_C
        
        #Hbond polynomials 
        donor_types   = sorted([int(d) for d in HBOND_POLYA.keys()])
        acceptor_set  = {int(a) for sub in HBOND_POLYA.values() for a in sub.keys() }
        acceptor_types = sorted(acceptor_set)
        
        D = len(donor_types)
        A = len(acceptor_types)
        
        degA = len(HBOND_POLYA[str(donor_types[0])][str(acceptor_types[0])]) - 1
        degB = len(HBOND_POLYB[str(donor_types[0])][str(acceptor_types[0])]) - 1
        degC = len(HBOND_POLYC[str(donor_types[0])][str(acceptor_types[0])]) - 1
        
        
        self.hbond_donor_types    = np.array(donor_types,   dtype=int)
        self.hbond_acceptor_types = np.array(acceptor_types,dtype=int)
        
        
        self.hbond_polyA = np.zeros((D, A, degA+1), dtype=float)
        self.hbond_polyB = np.zeros((D, A, degB+1), dtype=float)
        self.hbond_polyC = np.zeros((D, A, degC+1), dtype=float)
        
    
        for di, d in enumerate(donor_types):
            for ai, a in enumerate(acceptor_types):
                arrA = HBOND_POLYA  [str(d)][str(a)]
                arrB = HBOND_POLYB  [str(d)][str(a)]
                arrC = HBOND_POLYC  [str(d)][str(a)]
                self.hbond_polyA[di, ai, :] = arrA
                self.hbond_polyB[di, ai, :] = arrB
                self.hbond_polyC[di, ai, :] = arrC
        
        #-----halogen bond score data-----
        self.halogen_score = HalogenBondScore()
        halo_keys        = []
        halo_kxA_list    = []; halo_kyA_list    = []; halo_dimA_list    = []; halo_coeffsA = []
        halo_kxB_list    = []; halo_kyB_list    = []; halo_dimB_list    = []; halo_coeffsB = []
        halo_kxC_list    = []; halo_kyC_list    = []; halo_dimC_list    = []; halo_coeffsC = []
        halo_knots_x_list_A = []
        halo_knots_y_list_A = []
        halo_knots_x_list_B = []
        halo_knots_y_list_B = []
        halo_knots_x_list_C = []
        halo_knots_y_list_C = []
        
        # The key is (donor_idx, acceptor_idx) and the value is a tuple of 3 splines.
        for (i1, i2), (splineA, splineB, splineC) in self.halogen_score.spline_models.items():
            halo_keys.append((i1, i2))
            
            # — Pull knots, degrees, and coefficients for spline A
            txA, tyA = splineA.get_knots()
            halo_knots_x_list_A.append(np.array(txA, dtype=np.float64))
            halo_knots_y_list_A.append(np.array(tyA, dtype=np.float64))
            kxA, kyA = splineA.degrees
            nxA = len(txA) - kxA - 1
            nyA = len(tyA) - kyA - 1
            cA  = splineA.get_coeffs()
            assert cA.size == nxA * nyA, f"Halogen A[{i1},{i2}]: expected {nxA*nyA} coeffs, got {cA.size}"
            
            # — Pull knots, degrees, and coefficients for spline B
            txB, tyB = splineB.get_knots()
            halo_knots_x_list_B.append(np.array(txB, dtype=np.float64))
            halo_knots_y_list_B.append(np.array(tyB, dtype=np.float64))
            kxB, kyB = splineB.degrees
            nxB = len(txB) - kxB - 1
            nyB = len(tyB) - kyB - 1
            cB  = splineB.get_coeffs()
            assert cB.size == nxB * nyB, f"Halogen B[{i1},{i2}]: expected {nxB*nyB} coeffs, got {cB.size}"
        
            
            txC, tyC = splineC.get_knots()
            halo_knots_x_list_C.append(np.array(txC, dtype=np.float64))
            halo_knots_y_list_C.append(np.array(tyC, dtype=np.float64))
            kxC, kyC = splineC.degrees
            nxC = len(txC) - kxC - 1
            nyC = len(tyC) - kyC - 1
            cC  = splineC.get_coeffs()
            assert cC.size == nxC * nyC, f"Halogen C[{i1},{i2}]: expected {nxC*nyC} coeffs, got {cC.size}"
            
            
            # — Reshape coefficients and append all data to lists
            halo_coeffsA.append(cA.reshape((nxA, nyA)))
            halo_kxA_list.append(kxA); halo_kyA_list.append(kyA); halo_dimA_list.append((nxA, nyA))
        
            halo_coeffsB.append(cB.reshape((nxB, nyB)))
            halo_kxB_list.append(kxB); halo_kyB_list.append(kyB); halo_dimB_list.append((nxB, nyB))
        
            halo_coeffsC.append(cC.reshape((nxC, nyC)))
            halo_kxC_list.append(kxC); halo_kyC_list.append(kyC); halo_dimC_list.append((nxC, nyC))

            
        
        # Store the collected data as instance attributes, ready for C++
        # The keys are (M, 2) since there's no stack type
        self.halo_keys    = np.array(halo_keys, dtype=np.int32)
        
        # Spline A data
        self.halo_kxA     = np.array(halo_kxA_list, dtype=np.int32)  # (M,)
        self.halo_kyA     = np.array(halo_kyA_list, dtype=np.int32)  # (M,)
        self.halo_dimsA   = np.array(halo_dimA_list, dtype=np.int32) # (M,2)
        self.halo_coefA   = halo_coeffsA                              # List of (nx,ny) arrays
        self.halo_knots_xA = halo_knots_x_list_A                      # List of 1D numpy arrays
        self.halo_knots_yA = halo_knots_y_list_A
        
        # Spline B data
        self.halo_kxB     = np.array(halo_kxB_list, dtype=np.int32)
        self.halo_kyB     = np.array(halo_kyB_list, dtype=np.int32)
        self.halo_dimsB   = np.array(halo_dimB_list, dtype=np.int32)
        self.halo_coefB   = halo_coeffsB
        self.halo_knots_xB = halo_knots_x_list_B
        self.halo_knots_yB = halo_knots_y_list_B
        
        # Spline C data
        self.halo_kxC     = np.array(halo_kxC_list, dtype=np.int32)
        self.halo_kyC     = np.array(halo_kyC_list, dtype=np.int32)
        self.halo_dimsC   = np.array(halo_dimC_list, dtype=np.int32)
        self.halo_coefC   = halo_coeffsC
        self.halo_knots_xC = halo_knots_x_list_C
        self.halo_knots_yC = halo_knots_y_list_C
        
        
        self.w_nonbond  = 0.015435
        self.w_dsasa = -0.000325
        self.w_hphob = 2.429216
        self.w_electro = 0.006459
        self.w_ligand_torsion = 0.046476
        self.w_ligand_intra = 0.001517
        self.bias = -4.0204
        #new weights
        self.w_vdw = 0.015435 
        self.w_hbond = 0.003415 
        self.w_aromatic = 0.114082 
        self.w_halogen = 0.160168 
        self.w_hphob_enc = 0.034452 
        self.w_constraint = 1.0
        
        self.nb_cell = 4.5
        
        #add maps 
        #-----Add the density map----
        if not system.options.no_map and hasattr(system,'binding_site_maps') :
            self.no_map = False
            density_map = system.binding_site_maps[binding_site.key][0][0]
            
            self.binding_site_density_map_grid = density_map.density_map 
            self.binding_site_density_map_apix = density_map.apix 
            self.binding_site_density_map_origin = density_map.origin
            self.binding_site_density_map_resolution = density_map.resolution
            self.binding_site_density_map_sigma_coeff = 0.356
            self.mi_weight = system.options.mi_weight
            self.sci_weight = system.options.sci_weight
            
        else:
            self.no_map = True
           
            self.binding_site_density_map_grid = None
            self.binding_site_density_map_apix = None 
            self.binding_site_density_map_origin = None
            self.binding_site_density_map_resolution = None
            self.binding_site_density_map_sigma_coeff = None
            
            self.G_kernal = None
            self.grad_kernal = None
            self.G_lap_kernal = None
            self.r = None
            self.smoothed_map = None
            self.grad_map  = None
            self.laplacian_map = None
            self.atom_masses = None
            self.mi_weight = 0.0
        
            
    def __add__(self, other):
        """
        protein + ligand  ->  new protein object with ligand attrs added
        """
        if isinstance(other, PreCompDataLigand):
            
            merged = copy.deepcopy(self)
            _attach_attributes(merged, other, prefix="")  
            merged.__post_init__()
            return merged
        return NotImplemented  

    # optional in-place variant:  protein += ligand
    def __iadd__(self, other):
        if isinstance(other, PreCompDataLigand):
            _attach_attributes(self, other, prefix="")
            self.__post_init__()
            return self
        return NotImplemented
        
    def copy(self):
        return copy.deepcopy(self)
    
    def add_multi_site_bias(self, centroid, radius):
        
        
            
        self.binding_site_centroid = centroid 
        self.bias_radius = radius
        
        self.translation_points,  self.adjacency = filter_points_and_adjacency_by_radius(
            np.array(self.translation_points),
            self.adjacency,
            self.binding_site_origin,
            self.binding_site_grid_apix,
            self.binding_site_centroid,
            self.bias_radius
        )
        self._reset_ACO_simplex()
        
        
    def __post_init__(self):
        
        '''
        complete the initilisation of docking data once the ligand has been added.
        
        '''
        
        
        #-----scoring funciton terms-----
        n_protein_atoms = len(self.protein_atom_types)
        #int
        n_ligand_atoms = len(self.ligand_atom_types)
        #shape = (n_protein_atoms, n_ligand_atoms), type = float
        self.A_values = np.zeros((n_protein_atoms, n_ligand_atoms), dtype=np.float64)
        #shape = (n_protein_atoms, n_ligand_atoms), type = float
        self.B_values = np.zeros((n_protein_atoms, n_ligand_atoms), dtype=np.float64)
        #shape = (n_protein_atoms, n_ligand_atoms), type = float
        self.C_values = np.zeros((n_protein_atoms, n_ligand_atoms), dtype=np.float64)
        
        
        for i, p_idx in enumerate(self.protein_atom_types):
            for j, l_idx in enumerate(self.ligand_atom_types):
                # Use p_idx and l_idx as row/col in TABLE_A, etc.
                self.A_values[i, j] = TABLE_A[p_idx, l_idx]
                self.B_values[i, j] = TABLE_B[p_idx, l_idx]
                self.C_values[i, j] = TABLE_C[p_idx, l_idx]
        
        #-----docking ACO params----
        self.hbond_donor_acceptor_mask = compute_donor_acceptor_mask(self.protein_pdb_atom_ids, 
                                                                     self.protein_atom_types, 
                                                                     self.ligand_atom_types)
            
        self._reset_ACO_simplex()
        
        
        
    
        
        #------density map addition------
        if self.binding_site_density_map_grid is not None:
            
        
            
            #Build local Gaussian kernels for scoring 
            G, Gx, Gy, Gz, Glap, r = get_gaussian_kernels(self.binding_site_density_map_resolution, 
                                                          self.binding_site_density_map_apix[0],
                                                          cutoff=3.0,
                                                          sigma_coeff=0.356)
            
            grad_kernel_mag = np.sqrt(Gx**2 + Gy**2 + Gz**2)
            
            
            
            
            print("Pre-computing feature maps from ground truth ligand...")
            smoothed_map_data = ndimage.convolve(self.binding_site_density_map_grid, G, mode='constant', cval=0.0)
            dx_field = ndimage.convolve(smoothed_map_data, kernel_dx, mode='constant', cval=0.0)
            dy_field = ndimage.convolve(smoothed_map_data, kernel_dy, mode='constant', cval=0.0)
            dz_field = ndimage.convolve(smoothed_map_data, kernel_dz, mode='constant', cval=0.0)
            exp_grad_mag = np.sqrt(dx_field**2 + dy_field**2 + dz_field**2)
            laplacian_map = ndimage.laplace(smoothed_map_data, mode='constant', cval=0.0)
            
            self.G_kernal = G
            self.grad_kernal = grad_kernel_mag
            self.G_lap_kernal = Glap
            self.r = r
            self.smoothed_map = smoothed_map_data #why is this here?
            self.grad_map  = exp_grad_mag 
            self.laplacian_map = laplacian_map
            
            
            ccc0_map, ccc1_map, ccc2_map  = precompute_score_maps(smoothed_map_data, 
                                  exp_grad_mag,
                                  laplacian_map,
                                  G,
                                  grad_kernel_mag,
                                  Glap,
                                  self.binding_site_density_map_origin,
                                  self.binding_site_density_map_apix[0],
                                  r)
            
            self.smoothed_map_grid = ccc0_map
            self.smoothed_map_origin = self.binding_site_density_map_origin
            self.smoothed_map_apix = self.binding_site_density_map_apix 
            
            self.grad_map_grid  = ccc1_map
            self.grad_map_origin = self.binding_site_density_map_origin
            self.grad_map_apix = self.binding_site_density_map_apix 
            
            self.laplacian_map_grid = ccc2_map
            self.laplacian_map_origin = self.binding_site_density_map_origin
            self.laplacian_map_apix = self.binding_site_density_map_apix
            
            #remove!!
            #mi precomp data 
            mi = build_mi_assets_for_map(
                self.binding_site_density_map_grid,
                self.binding_site_density_map_origin,
                self.binding_site_density_map_apix,
                self.binding_site_density_map_resolution,
                bins=getattr(self, "mi_bins", 20),                # optional override
                sigma_coeff=getattr(self, "binding_site_density_map_sigma_coeff", 0.356)
            )
            
            
            self.mi_bins       = mi["mi_bins"]
            self.mi_map_bins_grid = mi["mi_map_bins"]         # uint8 [Z,Y,X]
            self.mi_map_bins_origin = mi["mi_origin"]           # float64[3]
            self.mi_map_bins_apix = mi["mi_apix"]             # float64[3]
            self.mi_k_offsets  = mi["mi_k_offsets"]        # int16 [N,3]
            self.mi_k_bins     = mi["mi_k_bins"]           # uint8 [N]
            self.mi_kernel_r   = mi["mi_kernel_r"]
            
            

        else:
            self.binding_site_density_map_grid = None
            self.binding_site_density_map_apix = None 
            self.binding_site_density_map_origin = None
            self.binding_site_density_map_resolution = None
            self.binding_site_density_map_sigma_coeff = None
            
            self.G_kernal = None
            self.grad_kernal = None
            self.G_lap_kernal = None
            self.r = None
            self.smoothed_map = None
            self.grad_map  = None
            self.laplacian_map = None
            self.atom_masses = None
            
            self.mi_bins       = None
            self.mi_map_bins_grid = None    
            self.mi_map_bins_origin = None
            self.mi_map_bins_apix = None
            self.mi_k_offsets  = None
            self.mi_k_bins     = None
            self.mi_kernel_r   = None
        
        
        
    
    def _prepare_torsion_only_docking(self, new_torsions, ligand, exc_atoms=[]):
        self.ligand_torsion_idxs = new_torsions
        self.ligand_torsion_profile = export_torsion_profile(ligand,  new_torsions)
        #THIS MUST CHANGE AS THE TORSION SCORES WILL NOT BE Eqivelent
        
        self.ligand_torsion_idxs = [i[1] for i in self.ligand_torsion_profile]
        self.ligand_torsion_scores = [i[2] for i in self.ligand_torsion_profile]
        self.end_torsions = len(new_torsions)
        self._reset_ACO_simplex()
        
        #add atoms to exclude during scoring
        #usually covenelent bonded atoms in covalent docking 
        self.exc_atoms = [ ]
        
        for i in self.ligand_heavy_atom_indexes:
            if i in exc_atoms:
                self.exc_atoms.append(0)
            else:
                self.exc_atoms.append(1)
        
        
    def _reset_ACO_simplex(self):
        self.all_arrays = [self.translation_points]
        self.simplex = [0.5, 0.5, 0.5]
        for i in range(3):
            self.all_arrays.append(np.arange(0, 360, 30, dtype=np.int32)) #changed from 5
            self.simplex.append(0.5)
            
        for tor in self.ligand_torsion_idxs:
            
            self.all_arrays.append(np.arange(0, 360, 30,dtype=np.int32))
            self.simplex.append(0.5)
            
        self.simplex  = np.array(self.simplex)
        
    
    def set_multi_sites(self, translation_points, origin, apix, distance_map, output = './', write= True):
        
            
        
        translation_coords = covert_idx_to_coords(np.array(translation_points),
                                                  origin,
                                                  np.array(apix)
                                                  )
        
        elbow_k, labels, inertias = kmeans_elbow_plot(translation_coords,show=False)
        
        lab_map = np.zeros(distance_map.shape)
        for idx, lab in zip(translation_points, labels):
            
                lab_map[idx[0], idx[1], idx[2]] = lab + 1
       
        
        dmap = EMMap(origin,
                     apix,
                     lab_map,
                     3.0)
        if write:
            dmap.write_mrc(os.path.join(output, 'cluster_map.mrc' ))
        
        all_translation_points = []
        all_translation_centroid_raidus = []
        for lab in np.unique(labels):
            
            mask = labels == lab
            
            new_translation_points = np.array(translation_points)[mask]
            centroid, raidus = centroid_and_radius(translation_coords[mask])
            all_translation_points.append(new_translation_points)
            all_translation_centroid_raidus.append((centroid, raidus))
       
        print('multi-site')
        self.split_site_translation_points = all_translation_points 
        self.split_site_translation_centroid_raidus = all_translation_centroid_raidus



def make_zero_grid(atom_coords,
                   spacing=1.0,
                   padding=2.0):
    """
    Create a 3D zero-filled grid that bounds a set of atoms with padding.

    Parameters
    ----------
    atom_coords : (N, 3) np.ndarray
        Atom coordinates in Å, columns [x, y, z].
    spacing : float, optional
        Grid spacing in Å between neighboring points along each axis.
    padding : float, optional
        Extra space in Å added beyond the min/max atom coordinates on each side.

    Returns
    -------
    origin : (3,) np.ndarray
        [x0, y0, z0] coordinate of the grid origin (lower corner), in Å.
    grid : np.ndarray
        3D array of zeros with shape (nz, ny, nx) covering the atoms + padding.
        Axis order is (Z, Y, X).
    """
    atom_coords = np.asarray(atom_coords, dtype=float)
    if atom_coords.ndim != 2 or atom_coords.shape[1] != 3:
        raise ValueError("atom_coords must be shape (N, 3)")

    # Bounds + padding
    mins = atom_coords.min(axis=0) - padding  # [xmin, ymin, zmin]
    maxs = atom_coords.max(axis=0) + padding  # [xmax, ymax, zmax]

    # Number of grid points along each axis (X, Y, Z)
    lengths = maxs - mins
    nx = int(np.ceil(lengths[0] / spacing)) + 1
    ny = int(np.ceil(lengths[1] / spacing)) + 1
    nz = int(np.ceil(lengths[2] / spacing)) + 1

    origin = mins.copy()  # [x0, y0, z0]

    # Zero-filled grid, axis order (Z, Y, X)
    grid = np.zeros((nz, ny, nx), dtype=float)

    return origin, grid


def get_position_input(protein_openff_structure):
    
    
    atoms = [i for i in protein_openff_structure.atoms if i.element > 1]
    positions = np.array([[i.xx, i.xy, i.xz] for i in atoms ])
    atom_radii = np.array( [get_van_der_waals_radius(i.element_name) for i in atoms] )
    
    return atoms, positions, atom_radii
    

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
           
#maptools 


def build_mi_assets_for_map(
    density_map: np.ndarray,
    origin_xyz,               # (x,y,z)
    apix_xyz,                 # (ax,ay,az)
    resolution: float,        # EM map resolution (Å)
    *,
    bins: int = 32,
    sigma_coeff: float = 0.356,
    cutoff_sigma: float = 3.0,
    eps: float = 1e-8,
    kernel_threshold_rel: float = 1e-3,
    robust_clip: tuple[float,float] | None = (1.0, 99.0),
):
    """
    Returns a dict containing everything C++ needs for fast MI:
      - mi_map_bins:   uint8 map of same shape as density_map (values in [0..bins-1])
      - mi_bins:       number of bins
      - mi_origin:     float64[3]
      - mi_apix:       float64[3]
      - mi_k_offsets:  int16[N,3] sparse offsets (dz,dy,dx)
      - mi_k_bins:     uint8[N] (bin per offset) in [0..bins-1]
      - mi_kernel_r:   int
    """
    assert density_map.ndim == 3, "density_map must be (Z,Y,X)"
    apix_xyz = np.asarray(apix_xyz, dtype=np.float64)
    origin_xyz = np.asarray(origin_xyz, dtype=np.float64)

    # Use only a *Gaussian-smoothed* intensity map for MI (no LoG / gradients)
    sigma_ang = sigma_coeff * float(resolution)      # Gaussian sigma in Å
    sigma_vox = sigma_ang / apix_xyz                 # per-axis sigma in voxels
    # scipy allows per-axis sigma; order=0 is fine (Gaussian filter handles smoothing)
    smoothed = ndimage.gaussian_filter(density_map.astype(np.float32, copy=False),
                                       sigma=sigma_vox, mode="constant", cval=0.0)

    # Robust min/max for binning (ignores outliers)
    if robust_clip is not None:
        lo, hi = np.percentile(smoothed, robust_clip)
    else:
        lo, hi = float(smoothed.min()), float(smoothed.max())
    if hi - lo < eps:  # flat map safeguard
        hi = lo + 1.0

    # Pre-bin the map to uint8
    scale = (bins - 1) / (hi - lo)
    mi_map_bins = np.clip(np.floor((smoothed - lo) * scale), 0, bins - 1).astype(np.uint8, copy=False)

    # Build a compact sparse Gaussian kernel and pre-bin it too
    # Kernel radius (in voxels) — take the *largest* axis to ensure coverage
    r = int(math.ceil(cutoff_sigma * sigma_vox.max()))
    # Build isotropic coords in voxel units relative to center (Z,Y,X)
    zz, yy, xx = np.mgrid[-r:r+1, -r:r+1, -r:r+1]
    # Continuous Gaussian in Å-accurate metric: use anisotropic sigma_vox per axis
    G = np.exp(-0.5 * ((xx / sigma_vox[0])**2 + (yy / sigma_vox[1])**2 + (zz / sigma_vox[2])**2)).astype(np.float32)

    # Normalize to [0,1] for binning, prune near-zero entries to make it sparse
    gmax = float(G.max())
    if gmax <= eps:
        raise ValueError("Gaussian kernel degenerate.")
    Gn = G / gmax
    mask = Gn >= (kernel_threshold_rel)
    dz, dy, dx = zz[mask].astype(np.int16), yy[mask].astype(np.int16), xx[mask].astype(np.int16)

    # Bin the kernel values too (same number of bins as the map)
    k_bins = np.clip(np.floor(Gn[mask] * (bins - 1)), 0, bins - 1).astype(np.uint8)

    return {
        "mi_map_bins": mi_map_bins,      # uint8 [Z,Y,X]
        "mi_bins": int(bins),
        "mi_origin": origin_xyz.astype(np.float64),   # float64[3]
        "mi_apix": apix_xyz.astype(np.float64),       # float64[3]
        "mi_k_offsets": np.stack([dz, dy, dx], axis=1),  # int16 [N,3] with (dz,dy,dx)
        "mi_k_bins": k_bins,            # uint8 [N]
        "mi_kernel_r": int(r),
        "mi_val_lo": float(lo),         # (optional metadata; C++ not required)
        "mi_val_hi": float(hi),
    }


def compute_water_map_grid_cpp(donor_Hs, acceptors, origin, apix, protein_xyz, 
                               nx,ny,nz):
    
    (A_map, D_map) = grid_maps.compute_water_site_maps(
        donor_Hs, acceptors,
        origin, apix,
        ((nz,ny,nx),(nz,ny,nx)),
        protein_xyz,
        r_opt=2.9, dr=0.6, min_protein_clearance=1.8
    )
    
    dmap = EMMap(origin, apix, A_map, 3.0)
    dmap.write_mrc('/Users/test/water_test/acceptor_map.mrc')
    dmap.density_map = D_map
    dmap.write_mrc('/Users/test/water_test/donor_map.mrc')
    import pdb 
    pdb.set_trace()

def compute_hydrophobic_grid_cpp(protein_positions, protein_hpi,
                         binding_site_map, binding_site_origin, apix,
                         cutoff_distance=6.0):
    
    
    hydrophobic_grid = np.zeros(binding_site_map.shape, dtype=np.float64)
    hydrophobic_grid = grid_maps.compute_hydrophobic_grid_gaussian(protein_positions, 
                                       protein_hpi,
                                       hydrophobic_grid, 
                                       binding_site_origin,
                                       np.array(apix),
                                       3.0,
                                       6.0)
                                       
    return hydrophobic_grid


def get_gaussian_kernels(resolution, apix, cutoff=3.0, sigma_coeff=0.356):
    """
    Builds discrete Gaussian, gradient-of-Gaussian, and LoG kernels.
    """
    sigma = sigma_coeff * resolution
    # Kernel radius in number of voxels
    r = int(math.ceil(cutoff * sigma / apix))
    coords = np.arange(-r, r + 1) * apix
    # meshgrid indexing='ij' gives (z,y,x) order which is what we want for kernels
    Z, Y, X = np.meshgrid(coords, coords, coords, indexing='ij')
    r2 = X**2 + Y**2 + Z**2
    G = np.exp(-r2 / (2 * sigma**2))
    # First derivatives
    Gx = - (X / (sigma**2)) * G
    Gy = - (Y / (sigma**2)) * G
    Gz = - (Z / (sigma**2)) * G
    # Laplacian-of-Gaussian
    Glap = ((r2 / sigma**4) - (3 / sigma**2)) * G
    return G, Gx, Gy, Gz, Glap, r


import numpy as np

def _gaussian_kernel1d(sigma, truncate=3.0):
    """1D Gaussian kernel normalized to sum=1."""
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k.astype(np.float32)

def _convolve1d_reflect(a, k, axis):
    """Separable 1D convolution with reflect padding along one axis."""
    pad = len(k) // 2
    if pad == 0:
        return a

    # Move target axis to last
    a_mv = np.moveaxis(a, axis, -1)
    # Reflect pad on last axis only
    pad_width = [(0, 0)] * (a_mv.ndim - 1) + [(pad, pad)]
    a_pad = np.pad(a_mv, pad_width, mode="reflect")

    # Flatten all leading dims into rows
    rows = a_pad.reshape(-1, a_pad.shape[-1])
    out = np.empty((rows.shape[0], a_mv.shape[-1]), dtype=np.float32)

    # 'valid' on padded rows gives original length
    for i in range(rows.shape[0]):
        out[i] = np.convolve(rows[i], k, mode="valid")

    out = out.reshape(a_mv.shape).astype(np.float32, copy=False)
    return np.moveaxis(out, -1, axis)

def _gaussian_blur3d(volume, sigma_vox, truncate=3.0):
    """
    Gaussian blur 3D volume.
    sigma_vox can be scalar or (sz, sy, sx) in voxels.
    """
    vol = volume.astype(np.float32, copy=False)

    # Fast path if SciPy is available
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(vol, sigma=sigma_vox, mode="reflect").astype(np.float32)
    except Exception:
        pass

    # Fallback: separable conv with reflect padding
    if np.isscalar(sigma_vox):
        sigmas = (float(sigma_vox), float(sigma_vox), float(sigma_vox))
    else:
        sigmas = tuple(float(s) for s in sigma_vox)

    out = vol
    # axis order is (z, y, x) for your arrays
    for axis, s in enumerate(sigmas):
        if s > 0:
            k = _gaussian_kernel1d(s, truncate=truncate)
            out = _convolve1d_reflect(out, k, axis=axis)
    return out.astype(np.float32, copy=False)

def _normalize_to_max1(m, eps=1e-12):
    """Scale so max becomes 1.0; clamp to [0,1]."""
    m = m.astype(np.float32, copy=False)
    mx = float(np.max(m))
    if mx > eps:
        m = m / mx
    # keep in [0,1] (and kill any tiny negatives from numeric blur)
    return np.clip(m, 0.0, 1.0).astype(np.float32, copy=False)

def precompute_score_maps(
    smoothed_map, grad_map, laplacian_map,
    G_kernel, grad_kernel, G_lap_kernel,
    origin, apix, r,
    # new knobs:
    blur_sigma_A=1.5,        # Å; increase (e.g. 2–3 Å) if maps are very sparse
    blur_truncate=3.0,
    normalize_max_to_1=True,
    blur_after_ccc=True,
    print_stats=True,
):
    """
    Pre-computes three maps where each voxel value is the local CCC score
    for the corresponding feature type; then optionally blurs + normalizes.
    """
    print("Pre-computing score component maps... (This may take some time)")
    assert smoothed_map.shape == grad_map.shape == laplacian_map.shape

    map_shape = smoothed_map.shape
    nz, ny, nx = map_shape

    ccc0_map = np.zeros(map_shape, dtype=np.float32)
    ccc1_map = np.zeros(map_shape, dtype=np.float32)
    ccc2_map = np.zeros(map_shape, dtype=np.float32)

    for z in range(nz):
        if z % 10 == 0:
            print(f"  Processing slice {z+1}/{nz}")
        for y in range(ny):
            for x in range(nx):
                current_pos = voxel_to_point(x, y, z, origin, apix)

                score0 = local_ccc(current_pos, smoothed_map,  G_kernel,     origin, apix, r)
                score1 = local_ccc(current_pos, grad_map,      grad_kernel,  origin, apix, r)
                score2 = local_ccc(current_pos, laplacian_map, G_lap_kernel, origin, apix, r)

                # clamp negatives (CCC can be [-1,1]; we want [0, ...])
                if score0 < 0.0: score0 = 0.0
                if score1 < 0.0: score1 = 0.0
                if score2 < 0.0: score2 = 0.0

                ccc0_map[z, y, x] = score0
                ccc1_map[z, y, x] = score1
                ccc2_map[z, y, x] = score2

    # ---- NEW: blur CCC maps to widen basin ----
    if blur_after_ccc and blur_sigma_A and blur_sigma_A > 0:
        # convert Å -> voxels; arrays are (z,y,x)
        sigma_vox = float(blur_sigma_A) / float(apix)
        if print_stats:
            print(f"Blurring CCC maps with Gaussian sigma={blur_sigma_A} Å (~{sigma_vox:.3f} vox), truncate={blur_truncate}")

        ccc0_map = _gaussian_blur3d(ccc0_map, sigma_vox=sigma_vox, truncate=blur_truncate)
        ccc1_map = _gaussian_blur3d(ccc1_map, sigma_vox=sigma_vox, truncate=blur_truncate)
        ccc2_map = _gaussian_blur3d(ccc2_map, sigma_vox=sigma_vox, truncate=blur_truncate)

        # keep non-negative after blur
        ccc0_map = np.clip(ccc0_map, 0.0, None)
        ccc1_map = np.clip(ccc1_map, 0.0, None)
        ccc2_map = np.clip(ccc2_map, 0.0, None)

    # ---- NEW: normalize so max becomes 1.0 ----
    if normalize_max_to_1:
        ccc0_map = _normalize_to_max1(ccc0_map)
        ccc1_map = _normalize_to_max1(ccc1_map)
        ccc2_map = _normalize_to_max1(ccc2_map)

    if print_stats:
        def _stats(name, m):
            return f"{name}: min={m.min():.6g} mean={m.mean():.6g} max={m.max():.6g} zero_frac={(m==0).mean():.4f}"
        print(_stats("CCC0", ccc0_map))
        print(_stats("CCC1", ccc1_map))
        print(_stats("CCC2", ccc2_map))

    print("Score map pre-computation complete.")
    return ccc0_map, ccc1_map, ccc2_map


def precompute_score_maps_v1(smoothed_map, grad_map, laplacian_map,
                           G_kernel, grad_kernel, G_lap_kernel,
                           origin, apix, r):
    """
    Pre-computes three maps where each voxel value is the local CCC score
    for the corresponding feature type.

    Args:
        smoothed_map (np.ndarray): The blurred density map.
        grad_map (np.ndarray): The gradient magnitude map.
        laplacian_map (np.ndarray): The Laplacian of Gaussian map.
        G_kernel (np.ndarray): The Gaussian kernel template.
        grad_kernel (np.ndarray): The gradient magnitude kernel template.
        G_lap_kernel (np.ndarray): The Laplacian of Gaussian kernel template.
        origin (tuple): The (x,y,z) real-world coordinates of the grid origin.
        apix (float): The voxel size in angstroms.
        r (int): The radius of the kernels in voxels.

    Returns:
        tuple: A tuple containing the three new score maps:
               (ccc0_map, ccc1_map, ccc2_map).
    """
    print("Pre-computing score component maps... (This may take some time)")
    # Ensure all maps have the same shape
    assert smoothed_map.shape == grad_map.shape == laplacian_map.shape

    map_shape = smoothed_map.shape
    nz, ny, nx = map_shape

    # Initialize the output score maps with zeros
    ccc0_map = np.zeros(map_shape, dtype=np.float32)
    ccc1_map = np.zeros(map_shape, dtype=np.float32)
    ccc2_map = np.zeros(map_shape, dtype=np.float32)

    # Iterate over every voxel in the grid
    for z in range(nz):
        # Print progress indicator
        if z % 10 == 0:
            print(f"  Processing slice {z+1}/{nz}")

        for y in range(ny):
            for x in range(nx):
                # Convert the current voxel index to a real-world position
                current_pos = voxel_to_point(x, y, z, origin, apix)

                # Calculate the CCC score for each feature type at this position
                score0 = local_ccc(current_pos, smoothed_map, G_kernel, origin, apix, r)
                score1 = local_ccc(current_pos, grad_map, grad_kernel, origin, apix, r)
                score2 = local_ccc(current_pos, laplacian_map, G_lap_kernel, origin, apix, r)

                # Store the scores in the new maps
                #
                
                if score0 < 0.0:
                    score0 = 0.0 
                
                if score1 < 0.0:
                    score1 = 0.0 
                
                if score2 < 0.0:
                    score2 = 0.0
                
                ccc0_map[z, y, x] = score0
                ccc1_map[z, y, x] = score1
                ccc2_map[z, y, x] = score2

    print("Score map pre-computation complete.")
    return ccc0_map, ccc1_map, ccc2_map


def local_ccc_cosine(atom_pos, field, kernel, origin, apix, r):
    """
    Computes local normalized cross-correlation (cosine similarity)
    between a 3D field and a template kernel around an atom position.

    This uses correlation-about-zero:

        CCC = <f, t> / (||f|| * ||t||)

    Parameters
    ----------
    atom_pos : (3,) array-like
        Real-space (x, y, z) coordinates.
    field : np.ndarray (Z, Y, X)
        3D density/feature map.
    kernel : np.ndarray (2r+1, 2r+1, 2r+1)
        Gaussian kernel on the same voxel grid.
    origin : (3,) array-like
        Map origin in real-space (x0, y0, z0).
    apix : float
        Voxel size (Å).
    r : int
        Kernel radius in voxels.

    Returns
    -------
    ccc : float
        Local cosine-style CCC in [-1, 1].
    """
    # coord -> voxel indices
    ix = int(round((atom_pos[0] - origin[0]) / apix))
    iy = int(round((atom_pos[1] - origin[1]) / apix))
    iz = int(round((atom_pos[2] - origin[2]) / apix))

    sz, sy, sx = field.shape

    z0, z1 = iz - r, iz + r + 1
    y0, y1 = iy - r, iy + r + 1
    x0, x1 = ix - r, ix + r + 1

    # require full kernel inside map
    if not (0 <= z0 and z1 <= sz and 0 <= y0 and y1 <= sy and 0 <= x0 and x1 <= sx):
        return 0.0

    sub = field[z0:z1, y0:y1, x0:x1]

    f = sub.astype(float).ravel()
    t = kernel.astype(float).ravel()

    num = np.dot(f, t)
    denom = math.sqrt(np.dot(f, f) * np.dot(t, t)) + 1e-12

    if denom < 1e-9:
        return 0.0

    ccc = num / denom
    # clamp for numerical safety
    if ccc > 1.0:
        ccc = 1.0
    elif ccc < -1.0:
        ccc = -1.0
    return ccc

import numpy as np
import math

def local_ccc(atom_pos, field, kernel, origin, apix, r, bg_value=0.0):
    """
    Computes local Pearson CCC between a 3D field and a template kernel
    around an atom position, allowing edges by padding the field
    with a constant background value outside the map.

    Parameters
    ----------
    atom_pos : (3,) array-like
        Real-space (x, y, z) coordinates.
    field : np.ndarray (Z, Y, X)
        3D map.
    kernel : np.ndarray (2r+1, 2r+1, 2r+1)
        Template kernel.
    origin : (3,) array-like
        Map origin (x0, y0, z0).
    apix : float
        Voxel size.
    r : int
        Kernel radius (kernel shape should be (2r+1)^3).
    bg_value : float
        Background value to use outside the map (e.g. 0.0 or global mean).
    """
    # Convert atom_pos to grid indices
    ix = int(round((atom_pos[0] - origin[0]) / apix))
    iy = int(round((atom_pos[1] - origin[1]) / apix))
    iz = int(round((atom_pos[2] - origin[2]) / apix))

    sz, sy, sx = field.shape
    kz, ky, kx = kernel.shape  # expect (2r+1, 2r+1, 2r+1)

    # Ideal field window for centered kernel
    z0 = iz - r
    z1 = iz + r + 1
    y0 = iy - r
    y1 = iy + r + 1
    x0 = ix - r
    x1 = ix + r + 1

    # Allocate a buffer for the field patch (same size as kernel)
    patch_f = np.full((kz, ky, kx), bg_value, dtype=float)

    # Overlap of this window with the actual field
    z0_f = max(z0, 0)
    z1_f = min(z1, sz)
    y0_f = max(y0, 0)
    y1_f = min(y1, sy)
    x0_f = max(x0, 0)
    x1_f = min(x1, sx)

    if z0_f < z1_f and y0_f < y1_f and x0_f < x1_f:
        # Corresponding indices in the kernel buffer
        kz0 = z0_f - z0
        ky0 = y0_f - y0
        kx0 = x0_f - x0
        kz1 = kz0 + (z1_f - z0_f)
        ky1 = ky0 + (y1_f - y0_f)
        kx1 = kx0 + (x1_f - x0_f)

        # Copy overlapping region from field into patch_f
        patch_f[kz0:kz1, ky0:ky1, kx0:kx1] = field[z0_f:z1_f, y0_f:y1_f, x0_f:x1_f]

    # Flatten both field patch and kernel
    f = patch_f.ravel()
    t = kernel.astype(float).ravel()

    # Mean-center (Pearson CCC)
    f_mean = np.mean(f)
    t_mean = np.mean(t)
    f_centered = f - f_mean
    t_centered = t - t_mean

    num = np.dot(f_centered, t_centered)
    denom_f = np.dot(f_centered, f_centered)
    denom_t = np.dot(t_centered, t_centered)
    denom = math.sqrt(denom_f * denom_t) + 1e-12

    if denom < 1e-9:
        return 0.0

    ccc = num / denom

    # Clamp
    if ccc > 1.0:
        ccc = 1.0
    elif ccc < -1.0:
        ccc = -1.0

    return ccc




def local_ccc_ori(atom_pos, field, kernel, origin, apix, r):
    """
    Computes local Pearson Correlation Coefficient (CCC) between a 3D field
    and a template kernel around an atom position.
    """
    # Convert atom_pos (x,y,z) to grid indices (ix, iy, iz)
    # Note: apix is assumed to be a single value here based on your script
    ix = int(round((atom_pos[0] - origin[0]) / apix))
    iy = int(round((atom_pos[1] - origin[1]) / apix))
    iz = int(round((atom_pos[2] - origin[2]) / apix))

    # Extract subvolume centered at (ix,iy,iz)
    sz, sy, sx = field.shape

    # Define slice boundaries
    z0, z1 = iz - r, iz + r + 1
    y0, y1 = iy - r, iy + r + 1
    x0, x1 = ix - r, ix + r + 1

    # Check if the entire kernel volume is inside the map
    if not (0 <= z0 and z1 <= sz and 0 <= y0 and y1 <= sy and 0 <= x0 and x1 <= sx):
        return 0.0  # Return 0 if atom is too close to the edge

    # Index the field with (z, y, x) slices
    sub = field[z0:z1, y0:y1, x0:x1]

    # Flatten both the sub-volume from the field and the kernel
    f = sub.flatten()
    t = kernel.flatten()

    # Mean-center both vectors
    f_mean = np.mean(f)
    t_mean = np.mean(t)
    f_centered = f - f_mean
    t_centered = t - t_mean

    # Compute dot products for numerator and denominator
    numerator = np.dot(f_centered, t_centered)
    denominator = math.sqrt(np.dot(f_centered, f_centered) * np.dot(t_centered, t_centered)) + 1e-12

    if denominator < 1e-9: # If either vector is flat, correlation is 0
        return 0.0

    return numerator / denominator

def voxel_to_point(x_idx, y_idx, z_idx, origin, apix):
    """Converts voxel indices to real-world coordinates."""
    # Note: apix is assumed to be a single value here
    real_world_x = origin[0] + x_idx * apix
    real_world_y = origin[1] + y_idx * apix
    real_world_z = origin[2] + z_idx * apix
    return (real_world_x, real_world_y, real_world_z)
#--------protein precomp tools    
def compute_donor_acceptor_mask(protein_pdb_atom_ids, protein_atom_types, ligand_atom_types):
    """
    Returns a 2D boolean array of shape (n_protein_atoms, n_ligand_atoms),
    where True indicates a potential donor–acceptor hydrogen bond pair 
    between the protein atom [i] and the ligand atom [j].
    """
    n_protein_atoms = len(protein_atom_types)
    n_ligand_atoms = len(ligand_atom_types)
    
    mask = np.zeros((n_protein_atoms, n_ligand_atoms), dtype=bool)
    
    for i in range(n_protein_atoms):
        p_type = protein_atom_types[i]
        p_res = protein_pdb_atom_ids[i]
        
        
        for j in range(n_ligand_atoms):
            l_type = ligand_atom_types[j]
            
            # If protein is donor & ligand is acceptor, or vice versa.
            if (p_type in HBOND_DONOR_ATOM_IDXS) and (l_type in HBOND_ACCEPTOR_ATOM_IDXS) or (p_type in HBOND_ACCEPTOR_ATOM_IDXS) and (l_type in HBOND_DONOR_ATOM_IDXS):
            
            #if is_protein_atom_donor(p_res[0], p_res[1]) and (l_type in HBOND_ACCEPTOR_ATOM_IDXS) or is_protein_atom_acceptor(p_res[0], p_res[1]) and (l_type in HBOND_DONOR_ATOM_IDXS):
            
                mask[i, j] = True
    
        
    return mask


def filter_points_and_adjacency_by_radius(
    points_indices,
    adjacency,
    origin,
    apix,
    center_point,
    radius
):
    """
    Filters grid indices and a corresponding adjacency list based on distance.

    This function takes a list of grid point indices and a matching adjacency
    list, and removes all entries that are outside a specified real-space
    radius from a center point. It is optimized for performance using NumPy's 
    vectorized operations.

    Args:
        points_indices (np.ndarray): 
            An array of shape (N, 3) containing the grid indices (z, y, x)
            to be filtered.

        adjacency (list[list[int]]):
            A list of lists representing the adjacency information. It must have
            a length of N, where each inner list corresponds to a point in
            `points_indices`.

        origin (np.ndarray or list/tuple): 
            The real-space coordinates (x, y, z) of the grid's origin.

        apix (np.ndarray or list/tuple): 
            The grid spacing (ax, ay, az) in Angstroms per pixel.

        center_point (np.ndarray or list/tuple): 
            The real-space coordinates (x, y, z) of the sphere's center.

        radius (float): 
            The distance cutoff in Angstroms. Points with a distance
            greater than this radius will be removed.

    Returns:
        tuple[np.ndarray, list[list[int]]]: 
            A tuple containing two elements:
            - A new array of shape (M, 3) with the filtered indices (M <= N).
            - A new list of lists of length M with the filtered adjacency info.
    """
    # 0. Basic validation to ensure inputs match
    if len(points_indices) != len(adjacency):
        raise ValueError(
            f"Input size mismatch: points_indices has {len(points_indices)} "
            f"entries, but adjacency has {len(adjacency)}."
        )

    # 1. Handle the edge case of an empty input array
    if len(points_indices) == 0:
        return np.array([], dtype=int).reshape(0, 3), []

    # 2. Ensure all inputs are NumPy arrays for vectorized math
    indices_arr = np.asarray(points_indices)
    origin_np = np.asarray(origin)
    apix_np = np.asarray(apix)
    center_point_np = np.asarray(center_point)

    # 3. Rearrange index columns from (z, y, x) to (x, y, z)
    indices_xyz = indices_arr[:, [2, 1, 0]]

    # 4. Convert all grid indices to real-space coordinates
    real_coords = origin_np + indices_xyz * apix_np

    # 5. Calculate squared distances from each point to the center
    squared_distances = np.sum((real_coords - center_point_np)**2, axis=1)

    # 6. Calculate the squared radius for efficient comparison
    radius_squared = radius**2

    # 7. Create a boolean mask where True indicates the point should be kept
    mask = squared_distances <= radius_squared

    # 8. Use the mask to filter both the NumPy array and the adjacency list
    filtered_indices = indices_arr[mask]
    
    # Use a list comprehension for efficient filtering of the adjacency list
    filtered_adjacency = [adj for adj, keep in zip(adjacency, mask) if keep]

    return filtered_indices, filtered_adjacency



def generate_fibonacci_sphere_vectors(samples=20):
    """
    Generates evenly distributed vectors on a sphere using the Fibonacci lattice method.
    Ensures the number of samples is even.
    """
    if samples % 2 != 0:
        samples += 1 # Ensure we have pairs of opposing vectors
        
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
        
    return np.array(points, dtype=np.float64)

def compute_encloure_grid_cpp(protein_positions,
                              protein_hpi, 
                              binding_site_map,
                              origin,
                              apix, 
                              samples=20):
    
    grid_shape = binding_site_map.shape
    hydrophobic_indices = protein_hpi > 0
    probe_vectors = generate_fibonacci_sphere_vectors(samples=20)
    
    enclosure_grid = grid_maps.compute_enclosure_grid(
        protein_positions.astype(np.float64),
        protein_hpi.astype(np.float64),
        binding_site_map,
        origin,
        apix,
        probe_vectors,
        ray_cutoff=4.0,
        axis_tolerance=2.0 # Allow atoms to be a bit off the direct ray path
    )
    
    smoothing_sigma = 1.0 
    smoothed_enclosure_grid = ndimage.gaussian_filter(enclosure_grid, sigma=smoothing_sigma)
    return smoothed_enclosure_grid


def get_protein_hpi(protein_structure):
    
    protein_hpi = []
    protein_positions = []
    for atom in protein_structure.atoms:
        if atom.element > 1.0:
            protein_hpi.append(get_xlogp(atom.residue.name, atom.name))
            protein_positions.append(np.array([atom.xx,atom.xy, atom.xz]))
            
    return  np.array(protein_positions), np.array(protein_hpi)
    
def get_xlogp(res, atom):
    try:
        return XlogP3().xlogp3_values[res][atom]
    except KeyError:
        return 0.0



def compute_protein_ring_types( residues, protein_positions):
    
    protein_rings = []
    protein_coords = []
    protein_heavy_idx = []
    for residue in residues:
        
        if residue.name in PROTEIN_RINGS:
            atom_names = PROTEIN_RINGS[residue.name]
            ring_types = RingType.from_residue_name(residue.name)
            ring_atom_coords, ring_atom_heavy_indices = get_protein_ring_atom_indices(residue, atom_names, protein_positions)
            
            #need to flatten as merged rings systems such as tryptophan are treated as 2 rings
            protein_heavy_idx += ring_atom_heavy_indices
            protein_rings += ring_types 
            protein_coords +=  ring_atom_coords
        
    
    protein_heavy_idx = [np.array(i, dtype=np.int32) for i in protein_heavy_idx]
    
    return protein_rings, protein_coords, protein_heavy_idx
    #prot ring
def get_protein_ring_atom_indices(residue, atom_names, protein_positions, tol=1e-8):
    
    atom_indices = []
    atom_heavy_idx = []
    
    for name, atom_ids in atom_names:
        indices = []
        heavy_idx = []
        for atom in residue.atoms:
            if atom.name in atom_ids:
                coord = np.array([atom.xx, atom.xy, atom.xz])
                indices.append(coord)
                
                matches = np.where(np.all(np.isclose(protein_positions, coord.reshape((1,3)), atol=tol), axis=1))[0]
                if matches:
                    heavy_idx.append(matches[0])
        
        atom_heavy_idx.append(heavy_idx)
        atom_indices.append(np.array(indices))
    return atom_indices, atom_heavy_idx




def resample_map(input_map, old_apix, new_apix, order=1):
    """
    Resample a 3D map to a new apix (voxel size) using interpolation.
    
    Parameters:
        input_map (np.ndarray): The 3D map (e.g., distance map, solvent map) to be resampled.
        old_apix (tuple): Original voxel dimensions (apix) as (x, y, z).
        new_apix (tuple): New desired voxel dimensions as (x, y, z).
        order (int): The order of the spline interpolation. Use order=0 for nearest-neighbor,
                     order=1 for linear (default), or higher for smoother interpolations.
                     
    Returns:
        np.ndarray: The resampled map.
    """
    # Calculate zoom factors for each dimension (old apix divided by new apix)
    zoom_factors = tuple(old / new for old, new in zip(old_apix, new_apix))
    
    # Resample the input map using the computed zoom factors
    resampled_map = ndimage.zoom(input_map, zoom=zoom_factors, order=order)
    
    return resampled_map



def compute_electostatic_grid_cpp( protein_positions, protein_charges,
                              binding_site_map,# np.array.shape #z,y,x
                              binding_site_origin, #x,y,z
                              apix,
                              c_factor = 332.06,
                              eps0=4.0,
                              k=0.2,
                              min_r=0.001):
    
    potential_grid = np.zeros(binding_site_map.shape, dtype=np.float64)
    potential_grid = grid_maps.compute_electostatic_grid_no_cutoff(protein_positions, 
                                                                   protein_charges,
                                                                   potential_grid,
                                                                   binding_site_origin,
                                                                   apix,
                                                                   c_factor, eps0, k, min_r)
    
    potential_grid = np.clip(potential_grid, -20, 20)
    return potential_grid




def expand_map(grid_map, old_apix, new_apix):
    
    
    old_apix = np.array(old_apix, dtype=float)
    new_apix = np.array(new_apix, dtype=float)
    old_shape = np.array(grid_map.shape, dtype=int)
    
    zoom_factors = old_apix / new_apix
    
    new_shape_float = old_shape * zoom_factors
    new_shape = np.round(new_shape_float).astype(int)

    # Create a new array of zeros with that shape
    new_grid_map = np.zeros(new_shape, dtype=np.float64)
    
    return new_grid_map



def compute_charges(protein_structure, protein_system):
    
    positions = protein_structure.positions.value_in_unit(unit.angstrom) 
    positions = np.array(positions)
    
    nonbonded_force = None
    for force in protein_system.getForces():
        if isinstance(force, NonbondedForce):
            nonbonded_force = force
            break
    if nonbonded_force is None:
        raise ValueError("No NonbondedForce found in the system.")
    
    charges = np.zeros(len(positions), dtype=np.float64)
    
    for i in range(len(positions)):
        q, sig, eps = nonbonded_force.getParticleParameters(i)
        charges[i] = q.value_in_unit(unit.elementary_charge)  # e
    
    return positions, charges
    



def get_protein_hydrogen_reference(mol):
    hydrogen_ref = []
    molHs = Chem.AddHs(mol,addCoords=True)
    positions = molHs.GetConformer().GetPositions()
    for atom in molHs.GetAtoms():
        if atom.GetSymbol() != 'H':
            nei_Hs = [i.GetIdx() for i in atom.GetNeighbors() if i.GetSymbol() == 'H']
            nei_Hs = [positions[i] for i in nei_Hs]
            hydrogen_ref.append( nei_Hs )
    return hydrogen_ref

def centroid_and_radius(points):
    """
    Parameters
    ----------
    points : array-like, shape (N, 3)
        Collection of 3-D Cartesian coordinates.

    Returns
    -------
    centroid : ndarray shape (3,)
        Mean (x, y, z) of all points.
    radius : float
        max ‖point − centroid‖₂ : the minimum radius that, centred on the
        centroid, encloses every point.
    """
    pts = np.asarray(points, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("Input must be an (N, 3) array of 3-D points.")

    centroid = pts.mean(axis=0)
    radius = np.linalg.norm(pts - centroid, axis=1).max()
    return centroid, radius


def kmeans_elbow_plot(
        X_proc,
        k_range=range(1, 11),
        *,
        init="k-means++",
        n_init=10,        # works on any scikit-learn version
        max_iter=300,
        random_state=42,
        show=True):
    
    ks, inertias = [], []

    for k in k_range:
        model = KMeans(n_clusters=k,
                       init=init,
                       n_init=n_init,
                       max_iter=max_iter,
                       random_state=random_state)
        model.fit(X_proc)
        ks.append(k)
        inertias.append(model.inertia_)

    # ----- plot -----
    if show:
        plt.figure(figsize=(6, 4))
        plt.plot(ks, inertias, marker="o")
        plt.xticks(ks)
        plt.xlabel("Number of clusters k")
        plt.ylabel("Inertia (WCSS)")
        plt.title("K-means elbow plot")
        plt.tight_layout()
        plt.show()

    
    kl = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
    elbow_k = int(kl.elbow) if kl.elbow else None
    
    
    final_km = KMeans(
        n_clusters=elbow_k,
        n_init=10,
        random_state=random_state,
        init="k-means++",
        max_iter=300,
    ).fit(X_proc)
    labels = final_km.labels_
    
    
    return elbow_k, labels, inertias

def covert_idx_to_coords(coords, origin, apix):
    coords_xyz = origin + coords[:, ::-1] * apix 
    return coords_xyz


def get_role_int(res, atom_id):
    role = 0
    
    if is_protein_atom_donor(res, atom_id):
        role += 1
    if is_protein_atom_acceptor(res, atom_id):
        role += 2
    
    return role

def get_valid_points_and_adjacency(site_mask, connectivity=6):
    """
    Given a 3D NumPy array 'site_mask' of shape (Nx, Ny, Nz) where True 
    indicates a valid point in the binding site, returns:

    1. valid_points: a list of (i, j, k) tuples for each True cell
    2. adjacency: a list of neighbor indices for each point
    
    Args:
        site_mask (np.ndarray): A boolean 3D array of shape (Nx, Ny, Nz).
        connectivity (int): 6 or 26, indicating how many directions 
                            we treat as neighbors in 3D.
                            6-connected => up, down, left, right, front, back
                            26-connected => includes diagonals as well.
                            
    Returns:
        valid_points (list[tuple]): List of (i, j, k) for all True cells.
        adjacency (list[list[int]]): adjacency[i] is a list of indices 
                                     of neighbors of valid_points[i].
    """

    Nx, Ny, Nz = site_mask.shape

    # 1. Collect all valid points
    valid_points = []
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if site_mask[i, j, k]:
                    valid_points.append((i, j, k))
    
    # 2. Map each valid point to an index
    point_to_index = {}
    for idx, (i, j, k) in enumerate(valid_points):
        point_to_index[(i, j, k)] = idx
    
    # 3. Build adjacency list
    adjacency = [[] for _ in range(len(valid_points))]

    # A. Define neighbor offsets
    #    6-connected => up to 6 neighbors (±1 in each axis), no diagonals.
    #    26-connected => includes diagonals in 3D.
    if connectivity == 6:
        # Up to 6 neighbors:
        # (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
        neighbor_offsets = [
            ( 1,  0,  0), (-1,  0,  0),
            ( 0,  1,  0), ( 0, -1,  0),
            ( 0,  0,  1), ( 0,  0, -1),
        ]
    elif connectivity == 26:
        # All combinations of -1,0,+1 in 3D except (0,0,0)
        neighbor_offsets = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if not (di == 0 and dj == 0 and dk == 0):
                        neighbor_offsets.append((di, dj, dk))
    else:
        raise ValueError("connectivity must be 6 or 26.")
    
    # B. For each valid point, check neighbors
    for idx, (i, j, k) in enumerate(valid_points):
        for (di, dj, dk) in neighbor_offsets:
            ni, nj, nk = i + di, j + dj, k + dk
            # Must be in bounds and True in site_mask
            if 0 <= ni < Nx and 0 <= nj < Ny and 0 <= nk < Nz:
                if site_mask[ni, nj, nk]:
                    # This neighbor is also valid => record adjacency
                    neighbor_idx = point_to_index[(ni, nj, nk)]
                    adjacency[idx].append(neighbor_idx)

    return valid_points, adjacency

#--------ligand precomp tools


def compute_halogen_bond_data(mol,atom_types, valid_atom_types = HALOGEN_DONOR_ATOM_IDXS):
    
    ligand_donor_indices = []
    ligand_donor_root_indices = []
    
    # Identify halogen bond donors in the ligand (e.g., Cl, Br, I)
    for atom, atom_type in zip(mol.GetAtoms(), atom_types):
        if atom_type in valid_atom_types:
            donor_idx = atom.GetIdx()
            # Get heavy (non-hydrogen) neighbors; assume first is the donor root
            heavy_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() != 'H']
            donor_root_idx = heavy_neighbors[0].GetIdx() if heavy_neighbors else None
            
            ligand_donor_indices.append(donor_idx)
            ligand_donor_root_indices.append(donor_root_idx)
    
    
    return np.array(ligand_donor_indices), np.array(ligand_donor_root_indices)
    


def get_ligand_hydrogen_reference(mol):
    molHs = Chem.AddHs(mol,addCoords=True)
    hydrogen_ref = []
    for atom in molHs.GetAtoms():
        if atom.GetSymbol() != 'H':
            nei_hs = [i.GetIdx() for i in atom.GetNeighbors() if i.GetSymbol() == 'H']
            hydrogen_ref.append( np.array(nei_hs, dtype = np.int32))
    return hydrogen_ref



def get_vdw_radius(element_symbol):
    pt = Chem.GetPeriodicTable()
    try:
        return pt.GetRvdw(element_symbol)
    except:
        return 1.8



def fragment_ligand(mol):
    """
    Fragment the molecule using BRICS bonds.
    
    Returns a list of fragment molecules that retain the original 3D coordinates.
    """
    # Fragment on BRICS bonds. This returns a new molecule with dummy atoms at break points.
    frag_mol = Chem.FragmentOnBRICSBonds(mol)
    # Get individual fragments as separate molecules. 'asMols=True' preserves separate fragments.
    frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
    atom_mapping = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=True)
    return frags, atom_mapping



def get_hydrophobic_groups( mol):
    fragments, atom_mapping = fragment_ligand(mol)
    
    fragment_data = {}
    for idx, (frag, atom_map) in enumerate(zip(fragments, atom_mapping)):
        logp_frag = compute_fragment_logP(frag)
        if logp_frag > 0.0:
            
            fragment_data[idx] = (atom_map, logp_frag)
        
    return fragment_data

def compute_fragment_logP(frag):
    """
    Compute the Crippen logP for the fragment.
    """
    return Crippen.MolLogP(frag)

def get_periodic_torsion_force(system):
    for force in system.getForces():
        if type(force).__name__ == 'PeriodicTorsionForce':
             return force,  force.getForceGroup() 

def get_force_energy(force, simulation):
    #state = simulation.context.getState(getEnergy=True)
    state = simulation.context.getState(getEnergy=True, groups={ force.getForceGroup() })
    energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    return energy

def export_torsion_profile(ligand,
                           torsion_lists,
                           platform,
                           output = './',
                           normalise = True,
                           write = False
                           ):
    
    integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picoseconds, 2*unit.femtoseconds)
    _platform = Platform.getPlatformByName(platform)
    system = ligand.complex_structure.createSystem()
    
    
    
    simulation = app.Simulation(ligand.complex_structure.topology, 
                            system, integrator, _platform)
    
    simulation.context.setPositions(ligand.complex_structure.positions)
    force_group = 0
    for force in system.getForces():
        force.setForceGroup(force_group)
        force_group += 1
        
    torsion_force, force_index = get_periodic_torsion_force(system)
    lig_copy = Chem.AddHs(ligand.mol, addCoords = True)
    all_torsion_profiles = []
    for torsion in torsion_lists:
        
        torsion_profile = []
        a1,a2,a3,a4 = torsion 
        
        atom_names = []
        
        for atom in ligand.complex_structure.atoms:
            if atom.idx == a1:
                a1_atom = atom 
            elif atom.idx == a2:
                a2_atom = atom 
            elif atom.idx == a3:
                a3_atom = atom 
            elif atom.idx == a4:
                a4_atom = atom
            
            
        for angle in range(0,360,1):
            
            rdMolTransforms.SetDihedralDeg(lig_copy.GetConformer(), a1,a2,a3,a4, angle)
            new_angle = rdMolTransforms.GetDihedralDeg(lig_copy.GetConformer(), a1,a2,a3,a4)
            positions = lig_copy.GetConformer().GetPositions() * unit.angstrom
            simulation.context.setPositions(positions)
            force_energy = get_force_energy(torsion_force, simulation)
            torsion_profile.append((int(new_angle), force_energy))
        
            
        #plt.plot([i[0] for i in torsion_profile], [i[1] for i in torsion_profile])
        
        
        min_energy = min([i[1] for i in torsion_profile])
        torsion_profile = [(round(i[0],3) , i[1] - min_energy ) for i in torsion_profile]
        max_energy = max([i[1] for i in torsion_profile])
        
        torsion_profile = [(round(i[0],3) , i[1] / max_energy ) for i in torsion_profile]
        
        # Step 3: Rescale to -1 to 1
        #torsion_profile = [(round(i[0]), 2 * i[1] - 1) for i in torsion_profile]
        
        all_torsion_profiles.append([[a1_atom.name, a2_atom.name, a3_atom.name, a4_atom.name ],
                                     [a1_atom.idx, a2_atom.idx, a3_atom.idx,a4_atom.idx],
                                     torsion_profile])
    
    
    return all_torsion_profiles


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


def get_torsion_lists_dup(mol):
    '''


    Parameters
    ----------
    mol : TYPE
        DESCRIPTION.

    Returns
    -------
    return_torsions : TYPE
        DESCRIPTION.

    '''
    mol = Chem.RemoveHs(mol)
    torsion_list = tfp.CalculateTorsionLists(mol)[0]
    return_torsions = []
    for tor in torsion_list:

        tor = tor[0]
        for t in tor:
            return_torsions.append(t)
            t = t[::-1]
            return_torsions.append(t)  # trying with just one set of torsions 
    
    
    
    
    return return_torsions

def get_torsion_lists(mol):
    '''


    Parameters
    ----------
    mol : TYPE
        DESCRIPTION.

    Returns
    -------
    return_torsions : TYPE
        DESCRIPTION.

    '''
    torsion_list = tfp.CalculateTorsionLists(mol)[0]
    return_torsions = []
    for tor in torsion_list:

        tor = tor[0]
        for t in tor:
            return_torsions.append(t)
            #t = t[::-1]
            #return_torsions.append(t)  # trying with just one set of torsions 
    
    donor_torsions = get_donor_h_torsions(mol)
   
    movedhs = set()
    for t in donor_torsions:
        if only_h_moves_on_rotation(mol, t):
            if t not in return_torsions:
                if t[3] not in movedhs:
                    return_torsions.append(t)
                    movedhs.add(t[3])
   
   
    return return_torsions


def get_donor_h_torsions(molH):
    """
    Enumerate all torsions around rotatable N/O–heavy bonds that carry
    explicit H's on the donor.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule (implicit H’s OK).

    Returns
    -------
    torsions : list of 4‐tuples of atom indices
        Each tuple is (A, B, C, D) where B–C is the rotatable bond
        and D is a hydrogen on the donor atom C.
    """
    
    # 2) SMARTS for a “rotatable bond” (Daylight‐style) from the Lipinski module
    rot_smarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
    rot_pat = Chem.MolFromSmarts(rot_smarts)  # :contentReference[oaicite:1]{index=1}

    torsions = []
    for i, j in molH.GetSubstructMatches(rot_pat):
        a1 = molH.GetAtomWithIdx(i)
        a2 = molH.GetAtomWithIdx(j)

        # decide which end is the H‐bond donor (N, S or O with an H neighbor)
        if a1.GetAtomicNum() in (7, 8, 16) and any(n.GetAtomicNum() == 1 for n in a1.GetNeighbors()):
            donor, pivot = a1, a2
        elif a2.GetAtomicNum() in (7, 8, 16) and any(n.GetAtomicNum() == 1 for n in a2.GetNeighbors()):
            donor, pivot = a2, a1
        else:
            continue

        
        for nbr in pivot.GetNeighbors():
            if nbr.GetIdx() == donor.GetIdx() or nbr.GetAtomicNum() == 1:
                continue
            A = nbr.GetIdx()
            B = pivot.GetIdx()
            C = donor.GetIdx()

            # build the dihedral A–B–C–D
            for hb in donor.GetNeighbors():
                if hb.GetAtomicNum() == 1:
                    D = hb.GetIdx()
                    torsions.append((A, B, C, D))
    return torsions

def only_h_moves_on_rotation(mol: Chem.Mol,
                             torsion: tuple[int,int,int,int],
                             angle_deg: float = 30.0,
                             tol: float = 1e-4) -> bool:
    """
    Rotate the dihedral defined by `torsion = (i,j,k,l)` by `angle_deg` degrees
    and return True if *only* atom `l` moved (within tolerance `tol`).

    Parameters
    ----------
    mol : Chem.Mol
        RDKit Mol with an embedded Conformer and explicit H atoms.
    torsion : tuple of int
        Four atom indices (i,j,k,l), where (j–k) is the rotatable bond
        and `l` is the H-atom on the donor.
    angle_deg : float, optional
        How many degrees to *add* to the current dihedral angle (default 30°).
    tol : float, optional
        Distance threshold (in Å) below which an atom is considered unmoved.

    Returns
    -------
    bool
        True if *only* atom `l` moved by more than `tol`, False otherwise.
    """
    
    mol_copy = Chem.Mol(mol)
    conf = mol_copy.GetConformer()

    n = conf.GetNumAtoms()
    before = [conf.GetAtomPosition(a) for a in range(n)]
    old_ang = rdMolTransforms.GetDihedralDeg(conf, *torsion)
    rdMolTransforms.SetDihedralDeg(conf, *torsion, old_ang + angle_deg)
    after = [conf.GetAtomPosition(a) for a in range(n)]

    # detect which atoms moved by > tol
    moved = []
    for pb, pa in zip(before, after):
        dx = pa.x - pb.x
        dy = pa.y - pb.y
        dz = pa.z - pb.z
        dist2 = dx*dx + dy*dy + dz*dz
        moved.append(dist2 > tol*tol)

    return moved.count(True) == 1 and moved[torsion[3]]


def find_best_ring_bond_to_break_(mol, ring_info=None):
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


import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType as HT

def _best_fit_plane_rms(conf, atom_ids):
    # returns (rms_distance_to_plane, max_abs_distance)
    pts = []
    for i in atom_ids:
        p = conf.GetAtomPosition(int(i))
        pts.append([p.x, p.y, p.z])
    pts = np.asarray(pts, dtype=float)

    ctr = pts.mean(axis=0)
    u, s, vh = np.linalg.svd(pts - ctr, full_matrices=False)
    normal = vh[-1]
    d = (pts - ctr) @ normal
    return float(np.sqrt(np.mean(d * d))), float(np.max(np.abs(d)))

def _amide_like_bonds(mol):
    """
    Returns a set of bond indices for common 'amide-like' linkages you almost never
    want to break for ring-flex (amide/urea/carbamate/sulfonamide-ish).
    """
    patt_smarts = [
        "[CX3](=[OX1])[NX3]",                 # amide / lactam / carbamate / urea C(=O)-N
        "[SX4](=[OX1])(=[OX1])[NX3]",         # sulfonamide S(=O)2-N
        "[PX4](=[OX1])([OX1])[NX3]",          # phosphoramidate-like (rare)
    ]
    out = set()
    for sm in patt_smarts:
        q = Chem.MolFromSmarts(sm)
        if q is None:
            continue
        for match in mol.GetSubstructMatches(q):
            # match includes at least C and N (or S and N). find the bond between first and last atom in match
            a = match[0]
            n = match[-1]
            b = mol.GetBondBetweenAtoms(int(a), int(n))
            if b is not None:
                out.add(b.GetIdx())
    return out

def find_best_ring_bond_to_break(
    mol,
    *,
    min_ring_size=5,
    min_sp3_fraction=0.70,
    planar_rms_cutoff=0.10,
    planar_maxdev_cutoff=0.20,
    confId=-1
):
    """
    Pick one breakable bond per eligible (simple) non-aromatic ring.

    Key guards:
      - ring is not fused/bridged/spiro: all ring atoms AND ring bonds must be in exactly 1 ring
      - ring is 'aliphatic enough': mostly SP3 atoms
      - candidate bond is single, non-conjugated, non-amide-like
    """
    ri = mol.GetRingInfo()
    if ri is None:
        return []

    atom_rings = list(ri.AtomRings())
    bond_rings = list(ri.BondRings())
    if not atom_rings:
        return []

    conf = None
    if mol.GetNumConformers() > 0:
        try:
            conf = mol.GetConformer(confId)
        except Exception:
            conf = mol.GetConformer()

    amide_bonds = _amide_like_bonds(mol)

    chosen = []

    for ring_atoms, ring_bonds in zip(atom_rings, bond_rings):
        if len(ring_atoms) < min_ring_size:
            continue

        #-----skip anything aromatic / partially aromatic----
        if any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring_atoms):
            continue
        if any(mol.GetBondWithIdx(b).GetIsAromatic() for b in ring_bonds):
            continue

        #-----skip fused/bridged/spiro ring systems-----
    
        if any(ri.NumAtomRings(a) != 1 for a in ring_atoms):
            continue
        if any(ri.NumBondRings(b) != 1 for b in ring_bonds):
            continue

        # -----require mostly SP3 ring atoms (rigidity heuristic) ---
        '''
        sp3_count = 0
        for a in ring_atoms:
            at = mol.GetAtomWithIdx(a)
            if (at.GetHybridization() == HT.SP3) and (not at.GetIsAromatic()):
                sp3_count += 1
        sp3_frac = sp3_count / float(len(ring_atoms))
        if sp3_frac < float(min_sp3_fraction):
            continue
        '''

        # -----Skip plannar rings-----
        if conf is not None:
            rms, mx = _best_fit_plane_rms(conf, ring_atoms)
            if (rms <= planar_rms_cutoff) and (mx <= planar_maxdev_cutoff):
                continue

        # --- Candidate bonds inside this ring ---
        candidates = []
        for bidx in ring_bonds:
            b = mol.GetBondWithIdx(bidx)

            if b.GetBondType() != Chem.BondType.SINGLE:
                continue
            if b.GetIsAromatic():
                continue
            if b.GetIsConjugated():  # RDKit exposes this directly :contentReference[oaicite:4]{index=4}
                continue
            if bidx in amide_bonds:
                continue

            a1 = mol.GetAtomWithIdx(b.GetBeginAtomIdx())
            a2 = mol.GetAtomWithIdx(b.GetEndAtomIdx())

            # keep breaks in the truly flexible part: both ends should be SP3
            if a1.GetHybridization() != HT.SP3 or a2.GetHybridization() != HT.SP3:
                continue

            # score: prefer C–C, then C–X; avoid "busy" atoms a bit
            score = 0
            if a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 6:
                score += 100
            elif a1.GetAtomicNum() == 6 or a2.GetAtomicNum() == 6:
                score += 20
            score -= (a1.GetTotalDegree() + a2.GetTotalDegree())  # mild preference for less substituted

            candidates.append((score, bidx))

        if not candidates:
            continue

        candidates.sort(reverse=True)
        chosen.append(candidates[0][1])

    return sorted(set(chosen))


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

def get_ligand_heavy_atom_indexes( mol):
    return np.array([atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != 'H'])

def get_hbond_role_int(atom_type):
    """
    Returns 'donor', 'acceptor', 'both', or 'none' based on membership in 
    HBOND_DONOR_ATOM_IDXS and HBOND_ACCEPTOR_ATOM_IDXS.
    """
    is_donor = (atom_type in HBOND_DONOR_ATOM_IDXS)
    is_acceptor = (atom_type in HBOND_ACCEPTOR_ATOM_IDXS)
    
    if is_donor and is_acceptor:
        return 3
    elif is_donor:
        return 1
    elif is_acceptor:
        return 2
    else:
        return 0

def get_exclusion_mask(mol, heavy_atom_indexes):
    '''
    Gets an exclusion mask that is True for atom indixes that are 0, 1 or 2 bonded atoms away. 
    True indicates atoms should be excluded from local scoreing 
    
    '''
    n_atoms = len(heavy_atom_indexes)
    exclusion_mask = np.zeros((n_atoms, n_atoms), dtype=bool)
    idx_map = {idx: i for i, idx in enumerate(heavy_atom_indexes)}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in idx_map:
            i = idx_map[idx]
            excluded_indices = get_intra_exclusion(atom)
            for excl_idx in excluded_indices:
                if excl_idx in idx_map:
                    j = idx_map[excl_idx]
                    exclusion_mask[i, j] = True
                    exclusion_mask[j, i] = True  # Symmetric
    # Exclude self-interactions
    np.fill_diagonal(exclusion_mask, True)
    return exclusion_mask

def get_intra_exclusion(atom):
    exclusion_indexes = set()
    for neighbor in atom.GetNeighbors():
        exclusion_indexes.add(neighbor.GetIdx())
        for neighbor2 in neighbor.GetNeighbors():
            exclusion_indexes.add(neighbor2.GetIdx())
    return list(exclusion_indexes)

def compute_bond_distances(mol, heavy_atom_indices):
    """
    Returns a 2D array of shape (n_heavy, n_heavy) giving the shortest 
    bond-path distance between each pair of heavy atoms in 'mol'.
    If there's no path (shouldn't happen within one ligand), set it high.
    """
    n_heavy = len(heavy_atom_indices)
    # Map each heavy atom index (RDKit) to a row/col in the output
    idx_map = {rd_idx: i for i, rd_idx in enumerate(heavy_atom_indices)}

    # Initialize with some large value
    bond_distances = np.full((n_heavy, n_heavy), fill_value=999, dtype=np.int32)
    # Distance to self is 0
    np.fill_diagonal(bond_distances, 0)

    # Build adjacency list for heavy atoms only
    adjacency = {rd_idx: [] for rd_idx in heavy_atom_indices}
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        # Only keep edges if both are heavy
        if a1 in idx_map and a2 in idx_map:
            adjacency[a1].append(a2)
            adjacency[a2].append(a1)

    # BFS for each heavy atom
    for start_idx in heavy_atom_indices:
        start_row = idx_map[start_idx]
        visited = {start_idx: 0}  # distance from start
        queue = deque([start_idx])
        while queue:
            curr = queue.popleft()
            curr_dist = visited[curr]
            for nbr in adjacency[curr]:
                if nbr not in visited:
                    visited[nbr] = curr_dist + 1
                    queue.append(nbr)
                    # Update bond_distances
                    bond_distances[start_row, idx_map[nbr]] = curr_dist + 1
                    bond_distances[idx_map[nbr], start_row] = curr_dist + 1
    return bond_distances





def compute_electrostatic_grid_cutoff_cpp(
    protein_positions,
    protein_charges,
    binding_site_map_shape,
    binding_site_origin,
    apix,
    c_factor=332.06,
    min_r=0.5,
    cutoff=12.0,
):
    pos   = np.asarray(protein_positions, dtype=float)
    q     = np.asarray(protein_charges,   dtype=float)
    shape = tuple(int(x) for x in binding_site_map_shape)
    origin= np.asarray(binding_site_origin, dtype=float)
    apix  = np.asarray(apix, dtype=float)
    
    grid = grid_maps.compute_electrostatic_grid_cutoff_cpp(
        pos, q, shape, origin, apix,
        float(c_factor),
        float(min_r),
        float(cutoff),
    )
    grid = np.clip(grid, -20, 20)
    return grid

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
            
            elements.append(atom.element.symbol)
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
    heavy_indices = [i for i, e in enumerate(elements) if e.upper() != "H"]
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



def make_spherical_kernel(radius_angstrom, spacing):
    """
    Create a 3D spherical kernel (hard sphere) in voxel units.

    Parameters
    ----------
    radius_angstrom : float
        Radius of the sphere in Å (e.g. vdW radius).
    spacing : float
        Grid spacing in Å (assumed isotropic).

    Returns
    -------
    kernel : 3D np.ndarray of float32
        Ones inside the sphere, zeros outside.
    """
    r_vox = radius_angstrom / float(spacing)
    half = int(np.ceil(r_vox))

    # voxel offsets
    dz = np.arange(-half, half + 1)
    dy = np.arange(-half, half + 1)
    dx = np.arange(-half, half + 1)

    zz, yy, xx = np.meshgrid(dz, dy, dx, indexing="ij")
    dist2_vox = xx**2 + yy**2 + zz**2

    kernel = (dist2_vox <= r_vox**2).astype(np.float32)
    return kernel





def make_enclosure_grids_for_atom_types(
        sasa_mask,
        spacing,
        vdw_radii_by_type,
        weight_grid=None
):
    """
    For each atom type, build an 'enclosure grid' by summing values
    in a hard sphere neighborhood whose radius is that atom's vdW radius.

    Parameters
    ----------
    sasa_mask : (nz,ny,nx) bool
        True where SASA / allowed solvent voxels.
    spacing : float
        Grid spacing in Å.
    vdw_radii_by_type : dict
        Mapping like {"C": 1.7, "N": 1.55, "O": 1.52, ...} in Å.
    weight_grid : (nz,ny,nx) float or None
        If None, enclosure = count of SASA voxels inside the sphere.
        If given, enclosure = sum of weight_grid values inside the sphere.
        Typical choice: propagated hydrophobic/logP field.

    Returns
    -------
    enclosure_grids : dict
        {atom_type: enclosure_grid}, each (nz,ny,nx) float32.
        Values outside SASA mask are set to 0.
    """
    sasa_mask = np.asarray(sasa_mask, dtype=bool)
    nz, ny, nx = sasa_mask.shape

    if weight_grid is None:
        base = sasa_mask.astype(np.float32)
    else:
        base = np.asarray(weight_grid, dtype=np.float32)
        # only care about SASA region
        base = base * sasa_mask.astype(np.float32)

    enclosure_grids = {}

    for atype, r_vdw in vdw_radii_by_type.items():
        kernel = make_spherical_kernel(radius_angstrom=r_vdw, spacing=spacing)

        # Convolve base (either 1 per SASA voxel, or hydrophobic field)
        enc = signal.convolve(
            base,
            kernel,
            mode="same",
            method="fft",
        ).astype(np.float32)
        

        # Force outside-SASA to zero so it's only meaningful where ligand can go
        enc[~sasa_mask] = 0.0

        enclosure_grids[atype] = enc
    
    return enclosure_grids


def propagate_logp_exp_decay_cpp(logp_centers,
                             logp_values,
                             origin,
                             grid_shape,
                             spacing,
                             sasa_mask=None,
                             cutoff=6.0):
    """
    Python wrapper around the C++ implementation.
    """
    centers = np.asarray(logp_centers, dtype=float)
    values  = np.asarray(logp_values,  dtype=float)
    origin  = np.asarray(origin,       dtype=float)
    nz, ny, nx = grid_shape

    if sasa_mask is None:
        field = grid_maps.propagate_logp_exp_decay_cpp(
            centers,
            values,
            origin,
            (int(nz), int(ny), int(nx)),
            float(spacing),
            None,
            float(cutoff),
        )
    else:
        sasa = np.asarray(sasa_mask, dtype=np.uint8)
        field = grid_maps.propagate_logp_exp_decay_cpp(
            centers,
            values,
            origin,
            (int(nz), int(ny), int(nx)),
            float(spacing),
            sasa,
            float(cutoff),
        )

    return field

def smooth_env_map_diffusion(env_index: np.ndarray,
                             solvent_mask: np.ndarray,
                             n_iter: int = 5,
                             alpha: float = 0.15):
    """
    Smooth env_index inside solvent_mask using simple diffusion / Laplacian smoothing.

    env_index    : float (Z,Y,X), e.g. depth_norm * constriction in [0,1].
    solvent_mask : bool  (Z,Y,X), True where solvent exists (SASA region).
    n_iter       : number of diffusion iterations (3–10 is usually enough).
    alpha        : diffusion coefficient (<= 1/6 for stability in 3D).

    Returns
    -------
    env_smooth : float32 (Z,Y,X), smoothed env_index, zero outside solvent.
    """

    env = np.asarray(env_index, dtype=np.float32).copy()
    mask = np.asarray(solvent_mask, dtype=bool)

    # Zero outside solvent to avoid nonsense leaking in
    env[~mask] = 0.0

    # Simple explicit diffusion scheme:
    # env_{t+1} = env_t + alpha * (sum(neighbors) - 6 * env_t)
    for _ in range(n_iter):
        # 6-neighbor values (periodic at box edges; usually fine if mask is interior)
        up    = np.roll(env,  1, axis=0)
        down  = np.roll(env, -1, axis=0)
        north = np.roll(env,  1, axis=1)
        south = np.roll(env, -1, axis=1)
        east  = np.roll(env,  1, axis=2)
        west  = np.roll(env, -1, axis=2)

        lap = (up + down + north + south + east + west - 6.0 * env)

        env = env + alpha * lap
        # Re-impose solvent region as Dirichlet-ish boundary
        #env[~mask] = 0.0

    # Optional: clip back to [0,1]
    env = np.clip(env, 0.0, 1.0).astype(np.float32)
    return env


def _widest_path_cpp(r_local, solvent_mask, seed_mask, connectivity=6, power=4):
    r_local = np.asarray(r_local, dtype=np.float32)
    solvent_mask = np.asarray(solvent_mask, dtype=np.uint8)
    seed_mask = np.asarray(seed_mask, dtype=np.uint8)

    R_bottle, flow_norm = grid_maps.widest_path_cpp(
        r_local,
        solvent_mask,
        seed_mask,
        int(connectivity),
        int(power),
    )
    return R_bottle, flow_norm


def compute_flow_widest_path_interface_seeds_cpp(
    protein_mask: np.ndarray,
    bulk_mask: np.ndarray,
    sasa_mask: np.ndarray,
    apix: float = 1.0,
    connectivity: int = 6,
    power: int = 4
):
    """
    Version where the search is restricted to the internal SASA region,
    and seeds are the SASA voxels that touch bulk (the interface).

    protein_mask : bool (Z,Y,X)
        True where protein/SES is.
    bulk_mask : bool (Z,Y,X)
        True for the bulk solvent region (outside).
    sasa_mask : bool (Z,Y,X)
        Internal solvent-accessible region that starts at the bulk interface.
    """

    assert protein_mask.shape == bulk_mask.shape == sasa_mask.shape
    nz, ny, nx = protein_mask.shape

    protein_mask = np.asarray(protein_mask, dtype=bool)
    bulk_mask    = np.asarray(bulk_mask,    dtype=bool)
    sasa_mask    = np.asarray(sasa_mask,    dtype=bool)

    # 1. Solvent region = internal SASA
    solvent_mask = sasa_mask

    # 2. Seeds: SASA voxels that touch bulk
    if connectivity == 6:
        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1, 1, 0] = True
        structure[1, 1, 2] = True
        structure[1, 0, 1] = True
        structure[1, 2, 1] = True
        structure[0, 1, 1] = True
        structure[2, 1, 1] = True
    elif connectivity == 26:
        structure = np.ones((3, 3, 3), dtype=bool)
        structure[1, 1, 1] = False
    else:
        raise ValueError("connectivity must be 6 or 26")
    
    
    bulk_dilated = binary_dilation(bulk_mask, structure=structure)
    # NOTE: this looks more consistent with the comment about "touching bulk":
    seed_mask = solvent_mask & bulk_dilated
    
    
    
    
    # 3. Local radius (EDT to protein / outside region)
    free_space = solvent_mask | bulk_mask
    r_local = distance_transform_edt(free_space).astype(np.float32)
    r_local *= sasa_mask.astype(np.float32)
    # If you want to factor in apix:
    # r_local *= float(apix)
    
    
    
    
    # 4. Widest-path via C++
    R_bottle, flow_norm = _widest_path_cpp(
        r_local,
        solvent_mask,
        seed_mask,
        connectivity=connectivity,
        power=power,
    )
    

    return r_local, R_bottle, flow_norm, seed_mask



def prepare_depth_for_mrc_cpp(depth_map, sasa_mask=None, fill_unreachable=None):
    depth = np.asarray(depth_map, dtype=np.float32)

    if sasa_mask is None and fill_unreachable is None:
        return grid_maps.prepare_depth_for_mrc_cpp(depth)

    if sasa_mask is None:
        # No SASA mask, only fill_unreachable
        return grid_maps.prepare_depth_for_mrc_cpp(depth, None, float(fill_unreachable))

    sasa = np.asarray(sasa_mask, dtype=np.uint8)

    if fill_unreachable is None:
        return grid_maps.prepare_depth_for_mrc_cpp(depth, sasa)
    else:
        return grid_maps.prepare_depth_for_mrc_cpp(depth, sasa, float(fill_unreachable))


def _depth_propagation_sub_cpp(bulk_mask_sub,
                               protein_mask_sub,
                               target_mask_sub,
                               spacing=1.0,
                               connectivity=6):
    """
    C++-accelerated replacement for _depth_propagation_sub.
    """
    bulk    = np.asarray(bulk_mask_sub,    dtype=np.uint8)
    protein = np.asarray(protein_mask_sub, dtype=np.uint8)
    target  = np.asarray(target_mask_sub,  dtype=np.uint8)

    depth = grid_maps.depth_propagation_cpp(
        bulk,
        protein,
        target,
        float(spacing),
        int(connectivity),
    )
    return depth

def compute_solvent_depth_map_morph_cpp(bulk_mask,
                                    protein_mask,
                                    spacing=1.0,
                                    sasa_mask=None,
                                    connectivity=6,
                                    crop=True,
                                    pad=3):
    bulk_mask    = np.asarray(bulk_mask,    dtype=bool)
    protein_mask = np.asarray(protein_mask, dtype=bool)
    nz, ny, nx = bulk_mask.shape

    if sasa_mask is not None:
        sasa_mask = np.asarray(sasa_mask, dtype=bool)
        target_mask = sasa_mask
    else:
        target_mask = ~protein_mask

    region = bulk_mask | target_mask

    if not np.any(region):
        depth = np.full((nz, ny, nx), np.nan, dtype=np.float32)
        return depth

    if crop:
        zz, yy, xx = np.where(region)
        z0 = max(int(zz.min()) - pad, 0)
        z1 = min(int(zz.max()) + pad + 1, nz)
        y0 = max(int(yy.min()) - pad, 0)
        y1 = min(int(yy.max()) + pad + 1, ny)
        x0 = max(int(xx.min()) - pad, 0)
        x1 = min(int(xx.max()) + pad + 1, nx)

        bulk_sub    = bulk_mask[z0:z1, y0:y1, x0:x1]
        protein_sub = protein_mask[z0:z1, y0:y1, x0:x1]
        target_sub  = target_mask[z0:z1, y0:y1, x0:x1]

        depth_sub = _depth_propagation_sub_cpp(
            bulk_sub,
            protein_sub,
            target_sub,
            spacing=spacing,
            connectivity=connectivity,
        )

        depth = np.full((nz, ny, nx), np.inf, dtype=np.float32)
        depth[z0:z1, y0:y1, x0:x1] = depth_sub
    else:
        depth = _depth_propagation_sub_cpp(
            bulk_mask,
            protein_mask,
            target_mask,
            spacing=spacing,
            connectivity=connectivity,
        )

    if sasa_mask is not None:
        depth[~sasa_mask] = np.nan

    return depth




def make_shell_kernel_R(R_sasa: float,
                        spacing: float,
                        shell_thickness: float = 1.0) -> np.ndarray:
    """
    Build a thin spherical shell kernel of radius R_sasa (Å),
    normalized so sum(kernel) ≈ 4π R_sasa^2 (Å^2).

    Parameters
    ----------
    R_sasa : float
        Shell radius in Å (here we'll use 3.0).
    spacing : float
        Grid spacing in Å.
    shell_thickness : float
        Thickness of the shell in Å (~0.5–1.0 works fine).

    Returns
    -------
    kernel : 3D np.ndarray (float32)
        Shell kernel.
    """
    R = float(R_sasa)
    spacing = float(spacing)

    R_vox = R / spacing                      # radius in voxels
    half_thick_vox = (shell_thickness / 2.0) / spacing

    r_vox = int(np.ceil(R_vox + half_thick_vox))
    size = 2 * r_vox + 1

    kernel = np.zeros((size, size, size), dtype=np.float32)
    cz = cy = cx = r_vox

    for dz in range(-r_vox, r_vox + 1):
        for dy in range(-r_vox, r_vox + 1):
            for dx in range(-r_vox, r_vox + 1):
                dist_vox = np.sqrt(dx*dx + dy*dy + dz*dz)
                dist_ang = dist_vox * spacing
                if abs(dist_ang - R) <= (shell_thickness / 2.0):
                    kernel[cz + dz, cy + dy, cx + dx] = 1.0

    shell_voxels = float(kernel.sum())
    if shell_voxels > 0.0:
        area_total = 4.0 * np.pi * R * R      # Å^2
        kernel *= (area_total / shell_voxels)

    return kernel


def build_delta_sasa_generic_grid(
    sasa_mask: np.ndarray,
    spacing: float,
    R_sasa: float = 3.0,
    shell_thickness: float = 1.0,
) -> np.ndarray:
    """
    Build a single generic ΔSASA grid using a shell radius R_sasa (Å).

    Parameters
    ----------
    sasa_mask : (nz,ny,nx) bool
        Protein SASA mask (True where solvent-accessible).
    spacing : float
        Grid spacing in Å.
    R_sasa : float
        Shell radius in Å (use 3.0 for your case).
    shell_thickness : float
        Thickness of the shell in Å.

    Returns
    -------
    delta_sasa_grid : (nz,ny,nx) float32
        Approximate ΔSASA (Å^2) if a generic atom is centered at each voxel.
    """
    sasa_mask = np.asarray(sasa_mask, dtype=bool)
    base = sasa_mask.astype(np.float32)   # 1 per SASA voxel

    kernel = make_shell_kernel_R(
        R_sasa=R_sasa,
        spacing=spacing,
        shell_thickness=shell_thickness,
    )

    delta = ndimage.convolve(
        base,
        kernel,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)

    # Outside SASA, just zero
    delta[~sasa_mask] = 0.0

    return delta

def crop_map_around_point(
    full_map: np.ndarray,
    origin: np.ndarray,
    apix,
    center: np.ndarray,
    box_size: np.ndarray,
):
    """
    Crop a 3D map around a central point (in Å) to a box of given size (in Å),
    and return the cropped map and its new origin.

    Parameters
    ----------
    full_map : (nz,ny,nx) array-like
        Full 3D map (e.g. density, SASA mask, etc.).
        Axis order: [Z, Y, X].
    origin : (3,) array-like
        [x0, y0, z0] Å coordinates for grid index (0,0,0).
    apix : float or (3,) array-like
        Grid spacing in Å. If scalar, assumed isotropic.
        If array-like, [ax, ay, az].
    center : (3,) array-like
        [cx, cy, cz] Å, central point around which to crop.
    box_size : float or (3,) array-like
        Desired box size in Å along X, Y, Z.
        If scalar, same size in all three directions.

    Returns
    -------
    sub_map : np.ndarray, shape (nz_sub, ny_sub, nx_sub)
        Cropped 3D submap, axis order [Z, Y, X].
    new_origin : np.ndarray, shape (3,)
        [x0_sub, y0_sub, z0_sub] Å for index (0,0,0) in the cropped map.
    """
    full_map = np.asarray(full_map)
    origin = np.asarray(origin, dtype=float)
    center = np.asarray(center, dtype=float)

    # Handle apix as scalar or 3-vector
    apix = np.asarray(apix, dtype=float)
    if apix.size == 1:
        ax = ay = az = float(apix)
    else:
        ax, ay, az = float(apix[0]), float(apix[1]), float(apix[2])

    # Handle box_size as scalar or 3-vector
    box_size = np.asarray(box_size, dtype=float)
    if box_size.size == 1:
        Lx = Ly = Lz = float(box_size)
    else:
        Lx, Ly, Lz = float(box_size[0]), float(box_size[1]), float(box_size[2])

    nz, ny, nx = full_map.shape

    # Center indices (nearest voxel to the requested Å position)
    ix_c = int(round((center[0] - origin[0]) / ax))
    iy_c = int(round((center[1] - origin[1]) / ay))
    iz_c = int(round((center[2] - origin[2]) / az))

    if not (0 <= ix_c < nx and 0 <= iy_c < ny and 0 <= iz_c < nz):
        raise ValueError("Center is outside the map bounds.")

    # Half-lengths in voxels (round to nearest)
    hx = int(round((Lx / 2.0) / ax))
    hy = int(round((Ly / 2.0) / ay))
    hz = int(round((Lz / 2.0) / az))

    # Index bounds (inclusive)
    ix_min = max(ix_c - hx, 0)
    ix_max = min(ix_c + hx, nx - 1)
    iy_min = max(iy_c - hy, 0)
    iy_max = min(iy_c + hy, ny - 1)
    iz_min = max(iz_c - hz, 0)
    iz_max = min(iz_c + hz, nz - 1)

    # Slice the map: [Z, Y, X]
    sub_map = full_map[iz_min:iz_max + 1,
                       iy_min:iy_max + 1,
                       ix_min:ix_max + 1].copy()

    # New origin in Å for the cropped box (index 0,0,0)
    new_origin = origin + np.array([
        ix_min * ax,
        iy_min * ay,
        iz_min * az,
    ], dtype=float)

    return sub_map, new_origin

def make_protein_and_solvent_masks(
        atom_coords,
        atom_radii,
        origin,
        grid_shape,
        spacing,
        probe_radius=1.4,
):
    protein_mask, solvent_mask = grid_maps.make_protein_and_solvent_masks_cpp(
        np.asarray(atom_coords, dtype=float),
        np.asarray(atom_radii,  dtype=float),
        np.asarray(origin,      dtype=float),
        tuple(grid_shape),
        float(spacing),
        float(probe_radius),
    )
    return protein_mask, solvent_mask

def final_sasa_mask_standalone(
    *,
    positions,
    atom_radii,
    grid_origin,
    grid_shape,
    grid_spacing,
    system_centroid,
    bulk_probe_radius=6.0,
    ses_probe_radius=1.2,
    crop_box_size=(30, 30, 30),
    R_sasa=3.0,
    shell_thickness=1.0,
):
    """
    Standalone version of final_sasa_mask.

    Returns
    -------
    bulk_solvent_mask : np.ndarray (int)
    protein_mask      : np.ndarray (int)
    sasa_mask         : np.ndarray (int)
    delta_sasa_generic_map : EMMap
    """
    
    
    t1 = time.perf_counter()

    # identify bulk solvent within grid
    _, bulk_solvent_mask = make_protein_and_solvent_masks(
        atom_coords=positions,
        atom_radii=atom_radii,
        origin=grid_origin,
        grid_shape=grid_shape,
        spacing=grid_spacing,
        probe_radius=bulk_probe_radius
    )

    # identify SES in grid
    protein_mask, _ = make_protein_and_solvent_masks(
        atom_coords=positions,
        atom_radii=atom_radii,
        origin=grid_origin,
        grid_shape=grid_shape,
        spacing=grid_spacing,
        probe_radius=ses_probe_radius
    )

    # sasa mask
    sasa_mask = protein_mask + bulk_solvent_mask
    sasa_mask = (sasa_mask == 0).astype(int)

    # binary close to fill in small holes
    structuring_element = ndimage.generate_binary_structure(3, 1)
    sasa_mask = ndimage.binary_closing(
        sasa_mask.copy(),
        structure=structuring_element
    ).astype(int)

    # crop sasa around binding region
    box_size = np.array(crop_box_size, dtype=float)
    box_center = np.asarray(system_centroid, dtype=float)
    
    sub_map, new_origin = crop_map_around_point(
        sasa_mask,
        grid_origin,
        grid_spacing,
        box_center,
        box_size,
    )
    '''
    delta_sasa_generic = build_delta_sasa_generic_grid(
        sasa_mask=sub_map,
        spacing=grid_spacing,
        R_sasa=R_sasa,
        shell_thickness=shell_thickness,
    )

    delta_sasa_generic_map = EMMap(
        new_origin,
        (grid_spacing, grid_spacing, grid_spacing),
        delta_sasa_generic,
        3.0
    )
    '''

    rt = time.perf_counter() - t1
    print("final_sasa_mask:", rt)

    return bulk_solvent_mask, protein_mask, sasa_mask#, delta_sasa_generic_map


def final_solvent_depth_mask_standalone(
    *,
    bulk_solvent_mask,
    protein_mask,
    sasa_mask,
    grid_spacing,
    connectivity=6,
    crop=True,
    pad=3,
):
    """
    Standalone version of final_solvent_depth_mask.

    Returns
    -------
    depth_map  : np.ndarray
    depth_clean: np.ndarray
    depth_norm : np.ndarray
    """
    t2 = time.perf_counter()

    depth_map = compute_solvent_depth_map_morph_cpp(
        bulk_mask=bulk_solvent_mask.astype(bool),
        protein_mask=protein_mask.astype(bool),
        spacing=grid_spacing,
        sasa_mask=sasa_mask.astype(bool),
        connectivity=connectivity,
        crop=crop,
        pad=pad,
    )

    depth_clean = prepare_depth_for_mrc_cpp(
        depth_map,
        sasa_mask=sasa_mask,
        fill_unreachable=0.0,
    )

    maxv = np.max(depth_clean) if np.max(depth_clean) > 0 else 1.0
    depth_norm = depth_clean / maxv

    rt2 = time.perf_counter() - t2
    print("final_solvent_depth_mask:", rt2)

    return depth_map, depth_clean, depth_norm


def get_flow_mask_standalone(
    *,
    protein_mask,
    bulk_solvent_mask,
    sasa_mask,
    grid_spacing,
    connectivity=6,
    power=4,
    n_iter=10,
):
    """
    Standalone version of get_flow_mask.

    Returns
    -------
    env_index        : np.ndarray
    env_index_smooth : np.ndarray
    """
    t2 = time.perf_counter()

    r_local, R_bottle, flow_norm, seed_mask = compute_flow_widest_path_interface_seeds_cpp(
        protein_mask,
        bulk_solvent_mask,
        sasa_mask,
        apix=grid_spacing,
        connectivity=connectivity,
        power=power
    )

    R_max = np.max(R_bottle) if np.max(R_bottle) > 0 else 1.0
    R_norm = R_bottle / R_max
    constriction = 1.0 - R_norm
    constriction = constriction * sasa_mask

    # Combine: high when deep AND constricted
    # (depth_norm is applied outside in the orchestrator)
    # Here we only produce the geometric constriction component if needed.
    # But to match your original behavior, we still compute env_index with
    # the caller's depth_norm before calling this function.
    # We'll keep this function simple: caller passes depth_norm? No —
    # we mirror your original layout by computing env_index outside.

    # NOTE: To preserve your original exact behavior,
    # this function expects the caller to multiply depth_norm * constriction.
    # We'll return constriction and let the orchestrator combine.
    # But since your original get_flow_mask sets env_index directly,
    # we include an optional path in the orchestrator.

    # For minimal change, we return constriction and smoothed of full env_index
    # will be computed in orchestrator once env_index is known.

    rt2 = time.perf_counter() - t2
    print("get_flow_mask (geometry prep):", rt2)

    return constriction


def smooth_env_index_standalone(env_index, sasa_mask, n_iter=10):
    """
    Mirror of your smoothing call.
    """
    return smooth_env_map_diffusion(env_index, sasa_mask, n_iter=n_iter)


def final_hydrophobic_grid_standalone(
    *,
    positions,
    atoms,
    grid_origin,
    grid_shape,
    grid_spacing,
    sasa_mask,
    env_index_smooth,
    cutoff=5.0,
    logp_smooth_sigma=1.5,
    vdw_radii_by_type=None,
):
    """
    Standalone version of final_hydrophobic_grid.

    Returns
    -------
    hydro_field_xlogp : np.ndarray
    hydro_enc_grid    : np.ndarray
    hphob_sub_origin  : np.ndarray
    hphob_sub_sasa_mask : np.ndarray
    hphob_sub_env_norm  : np.ndarray
    """
    t2 = time.perf_counter()

    if vdw_radii_by_type is None:
        vdw_radii_by_type = {"D": 2.0}

    logp = XlogP3()
    logp_values = np.array([logp.get_logp(a.residue.name, a.name) for a in atoms])

    hydro_field_xlogp = propagate_logp_exp_decay_cpp(
        logp_centers=positions,
        logp_values=logp_values,
        origin=grid_origin,
        grid_shape=grid_shape,
        spacing=grid_spacing,
        sasa_mask=sasa_mask,
        cutoff=cutoff,
    )

    hydro_field_xlogp = hydro_field_xlogp * sasa_mask
    hydro_field_xlogp = gaussian_filter(hydro_field_xlogp, sigma=logp_smooth_sigma)

    enclosure_grids = make_enclosure_grids_for_atom_types(
        sasa_mask=sasa_mask,
        spacing=grid_spacing,
        vdw_radii_by_type=vdw_radii_by_type,
        weight_grid=hydro_field_xlogp,
    )
    
    # Your original code loops keys but ends up storing only one grid.
    # We keep that behavior: last grid in dict order wins.
    hydro_enc_grid = None
    new_origin = grid_origin

    for k, grid in enclosure_grids.items():
        grid = grid * sasa_mask
        grid = grid * env_index_smooth

        valid_mask = (
            sasa_mask.astype(bool)
            & (env_index_smooth > 1e-4)
            & (grid > 0.0)
        )

        vals = grid[valid_mask]

        if vals.size > 0:
            q10, q50, q95 = np.percentile(vals, [25, 50, 99])
            base = np.zeros_like(grid, dtype=np.float32)
            denom = (q95 - q10 + 1e-8)
            base[valid_mask] = (grid[valid_mask] - q10) / denom
            base = np.clip(base, 0.0, 1.0)

            gamma = 2.0
            grid_norm = base ** gamma
        else:
            grid_norm = np.zeros_like(grid, dtype=np.float32)

        grid_norm[~valid_mask] = 0.0
        hydro_enc_grid = grid_norm

    hphob_sub_origin = new_origin
    hphob_sub_sasa_mask = sasa_mask
    hphob_sub_env_norm = env_index_smooth

    rt2 = time.perf_counter() - t2
    print("final_hydrophobic_grid:", rt2)

    return (
        hydro_field_xlogp,
        hydro_enc_grid,
        hphob_sub_origin,
        hphob_sub_sasa_mask,
        hphob_sub_env_norm,
    )


def final_electrostatic_map_standalone(
    *,
    protein_complex_structure,
    protein_complex_system,
    env_index_smooth,
    sasa_mask,
    grid_origin,
    grid_spacing,
    system_centroid,
    base_box_size=(30, 30, 30),
    electro_cutoff=12.0,
    c_factor=332.06,
    min_r=0.001,
):
    """
    Standalone version of final_electrostatic_map.

    Returns
    -------
    sub_env_index_smooth : np.ndarray
    sub_sasa_mask        : np.ndarray
    sub_elc_origin       : np.ndarray
    cpp_electrostatics   : np.ndarray
    cpp_scaled           : np.ndarray
    """
    t1 = time.perf_counter()

    prot_coords, prot_charges = compute_charges(
        protein_complex_structure,
        protein_complex_system,
        collapse_hydrogens=False
    )

    box_size = np.array(base_box_size, dtype=float) + np.array(
        (electro_cutoff, electro_cutoff, electro_cutoff), dtype=float
    )
    box_center = np.asarray(system_centroid, dtype=float)
    
    sub_map, new_origin = crop_map_around_point(
        env_index_smooth,
        grid_origin,
        grid_spacing,
        box_center,
        box_size,
    )
    
    sub_map_sasa, new_origin = crop_map_around_point(
        sasa_mask,
        grid_origin,
        grid_spacing,
        box_center,
        box_size,
    )

    sub_env_index_smooth = sub_map
    sub_sasa_mask = sub_map_sasa
    sub_elc_origin = new_origin

    cpp_electrostatics = compute_electrostatic_grid_cutoff_cpp(
        prot_coords,
        prot_charges,
        sub_map.shape,
        new_origin,
        np.array((grid_spacing, grid_spacing, grid_spacing)),
        c_factor=c_factor,
        min_r=min_r,
        cutoff=electro_cutoff
    )

    cpp_scaled = cpp_electrostatics * (sub_map + 0.5)

    t2 = time.perf_counter() - t1
    print("final_electrostatic_map:", t2)

    return sub_env_index_smooth, sub_sasa_mask, sub_elc_origin, cpp_electrostatics, cpp_scaled


def final_desolvation_grid_standalone(
    *,
    hydro_field_xlogp,
    hphob_sub_sasa_mask,
    hphob_sub_env_norm,
    cpp_electrostatics,
    sub_sasa_mask,
    sub_env_index_smooth,
    hydro_smooth_sigma=1.5,
    polar_smooth_sigma=1.5,
):
    """
    Standalone version of final_desolvation_grid.

    Returns
    -------
    desolv_polar_grid : np.ndarray
    desolv_hphob_grid : np.ndarray
    """
    t1 = time.perf_counter()

    # hydrophobic normalization
    h_min, h_max = np.percentile(hydro_field_xlogp, [5, 95])
    h_norm = (hydro_field_xlogp - h_min) / (h_max - h_min + 1e-8)
    h_norm = np.clip(h_norm, 0.0, 1.0)
    h_norm = h_norm * hphob_sub_sasa_mask

    # electro normalization (sub-box)
    phi_abs = np.abs(cpp_electrostatics)
    phi_abs = phi_abs * sub_sasa_mask

    vals = phi_abs[sub_sasa_mask.astype(bool)]
    if vals.size > 0:
        p5, p95 = np.percentile(vals, [5, 95])
    else:
        p5, p95 = 0.0, 1.0

    pol_norm = (phi_abs - p5) / (p95 - p5 + 1e-8)
    pol_norm = np.clip(pol_norm, 0.0, 1.0)
    pol_norm = pol_norm * sub_sasa_mask

    # env indices
    E = hphob_sub_env_norm
    E_sub = sub_env_index_smooth

    desolv_hphob_grid = E * h_norm
    desolv_hphob_grid = gaussian_filter(desolv_hphob_grid, sigma=hydro_smooth_sigma)

    desolv_polar_grid = E_sub * pol_norm
    desolv_polar_grid = gaussian_filter(desolv_polar_grid, sigma=polar_smooth_sigma)

    rt1 = time.perf_counter() - t1
    print("final_desolvation_grid:", rt1)

    return desolv_polar_grid, desolv_hphob_grid

def build_site_maps_standalone(
    *,
    # geometry / atoms
    positions,
    atom_radii,
    atoms,

    # grid
    grid_origin,
    grid_spacing,
    grid,  # used for shape

    # things that used to be from self.system
    system_centroid,
    protein_complex_structure,
    protein_complex_system,

    # optional tuning knobs to keep your old behavior adjustable
    crop_box_size=(30, 30, 30),
    electro_cutoff=12.0,
):
    """
    One fully-standalone pipeline that replaces the block you showed.
    It does NOT depend on `self` and does NOT require `self.system`.

    You must supply anything that was previously pulled from:
      - self.system.centroid[0]
      - self.system.protein.complex_structure
      - self.system.protein.complex_system

    Returns
    -------
    site_maps : dict
        {
          'env_scaled_map', 'electro_scaled_map', 'electro_raw_map',
          'hydrophob_raw_map', 'hydrophob_enc_map',
          'desolvation_polar_grid', 'desolv_hphob_grid',
          'delta_sasa_map'
        }
    """
    
    # 1) SASA / delta SASA
    #bulk_solvent_mask, protein_mask, sasa_mask, delta_sasa_map 
    bulk_solvent_mask, protein_mask, sasa_mask = final_sasa_mask_standalone(
        positions=positions,
        atom_radii=atom_radii,
        grid_origin=grid_origin,
        grid_shape=grid.shape,
        grid_spacing=grid_spacing,
        system_centroid=system_centroid,
        crop_box_size=crop_box_size,
    )

    # 2) depth
    depth_map, depth_clean, depth_norm = final_solvent_depth_mask_standalone(
        bulk_solvent_mask=bulk_solvent_mask,
        protein_mask=protein_mask,
        sasa_mask=sasa_mask,
        grid_spacing=grid_spacing,
    )

    # 3) flow / env index
    constriction = get_flow_mask_standalone(
        protein_mask=protein_mask,
        bulk_solvent_mask=bulk_solvent_mask,
        sasa_mask=sasa_mask,
        grid_spacing=grid_spacing,
    )

    env_index = depth_norm * constriction
    env_index_smooth = smooth_env_index_standalone(env_index, sasa_mask, n_iter=10)

    # 4) hydrophobic
    (
        hydro_field_xlogp,
        hydro_enc_grid,
        hphob_sub_origin,
        hphob_sub_sasa_mask,
        hphob_sub_env_norm,
    ) = final_hydrophobic_grid_standalone(
        positions=positions,
        atoms=atoms,
        grid_origin=grid_origin,
        grid_shape=grid.shape,
        grid_spacing=grid_spacing,
        sasa_mask=sasa_mask,
        env_index_smooth=env_index_smooth,
    )
    
    
    # 5) electrostatics (sub-box)
    (
        sub_env_index_smooth,
        sub_sasa_mask,
        sub_elc_origin,
        cpp_electrostatics,
        cpp_scaled,
    ) = final_electrostatic_map_standalone(
        protein_complex_structure=protein_complex_structure,
        protein_complex_system=protein_complex_system,
        env_index_smooth=env_index_smooth,
        sasa_mask=sasa_mask,
        grid_origin=grid_origin,
        grid_spacing=grid_spacing,
        system_centroid=system_centroid,
        base_box_size=crop_box_size,
        electro_cutoff=electro_cutoff,
    )
    
    
    env_scaled_map = EMMap(
        grid_origin,
        (grid_spacing, grid_spacing, grid_spacing),
        env_index_smooth,
        3.0
    )

    electro_scaled_map = EMMap(
        sub_elc_origin,
        (grid_spacing, grid_spacing, grid_spacing),
        cpp_scaled,
        3.0
    )

    electro_raw_map = EMMap(
        sub_elc_origin,
        (grid_spacing, grid_spacing, grid_spacing),
        cpp_electrostatics,
        3.0
    )

    hydrophob_raw_map = EMMap(
        hphob_sub_origin,
        (grid_spacing, grid_spacing, grid_spacing),
        hydro_field_xlogp,
        3.0
    )
    
    
    hydrophob_enc_map = EMMap(
        hphob_sub_origin,
        (grid_spacing, grid_spacing, grid_spacing),
        hydro_enc_grid,
        3.0
    )

    site_maps = {
        "env_scaled_map": env_scaled_map,
        "electro_scaled_map": electro_scaled_map,
        "electro_raw_map": electro_raw_map,
        "hydrophob_raw_map": hydrophob_raw_map,
        "hydrophob_enc_map": hydrophob_enc_map,
        
    }

    return site_maps



