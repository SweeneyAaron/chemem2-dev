# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

import os
import copy
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from ChemEM.messages import Messages
from ChemEM.data.data import (TABLE_A,
                              TABLE_B,
                              TABLE_C,
                              HBOND_POLYA,
                              HBOND_POLYB,
                              HBOND_POLYC,
                              AtomType,
                              HBOND_DONOR_ATOM_IDXS, 
                              HBOND_ACCEPTOR_ATOM_IDXS,
                              HALOGEN_ACCEPTOR_ATOM_IDXS)

from ChemEM.tools.aromatic_score import AromaticScore
from ChemEM.tools.halogen_bond_score import HalogenBondScore
from ChemEM.tools.biomolecule import (get_role_int,
                                      residue_atom_charge,
                                      get_hbond_direction, 
                                      compute_protein_ring_types,
                                      get_protein_hydrogen_reference,
                                      compute_charges)

from ChemEM.tools.geometry import get_valid_points_and_adjacency, kmeans_clustering_split
from ChemEM.tools.ligand import (get_van_der_waals_radius,
                                 get_ligand_heavy_atom_indexes,
                                 compute_bond_distances,
                                 find_best_ring_bond_to_break,
                                 remove_bonds_from_mol,
                                 get_imporper_torsion_restraints,
                                 get_torsion_lists,
                                 get_ligand_hydrogen_reference,
                                 per_atom_logp)


from ChemEM.tools.docking import compute_halogen_bond_data 

from ChemEM.tools.grid_maps import GridFactory
from ChemEM.tools.forces import export_torsion_profile 

###############################################################################
# Helpers restored from the original precompute_data.py
###############################################################################

def fragment_ligand(mol: Chem.Mol):
    """
    Fragment the molecule on BRICS bonds.
    Returns (fragments, atom_mapping) where atom_mapping gives original atom idxs
    per fragment.
    """
    frag_mol = Chem.FragmentOnBRICSBonds(mol)
    frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
    atom_mapping = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=True)
    return frags, atom_mapping


def compute_fragment_logP(frag: Chem.Mol) -> float:
    """Crippen fragment logP (same as original)."""
    return float(Crippen.MolLogP(frag))


def get_hydrophobic_groups(mol: Chem.Mol):
    """
    Same behavior as original:
    returns dict: frag_idx -> (atom_idx_tuple, logP) for fragments with logP > 0.
    """
    fragments, atom_mapping = fragment_ligand(mol)
    fragment_data = {}
    for idx, (frag, atom_map) in enumerate(zip(fragments, atom_mapping)):
        logp_frag = compute_fragment_logP(frag)
        if logp_frag > 0.0:
            fragment_data[idx] = (atom_map, logp_frag)
    return fragment_data


def select_ring_flip_torsions(mol: Chem.Mol, torsion_quads):
    """
    Same behavior as original select_ring_flip_torsions().
    """
    mol = Chem.RemoveHs(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    flip_torsions = []

    for (i, j, k, l) in torsion_quads:
        if max(i, j, k, l) >= mol.GetNumAtoms():
            continue

        if k in ring_atoms and l in ring_atoms and j not in ring_atoms and i not in ring_atoms:
            a, b, c, d = l, k, j, i
        elif i in ring_atoms and j in ring_atoms and k not in ring_atoms and l not in ring_atoms:
            a, b, c, d = i, j, k, l
        else:
            continue

        neis = mol.GetAtomWithIdx(b).GetNeighbors()
        neis = [n.GetIdx() for n in neis if n.GetIdx() not in [a, c]]
        if not neis:
            continue

        rank_a = ranks[a]
        if any(ranks[n] == rank_a for n in neis):
            continue

        flip_torsions.append((a, b, c, d))

    return list(set(flip_torsions))


def compute_donor_acceptor_mask(protein_pdb_atom_ids, protein_atom_types, ligand_atom_types):
    """
    Same behavior as original compute_donor_acceptor_mask():
    boolean mask (n_protein_atoms, n_ligand_atoms) for donor/acceptor pairs.
    """
    n_protein_atoms = len(protein_atom_types)
    n_ligand_atoms = len(ligand_atom_types)
    mask = np.zeros((n_protein_atoms, n_ligand_atoms), dtype=bool)

    for i in range(n_protein_atoms):
        p_type = int(protein_atom_types[i])
        for j in range(n_ligand_atoms):
            l_type = int(ligand_atom_types[j])
            if ((p_type in HBOND_DONOR_ATOM_IDXS) and (l_type in HBOND_ACCEPTOR_ATOM_IDXS)) or \
               ((p_type in HBOND_ACCEPTOR_ATOM_IDXS) and (l_type in HBOND_DONOR_ATOM_IDXS)):
                mask[i, j] = True
    return mask


class PreCompDataLigand:
    """
    Stores ligand topology, physics, and scoring parameters for the C++ engine.
    """
    def __init__(self, ligand, flexible_rings=False):
        
        self.mol = ligand.mol
        self.complex_structure = ligand.complex_structure # Required for torsion profiling
        # 1. Topology & Connectivity
        self._init_topology(ligand)
        # 2. Flexible Rings (Frag/Constraints)
        self._init_flexible_rings(ligand, flexible_rings)
        # 3. Torsions (Rotatable bonds)
        self._init_torsions(ligand)
        # 4. Physics (Radii, Charges, Hydrophobics)
        self._init_physics(ligand)
        # 5. Scoring Tables (Intra-molecular)
        self._init_scoring_tables()
        self._init_ring_flips()
        

    def _init_topology(self, ligand):
        """Basic atom indices and bond distances."""
        self.atom_masses = [a.GetMass() for a in self.mol.GetAtoms() if a.GetSymbol() != 'H']
        self.ligand_heavy_atom_indexes = get_ligand_heavy_atom_indexes(self.mol)
        self.ligand_heavy_end_index = np.max(self.ligand_heavy_atom_indexes) if len(self.ligand_heavy_atom_indexes) else 0
        
        # IMPORTANT: this must come from ligand.atom_types (original behavior)
        self.ligand_atom_types = np.array([t.idx for t in ligand.atom_types], dtype=np.int32)
        
        # Bond Distance Matrix (for exclusion masks)
        self.ligand_bond_distances = compute_bond_distances(
            self.mol, self.ligand_heavy_atom_indexes
        )
        
        # Default exclusion: 1 means included, 0 means excluded (usually covalent)
        self.exc_atoms = [1] * len(self.ligand_heavy_atom_indexes)

    def _init_flexible_rings(self, ligand, flexible_rings):
        """Handles ring fragmentation for flexibility."""
        self.break_bonds = False
        self.constrain_atoms = []
        self.constrain_atoms_dist = []
        self.improper_torsion_constraints = []
        self._new_ring_torsions = []

        if not flexible_rings:
            return

        bonds_to_break = find_best_ring_bond_to_break(self.mol)
        if not bonds_to_break:
            return

        frag_mol, atoms_to_constrain, dist_constraints = remove_bonds_from_mol(
            self.mol, bonds_to_break
        )

        if frag_mol:
            self.break_bonds = frag_mol
            self.constrain_atoms = atoms_to_constrain
            self.constrain_atoms_dist = dist_constraints
            
            # Calculate improper torsions to maintain stereochemistry at breaks
            self.improper_torsion_constraints = get_imporper_torsion_restraints(
                self.mol, atoms_to_constrain
            )
            
            # Identify new rotatable bonds created by breaking the ring
            candidate_torsions = get_torsion_lists(frag_mol)
            self._new_ring_torsions = [t for t in candidate_torsions if all(x < frag_mol.GetNumHeavyAtoms() for x in t)]

    def _init_torsions(self, ligand):
        """Calculates torsion profiles (energy vs angle) using OpenMM."""
        # Standard rotatable bonds
        self.torsion_lists = get_torsion_lists(self.mol)
        
        # Generate energy profiles for these torsions
        self.ligand_torsion_profile = export_torsion_profile(ligand, self.torsion_lists)
        
        # Extract indices and scores
        self.ligand_torsion_idxs = [i[1] for i in self.ligand_torsion_profile]
        self.ligand_torsion_scores = [i[2] for i in self.ligand_torsion_profile]
        
        # Merge ring torsions if they exist and aren't duplicates
        existing_torsions = set(tuple(t) for t in self.ligand_torsion_idxs)
        for t in self._new_ring_torsions:
            if tuple(t) not in existing_torsions:
                self.ligand_torsion_idxs.append(list(t))
                
        self.n_torsions = len(self.ligand_torsion_idxs)
        self.end_torsions = len(self.torsion_lists)

    def _init_physics(self, ligand):
        """Calculates chemical properties."""
        self.ligand_hydrophobic = get_hydrophobic_groups(self.mol)
        self.ligand_hydrophobic_cpp = list(self.ligand_hydrophobic.values())
        
        self.ligand_radii = np.array([get_van_der_waals_radius(atom.GetSymbol()) for atom in self.mol.GetAtoms()])
        self.ligand_hydrogen_idx = get_ligand_hydrogen_reference(self.mol)
        
        # Rings
        self.ligand_ring_types = getattr(ligand, 'ring_types', [])
        self.ligand_ring_type_ints = [i.idx for i in self.ligand_ring_types]
        self.ligand_ring_indices = getattr(ligand, 'ring_indices', [])
        
        # Halogen Bonds
        self.halogen_bond_donor_indices, self.halogen_bond_donor_root_indices = \
            compute_halogen_bond_data(self.mol, self.ligand_atom_types)
            
        # Charges & Mass
        self.MW = Descriptors.MolWt(self.mol)
        _, self.ligand_charges = compute_charges(ligand.complex_structure, ligand.complex_system)
        self.ligand_charges = list(self.ligand_charges)
        self.ligand_formal_charge = np.asarray(getattr(ligand, 'ligand_charge', []))
        
        # LogP & H-Bond
        self.per_atom_logp = np.array(per_atom_logp(self.mol))
        self.ligand_hbond_atom = [
            1 if i in HBOND_DONOR_ATOM_IDXS + HBOND_ACCEPTOR_ATOM_IDXS else 0 
            for i in self.ligand_atom_types
        ]

    def _init_ring_flips(self):
        """
        Restore original ring-flip torsion selection.
        Original used get_torsion_lists_dup(); here we emulate by adding reversed torsions.
        """
        torsions = list(get_torsion_lists(self.mol))
        torsions = torsions + [tuple(t[::-1]) for t in torsions]
        self.ring_flip_torsions = select_ring_flip_torsions(self.mol, torsions)

    def _init_scoring_tables(self):
        """Populates intra-molecular scoring matrices."""
        n = len(self.ligand_atom_types)
        self.LIGAND_INTRA_A_VALUES = np.zeros((n, n), dtype=np.float64)
        self.LIGAND_INTRA_B_VALUES = np.zeros((n, n), dtype=np.float64)
        self.LIGAND_INTRA_C_VALUES = np.zeros((n, n), dtype=np.float64)
        
        for i, t1 in enumerate(self.ligand_atom_types):
            for j, t2 in enumerate(self.ligand_atom_types):
                self.LIGAND_INTRA_A_VALUES[i, j] = TABLE_A[t1, t2]
                self.LIGAND_INTRA_B_VALUES[i, j] = TABLE_B[t1, t2]
                self.LIGAND_INTRA_C_VALUES[i, j] = TABLE_C[t1, t2]

    def __radd__(self, other):
        """Allows `protein + ligand` syntax."""
        if isinstance(other, PreCompDataProtein):
            return other + self
        return NotImplemented


class PreCompDataProtein:
    def __init__(self,
                 binding_site,
                 system,
                 site_maps=None,
                 bias_radius=12.0, 
                 split_site=False,
                 probe_radius=1.4,
                 grid_apix=(0.375, 0.375, 0.375),
                 grid_spacing=0.375,
                 pad_box=10.0
                 ):
        
        # ----- 1. Docking Setup & Parameters -----
        self.n_global_search = getattr(system.options, 'n_global_search', 1000)
        self.n_local_search = getattr(system.options, 'n_local_search', 10)
        self.ncpu = getattr(system.options, 'ncpu', 1)
        
        # Scoring constraints / annealing params
        self.repCap0 = 2.0
        self.repCap1 = 5.0
        self.repCap_discrete = 5.0
        self.repCap_inner_nm = 10.0 
        self.repCap_final_nm = 15.0 
        self.rms_cutoff = 2.0
        self.topN = 20 
        self.a_lo, self.a_mid, self.a_hi = 0.25, 0.45, 0.70
        self.iterations = 0

        # Weights
        self.w_nonbond = 0.015435
        self.w_dsasa = -0.000325
        self.w_hphob = 2.429216
        self.w_electro = 0.006459
        self.w_ligand_torsion = 0.046476
        self.w_ligand_intra = 0.001517
        self.bias = -4.0204
        self.w_vdw = 0.015435 
        self.w_hbond = 0.003415 
        self.w_aromatic = 0.114082 
        self.w_halogen = 0.160168 
        self.w_hphob_enc = 0.034452 
        self.w_constraint = 1.0
        self.nb_cell = 4.5

        # Salt bridge Buckingham boost (present in original PreCompDataProtein2)
        self.SB_A = -3.85504
        self.SB_B = 0.345362
        self.SB_C = -144.293
        
        
        # ----- 2. Protein Atom Data -----
        self.residues = binding_site.lining_residues
        self.protein_positions = binding_site.rdkit_lining_mol.GetConformer().GetPositions()
        
        # Initialize chemistry arrays
        self._init_protein_chemistry(binding_site.rdkit_lining_mol)

        # ----- 3. Scoring Matrices (Inter-molecular) -----
        
        self._init_scoring_tables()
        self._init_spline_scores() # Aromatic & Halogen splines

        # ----- 4. Binding Site Grid Points -----
        self.binding_site_grid_origin = np.array(binding_site.origin)
        self.binding_site_grid_apix = np.array(binding_site.apix)
        # Extract map shape
        self.binding_site_grid_nz, self.binding_site_grid_ny, self.binding_site_grid_nx = binding_site.distance_map.shape
        self.apix = grid_apix

        # Calculate valid search points
        mask = binding_site.distance_map > probe_radius
        self.translation_points, self.adjacency = get_valid_points_and_adjacency(mask, connectivity=26)

        # Set Centroid
        key = int(binding_site.key)
        if hasattr(system, 'centroid') and len(system.centroid) > key:
            self.binding_site_centroid = np.array(system.centroid[key])
        else:
            # Fallback centroid from translation points
            coords = self.binding_site_grid_origin + np.array(self.translation_points)[:,::-1] * self.binding_site_grid_apix
            self.binding_site_centroid = coords.mean(axis=0)
        
        self.bias_radius = bias_radius

        # Multi-site splitting logic
        if split_site:
            self._split_binding_site(self.translation_points, 
                                     self.binding_site_grid_origin, 
                                     self.binding_site_grid_apix, 
                                     system.output)

        # ----- 5. Grid Maps (SASA, Electro, etc.) -----
        if site_maps is None:
            # Rebuild all maps using GridFactory
            self._build_maps_from_scratch(system, grid_spacing, pad_box)
        else:
            self._load_maps(site_maps)

        # Restore original water map attributes (computed from protein roles)
        #self._init_water_map()

        # ----- 6. Density Map Features (CCC, MI) -----
        self._init_density_map_features(system, binding_site.key)

        # ----- 7. ACO Init -----
        self._reset_ACO_simplex()

    def _init_protein_chemistry(self, rdkit_mol):
        """Extracts atom types, roles, charges, vectors, and rings."""
        self.protein_pdb_atom_ids = []
        self.protein_atom_types = []
        self.protein_atom_roles = []
        self.protein_formal_charge = []
        self.protein_hbond_dirs = []

        atoms = list(rdkit_mol.GetAtoms())
        
        for atom in atoms:
            if atom.GetSymbol() == "H": continue
            
            res_name = atom.GetPDBResidueInfo().GetResidueName() if atom.GetPDBResidueInfo() else "UNK"
            atom_name = atom.GetPDBResidueInfo().GetName().strip() if atom.GetPDBResidueInfo() else atom.GetSymbol()
            
            # IDs
            self.protein_pdb_atom_ids.append((res_name, atom_name))
            
            # Types & Roles
            try:
                type_idx = AtomType.from_id(res_name, atom_name).idx
            except:
                type_idx = 0 # Fallback
            
            self.protein_atom_types.append(type_idx)
            
            role = get_role_int(res_name, atom_name)
            self.protein_atom_roles.append(role)
            
            # Physics
            self.protein_formal_charge.append(residue_atom_charge(res_name, atom_name))
            self.protein_hbond_dirs.append(get_hbond_direction(atom, role))

        # Convert to numpy
        self.protein_atom_types = np.array(self.protein_atom_types, dtype=np.int32)
        self.protein_atom_roles = np.array(self.protein_atom_roles, dtype=np.int32)
        self.protein_formal_charge = np.array(self.protein_formal_charge, dtype=float)
        self.protein_hbond_dirs = np.array(self.protein_hbond_dirs, dtype=float)
        
        # Radii & Hydrogens
        self.protein_radii = np.array([get_van_der_waals_radius(a.GetSymbol()) for a in atoms])
        self.protein_hydrogens = get_protein_hydrogen_reference(rdkit_mol)

        # Rings & Halogens
        self.protein_ring_types, self.protein_ring_coords, self.protein_ring_idx = \
            compute_protein_ring_types(self.residues, self.protein_positions)
        self.protein_ring_type_ints = np.array([i.idx for i in self.protein_ring_types], dtype=np.int32)
        
        self.halogen_bond_acceptor_indices, self.halogen_bond_acceptor_root_indices = \
            compute_halogen_bond_data(rdkit_mol, self.protein_atom_types, HALOGEN_ACCEPTOR_ATOM_IDXS)

    def _split_binding_site(self, points, origin, apix, output_dir):
        """Splits large binding site points into clusters."""
        print("Splitting binding site...")
        coords = origin + np.array(points)[:,::-1] * apix
        clusters = kmeans_clustering_split(coords, np.array(points), origin, apix)
        
        self.split_site_translation_points = [c['indices'] for c in clusters]
        self.split_site_translation_centroid_raidus = [(c['centroid'], c['radius']) for c in clusters]

    def _init_scoring_tables(self):
        """Populates H-bond polynomials and dummy pair matrices."""
        donor_types = sorted([int(d) for d in HBOND_POLYA.keys()])
        acceptor_types = sorted({int(a) for sub in HBOND_POLYA.values() for a in sub.keys()})
        
        degA = len(HBOND_POLYA[str(donor_types[0])][str(acceptor_types[0])]) - 1
        degB = len(HBOND_POLYB[str(donor_types[0])][str(acceptor_types[0])]) - 1
        degC = len(HBOND_POLYC[str(donor_types[0])][str(acceptor_types[0])]) - 1
        
        self.hbond_donor_types = np.array(donor_types, dtype=int)
        self.hbond_acceptor_types = np.array(acceptor_types, dtype=int)
        
        self.hbond_polyA = np.zeros((len(donor_types), len(acceptor_types), degA+1), dtype=float)
        self.hbond_polyB = np.zeros((len(donor_types), len(acceptor_types), degB+1), dtype=float)
        self.hbond_polyC = np.zeros((len(donor_types), len(acceptor_types), degC+1), dtype=float)
        
        for i, d in enumerate(donor_types):
            for j, a in enumerate(acceptor_types):
                self.hbond_polyA[i,j] = HBOND_POLYA[str(d)][str(a)]
                self.hbond_polyB[i,j] = HBOND_POLYB[str(d)][str(a)]
                self.hbond_polyC[i,j] = HBOND_POLYC[str(d)][str(a)]

    def _init_spline_scores(self):
        """Flattens Aromatic and Halogen splines for C++ consumption."""
        # Aromatic
        self.aromatic_score = AromaticScore()
        self._flatten_splines(self.aromatic_score.spline_models, prefix='arom')
        # Halogen
        self.halogen_score = HalogenBondScore()
        self._flatten_splines(self.halogen_score.spline_models, prefix='halo')
        

    def _flatten_splines(self, spline_dict, prefix):
        """
        Flatten spline_models into arrays/lists, matching original attribute names.
        Works for:
          - aromatic: keys are (i1, i2, stack)
          - halogen: keys are (i1, i2)
        """
        
        stack_map = {'p': 0, 't': 1}
        keys = []
        kxA_list = []; kyA_list = []; dimA_list = []; coeffsA = []
        kxB_list = []; kyB_list = []; dimB_list = []; coeffsB = []
        kxC_list = []; kyC_list = []; dimC_list = []; coeffsC = []
        
        knots_x_list_A = []; knots_y_list_A = []
        knots_x_list_B = []; knots_y_list_B = []
        knots_x_list_C = []; knots_y_list_C = []

        items = spline_dict.items() if hasattr(spline_dict, "items") else [(k, spline_dict[k]) for k in spline_dict]

        for k, v in items:
            # aromatic: k=(i1,i2,stack)
            if isinstance(k, tuple) and len(k) == 3:
                i1, i2, stack = k
                
                if stack in stack_map:
                    
                    keys.append((i1, i2, stack_map[stack]))
                else:
                    Messages.fatal_exception('PreComputedDataProtein', "Aromatic Spline Key contains unknown stack map type")
                    
                splineA, splineB, splineC = v
            # halogen: k=(i1,i2) and v=(A,B,C)
            elif isinstance(k, tuple) and len(k) == 2:
                i1, i2 = k
                keys.append((i1, i2))
                splineA, splineB, splineC = v
            else:
                raise ValueError(f"Unexpected spline key: {k}")

            # A
            txA, tyA = splineA.get_knots()
            knots_x_list_A.append(np.array(txA, dtype=np.float64))
            knots_y_list_A.append(np.array(tyA, dtype=np.float64))
            kxA, kyA = splineA.degrees
            nxA = len(txA) - kxA - 1
            nyA = len(tyA) - kyA - 1
            cA = splineA.get_coeffs()
            if cA.size != nxA * nyA:
                raise ValueError(f"{prefix} A[{k}]: expected {nxA*nyA} coeffs, got {cA.size}")
            coeffsA.append(cA.reshape((nxA, nyA)))
            kxA_list.append(kxA); kyA_list.append(kyA); dimA_list.append((nxA, nyA))

            # B
            txB, tyB = splineB.get_knots()
            knots_x_list_B.append(np.array(txB, dtype=np.float64))
            knots_y_list_B.append(np.array(tyB, dtype=np.float64))
            kxB, kyB = splineB.degrees
            nxB = len(txB) - kxB - 1
            nyB = len(tyB) - kyB - 1
            cB = splineB.get_coeffs()
            if cB.size != nxB * nyB:
                raise ValueError(f"{prefix} B[{k}]: expected {nxB*nyB} coeffs, got {cB.size}")
            coeffsB.append(cB.reshape((nxB, nyB)))
            kxB_list.append(kxB); kyB_list.append(kyB); dimB_list.append((nxB, nyB))

            # C
            txC, tyC = splineC.get_knots()
            knots_x_list_C.append(np.array(txC, dtype=np.float64))
            knots_y_list_C.append(np.array(tyC, dtype=np.float64))
            kxC, kyC = splineC.degrees
            nxC = len(txC) - kxC - 1
            nyC = len(tyC) - kyC - 1
            cC = splineC.get_coeffs()
            if cC.size != nxC * nyC:
                raise ValueError(f"{prefix} C[{k}]: expected {nxC*nyC} coeffs, got {cC.size}")
            coeffsC.append(cC.reshape((nxC, nyC)))
            kxC_list.append(kxC); kyC_list.append(kyC); dimC_list.append((nxC, nyC))

        setattr(self, f"{prefix}_keys", np.array(keys, dtype=np.int32))

        setattr(self, f"{prefix}_kxA", np.array(kxA_list, dtype=np.int32))
        setattr(self, f"{prefix}_kyA", np.array(kyA_list, dtype=np.int32))
        setattr(self, f"{prefix}_dimsA", np.array(dimA_list, dtype=np.int32))
        setattr(self, f"{prefix}_coefA", coeffsA)
        setattr(self, f"{prefix}_knots_xA", knots_x_list_A)
        setattr(self, f"{prefix}_knots_yA", knots_y_list_A)

        setattr(self, f"{prefix}_kxB", np.array(kxB_list, dtype=np.int32))
        setattr(self, f"{prefix}_kyB", np.array(kyB_list, dtype=np.int32))
        setattr(self, f"{prefix}_dimsB", np.array(dimB_list, dtype=np.int32))
        setattr(self, f"{prefix}_coefB", coeffsB)
        setattr(self, f"{prefix}_knots_xB", knots_x_list_B)
        setattr(self, f"{prefix}_knots_yB", knots_y_list_B)

        setattr(self, f"{prefix}_kxC", np.array(kxC_list, dtype=np.int32))
        setattr(self, f"{prefix}_kyC", np.array(kyC_list, dtype=np.int32))
        setattr(self, f"{prefix}_dimsC", np.array(dimC_list, dtype=np.int32))
        setattr(self, f"{prefix}_coefC", coeffsC)
        setattr(self, f"{prefix}_knots_xC", knots_x_list_C)
        setattr(self, f"{prefix}_knots_yC", knots_y_list_C)

        # Aromatic compatibility aliases (original exposed these)
        if prefix == "arom":
            self.arom_kx = self.arom_kxA.copy()
            self.arom_ky = self.arom_kyA.copy()
            self.arom_dims = self.arom_dimsA.copy()

    def _build_maps_from_scratch(self, system, spacing, padding):
        """Generates all physical grids using GridFactory."""
        
        
        atoms = [a for a in system.protein.complex_structure.atoms if a.element > 1]
        positions = np.array([[a.xx, a.xy, a.xz] for a in atoms])
        radii = np.array([get_van_der_waals_radius(a.element_name) for a in atoms])
        
        # Calculate Zero Grid Bounds
        origin, _ = GridFactory.make_zero_grid(positions, spacing, padding)
        shape = np.ceil((positions.max(0) - positions.min(0) + 2*padding) / spacing).astype(int)[::-1]
        
        # 1. SASA
        bulk, prot, sasa = GridFactory.generate_sasa_grids(
            positions, radii, origin, shape, spacing
        )
        
        # 2. Electrostatics
        raw_elec, scaled_elec = GridFactory.generate_electrostatics(
            system.protein.complex_structure, system.protein.complex_system,
            origin, shape, spacing, sasa
        )
        
        # 3. Hydrophobics (requires Flow/Constriction first)
        constriction = GridFactory.generate_solvent_flow(prot, bulk, sasa, spacing)
        env_index = constriction
        
        raw_hydro, enc_hydro = GridFactory.generate_hydrophobics(
            positions, atoms, origin, shape, spacing, sasa, env_index
        )
        
        #desolvation polar grid
        
        
        # Store properties required by C++ scoring
        self.env_scaled_grid = env_index
        self.env_scaled_origin = origin
        self.env_scaled_apix = np.array([spacing]*3)
        self.envz, self.envy, self.envx = self.env_scaled_grid.shape
        
        self.electro_scaled_grid = scaled_elec
        self.electro_scaled_origin = origin
        self.electro_scaled_apix = np.array([spacing]*3)
        self.electro_raw_grid = raw_elec
        self.electroz, self.electroy, self.electrox = self.electro_scaled_grid.shape
        self.electrorawz, self.electrrawoy, self.electrrawox = self.electro_raw_grid.shape
        
        self.hydrophob_enc_grid = enc_hydro
        self.hydrophob_enc_origin = origin
        self.hydrophob_enc_apix = np.array([spacing]*3)
        self.hydrophob_raw_grid = raw_hydro
        self.hydrophob_enc_z, self.hydrophob_enc_y, self.hydrophob_enc_x = self.hydrophob_enc_grid.shape
        self.hydrophob_raw_z, self.hydrophob_raw_y, self.hydrophob_raw_x = self.hydrophob_raw_grid.shape
        
        import pdb 
        pdb.set_trace()
        # Placeholder for desolvation (requires further calculation)
        self.desolvation_polar_grid = np.zeros_like(sasa, dtype=float)
        self.desolvation_polar_origin = origin
        self.desolvation_polar_apix = np.array([spacing]*3)
        self.desolv_hphob_grid = np.zeros_like(sasa, dtype=float)
        self.desolv_hphob_origin = origin
        self.desolv_hphob_apix = np.array([spacing]*3)
        self.desolvation_polar_z, self.desolvation_polar_y, self.desolvation_polar_x = self.desolvation_polar_grid.shape
        self.desolv_hphob_z, self.desolv_hphob_y, self.desolv_hphob_x = self.desolv_hphob_grid.shape
        
        self.delta_sasa_grid = np.zeros_like(sasa, dtype=float) # Placeholder
        self.delta_sasa_origin = origin
        self.delta_sasa_apix = np.array([spacing]*3)
        self.delta_sasa_z, self.delta_sasa_y, self.delta_sasa_x = self.delta_sasa_grid.shape

    def _load_maps(self, site_maps):
        """Loads pre-calculated maps from dictionary."""
        # Unpack dictionary into attributes expected by C++
        self.env_scaled_grid = site_maps['env_scaled_map'].density_map
        self.env_scaled_origin = site_maps['env_scaled_map'].origin
        self.env_scaled_apix = np.array(site_maps['env_scaled_map'].apix)
        self.envz, self.envy, self.envx = self.env_scaled_grid.shape
        
        self.electro_scaled_grid = site_maps['electro_scaled_map'].density_map
        self.electro_scaled_origin = site_maps['electro_scaled_map'].origin
        self.electro_scaled_apix = np.array(site_maps['electro_scaled_map'].apix)
        self.electro_raw_grid = site_maps['electro_raw_map'].density_map
        self.electro_raw_origin = site_maps['electro_raw_map'].origin
        self.electro_raw_apix = np.array(site_maps['electro_raw_map'].apix)
        self.electroz, self.electroy, self.electrox = self.electro_scaled_grid.shape
        self.electrorawz, self.electrrawoy, self.electrrawox = self.electro_raw_grid.shape
        
        self.hydrophob_enc_grid = site_maps['hydrophob_enc_map'].density_map
        self.hydrophob_enc_origin = site_maps['hydrophob_enc_map'].origin
        self.hydrophob_enc_apix = np.array(site_maps['hydrophob_enc_map'].apix)
        self.hydrophob_raw_grid = site_maps['hydrophob_raw_map'].density_map
        self.hydrophob_raw_origin = site_maps['hydrophob_raw_map'].origin
        self.hydrophob_raw_apix = np.array(site_maps['hydrophob_raw_map'].apix)
        self.hydrophob_enc_z, self.hydrophob_enc_y, self.hydrophob_enc_x = self.hydrophob_enc_grid.shape
        self.hydrophob_raw_z, self.hydrophob_raw_y, self.hydrophob_raw_x = self.hydrophob_raw_grid.shape
        
        self.desolvation_polar_grid = site_maps['desolvation_polar_grid'].density_map
        self.desolvation_polar_origin = site_maps['desolvation_polar_grid'].origin
        self.desolvation_polar_apix = np.array(site_maps['desolvation_polar_grid'].apix)
        
        self.desolv_hphob_grid = site_maps['desolv_hphob_grid'].density_map
        self.desolv_hphob_origin = site_maps['desolv_hphob_grid'].origin
        self.desolv_hphob_apix = np.array(site_maps['desolv_hphob_grid'].apix)
        self.desolvation_polar_z, self.desolvation_polar_y, self.desolvation_polar_x = self.desolvation_polar_grid.shape
        self.desolv_hphob_z, self.desolv_hphob_y, self.desolv_hphob_x = self.desolv_hphob_grid.shape
        
        self.delta_sasa_grid = site_maps['delta_sasa_map'].density_map
        self.delta_sasa_origin = site_maps['delta_sasa_map'].origin
        self.delta_sasa_apix = np.array(site_maps['delta_sasa_map'].apix)
        self.delta_sasa_z, self.delta_sasa_y, self.delta_sasa_x = self.delta_sasa_grid.shape

    def _init_water_map(self):
        """
        Original PreCompDataProtein2 always exposed:
          water_map_grid/origin/apix and watz/waty/watx
        Try to compute via ChemEM.grid_maps.compute_water_bridge_grid_cpp if available.
        Fallback: zeros grid with env_scaled shape.
        """
        origin = getattr(self, "env_scaled_origin", None)
        apix = getattr(self, "env_scaled_apix", None)
        grid = getattr(self, "env_scaled_grid", None)
        if origin is None or apix is None or grid is None:
            self.water_map_grid = None
            self.water_map_origin = None
            self.water_map_apix = None
            self.watz = self.waty = self.watx = 0
            return

        try:
            from ChemEM import grid_maps as _grid_maps
            if hasattr(_grid_maps, "compute_water_bridge_grid_cpp"):
                w_map = _grid_maps.compute_water_bridge_grid_cpp(
                    np.asarray(self.protein_positions),
                    np.asarray(self.protein_atom_roles),
                    grid.shape,
                    origin,
                    apix,
                    d0=5.0,
                    sigma=0.5,
                    max_dist=6.0
                )
                self.water_map_grid = w_map
            else:
                self.water_map_grid = np.zeros_like(grid, dtype=np.float32)
        except Exception:
            self.water_map_grid = np.zeros_like(grid, dtype=np.float32)

        self.water_map_origin = origin
        self.water_map_apix = np.array(apix)
        self.watz, self.waty, self.watx = self.water_map_grid.shape

    def _init_density_map_features(self, system, key):
        """Computes Density Map features (CCC, MI) if available."""
        if hasattr(system, 'binding_site_maps') and key in system.binding_site_maps:
            dmap_obj = system.binding_site_maps[key][0][0]
            self.binding_site_density_map = dmap_obj.density_map
            self.binding_site_density_map_apix = dmap_obj.apix
            self.binding_site_density_map_origin = dmap_obj.origin
            self.binding_site_density_map_resolution = dmap_obj.resolution
            self.binding_site_density_map_sigma_coeff = 0.356
            self.mi_weight = getattr(system.options, 'mi_weight', 0.0)
            self.sci_weight = getattr(system.options, 'sci_weight', 0.0)
            
            # Precompute CCC features
            feats = GridFactory.precompute_ccc_maps(
                self.binding_site_density_map, 
                self.binding_site_density_map_origin, 
                self.binding_site_density_map_apix, 
                self.binding_site_density_map_resolution
            )
            # Store features for C++
            self.G_kernal = feats['G']
            self.grad_kernal = feats['G_grad']
            self.G_lap_kernal = feats['G_lap']
            self.r = feats['r']
            self.smoothed_map = feats['smoothed']
            self.grad_map = feats['grad']
            self.laplacian_map = feats['laplacian']
            
            # Precompute MI assets
            mi_assets = GridFactory.build_mi_assets(
                self.binding_site_density_map,
                self.binding_site_density_map_origin,
                self.binding_site_density_map_apix,
                self.binding_site_density_map_resolution
            )
            self.mi_bins = 20
            self.mi_map_bins = mi_assets['bins']
            self.mi_origin = self.binding_site_density_map_origin
            self.mi_apix = self.binding_site_density_map_apix
            self.mi_k_offsets = mi_assets['k_offsets']
            self.mi_k_bins = mi_assets['k_weights'] # Mapped to bins
            self.mi_kernel_r = mi_assets['r']
            
        else:
            # Nullify if no map
            self.binding_site_density_map = None
            self.binding_site_density_map_apix = None
            self.binding_site_density_map_origin = None
            self.binding_site_density_map_resolution = None
            self.binding_site_density_map_sigma_coeff = None
            self.mi_weight = 0.0
            self.sci_weight = 0.0

            self.G_kernal = None
            self.grad_kernal = None
            self.G_lap_kernal = None
            self.r = None
            self.smoothed_map = None
            self.grad_map = None
            self.laplacian_map = None

            self.mi_bins = 0
            self.mi_map_bins = None
            self.mi_origin = None
            self.mi_apix = None
            self.mi_k_offsets = None
            self.mi_k_bins = None
            self.mi_kernel_r = None

    def _reset_ACO_simplex(self):
        """Resets the Ant Colony Optimization simplex arrays."""
        self.all_arrays = [self.translation_points]
        self.simplex = [0.5, 0.5, 0.5]
        
        # Add rotation dimensions (3x)
        for _ in range(3):
            self.all_arrays.append(np.arange(0, 360, 30, dtype=np.int32))
            self.simplex.append(0.5)
            
        # Add torsion dimensions
        if hasattr(self, 'ligand_torsion_idxs'):
            for _ in self.ligand_torsion_idxs:
                self.all_arrays.append(np.arange(0, 360, 30, dtype=np.int32))
                self.simplex.append(0.5)
                
        self.simplex = np.array(self.simplex)

    def __add__(self, other):
        """Supports protein + ligand merging."""
        if hasattr(other, 'ligand_atom_types'): # Duck typing check for PreCompDataLigand
            merged = copy.deepcopy(self)
            
            # Copy ligand attributes to merged object
            for k, v in vars(other).items():
                setattr(merged, k, copy.deepcopy(v))
            
            # Initialize intermolecular scoring matrices (A_values, etc.)
            # This logic was in __post_init__ originally
            merged._setup_intermolecular_matrices()
            merged._reset_ACO_simplex() # Reset because ligand added torsions
            return merged
        return NotImplemented

    def _setup_intermolecular_matrices(self):
        """Calculates interaction matrices (A, B, C) between protein and ligand."""
        n_p = len(self.protein_atom_types)
        n_l = len(self.ligand_atom_types)
        
        self.A_values = np.zeros((n_p, n_l), dtype=np.float64)
        self.B_values = np.zeros((n_p, n_l), dtype=np.float64)
        self.C_values = np.zeros((n_p, n_l), dtype=np.float64)
        
        for i, p_idx in enumerate(self.protein_atom_types):
            for j, l_idx in enumerate(self.ligand_atom_types):
                self.A_values[i, j] = TABLE_A[p_idx, l_idx]
                self.B_values[i, j] = TABLE_B[p_idx, l_idx]
                self.C_values[i, j] = TABLE_C[p_idx, l_idx]
                
        # Update H-bond mask
        self.hbond_donor_acceptor_mask = compute_donor_acceptor_mask(
            self.protein_pdb_atom_ids, self.protein_atom_types, self.ligand_atom_types
        )
        




