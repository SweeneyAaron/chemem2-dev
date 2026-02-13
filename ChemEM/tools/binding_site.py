# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple, Any, Optional, Dict

from ChemEM.data.binding_site_model import BindingSiteModel
from ChemEM.tools.geometry import compute_circumsphere, select_atoms_in_sphere
from ChemEM.tools.biomolecule import write_residues_to_pdb

import os
import numpy as np
import networkx as nx

from collections import defaultdict

from scipy.spatial import Delaunay, KDTree, distance_matrix
from scipy.cluster.hierarchy import fclusterdata

from scipy.ndimage import (distance_transform_edt,
                           center_of_mass, 
                           label,
                           generate_binary_structure,
                           binary_closing)
from scipy.spatial import cKDTree, distance
from skimage.morphology import binary_dilation
from skimage import morphology

#---------------------------------------
#Binding Site Grid tools
#---------------------------------------

def analyze_site_from_mask(
    site_mask_zyx: np.ndarray,
    protein_atoms: list,
    protein_coords_xyz: np.ndarray,
    protein_radii: np.ndarray,
    grid_origin_xyz: np.ndarray,
    grid_resolution: float,
    contact_cutoff_A: float = 2.0
) -> BindingSiteModel:
    
    #volume 
    num_voxels = np.sum(site_mask_zyx)
    site_volume = num_voxels * (grid_resolution ** 3)
    print(f"Site Volume: {site_volume:.2f} Å³ ({num_voxels} voxels)")
    #com
    centroid_indices_zyx = center_of_mass(site_mask_zyx)
    centroid_xyz = (np.array(centroid_indices_zyx)[::-1] * grid_resolution) + grid_origin_xyz
    print(f"Site Centroid (X,Y,Z): {np.round(centroid_xyz, 2)}")
    
    true_indices_zyx = np.array(np.where(site_mask_zyx))
    min_indices_zyx = true_indices_zyx.min(axis=1)
    max_indices_zyx = true_indices_zyx.max(axis=1)
    
    min_coords_xyz = (min_indices_zyx[::-1] * grid_resolution) + grid_origin_xyz
    max_coords_xyz = (max_indices_zyx[::-1] * grid_resolution) + grid_origin_xyz
    bounding_box_size = max_coords_xyz - min_coords_xyz
    
    site_voxel_coords_xyz = (true_indices_zyx.T[:, ::-1] * grid_resolution) + grid_origin_xyz
    
    # Use a KDTree for efficient nearest-neighbor search.
    print("Finding bounding residues using KDTree...")
    kdtree = cKDTree(site_voxel_coords_xyz)
    distances_to_site, _ = kdtree.query(protein_coords_xyz)
    
    # An atom is a "contact" if its distance to the site is less than its
    # own radius plus a cutoff distance.
    contact_atom_mask = distances_to_site < (protein_radii + contact_cutoff_A)
    unique_atom_indices = np.where(contact_atom_mask)[0]
    
    binding_site_atom_mask = distances_to_site < (protein_radii + 2.0)
    binding_site_atom_indices = np.where(binding_site_atom_mask)[0]
    # Get the unique residues from the contacting atoms
    bounding_residues = list(set(protein_atoms[i].residue for i in unique_atom_indices))
    lining_residues = list(set(protein_atoms[i].residue for i in binding_site_atom_indices))
    
    data = {
       'volume': site_volume,
       'binding_site_centroid': centroid_xyz,
       'min_coords': min_coords_xyz,
       'max_coords': max_coords_xyz,
       'bounding_box_size': bounding_box_size,
       'residues': bounding_residues,
       'unique_atom_indices': unique_atom_indices,
       'lining_residues': lining_residues,
       'source': 'Voxel Mask Analysis'
    }
    
    boundry_mask = find_binding_site_boundary(site_mask_zyx)
    distance_map = compute_distance_map(boundry_mask, grid_resolution)
    distance_map *= site_mask_zyx > 0.1
    data['distance_map'] = distance_map
    
    binding_site = BindingSiteModel.from_dict(data)
    
    return binding_site

def find_binding_site_boundary(binding_mask):
    eroded_mask = morphology.binary_erosion(binding_mask)
    boundary_mask = binding_mask ^ eroded_mask
    return boundary_mask

def compute_distance_map(boundary_mask, apix):
    inverted_boundary = ~boundary_mask
    distance_map = distance_transform_edt(inverted_boundary, sampling=apix)
    return distance_map

def find_site_at_point(
    pocket_grid_zyx: np.ndarray,
    point_xyz: tuple[float, float, float],
    grid_origin_xyz: np.ndarray,
    grid_resolution: float
) -> np.ndarray | None:
   
    point_vec = np.array(point_xyz)
    origin_vec = np.array(grid_origin_xyz)
    indices_float = (point_vec - origin_vec) / grid_resolution
    indices_xyz = np.round(indices_float).astype(int)

    # Convert (x, y, z) indices to NumPy (z, y, x) index order
    indices_zyx = (indices_xyz[2], indices_xyz[1], indices_xyz[0])
    if not all(0 <= idx < dim for idx, dim in zip(indices_zyx, pocket_grid_zyx.shape)):
        print("[Error] The specified point is outside the grid boundaries.")
        return None

    if not pocket_grid_zyx[indices_zyx]:
        print("[Error] The specified point is not within any binding site.")
        return None

    labeled_array, num_features = label(pocket_grid_zyx)
    
    if num_features == 0:
        return None 

    target_label = labeled_array[indices_zyx]
    final_mask = (labeled_array == target_label)
    return final_mask


def get_pocket_mask(protein_mask, apix , min_wall_voxels=1, min_pocket_voxels=5):
    
    grid_shape = protein_mask.shape
    pocket_masks = []
    
    for axis in range(3): # 0=X, 1=Y, 2=Z
        current_pocket_mask = np.zeros(grid_shape, dtype=bool)
        
        scan_mask = np.moveaxis(protein_mask, axis, 0)
        
        it = np.nditer(scan_mask[0, ...], flags=['multi_index'])
        while not it.finished:
            y, z = it.multi_index
            scan_rod = scan_mask[:, y, z] 
    
            in_protein = False
            pocket_candidate_start = -1
            wall_thickness = 0
            
            for x, is_protein in enumerate(scan_rod):
                if is_protein:
                    wall_thickness += 1
                    if pocket_candidate_start != -1:
                        if wall_thickness >= min_wall_voxels:
                            pocket_len = x - pocket_candidate_start
                            if pocket_len >= min_pocket_voxels:
                                indices = np.arange(pocket_candidate_start, x)
                                if axis == 0: current_pocket_mask[indices, y, z] = True
                                elif axis == 1: current_pocket_mask[y, indices, z] = True
                                else: current_pocket_mask[y, z, indices] = True
                        pocket_candidate_start = -1 # Reset
                else: # in solvent
                    if in_protein and wall_thickness >= min_wall_voxels:
                        pocket_candidate_start = x
                    wall_thickness = 0
                
                in_protein = is_protein
            
            it.iternext()
        pocket_masks.append(current_pocket_mask)
    
    if not pocket_masks:
        return None
    
    final_pocket_mask = (pocket_masks[0].astype(np.int8) + 
                         pocket_masks[1].astype(np.int8) + 
                         pocket_masks[2].astype(np.int8)) >= 2
    
    struct = generate_binary_structure(3, 1) # Use simple connectivity
    final_pocket_mask = binary_closing(final_pocket_mask, structure=struct, iterations=2)
    
    if not np.any(final_pocket_mask):
        print("[Warning] Line-scan method did not identify a qualifying pocket.")
        return None
    else:
        return final_pocket_mask

def calculate_ses_grid_zyx(
    atom_coords: np.ndarray,
    atom_radii: np.ndarray,
    probe_radius: float = 1.4,
    grid_resolution: float = 0.5
) -> (np.ndarray, np.ndarray, np.ndarray):
    
    if atom_coords.shape[0] != atom_radii.shape[0]:
        raise ValueError("atom_coords and atom_radii must have the same number of atoms.")

    print(f"Calculating TRUE SES for {len(atom_radii)} atoms...")
    print(f"Probe radius: {probe_radius} Å, Grid resolution: {grid_resolution} Å")

    buffer = probe_radius + grid_resolution
    min_coords = np.min(atom_coords - atom_radii[:, np.newaxis], axis=0) - buffer
    max_coords = np.max(atom_coords + atom_radii[:, np.newaxis], axis=0) + buffer
    grid_origin = min_coords

    x_range = np.arange(min_coords[0], max_coords[0], grid_resolution)
    y_range = np.arange(min_coords[1], max_coords[1], grid_resolution)
    z_range = np.arange(min_coords[2], max_coords[2], grid_resolution)

    gx, gy, gz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    grid_points = np.vstack([gx.ravel(), gy.ravel(), gz.ravel()]).T

    print(f"Grid dimensions (X, Y, Z): ({len(x_range)}, {len(y_range)}, {len(z_range)})")

    dist_grid_to_atoms = distance.cdist(grid_points, atom_coords)
    dist_to_vdw_surfaces = dist_grid_to_atoms - atom_radii
    min_dist_to_surface = np.min(dist_to_vdw_surfaces, axis=1)

    sas_mask_flat = min_dist_to_surface < probe_radius
    sas_mask_xyz = sas_mask_flat.reshape(len(x_range), len(y_range), len(z_range))
    solvent_centers_flat = min_dist_to_surface >= probe_radius
    solvent_centers_xyz = solvent_centers_flat.reshape(len(x_range), len(y_range), len(z_range))
    krnl_radius_grid = int(np.ceil(probe_radius / grid_resolution))
    k_range = np.arange(-krnl_radius_grid, krnl_radius_grid + 1)
    kx, ky, kz = np.meshgrid(k_range, k_range, k_range, indexing='ij')
    dilation_kernel = (kx**2 + ky**2 + kz**2) * grid_resolution**2 <= probe_radius**2
    
    print(f"Generated dilation kernel of shape: {dilation_kernel.shape}")

    print("Dilating solvent centers to find full solvent volume (this may take a moment)...")
    solvent_volume_xyz = binary_dilation(solvent_centers_xyz, dilation_kernel)

    ses_mask_xyz = sas_mask_xyz & ~solvent_volume_xyz
    ses_mask_zyx = ses_mask_xyz.transpose(2, 1, 0)
    
    grid_coords_xyz = grid_points.reshape(len(x_range), len(y_range), len(z_range), 3)

    print("Calculation complete.")
    print(f"Returning SES mask with shape (Z, Y, X): {ses_mask_zyx.shape}")
    
    return ses_mask_zyx, grid_coords_xyz, grid_origin

def generate_density_map(points, radii, grid_spacing=1.0, filename="density_map.mrc"):
    
    min_coords = np.min(points - radii[:, np.newaxis], axis=0)
    max_coords = np.max(points + radii[:, np.newaxis], axis=0)
    
    grid_shape = np.ceil((max_coords - min_coords) / grid_spacing).astype(int) + 1
    density_grid = np.zeros(grid_shape, dtype=np.float32)
    origin = min_coords

    for point, radius in zip(points, radii):
        
        lower = np.floor((point - radius - origin) / grid_spacing).astype(int)
        upper = np.ceil((point + radius - origin) / grid_spacing).astype(int)
        
        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, np.array(grid_shape))
        
        for x in range(lower[0], upper[0]):
            for y in range(lower[1], upper[1]):
                for z in range(lower[2], upper[2]):
                    # Calculate the real position of the grid point
                    grid_point = origin + np.array([x, y, z]) * grid_spacing
                    distance = np.linalg.norm(grid_point - point)
                    if distance <= radius:
                        density_grid[x, y, z] += np.exp(-distance**2 / (2 * (radius / 2.0)**2))

    density_grid = density_grid.astype(np.float32)
    density_grid = np.transpose(density_grid, (2, 1, 0)).astype(np.float32)
   
    return origin, density_grid.shape, density_grid

def create_binding_site_mask(density_shape, point, radii, apix, origin):
    binding_mask = np.zeros(density_shape, dtype=bool)
    for center_real, radius in zip(point,radii):
        
        center_voxel = (center_real - origin) / np.array(apix) #xyz
        center_voxel = np.round(center_voxel).astype(int) #xyz
        
        z, y, x = np.ogrid[:density_shape[0], :density_shape[1], :density_shape[2]]
        real_z = z * apix[2] + origin[2]
        real_y = y * apix[1] + origin[1]
        real_x = x * apix[0] + origin[0]
        distance_squared = ((real_x - center_real[0]) ** 2 +
                            (real_y - center_real[1]) ** 2 +
                            (real_z - center_real[2]) ** 2)
        sphere_mask = distance_squared <= radius ** 2
        binding_mask |= sphere_mask
    return binding_mask

#---------------------------------------
# Alpha Shape / Delaunay Tools
#---------------------------------------






def compute_alpha_shape_pockets(
    positions: np.ndarray,
    probe_min: float,
    probe_max: float,
    first_pass_thr: float,
    first_pass_min_size: int,
    second_pass_thr: float,
    third_pass_thr: float,
    n_overlaps: int,
    return_delaunay
) -> Dict[int, List[int]]:
    """
    Computes binding site pockets using an alpha-shape (Delaunay triangulation) approach.
    
    Args:
        positions: (N, 3) numpy array of atom positions.
        probe_min: Minimum probe sphere radius.
        probe_max: Maximum probe sphere radius.
        first_pass_thr: Distance threshold for initial clustering of tetrahedra.
        first_pass_min_size: Minimum number of tetrahedra in a first-pass cluster.
        second_pass_thr: Distance threshold for clustering centroids of first-pass clusters.
        third_pass_thr: Threshold for merging overlapping clusters (third pass).
        n_overlaps: Number of links required to merge clusters in the third pass.
        
    Returns:
        A dictionary mapping site labels (int) to lists of tetrahedron indices (List[int]).
        These indices refer to the simplices in the locally computed Delaunay triangulation.
        Note: The Delaunay object itself is not returned, so indices are only valid
        if the triangulation is recomputed or if the tetrahedra are extracted immediately
        (which this function does not do; it returns indices relative to the computed simplices).
        
        To make this robust, we return a dictionary where values are lists of 
        actual tetrahedron vertex INDICES (from the input `positions`).
        e.g. {0: [[idx1, idx2, idx3, idx4], ...], ...}
    """
    
    
    delaunay = Delaunay(positions)
    tetrahedra = delaunay.simplices
    
    
    candidate_indices = []
    candidate_centers = []
    
    # TODO! Vectorise circumsphere computation 

    for i, tetra in enumerate(tetrahedra):
        vertices = positions[tetra]
        center, radius = compute_circumsphere(vertices)
        
        if probe_min <= radius <= probe_max:
            candidate_indices.append(i)
            candidate_centers.append(center)
            
    if not candidate_indices:
        return {}

    candidate_centers = np.array(candidate_centers)
    candidate_tetrahedra = tetrahedra[candidate_indices] # Shape (M, 4)
    
    #
    # Cluster circumsphere centers to group nearby tetrahedra
    labels = fclusterdata(candidate_centers, t=first_pass_thr, criterion='distance')
    
    clusters_pass1 = defaultdict(list)
    for label, tetra_indices in zip(labels, candidate_tetrahedra):
        clusters_pass1[label].append(tetra_indices)
        
    # Filter by size
    first_pass_clusters = {
        lbl: tetras for lbl, tetras in clusters_pass1.items() 
        if len(tetras) >= first_pass_min_size
    }
    
    if not first_pass_clusters:
        return {}

    
    #second pass cluster
    cluster_centroids = []
    old_labels = []
    
    for label, tetras in first_pass_clusters.items():
        # Compute centroid of this cluster (mean of circumcenters of its tetrahedra)
        # Note: We need to re-compute circumcenters or store them. 
        # Recomputing for simplicity of data flow.
        centers = []
        for tet in tetras:
            c, _ = compute_circumsphere(positions[tet])
            centers.append(c)
        
        centroid = np.mean(centers, axis=0)
        cluster_centroids.append(centroid)
        old_labels.append(label)
        
    labels_2 = fclusterdata(cluster_centroids, t=second_pass_thr, criterion='distance')
    
    second_pass_clusters = defaultdict(list)
    for new_label, old_label in zip(labels_2, old_labels):
        second_pass_clusters[new_label].extend(first_pass_clusters[old_label])
        
    # 5. Third Pass Clustering (Overlap/Connectivity)
    ordered_labels = list(second_pass_clusters.keys())
    N_pockets = len(ordered_labels)
    label_to_index = {label: i for i, label in enumerate(ordered_labels)}
    
    linkage_counts = [{} for _ in range(N_pockets)]
    
    # Flatten all sphere centers for KDTree
    all_centers = []
    all_indices = [] # stores the pocket index (0 to N_pockets-1)
    
    for label, tetras in second_pass_clusters.items():
        pocket_index = label_to_index[label]
        for tet in tetras:
            c, _ = compute_circumsphere(positions[tet])
            all_centers.append(c)
            all_indices.append(pocket_index)
            
    all_centers = np.array(all_centers)
    tree = KDTree(all_centers)
    
    # Query neighbors
    # For every sphere, find neighbors within threshold
    # If neighbor belongs to different pocket, increment linkage
    query_results = tree.query_ball_point(all_centers, third_pass_thr)
    
    for i, idxs in enumerate(query_results):
        pid = all_indices[i]
        for nb_idx in idxs:
            if nb_idx == i: continue
            
            nb_pid = all_indices[nb_idx]
            if nb_pid != pid:
                if nb_pid not in linkage_counts[pid]:
                    linkage_counts[pid][nb_pid] = 0
                linkage_counts[pid][nb_pid] += 1

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(N_pockets))
    
    for a in range(N_pockets):
        for b, count in linkage_counts[a].items():
            if count >= n_overlaps:
                G.add_edge(a, b)
                
    merged_components = list(nx.connected_components(G))
    
    final_clusters = {}
    for new_id, component in enumerate(merged_components):
        merged_tetras = []
        for idx in component:
            original_label = ordered_labels[idx]
            merged_tetras.extend(second_pass_clusters[original_label])
        final_clusters[new_id] = merged_tetras
    
    
    if return_delaunay:
        return final_clusters, delaunay
    return final_clusters

#---------------------------------------
# Fallback / SES Extraction Tools
#---------------------------------------

def ses_ray_trace_binding_site(
    centroid: np.ndarray,
    centroid_key: int | str,
    atoms: list,
    positions: np.ndarray,
    atom_radii: np.ndarray,
    protein_openff_structure, # passed for writing PDBs
    system_output_dir: str,
    grid_spacing: float,
    fall_back_radius: float = 15.0
) -> Optional[BindingSiteModel]:
    """
    Performs the fallback binding site extraction using SES (Solvent Excluded Surface)
    when no Alpha Shape clusters are found near a centroid.
    
    Args:
        centroid: (3,) numpy array of the target centroid coordinates.
        centroid_key: Identifier for this site (used in filenames).
        atoms: List of atom objects (corresponding to positions).
        positions: (N, 3) numpy array of all protein atom coordinates.
        atom_radii: (N,) numpy array of all protein atom radii.
        protein_openff_structure: Structure object capable of being passed to write_residues_to_pdb.
        system_output_dir: Directory to write output files (PDBs, MRCs, tags).
        grid_spacing: Grid resolution in Angstroms.
        fall_back_radius: Radius to select atoms around the centroid for SES calculation.
        
    Returns:
        BindingSiteModel object if successful, None otherwise.
    """
    
    # 1. Select atoms in local sphere
    coords, rad = select_atoms_in_sphere(positions,
                                         atom_radii,
                                         centroid,
                                         fall_back_radius)
    
    # 2. Calculate SES Grid
    ses_mask, grid_coords, grid_origin = calculate_ses_grid_zyx(
        coords,
        rad,
        probe_radius=1.4,
        grid_resolution=grid_spacing
    )
    
    ses_mask = ses_mask.astype(int)
    
    # 3. Get Pocket Mask (Line Scan)
    pocket_mask = get_pocket_mask(ses_mask, (grid_spacing, grid_spacing, grid_spacing))
    
    # 4. Filter for specific site at centroid
    final_pocket_mask = find_site_at_point(
        pocket_mask, 
        centroid,
        grid_origin,
        grid_spacing
    )
    
    if final_pocket_mask is None:
        return None

    # 5. Analyze Site Properties
    site_data = analyze_site_from_mask(
        final_pocket_mask,
        atoms,
        positions,
        atom_radii,
        grid_origin,
        grid_spacing,
        contact_cutoff_A=6.0
    )
    
    # 6. Write Output Files (PDBs)
    pdb_path = os.path.join(system_output_dir, f'site_{centroid_key}.pdb')
    rdkit_mol = write_residues_to_pdb(
        site_data['residues'], 
        protein_openff_structure.positions, 
        pdb_path
    )
    
    pdb_lining_path = os.path.join(system_output_dir, f'site_lining_residues_{centroid_key}.pdb')
    rdkit_lining_mol = write_residues_to_pdb(
        site_data['lining_residues'], 
        protein_openff_structure.positions, 
        pdb_lining_path
    )
    
    # 7. Finalize Data Dictionary
    site_data['rdkit_mol'] = rdkit_mol
    site_data['lining_mol'] = rdkit_lining_mol
    
    # Add grid metadata
    site_data['origin'] = grid_origin 
    site_data['box_size'] = final_pocket_mask.shape 
    site_data['densmap'] = final_pocket_mask 
    site_data['apix'] = (grid_spacing, grid_spacing, grid_spacing)
    site_data['key'] = centroid_key
    
    # Create Model
    binding_site_model = BindingSiteModel.from_dict(site_data)
    
    # Write fallback tag
    source_tag = os.path.join(system_output_dir, 'fallback.tag')
    with open(source_tag, 'w') as f:
        f.write('')
        
    return binding_site_model





def compute_site_spheres(
    positions: np.ndarray,
    tetras: Sequence[Sequence[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each tetra (4 indices into `positions`), compute circumsphere center+radius.
    Returns:
        site_centers: (n, 3)
        site_radii:   (n,)
    """
    centers: List[np.ndarray] = []
    radii: List[float] = []

    for tetra in tetras:
        vertices = positions[np.asarray(tetra, dtype=int)]
        c, r = compute_circumsphere(vertices)
        centers.append(c)
        radii.append(r)

    return np.asarray(centers), np.asarray(radii)


def contact_atom_indices(
    site_centers: np.ndarray,
    site_radii: np.ndarray,
    atom_positions: np.ndarray,
    atom_radii: np.ndarray,
    extra_distance: float,
) -> np.ndarray:
    """
    Returns unique atom indices that contact ANY site sphere, where contact is:
        dist(center_i, atom_j) < site_radii[i] + atom_radii[j] + extra_distance
    """
    dist = distance_matrix(site_centers, atom_positions)  # (n_centers, n_atoms)
    radii_sum = site_radii[:, None] + atom_radii[None, :]
    contacts = dist < (radii_sum + extra_distance)
    return np.unique(np.where(contacts)[1])


def residues_from_atom_indices(
    atom_indices: Iterable[int],
    atoms: Sequence[Any],
) -> List[Any]:
    """
    Maps atom indices -> unique residues preserving first-seen order.
    Expects atoms[idx].residue exists.
    """
    seen = set()
    residues: List[Any] = []
    for idx in atom_indices:
        res = atoms[int(idx)].residue
        if res not in seen:
            seen.add(res)
            residues.append(res)
    return residues


def box_stats(site_centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        centroid, min_coords, max_coords, bounding_box_size
    """
    centroid = np.mean(site_centers, axis=0)
    min_coords = np.min(site_centers, axis=0)
    max_coords = np.max(site_centers, axis=0)
    return centroid, min_coords, max_coords, (max_coords - min_coords)


def density_and_distance_maps(
    site_centers: np.ndarray,
    site_radii: np.ndarray,
    grid_spacing: float,
    denmap_path: Optional[str] = None,
    density_cutoff: float = 0.1,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], np.ndarray, np.ndarray, Tuple[float, float, float], float]:
    """
    Builds density map and a boundary-derived distance map.

    Returns:
        origin, box_size, densmap, distance_map, apix, volume
    """
    origin, box_size, densmap = generate_density_map(
        site_centers,
        site_radii,
        grid_spacing=grid_spacing,
        filename=denmap_path,
    )

    apix = (grid_spacing, grid_spacing, grid_spacing)

    site_mask = create_binding_site_mask(
        densmap.shape,
        site_centers,
        site_radii,
        apix,
        origin,
    )
    boundary_mask = find_binding_site_boundary(site_mask)
    distance_map = compute_distance_map(boundary_mask, grid_spacing)

    # Match your original behavior: mask distance_map where density is low
    distance_map *= (densmap > density_cutoff)
    volume = float(np.sum(densmap > density_cutoff) * (grid_spacing ** 3))

    return origin, box_size, densmap, distance_map, apix, volume

#-----
#manual binding site defintion
#------

def make_grid_and_origin(
    box_size,
    centroid,
    grid_spacing,
    *,
    order="xyz",
    dtype=np.float32,
    ensure_odd=True,
    pad_voxels=0,
):
    c = np.asarray(centroid, dtype=float).reshape(3)

    if np.isscalar(box_size):
        L = np.array([float(box_size)] * 3, dtype=float)
    else:
        L = np.asarray(box_size, dtype=float).reshape(3)

    if grid_spacing <= 0:
        raise ValueError("grid_spacing must be > 0")

    n_xyz = np.ceil(L / float(grid_spacing)).astype(int)

    if ensure_odd:
        n_xyz += (n_xyz % 2 == 0).astype(int)

    if pad_voxels:
        n_xyz = n_xyz + 2 * int(pad_voxels)

    nx, ny, nz = map(int, n_xyz)

    half_extent = ((n_xyz - 1) * float(grid_spacing)) / 2.0
    origin = c - half_extent  # voxel-center at index 0 (x,y,z)

    if order == "zyx":
        grid = np.zeros((nz, ny, nx), dtype=dtype)
    elif order == "xyz":
        grid = np.zeros((nx, ny, nz), dtype=dtype)
    else:
        raise ValueError("order must be 'zyx' or 'xyz'")

    return grid, origin, (nx, ny, nz)


def make_grid_and_origin_from_radius(
    radius,
    centroid,
    grid_spacing,
    *,
    order="xyz",
    dtype=np.float32,
    ensure_odd=True,
    pad_voxels=0,
):
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if grid_spacing <= 0:
        raise ValueError("grid_spacing must be > 0")

    c = np.asarray(centroid, dtype=float).reshape(3)
    r = float(radius)
    s = float(grid_spacing)

    n = int(np.ceil((2.0 * r) / s))
    n_xyz = np.array([n, n, n], dtype=int)

    if ensure_odd:
        n_xyz += (n_xyz % 2 == 0).astype(int)

    if pad_voxels:
        n_xyz = n_xyz + 2 * int(pad_voxels)

    nx, ny, nz = map(int, n_xyz)

    half_extent = ((n_xyz - 1) * s) / 2.0
    origin = c - half_extent  # (x,y,z)

    if order == "zyx":
        grid = np.zeros((nz, ny, nx), dtype=dtype)
    elif order == "xyz":
        grid = np.zeros((nx, ny, nz), dtype=dtype)
    else:
        raise ValueError("order must be 'zyx' or 'xyz'")

    return grid, origin, (nx, ny, nz)


def _centroid_to_index(centroid_xyz, origin_xyz, grid_spacing, shape_xyz, order="zyx"):
    """Nearest voxel index to centroid (voxel centers)."""
    c = np.asarray(centroid_xyz, float)
    o = np.asarray(origin_xyz, float)
    s = float(grid_spacing)
    nx, ny, nz = map(int, shape_xyz)

    i = int(np.rint((c[0] - o[0]) / s))
    j = int(np.rint((c[1] - o[1]) / s))
    k = int(np.rint((c[2] - o[2]) / s))

    i = min(max(i, 0), nx - 1)
    j = min(max(j, 0), ny - 1)
    k = min(max(k, 0), nz - 1)

    return (k, j, i) if order == "zyx" else (i, j, k)


def _paint_excluded_spheres(excluded, origin_xyz, shape_xyz, grid_spacing, atom_positions, atom_radii, probe_radius, order="zyx"):
    """
    excluded: bool array (True = excluded)
    Paint spheres of radius (atom_radii + probe_radius) onto excluded grid using bounded sub-boxes.
    """
    ox, oy, oz = map(float, origin_xyz)
    nx, ny, nz = map(int, shape_xyz)
    s = float(grid_spacing)

    pos = np.asarray(atom_positions, dtype=float)
    rad = np.asarray(atom_radii, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("atom_positions must be shape (N,3)")
    if rad.ndim != 1 or rad.shape[0] != pos.shape[0]:
        raise ValueError("atom_radii must be shape (N,) matching atom_positions")

    pr = float(probe_radius)

    for (cx, cy, cz), r0 in zip(pos, rad):
        r = float(r0) + pr
        if not np.isfinite(r) or r <= 0:
            continue

        # xyz index bounds (voxel centers)
        i0 = int(np.floor((cx - r - ox) / s))
        i1 = int(np.ceil((cx + r - ox) / s))
        j0 = int(np.floor((cy - r - oy) / s))
        j1 = int(np.ceil((cy + r - oy) / s))
        k0 = int(np.floor((cz - r - oz) / s))
        k1 = int(np.ceil((cz + r - oz) / s))

        # clamp
        i0 = max(i0, 0); j0 = max(j0, 0); k0 = max(k0, 0)
        i1 = min(i1, nx - 1); j1 = min(j1, ny - 1); k1 = min(k1, nz - 1)
        if i1 < i0 or j1 < j0 or k1 < k0:
            continue

        xs = ox + np.arange(i0, i1 + 1, dtype=float) * s
        ys = oy + np.arange(j0, j1 + 1, dtype=float) * s
        zs = oz + np.arange(k0, k1 + 1, dtype=float) * s

        r2 = r * r

        if order == "zyx":
            zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")  # (k,j,i)
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= r2
            sub = excluded[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1]
            sub[mask] = True
        else:
            xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")  # (i,j,k)
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= r2
            sub = excluded[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1]
            sub[mask] = True


def sasa_accessible_component_mask(
    *,
    atom_positions: np.ndarray,
    atom_radii: np.ndarray,
    centroid: np.ndarray,
    grid_spacing: float,
    probe_radius: float = 1.4,   # water ~1.4 Å; set ~0.7 if you want "half water radius"
    # one of these:
    box_size=None,
    radius=None,
    # grid options
    order: str = "zyx",
    ensure_odd: bool = True,
    pad_voxels: int = 0,
    # component selection
    connectivity: int = 1,  # 1->6N, 2->18N, 3->26N
    keep_mode: str = "contains_or_closest",  # or "contains_only" / "closest_only"
    return_labels: bool = False,
):
    """
    Build probe-accessibility grid (excluded vs accessible), label accessible components,
    and keep the component containing the centroid voxel (or closest if centroid is excluded).

    Returns
    -------
    excluded : bool array
        True where voxel center is within (atom_radius + probe_radius) of any atom.
    accessible : bool array
        ~excluded
    kept : bool array
        Selected connected component within accessible.
    origin_xyz : (3,) float
    shape_xyz : (nx,ny,nz)
    (optional) labels, n_labels
    """
    if (box_size is None) == (radius is None):
        raise ValueError("Provide exactly one of box_size or radius.")

    centroid = np.asarray(centroid, dtype=float).reshape(3)

    if box_size is not None:
        _, origin, shape_xyz = make_grid_and_origin(
            box_size, centroid, grid_spacing,
            order=order, dtype=np.float32, ensure_odd=ensure_odd, pad_voxels=pad_voxels
        )
        # allocate excluded grid in chosen order
        nx, ny, nz = shape_xyz
        excluded = np.zeros((nz, ny, nx), dtype=bool) if order == "zyx" else np.zeros((nx, ny, nz), dtype=bool)
    else:
        _, origin, shape_xyz = make_grid_and_origin_from_radius(
            radius, centroid, grid_spacing,
            order=order, dtype=np.float32, ensure_odd=ensure_odd, pad_voxels=pad_voxels
        )
        nx, ny, nz = shape_xyz
        excluded = np.zeros((nz, ny, nx), dtype=bool) if order == "zyx" else np.zeros((nx, ny, nz), dtype=bool)

    # 1) paint excluded volume for probe centers
    _paint_excluded_spheres(
        excluded, origin, shape_xyz, grid_spacing,
        atom_positions, atom_radii, probe_radius, order=order
    )

    # 2) accessible space and labeling
    accessible = ~excluded

    structure = generate_binary_structure(3, connectivity)
    labels, nlab = label(accessible, structure=structure)

    kept = np.zeros_like(accessible, dtype=bool)
    if nlab == 0:
        if return_labels:
            return excluded, accessible, kept, origin, tuple(shape_xyz), labels, nlab
        return excluded, accessible, kept, origin, tuple(shape_xyz)

    c_idx = _centroid_to_index(centroid, origin, grid_spacing, shape_xyz, order=order)

    # 3) choose component
    chosen = 0
    if keep_mode in ("contains_only", "contains_or_closest"):
        chosen = int(labels[c_idx])

    if chosen == 0 and keep_mode in ("closest_only", "contains_or_closest"):
        # centroid lies in excluded (or in background): find nearest accessible voxel using distance transform
        # distance_transform_edt computes distance to nearest zero for non-zero entries,
        # so use excluded as input (excluded=True -> non-zero), nearest zero -> accessible voxel.
        dist, inds = distance_transform_edt(excluded, return_indices=True)
        nearest_idx = tuple(int(inds[d][c_idx]) for d in range(3))
        chosen = int(labels[nearest_idx])

    if chosen != 0:
        kept = labels == chosen

    if return_labels:
        return excluded, accessible, kept, origin, tuple(shape_xyz), labels, nlab
    return excluded, accessible, kept, origin, tuple(shape_xyz)




