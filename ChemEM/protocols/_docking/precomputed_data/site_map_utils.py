#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 15:05:03 2026

@author: aaron.sweeney
"""
import numpy as np
import time
from scipy.ndimage import (generate_binary_structure,
                           binary_closing, 
                           distance_transform_edt, 
                           binary_dilation,
                           gaussian_filter)
from ChemEM import grid_maps
from ChemEM.tools.density import crop_map_around_point
from ChemEM.data.data import XlogP3
from scipy import signal
from openmm import unit, NonbondedForce
#----------SASA mask-----

def get_sasa_mask(
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
    structuring_element = generate_binary_structure(3, 1)
    sasa_mask = binary_closing(
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
   

    rt = time.perf_counter() - t1
    print("final_sasa_mask:", rt)

    return bulk_solvent_mask, protein_mask, sasa_mask#, delta_sasa_generic_map


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


def get_solvent_depth_mask(
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


def get_flow_mask(
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

    # Solvent region = internal SASA
    solvent_mask = sasa_mask

    #Seeds: SASA voxels that touch bulk
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
    
    seed_mask = solvent_mask & bulk_dilated
    
    free_space = solvent_mask | bulk_mask
    r_local = distance_transform_edt(free_space).astype(np.float32)
    r_local *= sasa_mask.astype(np.float32)
    # If you want to factor in apix:
    # r_local *= float(apix)
    
    R_bottle, flow_norm = _widest_path_cpp(
        r_local,
        solvent_mask,
        seed_mask,
        connectivity=connectivity,
        power=power,
    )
    

    return r_local, R_bottle, flow_norm, seed_mask



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



def smooth_env_map_diffusion(env_index: np.ndarray,
                             sasa_mask: np.ndarray,
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
    mask = np.asarray(sasa_mask, dtype=bool)

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


def get_hydrophobic_grid(
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


def get_electrostatic_map(
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



