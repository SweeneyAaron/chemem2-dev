# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

from ChemEM.data.binding_site_model import BindingSiteModel
from ChemEM.parsers.EMMap import EMMap
import math
import numpy as np
from typing import List, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign

from scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import (map_coordinates, 
                           distance_transform_edt,
                           center_of_mass, 
                           label,
                           fourier_gaussian,
                           gaussian_filter,
                           generate_binary_structure,
                           binary_closing,
                           sobel,
                           label,
                           grey_opening,
                           grey_closing,
                           binary_dilation,
                           binary_propagation,
                           find_objects)



from scipy.spatial import cKDTree, distance
#from skimage.morphology import binary_dilation
from skimage import morphology
from skimage.filters import threshold_otsu

# -----------------------
# Small shared utilities
# -----------------------

def _as_xyz(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.size == 1:
        return np.array([float(v), float(v), float(v)], dtype=np.float64)
    return v


def _map_extent_xyz(emap):
    """
    Returns (x0,x1,y0,y1,z0,z1) in Å for an EMMap.
    EMMap.density_map shape is (z,y,x).
    """
    ox, oy, oz = map(float, emap.origin)
    ax, ay, az = _as_xyz(emap.apix)
    nz, ny, nx = emap.density_map.shape

    x0, x1 = ox, ox + ax * nx
    y0, y1 = oy, oy + ay * ny
    z0, z1 = oz, oz + az * nz
    return x0, x1, y0, y1, z0, z1


# -----------------------------------------
# MapTools 
# -----------------------------------------

class MapTools:
    @staticmethod
    def get_mi_weight(resolution: float) -> int:
        if resolution <= 5.0:
            return 100
        return 20

    @staticmethod
    def get_map_bounds(dens_map, pad: float = 0.0):
        if dens_map is None:
            return None

        ap = _as_xyz(dens_map.apix)
        map_bounds = [
            dens_map.x_origin + pad,
            (dens_map.x_origin + (dens_map.x_size * round(float(ap[0]), 2))) - pad,
            dens_map.y_origin + pad,
            (dens_map.y_origin + (dens_map.y_size * round(float(ap[1]), 2))) - pad,
            dens_map.z_origin + pad,
            (dens_map.z_origin + (dens_map.z_size * round(float(ap[2]), 2))) - pad,
        ]
        return map_bounds

    @staticmethod
    def get_map_segement(densmap, box_limits, threshold: float = 0.0):
        ap = _as_xyz(densmap.apix)

        x_min = math.ceil((box_limits[0] - densmap.x_origin) / ap[0])
        x_max = round((box_limits[1] - densmap.x_origin) / ap[0])

        y_min = math.ceil((box_limits[2] - densmap.y_origin) / ap[1])
        y_max = round((box_limits[3] - densmap.y_origin) / ap[1])

        z_min = math.ceil((box_limits[4] - densmap.z_origin) / ap[2])
        z_max = round((box_limits[5] - densmap.z_origin) / ap[2])

        new_map = densmap.copy()
        new_map.density_map = new_map.density_map[z_min:z_max, y_min:y_max, x_min:x_max]

        x_origin = new_map.x_origin + (x_min * ap[0])
        y_origin = new_map.y_origin + (y_min * ap[1])
        z_origin = new_map.z_origin + (z_min * ap[2])

        new_map.origin = (x_origin, y_origin, z_origin)
        new_map.resolution = densmap.resolution
        new_map.map_contour = densmap.map_contour
        return new_map

    @staticmethod
    def blur_model(mol, resolution, emmap=None):
        """
        Blur an RDKit mol into a density map that matches the grid of `emmap`.
        """
        if emmap is None:
            raise ValueError("blur_model requires an EMMap instance as `emmap` (template grid).")
        prot = MapTools._prepare_molmap_input(mol)
        densmap = MapTools.gaussian_blur(prot, resolution, emmap)
        return densmap

    @staticmethod
    def _prepare_molmap_input(mol, n_conf: int = 0):
        molmap_input = []
        atm_coords = mol.GetConformers()[n_conf].GetPositions()
        atom_props = mol.GetAtoms()

        for coord, prop in zip(atm_coords, atom_props):
            x = float(coord[0])
            y = float(coord[1])
            z = float(coord[2])
            atm_type = prop.GetSymbol()
            mass = float(prop.GetMass())
            atom_input = [atm_type, x, y, z, mass]
            molmap_input.append(atom_input)
        return molmap_input

    @staticmethod
    def gaussian_blur(
        prot,
        resolution,
        densMap,
        sigma_coeff: float = 0.356,
        normalise: bool = True,
    ):
        newMap = densMap.copy()
        newMap.density_map = np.zeros(densMap.box_size)

        sigma = sigma_coeff * resolution
        newMap = MapTools.make_atom_overlay_map(newMap, prot)

        fou_map = fourier_gaussian(fftn(newMap.density_map), sigma)
        newMap.density_map = np.real(ifftn(fou_map))

        if normalise:
            newMap.normalise()

        return newMap

    @staticmethod
    def make_atom_overlay_map(densMap, prot):
        densMap = densMap.copy()
        for atom in prot:
            pos = MapTools.mapGridPosition(densMap, atom)
            if pos:
                densMap.density_map[pos[2]][pos[1]][pos[0]] += pos[3]
        return densMap

    @staticmethod
    def mapGridPosition(densMap, atom):
        origin = densMap.origin
        apix = _as_xyz(densMap.apix)
        box_size = densMap.box_size

        x_pos = int(round((atom[1] - origin[0]) / apix[0], 0))
        y_pos = int(round((atom[2] - origin[1]) / apix[1], 0))
        z_pos = int(round((atom[3] - origin[2]) / apix[2], 0))

        if (
            (box_size[2] > x_pos >= 0)
            and (box_size[1] > y_pos >= 0)
            and (box_size[0] > z_pos >= 0)
        ):
            return x_pos, y_pos, z_pos, atom[4]
        return 0

    @staticmethod
    def split_density(dmap, struct=None, label_threshold_sigma=None, label_threshold=None):
        """
        Takes in a density map and creates a mask of disconnected densities.
        """
        if struct is None:
            struct = np.array(
                [
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                ]
            )

        masked_map = dmap.copy()

        if label_threshold_sigma is not None:
            threshold = MapTools.map_contour(dmap, t=label_threshold_sigma)
        elif label_threshold is not None:
            threshold = label_threshold
        else:
            threshold = 0.0

        masked_map.density_map = masked_map.density_map > threshold

        labels_arr, num_features = label(masked_map.density_map, structure=struct)
        labeled_map = masked_map.copy()
        labeled_map.density_map = labels_arr

        return masked_map, labeled_map, num_features

    @staticmethod
    def map_contour(densmap, t: float = -1.0):
        """
        Calculates map contour.
        """
        c1 = None
        if t != -1.0:
            zeropeak, ave, sigma1 = MapTools.peak_density(densmap)
            if zeropeak is not None:
                c1 = zeropeak + (t * sigma1)
            else:
                c1 = 0.0
        return c1

    @staticmethod
    def peak_density(densmap):
        freq, bins = np.histogram(densmap.density_map, 1000)
        ind = np.nonzero(freq == np.amax(freq))[0]
        peak = None

        ave, sigma = np.mean(densmap.density_map), np.std(densmap.density_map)

        for i in ind:
            val = (bins[i] + bins[i + 1]) / 2.0
            if val < float(ave) + float(sigma):
                peak = val

        if peak is None:
            peak = ave

        sigma1 = None
        if peak is not None:
            mask_array = densmap.density_map[densmap.density_map > peak]
            sigma1 = np.sqrt(np.mean(np.square(mask_array - peak)))

        return peak, ave, sigma1

    @staticmethod
    def model_contour(p_map, t: float = -1.0):
        """
        Calculates model contour.
        """
        c2 = None
        if t != -1.0:
            c2 = t * p_map.std
        return c2

    @staticmethod
    def fit_map(
        density_map,
        mol,
        N: int = 100,
        max_translation: float = 2.0,
        initial_step_size: float = 0.5,
        max_steps: int = 10000,
        sigma_coeff: float = 0.356,
        num_conformers: int = 50,
        rmsd_cutoff: float = 0.5,
        max_attempts: int = 1000,
    ):
        mol_copy, atomic_numbers, atomic_masses = MapTools.generate_initial_conformers(
            mol,
            num_conformers=num_conformers,
            rmsd_cutoff=rmsd_cutoff,
            max_attempts=max_attempts,
        )

        new_mol, ccc = MapTools.fit_mol_to_map(
            density_map,
            mol_copy,
            atomic_numbers,
            atomic_masses,
            sigma_coeff=sigma_coeff,
            N=N,
            initial_step_size=initial_step_size,
            max_steps=max_steps,
            max_translation=max_translation,
        )
        return new_mol, ccc

    @staticmethod
    def fit_mol_to_map(
        density_map,
        mol,
        atomic_numbers,
        atomic_masses,
        sigma_coeff: float = 0.356,
        N: int = 100,
        initial_step_size: float = 0.5,
        max_steps: int = 10000,
        max_translation: float = 2.0,
    ):
        # Keep import local to avoid heavy import at module load time
        from ChemEM import ligand_fitting

        center_of_mass = density_map.center_of_mass()
        sigma = density_map.resolution * sigma_coeff
        D_exp_flat = density_map.flatten()
        D_exp_dims = list(density_map.box_size)
        voxel_size = float(_as_xyz(density_map.apix)[0])
        origin = np.array(density_map.origin)

        all_best_mol = None
        all_best_ccc = -1

        for conf_id in range(mol.GetNumConformers()):
            conformer = mol.GetConformer(conf_id)
            translated_coords = MapTools.translate_to_point(conformer.GetPositions(), center_of_mass)
            atom_positions = [pos.reshape(3, 1) for pos in translated_coords]

            best_ccc, best_coords, all_cccs = ligand_fitting.global_search(
                atom_positions,
                atomic_masses,
                D_exp_flat,
                D_exp_dims,
                voxel_size,
                origin,
                sigma,
                N,
                initial_step_size,
                max_steps,
                max_translation,
            )

            mol = MapTools.assign_coords(mol, best_coords, conf_num=conf_id)

            if best_ccc > all_best_ccc:
                all_best_ccc = best_ccc
                all_best_mol = conf_id

        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        conf = mol.GetConformer(all_best_mol)
        new_conf = Chem.Conformer(conf)
        new_mol.AddConformer(new_conf, assignId=True)

        return new_mol, all_best_ccc

    @staticmethod
    def assign_coords(mol, new_coords, conf_num: int = 0):
        conformer = mol.GetConformer(conf_num)
        for i in range(mol.GetNumAtoms()):
            x, y, z = new_coords[i]
            conformer.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
        return mol

    @staticmethod
    def translate_to_point(atom_coords, point):
        centroid = np.mean(atom_coords, axis=0)
        translation_vector = np.array(point) - centroid
        translated_coords = atom_coords + translation_vector
        return translated_coords

    @staticmethod
    def generate_initial_conformers(mol, num_conformers: int = 50, rmsd_cutoff: float = 0.5, max_attempts: int = 1000):
        mol_copy = Chem.Mol(mol)
        mol_confs = MapTools.generate_conformers_with_min_rmsd(
            mol_copy,
            num_conformers,
            min_rmsd=rmsd_cutoff,
            max_attempts=max_attempts,
        )
        mol_confs = Chem.RemoveAllHs(mol_confs)
        atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        atomic_masses = np.array([MapTools.get_atomic_weight(z) for z in atomic_numbers])
        return mol_confs, atomic_numbers, atomic_masses

    @staticmethod
    def get_atomic_weight(atomic_num: int) -> float:
        pt = Chem.GetPeriodicTable()
        return pt.GetAtomicWeight(int(atomic_num))
    
    
    @staticmethod 
    def trim_map_to_atoms(
        dmap,
        coords_xyz,
        pad =  10.0,
        inplace = False
    ):
        
        density_map = dmap.density_map 
        nz, ny, nx = density_map.shape
        apix = np.asarray(dmap.apix, dtype=float) 
        origin = np.asarray(dmap.origin, dtype=float) 
        coords = np.asarray(coords_xyz, dtype=float)
        

        xyz_min = coords.min(axis=0) - float(pad)
        xyz_max = coords.max(axis=0) + float(pad)
        idx_min_xyz = np.floor((xyz_min - origin) / apix).astype(int)
        idx_max_xyz = np.ceil ((xyz_max - origin) / apix).astype(int)

        mins_zyx = np.array([idx_min_xyz[2], idx_min_xyz[1], idx_min_xyz[0]], dtype=int)
        maxs_zyx = np.array([idx_max_xyz[2], idx_max_xyz[1], idx_max_xyz[0]], dtype=int)

        mins_zyx = np.maximum(mins_zyx, 0)
        maxs_zyx = np.minimum(maxs_zyx, np.array([nz-1, ny-1, nx-1], dtype=int))
        
        zslice = slice(int(mins_zyx[0]), int(maxs_zyx[0]) + 1)
        yslice = slice(int(mins_zyx[1]), int(maxs_zyx[1]) + 1)
        xslice = slice(int(mins_zyx[2]), int(maxs_zyx[2]) + 1)

        submap = density_map[zslice, yslice, xslice]

        
        min_idx_xyz = np.array([int(mins_zyx[2]), int(mins_zyx[1]), int(mins_zyx[0])], dtype=int)
        new_origin = origin + apix * min_idx_xyz
        
        if inplace:
            #modify the density map inplace
            dmap.density_map = submap 
            dmap.origin = new_origin 
            return density_map 
        
        new_map =  EMMap(new_origin, dmap.apix, submap, dmap.resolution)
        return new_map
        
    
    @staticmethod
    def generate_conformers_with_min_rmsd(mol, num_conformers: int, min_rmsd: float, max_attempts: int = 1000):
        """
        Generate multiple conformers ensuring each is at least `min_rmsd` apart.
        """
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.pruneRmsThresh = float(min_rmsd)
        params.numThreads = 0

        pool_size = int(num_conformers) * 10
        conformer_ids = AllChem.EmbedMultipleConfs(mol, numConfs=pool_size, params=params)

        selected_conformers = []
        for conf_id in conformer_ids:
            if len(selected_conformers) >= num_conformers:
                break

            is_distinct = True
            for selected_id in selected_conformers:
                rmsd = rdMolAlign.GetBestRMS(mol, mol, conf_id, selected_id)
                if rmsd < min_rmsd:
                    is_distinct = False
                    break

            if is_distinct:
                selected_conformers.append(conf_id)

        if len(selected_conformers) < num_conformers:
            print(f"Warning: Only {len(selected_conformers)} conformers generated with min_rmsd {min_rmsd}")
        else:
            print(f"Successfully generated {len(selected_conformers)} conformers with min_rmsd {min_rmsd}")

        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()

        for conf_id in selected_conformers:
            conf = mol.GetConformer(conf_id)
            new_mol.AddConformer(conf, assignId=True)

        return new_mol


# ------------------------------------------------------
# Density/EMMap helper tools 
# ------------------------------------------------------

def _compute_union_grid(maps, tol=1e-6):
    if not maps:
        raise ValueError("maps must be a non-empty list of EMMap objects.")

    apix = _as_xyz(maps[0].apix)
    for m in maps[1:]:
        ap = _as_xyz(m.apix)
        if not np.allclose(ap, apix, atol=tol, rtol=0):
            raise ValueError("apix mismatch between maps. This function assumes constant apix.")

    bounds = [_map_extent_xyz(m) for m in maps]
    x0 = min(b[0] for b in bounds)
    x1 = max(b[1] for b in bounds)
    y0 = min(b[2] for b in bounds)
    y1 = max(b[3] for b in bounds)
    z0 = min(b[4] for b in bounds)
    z1 = max(b[5] for b in bounds)

    origin = np.array([x0, y0, z0], dtype=np.float64)

    ax, ay, az = apix
    nx = int(np.ceil((x1 - x0) / ax - 1e-12))
    ny = int(np.ceil((y1 - y0) / ay - 1e-12))
    nz = int(np.ceil((z1 - z0) / az - 1e-12))

    nx = max(nx, 1)
    ny = max(ny, 1)
    nz = max(nz, 1)

    shape = (nz, ny, nx)  # (z,y,x)
    return origin, apix, shape


def _paste_or_resample_into(
    src_data,
    src_origin,
    new_data,
    new_origin,
    apix_xyz,
    tol=1e-4,
):
    apix_xyz = _as_xyz(apix_xyz)

    src_origin = np.asarray(src_origin, dtype=np.float64)
    new_origin = np.asarray(new_origin, dtype=np.float64)

    off_xyz = (src_origin - new_origin) / apix_xyz
    off_zyx = np.array([off_xyz[2], off_xyz[1], off_xyz[0]], dtype=np.float64)

    off_int = np.floor(off_zyx + 1e-12).astype(int)
    off_frac = off_zyx - off_int

    if np.all(np.abs(off_frac) <= tol):
        data_use = src_data
        off_int = np.rint(off_zyx).astype(int)
        off_frac = np.zeros(3, dtype=np.float64)
    else:
        data_use = ndi_shift(
            src_data,
            shift=off_frac.tolist(),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )

    nz_s, ny_s, nx_s = data_use.shape
    nz_n, ny_n, nx_n = new_data.shape

    z0, y0, x0 = off_int.tolist()
    z1, y1, x1 = z0 + nz_s, y0 + ny_s, x0 + nx_s

    zz0, yy0, xx0 = max(z0, 0), max(y0, 0), max(x0, 0)
    zz1, yy1, xx1 = min(z1, nz_n), min(y1, ny_n), min(x1, nx_n)

    if zz1 <= zz0 or yy1 <= yy0 or xx1 <= xx0:
        return None, None

    sz0, sy0, sx0 = zz0 - z0, yy0 - y0, xx0 - x0
    sz1, sy1, sx1 = sz0 + (zz1 - zz0), sy0 + (yy1 - yy0), sx0 + (xx1 - xx0)

    new_slices = (slice(zz0, zz1), slice(yy0, yy1), slice(xx0, xx1))
    src_slices = (slice(sz0, sz1), slice(sy0, sy1), slice(sx0, sx1))

    new_data[new_slices] += data_use[src_slices]

    region_mask = np.ones((zz1 - zz0, yy1 - yy0, xx1 - xx0), dtype=bool)
    return region_mask, new_slices


def combine_emmaps_on_union_grid(
    maps,
    method="mean",
    tol=1e-4,
    return_resampled=False,
):
    if maps is None or len(maps) == 0:
        raise ValueError("maps must be a non-empty list of EMMap objects.")

    # local import to avoid circular import (EMMap may import MapTools)
    from ChemEM.EMMap import EMMap

    new_origin, apix_xyz, new_shape = _compute_union_grid(maps, tol=tol)

    if method in ("sum", "mean"):
        acc = np.zeros(new_shape, dtype=np.float64)
        count = np.zeros(new_shape, dtype=np.int32) if method == "mean" else None
    elif method == "max":
        acc = np.full(new_shape, -np.inf, dtype=np.float64)
        count = None
    elif method == "median":
        acc = None
        count = None
    else:
        raise ValueError(f"Unknown method: {method}")

    resampled_arrays = [] if (return_resampled or method == "median") else None

    for m in maps:
        src = np.asarray(m.density_map, dtype=np.float32)

        if method == "median":
            tmp = np.zeros(new_shape, dtype=np.float32)
            _paste_or_resample_into(
                src_data=src,
                src_origin=m.origin,
                new_data=tmp,
                new_origin=new_origin,
                apix_xyz=apix_xyz,
                tol=tol,
            )
            resampled_arrays.append(tmp)
        else:
            region_mask, new_slices = _paste_or_resample_into(
                src_data=src,
                src_origin=m.origin,
                new_data=acc,
                new_origin=new_origin,
                apix_xyz=apix_xyz,
                tol=tol,
            )

            if region_mask is not None and method == "mean":
                zs, ys, xs = new_slices
                count[zs, ys, xs] += 1

            if method == "max":
                tmp = np.zeros(new_shape, dtype=np.float32)
                _paste_or_resample_into(
                    src_data=src,
                    src_origin=m.origin,
                    new_data=tmp,
                    new_origin=new_origin,
                    apix_xyz=apix_xyz,
                    tol=tol,
                )
                acc = np.maximum(acc, tmp)

    if method == "mean":
        combined_data = np.zeros(new_shape, dtype=np.float32)
        mask = count > 0
        combined_data[mask] = (acc[mask] / count[mask]).astype(np.float32)
    elif method == "sum":
        combined_data = acc.astype(np.float32)
    elif method == "max":
        combined_data = acc.astype(np.float32)
        combined_data[~np.isfinite(combined_data)] = 0.0
    elif method == "median":
        stack = np.stack(resampled_arrays, axis=0).astype(np.float32)
        combined_data = np.median(stack, axis=0).astype(np.float32)

    combined = EMMap(
        origin=tuple(new_origin.tolist()),
        apix=tuple(apix_xyz.tolist()),
        density_map=combined_data,
        resolution=float(maps[0].resolution),
    )

    if return_resampled:
        aligned = []
        for m in maps:
            tmp_arr = np.zeros(new_shape, dtype=np.float32)
            _paste_or_resample_into(
                src_data=np.asarray(m.density_map, dtype=np.float32),
                src_origin=m.origin,
                new_data=tmp_arr,
                new_origin=new_origin,
                apix_xyz=apix_xyz,
                tol=tol,
            )
            aligned.append(
                EMMap(
                    origin=tuple(new_origin.tolist()),
                    apix=tuple(apix_xyz.tolist()),
                    density_map=tmp_arr,
                    resolution=float(m.resolution),
                )
            )
        return combined, aligned

    return combined


def segment_map2_on_map1_grid(
    map1,
    map2,
    order=1,
    cval=0.0,
    tol=1e-4,
):
    from ChemEM.EMMap import EMMap

    ap1 = _as_xyz(map1.apix)
    ap2 = _as_xyz(map2.apix)
    if not np.allclose(ap1, ap2, atol=tol, rtol=0):
        raise ValueError("apix mismatch. This function assumes identical apix for map1 and map2.")

    origin1 = np.asarray(map1.origin, dtype=np.float64)
    origin2 = np.asarray(map2.origin, dtype=np.float64)

    nz1, ny1, nx1 = map1.density_map.shape
    nz2, ny2, nx2 = map2.density_map.shape

    ax, ay, az = ap1

    off_xyz = (origin1 - origin2) / ap1
    off_zyx = np.array([off_xyz[2], off_xyz[1], off_xyz[0]], dtype=np.float64)

    frac = off_zyx - np.rint(off_zyx)
    if np.all(np.abs(frac) <= tol):
        off_int = np.rint(off_zyx).astype(int)

        z0, y0, x0 = off_int.tolist()

        out = np.full((nz1, ny1, nx1), cval, dtype=np.float32)

        z_src0 = max(z0, 0)
        y_src0 = max(y0, 0)
        x_src0 = max(x0, 0)
        z_src1 = min(z0 + nz1, nz2)
        y_src1 = min(y0 + ny1, ny2)
        x_src1 = min(x0 + nx1, nx2)

        if (z_src1 > z_src0) and (y_src1 > y_src0) and (x_src1 > x_src0):
            z_dst0 = z_src0 - z0
            y_dst0 = y_src0 - y0
            x_dst0 = x_src0 - x0
            z_dst1 = z_dst0 + (z_src1 - z_src0)
            y_dst1 = y_dst0 + (y_src1 - y_src0)
            x_dst1 = x_dst0 + (x_src1 - x_src0)

            out[z_dst0:z_dst1, y_dst0:y_dst1, x_dst0:x_dst1] = map2.density_map[
                z_src0:z_src1, y_src0:y_src1, x_src0:x_src1
            ]

        new_map = map1.copy()
        new_map.density_map = out
        new_map.origin = tuple(origin1.tolist())
        new_map.apix = tuple(ap1.tolist())
        new_map.resolution = float(getattr(map2, "resolution", getattr(map1, "resolution", 0.0)))
        return new_map

    z1_idx = np.arange(nz1, dtype=np.float64)
    y1_idx = np.arange(ny1, dtype=np.float64)
    x1_idx = np.arange(nx1, dtype=np.float64)

    Z1, Y1, X1 = np.meshgrid(z1_idx, y1_idx, x1_idx, indexing="ij")

    Xw = origin1[0] + X1 * ax
    Yw = origin1[1] + Y1 * ay
    Zw = origin1[2] + Z1 * az

    X2f = (Xw - origin2[0]) / ax
    Y2f = (Yw - origin2[1]) / ay
    Z2f = (Zw - origin2[2]) / az

    coords = np.array([Z2f, Y2f, X2f], dtype=np.float64)

    sampled = map_coordinates(
        map2.density_map.astype(np.float32),
        coords,
        order=order,
        mode="constant",
        cval=float(cval),
    ).astype(np.float32)

    new_map = map1.copy()
    new_map.density_map = sampled
    new_map.origin = tuple(origin1.tolist())
    new_map.apix = tuple(ap1.tolist())
    new_map.resolution = float(getattr(map2, "resolution", getattr(map1, "resolution", 0.0)))
    return new_map


def pad_emmap(
    emap,
    pad_voxels=0,
    pad_angstrom=None,
    mode="constant",
    constant_values=0.0,
):
    ap = np.array(emap.apix, dtype=np.float64)
    if ap.size == 1:
        ap = np.array([float(ap), float(ap), float(ap)], dtype=np.float64)

    if pad_angstrom is not None:
        if isinstance(pad_angstrom, (int, float)):
            padA = np.array([float(pad_angstrom)] * 3, dtype=np.float64)
        else:
            padA = np.array(pad_angstrom, dtype=np.float64)
            if padA.size != 3:
                raise ValueError("pad_angstrom must be float or length-3 tuple (x,y,z).")
        pad_xyz = np.ceil(padA / ap).astype(int)
        pad_voxels = tuple(int(x) for x in pad_xyz)

    if isinstance(pad_voxels, (int, np.integer)):
        p = int(pad_voxels)
        pad_width = ((p, p), (p, p), (p, p))
        pad_x = pad_y = pad_z = p
    else:
        if len(pad_voxels) == 3 and all(isinstance(v, (int, np.integer)) for v in pad_voxels):
            pad_x, pad_y, pad_z = (int(pad_voxels[0]), int(pad_voxels[1]), int(pad_voxels[2]))
            pad_width = ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x))
        elif len(pad_voxels) == 3 and all(len(v) == 2 for v in pad_voxels):
            pad_width = tuple((int(v[0]), int(v[1])) for v in pad_voxels)
            pad_z = pad_width[0][0]
            pad_y = pad_width[1][0]
            pad_x = pad_width[2][0]
        else:
            raise ValueError(
                "pad_voxels must be int, (pad_x,pad_y,pad_z), or "
                "((pz0,pz1),(py0,py1),(px0,px1))."
            )

    data = np.pad(
        emap.density_map,
        pad_width=pad_width,
        mode=mode,
        constant_values=constant_values if mode == "constant" else 0.0,
    ).astype(np.float32, copy=False)

    ox, oy, oz = map(float, emap.origin)
    new_origin = (
        ox - pad_x * ap[0],
        oy - pad_y * ap[1],
        oz - pad_z * ap[2],
    )

    out = emap.copy()
    out.origin = new_origin
    out.apix = tuple(ap.tolist())
    out.density_map = data
    return out

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
    
    # Use a KDTree for efficient nearest-neighbor search. This finds, for each
    # protein atom, the distance to the nearest voxel in the binding site.
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
    #print("Distance map computed.")
    return distance_map

def find_site_at_point(
    pocket_grid_zyx: np.ndarray,
    point_xyz: tuple[float, float, float],
    grid_origin_xyz: np.ndarray,
    grid_resolution: float
) -> np.ndarray | None:
    """
    Finds the specific, contiguous binding site containing a given point.

    This function maps a real-world (x, y, z) coordinate to the grid,
    identifies which disconnected component the point falls into, and returns
    the mask for that entire component.

    Args:
        pocket_grid_zyx (np.ndarray): A 3D boolean numpy array (in Z, Y, X order)
                                      where `True` indicates a voxel is part of a site.
        point_xyz (tuple): The (x, y, z) coordinate to check, in Angstroms.
        grid_origin_xyz (np.ndarray): The (x, y, z) coordinate of the grid's
                                      [0, 0, 0] corner, in Angstroms.
        grid_resolution (float): The side length of a single voxel, in Angstroms.

    Returns:
        np.ndarray | None: A 3D boolean mask of the same shape as the input,
                           containing only the binding site at the specified point.
                           Returns `None` if the point is outside the grid or
                           not within any site.
    """
   
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
                        # We just hit the second wall. Check if criteria are met.
                        if wall_thickness >= min_wall_voxels:
                            pocket_len = x - pocket_candidate_start
                            if pocket_len >= min_pocket_voxels:
                                # SUCCESS! We found a pocket segment.
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




'''
def calculate_ses_grid_zyx(
    atom_coords: np.ndarray,
    atom_radii: np.ndarray,
    probe_radius: float = 1.4,
    grid_resolution: float = 0.5
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Calculates a masked 3D grid for the true Solvent-Excluded Surface (SES)
    using a two-step carving method, returning a mask with (Z, Y, X) axis ordering.

    Args:
        atom_coords (np.ndarray): Array of atom coordinates (x, y, z), shape (N, 3).
        atom_radii (np.ndarray): Array of atom VDW radii, shape (N,).
        probe_radius (float): The radius of the solvent probe sphere.
        grid_resolution (float): The spacing between grid points in Å.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
        - ses_mask_zyx (np.ndarray): A 3D boolean array in (Z, Y, X) order.
        - grid_coords_xyz (np.ndarray): The coordinates of each point in a grid with
                                       (X, Y, Z) axis order.
        - grid_origin (np.ndarray): The [x, y, z] coordinate of the grid's origin.
    """
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
'''

def calculate_ses_grid_zyx(
    atom_coords: np.ndarray,
    atom_radii: np.ndarray,
    probe_radius: float = 1.4,
    grid_resolution: float = 0.5,
    map_shape_zyx: tuple | None = None,   # (Z, Y, X) voxels
    map_origin_xyz: tuple | None = None,  # (X, Y, Z) Å
    apix: float | None = None ,
    point_block: int = 200_000,           # NEW: control memory for grid chunks
    atom_block: int = 4_096         # Å/voxel
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Calculates a masked 3D grid for the true Solvent-Excluded Surface (SES)
    returning a mask with (Z, Y, X) axis ordering.

    If `map_shape_zyx`, `map_origin_xyz`, and optional `apix` are given, the SES
    is computed on that fixed grid. Otherwise, an auto-bounded grid is built
    using `grid_resolution`.

    Args:
        atom_coords (np.ndarray): (N, 3) array of atom coordinates (X, Y, Z) in Å.
        atom_radii (np.ndarray): (N,) array of VDW radii in Å.
        probe_radius (float): Solvent probe radius in Å.
        grid_resolution (float): Spacing between grid points in Å (used if `apix` not given).
        map_shape_zyx (tuple[int,int,int] | None): Target map shape in (Z, Y, X).
        map_origin_xyz (tuple[float,float,float] | None): Grid origin in Å (X, Y, Z).
        apix (float | None): Target pixel size in Å/voxel for fixed grid.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
        - ses_mask_zyx: 3D boolean array (Z, Y, X).
        - grid_coords_xyz: float array of shape (X, Y, Z, 3) with coordinates.
        - grid_origin: np.ndarray [X, Y, Z] origin in Å.
    """
    if atom_coords.shape[0] != atom_radii.shape[0]:
        raise ValueError("atom_coords and atom_radii must have the same number of atoms.")

    # --- Build grid either from user-specified map or auto-bounded ---
    if map_shape_zyx is not None and map_origin_xyz is not None:
        
        # Use provided map geometry
        apix = float(apix) if apix is not None else float(grid_resolution)
        nz, ny, nx = map_shape_zyx
        grid_origin = np.asarray(map_origin_xyz, dtype=float)
        
        x_range = grid_origin[0] + np.arange(nx) * apix
        y_range = grid_origin[1] + np.arange(ny) * apix
        z_range = grid_origin[2] + np.arange(nz) * apix
        
        
        buffer = probe_radius + apix
        min_needed = np.min(atom_coords - atom_radii[:, None], axis=0) - buffer
        max_needed = np.max(atom_coords + atom_radii[:, None], axis=0) + buffer
        grid_min = grid_origin
        grid_max = grid_origin + np.array([nx - 1, ny - 1, nz - 1], dtype=float) * apix
        if np.any(min_needed < grid_min) or np.any(max_needed > grid_max):
            print("Warning: Some atoms +/- (radius+probe) fall outside the provided grid.")
        
    else:
        # Auto-bounded grid 
        apix = float(grid_resolution)
        buffer = probe_radius + apix
        min_coords = np.min(atom_coords - atom_radii[:, np.newaxis], axis=0) - buffer
        max_coords = np.max(atom_coords + atom_radii[:, np.newaxis], axis=0) + buffer
        grid_origin = min_coords

        x_range = np.arange(min_coords[0], max_coords[0], apix)
        y_range = np.arange(min_coords[1], max_coords[1], apix)
        z_range = np.arange(min_coords[2], max_coords[2], apix)

    
    
    gx, gy, gz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    grid_points = np.vstack([gx.ravel(), gy.ravel(), gz.ravel()]).T
    min_dist_to_surface = _min_dist_to_vdw_surfaces_blocked(
        grid_points, atom_coords, atom_radii,
        point_block=point_block, atom_block=atom_block
    )
    
    
    # SAS mask = points within probe_radius of any VDW surface
    sas_mask_flat = min_dist_to_surface < probe_radius
    sas_mask_xyz = sas_mask_flat.reshape(len(x_range), len(y_range), len(z_range))
    solvent_centers_flat = min_dist_to_surface >= probe_radius
    solvent_centers_xyz = solvent_centers_flat.reshape(len(x_range), len(y_range), len(z_range))
    
    krnl_radius_grid = int(np.ceil(probe_radius / apix))
    k_range = np.arange(-krnl_radius_grid, krnl_radius_grid + 1)
    kx, ky, kz = np.meshgrid(k_range, k_range, k_range, indexing='ij')
    dilation_kernel = (kx**2 + ky**2 + kz**2) * (apix**2) <= (probe_radius**2)
    solvent_volume_xyz = binary_dilation(solvent_centers_xyz, structure=dilation_kernel)

    # Carve solvent volume from SAS to get SES
    ses_mask_xyz = sas_mask_xyz & ~solvent_volume_xyz
    ses_mask_zyx = ses_mask_xyz.transpose(2, 1, 0)
    grid_coords_xyz = grid_points.reshape(len(x_range), len(y_range), len(z_range), 3)

    return ses_mask_zyx, grid_coords_xyz, grid_origin

def _min_dist_to_vdw_surfaces_blocked(
    grid_points: np.ndarray,
    atom_coords: np.ndarray,
    atom_radii: np.ndarray,
    point_block: int = 200_000,
    atom_block: int = 4_096,
) -> np.ndarray:
    """
    Compute min distance from each grid point to any VDW surface without
    allocating the full (M x N) distance matrix. Returns shape (M,).
    """
    M = grid_points.shape[0]
    min_dist = np.full(M, np.inf, dtype=np.float64)

    for i0 in range(0, M, point_block):
        i1 = min(i0 + point_block, M)
        gp = grid_points[i0:i1]                   # (P, 3)
        md = np.full(gp.shape[0], np.inf)

        for j0 in range(0, atom_coords.shape[0], atom_block):
            j1 = min(j0 + atom_block, atom_coords.shape[0])
            ac = atom_coords[j0:j1]               # (A, 3)
            ar = atom_radii[j0:j1]                # (A,)

            # distances to atom centers for this block
            d = cdist(gp, ac)                     # (P, A)
            d -= ar[None, :]                      # subtract VDW radii -> distance to surfaces
            md = np.minimum(md, np.min(d, axis=1))

        min_dist[i0:i1] = md

    return min_dist

def generate_density_map(points, radii, grid_spacing=1.0, filename="density_map.mrc"):
    """
    Generate a 3D density map from points and radii and save it as an MRC file.

    Parameters:
        points (numpy.ndarray): An (N, 3) array of points [x, y, z].
        radii (numpy.ndarray): An array of radii for each point.
        grid_spacing (float): The spacing of the grid in Å. Default is 1.0 Å.
        filename (str): The name of the output MRC file.
    """
    
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




def create_binding_site_mask(density_shape_zyx, points_xyz, radii_A, apix_xyz, origin_xyz):
    """
    density_shape_zyx : (nz, ny, nx)
    points_xyz        : iterable of (x,y,z) centers in Å
    radii_A           : iterable of radii in Å
    apix_xyz          : (ax, ay, az) Å/voxel
    origin_xyz        : (ox, oy, oz) Å (real-space coord of voxel [0,0,0] in x,y,z)
    """
    nz, ny, nx = map(int, density_shape_zyx)
    ax, ay, az = map(float, apix_xyz)
    ox, oy, oz = map(float, origin_xyz)

    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    radii_A = np.asarray(radii_A, dtype=np.float64)

    # output
    mask = np.zeros((nz, ny, nx), dtype=bool)

    # centers in voxel coordinates (x,y,z) as float (keep sub-voxel precision)
    centers_vox = (points_xyz - np.array([ox, oy, oz])) / np.array([ax, ay, az])

    for (cx, cy, cz), r in zip(centers_vox, radii_A):
        if r <= 0:
            continue
        r2 = r * r

        # bounding box in voxel indices
        rx = r / ax
        ry = r / ay
        rz = r / az

        x0 = max(int(np.floor(cx - rx)), 0)
        x1 = min(int(np.ceil (cx + rx)), nx - 1)
        y0 = max(int(np.floor(cy - ry)), 0)
        y1 = min(int(np.ceil (cy + ry)), ny - 1)
        z0 = max(int(np.floor(cz - rz)), 0)
        z1 = min(int(np.ceil (cz + rz)), nz - 1)

        if x0 > x1 or y0 > y1 or z0 > z1:
            continue

        # local grid (z,y,x) indices
        zz, yy, xx = np.ogrid[z0:z1+1, y0:y1+1, x0:x1+1]

        # distance^2 in Å^2 (anisotropic apix supported)
        dx2 = ((xx - cx) * ax) ** 2
        dy2 = ((yy - cy) * ay) ** 2
        dz2 = ((zz - cz) * az) ** 2
        sphere = (dx2 + dy2 + dz2) <= r2

        mask[z0:z1+1, y0:y1+1, x0:x1+1] |= sphere

    return mask




def otsu_mask_from_map(volume: np.ndarray, ignore_nan: bool = False):
    """
    Apply Otsu thresholding to a 3D map.
    Returns (binary_mask, threshold).
    """
    if ignore_nan:
        data = volume[np.isfinite(volume)]
    else:
        data = volume.ravel()
    
    
    data = data[data != 0] 
    t = threshold_otsu(data)          
    mask = volume >= t                # boolean mask, same shape as volume
    return mask, t


def sobel_filtered_map(volume, normalise=True):
    gx = sobel(volume, axis=0)
    gy = sobel(volume, axis=1)
    gz = sobel(volume, axis=2)
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    
    if normalise:
        # Normalize the gradient magnitude to be between 0 and 1.
        grad_min = grad_mag.min()
        grad_max = grad_mag.max()
        if grad_max > grad_min:
            grad_mag = (grad_mag - grad_min) / (grad_max - grad_min)
        else:
            # In case the gradient magnitude is constant, return an array of zeros.
            grad_mag = np.zeros_like(grad_mag)
    
    return grad_mag




def grow_ligand_region(seed_mask, binding_mask, gradient_map, grad_threshold, *, connectivity=1, keep_seed=True):
    """
    Faster version of your region growing:
    grow seed within (binding_mask AND gradient >= threshold).
    """
    seed_mask = seed_mask.astype(bool, copy=False)
    binding_mask = binding_mask.astype(bool, copy=False)

    # Only allow growth where both conditions hold
    allowed = binding_mask & (gradient_map >= grad_threshold)

    # Your original loop keeps seed voxels even if they fail the threshold.
    # binary_propagation will never output True outside its mask, so optionally
    # allow the seed itself.
    if keep_seed:
        allowed = allowed | seed_mask

    struct = generate_binary_structure(3, connectivity)

    
    return binary_propagation(seed_mask, struct, mask=allowed)


def grow_ligand_region_(seed_mask, binding_mask, gradient_map, grad_threshold):
    """
    Grow a region starting from seed_mask, but only add voxels whose gradient value
    (from gradient_map) is above grad_threshold. This is intended to expand the ligand
    density into the surrounding binding site, but stop growing when the gradient 
    drops below grad_threshold (indicating a less-defined or bridging boundary).
    
    Parameters:
    -----------
    seed_mask : 3D numpy array (boolean)
        Binary mask of the initial ligand seed region.
    binding_mask : 3D numpy array (boolean)
        Binary mask of the overall binding site.
    gradient_map : 3D numpy array (float)
        Normalized gradient map (values between 0 and 1) from the Sobel operator.
    grad_threshold : float
        The threshold for the gradient value. Only neighboring voxels with 
        gradient_map >= grad_threshold will be added to the region.
        
    Returns:
    --------
    grown_region : 3D numpy array (boolean)
        Binary mask of the grown region.
    """
    
    grown_region = seed_mask.copy()
    structure = generate_binary_structure(3, 1)  # 3D connectivity
    changed = True
    
    while changed:
        changed = False
        
        dilated = binary_dilation(grown_region, structure=structure)
        candidates = dilated & binding_mask & (~grown_region)
        valid = candidates & (gradient_map >= grad_threshold)
        if np.any(valid):
            grown_region |= valid
            changed = True
    
    return grown_region




def extract_ligand_density(
    density_map,
    binding_mask,
    apix,
    high_threshold_sigma=2.0,
    grad_threshold=0.4,
):
    # --- mask map to binding site ---
    binding_density = density_map * (binding_mask > 0.0)

    non_zero_values = binding_density[binding_density > 0.0]
    if non_zero_values.size == 0:
        return [], []

    high_threshold = float(np.std(non_zero_values)) * float(high_threshold_sigma)

    # seeds: voxels confidently in ligand density
    thr_mask = (binding_density > high_threshold)

    # gradient gate (same as before)
    density_smoothed = gaussian_filter(binding_density, sigma=1.0)
    mag_grad = sobel_filtered_map(density_smoothed)

    struct = generate_binary_structure(3, 1)

    # label seeds (these define the feature IDs / per-feature stats)
    seed_labels, num_features = label(thr_mask, structure=struct)
    if num_features == 0:
        return [], []

    seed_slices = find_objects(seed_labels)

    # --- ONE global grow + relabel ---
    # In your original grow_ligand_region you used:
    #   binding_mask_for_grow = (binding_density > 0.0)
    #   valid = candidates & (mag_grad >= grad_threshold)
    #allowed_growth = (binding_density > 0.0) & (mag_grad >= grad_threshold)
    allowed_growth = (binding_density > 0.0) & (mag_grad <= grad_threshold)

    # keep seeds even if they fail allowed_growth (matches your original: you never remove seeds)
    grown_all = binary_propagation(thr_mask, struct, mask=(allowed_growth | thr_mask))

    grown_labels, num_grown = label(grown_all, structure=struct)
    grown_slices = find_objects(grown_labels)

    # precompute constants used in feature dicts
    mean_masked = float(binding_density.mean())
    pos = density_map[density_map > 0.0]
    amp_no_mask = float(pos.mean()) if pos.size else 0.0
    voxel_vol = float(np.prod(apix))

    ligand_features = []
    ligand_densities = []

    # for each original seed-feature, pick the grown component(s) it connects to
    for i in range(1, num_features + 1):
        sl_seed = seed_slices[i - 1]
        if sl_seed is None:
            continue

        seed_local = (seed_labels[sl_seed] == i)

        # stats are based on the *seed* region (same as your old code)
        vals = binding_density[sl_seed][seed_local]
        if vals.size == 0:
            continue

        dist_vals = binding_mask[sl_seed][seed_local]
        centroid = float(dist_vals.mean())

        map_features = {
            "centroid": centroid,
            "volume": float(seed_local.sum()) * voxel_vol,
            "mean_non_zero_density_value": float(vals.mean()),
            "map_amplitude_thr_no_mask": amp_no_mask,
            "map_amplitude_thr_masked": mean_masked,
            "feature_id": i,
        }

        # Find which grown component(s) this seed reaches.
        # Use a 1-voxel dilation so seeds that are adjacent to allowed_growth
        # latch onto the right grown component (matches your first iteration).
        seed_dil = binary_dilation(seed_local, structure=struct, iterations=1)
        comp_ids = np.unique(grown_labels[sl_seed][seed_dil])
        comp_ids = comp_ids[comp_ids != 0]

        grown_mask = np.zeros_like(thr_mask, dtype=bool)

        # Always keep the seed itself (matches your original)
        grown_mask[sl_seed] |= seed_local

        # Add the full grown component(s), using find_objects slices to avoid full-volume compares
        for cid in comp_ids:
            slc = grown_slices[cid - 1]
            if slc is None:
                continue
            grown_mask[slc] |= (grown_labels[slc] == cid)

        # Build the density map for this feature (same output type/shape as before)
        full_ligand_density = np.zeros_like(binding_density)
        full_ligand_density[grown_mask] = binding_density[grown_mask]

        ligand_features.append(map_features)
        ligand_densities.append(full_ligand_density)

    return ligand_densities, ligand_features




def extract_ligand_densit_y(density_map, 
                           binding_mask,
                           apix,
                           high_threshold_sigma=2.0,
                           grad_threshold=0.4
                           ):
    
    
    
    binding_density = density_map *  (binding_mask > 0.0)
    non_zero_values = binding_density[binding_density > 0.0]
    high_threshold = np.std(non_zero_values) * high_threshold_sigma 
    thr_map = binding_density * (binding_density > high_threshold)
    density_smoothed = gaussian_filter(binding_density, sigma=1.0)
    mag_grad = sobel_filtered_map(density_smoothed)
    
    labels, num_features = label(thr_map, structure = generate_binary_structure(3, 1))
    ligand_features = []
    ligand_densities = []
    
    mean_masked = np.mean(binding_density.ravel())
    if num_features > 0:
       
        for i in range(num_features):
            if i == 0:
                continue 
            
            map_features = {}
            dist_map = binding_mask[labels ==i]
            centroid = np.mean(dist_map)
            
            map_features['centroid'] = centroid 
            
            feature_map = binding_density * (labels == i)
            values_above = feature_map[labels == i]
            
            
            volume = len(values_above) * np.product(apix)
            map_features['volume'] = volume
            map_features['mean_non_zero_density_value'] = values_above.mean() 
            map_features['map_amplitude_thr_no_mask'] = density_map[density_map > 0].mean()
            map_features['map_amplitude_thr_masked'] = mean_masked
            map_features['feature_id'] = i 
            
            
            full_ligand_density_mask =  grow_ligand_region(feature_map > 0.0,
                                                     binding_density > 0.0,
                                                     mag_grad,
                                                     grad_threshold)
            full_ligand_density = binding_density * full_ligand_density_mask 
            ligand_features.append(map_features)
            ligand_densities.append(full_ligand_density)
                    
    return ligand_densities, ligand_features




def extract_ligand_density_otsu(full_map, resolution, sigma_coeff=0.356):
    full_map_threshold = threshold_otsu(full_map.ravel())

    map_copy = full_map.copy()
    structuring_element = generate_binary_structure(3, 1)
    image_3d = map_copy * (map_copy  > 0)
    eroded_image = grey_opening(image_3d, footprint=structuring_element)
    flattened = eroded_image.ravel() 
    
    global_thresh = threshold_otsu(flattened)
    segmented_density = get_disconected_densities(eroded_image,
                                                  global_thresh,
                                                  struct=structuring_element)
    
    sigma = get_sigma(resolution, sigma_coeff=0.356)
    #exclude 0 label 0 is background!!
    
    feature_maps = []
    for num in np.unique(segmented_density)[1:]:
        mask_1 = (segmented_density == num)
        map_1 = eroded_image * mask_1
        map_1_closed = grey_closing(map_1, footprint=structuring_element)
        smooth_mask = smooth_image(map_1_closed , sigma)
        flattened = smooth_mask.ravel()
        smooth_thresh = threshold_otsu(flattened)
        threshold_image = smooth_mask * (smooth_mask > smooth_thresh)
        masked_region = map_copy * (threshold_image > 0)
        n_voxels_above_zero, n_voxels_above_otsu, percent_above_otsu = get_percent_over_threshold( masked_region, full_map_threshold)
        feature_maps.append([ ( n_voxels_above_zero, n_voxels_above_otsu, percent_above_otsu),
                           masked_region])
    return feature_maps

def  get_percent_over_threshold(masked_map, threshold):
    
    count_above_zero = np.sum(masked_map > 0)
    
    count_above_otsu = np.sum(masked_map > threshold)
    percent = round(count_above_otsu / count_above_zero , 2)
    
    return count_above_zero, count_above_otsu, percent

def smooth_image(image, sigma):
    smoothed_image = gaussian_filter(image, sigma=sigma)
    return smoothed_image

def get_sigma(resolution, sigma_coeff=0.356):
    return resolution * sigma_coeff

def get_disconected_densities(image, threshold, struct = None):
    if struct is None:
        struct = generate_binary_structure(3, 1)
        
    bool_image = (image > threshold)
    labels, num_features = label(bool_image, structure = struct)
    return labels

def extract_min_bounding_box(volume: np.ndarray, thr: float = 0.0, pad: int = 0):
    """
    Return (subvolume, (z0,y0,x0), (z1,y1,x1)) for the minimal bbox containing values > thr.
    If no voxels > thr, returns None.
    pad: optional integer padding (in voxels) around the bbox, clipped to bounds.
    """
    pad = int(pad)
    mask = volume > thr
    if not np.any(mask):
        return None
    
    

    z_idx, y_idx, x_idx = np.where(mask)
    z0, z1 = z_idx.min(), z_idx.max()
    y0, y1 = y_idx.min(), y_idx.max()
    x0, x1 = x_idx.min(), x_idx.max()

    if pad > 0:
        z0 = max(0, z0 - pad); y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
        z1 = min(volume.shape[0]-1, z1 + pad)
        y1 = min(volume.shape[1]-1, y1 + pad)
        x1 = min(volume.shape[2]-1, x1 + pad)
    
    sub = volume[z0:z1+1, y0:y1+1, x0:x1+1]
    return sub, (z0, y0, x0), (z1, y1, x1)


def get_map_features(density_map, 
                     feature_map,
                     distance_mask,
                     apix,
                     feature_id = -1
                     ):
    map_features = {}
    
    binding_density = density_map *  (distance_mask > 0.0)
    
    #
    dist_map = distance_mask[feature_map > 0.0]
    centroid = np.mean(dist_map)
    
    values_above = feature_map[feature_map > 0.0]
    volume = len(values_above) * np.product(apix)
    map_features['volume'] = volume
    map_features['centroid'] = centroid 
    map_features['mean_non_zero_density_value'] = values_above.mean() 
    map_features['map_amplitude_thr_no_mask'] = density_map[density_map > 0].mean()
    map_features['map_amplitude_thr_masked'] = np.mean(binding_density.ravel())
    
    return map_features



def extract_subvolume_from_grid(map_origin, apix, density, box_size, grid_origin=None, resolution=None):
    """
    Extracts a sub-volume (grid) from a density map based on a specified box size.
    
    Args:
        map_origin (np.ndarray): Real-space origin of the full density map (order: x, y, z).
        apix (float or array-like): Pixel size in Å per voxel (order: x, y, z if array-like).
        density (np.ndarray): 3D density map (shape: (z, y, x)).
        box_size (list or np.ndarray): Size of the subvolume in Å (order: x, y, z).
        grid_origin (np.ndarray, optional): Real-space coordinate (order: x, y, z) for the subvolume's origin.
            If None, defaults to the full map's origin.
        resolution (float, optional): Resolution of the subvolume (to be passed to the EMMap).
    
    Returns:
        EMMap: An EMMap object containing:
            - new_origin: The sub-volume's origin in real-space.
            - apix: The voxel size (unchanged).
            - new_density: The extracted density sub-volume.
            - resolution: The provided resolution.
    
    Raises:
        ValueError: If the computed subvolume is empty or the specified grid extends beyond map bounds.
    """
    import numpy as np

    if grid_origin is None:
        grid_origin = map_origin

    map_origin = np.array(map_origin)
    grid_origin = np.array(grid_origin)
    apix = np.array(apix)
    box_size = np.array(box_size)
    grid_origin_voxel = np.floor((grid_origin - map_origin) / apix).astype(int)[::-1]

    box_voxels = box_size.astype(int)
    if np.any(box_voxels <= 0):
        raise ValueError("Box size is too small to extract any voxels in one or more dimensions.")

    min_voxel = grid_origin_voxel
    max_voxel = grid_origin_voxel + box_voxels - 1
    min_voxel = np.maximum(min_voxel, 0)
    max_voxel = np.minimum(max_voxel, np.array(density.shape) - 1)
    if np.any(max_voxel < min_voxel):
        raise ValueError("The specified grid extends outside the density map bounds.")

    slices = tuple(slice(min_idx, max_idx + 1) for min_idx, max_idx in zip(min_voxel, max_voxel))

    new_density = density[slices].copy()
    new_origin = map_origin + (min_voxel[::-1] * apix)

    #return new_origin, apix, new_density, resolution
    return EMMap(new_origin, apix, new_density, resolution)


def extract_map_segment_around_atoms(densmap,
                                     atom_coords,
                                     padding = 0.0):
    
    density_map = densmap.density_map
    apix = np.array(densmap.apix)
    origin = np.array(densmap.origin)
    
    nz, ny, nx = density_map.shape
    ax, ay, az = apix 
    
    x_idx = (atom_coords[:, 0] - origin[0]) / ax
    y_idx = (atom_coords[:, 1] - origin[1]) / ay
    z_idx = (atom_coords[:, 2] - origin[2]) / az
    
    pad_x = padding / ax
    pad_y = padding / ay
    pad_z = padding / az
    x_min = np.floor(np.min(x_idx) - pad_x)
    x_max = np.ceil(np.max(x_idx) + pad_x)
    y_min = np.floor(np.min(y_idx) - pad_y)
    y_max = np.ceil(np.max(y_idx) + pad_y)
    z_min = np.floor(np.min(z_idx) - pad_z)
    z_max = np.ceil(np.max(z_idx) + pad_z)
    x0 = int(max(0, x_min))
    x1 = int(min(nx, x_max + 1))  # +1 for exclusive upper bound
    y0 = int(max(0, y_min))
    y1 = int(min(ny, y_max + 1))
    z0 = int(max(0, z_min))
    z1 = int(min(nz, z_max + 1))
    
    submap = density_map[z0:z1, y0:y1, x0:x1]
    
    new_origin = origin + np.array([
        x0 * ax,  # X
        y0 * ay,  # Y
        z0 * az,  # Z
    ], dtype=float)
    
    site_distance_map = EMMap(new_origin, apix, submap, densmap.resolution)
    return site_distance_map



    
    

def site_from_densmap(densmap, 
                      protein_structure,
                      distance_map,
                      distance_cutoff = 6.0,
                      density_threshold = 0.0,
                      padding=0.0
                      
                      ):
    density_map = densmap.density_map
    apix = np.array(densmap.apix)
    origin = np.array(densmap.origin)
    
    atoms = [i for i in protein_structure.atoms if i.element > 1]
    positions = np.array([[i.xx, i.xy, i.xz] for i in atoms ])
    residue_ids = [i.residue for i in atoms]
    mask = density_map > density_threshold
    if not np.any(mask):
       return set()
    
    
    
    mask_indices = np.argwhere(mask)
    z_idx = mask_indices[:, 0]
    y_idx = mask_indices[:, 1]
    x_idx = mask_indices[:, 2]
    
    ax, ay, az = apix  # spacing for X, Y, Z
    map_points = np.column_stack([
        origin[0] + x_idx * ax,  # X
        origin[1] + y_idx * ay,  # Y
        origin[2] + z_idx * az,  # Z
    ])
    
    
    tree = cKDTree(map_points)
    dists, _ = tree.query(positions, k=1)
    #grab all the residues wthin a given distance of the segmented map.
    binding_residues = {res_id for res_id, d in zip(residue_ids, dists) if d <= distance_cutoff}
    
    site_coords = []
    unique_atom_indices = []
    for res in binding_residues:
        for atom in res.atoms:
            if atom.element > 1:
                site_coords.append([atom.xx, atom.xy, atom.xz])
                unique_atom_indices.append(atom.idx)
    
    site_coords = np.array(site_coords)
    
    
    site_distance_map = extract_map_segment_around_atoms(distance_map, 
                                     site_coords,
                                     padding = padding)
    
    resampled_density = resample_map_onto_target(densmap, site_distance_map)
    
    return binding_residues, np.array(unique_atom_indices), site_distance_map, resampled_density
    

    
def resample_map_onto_target(source, target, order=1, cval=0.0):
    '''
    Resample `source_map` onto the grid of `target_map`.
    '''
    
    source_map = np.asarray(source.density_map, dtype=float)
    target_shape = target.density_map.shape

    source_origin = np.asarray(source.origin, dtype=float)
    source_apix   = np.asarray(source.apix, dtype=float)

    target_origin = np.asarray(target.origin, dtype=float)
    target_apix   = np.asarray(target.apix, dtype=float)
    
    
    ax_s, ay_s, az_s = source_apix
    ax_t, ay_t, az_t = target_apix

    nz_t, ny_t, nx_t = target_shape

    # 1. Build index grid for target map: (Z, Y, X)
    z_t, y_t, x_t = np.indices(target_shape, dtype=float)

    
    X = target_origin[0] + x_t * ax_t
    Y = target_origin[1] + y_t * ay_t
    Z = target_origin[2] + z_t * az_t
    
    # 3. Convert real-space coordinates -> source voxel indices
    #    (remember: indices are (z, y, x); apix/origin are (X, Y, Z))
    x_s = (X - source_origin[0]) / ax_s
    y_s = (Y - source_origin[1]) / ay_s
    z_s = (Z - source_origin[2]) / az_s
    
    # 4. Interpolate source_map at these (z_s, y_s, x_s) positions
    coords = np.vstack([
        z_s.ravel(),
        y_s.ravel(),
        x_s.ravel(),
    ])
    
    resampled_flat = map_coordinates(
        source_map,
        coords,
        order=order,
        mode="constant",
        cval=cval,
    )
    
    resampled = resampled_flat.reshape(target_shape)
    
    resampled_map = target.copy()
    resampled_map.density_map = resampled
    return resampled_map




def compute_p_values(density_map, bg_mean, bg_std):
    """
    Compute one-sided p-values for each voxel assuming Gaussian noise distribution.
    p-value = P(X >= value) under N(bg_mean, bg_std^2).
    """
    # Z-score each voxel
    z_map = (density_map - bg_mean) / (bg_std + 1e-12)
    
    # p-values = 1 - CDF(z)
    p_values = 1 - norm.cdf(z_map)
    return p_values


def benjamini_yekutieli(pvals, alpha=0.05):
    """
    Benjamini-Yekutieli procedure for FDR control.
    This method handles arbitrary dependencies.
    
    Returns array of q-values.
    """
    # Flatten p-values and sort
    pvals_flat = pvals.flatten()
    n = len(pvals_flat)
    sort_indices = np.argsort(pvals_flat)
    sorted_pvals = pvals_flat[sort_indices]
    
    # BY correction factor for arbitrary dependency
    # harmonic_n = sum_{i=1}^n (1/i)
    harmonic_n = np.sum(1.0 / np.arange(1, n+1))
    
    # Compute thresholded q-values
    qvals = np.zeros_like(sorted_pvals)
    min_ratio = 1.0
    for i in reversed(range(n)):
        current = sorted_pvals[i] * n * harmonic_n / (i+1)
        if current < min_ratio:
            min_ratio = current
        qvals[i] = min_ratio
    
    # Revert to original ordering
    qvals_original = np.empty_like(qvals)
    qvals_original[sort_indices] = qvals
    return qvals_original.reshape(pvals.shape)

def estimate_background_distribution(density_map, cube_size=10, num_cubes=4):
    """
    Estimate the background noise distribution parameters (mean, std).
    For simplicity, we pick `num_cubes` regions along the principal axes far from the particle center.
    Assumes the particle is roughly centered.
    """
    nx, ny, nz = density_map.shape
    #nz,ny,nx = density_map.shape
    half_x, half_y, half_z = nx//2, ny//2, nz//2
    
    # Define cube corners far outside the center. For instance:
    # 1) along +x direction, 2) -x direction, 3) +y direction, 4) -y direction.
    # Adjust these based on known map orientation and empty regions.
    coords = [
        (nx - cube_size, nx, half_y, half_y+cube_size, half_z, half_z+cube_size),
        (0, cube_size, half_y, half_y+cube_size, half_z, half_z+cube_size),
        (half_x, half_x+cube_size, ny - cube_size, ny, half_z, half_z+cube_size),
        (half_x, half_x+cube_size, 0, cube_size, half_z, half_z+cube_size),
    ]
    
    # Extract background voxels
    background_voxels = []
    for (xstart, xend, ystart, yend, zstart, zend) in coords:
        background_voxels.append(density_map[xstart:xend, ystart:yend, zstart:zend].ravel())
    background_voxels = np.concatenate(background_voxels)
    
    # Estimate mean and std from these voxels
    bg_mean = np.mean(background_voxels)
    bg_std = np.std(background_voxels)
    return bg_mean, bg_std

def coords_to_grid(coords, origin, voxel_size):
    """
    Convert atomic coordinates in Å to voxel indices in the map.
    
    Arguments:
    - coords: Nx3 array of atomic coordinates in Å.
    - origin: The origin of the map in Å (x0, y0, z0) corresponding to map[0,0,0].
    - voxel_size: The size of each voxel in Å/pixel (scalar or tuple).
    
    Returns:
    - grid_coords: Nx3 array of integer voxel indices.
    """
    # If voxel_size is scalar, make it a tuple (vx, vy, vz)
    if np.isscalar(voxel_size):
        voxel_size = (voxel_size, voxel_size, voxel_size)
        
    grid_coords = (coords - origin) / np.array(voxel_size)
    return np.round(grid_coords).astype(int)

def create_mask_from_model(density_shape, atom_indices, radius_pixels):
    """
    Create a binary mask for the protein by marking voxels within a certain 
    radius (in pixels) of any atom position.
    
    Arguments:
    - density_shape: shape of the 3D map (nz, ny, nx)
    - atom_indices: Nx3 array of voxel indices of atoms (v,y,z)
    - radius_pixels: the radius around each atom (in voxel units) to consider inside
    
    Returns:
    - mask: 3D boolean array, True = inside protein envelope
    """
    mask = np.zeros(density_shape, dtype=bool)
    
    # A simple approach: for each atom, mark a sphere of radius_pixels as True.
    # For large maps, this might be slow. Consider more efficient methods or spatial data structures.
    
    nz, ny, nx = density_shape
    rr = radius_pixels**2
    for x_a, y_a, z_a in atom_indices:
        # Determine bounding box for the sphere
        x_min = max(x_a - radius_pixels, 0)
        x_max = min(x_a + radius_pixels + 1, nx)
        y_min = max(y_a - radius_pixels, 0)
        y_max = min(y_a + radius_pixels + 1, ny)
        z_min = max(z_a - radius_pixels, 0)
        z_max = min(z_a + radius_pixels + 1, nz)
        
        # Compute distances within this sub-box
        xs, ys, zs = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
        dist_sq = (xs - x_a)**2 + (ys - y_a)**2 + (zs - z_a)**2
        mask[z_min:z_max, y_min:y_max, x_min:x_max] |= (dist_sq <= rr)
    
    return mask


def refine_mask(mask, dilation_iterations=2, smooth_sigma=1.0):
    """
    Refine the mask by dilating and smoothing, which can help ensure the entire molecular
    envelope is captured. 
    
    - binary_dilation expands the mask slightly
    - gaussian_filter on a float version and thresholding can create a smoother boundary
    
    Returns a refined boolean mask.
    """
    refined = binary_dilation(mask, iterations=dilation_iterations)
    # Smooth the refined mask by converting to float, applying Gaussian, and thresholding
    smoothed = gaussian_filter(refined.astype(float), sigma=smooth_sigma)
    # Threshold at 0.5 to get a cleaner envelope
    refined_mask = (smoothed > 0.5)
    return refined_mask

def estimate_background_stats(density_map, protein_mask):
    """
    Estimate background noise statistics from voxels outside the protein mask.
    """
    outside_voxels = density_map[~protein_mask].ravel()
    bg_mean = np.mean(outside_voxels)
    bg_std = np.std(outside_voxels)
    return bg_mean, bg_std
    
    


        
