#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

import os 
import sys
import numpy as np 
import json 
from ChemEM.messages import Messages
from ChemEM.parsers.EMMap import EMMap
from ChemEM.tools.ligand import get_van_der_waals_radius
from ChemEM.tools.geometry import compute_circumsphere
from ChemEM.tools.density import (MapTools,
                                  calculate_ses_grid_zyx,
                                  create_binding_site_mask,
                                  otsu_mask_from_map,
                                  find_binding_site_boundary,
                                  compute_distance_map,
                                  extract_ligand_density,
                                  extract_ligand_density_otsu,
                                  get_disconected_densities,
                                  extract_min_bounding_box,
                                  get_map_features,
                                  extract_subvolume_from_grid,
                                  site_from_densmap)
from ChemEM.data.binding_site_model import BindingSiteModel
from ChemEM.tools.biomolecule import write_residues_to_pdb

from scipy.spatial import Delaunay



class AlphaMask:
    def __init__(self, system):
        self.system = system 
        self.sliced_density_maps = {}
        #don't need these any more if i call binding site before
        
        self.PROBE_SPHERE_MIN = 3.0  # Minimum radius for binding site detection
        self.PROBE_SPHERE_MAX = 6.0
        self.pad = system.options.alpha_pad
        self.dist_cutoff = 6.0
        self.percent_above_otsu_threshold = 0.0 # values between 0. and 1.
        self.non_zero_voxel_count_threshold = 50 #basically size !!! 
        
        
        self._removed_binding_site_keys = []
        self._otsu_thr = None
        self._n_labeled_features = None
        self._pre_binding_sites_count = None
        self._pre_binding_site_maps_count = None
        
    def mk_output(self):
        self.output = os.path.join(self.system.output, 'alpha_mask')
        try:
            os.mkdir(self.output)
        except FileExistsError:
            pass
    
    def get_protein_structure(self):
        #TODO! system ligands!!
        self.protein_openff_structure =  self.system.protein.complex_structure
    
    def get_position_input(self):
        
        #can probably move this to the actual protein object so it gets calculated once
        self.atoms = [i for i in self.protein_openff_structure.atoms if i.element > 1]
        self.positions = np.array([[i.xx, i.xy, i.xz] for i in self.atoms ])
        self.atom_radii = np.array( [get_van_der_waals_radius(i.element_name) for i in self.atoms] )
    
    def set_grid_spacing(self):
        self.grid_spacing = self.system.density_map.apix[0]
        
        
    def get_delaunay(self):
        #Don't recompute 
        if getattr(self.system.protein, 'delaunay', None) is None:
            self.delaunay = Delaunay(self.positions)
            self.tetrahedra = self.delaunay.simplices 
        else:
            self.delaunay = self.system.protein.delaunay
            self.tetrahedra = self.delaunay.simplices 
        
    def calculate_circumsphere(self):
        self.circumspheres = []
        for tetra in self.tetrahedra:
            vertices = self.positions[tetra]
            
            center, radius = compute_circumsphere(vertices)
            
            self.circumspheres.append((center, radius))
    
    def filter_circumspheres(self):
        self.candidate_tetrahedra = []
        self.candidate_centers = []
        self.candidate_radii = []

        for i, (center, radius) in enumerate(self.circumspheres):
            if self.system.options.probe_sphere_min <= radius <= self.system.options.probe_sphere_max:
                self.candidate_tetrahedra.append(self.tetrahedra[i])
                self.candidate_centers.append(center)
                self.candidate_radii.append(radius)
    
    def get_mask(self):
        
        '''
        this function is very slow
       
        '''
        
        
        copy_map = MapTools.trim_map_to_atoms(self.system.confidence_map.copy(),
                                              self.positions,
                                              self.pad,
                                              inplace=False)
        
        
        shape = copy_map.density_map.shape
        
        
        if self.system.options.ses_mask:
            #use SASA mask
            site_mask, _, _ = calculate_ses_grid_zyx(
                self.positions,
                self.atom_radii,
                map_shape_zyx = shape,
                map_origin_xyz = copy_map.origin,  
                apix = copy_map.apix[0]
            ) 
            #flip mask == sasa 
            site_mask = ~site_mask
        else:
            #use alpha sphere intermediate spheres for sasa
            site_mask = create_binding_site_mask(shape, 
                                             self.candidate_centers, 
                                             self.candidate_radii, 
                                             copy_map.apix,
                                             copy_map.origin)
        
        
        
        masked_map = copy_map.copy()
        masked_map.density_map = copy_map.density_map * site_mask
        
        
        if not self.system.options.no_otsu_filter_ses_mask:
            otsu_filtered_map, thr = otsu_mask_from_map( masked_map.density_map.copy())
            masked_map.density_map = masked_map.density_map.copy() * otsu_filtered_map 
        
        
        
        if not self.system.options.no_boundry:
            if self.system.options.ses_mask:
                b_mask = create_binding_site_mask(shape, 
                                                  self.candidate_centers, 
                                                  self.candidate_radii, 
                                                  copy_map.apix,
                                                  copy_map.origin)
            else:
                b_mask = site_mask.copy()
            
            
            boundry_mask = find_binding_site_boundary(b_mask)
            #apply boundry 
            distance_map = compute_distance_map(boundry_mask, copy_map.apix)
            masked_map.density_map *= distance_map
            #out_fn = os.path.join(self.output, 'alpha_boundry_applied_1.mrc')
            #masked_map.write_mrc(out_fn)
            
        else:
            distance_map = compute_distance_map(site_mask, copy_map.apix)
           # out_fn = os.path.join(self.output, 'alpha_boundry_applied_2.mrc')
            #masked_map.write_mrc(out_fn)
        
        
        #out_fn = os.path.join(self.output, 'alpha_masked_ses.mrc')
        #masked_map.write_mrc(out_fn)
        
        distance_copy = copy_map.copy() 
        distance_copy.density_map = distance_map 

        #out_fn = os.path.join(self.output, 'alpha_distance_map.mrc')
        #distance_copy.write_mrc(out_fn)
        
        
        self.distance_map = distance_copy
        self.masked_density = masked_map
       # out_fn = os.path.join(self.output, 'alpha_masked_map_final.mrc')
       # masked_map.write_mrc(out_fn)
        
        
    def significant_features(self):
        '''
        extract subvolume from density_map -> self.sliced_density_maps
        
        '''
        self._slice_binding_site_density()
        self._get_significant_features()
        
        
    def _slice_binding_site_density(self):
        ''' 
            extracts a sub-volume from the full map. uses the confidence filtered map.
        '''
        
        none_keys = []
        for key, binding_site in self.system.binding_sites.items():
            try:
                site_centers = binding_site.site_centers  # (x, y, z)
                site_radii = binding_site.site_radii 
                
                density_map_slice = extract_subvolume_from_grid(self.system.confidence_map.origin,
                                                                np.array(self.system.confidence_map.apix),
                                                                self.system.confidence_map.density_map,
                                                                binding_site.box_size, 
                                                                grid_origin=binding_site.origin,
                                                                resolution=self.system.confidence_map.resolution)
                if density_map_slice is None:
                    none_keys.append(key)
                    continue
                
                self.sliced_density_maps[key] = density_map_slice
            
            except Exception as e:
                self.system.log(Messages.chemem_warning(self.__name__, "significant_features", f'Removing site {key} due to Error in resample: {e}'))
                none_keys.append(key)
                continue
        
        for key in none_keys:
            del self.system.binding_sites[key]
            
        
    def _get_significant_features(self):
        '''
        should mutate
        self.system.binding_sites = new_binding_sites 
        self.system.binding_site_maps = new_maps 
        '''
        
        ligand_densities = []
        ligand_features = []
        binding_site_keys = []
        for key, binding_site in self.system.binding_sites.items():
            if key not in self.sliced_density_maps:
                continue
            distance_map = binding_site.distance_map
            densmap = self.sliced_density_maps[key]
             
            ligand_dens, ligand_feat = extract_ligand_density(densmap.density_map, 
                                                              distance_map, 
                                                              densmap.apix,
                                                              high_threshold_sigma=self.system.options.sf_sigma_thr)
            
            ligand_densities += ligand_dens 
            ligand_features += ligand_feat 
            binding_site_keys += [key for _ in ligand_dens]
            
            if not ligand_features:
                continue
            
            max_thr = np.max([i['map_amplitude_thr_no_mask'] for i in ligand_features])
        
        site_ligands = {}
        for density, features, key in zip(ligand_densities, ligand_features, binding_site_keys):
            new_map = self.sliced_density_maps[key].copy()
            new_map.density_map = density
            if key in site_ligands:
                site_ligands[key].append((new_map, features))
            else:
                site_ligands[key] = [(new_map, features)]
            
        
        combined_dic = {}
        for key, value in site_ligands.items():
            combined_map = None
            summed_values = {}
            ligand_count = 0
            for vals in value:
                #cutoffs
                if vals[1]['centroid'] < self.system.options.sf_centroid_thr:
                    continue
                elif vals[1]['volume'] < self.system.options.sf_volume_thr:
                    continue 
                elif vals[1]['mean_non_zero_density_value'] < vals[1]['map_amplitude_thr_no_mask'] * self.system.options.sf_amp_frac:
                    continue
                
                #start a new map if needed
                if combined_map is None:
                    combined_map = vals[0].copy()
                else:
                    combined_map.density_map += vals[0].density_map
                
                for metric_key, metric_value in vals[1].items():
                    # If the key is new, initialize it with 0 before adding
                    summed_values[metric_key] = summed_values.get(metric_key, 0) + metric_value
                
                ligand_count += 1
            averaged_values = {}
            if ligand_count > 0:
                for metric_key, total_sum in summed_values.items():
                    averaged_values[metric_key] = total_sum / ligand_count
            
                new_dmap_out = os.path.join(self.output, f'combined_map_{key}.mrc')
                #combined_map.write_mrc(new_dmap_out)
                combined_dic[key] = [(combined_map,averaged_values)]
        
        
        self.system.binding_site_maps = combined_dic 
    
    def get_features(self):
        '''
            -- build a “combined feature density” map (two segmentation modes)
            -- label disconnected components
            -- for each component: bbox-> submap-> centroid-> write mrc
            -- decide “belongs to existing site?” vs “make a new site”
            -- write outputs + mutate self.system state
        '''
        
        apix = np.array(self.masked_density.apix, dtype=float)
        base_origin = np.array(self.masked_density.origin, dtype=float)
        combined_map = self._build_combined_map()
        features = self._label_features(combined_map)
        site_id = self._get_next_site_id_start()
        
        centroids: dict[str, list[float]] = {}
        system_sites: dict[int, list[tuple[EMMap, dict]]] = {}
        new_binding_sites: dict[int, BindingSiteModel] = {}
        
        for num in np.unique(features)[1:]:
            densmap = combined_map * (features == num)
            
            sub_map, out_name, sub_centroid, binding_site_key, binding_site_values = \
                self._feature_bbox_to_submap(
                    num=num,
                    densmap=densmap,
                    apix=apix,
                    base_origin=base_origin,
                )
            
            if sub_map is None:
                continue
    
            # write bbox map + record centroid
            centroids[out_name] = sub_centroid
            #sub_map.write_mrc(os.path.join(self.output, out_name))
    
            feat = self._compute_feature_stats(num=num, densmap=densmap, apix=apix)
            feat["feature_id"] = num
    
            if binding_site_key is not None:
                self._add_feature_to_existing_site(
                    system_sites=system_sites,
                    binding_site_key=binding_site_key,
                    binding_site_values=binding_site_values,
                    densmap=densmap,
                    feat=feat,
                    apix=apix,
                    base_origin=base_origin,
                )
            else:
                binding_site, resampled_density = self._create_new_site_from_submap(
                    site_id=site_id,
                    sub_map=sub_map,
                    sub_centroid=sub_centroid,
                )
                new_binding_sites[site_id] = binding_site
                system_sites[site_id] = [(resampled_density, feat)]
                site_id += 1
    
        self._write_feature_outputs(
            combined_map=combined_map,
            centroids=centroids,
            system_sites=system_sites,
            new_binding_sites=new_binding_sites,
        )
        
    def _get_next_site_id_start(self) -> int:
        """Return the next free binding-site id (max existing key + 1), or 0 if none."""
        keys = list(self.system.binding_sites.keys())
        return (max(keys) + 1) if keys else 0
    
    
    def _build_combined_map(self) -> np.ndarray:
        
        combined_map = np.zeros(self.masked_density.density_map.shape, dtype=float)
    
        if not self.system.options.otsu_segment:
            ligand_dens, ligand_feat = extract_ligand_density(
                self.masked_density.density_map,
                self.distance_map.density_map,
                self.masked_density.apix,
                high_threshold_sigma=self.system.options.sf_sigma_thr,
                grad_threshold=self.system.options.grad_thr,
            )
    
            for dens, feat in zip(ligand_dens, ligand_feat):
                if feat["centroid"] < self.system.options.sf_centroid_thr:
                    continue
                if feat["volume"] < self.system.options.sf_volume_thr:
                    continue
                if feat["mean_non_zero_density_value"] < (
                    feat["map_amplitude_thr_no_mask"] * self.system.options.sf_amp_frac
                ):
                    continue
                combined_map += dens
            return combined_map
    
        feature_maps = extract_ligand_density_otsu(
            self.masked_density.density_map,
            self.masked_density.resolution,
            sigma_coeff=self.system.options.sigma_coeff,
        )
    
        for sig_feat in feature_maps:
            if sig_feat[0][2] > self.percent_above_otsu_threshold and sig_feat[0][0] > self.non_zero_voxel_count_threshold:
                combined_map += sig_feat[1]
    
        return combined_map

    def _label_features(self, combined_map: np.ndarray) -> np.ndarray:
        return get_disconected_densities(combined_map, 0.0)
    
    def _feature_bbox_to_submap(self, *, num, densmap, apix, base_origin):
        
        bbox = extract_min_bounding_box(densmap, thr=0.0, pad=1.0)
        if bbox is None:
            return None, None, None, None, None
    
        subvol, (z0, y0, x0), (z1, y1, x1) = bbox
    
        new_origin = base_origin + np.array([x0 * apix[0], y0 * apix[1], z0 * apix[2]], dtype=float)
        sub_map = EMMap(new_origin, apix, subvol, self.masked_density.resolution)
    
        sub_centroid = sub_map.center_of_mass()
        out_name = f"feature_{num}_bbox.mrc"
    
        binding_site_key, binding_site_values = self._centroid_in_existing_site(sub_centroid)
    
        return sub_map, out_name, sub_centroid, binding_site_key, binding_site_values

    def _centroid_in_existing_site(self, centroid):
        for k, v in self.system.binding_sites.items():
            mn = v.min_coords
            mx = v.max_coords
            if (mn[0] < centroid[0] < mx[0]) and (mn[1] < centroid[1] < mx[1]) and (mn[2] < centroid[2] < mx[2]):
                return k, v
        return None, None
    
    def _compute_feature_stats(self, *, num, densmap, apix) -> dict:
        return get_map_features(
            self.masked_density.density_map,
            densmap,
            self.distance_map.density_map,
            apix,
            num,
        )

    def _add_feature_to_existing_site(self, *, system_sites, binding_site_key, binding_site_values,
                                 densmap, feat, apix, base_origin):
        new_map = extract_subvolume_from_grid(
            base_origin,
            apix,
            densmap,
            binding_site_values.box_size,
            grid_origin=binding_site_values.origin,
            resolution=self.masked_density.resolution,
        )
        
        system_sites.setdefault(binding_site_key, []).append((new_map, feat))
    
    def _create_new_site_from_submap(self, *, site_id: int, sub_map: EMMap, sub_centroid):
        
        """
        TODO! This should be a stand alone function
        """
        residues, unique_atom_indices, site_distance_map, resampled_density = site_from_densmap(
            sub_map,
            self.protein_openff_structure,
            self.distance_map,
            distance_cutoff=6.0,
            density_threshold=0.0,
        )
    
        #site_distance_map.write_mrc(os.path.join(self.system.output, f"site_{site_id}_dist_map_force_new_site.mrc"))
    
        site_min_coords = site_distance_map.origin
        site_max_coords = site_distance_map.origin + (
            np.array([site_distance_map.x_size, site_distance_map.y_size, site_distance_map.z_size]) * site_distance_map.apix
        )
        bounding_box_size = (
            np.array([site_distance_map.x_size, site_distance_map.y_size, site_distance_map.z_size]) * site_distance_map.apix
        )
    
        pdb_path = os.path.join(self.system.output, f"site_{site_id}_force_new_site.pdb")
        rdkit_mol = write_residues_to_pdb(residues, self.protein_openff_structure.positions, pdb_path, write=False)
    
        pdb_path = os.path.join(self.system.output, f"site_lining_residues_{site_id}.pdb")
        rdkit_lining_mol = write_residues_to_pdb(residues, self.protein_openff_structure.positions, pdb_path)
    
        data = {
            "residues": residues,
            "lining_residues": residues,
            "tetrahedrals": [],
            "unique_atom_indices": unique_atom_indices,
            "binding_site_centroid": sub_centroid,
            "min_coords": site_min_coords,
            "max_coords": site_max_coords,
            "bounding_box_size": bounding_box_size,
            "site_centers": [],
            "site_radii": [],
            "volume": np.sum(site_distance_map.density_map > 0.1) * (site_distance_map.apix ** 3),
            "distance_map": site_distance_map.density_map,
            "densmap": site_distance_map.density_map,
            "origin": site_distance_map.origin,
            "box_size": site_distance_map.box_size,
            "apix": site_distance_map.apix,
            "key": site_id,
            "rdkit_mol": rdkit_mol,
            "rdkit_lining_mol": rdkit_lining_mol,
            "opening_points": [],
            "candidate_opening_points": [],
            "solvent_open_map": np.zeros(site_distance_map.density_map.shape),
        }
    
        return BindingSiteModel.from_dict(data), resampled_density

    def _write_feature_outputs(self, *, combined_map, centroids, system_sites, new_binding_sites):
        copy_map = self.masked_density.copy()
        copy_map.density_map = combined_map
        copy_map.write_mrc(os.path.join(self.output, "combined_map.mrc"))
    
        with open(os.path.join(self.output, "centroids.json"), "w") as f:
            json.dump(centroids, f, ensure_ascii=False, indent=2)
    
        self.system.binding_site_maps = system_sites
        '''
        for k, v in system_sites.items():
            base = os.path.join(self.output, f"subsite_map_{k}")
            for n, (mp, _) in enumerate(v):
                mp.write_mrc(base + f"{n}.mrc")
        '''
        for k, v in new_binding_sites.items():
            self.system.binding_sites[k] = v
    
    def handle_centroids(self):
        
        if self.system.centroid:
            
            new_maps = {}
            new_binding_sites = {}
            
            for key, site in self.system.binding_sites.items():
                min_coords = site.origin
                max_coords = site.origin + (site.apix * np.array([site.box_size[2],site.box_size[1],site.box_size[0]] ))
                
                for centroid in self.system.centroid:
                    if np.all(centroid > min_coords) and np.all(centroid < max_coords):
                        
                        
                        if key in self.system.binding_site_maps:
                            new_binding_sites[key] = site 
                            new_maps[key] = self.system.binding_site_maps[key]
                            break
        
            
            self.system.binding_sites = new_binding_sites 
            self.system.binding_site_maps = new_maps 
    
    def log(self):
        """Log a summary of what AlphaMask produced."""
        log = self.system.log
    
        log(Messages.create_centered_box("Alpha-mask summary"))
        log(f"Output directory: {getattr(self, 'output', None)}")
    
        # Options / mode
        mode = "significant_features" if self.system.options.segment_binding_sites else (
            "otsu_segment" if self.system.options.otsu_segment else "sigma+grad_segment"
        )
        log(f"Segmentation mode: {mode}")
        log(f"SES mask: {bool(self.system.options.ses_mask)}")
        log(f"Apply Otsu-to-masked-map: {bool(not self.system.options.no_otsu_filter_ses_mask)}")
        if self._otsu_thr is not None:
            log(f"Otsu threshold used: {self._otsu_thr:.6g}")
        log(f"Apply boundary distance weighting: {bool(not self.system.options.no_boundry)}")
    
        # Geometry counts
        n_atoms = len(getattr(self, "atoms", []) or [])
        log(f"Protein heavy atoms used: {n_atoms}")
    
        n_tetra = int(getattr(getattr(self, "tetrahedra", None), "shape", [0])[0]) if getattr(self, "tetrahedra", None) is not None else 0
        n_spheres = len(getattr(self, "circumspheres", []) or [])
        n_candidates = len(getattr(self, "candidate_centers", []) or [])
        log(f"Delaunay tetrahedra: {n_tetra}")
        log(f"Circumspheres computed: {n_spheres}")
        log(f"Candidate alpha-spheres kept (radius filter): {n_candidates}  "
            f"(min/max = {self.system.options.probe_sphere_min}/{self.system.options.probe_sphere_max} Å)")
    
        if getattr(self, "candidate_radii", None):
            r = np.asarray(self.candidate_radii, dtype=float)
            if r.size:
                log(f"Candidate radii stats (Å): min={float(np.min(r)):.2f}, "
                    f"median={float(np.median(r)):.2f}, max={float(np.max(r)):.2f}")
    
        # Map info
        md = getattr(self, "masked_density", None)
        if md is not None:
            log(f"Masked map: shape(z,y,x)={tuple(md.density_map.shape)}, "
                f"apix={tuple(map(float, md.apix))}, origin={tuple(map(float, md.origin))}, "
                f"resolution={getattr(md, 'resolution', None)}")
    
        # Feature counts
        if self._n_labeled_features is not None:
            log(f"Labeled disconnected features in combined map: {self._n_labeled_features}")
    
        # System outputs (binding sites / maps)
        sites = getattr(self.system, "binding_sites", {}) or {}
        maps = getattr(self.system, "binding_site_maps", {}) or {}
    
        if self._pre_binding_sites_count is not None:
            log(f"Binding sites: {self._pre_binding_sites_count} -> {len(sites)}")
        else:
            log(f"Binding sites: {len(sites)}")
    
        if self._pre_binding_site_maps_count is not None:
            log(f"Binding-site maps: {self._pre_binding_site_maps_count} -> {len(maps)}")
        else:
            log(f"Binding-site maps: {len(maps)}")
    
        if self._removed_binding_site_keys:
            log(f"Removed sites during slicing/resampling: {sorted(set(self._removed_binding_site_keys))}")
    
        if maps:
            log("Per-site feature-map counts:")
            for k in sorted(maps.keys()):
                try:
                    log(f"  - site {k}: {len(maps[k])} map(s)")
                except Exception:
                    log(f"  - site {k}: <unavailable>")
    
        if getattr(self.system, "centroid", None):
            log(f"Centroid filter applied: {len(self.system.centroid)} centroid(s)")
    
        

    def run(self):
        if self.system.options.no_map or self.system.density_map is None:
            self.system.log(Messages.chemem_warning(self.__class__, "run", "[Warning] No map defined, skipping alpha-mask"))
            return 
        
        self.system.log(Messages.create_centered_box('Alpha-masking'))
        
        self._pre_binding_sites_count = len(getattr(self.system, "binding_sites", {}) or {})
        self._pre_binding_site_maps_count = len(getattr(self.system, "binding_site_maps", {}) or {})

        
        self.mk_output()
        self.get_protein_structure()
        self.set_grid_spacing()
        self.get_position_input()
        import time 
        t1 = time.perf_counter()
        
        self.get_delaunay()
        t2 = time.perf_counter()
        rt = t2 - t1
        print('get_delauny', rt)
        
        t1 = time.perf_counter()
        self.calculate_circumsphere()
        t2 = time.perf_counter()
        rt = t2 - t1
        print('calc_circum', rt)
        
        t1 = time.perf_counter()
        self.filter_circumspheres()
        t2 = time.perf_counter()
        rt = t2 - t1
        print('filt_circum', rt)
        
        t1 = time.perf_counter()
        self.get_mask() 
        t2 = time.perf_counter()
        rt = t2 - t1
        print('get_mask', rt)
        
        
        if self.system.options.segment_binding_sites:
            self.system.log('-- Using Significant Feature Extraction')
            self.significant_features()
            
        else:
            if self.system.options.force_new_site:
                self.system.binding_sites = {}
            t1 = time.perf_counter()
            self.get_features() 
            t2 = time.perf_counter()
            rt = t2 - t1
            print('get_feat', rt)
            
        t1 = time.perf_counter()
        self.handle_centroids()
        t2 = time.perf_counter()
        rt = t2 - t1
        print('handel_cent', rt)
        self.log()
        
        #TODO! add loggin
        
    
    