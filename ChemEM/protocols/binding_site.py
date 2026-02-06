# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>
"""

this file contains the protocols for automatically or manually defining binding sites

binding sites are defined using either 
-- an alpha-spheres method
-- environment grid method TODO!
-- ray-tracing grid method (defualt fallback method)

"""

import os
import numpy as np 
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from collections import defaultdict

from ChemEM.parsers.EMMap import EMMap
from ChemEM.data.binding_site_model import BindingSiteModel
from ChemEM.tools.ligand import get_van_der_waals_radius
from ChemEM.tools.geometry import compute_circumsphere, select_atoms_in_sphere

from ChemEM.tools.binding_site import (calculate_ses_grid_zyx, 
                                  get_pocket_mask , 
                                  find_site_at_point,
                                  analyze_site_from_mask,
                                  generate_density_map,
                                  create_binding_site_mask,
                                  find_binding_site_boundary,
                                  compute_distance_map,
                                  compute_alpha_shape_pockets,
                                  ses_ray_trace_binding_site,
                                  compute_site_spheres,
                                  contact_atom_indices,
                                  residues_from_atom_indices,
                                  box_stats,
                                  density_and_distance_maps) 



from ChemEM.tools.biomolecule import write_residues_to_pdb
from ChemEM.messages import Messages 


class BindingSite:
    
    def __init__(self, system):
        self.system = system 
        self.binding_sites = {}
        #fall back binding site getter 
        self.fall_back_radius = 15.0
        self.padding = 6.0
        
    def get_protein_structure(self):
        self.protein_openff_structure =  self.system.protein.complex_structure
        
    def get_position_input(self):
        self.atoms = [i for i in self.protein_openff_structure.atoms if i.element > 1]
        self.positions = np.array([[i.xx, i.xy, i.xz] for i in self.atoms ])
        self.atom_radii = np.array( [get_van_der_waals_radius(i.element_name) for i in self.atoms] )
        
    def set_grid_spacing(self):
        if self.system.density_map is not None:
            self.grid_spacing = self.system.density_map.apix[0]
        else:
            self.grid_spacing = self.system.options.binding_site_grid_spacing
    
    def get_centroid_binding_sites(self):
        
        ligand_binding_clusters = {}
        
        for num, centroid in enumerate(self.system.centroid):
            centroid = np.array(centroid)
            centroid_to_clusters = []
            
            # self.pockets contains {label: [ [atom_idx1, atom_idx2...], ... ] }
            for cluster_label, tetras in self.pockets.items():
                site_centers = []
                site_radii = []
                for tetra in tetras:
                    vertices = self.positions[tetra]
                    center, radius = compute_circumsphere(vertices)
                    site_centers.append(center)
                    site_radii.append(radius)
                
                site_centers = np.array(site_centers)
                site_radii = np.array(site_radii)
                
                centroid_reshaped = centroid.reshape(1, -1)
                distances = cdist(centroid_reshaped, site_centers)
                contacts = distances < site_radii[np.newaxis, :]
                centroid_indices, _ = np.where(contacts)
                if len(centroid_indices) > 0:
                    centroid_to_clusters += tetras
            
            centroid_key = num 
            
            if centroid_to_clusters:
                
                ligand_binding_clusters[centroid_key] = centroid_to_clusters
            else:
                
                # Fall back to SES extraction via helper function
                fallback_site = ses_ray_trace_binding_site(
                    centroid=centroid,
                    centroid_key=centroid_key,
                    atoms=self.atoms,
                    positions=self.positions,
                    atom_radii=self.atom_radii,
                    protein_openff_structure=self.protein_openff_structure,
                    system_output_dir=self.system.output,
                    grid_spacing=self.grid_spacing,
                    fall_back_radius=self.fall_back_radius
                )
                
                if fallback_site:
                    self.binding_sites[centroid_key] = fallback_site
        
        self.ligand_binding_clusters = ligand_binding_clusters

    def get_binding_sites(self):
        """
        For each binding site, identify bounding/lining residues and compute properties.
        """
    
        for site_label, tetras in self.ligand_binding_clusters.items():
            if not tetras:
                continue
    
            # 1) Sphere representation of site
            site_centers, site_radii = compute_site_spheres(self.positions, tetras)
    
            # 2) Bounding residues (padding)
            bounding_atom_indices = contact_atom_indices(
                site_centers=site_centers,
                site_radii=site_radii,
                atom_positions=self.positions,
                atom_radii=self.atom_radii,
                extra_distance=self.padding,
            )
            bounding_residues = residues_from_atom_indices(bounding_atom_indices, self.atoms)
    
            # 3) Lining residues (config distance)
            lining_atom_indices = contact_atom_indices(
                site_centers=site_centers,
                site_radii=site_radii,
                atom_positions=self.positions,
                atom_radii=self.atom_radii,
                extra_distance=self.system.options.lining_residue_distance,
            )
            lining_residues = residues_from_atom_indices(lining_atom_indices, self.atoms)
    
            # 4) Simple geometric stats
            binding_site_centroid, min_coords, max_coords, bounding_box_size = box_stats(site_centers)
    
            # 5) Maps + volume
            denmap_path = os.path.join(self.system.output, f"site_{site_label}.mrc")
            distmap_path = os.path.join(self.system.output, f"site_{site_label}_distance.mrc")
            pdb_path = os.path.join(self.system.output, f"site_{site_label}.pdb")
    
            origin, box_size, densmap, distance_map, apix, site_volume = density_and_distance_maps(
                site_centers=site_centers,
                site_radii=site_radii,
                grid_spacing=self.grid_spacing,
                denmap_path=denmap_path,
                density_cutoff=0.1,
            )
    
            # Optional debug output (kept explicit + separate path)
            dmap = EMMap(origin, apix, distance_map, 3.0)
            dmap.write_mrc(distmap_path)
    
            data = {
                "key": site_label,
                "tetrahedrals": tetras,
                "site_centers": site_centers,
                "site_radii": site_radii,
                "binding_site_centroid": binding_site_centroid,
                "min_coords": min_coords,
                "max_coords": max_coords,
                "bounding_box_size": bounding_box_size,
                "origin": origin,
                "box_size": box_size,
                "densmap": densmap,
                "distance_map": distance_map,
                "apix": apix,
                "volume": site_volume,
                "bounding_atom_indices": bounding_atom_indices,
                "lining_atom_indices": lining_atom_indices,
                "residues": bounding_residues,
                "lining_residues": lining_residues,
            }
    
            
            data["rdkit_mol"] = write_residues_to_pdb(
                bounding_residues,
                self.protein_openff_structure.positions,
                pdb_path,
                write=False,
            )
            
            data["rdkit_lining_mol"] = write_residues_to_pdb(
                lining_residues,
                self.protein_openff_structure.positions,
                pdb_path.replace(".pdb", "_lining.pdb"),
            )
    
            self.binding_sites[site_label] = BindingSiteModel.from_dict(data)


    
    def log(self):
        """
        Generates a summary log of the binding site detection process and 
        writes it to the system log.
        """
        opts = self.system.options
        num_sites = len(self.binding_sites)
        
        # --- Build Header and Parameters ---
        log_lines = []
        log_lines.append("\n" + "="*60)
        log_lines.append(f"BINDING SITE DETECTION SUMMARY")
        log_lines.append("="*60)
        log_lines.append(f"Total Sites Identified: {num_sites}")
        log_lines.append("-" * 60)
        log_lines.append(f"Parameters Used:")
        log_lines.append(f"  Grid Spacing:       {self.grid_spacing:.2f} Å")
        log_lines.append(f"  Probe Radius:       {opts.probe_sphere_min:.1f} - {opts.probe_sphere_max:.1f} Å")
        log_lines.append(f"  Clustering Thr:     Pass 1: {opts.first_pass_thr:.2f}, Pass 2: {opts.second_pass_thr:.2f}")
        log_lines.append(f"  Merge Overlaps:     {opts.n_overlaps} links (Thr: {opts.third_pass_thr:.2f} Å)")
        log_lines.append("-" * 60)

        # --- Log Details for Each Site ---
        if num_sites == 0:
            log_lines.append("No binding sites were found matching the criteria.")
        else:
            sorted_keys = sorted(self.binding_sites.keys(), key=lambda x: str(x))
            
            for key in sorted_keys:
                site = self.binding_sites[key]
                
                # Safely access attributes (assuming BindingSiteModel attributes)
                vol = getattr(site, 'volume', 0.0)
                centroid = getattr(site, 'binding_site_centroid', np.array([0.0, 0.0, 0.0]))
                n_residues = len(getattr(site, 'residues', []))
                n_lining = len(getattr(site, 'lining_residues', []))
                
                # Format Centroid
                c_str = f"[{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}]"
                
                log_lines.append(f"Site ID: {key}")
                log_lines.append(f"  Volume:            {vol:.2f} Å³")
                log_lines.append(f"  Centroid (x,y,z):  {c_str}")
                log_lines.append(f"  Bounding Residues: {n_residues}")
                log_lines.append(f"  Lining Residues (<= {self.system.options.lining_residue_distance} Å):   {n_lining}")
                log_lines.append("." * 60)

        log_lines.append("="*60 + "\n")
        
        # --- Write to System Log ---
        full_message = "\n".join(log_lines)
        self.system.log(full_message)
    
    def run(self):
        self.system.log(Messages.create_centered_box('Binding Site Segmentation'))
        
        self.get_protein_structure()
        self.set_grid_spacing()
        self.get_position_input()
        
        if self.system.options.force_new_site:
            #TODO!
            
            pass
        else:
            # Automated binding site segmentation
            self.pockets = compute_alpha_shape_pockets(
                positions=self.positions,
                probe_min=self.system.options.probe_sphere_min,
                probe_max=self.system.options.probe_sphere_max,
                first_pass_thr=self.system.options.first_pass_thr,
                first_pass_min_size=self.system.options.fist_pass_cluster_size,
                second_pass_thr=self.system.options.second_pass_thr,
                third_pass_thr=self.system.options.third_pass_thr,
                n_overlaps=self.system.options.n_overlaps
            )
            
            if (self.system.centroid is not None) and len(self.system.centroid):
                self.get_centroid_binding_sites()
            else:
                self.ligand_binding_clusters = self.pockets
        
        self.get_binding_sites()
        
        self.system.binding_sites = self.binding_sites
        self.log()
        
        
        
        
        