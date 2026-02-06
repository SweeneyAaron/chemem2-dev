#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 13:25:11 2025

@author: aaron.sweeney
"""
import os 
import pickle
import numpy as np


class AromaticScore:
    def __init__(self):
        """
        Loads all spline scoring objects from the package location:
        ChemEM/ChemDock2/aromatic.
        The files are expected to be named as '{idx1}-{idx2}-{stack}.pkl'
        """
        self.spline_models = {}  # keyed by (base_idx, angled_idx, stack_type)
        # Get the directory path using pkg_resources. Adjust the package path as needed.
        #TODO Switch back to pkg_resources when packaging
        #aromatic_dir = pkg_resources.resource_filename("ChemEM.ChemDock2", "aromatic")
        current_dir = os.path.dirname(__file__)
        aromatic_dir = os.path.abspath(os.path.join(current_dir, "..", "data","ECHO_dock", "aromatic"))
        for filename in os.listdir(aromatic_dir):
            if filename.endswith(".pkl"):
                # Expected format: "{idx1}-{idx2}-{stack}.pkl"
                base_name = filename[:-4]  # remove .pkl
                parts = base_name.split('-')
                if len(parts) == 3:
                    try:
                        idx1 = int(parts[0])
                        idx2 = int(parts[1])
                    except ValueError:
                        continue  # Skip files that do not follow naming convention.
                    stack_type = parts[2]  # 'p' or 't'
                    filepath = os.path.join(aromatic_dir, filename)
                    with open(filepath, "rb") as f:
                        # Each file contains a tuple of spline functions: (spline_A, spline_B, spline_C)
                        spline_A, spline_B, spline_C = pickle.load(f)
                    self.spline_models[(idx1, idx2, stack_type)] = (spline_A, spline_B, spline_C)

    def predict_params(self, spline_tuple, angle, offset):
        """
        Predicts spline parameters A, B, and C for a given angle and offset.
        """
        spline_A, spline_B, spline_C = spline_tuple
        A_pred = float(spline_A(angle, offset))
        B_pred = float(spline_B(angle, offset))
        C_pred = float(spline_C(angle, offset))
        return A_pred, B_pred, C_pred

    def buckingham_potential(self, r, A, B, C):
        """
        Computes the Buckingham potential given distance r and parameters.
        """
        return A * np.exp(-B * r) - C / (r ** 6)

    @staticmethod
    def compute_centroid(atoms):
        """
        Computes the centroid of a set of atoms.
        Assumes atoms is an iterable of numpy arrays (3D coordinates).
        """
        return np.mean(np.array(atoms), axis=0)
    
    @staticmethod
    def compute_normal(atoms):
        """
        Compute a unit normal vector to the plane defined by the first three atoms.
        Assumes the atoms array has at least three points.
        """
        p1, p2, p3 = atoms[:3]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm == 0:
            raise ValueError("Cannot compute normal: points are collinear.")
        return n / norm
    
    @staticmethod
    def compute_normal(atoms):
        """
        Computes a normal vector for a ring given atom positions.
        For simplicity, uses the first three atoms.
        """
        atoms = np.array(atoms)
        if len(atoms) < 3:
            raise ValueError("Ring atoms < 3")
        v1 = atoms[1] - atoms[0]
        v2 = atoms[2] - atoms[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0:
           raise ValueError("Degenerate ring: atoms are collinear.")
           
        return normal / norm
    
    
    def score_interaction(self, ring_type1, ring_type2, atoms1, atoms2, index1, index2):
        """
        Scores aromatic interactions between two rings.
        
        Parameters:
            ring_type1 (str): e.g. "benzene", "furan", etc.
            ring_type2 (str): e.g. "benzene", "pyrrole", etc.
            atoms1 (iterable): List of 3D coordinates (numpy arrays) for ring1.
            atoms2 (iterable): List of 3D coordinates for ring2.
        
        Returns:
            float or None: The computed interaction score, or None if cutoffs are not met.
        """
        
        '''
        test value!!
        dist: 5.267
        ang: 70.51
        offset: 0.123
        '''
        
        # Ensure lowercase keys
        idx1 = ring_type1.idx
        idx2 = ring_type2.idx

        # Compute centroids and center-to-center distance.
        centroid1 = self.compute_centroid(atoms1)
        centroid2 = self.compute_centroid(atoms2)
        d = np.round(np.linalg.norm(centroid2 - centroid1), 3)
        if not (2.0 < d < 8.0):
            return None, None  # Distance cutoff not met.
        
        
        # Compute ring normals.
        try:
            normal1 = self.compute_normal(atoms1)
            normal2 = self.compute_normal(atoms2)
        except ValueError:
            return None, None
        
        
        # --- Offset calculation for ring1's plane ---
        # Project centroid2 onto the plane of ring1:
        # distance component along ring1's normal:
        d1 = centroid2 - centroid1
        comp1 = np.dot(d1, normal1)
        proj_centroid2_on_ring1 = centroid2 - comp1 * normal1
        offset1 = np.linalg.norm(proj_centroid2_on_ring1 - centroid1)
        
        # --- Offset calculation for ring2's plane ---
        # Project centroid1 onto the plane of ring2:
        d2 = centroid1 - centroid2  # which is just -(centroid2 - centroid1)
        comp2 = np.dot(d2, normal2)
        proj_centroid1_on_ring2 = centroid1 - comp2 * normal2
        offset2 = np.linalg.norm(proj_centroid1_on_ring2 - centroid2)
        
        # Return the smaller of the two offsets.
        offset =  np.round(min(offset1, offset2),3)
        if offset >= 2.5:
            return None, None
        
        dot = np.sum(normal2 * normal1)
        selected_angle = np.degrees(np.arccos(np.clip(np.abs(dot), -1.0, 1.0)))
        if selected_angle > 90:
            selected_angle = 180.0 - selected_angle
        selected_angle =np.round(selected_angle,2)
        
        vec_unit = d1 / np.linalg.norm(d1)
        
        #debug for test 
        offset = 0.368
        selected_angle = 68.2
        d = 7.724
        
        
        if (0.0 <= selected_angle <= 25.0):
            stacking_type = 'p'
            key_idx1 = idx1 
            key_idx2 = idx2
            key = (idx1, idx2,'p')
            if (idx1, idx2,'p')  in self.spline_models:
                key = (idx1, idx2,'p')
            elif (idx2, idx1,'p') in self.spline_models:
                key = (idx2, idx1,'p') 
            else:
                return None, None
        elif (65.0 <= selected_angle <= 90.0):
            stacking_type = 't'
            if idx1 == idx2:
                key = (idx2, idx1,'t') 
            else:
                # Use the connecting vector method to decide which ring is perpendicular.
                # Compute dot products between each ring's normal and the connecting vector.
                # Here, higher dot product means the normal is more aligned with the connecting vector,
                # which implies that ring is rotated (its plane is more perpendicular to the connecting line).
                lig_dot = np.abs(np.dot(normal1, vec_unit))
                prot_dot = np.abs(np.dot(normal2, -vec_unit))
                # If ligand (ring1) has a higher dot, it is the rotated ring.
                if lig_dot > prot_dot:
                    key = (idx2, idx1, 't')  # non-rotated ring first (ring2)
                else:
                    key = (idx1, idx2, 't')  # non-rotated ring first (ring1)
                
                if key not in self.spline_models:
                    key = (key[1], key[0], 't')
                    
                if key not in self.spline_models:
                    return None, None
        else:
            return None, None
                
        
        
        # Retrieve the corresponding spline model.
        spline_tuple = self.spline_models[key]
        # Predict spline parameters given the selected angle and offset.
        A_pred, B_pred, C_pred = self.predict_params(spline_tuple, selected_angle, offset)
        # Compute the interaction score using the Buckingham potential.
        score = self.buckingham_potential(d, A_pred, B_pred, C_pred)
        
        # --- Atom-Atom clash Score ---
        atoms1_arr = np.array(atoms1)  # shape (N1, 3)
        atoms2_arr = np.array(atoms2)  # shape (N2, 3)
        diff = atoms1_arr[:, np.newaxis, :] - atoms2_arr[np.newaxis, :, :]  # (N1, N2, 3)
        dist_matrix = np.linalg.norm(diff, axis=2)  # (N1, N2)
        d_safe = np.where(dist_matrix < 2.0, 2.0, dist_matrix)
        # Force using the p-stack parameters with offset=0.0 and angle=0.0.
        if (idx1, idx2, 'p') in self.spline_models:
            p_key = (idx1, idx2, 'p')
        elif (idx2, idx1, 'p') in self.spline_models:
            p_key = (idx2, idx1, 'p')
        else: 
            return score, None
        
        A_p, B_p, C_p = self.predict_params(self.spline_models[p_key], 0.0, 0.0)
        atom_atom_scores = A_p * np.exp(-B_p * d_safe) - C_p / (d_safe**6)
        # Divide the score by 6 for each atom-atom pair.
        atom_atom_scores /= 6.0
        # Set negative scores to 0.
        atom_atom_scores[atom_atom_scores < 0.0] = 0.0
        # Cap the score at 10.0.
        atom_atom_scores = np.minimum(atom_atom_scores, 10.0)

        return score, atom_atom_scores