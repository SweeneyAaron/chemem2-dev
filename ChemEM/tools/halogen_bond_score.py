#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 13:28:55 2025

@author: aaron.sweeney
"""

import os 
import pickle
import numpy as np


class HalogenBondScore:
    def __init__(self):
        """
        Loads all spline scoring objects from the package location:
        ChemEM/ChemDock2/halogen.
        The files are expected to be named as '{idx_donor}-{idx_acceptor}.pkl'
        """
        self.spline_models = {} 
        current_dir = os.path.dirname(__file__)
        aromatic_dir = os.path.abspath(os.path.join(current_dir, "..", "data", "ECHO_dock","halogen"))
        for filename in os.listdir(aromatic_dir):
            if filename.endswith(".pkl"):
                # Expected format: "{idx1}-{idx2}.pkl"
                base_name = filename[:-4]  # remove .pkl
                parts = base_name.split('-')
                if len(parts) == 2:
                    try:
                        idx1 = int(parts[0])
                        idx2 = int(parts[1])
                    except ValueError:
                        
                        continue  # Skip files that do not follow naming convention.
                    
                    filepath = os.path.join(aromatic_dir, filename)
                    with open(filepath, "rb") as f:
                        # Each file contains a tuple of spline functions: (spline_A, spline_B, spline_C)
                        spline_A, spline_B, spline_C = pickle.load(f)
                    self.spline_models[(idx1, idx2)] = (spline_A, spline_B, spline_C)
    
    def predict_params(self, spline_tuple, angle_s1, angle_s2):
        """
        Predicts spline parameters A, B, and C for a given angle and offset.
        """
        spline_A, spline_B, spline_C = spline_tuple
        A_pred = float(spline_A(angle_s1, angle_s2))
        B_pred = float(spline_B(angle_s1, angle_s2))
        C_pred = float(spline_C(angle_s1, angle_s2))
        return A_pred, B_pred, C_pred
    
    def buckingham_potential(self, r, A, B, C):
        """
        Computes the Buckingham potential given distance r and parameters.
        """
        return A * np.exp(-B * r) - C / (r ** 6)
    
    @staticmethod
    def compute_angle(point1, point2, point3):
        point1 = np.array(point1)
        point2 = np.array(point2)
        point3 = np.array(point3)
        ba = point1 - point2
        bc = point3 - point2
        cosine_angle = np.dot(ba, bc) / \
            (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def score_interaction(self, 
                          donor_atom, 
                          donor_root_atom, 
                          acceptor_atom, 
                          acceptor_root_atom,
                          donor_atom_type,
                          acceptor_atom_type):
        
        d = np.round(np.linalg.norm(donor_atom - acceptor_atom), 3)
        if not (2.0 < d < 5.0):
            return None # Distance cutoff not met.
        
        #angle, s1 is donor-donor root-acceptor 
        #offset, s2 is acceptor-root - acceptor - donor
        angle_s1 = self.compute_angle(donor_atom, donor_root_atom, acceptor_atom)
        angle_s2 = self.compute_angle(acceptor_root_atom, acceptor_atom, donor_atom)
        
        if angle_s1 < 50.0:
            #clip angle for scoring function
            angle_s1 = 180.0 - angle_s1
        
        
        
        
        
        if angle_s1 < 130.0:
            return None 
        if not (90 < angle_s2 < 150):
            return None 
        
        spline_tuple = self.spline_models[(donor_atom_type, acceptor_atom_type)]
        A_pred, B_pred, C_pred = self.predict_params(spline_tuple, angle_s1, angle_s2)
        score = self.buckingham_potential(d, A_pred, B_pred, C_pred)
        
        return score