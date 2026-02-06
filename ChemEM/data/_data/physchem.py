# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


class XlogP3():
    ''' Calculated logp values for protein atoms'''
    def __init__(self):
        self.xlogp3_values = {'GLY': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148, 'N':-0.2610,'CA':-0.0821},
     'ALA': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148, 'N':-0.2610,'CA':-0.1426,'CB':0.5201},
     'VAL': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':0.1485,'CG1':0.5201,'CG2':0.5201},
     'ILE': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':0.1485,'CG2':0.5201,'CG1':0.3436,'CD1':0.5201},
     'LEU': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':0.3436,'CG':0.1485,'CD1':0.5201,'CD2':0.5201},
     'MET': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':0.3436,'CG':-0.0821,'SD':0.4125,'CE':0.0402},
     'PHE': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':0.2718,'CG':0.1911,'CD1':0.3157,'CD2':0.3157,'CE1':0.3157,'CE2':0.3157,'CZ':0.3157},
     'TYR': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':0.2718,'CG':0.1911,'CD1':0.3157,'CD2':0.3157,'CE1':0.3157,'CE2':0.3157,'CZ':-0.0112,'OH':-0.0381},
     'TRP': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':0.2718, 'CG':0.1911,'CD1':-0.1039,'CD2':0.1911,'NE1':0.2172,'CE2':-0.0112,'CE3':0.3157,'CZ2':0.3157,'CZ3':0.3157,'CH2':0.3157},
     'SER': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':-0.0821,'OG':-0.4802},
     'THR': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':-0.1426,'CG2':0.5240,'OG1':-0.4802},
     'ASN': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':0.3436,'CG':-0.8076,'OD1':0.7148,'ND2':-0.6414},
     'GLN': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':0.3436,'CG':0.3436,'CD':-0.8076,'OE1':0.7148,'NE2':-0.6414},
     'ASP': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':0.3436,'CG':-0.8076,'OD1':-0.4802,'OD2':-0.4802},
     'GLU': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':0.3436,'CG':0.3436,'CD':-0.8076,'OE1':-0.4802,'OE2':-0.4802},
     'ARG': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':0.3436,'CG':0.3436,'CD':-0.0821,'NE':-0.2610,'CZ':-0.8076,'NH1':-0.7445,'NH2':-0.7445},
     'LYS': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':0.3436,'CG':0.3436,'CD':0.3436,'CE':-0.0821,'NZ':-0.7445},
     'HIS': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':0.2718,'CG':-0.1874,'ND1':0.3181,'CD2':-0.1039,'CE1':-0.1039,'NE2':0.3181},
     'CYS': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':-0.2610,'CA':-0.1426,'CB':0.0821,'SG':0.4927},
     'PRO': {'C': -0.8076, 'O':0.7148, 'OXT':0.7148,'N':0.3333,'CA':-0.1426,'CB':0.3436,'CG':0.3436,'CD':-0.0821}
     }
    def get_logp(self, residue_name, atom_name):
        try:
            return self.xlogp3_values[residue_name][atom_name]
        except:
            return 0.0


class IonData:
    '''Basic data related to ions'''
    def __init__(self):
        self.ion_ids = ["Ca", "Mn", "Mg", "Fe", "Cu", "Zn"]
        self.coordinate_residues = ['ASP', 'GLU', 'SER', 'THR', 'TYR', 'HIS', 'CYS']
        self.ion_protein_distances = {
            "CA": {"HOH": 2.39, "ASP": 2.36, "GLU": 2.36, "SER": 2.43, "THR": 2.43, "TYR": 2.20, "HIS": 2.38, "CYS": 2.56, "O": 2.36},
            "MG": {"HOH": 2.07, "ASP": 2.26, "GLU": 2.26, "SER": 2.10, "THR": 2.10, "TYR": 1.87, "HIS": 2.05, "CYS": 2.03, "O": 2.26},
            "MN": {"HOH": 2.21, "ASP": 2.21, "GLU": 2.21, "SER": 2.25, "THR": 2.25, "TYR": 1.88, "HIS": 2.19, "CYS": 2.35, "O": 2.21},
            "FE": {"HOH": 2.09, "ASP": 2.01, "GLU": 2.01, "SER": 2.13, "THR": 2.13, "TYR": 1.93, "HIS": 2.08, "CYS":  2.27, "O": 2.01},
            "CU": {"HOH": 1.97, "ASP": 1.96, "GLU": 1.96, "SER": 2.00, "THR": 2.00, "TYR": 1.90, "HIS": 1.99, "CYS":  2.17, "O": 1.96},
            "ZN": {"HOH": 2.09, "ASP": 2.04, "GLU": 2.04, "SER": 2.14, "THR": 2.14, "TYR": 1.95, "HIS": 2.00, "CYS":  2.29, "O": 2.04}
            }
        self.ion_ligand_distances = {
            "CA": {"O": 2.43, "N": 2.38, "ELSE": 2.56},
            "MG": {"O": 2.10, "N": 2.05, "ELSE": 2.03},
            "MN": {"O": 2.25, "N": 2.19, "ELSE": 2.35},
            "FE": {"O": 2.13, "N": 2.08, "ELSE": 2.27},
            "CU": {"O": 2.00, "N": 1.99, "ELSE": 2.17},
            "ZN": {"O": 2.14, "N": 2.00, "ELSE": 2.29}
            }
        self.ion_distance_tolerence = 0.5
        self.ion_angle_deviation = 20.0
        self.ion_rms_deviation = 0.5
        self.ion_order_relation = {
            "Octahedral": ["Square Pyrimidal", "Square Planar", "linear"],
            "Square Planar": ["linear"],
            "linear": [],
            "Trigonal Bipyramidal": ["Triganal Planer"],
            "Triganal Planer": [],
            "Square Pyrimidal": ["Square Planar", "linear"],
            "Tetrahedral" :  [],
            "Pentagonal Bipyrimidal": []
            }
        self.ion_order_relation_rms = 1.5
