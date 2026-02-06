# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


SYSTEM_ATTRS = ['centroid', 'segment_centroid', 'exclude', 'segment_dimensions', 'output', 'auto_split_radius',
                'ligand_id','difference_map_id', 'full_map_id','protein_id',
                'mi_weight', 'global_k', 'docking_radius', 'multiligand','platform', 'cutoff',
                'flexible_side_chains', 'solvent', 'n_cpu', 'label_threshold_sigma', 'label_threshold',
                'n_ants', 'theta',  'rho', 'sigma' , 'max_iterations',
                'post_process_num_solutions', 'post_process_solution', 'refine_side_chains',
                'cycles', 'start_temp', 'norm_temp', 'top_temp', 'temperature_step',
                'pressure', 'barostatInterval', 'initial_heating_interval',
                'heating_interval', 'steps', 'pdb_gt_id' , 'generate_diverse_solutions', 'hold_fragment', 'overwrite']

RESIDUE_NAMES = ['CYS','MET','GLY','ASP','ALA','VAL','PRO','PHE','ASN','THR',
                 'HIS','GLN','ARG','TRP','ILE','SER','LYS','LEU','GLU','TYR']

PROTEIN_RINGS = {"TRP": [("Benzene", ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2")), ("Pyrrole",("CG", "CD1", "NE1", "CD2", "CE2") )],
                 "PHE": [("Benzene", ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"))],
                 "TYR": [("Benzene", ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"))],
                 "HIS": [("Imidazole", ("CG", "CD2", "ND1", "CE1", "NE2"))]}

PROTEIN_DONOR_ATOM_IDS = {
    # Non-polar, aliphatic
    'GLY': {'N'},
    'ALA': {'N'},
    'VAL': {'N'},
    'ILE': {'N'},
    'LEU': {'N'},
    'MET': {'N'},

    # Proline is an exception: its backbone N lacks a hydrogen
    'PRO': set(),

    # Aromatic
    'PHE': {'N'},
    'TRP': {'N'},#needs to be fixed with atom typing!! #'NE1'},
    'TYR': {'N', 'OH'},

    # Polar, uncharged
    'ASN': {'N', 'ND2'},
    'CYS': {'N', 'SG'},
    'GLN': {'N', 'NE2'},
    'SER': {'N', 'OG'},
    'THR': {'N', 'OG1'},

    # Negatively charged (acidic)
    'ASP': {'N'},
    'GLU': {'N'},

    # Positively charged (basic)
    'ARG': {'N', 'NE', 'NH1', 'NH2'},
    'HIS': {'N', 'ND1'}, # needs to be fixed!! 'ND1', 'NE2'},  # Depending on protonation state
    'LYS': {'N', 'NZ'},
}

PROTEIN_ACCEPTOR_ATOM_IDS = {
    # Non-polar, aliphatic
    'GLY': {'O'},
    'ALA': {'O'},
    'VAL': {'O'},
    'ILE': {'O'},
    'LEU': {'O'},
    'MET': {'O', 'SD'}, # SD is a weak acceptor

    # Proline's backbone oxygen is a standard acceptor
    'PRO': {'O'},

    # Aromatic
    'PHE': {'O'},
    'TRP': {'O'},
    'TYR': {'O', 'OH'},

    # Polar, uncharged
    'ASN': {'O', 'OD1'},
    'CYS': {'O'},          # SG is a very weak acceptor, often ignored
    'GLN': {'O', 'OE1'},
    'SER': {'O', 'OG'},
    'THR': {'O', 'OG1'},

    # Negatively charged (acidic)
    'ASP': {'O', 'OD1', 'OD2'},
    'GLU': {'O', 'OE1', 'OE2'},

    # Positively charged (basic)
    'ARG': {'O'},
    'HIS': {'O', 'NE2'},  # Depending on protonation state
    'LYS': {'O'},
}

def is_protein_atom_donor(residue_code, atom_id):
    return atom_id in PROTEIN_DONOR_ATOM_IDS.get(residue_code, set())

def is_protein_atom_acceptor(residue_code, atom_id):
    return atom_id in PROTEIN_ACCEPTOR_ATOM_IDS.get(residue_code, set())
