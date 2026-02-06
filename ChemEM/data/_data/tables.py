# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


import json
from importlib import resources

import numpy as np


def load_json(filename):
    
    with resources.open_text('ChemEM.data.ECHO_dock', filename) as f:
        data = json.load(f)
    return data


TABLE_A = np.array([row for row in load_json('TableA.json')])
TABLE_B = np.array([row for row in load_json('TableB.json')])
TABLE_C = np.array([row for row in load_json('TableC.json')])

HBOND_POLYA = load_json('HBOND_POLY_A.json')
HBOND_POLYB = load_json('HBOND_POLY_B.json')
HBOND_POLYC = load_json('HBOND_POLY_C.json')

HBOND_DONOR_ATOM_IDXS = [13, 15, 19, 23, 24, 28, 37, 40, 42, 43]
#i dont think 14 can accept?

HBOND_ACCEPTOR_ATOM_IDXS = [13, 14, 15, 16, 17, 18, 19,36, 20, 21, 22, 23, 24, 25, 26, 28, 38, 39, 41]
HALOGEN_DONOR_ATOM_IDXS = [0,1,3]
HALOGEN_ACCEPTOR_ATOM_IDXS =  [19, 20, 21, 22, 24, 24, 25, 25, 26, 26, 27, 27,38, 39]
