# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


import numpy as np


def all_pairwise_distances_leq(points, m):
    points = np.asarray(points, dtype=float)
    diffs = points[:, None, :] - points[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    iu = np.triu_indices(len(points), k=1)
    return np.all(d2[iu] <= m * m)