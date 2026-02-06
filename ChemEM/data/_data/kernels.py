# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

import numpy as np

# A kernel for the X derivative
kernel_dx = np.array([[[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]],

                      [[0, 0, 0],
                       [-1, 0, 1],  # Change along the X-axis
                       [0, 0, 0]],

                      [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]]) / 2.0  # Normalize by dividing by 2

# A kernel for the Y derivative
kernel_dy = np.array([[[0, 0, 0],
                       [0,-1, 0],  # Change along the Y-axis
                       [0, 0, 0]],

                      [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]],

                      [[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]]]) / 2.0

# A kernel for the Z derivative
kernel_dz = np.array([[[0,-1, 0],
                       [0, 0, 0],
                       [0, 0, 0]],

                      [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]],

                      [[0, 1, 0],
                       [0, 0, 0],
                       [0, 0, 0]]]) / 2.0
