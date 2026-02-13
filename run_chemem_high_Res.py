#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 15:03:09 2025

@author: aaron.sweeney
"""

import os 

('6x40', [2.7747796082366105, 'Ligand_0.sdf'])
('6tte', [5.7174861838923565, 'Ligand_14.sdf'])
('6udp', [3.8911447214769885, 'Ligand_16.sdf'])
('6peq', [4.343990175411781, 'Ligand_37.sdf'])
('6tti', [6.4684494391564265, 'Ligand_9.sdf'])
('7cfm', [4.77232971441658, 'Ligand_27.sdf'])
('6x3x', [5.159704140791408, 'Ligand_2.sdf'])


conf_out = '/Users/aaron.sweeney/Documents/ChemEM2/docking/confs_mi'

confs = [i for i in os.listdir(conf_out) if i.endswith('.conf')]

not_done = []

#'7c7q and 6vfx are problems'


#not_done = ['6x40', '6tte', '6udp', '6ttq', '6a95', '6uz8', '6uqe', '6kpf', '6tti', '7jjo', '6vfx', '6nyy', '7cfm']
not_done = ['7jjo']
flexible_rings= ['6tte', '6vfx']
alpha_mask = ['6oo3', '6nyy']


do = ['6nyy', '6oo3']
for c in confs:
    
    if c[0:4] not in do:
        continue
    
    #if c[0:4] not in not_done:
    #    continue
    
    print('PDB', '-' * 60, c)
    conf_path = os.path.join(conf_out, c)
    #com = f'python -m ChemEM {conf_path}  --dock --flexible-rings'
    
    
    if c[0:4] in flexible_rings:
        com = f'python -m ChemEM {conf_path}  --dock --flexible-rings --minimize-docking --segment-binding-sites --energy-cutoff 3.0 --cluster-docking 1.0'
    elif c[0:4] in alpha_mask:
        com = f'python -m ChemEM {conf_path}  --dock --bias-radius 9.0  --minimize-docking  --no-otsu-filter-ses-mask --energy-cutoff 3.0 --cluster-docking 1.0'
    else:
        com = f'python -m ChemEM {conf_path}  --dock --minimize-docking --segment-binding-sites --energy-cutoff 3.0 --cluster-docking 1.0'
    
    try:
        os.system(com)
    except Exception as e:
        not_done.append((c,e))

