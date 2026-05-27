# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>
from ChemEM.messages import Messages
from ChemEM.tools.resources import default_cpu_budget
import sys 
import os 

class System:
    '''
    A protein ligand docking/MD system.
    This is passed to a procol for and contains all the data
    to run a protocol. 
    
    Protocol objects modify the state of the system to return results.
    
    '''
    def __init__(self, 
                 protein = None,
                 ligand = None,
                 centroid = None,
                 maps = None):
        
        self.protein = protein
        self.ligand = ligand
        self.centroid = centroid
        self.density_map = maps
        self.protocols = []
        #protocol flags
        self.docked = False 
        
        #running options
        # CPUS_PER_SITE is resolved at protocol-run time by
        # ChemEM.tools.resources.resolve_cpus_per_site(), which honours an
        # explicit override here or on system.options.cpus_per_site and
        # otherwise picks max(2, total_cpus // 4). Leave as None so the
        # resolver applies the heuristic on small machines instead of pinning
        # to a value that collapses split-site parallelism.
        self.CPUS_PER_SITE = None
        self.ncpu = default_cpu_budget()
        self.n_cpu = self.ncpu
        self.n_cpus = self.ncpu
        self._log = ''
   
    def run(self):
        for protocol in self.protocols:
            
            try:
                protocol.run()
            except Exception as e:
                self.log(Messages.fatal_exception(protocol.__class__, e))
                sys.exit()
    
    def log(self, string):
        print(string)
        self._log += string  + '\n'
    
    def write_log(self):
        output = getattr(self, 'output', '.')
        file = os.path.join(output, 'log.out')
        with open(file, 'w') as f:
            f.write(self._log)
            
    def add_protocol(self, protocol):
        self.protocols.append(protocol)
    
    def run_protocol(self, protocol):
         protocol(self).run()
