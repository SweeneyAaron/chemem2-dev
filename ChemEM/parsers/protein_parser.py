# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>
import os
from .models import Protein
from .components import Components
from .parse_forcefield import  build_forcefeilds_from_components
from .remodel.pipeline import remodel_from_fixer
from typing import List
from ChemEM.messages import Messages
import parmed
from openmm import app
from .remodel.topology_ops import ensure_water_geometry_types
from .interchange import modeller_to_parmed
from .mapping import build_residue_map_by_positions


class ProteinParser:
    
    # -------------------------
    # Main entry point
    # -------------------------
    @staticmethod
    def load_protein_structure(protein_file: str, 
                               forcefield: List[str], 
                               request_implicit:bool = True,
                               force_ff: bool = False,
                               to_parmed : bool = True,
                               map_tol : float = 1e-3) -> Protein:
        '''
        Factory for returning a Protein model from a .pdb | .mmcif file
        Parameters
        ----------
        protein_file : str 
            protein .pdb | .mmcif file path
        forcefield : List[str]
            Explicit Forcefield parameters to set for biomoleculs with openMM
            Valid terms: 
                --
            
        prefer_water : str, optional
            The prefered water model to use if the structure contains explicit water.
            The default is "amber14/tip3p.xml".

        Returns
        -------
        Protein
            A ChemEM Protein object representing the structural data needed to run ChemEM.

        '''
        
        if not os.path.exists(protein_file):
            raise RuntimeError(Messages.fatal_exception('ProteinParser', f'[ERROR] input file not found: {protein_file}'))
            
        
        #---- idnetify included components
        pdb = app.PDBFile(protein_file)
        #needed for mapping
        original_topology = pdb.topology
        original_positions = pdb.positions
        
        comp_report = Components.scan_components(pdb.topology)
        comp_report.print_component_report()
        ff = build_forcefeilds_from_components(comp_report,
                                               forcefield, 
                                               force_ff = force_ff,
                                               request_implicit=request_implicit)
        
        modeller = remodel_from_fixer(protein_file, ff, split_chains = True)
        
        #------conversions
        if to_parmed:
            modeller, system = modeller_to_parmed(modeller, ff)
            if comp_report.has_waters:
                #This is nessicery due to a bug in paramed v4.2.2
                patched = ensure_water_geometry_types(modeller,water_model="tip3p")
                print(f"[ParmEd] Patched water geometry on {patched} residues.")
        
        else:
            system = ff.createSystem(modeller.topology)
        
        
        #------mapping 
        residue_map = build_residue_map_by_positions(
            original_topology,
            original_positions,
            modeller.topology,
            modeller.positions,
            tol_ang=map_tol,
        )

        
        return Protein(
            protein_file,
            system,
            modeller,
            ff,
            residue_map=residue_map,
        )
        #return modeller, residue_map
        
        
       