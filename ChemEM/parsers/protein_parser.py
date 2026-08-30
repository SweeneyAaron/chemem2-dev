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
                               map_tol : float = 1e-5,
                               prep = None,
                               cache = None) -> Protein:
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
        
        # ---- prepare (repair + protonate), reusing a cached result when possible.
        # This is the only expensive, previously non-deterministic step; everything
        # after it is a pure function of the prepared coordinates.
        cache_key = None
        modeller = None
        if cache is not None:
            cache_key = cache.key(
                protein_file,
                forcefield_files=getattr(ff, "_chemem_ff_files", None) or list(forcefield or []),
                request_implicit=request_implicit,
                force_ff=force_ff,
                prep=prep,
            )
            modeller = cache.load(cache_key)

        cache_hit = modeller is not None
        if not cache_hit:
            modeller = remodel_from_fixer(protein_file, ff, split_chains=True, prep=prep)

        # `modeller` is rebound to a ParmEd Structure below; keep the prepared
        # topology/positions so the cache stores the thing it can reload.
        prepared = modeller

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

        # The residue map matches original<->prepared backbone atoms to 1e-5 A, so
        # it is the canary for any coordinate corruption: a cache that perturbed
        # coordinates would return an EMPTY map with no error, silently breaking
        # get_residue_mapping, --manual-site and covalent specs. Refuse to trust a
        # cache entry that maps fewer residues than the cold path did.
        n_mapped = len(residue_map)
        if cache is not None:
            if cache_hit:
                expected = cache.expected_mapped(cache_key)
                if expected is not None and n_mapped != expected:
                    print(f"[prep] WARNING: cached protein mapped {n_mapped} residues, "
                          f"expected {expected}; discarding cache entry and re-preparing.")
                    cache.invalidate(cache_key)
                    return ProteinParser.load_protein_structure(
                        protein_file, forcefield,
                        request_implicit=request_implicit, force_ff=force_ff,
                        to_parmed=to_parmed, map_tol=map_tol, prep=prep, cache=None,
                    )
            else:
                cache.store(cache_key, prepared, n_mapped=n_mapped)

        if n_mapped == 0 and len(list(original_topology.residues())):
            print("[prep] WARNING: residue map is empty -- get_residue_mapping, "
                  "--manual-site and covalent atom specs will not resolve.")

        return Protein(
            protein_file,
            system,
            modeller,
            ff,
            residue_map=residue_map,
        )
        #return modeller, residue_map
        
        
       