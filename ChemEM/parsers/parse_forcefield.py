# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from typing import List
from openmm import app
from ChemEM.messages import Messages
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


def _dedup_preserve_order(iterable: Iterable) -> List:
    
    seen = []
    for i in iterable:
        if i not in seen:
            seen.append(i)
            
    return seen
    
@dataclass
class AMBER_FF:
    
    default_protein :str = "amber14/protein.ff14SB.xml" 
    default_dna: str = "amber14/DNA.OL15.xml"
    default_rna :str = "amber14/RNA.OL3.xml"
    default_lipid: str = "amber14/lipid17.xml"
    default_glycam: str = "amber14/GLYCAM_06j-1.xml"
    default_explicit: str = "amber14/tip3pfb.xml"
    default_implicit :str = "implicit/gbn2.xml"

    supported_protein: Tuple[str, ...] = (
        "amber14/protein.ff14SB.xml",
        "amber14/protein.ff15ipq.xml",
        "amber19/protein.ff19SB.xml",
        "amber19/protein.ff19ipq.xml",
        "amber14-all.xml")
        
    supported_dna: Tuple[str, ...] = (
        "amber14/DNA.OL15.xml",
        "amber14/DNA.bsc1.xml",
        "amber19/DNA.OL21.xml",
        "amber14-all.xml")
        
    
    supported_rna : Tuple[str, ...] = (
        "amber14/RNA.OL3.xml",
        "amber14-all.xml")
    
    supported_lipid : Tuple[str, ...] = (
        "amber14/lipid17.xml",
        "amber19/lipid21.xml",
        "amber14-all.xml")
        
        
    supported_glycam : Tuple[str, ...] = (
        "amber14/GLYCAM_06j-1.xml")
    
    supported_explicit: Tuple[str, ...] = (
        "amber14/tip3p.xml",
        "amber14/tip3pfb.xml",
        "amber14/tip4pew.xml",
        "amber14/tip4pfb.xml",
        "amber14/spce.xml",
        "amber14/opc.xml",
        "amber14/opc3.xml"
        )
    
    supported_implicit : Tuple[str, ...] = (
        "implicit/hct.xml",
        "implicit/obc1.xml",
        "implicit/obc2.xml",
        "implicit/gbn.xml",
        "implicit/gbn2.xml"
        )
    
    @classmethod
    def get_forcefield(cls, rep, forcefeild = None, request_implicit = True):
        ff = []
    
        if rep.protein_res:
            ff.append(cls.get_supported_forcefeild(forcefeild, cls.supported_protein, cls.default_protein))
        
        if rep.dna:
            ff.append(cls.get_supported_forcefeild(forcefeild, cls.supported_dna, cls.default_dna))
        
        if rep.rna:
            ff.append(cls.get_supported_forcefeild(forcefeild, cls.supported_rna, cls.default_rna))
        
        if rep.glycans:
            ff.append(cls.get_supported_forcefeild(forcefeild, cls.supported_glycam, cls.default_glycam))
        
        if rep.waters:
            ff.append(cls.get_supported_forcefeild(forcefeild, cls.supported_explicit, cls.default_explicit))
        
        if request_implicit or (rep.ions and not rep.waters):
            ff.append(cls.get_supported_forcefeild(forcefeild, cls.supported_implicit, cls.default_implicit))
        
        if not ff:
            raise RuntimeError(Messages.fatal_exception(__file__, "[Error] no forecfields identified"))
        
        ff = _dedup_preserve_order(ff)
        return ff
        
    @staticmethod
    def get_supported_forcefeild(forcefield, supported_ffs, default) -> str:
        
        #check there is only 1 forcefield requested 
        if forcefield is None:
            return default
        
        requested = [i for i in forcefield if i in supported_ffs]
        
        if len(requested) > 1:
            raise RuntimeError(Messages.fatal_exception(__file__, f"[Error] Duplicate component forcefields requested : {requested}"))
        
        elif len(requested) == 1:
            return requested[0]
        else: 
            return default
    

@dataclass
class CHARMM_FF:
    pass

def ff_loads(files: List[str]):
    try:
        ff = app.ForceField(*files)
        return ff
    except Exception as e:
        return None
    
def build_forcefeilds_from_components(rep, 
                                     forcefields = None, 
                                     force_ff = False, 
                                     request_implicit=True):
    
    if force_ff and (forcefields is not None):
        ff = ff_loads(forcefields)
        if ff is None:
            raise RuntimeError(Messages.fatal_exception(__file__, "[ERROR] failed to compile forcefields : {forcefields}"))
        #just make it
    
    force_family = AMBER_FF()
    
    #get force family 
    if forcefields is not None:
        force_family = get_force_family(forcefields)
    
    #get components 
    ff = force_family.get_forcefield(rep, forcefields,request_implicit=request_implicit)
    force = ff_loads(ff)
    
    #build and return
    if force is None:
        raise RuntimeError(Messages.fatal_exception(__file__, f"[ERROR] failed to compile forcefields : {ff}"))
    
    print(f'-- Sucessfully Built Forcefeild from {ff}')
    
    return force
    
    
def get_force_family(forcefields):
    amber_family = ['amber' in i for i in forcefields]
    charmm_family = ['charmm' in i for i in forcefields]
    explict = ['explicit' in i for i in forcefields]
    
    if any(amber_family) and any(charmm_family):
        raise RuntimeError(Messages.fatal_exception(__file__, "[ERROR] Can't mix Charmm and Amber Force families"))
    
    if any(amber_family):
        return AMBER_FF
    
    if any(charmm_family):
        return CHARMM_FF
    
    if all(explict):
        #the user just specifying a change in explicit solvent
        #return te default and change the explict solvent
        return AMBER_FF
    
