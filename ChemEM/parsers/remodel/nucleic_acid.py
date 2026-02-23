# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

 

_RNA_RESNAMES = {"A", "C", "G", "U", "I"}
_DNA_RESNAMES = {"DA", "DC", "DG", "DT", "DU", "DI"}
_NA_CANON = _RNA_RESNAMES | _DNA_RESNAMES

def norm_atom_name(name: str) -> str:
    # normalize star/prime variants (e.g. O3* vs O3')
    return name.replace("*", "'").strip()

def trim_na_5prime_phosphates( modeller) -> int:
    """
    Remove P/OP1/OP2/OP3 (and O1P/O2P) from the FIRST residue of each NA chain.
    """
    to_delete = []
    for ch in modeller.topology.chains():
        first = None
        for r in ch.residues():
            first = r
            break
        if first is None:
            continue
        if first.name not in _NA_CANON:
            continue
        for a in first.atoms():
            n = norm_atom_name(a.name)
            if n in ("P", "OP1", "OP2", "OP3", "O1P", "O2P"):
                to_delete.append(a)
    if to_delete:
        
        modeller.delete(to_delete)
    return len(to_delete)

