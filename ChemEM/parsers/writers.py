# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


from openmm.app import PDBFile
from rdkit import Chem 

def save_structure_openmm(structure, filename):
    """Saves the structure using OpenMM, forcing it to keep original IDs."""
    print(f"Saving full structure to {filename}...")
    
    with open(filename, 'w') as f:
        PDBFile.writeFile(
            structure.topology, 
            structure.positions, 
            f, 
            keepIds=True  # <--- This is the magic flag!
        )

def save_structure_parmed(structure, filename="refined_complex.pdb"):
    """Saves the ParmEd structure, maintaining original Res IDs and Chains."""
    print(f"Saving full structure to {filename}...")

    structure.save(filename, overwrite=True)
    

def write_to_sdf(mol: Chem.Mol, file_name: str, conf_id: int = 0) -> None:
    """Write one conformer to SDF."""
    w = Chem.SDWriter(file_name)
    w.write(mol, confId=conf_id)
    w.close()