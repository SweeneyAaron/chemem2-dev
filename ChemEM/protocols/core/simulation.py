# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>
from openmm.app import NoCutoff, HBonds 
from openmm import unit
import numpy as np




def first_missing_params(struct):
    for b in struct.bonds:
        if b.type is None: return ("bond", [b.atom1, b.atom2])
    for a in struct.angles:
        if a.type is None: return ("angle", [a.atom1, a.atom2, a.atom3])
    for d in struct.dihedrals:
        if d.type is None: return ("dihedral", [d.atom1, d.atom2, d.atom3, d.atom4])
    for i in getattr(struct, "impropers", []):
        if i.type is None: return ("improper", [i.atom1, i.atom2, i.atom3, i.atom4])
    return None

def _is_ligand_object(obj):
    """Duck-type check: Ligand has .complex_structure and .covalent_link."""
    return hasattr(obj, "complex_structure") and hasattr(obj, "covalent_link")


def merge_structures(protein, ligand_structures):
    """Merge protein + ligand parmed Structures via parmed's `+=` operator.

    ligand_structures is a list of parmed Structure objects (not Ligand).
    """
    if not isinstance(ligand_structures, list):
        ligand_structures = [ligand_structures]

    for lig in ligand_structures:
        for i, res in enumerate(lig.residues):
            if res.name == 'UNL':
                res.name = f'LIG_{i}'

    complex_struc = protein
    for lig in ligand_structures:
        complex_struc += lig
    return complex_struc


def finalize_system_from_structure(complex_struc, solvent):
    kwargs = {
        'nonbondedMethod': NoCutoff,
        'nonbondedCutoff': 9.0 * unit.angstrom,
        'constraints': HBonds,
        'removeCMMotion': True,
    }

    if solvent:
        kwargs['implicitSolvent'] = solvent
    else:
        kwargs['rigidWater'] = True

    miss = first_missing_params(complex_struc)
    if miss:
        kind, atoms = miss
        print(f"{kind} missing for:")
        for at in atoms:
            print(f"  {at.residue.name}:{at.name} (idx {at.idx})")
    else:
        print("No missing valence params found.")

    system = complex_struc.createSystem(**kwargs)
    return complex_struc, system


def create_system(protein, ligand_list, solvent):
    """Build a merged OpenMM system from protein + ligands.

    `ligand_list` may be a list of parmed Structures (legacy) OR a list of
    ChemEM Ligand objects. If Ligand objects are passed, any CovalentLinkSpec
    attached to them is applied (junction bond/angle/dihedral terms injected
    into the merged structure and fragment-derived charges copied onto the
    ligand) before createSystem is called.
    """
    if not isinstance(ligand_list, list):
        ligand_list = [ligand_list]

    ligand_objects = None
    if ligand_list and _is_ligand_object(ligand_list[0]):
        ligand_objects = ligand_list
        ligand_structures = [lig.complex_structure for lig in ligand_objects]
    else:
        ligand_structures = ligand_list

    complex_struc = merge_structures(protein, ligand_structures)

    if ligand_objects is not None and any(
        getattr(l, "covalent_link", None) for l in ligand_objects
    ):
        # Lazy import to avoid a circular import at module load time.
        from ChemEM.parsers.covalent_fragment import inject_covalent_bonds
        inject_covalent_bonds(complex_struc, ligand_objects)

    return finalize_system_from_structure(complex_struc, solvent)

def identify_indices_from_name(complex_structure):
        all_ligand_indices = [
            a.idx for a in complex_structure.atoms 
            if a.residue.name.startswith('LIG')
        ]
        ligand_heavy_indices = [
            i for i in all_ligand_indices 
            if complex_structure.atoms[i].element != 1
        ]
        protein_heavy_indices = [
            a.idx for a in complex_structure.atoms
            if not a.residue.name.startswith('LIG') and a.element != 1
        ]
        return all_ligand_indices, ligand_heavy_indices
    

    

def check_pose_forces(
    simulation, 
    ligand_indices, 
    max_force_kj_per_mol_nm: float = 2e5, 
    rms_force_kj_per_mol_nm: float = 5e4
):
    """
    Checks if the forces on the ligand are blowing up.
    Returns (is_bad: bool, reason: str)
    """
    st = simulation.context.getState(getEnergy=True, getForces=True)
    frc = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    
    lig_idx = np.array(ligand_indices, dtype=int)
    lig_f = frc[lig_idx]
    lig_norm = np.linalg.norm(lig_f, axis=1)
    
    fmax = float(lig_norm.max()) if lig_norm.size else 0.0
    frms = float(np.sqrt(np.mean(lig_norm**2))) if lig_norm.size else 0.0

    if (not np.isfinite(fmax)) or (not np.isfinite(frms)):
        return True, f"non-finite force: max={fmax}, rms={frms}"

    # Uncomment if you want strict numerical cutoffs
    # if fmax > max_force_kj_per_mol_nm or frms > rms_force_kj_per_mol_nm:
    #     return True, f"too-large force: max={fmax:.3g}, rms={frms:.3g}"

    return False, "ok"

def run_biased_md_burst(
    simulation,
    ligand_indices,
    intended_lig_nm,
    md_ps: float,
    dt_ps: float = 0.001,
    md_pre_min_iters: int = 0,
    md_temp_K: float = 150.0,
    md_seed: int = 1,
    md_report_ps: float = 0.0,
    pose_index: int = 1
):
    """Runs a short biased MD burst."""
    if md_pre_min_iters > 0:
        simulation.minimizeEnergy(maxIterations=int(md_pre_min_iters))
    
    try:
        simulation.context.setVelocitiesToTemperature(
            md_temp_K * unit.kelvin,
            int(md_seed + (pose_index - 1))
        )
    except TypeError:
        simulation.context.setVelocitiesToTemperature(md_temp_K * unit.kelvin)
    
    nsteps = int(round(float(md_ps) / float(dt_ps)))
    if nsteps < 1: nsteps = 1
    
    if md_report_ps > 0:
        chunk = max(1, int(round(float(md_report_ps) / float(dt_ps))))
        done = 0
        while done < nsteps:
            step_now = min(chunk, nsteps - done)
            simulation.step(step_now)
            done += step_now

            st = simulation.context.getState(getEnergy=True, getPositions=True)
            E = st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

            dA = np.linalg.norm(pos[ligand_indices] - intended_lig_nm, axis=1) * 10.0
            print(
                f"  [pose {pose_index} MD {done*dt_ps:.1f} ps] E={E:.2f} kcal/mol | "
                f"lig rmsΔ={np.sqrt(np.mean(dA**2)):.3f} Å maxΔ={dA.max():.3f} Å"
            )
    else:
        simulation.step(nsteps)
        
def get_atom_mapping(full_structure, local_structure):
    
    full_atom_map = {}
    #Key: (Chain_ID, Res_ID, Atom_Name)
    for residue in full_structure.topology.residues():
        chain_id = residue.chain.id 
        res_id = residue.id 
        for atom in residue.atoms():
            key = (chain_id, res_id, atom.name)
            
            full_atom_map[key] = atom.index
    
    mapping = []
    for residue in local_structure.topology.residues():
        chain_id = residue.chain.id 
        res_id = residue.id 
        for atom in residue.atoms():
            key = (chain_id, res_id, atom.name)
            atom_idx = atom.index
    
            if key in full_atom_map:
                mapping.append((atom_idx, full_atom_map[key]))
            else:
                
                print(f"Warning: Atom {key} in local structure not found in full structure.")
    return mapping
    
    


def update_global_positions(full_structure, local_structure):
    """
    Updates the coordinates of the full structure using the minimized 
    coordinates from the local structure.
    """
    # 1. Get the exact index mapping
    mapping = get_atom_mapping(full_structure, local_structure)
    if not mapping:
        print("Error: No matching atoms found to update.")
        return 0
    
    # 2. Safely extract position arrays and their units
    # Fallback to angstroms if no unit is attached
    pos_unit = getattr(full_structure.positions, 'unit', unit.angstrom)
    
    # Copy to numpy arrays for fast indexing
    full_pos = np.array(full_structure.positions.value_in_unit(pos_unit))
    local_pos = np.array(local_structure.positions.value_in_unit(pos_unit))
    
    # 3. Apply the updates
    for local_idx, full_idx in mapping:
        full_pos[full_idx] = local_pos[local_idx]
        
    # 4. Re-attach units and save back to the global structure
    full_structure.positions = full_pos * pos_unit
    
    return len(mapping)


def update_ligand_positions(local_structure, ligand_objects):
    """
    Extracts minimized ligand coordinates from the local environment structure 
    and updates the original ligand objects.
    
    Parameters:
    -----------
    local_structure : parmed.Structure
        The minimized environment structure (env.complex_structure).
    ligand_objects : list
        The list of original ligand objects that contain the .set_positions() method.
    """
    num_ligands = len(ligand_objects)
    if num_ligands == 0:
        return 0

    # Because ligands were added to the environment last, they are the last N residues.
    # We slice the last N residues. (e.g., if 2 ligands, we get [-2, -1])
    ligand_residues = local_structure.residues[-num_ligands:]
    
    # Safely extract the full local coordinate array in Angstroms
    pos_unit = getattr(local_structure.positions, 'unit', unit.angstrom)
    full_pos_angstrom = np.array(local_structure.positions.value_in_unit(pos_unit))
    
    updated_count = 0
    
    # zip() pairs the first ligand object with the first ligand residue, etc.
    for lig_obj, residue in zip(ligand_objects, ligand_residues):
        
        # Extract the global array indices for the atoms in this specific residue,
        # perfectly preserving the order they appear in the residue object.
        atom_indices = [atom.idx for atom in residue.atoms]
        
        # Slice the numpy array using those indices
        lig_coords_angstrom = full_pos_angstrom[atom_indices]
        
        # Sanity check: Ensure we aren't about to crash the set_positions method
        expected_atoms = len(lig_obj.complex_structure.atoms)
        if len(lig_coords_angstrom) != expected_atoms:
            print(f"Warning: Atom count mismatch for ligand {residue.name}. "
                  f"Expected {expected_atoms}, got {len(lig_coords_angstrom)}.")
            continue
            
        # Update your original object!
        lig_obj.set_positions(lig_coords_angstrom)
        updated_count += 1
        
    return updated_count


