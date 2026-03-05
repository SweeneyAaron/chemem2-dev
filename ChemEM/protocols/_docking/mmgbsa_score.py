#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 12:54:59 2025

@author: aaron.sweeney
"""


from openmm.app import HBonds, OBC2
from openmm import( unit,
                   CustomNonbondedForce,
                   NonbondedForce, 
                   CustomGBForce, 
                   Context, 
                   VerletIntegrator, 
                   CustomExternalForce, 
                   GBSAOBCForce,
                   )
import parmed as pmd
import mdtraj as md
import numpy as np
import copy 

from dataclasses import dataclass
from typing import Dict, List
from rdkit import Chem
from rdkit.Geometry import Point3D

GAMMA= 0.005 
BETA = 0.0

@dataclass
class PoseScore:
    ligand_name : str
    pose_idx    : int
    deltaG          : float
    components  : Dict[str, float]      # {"ΔEEL": …, "ΔVDW": …, …}

    def to_row(self):
        base = {"ligand": self.ligand_name, "pose": self.pose_idx, "deltaG": self.deltaG}
        base.update(self.components)
        return base
    
    def _line(self):
        return f'{self.ligand_name}, pose {self.pose_idx}: {str(self.components)} \n '


def write_mmgbsa_scores(scores : List[PoseScore], file : str):
    out = ''
    
    for sc in scores:
        out += sc._line()
    with open(file, 'w') as f:
        f.write(out)

def mmgbsa_score_docked_poses(ligand,
                       protein,
                       take_top : int = -1,
                       ) -> List[PoseScore]:
    
    
    if not hasattr(ligand, 'docked'):
        raise AttributeError("Ligand object has no attribute docked")
    
    if take_top == -1:
        #score all docked poses
        take_top = len(ligand.docked)
        

    pose_scores: List[PoseScore] = []
    
    for pose_idx, docked_pose in enumerate(ligand.docked):
        if pose_idx >= take_top:
            break 
        score = score_single_pose(docked_pose.position, 
                                        ligand,
                                        protein, 
                                        pose_idx)
        
        pose_scores.append(score)
    
    ligand.mmgbsa_scores = pose_scores
    return pose_scores
    
def mmgbsa_score_pose(ligand,
               protein,
               
               ) -> List[PoseScore]:
    
    score = score_single_pose(ligand.mol.GetConformer().GetPositions(),
                              ligand, 
                              protein,
                              )
    ligand.mmgbsa_scores = [score]
    return [score]
        
def ligand_traj_to_sdf(traj ,ligand ,outfile):
    lig_atom_idx = traj.topology.select(f"resname {ligand.ligand_id}")
    if not lig_atom_idx.size:
        print("[Warning] ligand traj contains no atoms")
        return 
    
    lig_coords_A = traj.xyz[-1, lig_atom_idx, :] * 10.0 
    rdmol = Chem.Mol(ligand.mol)
    if rdmol.GetNumAtoms() == lig_coords_A.shape[0]:
        conf = Chem.Conformer(rdmol.GetNumAtoms())
        for i, (x, y, z) in enumerate(lig_coords_A):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        rdmol.RemoveAllConformers()
        rdmol.AddConformer(conf, assignId=True)
        
        with Chem.SDWriter(outfile) as w:
            w.write(rdmol)
        
def score_single_pose(positions,
                      ligand,
                      protein,
                      pose_idx: str | None = None ):
    
    
    if pose_idx is None:
        pose_idx = 0

    complex_struct = build_complex(protein.complex_structure,
                                   ligand, 
                                   positions)
    
    traj = parmed_structure_to_single_frame_traj(complex_struct) 
    comps, deltaG = mmgbsa_from_traj(complex_struct, traj)
    
    return PoseScore(ligand_name = ligand.ligand_int,
                     pose_idx    = pose_idx,
                     deltaG      = deltaG,
                     components  = comps)
    
    
def parmed_structure_to_single_frame_traj(struct):
    """
    Convert a ParmEd Structure into a single-frame mdtraj.Trajectory.

    Parameters
    ----------
    struct : parmed.Structure

    Returns
    -------
    mdtraj.Trajectory
        Single-frame trajectory (1, n_atoms, 3)
    """
    

    # ParmEd's positions (Quantity, typically in angstrom)
    pos = getattr(struct, "positions", None)

    if pos is None:
        raise ValueError("ParmEd Structure has no positions set.")
        
    xyz_nm = np.asarray(pos.value_in_unit(unit.nanometer), dtype=float)

    return md.Trajectory(
        xyz=xyz_nm[None, :, :],  # add frame axis
        topology=md.Topology.from_openmm(struct.topology),
    )

def build_complex(protein_struct, ligand, pose_xyz_A):
    lig = copy.deepcopy(ligand.complex_structure)
    for atom, xyz in zip(lig.atoms, pose_xyz_A):
        atom.xx, atom.xy, atom.xz = xyz
    return copy.deepcopy(protein_struct) + lig


def mmgbsa_from_traj(complex_struct, traj, gamma = GAMMA, beta = BETA):
    totals = {"EEL": [], "VDW": [], "EGB": [], "ECAV": []}
    rec_sys, lig_sys, cmp_sys, rec_idx, lig_idx = make_system_triplet(complex_struct)
    
    
    for frame in traj:
        pos_nm = frame.xyz[0] * unit.nanometer
        rec_pos = pos_nm[rec_idx]
        lig_pos = pos_nm[lig_idx]

        dEEL, dVDW, dEGB, dECAV = frame_mmgbsa(
                                    cmp_sys, rec_sys, lig_sys,
                                    pos_nm, rec_pos, lig_pos, frame, gamma, beta)
        
        for k,v in zip(totals, (dEEL,dVDW,dEGB,dECAV)):
            totals[k].append(v)

    avg = {k: float(np.mean(v)) for k,v in totals.items()}
    avg["deltaG"] = sum(avg.values())
    return avg, avg["deltaG"]

def frame_mmgbsa( cmp_sys, rec_sys, lig_sys,
              pos, rec_pos, lig_pos, frame, gamma, beta):

    eel_c, vdw_c, egb_c, ecav_c = compute_frame_energies(cmp_sys, pos,      frame, gamma, beta)
    eel_r, vdw_r, egb_r, ecav_r = compute_frame_energies(rec_sys, rec_pos, frame, gamma, beta)
    eel_l, vdw_l, egb_l, ecav_l = compute_frame_energies(lig_sys, lig_pos, frame, gamma, beta)

    return (eel_c - (eel_r + eel_l),
            vdw_c - (vdw_r + vdw_l),
            egb_c - (egb_r + egb_l),
            ecav_c - (ecav_r + ecav_l))

def make_system_triplet(complex_struct: pmd.Structure):
    """
    Build OpenMM Systems for:
        • complex  (protein + ligand)
        • receptor (protein only)
        • ligand   (ligand only)

    Also caches atom-index masks so we can slice coordinates fast.
    Returns
    -------
    rec_sys, lig_sys, cmp_sys : tuple[openmm.System, openmm.System, openmm.System]
    """

    
    rec_idx, lig_idx = [], []
    for atom in complex_struct.atoms:
        if atom.residue.name.startswith("LIG"):
            lig_idx.append(atom.idx)
        else:
            rec_idx.append(atom.idx)


    receptor_struct = complex_struct[rec_idx]
    ligand_struct   = complex_struct[lig_idx]
    
    common_kwargs = dict(nonbondedCutoff=1.2*unit.nanometers,
                         constraints=HBonds,
                         removeCMMotion=False,
                         implicitSolvent=OBC2)
    
    rec_sys = receptor_struct.createSystem(**common_kwargs)
    lig_sys = ligand_struct.createSystem(**common_kwargs)
    cmp_sys = complex_struct.createSystem(**common_kwargs)
    
    # ----------------------------------------------------------------
    # 4) put forces into groups so MM/GBSA code can pick them apart
    # ----------------------------------------------------------------
    rec_sys = split_nonbonded_terms(rec_sys); rec_sys = modify_system(rec_sys)
    lig_sys = split_nonbonded_terms(lig_sys); lig_sys = modify_system(lig_sys)
    cmp_sys = split_nonbonded_terms(cmp_sys); cmp_sys = modify_system(cmp_sys)

    return rec_sys, lig_sys, cmp_sys, rec_idx, lig_idx



# ---------------------------------------------------------------------
# Utility -------------------------------------------------------------
# ---------------------------------------------------------------------


def compute_frame_energies(system, positions, frame, gamma, beta):
    intg   = VerletIntegrator(0.002*unit.picoseconds)
    ctx    = Context(system, intg)
    ctx.setPositions(positions)

    EEL   = ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy()
    VDW   = ctx.getState(getEnergy=True, groups={2}).getPotentialEnergy()
    EGB   = ctx.getState(getEnergy=True, groups={3}).getPotentialEnergy()

    # SASA in Å² → nm² for mdtraj
    sasa  = md.shrake_rupley(frame)[0].sum() * 100.0          # Å²
    ECAV  = gamma * sasa + beta                               # kcal/mol

    # Convert to kcal/mol
    conv  = unit.kilocalories_per_mole
    return (EEL.value_in_unit(conv),
            VDW.value_in_unit(conv),
            EGB.value_in_unit(conv),
            ECAV)

def split_nonbonded_terms(system):
    """
    Replace the default NonbondedForce with two forces:
      • group 1 → pure Coulomb  (EEL)
      • group 2 → pure LJ       (VDW)
    GB / PB (group 3) remains untouched.
    """
    nb      = None
    nb_i    = None
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, NonbondedForce):
            nb, nb_i = f, i
            break
    assert nb is not None, "System has no NonbondedForce"

    # ------------------------------
    # 1. Coulomb-only NonbondedForce
    # ------------------------------
    coul_force = NonbondedForce()
    coul_force.setCutoffDistance(nb.getCutoffDistance())
    coul_force.setNonbondedMethod(nb.getNonbondedMethod())
    coul_force.setReactionFieldDielectric(nb.getReactionFieldDielectric())
    coul_force.setEwaldErrorTolerance(nb.getEwaldErrorTolerance())
    coul_force.setUseDispersionCorrection(False)
    zero_len  = 0.0 * unit.nanometer
    zero_eps  = 0.0 * unit.kilojoule_per_mole
    
    
    # ------------------------------
    # 2. Pure-LJ CustomNonbondedForce
    # ------------------------------
    lj_force = CustomNonbondedForce("4*epsilon*((sigma/r)^12-(sigma/r)^6);"
                                    "sigma = 0.5*(sigma1+sigma2);"
                                    "epsilon = sqrt(epsilon1*epsilon2)")
    lj_force.addPerParticleParameter("sigma")
    lj_force.addPerParticleParameter("epsilon")
    lj_force.setCutoffDistance(nb.getCutoffDistance())
    lj_force.setNonbondedMethod(nb.getNonbondedMethod())

    # Add particles to both forces
    for idx in range(nb.getNumParticles()):
        charge, sigma, eps = nb.getParticleParameters(idx)
        # Coulomb-only: keep charge, zero ε
        #coul_force.addParticle(charge, unit.Quantity(0.0, eps.unit), unit.Quantity(0.0, eps.unit))
        coul_force.addParticle(charge, sigma, zero_eps) 
        
        # LJ-only   : zero charge, keep σ,ε
        lj_force.addParticle([sigma, eps])

    # Copy exclusions / 1-4s
    for i in range(nb.getNumExceptions()):
        i1, i2, q, sig, eps = nb.getExceptionParameters(i)
        #coul_force.addException(i1, i2, q, unit.Quantity(0.0, eps.unit), unit.Quantity(0.0, eps.unit))
        coul_force.addException(i1, i2, q, sig, zero_eps) 
        lj_force.addExclusion(i1, i2)

    # Remove old NonbondedForce and attach new ones
    system.removeForce(nb_i)
    coul_force.setForceGroup(1)   # EEL
    lj_force.setForceGroup(2)     # VDW
    system.addForce(coul_force)
    system.addForce(lj_force)

    return system

def modify_system(system):
    for f in system.getForces():
        if isinstance(f, CustomGBForce) or isinstance(f, GBSAOBCForce):
            f.setForceGroup(3)  # GB
    return system

def _modify_system( system):
       """
       Assign energy groups to forces for MMGBSA calculations.

       Parameters:
           system: OpenMM System object.

       Returns:
           system: Modified OpenMM System object with assigned force groups.
       """
       for force in system.getForces():
           if isinstance(force, NonbondedForce):
               force.setForceGroup(1)  # Electrostatics + VDW
           elif isinstance(force, CustomGBForce):
               print('DEBUG: has GB Force')
               force.setForceGroup(3)  # GB interactions
           elif isinstance(force, CustomExternalForce):
               force.setForceGroup(4)  # Custom map-based forces
           else:
               force.setForceGroup(0)  # Default group for other forces
       return system


def poses_to_trajectory(protein_structure: pmd.Structure,
                        ligand_template: pmd.Structure,
                        poses_angstrom: np.ndarray) -> md.Trajectory:
    """
    Build mdtraj.Trajectory from an array of docked ligand poses.

    Parameters
    ----------
    protein_structure : parmed.Structure
        Apo protein with coordinates (Å) that will stay fixed.
    ligand_template   : parmed.Structure
        Ligand topology *template* (coordinates will be overwritten).
    poses_angstrom    : np.ndarray, shape (n_poses, n_atoms, 3)
        Docked ligand coordinates in Å.

    Returns
    -------
    traj : mdtraj.Trajectory
        All protein–ligand complexes as a single trajectory (units: nm).
    """
    complexes = []

    # ---- build a complex structure for every pose ------------------
    for pose_xyz in poses_angstrom:                             # loop over poses
        lig = copy.deepcopy(ligand_template)                 # keep template intact

        # overwrite ligand coordinates (ParmEd stores Å)
        for atom, xyz in zip(lig.atoms, pose_xyz):
            atom.xx, atom.xy, atom.xz = xyz                     # already in Å

        # concatenate protein + ligand -> one complex structure
        complexes.append(protein_structure + lig)

    # ---- convert complex list -> mdtraj ---------------------------
    md_topology = md.Topology.from_openmm(complexes[0].topology)

    # stack (n_poses, n_atoms, 3) array in **nanometres** for MDTraj
    xyz_nm = np.stack([
        np.array([[a.xx, a.xy, a.xz] for a in c.atoms]) * 0.1   # Å → nm
        for c in complexes
    ])

    return md.Trajectory(xyz_nm, md_topology)

    
