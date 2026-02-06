#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 12:54:59 2025

@author: aaron.sweeney
"""
import warnings
from openmm.app import ForceField as OpenMMForceField
from openmm.app import (Modeller, 
                        HBonds, 
                        NoCutoff, 
                        PME, 
                        OBC2, 
                        GBn2, 
                        Simulation, 
                        StateDataReporter,
                        PDBReporter,
                        PDBFile)
from openmm import( unit,
                   LangevinIntegrator,
                   Platform, 
                   CustomCompoundBondForce,
                   CustomNonbondedForce,
                   MonteCarloBarostat, 
                   NonbondedForce, 
                   CustomGBForce, 
                   Context, 
                   VerletIntegrator, 
                   CustomExternalForce, 
                   GBSAOBCForce,
                   OpenMMException)
import parmed as pmd
import mdtraj as md
import numpy as np
import copy 
import time
from dataclasses import dataclass, asdict
from typing import Dict, List
import json 
import os
from rdkit import Chem
from rdkit.Geometry import Point3D


def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


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


class MMGBSAScore:
    def __init__(self, 
                 output,
                 protein,
                 ligand,
                 platform,
                 take_top = -1,
                 minimize = False,
                 post_minimize = False,
                 pin_atoms = "binding_site",
                 binding_site_radius = 0.6,
                 pin_k      = 500 ,
                 md_dt      = 2.0 ,
                 md_ns      = 0,
                 md_temp    = 300 ,
                 md_save    = 10.0
                 
                 ):
        self.output = output 
        self.ligand = ligand
        self.protein = protein 
        self.platform = platform 
        self.traj_file = None 
        self.gamma = 0.005 # kcal/mol/Å²   (non-polar surface tension term)
        self.beta = 0.0 # kcal/mol      (non-polar offset)
        
        #-----traj data------
        self._trajectories = []
        self._receptor_indices = []
        self._ligand_indices = []
        self.results = []
        
        self._protein_struct = None 
        self._lig_templates = None
        
        
        self.take_top = take_top 
        self._do_min = minimize
        self._post_min = post_minimize 
        self._pin_atoms = pin_atoms
        self._binding_site_radius = binding_site_radius * unit.nanometer
        self._pin_k = pin_k * unit.kilocalories_per_mole/unit.angstrom**2
        self._md_ns = md_ns #nano seconds 
        self._md_dt = md_dt * unit.femtoseconds #fs 
        self._md_temp = md_temp * unit.kelvin
        self._md_save  = md_save
        
        #-----MD parameters-----
        
        
        
        
        
    def run(self):
        self.output = os.path.join(self.output, 'mmgbsa_rescore')
        try:
            os.mkdir(self.output)
        except FileExistsError:
            pass
       
        self._prepare_complex_templates()
        
        self._score_all_poses()
        
        
    def _score_all_poses(self):
        
        out_file = 'MMGBSA_scores:\n'
        
        
        for lig_num, lig in enumerate(self.ligand):
            
            _pose_scores = []
            
            output = os.path.join(self.output, f'Ligand_{lig_num}')
            mkdir(output)
            
            if lig.docked:
            
                for pidx, docked_pose in enumerate(lig.docked):
                    
                    if self.take_top == -1 or (self.pidx < self.take_top):
                        
                        traj, score = self._score_single_pose(lig_num, pidx, docked_pose.position)
                        
                        traj_id = f'Ligand_{lig_num}_pose_{pidx}'
                        
                        file = os.path.join(output, f'{traj_id}.pdb')
                        traj.save_pdb(file)
                        score_file = os.path.join(output, f'{traj_id}_score.json')
                        
                        with open(score_file, 'w') as f:
                            json.dump(score.components, f)
                        
                        #HERERERE
                        lig_atom_idx = traj.topology.select(f"resname LIG{lig_num}")
                        if lig_atom_idx.size:
                            lig_coords_A = traj.xyz[-1, lig_atom_idx, :] * 10.0 
                            rdmol = Chem.Mol(lig.mol)
                            if rdmol.GetNumAtoms() == lig_coords_A.shape[0]:
                                conf = Chem.Conformer(rdmol.GetNumAtoms())
                                for i, (x, y, z) in enumerate(lig_coords_A):
                                    conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
                                rdmol.RemoveAllConformers()
                                rdmol.AddConformer(conf, assignId=True)
                                sdf_path = os.path.join(output, f"Ligand_{lig_num}_pose_{pidx}.sdf")
                                with Chem.SDWriter(sdf_path) as w:
                                    w.write(rdmol)
                        
                        #I need to write the ligand to an sdf file from the traj!!!
                        out_file += f'Ligand {lig_num}, pose {pidx}:'
                        out_file += str(score.components) 
                        out_file += '\n'
                        
                        _pose_scores.append(score)
            
                
                lig.mmgbsa_scores = _pose_scores
                
                
                out_file += '\n'
            
            #start here with loaded ligands!!
            else:
                pidx = 0
                
                traj, score = self._score_single_pose(lig_num, 
                                                      pidx,
                                                      lig.mol.GetConformer().GetPositions())
                
                traj_id = f'Ligand_{lig_num}_pose_{pidx}'
                file = os.path.join(self.output, f'{traj_id}.pdb')
                traj.save_pdb(file)
                score_file = os.path.join(self.output, f'{traj_id}_score.json')
                with open(score_file, 'w') as f:
                    json.dump(score.components, f)
                
                out_file += f'Ligand {lig_num}, pose {pidx}:'
                out_file += str(score.components) 
                out_file += '\n'
                
                _pose_scores.append(score)
    
        
                lig.mmgbsa_scores = _pose_scores
        
        
                out_file += '\n'
                
            
        outfn = os.path.join(self.output, f'mmgbsa_scores.txt')
        with open(outfn, 'w') as f:
            f.write(out_file)
        
        
    
    def _prepare_complex_templates(self):
        self._protein_struct = self.protein.complex_structure
        self._lig_templates  = {
            num : lig.complex_structure for num,lig in enumerate(self.ligand)
        }
    
    def _score_single_pose(self, lig_num, pose_idx,
                           pose_xyz_A):
        
        
        complex_struct = self._build_complex(lig_num, pose_xyz_A)
        
        traj           = self._refine(complex_struct)        # min+optional MD
        
        comps, deltaG      = self._mmgbsa_average(complex_struct, traj)
        
        return traj, PoseScore(ligand_name = lig_num,
                         pose_idx    = pose_idx,
                         deltaG          = deltaG,
                         components  = comps)
    
    def _build_complex(self, lig_num, pose_xyz_A):
        lig = copy.deepcopy(self._lig_templates[lig_num])
        for atom, xyz in zip(lig.atoms, pose_xyz_A):
            atom.xx, atom.xy, atom.xz = xyz
        return copy.deepcopy(self._protein_struct) + lig
    
    def _refine(self, complex_struct):
        """
        1. createSystem  2. optional pin restraints  3. minimise
        4. optional MD   5. return mdtraj.Trajectory (≥1 frame, even on failure)
        """
    
        # ---------------------------------------------------------------------
        #  Helpers
        # ---------------------------------------------------------------------
        
        def _add_frame(kind: str, tag: str = "") -> bool:
            """
            Record current coordinates and their 'kind' (one of: 'initial','min','md','post-min').
            Returns True if accepted, False if skipped due to NaNs.
            """
            state = sim.context.getState(getPositions=True)
            pos   = state.getPositions(asNumpy=True)
        
            # Detect NaNs *before* we stash the frame
            if np.isnan(pos.value_in_unit(unit.nanometer)).any():
                warnings.warn(f"Skipped frame '{tag or kind}' because coordinates contain NaN.")
                return False
        
            frames.append(pos)
            kinds.append(kind)
            if tag:
                print(f"[{tag}] recorded frame {len(frames)}")
            return True
        
        def _select_indices(kinds: list[str]) -> list[int]:
            # Primary policy
            if self._md_ns > 0.0:
                if self._post_min:
                    idx = [i for i, k in enumerate(kinds) if k == "post-min"]
                    if idx:
                        return idx
                    # fallback: final MD frame if post-min failed to record
                    md_idx = [i for i, k in enumerate(kinds) if k == "md"]
                    if md_idx:
                        return [md_idx[-1]]
                else:
                    md_idx = [i for i, k in enumerate(kinds) if k == "md"]
                    if md_idx:
                        return md_idx
        
            if self._do_min:
                min_idx = [i for i, k in enumerate(kinds) if k == "min"]
                if min_idx:
                    return min_idx
        
            init_idx = [i for i, k in enumerate(kinds) if k == "initial"]
            if init_idx:
                return init_idx
        
            # Absolute fallback: keep whatever we have
            return list(range(len(kinds)))
            
        # ---------------------------------------------------------------------
        #  System construction
        # ---------------------------------------------------------------------
        system = complex_struct.createSystem(
            nonbondedCutoff=1.2 * unit.nanometers,
            constraints=HBonds,
            removeCMMotion=False,
            implicitSolvent=OBC2,
        )
    
        # split forces for energy groups
        #system = split_nonbonded_terms(system)
        #system = modify_system(system)
    
        # ---------------------------------------------------------------------
        #  Optional positional restraints
        # ---------------------------------------------------------------------
        pin_idx = []
        if self._pin_atoms == "protein_backbone":
            pin_idx = [
                a.idx for a in complex_struct.atoms
                if (not a.residue.name.startswith('LIG')) and (a.name in ("N", "CA", "C", "O"))
            ]
    
        elif self._pin_atoms == "binding_site":
            md_top = md.Topology.from_openmm(complex_struct.topology)
            xyz_nm = np.array([[a.xx, a.xy, a.xz] for a in complex_struct.atoms]) * 0.1
            temp_traj = md.Trajectory(xyz_nm, md_top)
    
            ligand_indices          = temp_traj.topology.select('resname LIG')
            protein_indices         = temp_traj.topology.select('protein')
            protein_backbone_indices = temp_traj.topology.select('protein and backbone')
            protein_sidechain_indices = temp_traj.topology.select('protein and sidechain')
    
            nearby_protein_atoms = md.compute_neighbors(
                temp_traj,
                self._binding_site_radius.value_in_unit(unit.nanometer),
                query_indices=ligand_indices,
                haystack_indices=protein_indices,
            )
            nearby_atom_indices   = {idx for pair in nearby_protein_atoms for idx in pair}
            nearby_residue_indices = {
                temp_traj.topology.atom(i).residue.index for i in nearby_atom_indices
            }
    
            mobile_sidechain_indices = [
                idx for idx in protein_sidechain_indices
                if temp_traj.topology.atom(idx).residue.index in nearby_residue_indices
            ]
    
            pinned_backbone = set(protein_backbone_indices)
            pinned_sidechain = set(protein_sidechain_indices) - set(mobile_sidechain_indices)
            pin_idx = list(pinned_backbone | pinned_sidechain)
        
        #add secondary structure
        
        elif isinstance(self._pin_atoms, (list, tuple)):
            pin_idx = list(self._pin_atoms)
    
        if pin_idx:
            k = self._pin_k  # kcal/mol/Å²
            rest = CustomExternalForce(
                "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
            )
            rest.addGlobalParameter("k", k)
            rest.addPerParticleParameter("x0")
            rest.addPerParticleParameter("y0")
            rest.addPerParticleParameter("z0")
            for idx in pin_idx:
                pos = complex_struct.atoms[idx]
                rest.addParticle(idx, [pos.xx * 0.1, pos.xy * 0.1, pos.xz * 0.1])  # Å→nm
            rest.setForceGroup(5)  # separate force group
            system.addForce(rest)
    
        # ---------------------------------------------------------------------
        #  Simulation object
        # ---------------------------------------------------------------------
        platform   = Platform.getPlatformByName(self.platform)
        integrator = LangevinIntegrator(self._md_temp, 1.0 / unit.picosecond, self._md_dt)
        sim        = Simulation(complex_struct.topology, system, integrator, platform)
    
        start_pos = np.array([[a.xx, a.xy, a.xz] for a in complex_struct.atoms]) * 0.1  # Å→nm
        sim.context.setPositions(start_pos * unit.nanometer)
    
        frames = []
        kinds = []
        _add_frame("initial", "initial")
    
        # ---------------------------------------------------------------------
        #  Energy minimisation
        # ---------------------------------------------------------------------
        if self._do_min:
            print("Minimising …")
            t0 = time.perf_counter()
            try:
                sim.minimizeEnergy(
                    tolerance=100 * unit.kilojoule_per_mole / unit.nanometer,
                    maxIterations=500,
                )
            except OpenMMException as e:
                warnings.warn(f"Minimisation failed: {e}.  Continuing with last stable coordinates.")
            finally:
                
                _add_frame("min", "min")
                print(f"… done in {time.perf_counter() - t0:.1f}s\n" + "-" * 50)
    
        # ---------------------------------------------------------------------
        #  MD sampling
        # ---------------------------------------------------------------------
        if self._md_ns > 0.0:
            print("MD sampling …")
            steps_per_ps = int(unit.picoseconds / self._md_dt)
            total_steps  = int(self._md_ns * 1000 * steps_per_ps)
            stride_steps = int(self._md_save * steps_per_ps)
    
            completed = 0
            try:
                for step in range(0, total_steps, stride_steps):
                    sim.step(stride_steps)
                    completed = step + stride_steps
                    if not self._post_min:
                        _add_frame(f"MD {completed}/{total_steps}",  f"MD {completed}/{total_steps}")
            except OpenMMException as e:
                warnings.warn(
                    f"MD integration failed after {completed} steps: {e}."
                    " Returning trajectory up to this point."
                )
    
            # Final post-minimisation for post-min mode
            if self._post_min:
                print("Post-minimisation …")
                try:
                    sim.minimizeEnergy(
                        tolerance=100 * unit.kilojoule_per_mole / unit.nanometer,
                        maxIterations=2000,
                    )
                except OpenMMException as e:
                    warnings.warn(f"Post-minimisation failed: {e}.  Using last MD frame.")
                finally:
                    _add_frame("post-min", "post-min")
    
        # ---------------------------------------------------------------------
        #  Build trajectory
        # ---------------------------------------------------------------------
        if not frames:                        # <-- fallback instead of raising
            warnings.warn(
                "No simulation frames were recorded; "
                "returning the original input structure."
            )
            xyz0_nm = np.array(
                [[a.xx, a.xy, a.xz] for a in complex_struct.atoms]
            ) * 0.1                           # Å → nm
            traj = md.Trajectory(
                xyz0_nm[None, ...],           # add a frame dimension (shape: 1 × N × 3)
                md.Topology.from_openmm(complex_struct.topology)
            )
            return traj
        
        selected = _select_indices(kinds)
        if not selected:
            print('[Warning] No frames selected')
            selected = list(range(len(kinds)))
        
        # 4) Build trajectory from selected frames
        xyz_nm = np.array([
            [vec.value_in_unit(unit.nanometer) for vec in frames[i]]
            for i in selected
        ])
        traj = md.Trajectory(xyz_nm, md.Topology.from_openmm(complex_struct.topology))
        return traj

    
    def _mmgbsa_average(self,complex_struct, traj):
        totals = {"EEL": [], "VDW": [], "EGB": [], "ECAV": []}
        rec_sys, lig_sys, cmp_sys = self._make_system_triplet(complex_struct)
        
        
        for frame in traj:
            pos_nm = frame.xyz[0] * unit.nanometer
            rec_pos = pos_nm[self._rec_idx]
            lig_pos = pos_nm[self._lig_idx]

            dEEL, dVDW, dEGB, dECAV = self._frame_mmgbsa(
                                        cmp_sys, rec_sys, lig_sys,
                                        pos_nm, rec_pos, lig_pos, frame)
            
            for k,v in zip(totals, (dEEL,dVDW,dEGB,dECAV)):
                totals[k].append(v)

        avg = {k: float(np.mean(v)) for k,v in totals.items()}
        avg["deltaG"] = sum(avg.values())
        return avg, avg["deltaG"]
    
    def _frame_mmgbsa(self, cmp_sys, rec_sys, lig_sys,
                  pos, rec_pos, lig_pos, frame):

        eel_c, vdw_c, egb_c, ecav_c = compute_frame_energies(cmp_sys, pos,      frame, self.gamma, self.beta)
        eel_r, vdw_r, egb_r, ecav_r = compute_frame_energies(rec_sys, rec_pos, frame, self.gamma, self.beta)
        eel_l, vdw_l, egb_l, ecav_l = compute_frame_energies(lig_sys, lig_pos, frame, self.gamma, self.beta)
    
        return (eel_c - (eel_r + eel_l),
                vdw_c - (vdw_r + vdw_l),
                egb_c - (egb_r + egb_l),
                ecav_c - (ecav_r + ecav_l))
    
    def _make_system_triplet(self, complex_struct: pmd.Structure):
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
    
        # ----------------------------------------------------------------
        # 1) identify ligand vs protein atoms in the complex
        #    (here: any residue whose name starts with 'LIG' → ligand)
        # ----------------------------------------------------------------
        rec_idx, lig_idx = [], []
        for atom in complex_struct.atoms:
            if atom.residue.name.startswith("LIG"):
                lig_idx.append(atom.idx)
            else:
                rec_idx.append(atom.idx)
    
        # cache masks for later slicing
        self._rec_idx = rec_idx
        self._lig_idx = lig_idx
    
        # ----------------------------------------------------------------
        # 2) split ParmEd structures
        # ----------------------------------------------------------------
        # Deep-copy once → strip away the other component
        #receptor_struct = copy.deepcopy(complex_struct)
        #ligand_struct   = copy.deepcopy(complex_struct)
    
        # remove ligand atoms from receptor_struct
        receptor_struct = complex_struct[rec_idx]
        ligand_struct   = complex_struct[lig_idx]
        '''
        for idx in sorted(lig_idx, reverse=True):
            receptor_struct.atoms[idx].exclude = False   # make sure remove_atom works
            import pdb 
            pdb.set_trace()
            receptor_struct.remove_atom(receptor_struct.atoms[idx])
            
            
        # remove protein atoms from ligand_struct
        for idx in sorted(rec_idx, reverse=True):
            ligand_struct.atoms[idx].exclude = False
            ligand_struct.remove_atom(ligand_struct.atoms[idx])
        '''
        # ----------------------------------------------------------------
        # 3) create OpenMM Systems (NoCutoff + implicit OBC2, as used elsewhere)
        # ----------------------------------------------------------------
        '''
        common_kwargs = dict(nonbondedMethod=NoCutoff,
                             constraints=HBonds,
                             removeCMMotion=False,
                             implicitSolvent=OBC2)
        '''
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
    
        return rec_sys, lig_sys, cmp_sys


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

    
