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
                   VerletIntegrator,
                   CustomExternalForce,
                   GBSAOBCForce,
                   LocalEnergyMinimizer,
                   )
import parmed as pmd
import mdtraj as md
import numpy as np
import copy
from scipy.spatial import cKDTree

from dataclasses import dataclass
from typing import Dict, List
from rdkit import Chem
from rdkit.Geometry import Point3D
from ChemEM.tools.resources import make_openmm_context

GAMMA= 0.005 
BETA = 0.0

@dataclass
class PoseScore:
    ligand_name : str
    pose_idx    : int
    deltaG          : float
    components  : Dict[str, float]      # {"ΔEEL": …, "ΔVDW": …, …}
    min_shift_A : float = None          # ligand RMSD moved by pre-min (None if not minimised)

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
        
# --------------------------------------------------------------------------- #
# Pocket minimisation: relax the ligand inside a small pinned pocket of surrounding
# residues, then score the FULL complex (no minimise). Truncating the minimisation
# system to the pocket makes each force eval ~10-100x cheaper than a whole-protein
# minimise, while the 1.2 nm nonbonded cutoff means the pocket already contains every
# atom that interacts with the ligand.
# --------------------------------------------------------------------------- #
def _pocket_precompute(complex_struct, rec_idx, rec_pos_A):
    """Per-ligand-identity constants for pocket selection: a KDTree over the receptor
    atoms and a receptor residue -> complex-atom-index map (for whole-residue expansion)."""
    atoms = list(complex_struct.atoms)
    row_res = np.empty(len(rec_idx), dtype=np.int64)
    res_atoms = {}
    for row, ai in enumerate(np.asarray(rec_idx, dtype=int)):
        residue_idx = int(atoms[int(ai)].residue.idx)
        row_res[row] = residue_idx
        res_atoms.setdefault(residue_idx, []).append(int(ai))
    return {"tree": cKDTree(np.asarray(rec_pos_A, dtype=float)),
            "row_res": row_res, "res_atoms": res_atoms}


def _select_pocket_atoms(precomp, lig_coords_A, cutoff_A):
    """Sorted tuple of receptor complex-atom indices belonging to any residue that has an
    atom within cutoff_A of any ligand atom (whole residues, deterministic for these coords)."""
    hits = precomp["tree"].query_ball_point(np.asarray(lig_coords_A, dtype=float), r=float(cutoff_A))
    rows = {int(r) for sub in hits for r in sub}
    residues = {int(precomp["row_res"][r]) for r in rows}
    out = []
    for residue_idx in residues:
        out.extend(precomp["res_atoms"][residue_idx])
    return tuple(sorted(out))


def _pocket_min_context(complex_struct, pocket_prot, lig_idx, complex_pos_nm,
                        pin_k_kcal_per_mol_A2, resource_owner, platform_name, ncpu, platform_properties):
    """Build the truncated pocket system (pocket protein + ligand), pin the pocket protein
    atoms to their reference positions, and return a reusable minimisation Context + metadata.
    Atom order in the pocket system is [sorted pocket protein ... ligand]."""
    from ChemEM.protocols.core.forces import ForceBuilder
    lig_list = [int(i) for i in np.asarray(lig_idx, dtype=int)]
    pocket_idx = list(pocket_prot) + lig_list
    pocket_struct = complex_struct[pocket_idx]
    pocket_sys = pocket_struct.createSystem(nonbondedCutoff=1.2*unit.nanometers,
                                            constraints=HBonds, removeCMMotion=False,
                                            implicitSolvent=OBC2)
    n_pp = len(pocket_prot)
    pocket_ref_nm = np.concatenate(
        [complex_pos_nm[list(pocket_prot)], complex_pos_nm[np.asarray(lig_list, dtype=int)]], axis=0)
    pin = ForceBuilder.create_positional_pin(list(range(n_pp)), pocket_ref_nm, k_name="k_static_pin")
    pocket_sys.addForce(pin)
    ctx = _reusable_context(pocket_sys, resource_owner, platform_name, ncpu, platform_properties)
    ctx.setParameter("k_static_pin", float(pin_k_kcal_per_mol_A2) * 418.4)  # k*dx^2, kcal/A^2 -> kJ/nm^2
    return {"ctx": ctx, "n_pp": n_pp, "n_pocket": pocket_ref_nm.shape[0],
            "prot_ref_nm": pocket_ref_nm[:n_pp].copy()}


def _pocket_minimise(complex_struct, rec_idx, lig_idx, complex_pos_nm, cutoff_A, max_iters,
                     resource_owner, platform_name, ncpu, platform_properties,
                     precomp=None, pocket_cache=None, pin_k_kcal_per_mol_A2=1000.0):
    """Relax the ligand inside a cutoff_A pinned pocket. Returns (full complex positions with
    the ligand relaxed [nm], ligand_rmsd_shift_A). The pocket system is keyed on its atom-set in
    pocket_cache (poses at one site share it); with pocket_cache=None it is built fresh."""
    rec_idx = np.asarray(rec_idx, dtype=int)
    lig_idx = np.asarray(lig_idx, dtype=int)
    if precomp is None:
        precomp = _pocket_precompute(complex_struct, rec_idx, complex_pos_nm[rec_idx] * 10.0)
    pocket_prot = _select_pocket_atoms(precomp, complex_pos_nm[lig_idx] * 10.0, cutoff_A)
    if not pocket_prot:
        return complex_pos_nm, None                       # no receptor near the ligand
    pk = pocket_cache.get(pocket_prot) if pocket_cache is not None else None
    if pk is None:
        pk = _pocket_min_context(complex_struct, pocket_prot, lig_idx, complex_pos_nm,
                                 pin_k_kcal_per_mol_A2, resource_owner, platform_name, ncpu,
                                 platform_properties)
        if pocket_cache is not None:
            pocket_cache[pocket_prot] = pk
    n_pp = pk["n_pp"]
    pocket_pos = np.empty((pk["n_pocket"], 3), dtype=float)
    pocket_pos[:n_pp] = pk["prot_ref_nm"]
    pocket_pos[n_pp:] = complex_pos_nm[lig_idx]
    ctx = pk["ctx"]
    ctx.setPositions(pocket_pos * unit.nanometer)
    LocalEnergyMinimizer.minimize(ctx, maxIterations=int(max_iters))
    pocket_after = np.asarray(ctx.getState(getPositions=True).getPositions(asNumpy=True)
                              .value_in_unit(unit.nanometer), dtype=float)
    relaxed_lig_nm = pocket_after[n_pp:]
    out = complex_pos_nm.copy()
    out[lig_idx] = relaxed_lig_nm
    d = (relaxed_lig_nm - complex_pos_nm[lig_idx]) * 10.0
    shift = float(np.sqrt((d * d).sum(axis=1).mean())) if len(lig_idx) else None
    return out, shift


def minimise_ligand_in_complex(complex_struct,
                               cutoff_A=12.0,
                               max_iters=300,
                               pin_k_kcal_per_mol_A2=1000.0,
                               resource_owner=None,
                               platform_name=None,
                               ncpu=None,
                               platform_properties=None):
    """Relax the ligand inside a ~cutoff_A pinned pocket (non-cached path), then return
    (relaxed_complex_struct, ligand_rmsd_shift_A). Only the ligand moves; the receptor keeps
    its reference coordinates. The full-complex MM-GBSA is scored separately (no minimise)."""
    struct = copy.deepcopy(complex_struct)
    rec_idx = [a.idx for a in struct.atoms if not a.residue.name.startswith("LIG")]
    lig_idx = [a.idx for a in struct.atoms if a.residue.name.startswith("LIG")]
    complex_pos_nm = np.asarray(struct.positions.value_in_unit(unit.nanometer), dtype=float)
    out, shift = _pocket_minimise(struct, rec_idx, lig_idx, complex_pos_nm, cutoff_A, max_iters,
                                  resource_owner, platform_name, ncpu, platform_properties,
                                  pin_k_kcal_per_mol_A2=pin_k_kcal_per_mol_A2)
    struct.coordinates = out * 10.0                       # nm -> A, update full complex
    return struct, shift


# --------------------------------------------------------------------------- #
# Per-case system/Context reuse (opt-in via reuse_cache).
#
# For every pose of a case the receptor is unchanged and, for poses of the SAME
# molecule, the complex/ligand topologies are unchanged — so the 3 OpenMM systems
# + Contexts can be built ONCE per ligand-identity and reused across poses via
# setPositions(), instead of rebuilt every pose. Bit-identical to the per-pose
# path (same systems + same positions => same energies).
# --------------------------------------------------------------------------- #
def _mmgbsa_cache_key(ligand):
    """Atom-order-SENSITIVE key over the parameterised ligand structure. The cached OpenMM
    systems reuse per-atom parameters and per-pose coordinates BY INDEX, so an entry may be
    reused only for a ligand whose complex_structure atom indexing is identical — a canonical
    SMILES would mis-map coordinates onto the wrong atoms. Poses of the same docked ligand
    share the atom order, so they still dedupe."""
    st = ligand.complex_structure
    atoms = tuple(getattr(a, "atomic_number", None) or getattr(a, "element", 0) for a in st.atoms)
    bonds = tuple(sorted((b.atom1.idx, b.atom2.idx) for b in st.bonds))
    return (atoms, bonds)


def _reusable_context(system, resource_owner, platform_name, ncpu, platform_properties):
    intg = VerletIntegrator(0.002*unit.picoseconds)
    pn = platform_name if platform_name is not None else getattr(resource_owner, "platform", None)
    return make_openmm_context(system, intg, platform_name=pn, source=resource_owner,
                               ncpu=ncpu, platform_properties=platform_properties)


def _build_mmgbsa_entry(protein, ligand, first_pose_A, resource_owner,
                        platform_name, ncpu, platform_properties, minimise_ligand, cutoff_A=12.0):
    """Built once per ligand-identity: the 3 full-complex systems + reusable Contexts and the
    constant receptor positions / complex topology used for every pose. When minimising, also the
    pocket-selection precompute + a per-pocket Context cache (poses at one site share the pocket)."""
    complex_struct = build_complex(protein.complex_structure, ligand, first_pose_A)
    rec_sys, lig_sys, cmp_sys, rec_idx, lig_idx = make_system_triplet(complex_struct)
    rec_idx = np.asarray(rec_idx, dtype=int)
    lig_idx = np.asarray(lig_idx, dtype=int)
    complex_pos_nm = np.asarray(complex_struct.positions.value_in_unit(unit.nanometer), dtype=float)
    entry = {
        "rec_idx": rec_idx,
        "lig_idx": lig_idx,
        "n_total": complex_pos_nm.shape[0],
        "prot_ref_nm": complex_pos_nm[rec_idx].copy(),        # receptor positions (constant)
        "cmp_top": md.Topology.from_openmm(complex_struct.topology),
        "cmp_ctx": _reusable_context(cmp_sys, resource_owner, platform_name, ncpu, platform_properties),
        "rec_ctx": _reusable_context(rec_sys, resource_owner, platform_name, ncpu, platform_properties),
        "lig_ctx": _reusable_context(lig_sys, resource_owner, platform_name, ncpu, platform_properties),
        "minimise": bool(minimise_ligand),
    }
    # The receptor never moves (pocket minimise relaxes only the ligand), so its EEL/VDW/EGB are
    # constant across every pose — compute once here and reuse, saving one whole-protein energy
    # eval per pose. Bit-identical: rec positions are always entry["prot_ref_nm"].
    entry["rec_energy"] = _mm_group_energies(entry["rec_ctx"], entry["prot_ref_nm"])
    if minimise_ligand:
        entry["complex_struct"] = complex_struct
        entry["cutoff_A"] = float(cutoff_A)
        entry["pocket_precomp"] = _pocket_precompute(complex_struct, rec_idx,
                                                     complex_pos_nm[rec_idx] * 10.0)
        entry["pocket_cache"] = {}
        entry["ctx_args"] = (resource_owner, platform_name, ncpu, platform_properties)
    return entry


def _mm_group_energies(ctx, positions_nm):
    """EEL/VDW/EGB (kcal/mol) for a cached Context (setPositions, no rebuild)."""
    ctx.setPositions(positions_nm * unit.nanometer)
    conv = unit.kilocalories_per_mole
    eel = ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(conv)
    vdw = ctx.getState(getEnergy=True, groups={2}).getPotentialEnergy().value_in_unit(conv)
    egb = ctx.getState(getEnergy=True, groups={3}).getPotentialEnergy().value_in_unit(conv)
    return eel, vdw, egb


def _score_with_entry(entry, pose_A, gamma, beta, minimise_ligand, minimise_iters):
    """One pose through the cached systems/Contexts. Mirrors frame_mmgbsa exactly:
    the SASA of the COMPLEX frame is used for all three ECAV terms, so
    dECAV = ecav_c - (ecav_r + ecav_l) = -ecav."""
    rec_idx, lig_idx = entry["rec_idx"], entry["lig_idx"]
    complex_pos_nm = np.empty((entry["n_total"], 3), dtype=float)
    complex_pos_nm[rec_idx] = entry["prot_ref_nm"]
    lig_pre_nm = np.asarray(pose_A, dtype=float) / 10.0
    complex_pos_nm[lig_idx] = lig_pre_nm

    min_shift = None
    if minimise_ligand and entry.get("minimise"):
        ro, pn, ncpu, pp = entry["ctx_args"]
        complex_pos_nm, min_shift = _pocket_minimise(
            entry["complex_struct"], rec_idx, lig_idx, complex_pos_nm,
            entry["cutoff_A"], minimise_iters, ro, pn, ncpu, pp,
            precomp=entry["pocket_precomp"], pocket_cache=entry["pocket_cache"])

    lig_pos = complex_pos_nm[lig_idx]
    eel_c, vdw_c, egb_c = _mm_group_energies(entry["cmp_ctx"], complex_pos_nm)
    eel_r, vdw_r, egb_r = entry["rec_energy"]            # constant receptor (see _build_mmgbsa_entry)
    eel_l, vdw_l, egb_l = _mm_group_energies(entry["lig_ctx"], lig_pos)
    frame = md.Trajectory(xyz=complex_pos_nm[None, :, :], topology=entry["cmp_top"])
    ecav = gamma * (md.shrake_rupley(frame)[0].sum() * 100.0) + beta
    comps = {
        "EEL": eel_c - (eel_r + eel_l),
        "VDW": vdw_c - (vdw_r + vdw_l),
        "EGB": egb_c - (egb_r + egb_l),
        "ECAV": ecav - (ecav + ecav),
    }
    return comps, float(sum(comps.values())), min_shift


def _score_single_pose_cached(positions, ligand, protein, pose_idx, resource_owner,
                              platform_name, ncpu, platform_properties,
                              minimise_ligand, minimise_iters, minimise_cutoff_A=12.0):
    cache = getattr(resource_owner, "_mmgbsa_cache", None)
    if cache is None:
        cache = {}
        resource_owner._mmgbsa_cache = cache
    key = (_mmgbsa_cache_key(ligand), bool(minimise_ligand),
           round(float(minimise_cutoff_A), 3) if minimise_ligand else None)
    entry = cache.get(key)
    if entry is None:
        entry = _build_mmgbsa_entry(protein, ligand, positions, resource_owner,
                                    platform_name, ncpu, platform_properties, minimise_ligand,
                                    cutoff_A=minimise_cutoff_A)
        cache[key] = entry
    comps, deltaG, min_shift = _score_with_entry(entry, positions, GAMMA, BETA,
                                                 minimise_ligand, minimise_iters)
    return PoseScore(ligand_name=ligand.ligand_int, pose_idx=pose_idx,
                     deltaG=deltaG, components=comps, min_shift_A=min_shift)


def score_single_pose(positions,
                      ligand,
                      protein,
                      pose_idx: str | None = None,
                      resource_owner=None,
                      platform_name=None,
                      ncpu=None,
                      platform_properties=None,
                      minimise_ligand=False,
                      minimise_iters=300,
                      minimise_cutoff_A=12.0,
                      reuse_cache=False):


    if pose_idx is None:
        pose_idx = 0

    if reuse_cache and resource_owner is not None:
        return _score_single_pose_cached(
            positions, ligand, protein, pose_idx, resource_owner,
            platform_name, ncpu, platform_properties, minimise_ligand, minimise_iters,
            minimise_cutoff_A=minimise_cutoff_A,
        )

    if minimise_ligand:
        # Non-cached minimise path: build a throwaway entry and score through the SAME code as
        # the cached case, so it is bit-identical to reuse_cache=True (and avoids the separate
        # mmgbsa_from_traj coordinate path). The entry is not stored, so every pose rebuilds.
        entry = _build_mmgbsa_entry(protein, ligand, positions, resource_owner,
                                    platform_name, ncpu, platform_properties,
                                    minimise_ligand=True, cutoff_A=minimise_cutoff_A)
        comps, deltaG, min_shift = _score_with_entry(entry, positions, GAMMA, BETA,
                                                     True, minimise_iters)
        return PoseScore(ligand_name=ligand.ligand_int, pose_idx=pose_idx,
                         deltaG=deltaG, components=comps, min_shift_A=min_shift)

    complex_struct = build_complex(protein.complex_structure,
                                   ligand,
                                   positions)

    traj = parmed_structure_to_single_frame_traj(complex_struct)
    comps, deltaG = mmgbsa_from_traj(
        complex_struct,
        traj,
        resource_owner=resource_owner,
        platform_name=platform_name,
        ncpu=ncpu,
        platform_properties=platform_properties,
    )

    return PoseScore(ligand_name = ligand.ligand_int,
                     pose_idx    = pose_idx,
                     deltaG      = deltaG,
                     components  = comps,
                     min_shift_A = None)
    
    
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


def mmgbsa_from_traj(
    complex_struct,
    traj,
    gamma=GAMMA,
    beta=BETA,
    resource_owner=None,
    platform_name=None,
    ncpu=None,
    platform_properties=None,
):
    totals = {"EEL": [], "VDW": [], "EGB": [], "ECAV": []}
    rec_sys, lig_sys, cmp_sys, rec_idx, lig_idx = make_system_triplet(complex_struct)
    
    
    for frame in traj:
        pos_nm = frame.xyz[0] * unit.nanometer
        rec_pos = pos_nm[rec_idx]
        lig_pos = pos_nm[lig_idx]

        dEEL, dVDW, dEGB, dECAV = frame_mmgbsa(
                                    cmp_sys, rec_sys, lig_sys,
                                    pos_nm, rec_pos, lig_pos, frame, gamma, beta,
                                    resource_owner=resource_owner,
                                    platform_name=platform_name,
                                    ncpu=ncpu,
                                    platform_properties=platform_properties)
        
        for k,v in zip(totals, (dEEL,dVDW,dEGB,dECAV)):
            totals[k].append(v)

    avg = {k: float(np.mean(v)) for k,v in totals.items()}
    avg["deltaG"] = sum(avg.values())
    return avg, avg["deltaG"]

def frame_mmgbsa( cmp_sys, rec_sys, lig_sys,
              pos, rec_pos, lig_pos, frame, gamma, beta,
              resource_owner=None,
              platform_name=None,
              ncpu=None,
              platform_properties=None):

    kwargs = dict(
        resource_owner=resource_owner,
        platform_name=platform_name,
        ncpu=ncpu,
        platform_properties=platform_properties,
    )
    eel_c, vdw_c, egb_c, ecav_c = compute_frame_energies(cmp_sys, pos,      frame, gamma, beta, **kwargs)
    eel_r, vdw_r, egb_r, ecav_r = compute_frame_energies(rec_sys, rec_pos, frame, gamma, beta, **kwargs)
    eel_l, vdw_l, egb_l, ecav_l = compute_frame_energies(lig_sys, lig_pos, frame, gamma, beta, **kwargs)

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


def compute_frame_energies(
    system,
    positions,
    frame,
    gamma,
    beta,
    resource_owner=None,
    platform_name=None,
    ncpu=None,
    platform_properties=None,
):
    intg   = VerletIntegrator(0.002*unit.picoseconds)
    platform_name = platform_name if platform_name is not None else getattr(resource_owner, "platform", None)
    ctx    = make_openmm_context(
        system,
        intg,
        platform_name=platform_name,
        source=resource_owner,
        ncpu=ncpu,
        platform_properties=platform_properties,
    )
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

    
