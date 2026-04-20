# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>



from tqdm.auto import tqdm
from contextlib import nullcontext
import time
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union

from openmm import (
    unit, app, 
    LangevinIntegrator, Platform, 
    MonteCarloBarostat,
    Vec3
)
from openmm.app import NoCutoff, HBonds, PDBFile
from openmm import CustomExternalForce


from ChemEM.protocols.core.simulation import (create_system, 
                                              identify_indices_from_name,
                                              run_biased_md_burst,
                                              check_pose_forces)

from ChemEM.protocols.core.forces import (add_density_map_force,
                                          add_sse_rmsd_forces,
                                          ForceBuilder)

from ChemEM.protocols.core.biomolecule import (find_atoms_outside_ligand,
                                               select_atoms,
                                               create_structure_subset)

from openmm import CustomExternalForce, CustomCompoundBondForce, CustomCentroidBondForce

class ProgressLogger:
    @staticmethod
    def get_bar(total, desc, unit="steps", leave=False, active=True):
        if active and total > 0:
            return tqdm(total=total, desc=desc, unit=unit, leave=leave)
        return nullcontext()
    
    @staticmethod
    def human_ps(ps: float) -> str:
        if ps >= 1000.0: return f"{ps/1000.0:.1f} ns"
        if ps >= 1.0: return f"{ps:.1f} ps"
        return f"{ps*1000.0:.1f} fs"



class ChemEMSimulationSetup:
    
    def __init__(self, 
                 protein_structure,
                 ligand_structure,
                 residues=None,
                 density_map=None,
                 padding=1.0,
                 solvent=app.GBn2,
                 platform_name='CPU',
                 # Restraint settings
                 restrain_side_chains = True,
                 protein_restraint='protein',  # 'protein', 'sse', 'none'
                 sse_groups: Optional[List[List[int]]] = None,
                 sse_k: float = 50.0,
                 pin_k: float = 10000.0,
                 localise: bool = True,
                 global_k : float = 150.0,
                 smooth_sigma_A : float = 0.0,
                 smooth_sigma_vox: float = 0.0,
                 pin_specs: Optional[List[str]] = None,
                 distance_specs: Optional[List[str]] = None):
        
        self.density_map = density_map
        self.global_k = global_k
        self.smooth_sigma_A = smooth_sigma_A
        
        if residues is not None:
            protein_structure = create_structure_subset(protein_structure, residues)
        
        self.complex_structure, self.complex_system = create_system(
            protein_structure, ligand_structure, solvent
        )
        
        #-----Cache ligand indexes
        self.all_ligand_indices, self.ligand_heavy_indices = identify_indices_from_name(self.complex_structure)
        
        #-----Density Map Forces
        if self.density_map:
            
            add_density_map_force(self.complex_system, 
                                  self.complex_structure,
                                  self.density_map, 
                                  self.global_k, 
                                  padding, 
                                  smooth_sigma_vox=smooth_sigma_vox)

        
        # -----Integrator & Platform Setup
        self.integrator = LangevinIntegrator(
            300 * unit.kelvin, 
            1.0 / unit.picoseconds, 
            1.0 * unit.femtoseconds
        )
        self.platform = Platform.getPlatformByName(platform_name)
        props = {"Precision": "single"} if platform_name != 'CPU' else {}
        
        self.simulation = app.Simulation(
            self.complex_structure.topology,
            self.complex_system,
            self.integrator,
            self.platform,
            platformProperties=props
        )
        
        self.simulation.context.setPositions(self.complex_structure.positions)
        
        # 5. Restraints
        self.restraint_config = {
            'mode': protein_restraint, 
            'sse_groups': sse_groups, 
            'sse_k': sse_k, 
            'pin_k': pin_k,
            'localise': localise,
            'restrain_sidechains': restrain_side_chains,
            'pin_specs': list(pin_specs or []),
            'distance_specs': list(distance_specs or []),
        }
        self._setup_protein_restraints()
        self._setup_user_restraints()
    
    
    def _setup_protein_restraints(self):
        mode = self.restraint_config['mode']
        if mode == 'none': return
        
        state = self.simulation.context.getState(getPositions=True)
        ref_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        
        # --- SSE Mode ---
        if mode == 'sse':
            add_sse_rmsd_forces(
                complex_system=self.complex_system,
                complex_structure=self.complex_structure,
                ref_nm=ref_nm,
                sse_groups=self.restraint_config['sse_groups'],
                sse_k=self.restraint_config['sse_k'],
                localise=self.restraint_config['localise'],
                all_ligand_indices=self.all_ligand_indices,
                ligand_heavy_indices=self.ligand_heavy_indices
            )
            return  # Exit here as SSE forces are fully handled
        
        # --- Protein Pin Mode ---
        
        self.protein_heavy_indices = [
            a.idx for a in self.complex_structure.atoms
            if not a.residue.name.startswith('LIG') and a.element != 1
        ]
        
        ligand_set = set(self.all_ligand_indices)
        atoms_to_restrain = select_atoms(
            self.complex_structure, 
            indices=self.protein_heavy_indices, 
            exclude_indices=ligand_set,
        )
        
        if self.restraint_config['localise']:
            outside_indices = find_atoms_outside_ligand(
                self.complex_structure, self.ligand_heavy_indices,
                exclude_backbone=False if self.restraint_config['restrain_sidechains'] else True
            )
            
            atoms_to_restrain = [i for i in atoms_to_restrain if i not in outside_indices]
       
        if atoms_to_restrain:
            pin_f = ForceBuilder.create_positional_pin(
                atoms_to_restrain, ref_nm, k_name="k_static_pin"
            )
            self.complex_system.addForce(pin_f)
            self.simulation.context.reinitialize(preserveState=True)
            self.simulation.context.setParameter("k_static_pin", self.restraint_config['pin_k'])
    
    '''
    def _parse_atom_spec(self, spec: str):
        parts = [p.strip() for p in str(spec).split(":") if p.strip()]
        if len(parts) == 4:
            chain_id, residue_name, residue_number, atom_name = parts
        elif len(parts) == 3:
            chain_id = None
            residue_name, residue_number, atom_name = parts
        else:
            raise ValueError(
                f"Invalid atom spec '{spec}'. Expected CHAIN:RESNAME:RESNUM:ATOM or RESNAME:RESNUM:ATOM."
            )
        return chain_id, residue_name, residue_number, atom_name
    '''
    def _parse_atom_spec(self, spec: str):
        parts = [p.strip() for p in spec.split(":")]
    
        # Ligand spec: LIG:<ligand_index>:<atom_name>
        if len(parts) == 3 and parts[0].upper() == "LIG":
            _, lig_idx, atom_name = parts
            try:
                lig_idx = int(lig_idx)
            except ValueError:
                raise ValueError(
                    f"Invalid ligand atom spec '{spec}'. "
                    f"Expected LIG:<ligand_index>:<atom_name>, e.g. LIG:0:O6"
                )
            return ("ligand", lig_idx, atom_name)
    
        # Protein / polymer spec: <chain_id>:<resname>:<resnum>:<atom_name>
        if len(parts) == 4:
            chain_id, residue_name, residue_number, atom_name = parts
            try:
                residue_number = int(residue_number)
            except ValueError:
                raise ValueError(
                    f"Invalid protein atom spec '{spec}'. "
                    f"Expected <chain>:<resname>:<resnum>:<atom>, e.g. A:ASP:45:OD1"
                )
            return ("protein", chain_id, residue_name, residue_number, atom_name)
    
        raise ValueError(
            f"Invalid atom spec '{spec}'. "
            f"Use either <chain>:<resname>:<resnum>:<atom> or LIG:<ligand_index>:<atom>"
        )


    def _resolve_atom_spec(self, spec: str) -> int:
        parsed = self._parse_atom_spec(spec)
    
        if parsed[0] == "ligand":
            _, lig_idx, atom_name = parsed
            target_resname = f"LIG{lig_idx}"
    
            matches = []
            for res in self.complex_structure.topology.residues():
                if res.name != target_resname:
                    continue
    
                atoms = [atom for atom in res.atoms() if atom.name == atom_name]
                matches.extend(atoms)
    
            if not matches:
                raise ValueError(
                    f"Could not resolve ligand atom spec '{spec}'. "
                    f"Expected residue named '{target_resname}' in the refinement structure."
                )
            if len(matches) > 1:
                match_str = ", ".join(
                    f"{a.residue.name}:{a.residue.id}:{a.name}" for a in matches
                )
                raise ValueError(
                    f"Ligand atom spec '{spec}' is ambiguous in the current refinement structure. "
                    f"Matches: {match_str}"
                )
    
            return int(matches[0].index)
    
        # protein path
        _, chain_id, residue_name, residue_number, atom_name = parsed
    
        matches = []
        for res in self.complex_structure.topology.residues():
            if not (
                res.chain.id == chain_id
                and res.name == residue_name
                and int(res.id) == int(residue_number)
            ):
                continue
    
            atoms = [atom for atom in res.atoms() if atom.name == atom_name]
            if len(atoms) != 1:
                raise RuntimeError(
                    f"[Error] Expected exactly one atom for protein spec '{spec}', "
                    f"found {len(atoms)}"
                )
    
            matches.append(atoms[0])
    
        if not matches:
            raise ValueError(
                f"Could not resolve atom spec '{spec}' in the current refinement structure."
            )
        if len(matches) > 1:
            match_str = ", ".join(
                f"{a.residue.chain.id}:{a.residue.name}:{a.residue.id}:{a.name}"
                for a in matches
            )
            raise ValueError(
                f"Atom spec '{spec}' is ambiguous in the current refinement structure. "
                f"Matches: {match_str}"
            )
    
        return int(matches[0].index)

    @staticmethod
    def _residue_numbers_for_matching(residue):
        values = set()
        for attr in ("number", "idx", "resnum"):
            value = getattr(residue, attr, None)
            if value is not None:
                values.add(str(value).strip())
        return values
    '''
    def __resolve_atom_spec(self, spec: str) -> int:
        chain_id, residue_name, residue_number, atom_name = self._parse_atom_spec(spec)
        #DEBUG
        print(chain_id, residue_name, residue_number, atom_name)
        #END DEBUG
        matches = []
        
                
        for res in self.complex_structure.topology.residues():
            
            
            if not all([res.chain.id == chain_id,  res.name == residue_name, int(res.id) == int(residue_number)]):
                continue
            
            
            
            atoms = [atom for atom in res.atoms() if atom.name == atom_name]
            
            if len(atoms) != 1:
                
                raise RuntimeError(f"[Error] Protein multiple atoms with same spec {spec}")
            
            matches.append(atoms[0])
        

        if not matches:
            _res = [i for i in self.complex_structure.topology.residues() if i.name.startswith("LIG")]
            import pdb 
            pdb.set_trace()
            
            raise ValueError(f"Could not resolve atom spec '{spec}' in the current refinement structure.")
        if len(matches) > 1:
            match_str = ", ".join(
                f"{getattr(a.residue, 'chain', '')}:{a.residue.name}:{getattr(a.residue, 'number', getattr(a.residue, 'idx', '?'))}:{a.name}"
                for a in matches
            )
            raise ValueError(
                f"Atom spec '{spec}' is ambiguous in the current refinement structure. Matches: {match_str}"
            )
            
        
        return int(matches[0].index)
    '''
    def _parse_distance_spec(self, spec: str):
        parts = [p.strip() for p in str(spec).split(";")]
        
        if len(parts) != 3:
            raise ValueError(
                f"Invalid distance spec '{spec}'. Expected <atom1-spec>;<atom2-spec>;<distance-in-A>."
            )
        atom1_spec, atom2_spec, distance_A = parts
        return atom1_spec, atom2_spec, float(distance_A)

    def _setup_user_restraints(self):
        pin_specs = self.restraint_config.get('pin_specs', [])
        distance_specs = self.restraint_config.get('distance_specs', [])
        if not pin_specs and not distance_specs:
            return

        state = self.simulation.context.getState(getPositions=True)
        ref_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        needs_reinit = False

        if pin_specs:
            pin_indices = [self._resolve_atom_spec(spec) for spec in pin_specs]
            pin_force = ForceBuilder.create_positional_pin(
                pin_indices,
                ref_nm,
                k_name="k_user_pin",
            )
            self.complex_system.addForce(pin_force)
            needs_reinit = True
            print(f"Added {len(pin_indices)} user pin restraint(s).")

        if distance_specs:
            restraints = []
            for spec in distance_specs:
                
                atom1_spec, atom2_spec, distance_A = self._parse_distance_spec(spec)
                atom1_idx = self._resolve_atom_spec(atom1_spec)
                atom2_idx = self._resolve_atom_spec(atom2_spec)
                restraints.append((atom1_idx, atom2_idx, distance_A * 0.1))

            dist_force = ForceBuilder.create_distance_restraint(
                restraints,
                k_name="k_user_distance",
            )
            self.complex_system.addForce(dist_force)
            needs_reinit = True
            print(f"Added {len(restraints)} user distance restraint(s).")

        if needs_reinit:
            self.simulation.context.reinitialize(preserveState=True)
            k_value = float(self.restraint_config['pin_k'])
            try:
                self.simulation.context.setParameter("k_user_pin", k_value)
            except Exception:
                pass
            try:
                self.simulation.context.setParameter("k_user_distance", k_value)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API for Force & State Management (Used by Minimizers)
    # ------------------------------------------------------------------
    
    def set_ligand_flatbottom_tether(self, ref_pos_nm, *, r0_A=0.1, k_kcal_per_mol_A2=0.2):
        self._ensure_ligand_flatbottom_tether()
        r0_nm = float(r0_A) * 0.1
        k_kj_per_mol_nm2 = float(k_kcal_per_mol_A2) * 418.4
    
        f = self._lig_fb_tether
        for p, atom_idx in enumerate(self._lig_fb_atoms):
            x0, y0, z0 = map(float, ref_pos_nm[int(atom_idx)])
            f.setParticleParameters(p, int(atom_idx), [x0, y0, z0])
    
        f.updateParametersInContext(self.simulation.context)
        self.simulation.context.setParameter("r0", r0_nm)
        self.simulation.context.setParameter("k_tether", k_kj_per_mol_nm2)
    
    def clear_ligand_flatbottom_tether(self):
        if hasattr(self, "_lig_fb_tether"):
            self.simulation.context.setParameter("k_tether", 0.0)

    def _ensure_ligand_flatbottom_tether(self):
        if hasattr(self, "_lig_fb_tether"): return
        dist = "sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0))"
        expr = f"0.5*k_tether*step({dist}-r0)*({dist}-r0)^2"
        f = CustomExternalForce(expr)

        f.addGlobalParameter("k_tether", 0.0)
        f.addGlobalParameter("r0", 0.0)
        f.addPerParticleParameter("x0")
        f.addPerParticleParameter("y0")
        f.addPerParticleParameter("z0")
    
        self._lig_fb_atoms = list(self.ligand_heavy_indices)
        for atom_idx in self._lig_fb_atoms:
            f.addParticle(int(atom_idx), [0.0, 0.0, 0.0])
    
        self.complex_system.addForce(f)
        self._lig_fb_tether = f
        self.simulation.context.reinitialize(preserveState=True)

    def set_ligand_tether(self, ref_pos_nm: np.ndarray, k_kcal_per_mol_A2: float):
        self._ensure_ligand_tether_force()
        k_kj_per_mol_nm2 = float(k_kcal_per_mol_A2) * 418.4
    
        f = self._lig_tether_force
        for p, atom_idx in enumerate(self._lig_tether_atoms):
            x0, y0, z0 = map(float, ref_pos_nm[int(atom_idx)])
            f.setParticleParameters(p, int(atom_idx), [x0, y0, z0])
    
        f.updateParametersInContext(self.simulation.context) 
        self.simulation.context.setParameter("k_tether", k_kj_per_mol_nm2)
    
    def clear_ligand_tether(self):
        if hasattr(self, "_lig_tether_force"):
            self.simulation.context.setParameter("k_tether", 0.0)

    def _ensure_ligand_tether_force(self):
        if hasattr(self, "_lig_tether_force"): return
        f = CustomExternalForce("0.5*k_tether*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        f.addGlobalParameter("k_tether", 0.0)
        f.addPerParticleParameter("x0")
        f.addPerParticleParameter("y0")
        f.addPerParticleParameter("z0")
    
        self._lig_tether_atoms = list(self.ligand_heavy_indices)
        for atom_idx in self._lig_tether_atoms:
            f.addParticle(int(atom_idx), [0.0, 0.0, 0.0])
    
        self.complex_system.addForce(f)
        self._lig_tether_force = f
        self.simulation.context.reinitialize(preserveState=True)

    # ------------------------------------------------------------------
    # Debugging Helpers
    # ------------------------------------------------------------------

    def debug_min_state(self, tag: str, intended_lig_nm=None):
        st = self.simulation.context.getState(getEnergy=True, getPositions=True, getForces=True)
        E = st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        frc = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    
        lig_idx = np.array(self.all_ligand_indices, dtype=int)
        lig_f = frc[lig_idx]
        lig_f_norm = np.linalg.norm(lig_f, axis=1)
        max_f = float(lig_f_norm.max()) if lig_f_norm.size else 0.0
        rms_f = float(np.sqrt(np.mean(lig_f_norm**2))) if lig_f_norm.size else 0.0
    
        msg = f"[{tag}] E={E:.2f} kcal/mol | ligand max|F|={max_f:.3g} | ligand rms|F|={rms_f:.3g}"
        if intended_lig_nm is not None:
            d = pos[lig_idx] - intended_lig_nm
            dA = np.linalg.norm(d, axis=1) * 10.0
            msg += f" | ligand rms disp={float(np.sqrt(np.mean(dA**2))):.3f} Å | max={float(dA.max()):.3f} Å"
    
        for pname in ("k_static_pin", "k_stage_heavy", "k_stage_bb", "global_k"):
            try: msg += f" | {pname}={self.simulation.context.getParameter(pname)}"
            except Exception: pass
        print(msg)
    
    def get_ligand_com_nm(self, pos_nm=None):
        if pos_nm is None:
            st = self.simulation.context.getState(getPositions=True)
            pos_nm = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    
        pos_nm = np.asarray(pos_nm, dtype=np.float64)
        if pos_nm.ndim != 2 or pos_nm.shape[1] != 3:
            pos_nm = np.asarray([[float(v[0]), float(v[1]), float(v[2])] for v in pos_nm], dtype=np.float64)
    
        lig_idx = np.asarray(self.all_ligand_indices, dtype=int)
        xyz = pos_nm[lig_idx]
    
        masses = np.array(
            [self.complex_system.getParticleMass(int(i)).value_in_unit(unit.dalton) for i in lig_idx],
            dtype=np.float64
        )
        total_mass = masses.sum()
        if total_mass <= 0.0:
            raise ValueError("Ligand COM tether failed: total ligand mass is zero.")
    
        return (xyz * masses[:, None]).sum(axis=0) / total_mass

    
    def _ensure_ligand_com_tether_force(self):
        if hasattr(self, "_lig_com_tether_force"):
            return
    
        lig_idx = [int(i) for i in self.all_ligand_indices]
    
        # d = COM distance to reference point
        # soft = smooth approximation to max(0, d-r0)
        # energy = 0.5*k*soft^2
        #
        # alpha controls how sharp the switch is:
        # larger alpha -> closer to hard flat-bottom
        # smaller alpha -> smoother onset
        expr = (
            "0.5*lig_com_k*soft^2;"
            "soft=log(1+exp(lig_com_alpha*(d-lig_com_r0)))/lig_com_alpha;"
            "d=pointdistance(x1,y1,z1,x0,y0,z0)"
        )
    
        f = CustomCentroidBondForce(1, expr)
        f.addGlobalParameter("lig_com_k", 0.0)
        f.addGlobalParameter("lig_com_r0", 0.0)
        f.addGlobalParameter("lig_com_alpha", 80.0)  # nm^-1, sharper switch at larger values
        f.addPerBondParameter("x0")
        f.addPerBondParameter("y0")
        f.addPerBondParameter("z0")
    
        # By default, group center uses particle masses -> COM
        group_idx = f.addGroup(lig_idx)
        bond_idx = f.addBond([group_idx], [0.0, 0.0, 0.0])
    
        self.complex_system.addForce(f)
        self._lig_com_tether_force = f
        self._lig_com_group_idx = int(group_idx)
        self._lig_com_tether_bond_idx = int(bond_idx)
    
        self.simulation.context.reinitialize(preserveState=True)
    
    
    def set_ligand_com_tether(
        self,
        ref_pos_nm,
        *,
        r0_A=0.5,
        k_kcal_per_mol_A2=5.0,
        alpha_per_nm=80.0,
    ):
        """
        Smooth flat-bottom harmonic tether on the ligand COM.
    
        Inside r0_A: ~no restoring force
        Outside r0_A: restoring force grows with distance
        """
        self._ensure_ligand_com_tether_force()
    
        ref_com_nm = np.asarray(self.get_ligand_com_nm(ref_pos_nm), dtype=np.float64).reshape(3,)
        params = [float(ref_com_nm[0]), float(ref_com_nm[1]), float(ref_com_nm[2])]
    
        f = self._lig_com_tether_force
        f.setBondParameters(
            self._lig_com_tether_bond_idx,
            [int(self._lig_com_group_idx)],
            params
        )
        f.updateParametersInContext(self.simulation.context)
    
        r0_nm = float(r0_A) * 0.1
        k_kj_per_mol_nm2 = float(k_kcal_per_mol_A2) * 418.4
    
        self.simulation.context.setParameter("lig_com_r0", r0_nm)
        self.simulation.context.setParameter("lig_com_k", k_kj_per_mol_nm2)
        self.simulation.context.setParameter("lig_com_alpha", float(alpha_per_nm))
    
    
    def clear_ligand_com_tether(self):
        if hasattr(self, "_lig_com_tether_force"):
            self.simulation.context.setParameter("lig_com_k", 0.0)
        
    
    
    
    
    
        

class BatchPoseMinimizer:
    def __init__(self, env):
        self.env = env
        
    def run(self, 
            ligand_poses_angstrom, 
            do_biased_md=False, 
            md_ps=5.0, 
            dt_ps=0.001,
            md_temp_K=150.0,
            md_seed=1,
            md_pre_min_iters=0,
            md_report_ps=0.0,
            max_iters=0):
        
        results = []
        print(f"Minimizing {len(ligand_poses_angstrom)} poses...")

       
        initial_pos = self.env.simulation.context.getState(getPositions=True).getPositions()

        for i, pose_ang in enumerate(ligand_poses_angstrom):
            t_start = time.perf_counter()
            print(f"\n--- Pose {i+1} ---")

            try:
               
                self.env.simulation.context.setPositions(initial_pos)
                curr_pos_nm = np.array(initial_pos.value_in_unit(unit.nanometer), copy=True)
                curr_pos_nm[self.env.all_ligand_indices] = pose_ang / 10.0

                # IMPORTANT: set with units
                self.env.simulation.context.setPositions(curr_pos_nm * unit.nanometer)

                # Enforce constraints after setPositions
                try:
                    self.env.simulation.context.applyConstraints(1e-6)
                except Exception:
                    pass

                intended_lig_nm = np.array(curr_pos_nm)[self.env.all_ligand_indices]

                bad, reason = check_pose_forces(
                    self.env.simulation, 
                    self.env.all_ligand_indices
                )
                if bad:
                    print(f"[pose {i+1}] SKIP (pre-check): {reason}")
                    continue

                self.env.set_ligand_flatbottom_tether(curr_pos_nm, r0_A=0.4, k_kcal_per_mol_A2=0.2)

                
                if do_biased_md and md_ps > 0:
                    run_biased_md_burst(
                        simulation=self.env.simulation,
                        ligand_indices=self.env.all_ligand_indices,
                        intended_lig_nm=intended_lig_nm,
                        md_ps=md_ps,
                        dt_ps=dt_ps,
                        md_temp_K=md_temp_K,
                        md_seed=md_seed,
                        md_pre_min_iters=md_pre_min_iters,
                        md_report_ps=md_report_ps,
                        pose_index=i+1
                    )

                    
                    bad, reason = check_pose_forces(
                        self.env.simulation, 
                        self.env.all_ligand_indices,
                        
                    )
                    if bad:
                        print(f"[pose {i+1}] SKIP (post-MD): {reason}")
                        continue

                
                self.env.simulation.minimizeEnergy(maxIterations=max_iters)

                
                bad, reason = check_pose_forces(
                    self.env.simulation, 
                    self.env.all_ligand_indices,
                    
                )
                if bad:
                    print(f"[pose {i+1}] SKIP (post-min): {reason}")
                    continue

                
                state = self.env.simulation.context.getState(getPositions=True, getEnergy=True)
                final_pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

                dA = np.linalg.norm(final_pos[self.env.all_ligand_indices] - intended_lig_nm, axis=1) * 10.0
                print(f"  -> pose {i+1} ligand move: rms={np.sqrt(np.mean(dA**2)):.3f} Å max={dA.max():.3f} Å")

                final_lig = final_pos[self.env.all_ligand_indices] * 10.0
                energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
                
                results.append(final_lig)
                print(f" Pose {i+1} Finished: {time.perf_counter() - t_start:.2f}s | E: {energy:.2f} kcal/mol")

            except FloatingPointError as e:
                print(f"[pose {i+1}] SKIP (non-finite): {e}")

            except Exception as e:
                print(f"[pose {i+1}] SKIP (exception): {type(e).__name__}: {e}")

            finally:
                # CRITICAL: Always clear the tether so a bad pose doesn't poison the next pose
                try:
                    self.env.clear_ligand_flatbottom_tether()
                except Exception:
                    pass
        #reset the simulation as its a refernece to the actual simulation
        self.env.simulation.context.setPositions(initial_pos)
        return results


class MinimiseInPlace:
    def __init__(self, env):
        self.env = env 
    
    def run(self, 
            do_biased_md=False, 
            md_ps=5.0, 
            dt_ps=0.001,
            md_temp_K=150.0,
            max_iters=0):
        
        print("Starting In-Place Minimization...")
        t_start = time.perf_counter()

        # 1. Pre-Check
        bad, reason = check_pose_forces(
            self.env.simulation, 
            self.env.all_ligand_indices
        )
        if bad:
            print(f"[SKIP (pre-check)]: {reason}")
            return None
        
        
        st = self.env.simulation.context.getState(getPositions=True)
        current_pos_nm = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        intended_lig_nm = current_pos_nm[self.env.all_ligand_indices]

        
        if do_biased_md and md_ps > 0:
            print(f"Running Biased MD ({md_ps} ps)...")
            run_biased_md_burst(
                simulation=self.env.simulation,
                ligand_indices=self.env.all_ligand_indices,
                intended_lig_nm=intended_lig_nm,
                md_ps=md_ps,
                dt_ps=dt_ps,
                md_temp_K=md_temp_K,
                md_report_ps=0.0 # Change to >0 if you want the logs
            )

        
        print("Minimizing Energy...")
        self.env.simulation.minimizeEnergy(maxIterations=max_iters)

        
        state = self.env.simulation.context.getState(getPositions=True, getEnergy=True)
        final_pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

        
        self.env.complex_structure.positions = final_pos * unit.angstrom

        print(f"Finished in {time.perf_counter() - t_start:.2f}s | Final E: {energy:.2f} kcal/mol")
        return energy


'''
@dataclass
class AnnealingConfig:
    """Configuration for simulated annealing schedule matching the paper protocol."""
    minimise_before: bool = True
    minimise_after: bool = True
    base_T: float = 300.0          # Starting and ending temperature
    heat_to_K: float = 315.0       # Max temperature
    temp_step_K: float = 1.0       # "temperature steps of 1K"
    md_steps_per_T: int = 1000     # "1000 steps at each temperature step"
    high_hold_ps: float = 0.0      # Optional hold at max temp (0 in paper)
    cycles: int = 4                # "4 by default"


class SimulatedAnnealingInPlace:
    
    def __init__(self, env):
        self.env = env 
        self._initial_pos_nm = None
    
    def run(self, config=None):
        if config is None:
            config = AnnealingConfig()
        t_start = time.perf_counter()
        print('\nStarting Simulated Annealing In-Place...')
        
        # 1. Capture the exact starting coordinates
        st_initial = self.env.simulation.context.getState(getPositions=True)
        self._initial_pos_nm = st_initial.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        
        if config.minimise_before:
            print("Running pre-minimization...")
            self.env.simulation.minimizeEnergy()
            
            
        for cycle in range(config.cycles):
            print(f"\n--- Starting Cycle {cycle + 1} of {config.cycles} ---")
            self.cycle(config)
            
        
        if config.minimise_after:
            print("\nRunning final energy minimization...")
            self.env.simulation.minimizeEnergy()
            
        
        
        state = self.env.simulation.context.getState(getPositions=True, getEnergy=True)
        final_pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        self.env.complex_structure.positions = final_pos * unit.angstrom

        print(f"Finished in {time.perf_counter() - t_start:.2f}s | Final E: {energy:.2f} kcal/mol")
        
        
        return energy
        
    def cycle(self, config):
        # RAMP UP
        ramp_up_temps = np.arange(
            config.base_T + config.temp_step_K,
            config.heat_to_K + config.temp_step_K,
            config.temp_step_K
        )
        for t in ramp_up_temps:
            self._set_T(float(t))
            self.env.simulation.step(config.md_steps_per_T)
        
        # HOLD
        if config.high_hold_ps > 0:
            hold_steps = int(config.high_hold_ps / 0.001)
            self.env.simulation.step(hold_steps)
        
        # RAMP DOWN
        ramp_down_temps = np.arange(
            config.heat_to_K - config.temp_step_K,
            config.base_T - config.temp_step_K,
            -config.temp_step_K
        )
        for t in ramp_down_temps:
            self._set_T(float(t))
            self.env.simulation.step(config.md_steps_per_T)
        
    def get_energy(self):
        state = self.env.simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        
    def _set_T(self, k):
        self.env.integrator.setTemperature(k * unit.kelvin)
'''


@dataclass
class AnnealingConfig:
    """Configuration for simulated annealing with ligand COM tether."""
    minimise_before: bool = True
    minimise_after: bool = True

    base_T: float = 50.0
    heat_to_K: float = 150.0
    temp_step_K: float = 5.0
    md_steps_per_T: int = 250
    high_hold_ps: float = 0.0
    cycles: int = 1

    seed: int = 1

    # COM tether settings
    use_com_tether: bool = True
    com_tether_r0_A: float = 0.25
    com_tether_k_kcal_per_mol_A2: float = 10.0
    com_tether_alpha_per_nm: float = 80.0

    # Monitoring only
    report_every_cycle: bool = True
    debug_stage_drifts: bool = True


class SimulatedAnnealingInPlace:

    def __init__(self, env):
        self.env = env
        self._initial_pos_nm = None

    def run(self, config=None):
        if config is None:
            config = AnnealingConfig()

        t_start = time.perf_counter()
        print("\nStarting Simulated Annealing In-Place...")

        bad, reason = check_pose_forces(
            self.env.simulation,
            self.env.all_ligand_indices
        )
        if bad:
            print(f"[SKIP (pre-check)]: {reason}")
            return None

        # Reference must be captured BEFORE any minimisation
        st_initial = self.env.simulation.context.getState(getPositions=True)
        self._initial_pos_nm = (
            st_initial.getPositions(asNumpy=True)
            .value_in_unit(unit.nanometer)
            .copy()
        )

        ref_pos_nm = self._initial_pos_nm.copy()
        ref_com_nm = np.asarray(self.env.get_ligand_com_nm(ref_pos_nm), dtype=float)

        def _state_pos_energy():
            st = self.env.simulation.context.getState(getPositions=True, getEnergy=True)
            pos_nm = np.asarray(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=float)
            energy = float(st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
            return pos_nm, energy

        def _com_drift_A(pos_nm):
            lig_com_nm = np.asarray(self.env.get_ligand_com_nm(pos_nm), dtype=float)
            return float(np.linalg.norm(lig_com_nm - ref_com_nm) * 10.0)

        def _print_stage(label):
            if not config.debug_stage_drifts:
                return
            pos_nm, energy = _state_pos_energy()
            drift_A = _com_drift_A(pos_nm)
            print(f"[debug] {label}: E={energy:.2f} kcal/mol | ligand COM drift={drift_A:.3f} Å")

        try:
            # Turn tether on from the very start
            if config.use_com_tether:
                self.env.set_ligand_com_tether(
                    ref_pos_nm,
                    r0_A=config.com_tether_r0_A,
                    k_kcal_per_mol_A2=config.com_tether_k_kcal_per_mol_A2,
                    alpha_per_nm=config.com_tether_alpha_per_nm,
                )

            _print_stage("after tether set / initial reference")

            if config.minimise_before:
                print("Running pre-minimization...")
                self.env.simulation.minimizeEnergy()
                _print_stage("after pre-minimization")

            self.env.integrator.setTemperature(config.base_T * unit.kelvin)
            self.env.simulation.context.setVelocitiesToTemperature(
                config.base_T * unit.kelvin,
                config.seed
            )

            for cycle_idx in range(config.cycles):
                print(f"\n--- Starting Cycle {cycle_idx + 1} of {config.cycles} ---")
                self.cycle(config)

                pos_nm, energy = _state_pos_energy()
                com_drift_A = _com_drift_A(pos_nm)

                bad, reason = check_pose_forces(
                    self.env.simulation,
                    self.env.all_ligand_indices
                )
                if bad:
                    print(f"[cycle {cycle_idx + 1}] Bad forces detected: {reason}")
                    return None

                if config.report_every_cycle:
                    print(
                        f"Cycle {cycle_idx + 1}: "
                        f"E={energy:.2f} kcal/mol | "
                        f"ligand COM drift={com_drift_A:.3f} Å"
                    )

            _print_stage("before final minimization")

            if config.minimise_after:
                print("\nRunning final energy minimization...")
                self.env.simulation.minimizeEnergy()
                _print_stage("after final minimization")

            state = self.env.simulation.context.getState(getPositions=True, getEnergy=True)
            final_pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

            self.env.complex_structure.positions = final_pos * unit.angstrom

            final_pos_nm = np.asarray(final_pos, dtype=float) / 10.0
            final_com_drift_A = _com_drift_A(final_pos_nm)

            print(
                f"Finished in {time.perf_counter() - t_start:.2f}s | "
                f"Final E: {energy:.2f} kcal/mol | "
                f"Final COM drift: {final_com_drift_A:.3f} Å"
            )
            return energy

        finally:
            try:
                self.env.clear_ligand_com_tether()
            except Exception:
                pass

    def cycle(self, config):
        ramp_up_temps = np.arange(
            config.base_T + config.temp_step_K,
            config.heat_to_K + config.temp_step_K,
            config.temp_step_K
        )
        for T in ramp_up_temps:
            self._set_T(float(T))
            self.env.simulation.step(config.md_steps_per_T)

        if config.high_hold_ps > 0:
            dt_ps = self.env.integrator.getStepSize().value_in_unit(unit.picoseconds)
            hold_steps = int(round(config.high_hold_ps / dt_ps))
            if hold_steps > 0:
                self.env.simulation.step(hold_steps)

        ramp_down_temps = np.arange(
            config.heat_to_K - config.temp_step_K,
            config.base_T - config.temp_step_K,
            -config.temp_step_K
        )
        for T in ramp_down_temps:
            self._set_T(float(T))
            self.env.simulation.step(config.md_steps_per_T)

    def get_energy(self):
        state = self.env.simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    def _set_T(self, temperature_K: float):
        self.env.integrator.setTemperature(temperature_K * unit.kelvin)