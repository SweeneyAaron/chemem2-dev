# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from scipy import ndimage
from openmm import (
    CustomCompoundBondForce,
    CustomExternalForce,
    CustomBondForce,
    CustomCVForce,
    RMSDForce,
    Continuous3DFunction,
    unit,
    LangevinIntegrator,
    app, 
    Platform
)
from scipy.spatial import cKDTree
from .biomolecule import find_atoms_outside_ligand


def add_sse_rmsd_forces(
    complex_system,
    complex_structure,
    ref_nm,
    sse_groups,
    ligand_set,
    ligand_heavy_indices,
    sse_k: float,
    localise: bool = False
):
    """
    Filters SSE groups, builds RMSD restraints, and adds them to the system.
    """
    if not sse_groups: 
        raise ValueError("SSE mode requires sse_groups")
        
    for grp in sse_groups:
        # Filter out ligand atoms
        valid_grp = [i for i in grp if i not in ligand_set]
        
        # Filter out atoms too far from the ligand (if localise is True)
        if localise:
            outside = find_atoms_outside_ligand(
                complex_structure, ligand_heavy_indices
            )
            valid_grp = [i for i in valid_grp if i not in outside]
        
        # Build and add the force
        rmsd_force = ForceBuilder.create_rmsd_restraint(ref_nm, valid_grp, sse_k)
        if rmsd_force:
            complex_system.addForce(rmsd_force)

def add_density_map_force(complex_system, 
                          complex_structure,
                          density_map, 
                          global_k, 
                          padding, 
                          smooth_sigma_vox=0.0):
    
        print("Processing Density Map...")
        print('-- Global k:', global_k)
        map_force = ForceBuilder.create_map_potential(
            density_map, 
            global_k,
            smooth_sigma_vox=smooth_sigma_vox
        )
        for atom in complex_structure.atoms:
            if atom.element != 1:
                map_force.addBond([atom.idx])
        complex_system.addForce(map_force)

class ForceBuilder:
    """Helper class to construct OpenMM Force objects."""

    @staticmethod
    def create_continuous_3d_function(density_map_obj, blur=0.0, normalise= False):
        """
        Builds a Continuous3DFunction from a density map object safely.
        """
        vol_zyx = density_map_obj.density_map.astype(np.float32, copy=False)
        
        if blur and float(blur) > 0.0:
            vol_zyx = ndimage.gaussian_filter(vol_zyx, float(blur))

        # Thresholding if map_contour exists
        c_level = getattr(density_map_obj, "map_contour", None)
        if c_level is not None:
             vol_zyx = vol_zyx * (vol_zyx >= float(c_level))
        
        if normalise:
            vmax = float(np.max(np.abs(vol_zyx))) or 1.0
            vol_zyx = (vol_zyx / vmax).astype(np.float32, copy=False)

        # Flatten in C-order (z,y,x) -> (x fastest)
        vol_c = np.ascontiguousarray(vol_zyx)
        nz, ny, nx = vol_c.shape
        values = vol_c.ravel(order="C")
        
        
        
        # Coordinate bounds (Angstrom to Nanometer)
        ox, oy, oz = map(float, density_map_obj.origin)
        ax, ay, az = map(float, density_map_obj.apix)
        A2NM = 0.1
        
        xmin = (ox - 0.5 * ax) * A2NM; xmax = (ox + nx * ax - 0.5 * ax) * A2NM
        ymin = (oy - 0.5 * ay) * A2NM; ymax = (oy + ny * ay - 0.5 * ay) * A2NM
        zmin = (oz - 0.5 * az) * A2NM; zmax = (oz + nz * az - 0.5 * az) * A2NM

        return Continuous3DFunction(
            int(nx), int(ny), int(nz), values,
            float(xmin), float(xmax),
            float(ymin), float(ymax),
            float(zmin), float(zmax)
        )

    @staticmethod
    def create_map_potential(density_map_obj,
                             global_k: float,
                             smooth_sigma_vox: float = 0.0,
                             smooth_sigma_A: float = 0.0,
                             normalise: bool = True,
                             force_group: int = 7):
        
        """Creates the grid-based map potential force."""
        
        blur_vox = 0.0
        if smooth_sigma_A and float(smooth_sigma_A) > 0.0:
            ax, ay, az = map(float, density_map_obj.apix)
            # Convert Angstrom to voxel sigma per axis, then average for a single sigma.
            # (ndimage.gaussian_filter supports per-axis sigmas too; see note below.)
            sig_x = float(smooth_sigma_A) / max(ax, 1e-8)
            sig_y = float(smooth_sigma_A) / max(ay, 1e-8)
            sig_z = float(smooth_sigma_A) / max(az, 1e-8)
            blur_vox = (sig_x + sig_y + sig_z) / 3.0
        elif smooth_sigma_vox and float(smooth_sigma_vox) > 0.0:
            blur_vox = float(smooth_sigma_vox)
        
        
        func = ForceBuilder.create_continuous_3d_function(density_map_obj, 
                                                          blur=blur_vox,
                                                          normalise= normalise)
        cf = CustomCompoundBondForce(1, "")
        cf.addTabulatedFunction("map_potential", func)
        cf.addGlobalParameter("global_k", float(global_k))
        # Natural coordinate order x, y, z
        cf.setEnergyFunction("-global_k * map_potential(x1, y1, z1)")
        cf.setForceGroup(force_group)
        return cf

    @staticmethod
    def create_positional_pin(atom_indices, ref_positions_nm, k_name="k_pin"):
        """Creates a CustomExternalForce for pinning atoms."""
        expr = f"{k_name}*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
        f = CustomExternalForce(expr)
        f.addGlobalParameter(k_name, 0.0 * unit.kilojoule_per_mole / unit.nanometer**2)
        f.addPerParticleParameter("x0")
        f.addPerParticleParameter("y0")
        f.addPerParticleParameter("z0")

        for i in atom_indices:
            x0, y0, z0 = map(float, ref_positions_nm[int(i)])
            f.addParticle(int(i), [x0 * unit.nanometer, y0 * unit.nanometer, z0 * unit.nanometer])
        
        return f

    @staticmethod
    def create_rmsd_restraint(ref_positions_nm, atom_indices, k_rmsd, mode="spoke", flat_bottom_A=0.0):
        """
        Creates an ENM-like restraint (Elastic Network Model) or RMSD restraint.
        modes: 'spoke' (minimal edges), 'dense' (cutoff based).
        """
        ids = list(dict.fromkeys(atom_indices))
        if len(ids) < 3:
            return None

        ref_nm = np.asarray(ref_positions_nm, dtype=float)
        
        # Select anchors (farthest pair + max area third point)
        coords = ref_nm[ids]
        dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        i, j = np.unravel_index(np.argmax(dmat), dmat.shape)
        a, b = ids[int(i)], ids[int(j)]
        
        # Find C
        ab = ref_nm[b] - ref_nm[a]
        best, c = -1.0, None
        for k in ids:
            if k in (a, b): continue
            area2 = np.linalg.norm(np.cross(ab, ref_nm[k] - ref_nm[a]))
            if area2 > best:
                best, c = area2, k
        if c is None: 
            c = next(x for x in ids if x not in (a, b))

        # Force Expression
        if flat_bottom_A > 0.0:
            expr = "step(abs(r-r0)-delta) * k * (abs(r-r0)-delta)^2"
        else:
            expr = "k*(r-r0)^2"

        cf = CustomBondForce(expr)
        cf.addGlobalParameter("k", float(k_rmsd) * unit.kilojoule_per_mole / unit.nanometer**2)
        cf.addPerBondParameter("r0")
        
        if flat_bottom_A > 0.0:
            cf.addGlobalParameter("delta", (flat_bottom_A * 0.1) * unit.nanometer)

        def _add(i_idx, j_idx):
            r0 = float(np.linalg.norm(ref_nm[i_idx] - ref_nm[j_idx]))
            cf.addBond(int(i_idx), int(j_idx), [r0 * unit.nanometer])

        # Base triangle
        _add(a, b); _add(a, c); _add(b, c)

        if mode == "spoke":
            for d in ids:
                if d not in (a, b, c):
                    _add(a, d); _add(b, d); _add(c, d)
        elif mode == "dense":
            cutoff_nm = 0.2 # 2.0 Angstrom default
            tree = cKDTree(ref_nm[ids])
            pairs = tree.query_pairs(cutoff_nm, output_type="ndarray")
            for ii, jj in pairs:
                _add(ids[ii], ids[jj])
        
        return cf

    @staticmethod
    def _normalize_geom_name(name: str) -> str:
        n = " ".join(str(name).strip().lower().split())
        return {
            "triganal planer": "trigonal planar",
            "square pyrimidal": "square pyramidal",
            "pentagonal bipyrimidal": "pentagonal bipyramidal",
        }.get(n, n)

    @staticmethod
    def _coordination_template_dirs(geometry: str) -> np.ndarray:
        """
        Returns (CN,3) unit vectors for canonical coordination site directions.
        IMPORTANT: the ORDER of these sites defines the angle targets.
        """
        g = ForceBuilder._normalize_geom_name(geometry)
        s3 = np.sqrt(3.0)

        if g == "octahedral":
            V = np.array([[ 1,0,0],[-1,0,0],[0, 1,0],[0,-1,0],[0,0, 1],[0,0,-1]], float)
        elif g == "square planar":
            V = np.array([[ 1,0,0],[-1,0,0],[0, 1,0],[0,-1,0]], float)
        elif g == "linear":
            V = np.array([[0,0, 1],[0,0,-1]], float)
        elif g == "trigonal planar":
            V = np.array([[1,0,0],[-0.5,  0.5*s3,0],[-0.5, -0.5*s3,0]], float)
        elif g == "tetrahedral":
            V = np.array([[ 1,  1,  1],
                          [ 1, -1, -1],
                          [-1,  1, -1],
                          [-1, -1,  1]], float)
        elif g == "trigonal bipyramidal":
            V = np.array([[0,0, 1],[0,0,-1],[1,0,0],[-0.5,  0.5*s3,0],[-0.5, -0.5*s3,0]], float)
        elif g == "square pyramidal":
            V = np.array([[ 1,0,0],[-1,0,0],[0, 1,0],[0,-1,0],[0,0,1]], float)
        elif g == "pentagonal bipyramidal":
            eq = []
            for k in range(5):
                ang = np.deg2rad(72.0*k)
                eq.append([np.cos(ang), np.sin(ang), 0.0])
            V = np.array([[0,0, 1],[0,0,-1]] + eq, float)
        else:
            raise ValueError(f"Unknown coordination geometry: {geometry!r} (normalized {g!r})")

        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        return V

    @staticmethod
    def create_ion_coordination_restraints(
        ion_idx: int,
        coord_atom_indices: Sequence[int],
        *,
        # Distances
        target_dist_A: float | Sequence[float],
        flat_bottom_A: float = 0.2,
        k_dist_name: str = "k_ion_dist",
        # Angles
        include_angles: bool = False,
        geometry: Optional[str] = None,
        angle_pairs: str = "all",  # "all" is simplest/most robust
        k_ang_name: str = "k_ion_ang",
        # Force groups
        force_group_dist: int = 20,
        force_group_ang: int = 21,
    ) -> Dict[str, Any]:
        """
        Build OpenMM forces for metal coordination.

        Parameters
        ----------
        ion_idx
            Atom index of the ion.
        coord_atom_indices
            Atom indices of coordinating atoms (donors + dummy water O),
            ORDERED by coordination site index (0..CN-1).
            If you used site_to_atom mapping: [site_to_atom[i] for i in range(CN)].
        target_dist_A
            Either a single distance in Å applied to all partners, or a list/tuple
            of length len(coord_atom_indices) for per-partner distances.
        flat_bottom_A
            Half-width (Å) of the flat-bottom region for distance restraints.
            Set to 0 to use a simple harmonic.
        include_angles
            If True, adds partner–ion–partner angle restraints.
        geometry
            Required if include_angles=True. One of your geometry strings.
        angle_pairs
            "all" restrains every pair (recommended initially).
        k_dist_name, k_ang_name
            Names of global parameters so you can stage/anneal them via context.
        force_group_dist, force_group_ang
            Force groups for debugging.

        Returns
        -------
        dict with keys:
            - "distance_force": CustomBondForce
            - "angle_force": CustomCompoundBondForce or None
            - "k_dist_name": str
            - "k_ang_name": str
            - "cn": int
        """
        ion_idx = int(ion_idx)
        coord_atom_indices = [int(i) for i in coord_atom_indices]
        cn = len(coord_atom_indices)

        # --- target distances (nm) ---
        if isinstance(target_dist_A, (list, tuple, np.ndarray)):
            if len(target_dist_A) != cn:
                raise ValueError(f"target_dist_A must have length {cn}, got {len(target_dist_A)}")
            r0_nm_list = [float(r) * 0.1 for r in target_dist_A]
        else:
            r0_nm_list = [float(target_dist_A) * 0.1] * cn

        # --- distance force ---
        if flat_bottom_A and float(flat_bottom_A) > 0.0:
            expr = "step(abs(r-r0)-delta) * {k} * (abs(r-r0)-delta)^2".format(k=k_dist_name)
        else:
            expr = f"{k_dist_name}*(r-r0)^2"

        dist = CustomBondForce(expr)
        dist.addGlobalParameter(
            k_dist_name, 0.0 * unit.kilojoule_per_mole / unit.nanometer**2
        )
        dist.addPerBondParameter("r0")
        if flat_bottom_A and float(flat_bottom_A) > 0.0:
            dist.addGlobalParameter("delta", float(flat_bottom_A) * 0.1 * unit.nanometer)

        for aidx, r0_nm in zip(coord_atom_indices, r0_nm_list):
            dist.addBond(ion_idx, int(aidx), [r0_nm * unit.nanometer])

        dist.setForceGroup(int(force_group_dist))

        # --- optional angle force ---
        ang = None
        if include_angles:
            if geometry is None:
                raise ValueError("geometry must be provided when include_angles=True")
        
            dirs = ForceBuilder._coordination_template_dirs(geometry)
            if dirs.shape[0] != cn:
                raise ValueError(
                    f"Geometry {geometry!r} implies CN={dirs.shape[0]} but you gave {cn} coord atoms."
                )
        
            if angle_pairs != "all":
                raise ValueError("Only angle_pairs='all' is implemented right now (safe default).")
        
            ang = CustomCompoundBondForce(
                3,
                f"{k_ang_name}*(angle(p1,p2,p3)-theta0)^2"
            )
            ang.addGlobalParameter(
                k_ang_name, 0.0 * unit.kilojoule_per_mole / unit.radian**2
            )
            ang.addPerBondParameter("theta0")
        
            for i in range(cn):
                for j in range(i + 1, cn):
                    cos_t = float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0))
                    theta0 = float(np.arccos(cos_t))
                    ang.addBond(
                        [int(coord_atom_indices[i]), int(ion_idx), int(coord_atom_indices[j])],
                        [theta0 * unit.radian],
                    )
        
            ang.setForceGroup(int(force_group_ang))

        return {
            "distance_force": dist,
            "angle_force": ang,
            "k_dist_name": k_dist_name,
            "k_ang_name": k_ang_name,
            "cn": cn,
        }


def get_periodic_torsion_force(system):
    for force in system.getForces():
        if type(force).__name__ == 'PeriodicTorsionForce':
             return force,  force.getForceGroup() 

def get_force_energy(force, simulation):
    #state = simulation.context.getState(getEnergy=True)
    state = simulation.context.getState(getEnergy=True, groups={ force.getForceGroup() })
    energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    return energy


def export_torsion_profile(ligand,
                           torsion_lists,
                           platform = 'OpenCL',
                           output = './',
                           normalise = True,
                           write = False
                           ):
    
    integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picoseconds, 2*unit.femtoseconds)
    _platform = Platform.getPlatformByName(platform)
    system = ligand.complex_structure.createSystem()
    
    
    
    simulation = app.Simulation(ligand.complex_structure.topology, 
                            system, integrator, _platform)
    
    simulation.context.setPositions(ligand.complex_structure.positions)
    force_group = 0
    for force in system.getForces():
        force.setForceGroup(force_group)
        force_group += 1
        
    torsion_force, force_index = get_periodic_torsion_force(system)
    lig_copy = Chem.AddHs(ligand.mol, addCoords = True)
    all_torsion_profiles = []
    for torsion in torsion_lists:
        
        torsion_profile = []
        a1,a2,a3,a4 = torsion 
        
        atom_names = []
        
        for atom in ligand.complex_structure.atoms:
            if atom.idx == a1:
                a1_atom = atom 
            elif atom.idx == a2:
                a2_atom = atom 
            elif atom.idx == a3:
                a3_atom = atom 
            elif atom.idx == a4:
                a4_atom = atom
            
            
        for angle in range(0,360,1):
            
            rdMolTransforms.SetDihedralDeg(lig_copy.GetConformer(), a1,a2,a3,a4, angle)
            new_angle = rdMolTransforms.GetDihedralDeg(lig_copy.GetConformer(), a1,a2,a3,a4)
            positions = lig_copy.GetConformer().GetPositions() * unit.angstrom
            simulation.context.setPositions(positions)
            force_energy = get_force_energy(torsion_force, simulation)
            torsion_profile.append((int(new_angle), force_energy))
        
            
        #plt.plot([i[0] for i in torsion_profile], [i[1] for i in torsion_profile])
        
        
        min_energy = min([i[1] for i in torsion_profile])
        torsion_profile = [(round(i[0],3) , i[1] - min_energy ) for i in torsion_profile]
        max_energy = max([i[1] for i in torsion_profile])
        
        torsion_profile = [(round(i[0],3) , i[1] / max_energy ) for i in torsion_profile]
        
        # Step 3: Rescale to -1 to 1
        #torsion_profile = [(round(i[0]), 2 * i[1] - 1) for i in torsion_profile]
        
        all_torsion_profiles.append([[a1_atom.name, a2_atom.name, a3_atom.name, a4_atom.name ],
                                     [a1_atom.idx, a2_atom.idx, a3_atom.idx,a4_atom.idx],
                                     torsion_profile])
    
    
    return all_torsion_profiles
