# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

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


class ForceBuilder:
    """Helper class to construct OpenMM Force objects."""

    @staticmethod
    def create_continuous_3d_function(density_map_obj, blur=0.0):
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
    def create_map_potential(density_map_obj, global_k: float, force_group: int = 7):
        """Creates the grid-based map potential force."""
        func = ForceBuilder.create_continuous_3d_function(density_map_obj)
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
