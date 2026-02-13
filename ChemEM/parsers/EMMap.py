# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

import copy
import numpy as np
import mrcfile

from scipy.fftpack import fftn, ifftn
from scipy.ndimage import fourier_gaussian

class EMMap:
    def __init__(self, origin, apix, density_map, resolution):
        self.origin = origin              # (x,y,z) Å
        self.apix = apix                  # (ax,ay,az) Å/px
        self.density_map = density_map    # np.ndarray
        self.resolution = resolution
        self.map_contour = 0.0

    # ----------------------------
    # Geometry / convenience props
    # ----------------------------
    @property
    def box_size(self):
        # (z,y,x)
        return self.density_map.shape

    @property
    def x_size(self):
        return self.box_size[2]

    @property
    def y_size(self):
        return self.box_size[1]

    @property
    def z_size(self):
        return self.box_size[0]

    @property
    def x_origin(self):
        return self.origin[0]

    @property
    def y_origin(self):
        return self.origin[1]

    @property
    def z_origin(self):
        return self.origin[2]

    @property
    def std(self):
        return np.std(self.density_map)

    @property
    def mean(self):
        return np.mean(self.density_map)

    # ----------------------------
    # Core methods
    # ----------------------------
    def voxel_to_point(self, x, y, z):
        """Voxel (x,y,z) -> world (x,y,z) in Å."""
        real_world_x = self.origin[0] + x * self.apix[0]
        real_world_y = self.origin[1] + y * self.apix[1]
        real_world_z = self.origin[2] + z * self.apix[2]
        return (real_world_x, real_world_y, real_world_z)

    def normalise(self):
        if self.std != 0:
            self.density_map = (self.density_map - self.mean) / self.std

    def set_map_contour(self):
        self.map_contour = 0.0
        # self.map_contour = MapTools.map_contour(self, t=3.0)

    def flatten(self):
        return self.density_map.flatten().tolist()

    def center_of_mass(self):
        """
        Center of mass of non-zero voxels (weighted by density) in world Å.
        """
        if not isinstance(self.density_map, np.ndarray) or self.density_map.ndim != 3:
            raise ValueError("density_map must be a 3D NumPy array.")

        indices = np.array(np.nonzero(self.density_map)).T  # (N,3) in (z,y,x)
        if indices.size == 0:
            raise ValueError("density_map contains no non-zero voxels.")

        weights = self.density_map[indices[:, 0], indices[:, 1], indices[:, 2]]
        com_grid_zyx = np.average(indices, axis=0, weights=weights)  # z,y,x
        com_grid_xyz = com_grid_zyx[::-1]  # x,y,z

        com_real = np.array(self.origin, dtype=float) + np.array(self.apix, dtype=float) * com_grid_xyz
        return tuple(com_real)
    
    def submap(self, origin, box_size, *, pad_value=0.0):
        """
        Return a new EMMap that is a sub-volume of this map.

        Parameters
        ----------
        origin : (x,y,z)
            World-space origin (Å) of the requested sub-box corner.
        box_size : (bz,by,bx)
            Requested size in voxels in (z,y,x) order (same as EMMap.box_size).
        pad_value : float
            Fill value if requested box extends outside the map.

        Returns
        -------
        EMMap
            New EMMap with:
              - origin = requested origin (Å)
              - apix = same as parent
              - resolution = same as parent
              - density_map shape = (bz,by,bx) in (z,y,x)
        """
        if not isinstance(self.density_map, np.ndarray) or self.density_map.ndim != 3:
            raise ValueError("density_map must be a 3D NumPy array.")
        if len(origin) != 3:
            raise ValueError("origin must be length-3 (x,y,z) in Å.")
        if len(box_size) != 3:
            raise ValueError("box_size must be length-3 (bz,by,bx) in voxels (z,y,x).")

        bz, by, bx = (int(box_size[0]), int(box_size[1]), int(box_size[2]))
        if bx <= 0 or by <= 0 or bz <= 0:
            raise ValueError("box_size entries must be positive integers.")

        origin = np.asarray(origin, dtype=float)
        map_origin = np.asarray(self.origin, dtype=float)
        apix = np.asarray(self.apix, dtype=float)

        # world -> voxel corner indices (x,y,z)
        start_xyz = np.rint((origin - map_origin) / apix).astype(int)
        sx, sy, sz = int(start_xyz[0]), int(start_xyz[1]), int(start_xyz[2])

        # requested voxel ranges in voxel coords (x,y,z)
        ex, ey, ez = sx + bx, sy + by, sz + bz

        nx, ny, nz = self.x_size, self.y_size, self.z_size  # sizes in voxel coords (x,y,z)

        # overlap with map bounds
        ix0, ix1 = max(0, sx), min(nx, ex)
        iy0, iy1 = max(0, sy), min(ny, ey)
        iz0, iz1 = max(0, sz), min(nz, ez)

        sub = np.full((bz, by, bx), pad_value, dtype=self.density_map.dtype)

        if ix1 > ix0 and iy1 > iy0 and iz1 > iz0:
            # where to paste in output (x,y,z offsets)
            ox0 = ix0 - sx
            oy0 = iy0 - sy
            oz0 = iz0 - sz

            ox1 = ox0 + (ix1 - ix0)
            oy1 = oy0 + (iy1 - iy0)
            oz1 = oz0 + (iz1 - iz0)

            # src/dst are both (z,y,x)
            sub[oz0:oz1, oy0:oy1, ox0:ox1] = self.density_map[iz0:iz1, iy0:iy1, ix0:ix1]

        return EMMap(tuple(origin.tolist()), tuple(float(a) for a in self.apix), sub, self.resolution)

    def write_mrc(self, outfile):
        if not outfile.endswith(".mrc"):
            outfile += ".mrc"

        data = self.density_map.astype("float32", copy=False)
        with mrcfile.new(outfile, overwrite=True) as mrc:
            mrc.set_data(data)

            mrc.header.nxstart = 0
            mrc.header.nystart = 0
            mrc.header.nzstart = 0

            mrc.header.mx = self.x_size
            mrc.header.my = self.y_size
            mrc.header.mz = self.z_size

            mrc.header.mapc = 1
            mrc.header.mapr = 2
            mrc.header.maps = 3

            mrc.header.cellb.alpha = 90
            mrc.header.cellb.beta = 90
            mrc.header.cellb.gamma = 90

            mrc.header.origin.x = float(self.origin[0])
            mrc.header.origin.y = float(self.origin[1])
            mrc.header.origin.z = float(self.origin[2])

            # voxel_size is stored as x,y,z
            mrc.voxel_size = tuple(float(x) for x in self.apix)

            # Optional header extras if you keep them on the instance
            if hasattr(self, "ispg"):
                mrc.header.ispg = self.ispg
            if hasattr(self, "extra1"):
                mrc.header.extra1 = self.extra1
            if hasattr(self, "extra2"):
                mrc.header.extra2 = self.extra2
            if hasattr(self, "exttyp"):
                mrc.header.exttyp = self.exttyp
            if hasattr(self, "extended_header"):
                mrc.set_extended_header(self.extended_header)

    def copy(self):
        return copy.deepcopy(self)

    # ----------------------------
    # Constructors
    # ----------------------------
    @classmethod
    def from_mrc(cls, filename, resolution=0.0):
        with mrcfile.open(filename, mode="r") as mrc:
            data = np.array(mrc.data, copy=True)
            origin = (float(mrc.header.origin.x), float(mrc.header.origin.y), float(mrc.header.origin.z))
            apix = mrc.voxel_size
            apix = (float(apix.x), float(apix.y), float(apix.z))
        return cls(origin, apix, data, resolution)

    @classmethod
    def from_model(cls, mol, resolution, origin, apix, box_size, sigma_coeff=0.356, normalise=True):
        """
        Create an EMMap by blurring an RDKit model using Fourier-space Gaussian blur.
        """
        data = np.zeros(box_size, dtype=float)
        emap = cls(origin, apix, data, resolution)

        # Prepare atomic input
        prot = _prepare_molmap_input(mol)
        overlay_map = _make_atom_overlay_map(emap, prot)

        sigma = sigma_coeff * resolution
        fou = fourier_gaussian(fftn(overlay_map.density_map), sigma)
        emap.density_map = np.real(ifftn(fou))

        if normalise:
            emap.normalise()

        return emap

def _prepare_molmap_input(mol, n_conf: int = 0):
    molmap_input = []
    atm_coords = mol.GetConformers()[n_conf].GetPositions()
    atom_props = mol.GetAtoms()

    for coord, prop in zip(atm_coords, atom_props):
        x = float(coord[0])
        y = float(coord[1])
        z = float(coord[2])
        atm_type = prop.GetSymbol()
        mass = float(prop.GetMass())
        atom_input = [atm_type, x, y, z, mass]
        molmap_input.append(atom_input)
    return molmap_input


def _make_atom_overlay_map(densMap, prot):
    densMap = densMap.copy()
    for atom in prot:
        pos = _map_grid_position(densMap, atom)
        if pos:
            densMap.density_map[pos[2]][pos[1]][pos[0]] += pos[3]
    return densMap

def _map_grid_position(densMap, atom):
    origin = densMap.origin
    apix = _as_xyz(densMap.apix)
    box_size = densMap.box_size

    x_pos = int(round((atom[1] - origin[0]) / apix[0], 0))
    y_pos = int(round((atom[2] - origin[1]) / apix[1], 0))
    z_pos = int(round((atom[3] - origin[2]) / apix[2], 0))

    if (
        (box_size[2] > x_pos >= 0)
        and (box_size[1] > y_pos >= 0)
        and (box_size[0] > z_pos >= 0)
    ):
        return x_pos, y_pos, z_pos, atom[4]
    return 0

def _as_xyz(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.size == 1:
        return np.array([float(v), float(v), float(v)], dtype=np.float64)
    return v

