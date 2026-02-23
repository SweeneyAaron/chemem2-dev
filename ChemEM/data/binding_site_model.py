# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from ChemEM.parsers.EMMap import EMMap

def _np3(v, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if v is None:
        return np.asarray(default, dtype=float)
    a = np.asarray(v, dtype=float)
    if a.shape != (3,):
        raise ValueError(f"Expected shape (3,), got {a.shape}")
    return a


@dataclass
class BindingSiteModel:
    """
    Canonical representation of a binding site produced by BindingSiteProtocol.
    This replaces the loose dict used previously.
    """
    key: int = 0
    source: str = "auto"  # "auto" or "manual"

    residues: List[Any] = field(default_factory=list)
    lining_residues: List[Any] = field(default_factory=list)
    tetrahedrals: Any = None

    unique_atom_indices: List[int] = field(default_factory=list)

    binding_site_centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    min_coords: np.ndarray = field(default_factory=lambda: np.zeros(3))
    max_coords: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bounding_box_size: np.ndarray = field(default_factory=lambda: np.zeros(3))

    site_centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    site_radii: np.ndarray = field(default_factory=lambda: np.zeros((0,)))

    volume: float = 0.0

    # grid/meta
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    apix: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    box_size: Optional[Tuple[int, int, int]] = None  # (z,y,x)

    densmap: Optional[np.ndarray] = None
    distance_map: Optional[np.ndarray] = None

    # optional RDKit views
    rdkit_mol: Any = None
    rdkit_lining_mol: Any = None

    # opening points (optional)
    openings: Optional[List[np.ndarray]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BindingSiteModel":
        obj = cls()
        for k, v in d.items():
            if not hasattr(obj, k):
                continue
            setattr(obj, k, v)

        obj.binding_site_centroid = _np3(obj.binding_site_centroid)
        obj.min_coords = _np3(obj.min_coords)
        obj.max_coords = _np3(obj.max_coords)
        obj.bounding_box_size = _np3(obj.bounding_box_size)

        obj.site_centers = np.asarray(obj.site_centers, dtype=float).reshape((-1, 3))
        obj.site_radii = np.asarray(obj.site_radii, dtype=float).reshape((-1,))
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """For backwards compatibility / JSON output."""
        return {
            "key": self.key,
            "source": self.source,
            "residues": self.residues,
            "lining_residues": self.lining_residues,
            "tetrahedrals": self.tetrahedrals,
            "unique_atom_indices": self.unique_atom_indices,
            "binding_site_centroid": self.binding_site_centroid.tolist(),
            "min_coords": self.min_coords.tolist(),
            "max_coords": self.max_coords.tolist(),
            "bounding_box_size": self.bounding_box_size.tolist(),
            "site_centers": self.site_centers.tolist(),
            "site_radii": self.site_radii.tolist(),
            "volume": float(self.volume),
            "origin": tuple(self.origin),
            "apix": tuple(self.apix),
            "box_size": self.box_size,
            "openings": None if self.openings is None else [o.tolist() for o in self.openings],
        }
    

    def copy(self, *, new_key: Optional[int] = None, copy_grids: bool = False) -> "BindingSiteModel":
        """
        Clone the site. By default, does NOT copy densmap/distance_map arrays (often huge),
        because for splitting you typically only need geometry + residues.
        """
        c = replace(self)  # shallow copy of the dataclass
        if new_key is not None:
            c.key = int(new_key)

        # numpy vectors
        c.binding_site_centroid = np.array(self.binding_site_centroid, dtype=float, copy=True)
        c.min_coords = np.array(self.min_coords, dtype=float, copy=True)
        c.max_coords = np.array(self.max_coords, dtype=float, copy=True)
        c.bounding_box_size = np.array(self.bounding_box_size, dtype=float, copy=True)

        c.site_centers = np.array(self.site_centers, dtype=float, copy=True).reshape((-1, 3))
        c.site_radii = np.array(self.site_radii, dtype=float, copy=True).reshape((-1,))

        # lists
        c.residues = list(self.residues)
        c.lining_residues = list(self.lining_residues)
        c.unique_atom_indices = list(self.unique_atom_indices)

        # grids (optional)
        if copy_grids:
            c.densmap = None if self.densmap is None else np.array(self.densmap, copy=True)
            c.distance_map = None if self.distance_map is None else np.array(self.distance_map, copy=True)
        else:
            c.densmap = None
            c.distance_map = None

        return c
    def write_site(self, file):
        
        site = EMMap(self.origin, self.apix, self.distance_map, 0.0)
        site.write_mrc(file)
        
    def _bbox_extents(self) -> np.ndarray:
        """
        Returns bbox extents in Å as (x, y, z).
        Falls back to max_coords - min_coords if bounding_box_size missing/zero.
        """
        b = np.asarray(self.box_size, dtype=float).reshape(3)
        if np.allclose(b, 0.0):
            b = np.asarray(self.max_coords, dtype=float) - np.asarray(self.min_coords, dtype=float)
        return np.abs(b)

    def _get_pocket_mask(self, dens_threshold: float = 0.0) -> Optional[np.ndarray]:
        """
        Convert densmap to a boolean mask.
        Assumes densmap > dens_threshold means pocket voxels.
        """
        if self.distance_map is None:
            return None
        d = np.asarray(self.distance_map)
        if d.size == 0:
            return None
        # Handles bool masks and float score maps
        if d.dtype == np.bool_:
            return d.copy()
        return np.isfinite(d) & (d > dens_threshold)

    @staticmethod
    def _surface_voxel_count(mask: np.ndarray) -> int:
        """
        Count pocket voxels with at least one 6-neighbour outside the mask.
        mask shape: (z, y, x)
        """
        if mask is None or mask.size == 0:
            return 0
        if not np.any(mask):
            return 0

        p = np.pad(mask.astype(bool), ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=False)
        core = p[1:-1, 1:-1, 1:-1]

        n_zm = p[:-2, 1:-1, 1:-1]
        n_zp = p[2:,  1:-1, 1:-1]
        n_ym = p[1:-1, :-2, 1:-1]
        n_yp = p[1:-1, 2:,  1:-1]
        n_xm = p[1:-1, 1:-1, :-2]
        n_xp = p[1:-1, 1:-1, 2:]

        interior = core & n_zm & n_zp & n_ym & n_yp & n_xm & n_xp
        surface = core & (~interior)
        return int(np.count_nonzero(surface))

    def compute_geometric_features(
        self,
        dens_threshold: float = 0.0,
        depth_thresholds: Tuple[float, float] = (2.0, 4.0),
        store: bool = True,
    ) -> Dict[str, float]:
        
        
        feats: Dict[str, float] = {}
        eps = 1e-8
        feats["voxel_count"] = np.sum(self.distance_map.ravel() > 0.0) 
        feats["voxel_volume"] = np.product(self.apix)
        feats["volume"] = feats["voxel_count"] * feats["voxel_volume"] 
        feats["bbox_z"], feats["bbox_y"],feats["bbox_x"] = self.distance_map.shape
        feats["bbox_voxel_volume"] = feats["bbox_z"]* feats["bbox_y"]*feats["bbox_x"]
        feats["bbox_volume"] = feats["bbox_voxel_volume"] * feats["voxel_volume"]
        feats['voxel_fill_fraction_in_voxel_bbox'] = feats["volume"] / feats["bbox_volume"]
        
        mask = self._get_pocket_mask(dens_threshold=dens_threshold)
        voxel_count = int(np.count_nonzero(mask))
        feats["pocket_voxel_count"] = float(voxel_count)
        feats["pocket_voxel_volume"] = float(voxel_count * feats["voxel_volume"])
        surface_count = self._surface_voxel_count(mask)
        feats["surface_voxel_count"] = float(surface_count)
        feats["surface_voxel_fraction"] = float(surface_count / (voxel_count + eps))
        
        #pca
        ax, ay, az = self.apix
        zyx = np.argwhere(mask)
        origin = np.asarray(self.origin, dtype=float).reshape(3)
        xyz_idx = zyx[:, ::-1].astype(float)  # (x,y,z)
        xyz = origin[None, :] + xyz_idx * np.asarray([ax, ay, az])[None, :]
        if xyz.shape[0] >= 3:
            mu = xyz.mean(axis=0)
            X = xyz - mu[None, :]
            cov = (X.T @ X) / max(1, X.shape[0] - 1)
            try:
                evals = np.linalg.eigvalsh(cov)
                evals = np.sort(np.clip(evals, 0.0, None))[::-1]  # descending
            except np.linalg.LinAlgError:
                evals = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            evals = np.array([0.0, 0.0, 0.0], dtype=float)

        v1, v2, v3 = [float(v) for v in evals]
        feats["pca_var1"] = v1
        feats["pca_var2"] = v2
        feats["pca_var3"] = v3
        feats["pca_ratio_12"] = v1 / (v2 + eps)
        feats["pca_ratio_13"] = v1 / (v3 + eps)
        feats["pca_ratio_23"] = v2 / (v3 + eps)
        
        
        return feats

        

def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _percentile_safe(a: np.ndarray, q: float, default=0.0) -> float:
    if a.size == 0:
        return float(default)
    return _safe_float(np.percentile(a, q), default=default)


def _mean_safe(a: np.ndarray, default=0.0) -> float:
    if a.size == 0:
        return float(default)
    return _safe_float(np.mean(a), default=default)


def _std_safe(a: np.ndarray, default=0.0) -> float:
    if a.size == 0:
        return float(default)
    return _safe_float(np.std(a), default=default)


def _max_safe(a: np.ndarray, default=0.0) -> float:
    if a.size == 0:
        return float(default)
    return _safe_float(np.max(a), default=default)
