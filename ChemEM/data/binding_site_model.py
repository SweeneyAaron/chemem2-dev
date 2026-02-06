# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


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
