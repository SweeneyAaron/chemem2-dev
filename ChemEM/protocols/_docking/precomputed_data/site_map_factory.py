from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from ChemEM.parsers.EMMap import EMMap
from .site_map_utils import (get_sasa_mask,
                             get_solvent_depth_mask,
                             get_flow_mask,
                             smooth_env_map_diffusion,
                             get_hydrophobic_grid,
                             get_electrostatic_map)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteMapFactoryConfig:
    """
    Configuration for building site maps.

    These defaults preserve the current behavior of build_site_maps_standalone().
    """
    crop_box_size: Tuple[float, float, float] = (30.0, 30.0, 30.0)
    electro_cutoff: float = 12.0

    # Flow/env smoothing
    env_smooth_n_iter: int = 10

    # Optional passthrough tuning (kept here for future maintainability)
    depth_connectivity: int = 6
    depth_crop: bool = True
    depth_pad: int = 3

    flow_connectivity: int = 6
    flow_power: int = 4

    # Hydrophob defaults are currently inside final_hydrophobic_grid_standalone(),
    # but kept here so you can route them through later if desired.
    hydro_cutoff: float = 5.0
    hydro_logp_smooth_sigma: float = 1.5
    hydro_vdw_radii_by_type: Optional[Mapping[str, float]] = None

    # Electrostatics defaults
    electro_c_factor: float = 332.06
    electro_min_r: float = 0.001


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

class SiteMapFactory:
    """
    Factory class for building site maps from protein/binding-site inputs.

    This class is a refactor of build_site_maps_standalone() that keeps the same
    inputs and same return structure, but organizes the pipeline into readable
    steps with validation and small helper methods.
    """

    def __init__(self, config: Optional[SiteMapFactoryConfig] = None):
        
        self.config = config or SiteMapFactoryConfig()

    # -------------------------
    # Public API
    # -------------------------

    def build(
        self,
        *,
        # geometry / atoms
        positions,
        atom_radii,
        atoms,

        # grid
        grid_origin,
        grid_spacing,
        grid,  # used for shape

        # things that used to be from self.system
        system_centroid,
        protein_complex_structure,
        protein_complex_system,
    ) -> Dict[str, Any]:
        """
        Build the site maps dictionary (same output keys as current standalone function).
        
        Returns
        -------
        dict
            {
                "env_scaled_map": EMMap,
                "electro_scaled_map": EMMap,
                "electro_raw_map": EMMap,
                "hydrophob_raw_map": EMMap,
                "hydrophob_enc_map": EMMap,
            }
        """
        self._validate_inputs(
            positions=positions,
            atom_radii=atom_radii,
            grid_origin=grid_origin,
            grid_spacing=grid_spacing,
            grid=grid,
            system_centroid=system_centroid,
        )

        grid_shape = grid.shape
        spacing3 = self._spacing3(grid_spacing)

        # 1) SASA masks
        bulk_solvent_mask, protein_mask, sasa_mask = self._build_sasa_masks(
            positions=positions,
            atom_radii=atom_radii,
            grid_origin=grid_origin,
            grid_shape=grid_shape,
            grid_spacing=grid_spacing,
            system_centroid=system_centroid,
        )

        # 2) Depth maps
        depth_map, depth_clean, depth_norm = self._build_depth_maps(
            bulk_solvent_mask=bulk_solvent_mask,
            protein_mask=protein_mask,
            sasa_mask=sasa_mask,
            grid_spacing=grid_spacing,
        )

        # 3) Flow/constriction + env index
        constriction = self._build_constriction_map(
            protein_mask=protein_mask,
            bulk_solvent_mask=bulk_solvent_mask,
            sasa_mask=sasa_mask,
            grid_spacing=grid_spacing,
        )

        env_index = depth_norm * constriction
        env_index_smooth = self._smooth_env_index(env_index=env_index, sasa_mask=sasa_mask)

        # 4) Hydrophobic maps
        (
            hydro_field_xlogp,
            hydro_enc_grid,
            hphob_sub_origin,
            hphob_sub_sasa_mask,
            hphob_sub_env_norm,
        ) = self._build_hydrophobic_maps(
            positions=positions,
            atoms=atoms,
            grid_origin=grid_origin,
            grid_shape=grid_shape,
            grid_spacing=grid_spacing,
            sasa_mask=sasa_mask,
            env_index_smooth=env_index_smooth,
        )

        # 5) Electrostatic maps (sub-box)
        (
            sub_env_index_smooth,
            sub_sasa_mask,
            sub_elc_origin,
            cpp_electrostatics,
            cpp_scaled,
        ) = self._build_electrostatic_maps(
            protein_complex_structure=protein_complex_structure,
            protein_complex_system=protein_complex_system,
            env_index_smooth=env_index_smooth,
            sasa_mask=sasa_mask,
            grid_origin=grid_origin,
            grid_spacing=grid_spacing,
            system_centroid=system_centroid,
        )

        # 6) Wrap into EMMap objects (same as before)
        env_scaled_map = self._make_emmap(grid_origin, spacing3, env_index_smooth)
        depth_norm = self._make_emmap(grid_origin, spacing3,  depth_norm)
        electro_scaled_map = self._make_emmap(sub_elc_origin, spacing3, cpp_scaled)
        electro_raw_map = self._make_emmap(sub_elc_origin, spacing3, cpp_electrostatics)

        hydrophob_raw_map = self._make_emmap(hphob_sub_origin, spacing3, hydro_field_xlogp)
        hydrophob_enc_map = self._make_emmap(hphob_sub_origin, spacing3, hydro_enc_grid)
        
        site_maps = {
            "env_scaled_map": env_scaled_map,
            "electro_scaled_map": electro_scaled_map,
            "electro_raw_map": electro_raw_map,
            "hydrophob_raw_map": hydrophob_raw_map,
            "hydrophob_enc_map": hydrophob_enc_map,
        }

        return site_maps

    # -------------------------
    # Stage methods
    # -------------------------

    def _build_sasa_masks(
        self,
        *,
        positions,
        atom_radii,
        grid_origin,
        grid_shape,
        grid_spacing,
        system_centroid,
    ):
        return get_sasa_mask(
            positions=positions,
            atom_radii=atom_radii,
            grid_origin=grid_origin,
            grid_shape=grid_shape,
            grid_spacing=grid_spacing,
            system_centroid=system_centroid,
            crop_box_size=self.config.crop_box_size,
        )

    def _build_depth_maps(
        self,
        *,
        bulk_solvent_mask,
        protein_mask,
        sasa_mask,
        grid_spacing,
    ):
        # You can route config depth params here if/when needed.
        return get_solvent_depth_mask(
            bulk_solvent_mask=bulk_solvent_mask,
            protein_mask=protein_mask,
            sasa_mask=sasa_mask,
            grid_spacing=grid_spacing,
            connectivity=self.config.depth_connectivity,
            crop=self.config.depth_crop,
            pad=self.config.depth_pad,
        )

    def _build_constriction_map(
        self,
        *,
        protein_mask,
        bulk_solvent_mask,
        sasa_mask,
        grid_spacing,
    ):
        return get_flow_mask(
            protein_mask=protein_mask,
            bulk_solvent_mask=bulk_solvent_mask,
            sasa_mask=sasa_mask,
            grid_spacing=grid_spacing,
            connectivity=self.config.flow_connectivity,
            power=self.config.flow_power,
        )

    def _smooth_env_index(self, *, env_index, sasa_mask):
        return smooth_env_map_diffusion(
            env_index=env_index,
            sasa_mask=sasa_mask,
            n_iter=self.config.env_smooth_n_iter,
        )

    def _build_hydrophobic_maps(
        self,
        *,
        positions,
        atoms,
        grid_origin,
        grid_shape,
        grid_spacing,
        sasa_mask,
        env_index_smooth,
    ):
        return get_hydrophobic_grid(
            positions=positions,
            atoms=atoms,
            grid_origin=grid_origin,
            grid_shape=grid_shape,
            grid_spacing=grid_spacing,
            sasa_mask=sasa_mask,
            env_index_smooth=env_index_smooth,
            cutoff=self.config.hydro_cutoff,
            logp_smooth_sigma=self.config.hydro_logp_smooth_sigma,
            vdw_radii_by_type=self.config.hydro_vdw_radii_by_type,
        )

    def _build_electrostatic_maps(
        self,
        *,
        protein_complex_structure,
        protein_complex_system,
        env_index_smooth,
        sasa_mask,
        grid_origin,
        grid_spacing,
        system_centroid,
    ):
        return get_electrostatic_map(
            protein_complex_structure=protein_complex_structure,
            protein_complex_system=protein_complex_system,
            env_index_smooth=env_index_smooth,
            sasa_mask=sasa_mask,
            grid_origin=grid_origin,
            grid_spacing=grid_spacing,
            system_centroid=system_centroid,
            base_box_size=self.config.crop_box_size,
            electro_cutoff=self.config.electro_cutoff,
            c_factor=self.config.electro_c_factor,
            min_r=self.config.electro_min_r,
        )

    # -------------------------
    # Small helpers
    # -------------------------

    def _make_emmap(self, origin, apix3, data):
        # EMMap must exist in your module namespace.
        return EMMap(
            np.asarray(origin, dtype=float),
            tuple(float(x) for x in apix3),
            np.asarray(data),
            3.0,
        )

    @staticmethod
    def _spacing3(grid_spacing) -> Tuple[float, float, float]:
        s = np.asarray(grid_spacing, dtype=float).reshape(-1)
        if s.size == 1:
            return (float(s[0]), float(s[0]), float(s[0]))
        if s.size == 3:
            return (float(s[0]), float(s[1]), float(s[2]))
        raise ValueError(f"grid_spacing must be scalar or length-3, got shape {s.shape}")

    @staticmethod
    def _validate_inputs(
        *,
        positions,
        atom_radii,
        grid_origin,
        grid_spacing,
        grid,
        system_centroid,
    ) -> None:
        pos = np.asarray(positions)
        rad = np.asarray(atom_radii)

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions must have shape (N,3), got {pos.shape}")
        if rad.ndim != 1:
            raise ValueError(f"atom_radii must be 1D, got {rad.shape}")
        if len(pos) != len(rad):
            raise ValueError(
                f"positions and atom_radii length mismatch: {len(pos)} vs {len(rad)}"
            )

        go = np.asarray(grid_origin, dtype=float).reshape(-1)
        if go.size != 3:
            raise ValueError(f"grid_origin must be length 3, got {go.shape}")

        sc = np.asarray(system_centroid, dtype=float).reshape(-1)
        if sc.size != 3:
            raise ValueError(f"system_centroid must be length 3, got {sc.shape}")

        if not hasattr(grid, "shape") or len(grid.shape) != 3:
            raise ValueError("grid must be a 3D array-like object with .shape")

        # Validate spacing through the same normalization logic
        _ = SiteMapFactory._spacing3(grid_spacing)


# -----------------------------------------------------------------------------
# Backwards-compatible wrapper (same signature as your old function)
# -----------------------------------------------------------------------------

def build_site_maps_standalone(
    *,
    # geometry / atoms
    positions,
    atom_radii,
    atoms,

    # grid
    grid_origin,
    grid_spacing,
    grid,  # used for shape

    # things that used to be from self.system
    system_centroid,
    protein_complex_structure,
    protein_complex_system,

    # optional tuning knobs
    crop_box_size=(30, 30, 30),
    electro_cutoff=12.0,
):
    """
    Backwards-compatible wrapper around SiteMapFactory.

    Same inputs and same returned site_maps dict as before.
    """
    cfg = SiteMapFactoryConfig(
        crop_box_size=tuple(float(x) for x in crop_box_size),
        electro_cutoff=float(electro_cutoff),
    )
    factory = SiteMapFactory(config=cfg)

    return factory.build(
        positions=positions,
        atom_radii=atom_radii,
        atoms=atoms,
        grid_origin=grid_origin,
        grid_spacing=grid_spacing,
        grid=grid,
        system_centroid=system_centroid,
        protein_complex_structure=protein_complex_structure,
        protein_complex_system=protein_complex_system,
    )

