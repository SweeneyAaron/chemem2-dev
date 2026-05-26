# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>
"""
Lining-residue refinement.

When a protein is fit into a cryo-EM map without its ligands/ions present,
sidechains lining a binding pocket can drift into densities that actually
belong to the (not-yet-placed) ligand. This protocol detects those misfit
sidechains and refines them out of the ligand-density region using a custom
repulsive pocket force together with the normal density force.

Detection uses a density-blob centroid method:

  1. Crop the raw density map to each binding site's bounding box (no SES
     mask — we need drifted sidechains to be visible).
  2. Threshold + connected-component label.
  3. For each component compute density-weighted centroid and sample the
     site's distance_map at that centroid (pocket depth in Å).
  4. Keep components whose volume >= lr_blob_vol_min and whose centroid is
     at least lr_centroid_depth_thr deep in the pocket, and (by default)
     whose voxels do not bridge to backbone density.
  5. Any lining sidechain with >= lr_misfit_frac of its heavy atoms inside a
     surviving blob is flagged as misfit.

Refinement rebuilds a localised OpenMM simulation around the flagged atoms
(plus their first-shell neighbours), applies the normal density force, a
soft positional pin on every heavy atom EXCEPT the flagged sidechain atoms,
and a grid-based repulsive pocket force that acts only on the flagged atoms.
"""

import os
import time

import numpy as np
from scipy.ndimage import generate_binary_structure, label

from openmm import unit

from ChemEM.parsers.EMMap import EMMap
from ChemEM.messages import Messages
from ChemEM.tools.map_q_score import MapGrid
from ChemEM.tools.density import extract_subvolume_from_grid

from ChemEM.protocols.core.forces import ForceBuilder
from ChemEM.protocols.refine.pose_minimiser import ChemEMSimulationSetup


_BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT"}


def _atom_key(atom):
    chain = getattr(atom.residue, "chain", "")
    number = getattr(atom.residue, "number", getattr(atom.residue, "idx", -1))
    try:
        number = int(number)
    except (TypeError, ValueError):
        number = -1
    return (str(chain), number, str(atom.residue.name), str(atom.name))


def _res_key(residue):
    number = getattr(residue, "number", getattr(residue, "idx", -1))
    try:
        number = int(number)
    except (TypeError, ValueError):
        number = -1
    return (str(getattr(residue, "chain", "")), number, str(residue.name))


def _residues_near_points(points, complex_structure, distance_cutoff):
    """
    Return the list of residues in `complex_structure` with at least one atom
    within `distance_cutoff` Å of any point in `points`. Preserves full-structure
    atom indices (unlike get_residue_subset_from_points which returns a subset).
    """
    selected_residues = []
    seen = set()
    cutoff2 = float(distance_cutoff) * float(distance_cutoff)
    points = np.asarray(points, dtype=float)

    for res in complex_structure.residues:
        atoms = list(res.atoms)
        if not atoms:
            continue
        res_xyz = np.array([[a.xx, a.xy, a.xz] for a in atoms], dtype=float)
        diffs = res_xyz[:, None, :] - points[None, :, :]
        d2 = np.sum(diffs * diffs, axis=-1)
        if np.any(d2 <= cutoff2):
            key = _res_key(res)
            if key not in seen:
                seen.add(key)
                selected_residues.append(res)
    return selected_residues


class LiningRefine:
    """
    Detect and refine lining sidechains that have drifted into ligand density.
    Requires `binding_site` to have already been run and a density_map to be
    loaded on the system.
    """

    def __init__(self, system):
        self.system = system
        self.output = os.path.join(system.output, "lining_refine")
        os.makedirs(self.output, exist_ok=True)

        self.per_site_detections = {}
        self.flagged_residue_keys = set()
        self.flagged_atom_keys = set()
        self._refine_stats = None

    # ------------------------------------------------------------------
    # Stage 1 — detection
    # ------------------------------------------------------------------

    def detect(self):
        if self.system.density_map is None:
            self.system.log(Messages.chemem_warning(
                "LiningRefine", "detect",
                "system.density_map is not loaded — skipping detection.",
            ))
            return

        if not getattr(self.system, "binding_sites", None):
            self.system.log(Messages.chemem_warning(
                "LiningRefine", "detect",
                "system.binding_sites is empty — nothing to do.",
            ))
            return

        opts = self.system.options
        full_map = self.system.density_map.density_map
        map_mean = float(np.mean(full_map))
        map_std = float(np.std(full_map))
        intensity_cutoff = map_mean + float(opts.lr_sigma_thr) * map_std

        voxel_vol = float(np.prod(np.asarray(self.system.density_map.apix, dtype=float)))
        struct = generate_binary_structure(3, 1)
        protein_struct = self.system.protein.complex_structure

        for site_key, site in self.system.binding_sites.items():
            detection = self._detect_site(
                site=site,
                site_key=site_key,
                protein_struct=protein_struct,
                intensity_cutoff=intensity_cutoff,
                voxel_vol=voxel_vol,
                struct=struct,
                opts=opts,
            )
            if detection is None:
                continue
            self.per_site_detections[site_key] = detection
            self.flagged_residue_keys.update(detection["misfit_residue_keys"])
            self.flagged_atom_keys.update(detection["flagged_atom_keys"])

    def _detect_site(self, site, site_key, protein_struct,
                     intensity_cutoff, voxel_vol, struct, opts):
        if site.distance_map is None:
            return None
        if not site.lining_residues:
            return None

        min_c = np.asarray(site.min_coords, dtype=float)
        max_c = np.asarray(site.max_coords, dtype=float)
        pad = float(opts.lr_crop_pad)
        grid_origin = min_c - pad
        box_size_A = (max_c - min_c) + 2.0 * pad

        apix = np.asarray(self.system.density_map.apix, dtype=float)
        box_voxels = np.maximum(1, np.ceil(box_size_A / apix)).astype(int)

        try:
            submap = extract_subvolume_from_grid(
                self.system.density_map.origin,
                apix,
                self.system.density_map.density_map,
                box_voxels,
                grid_origin=grid_origin,
                resolution=self.system.density_map.resolution,
            )
        except Exception as exc:
            self.system.log(Messages.chemem_warning(
                "LiningRefine", "_detect_site",
                f"site {site_key}: subvolume extraction failed: {exc}",
            ))
            return None
        if submap is None:
            return None

        cropped = np.asarray(submap.density_map, dtype=float)
        if cropped.size == 0:
            return None

        crop_origin = np.asarray(submap.origin, dtype=float)
        crop_apix = np.asarray(submap.apix, dtype=float)
        nz, ny, nx = cropped.shape

        thr_mask = cropped > intensity_cutoff
        if not np.any(thr_mask):
            return self._empty_detection(site_key)

        labels_arr, num_features = label(thr_mask, structure=struct)
        if num_features == 0:
            return self._empty_detection(site_key)

        dist_mapgrid = MapGrid(
            data=np.asarray(site.distance_map, dtype=np.float32),
            origin_xyz=np.asarray(site.origin, dtype=float),
            apix_xyz=np.asarray(site.apix, dtype=float),
        )

        candidate_blobs = {}
        min_vol = float(opts.lr_blob_vol_min)
        min_depth = float(opts.lr_centroid_depth_thr)

        for comp_id in range(1, num_features + 1):
            comp_mask = (labels_arr == comp_id)
            n_vox = int(np.count_nonzero(comp_mask))
            vol_A3 = n_vox * voxel_vol
            if vol_A3 < min_vol:
                continue

            zs, ys, xs = np.where(comp_mask)
            weights = cropped[zs, ys, xs]
            wsum = float(weights.sum())
            if wsum <= 0.0:
                continue
            cx = float((xs * weights).sum() / wsum)
            cy = float((ys * weights).sum() / wsum)
            cz = float((zs * weights).sum() / wsum)
            centroid_xyz = crop_origin + np.array([cx, cy, cz], dtype=float) * crop_apix

            depth_samples = dist_mapgrid.sample_trilinear(centroid_xyz.reshape(1, 3))
            centroid_depth = float(depth_samples[0]) if np.isfinite(depth_samples[0]) else 0.0
            if centroid_depth < min_depth:
                continue

            candidate_blobs[comp_id] = {
                "volume_A3": vol_A3,
                "centroid_xyz": centroid_xyz,
                "centroid_depth": centroid_depth,
            }

        if not candidate_blobs:
            return self._empty_detection(site_key, num_features=num_features)

        # Backbone-bridge filter: drop blobs touching backbone density voxels.
        if not opts.lr_allow_backbone_bridge:
            bridge_vox = max(
                1,
                int(np.ceil(float(opts.lr_backbone_bridge_dist) / float(np.mean(crop_apix)))),
            )
            backbone_mask = np.zeros_like(cropped, dtype=bool)
            for atom in protein_struct.atoms:
                if atom.element <= 1:
                    continue
                if str(atom.name).strip() not in {"N", "CA", "C", "O"}:
                    continue
                vx = (atom.xx - crop_origin[0]) / crop_apix[0]
                vy = (atom.xy - crop_origin[1]) / crop_apix[1]
                vz = (atom.xz - crop_origin[2]) / crop_apix[2]
                ix = int(np.floor(vx)); iy = int(np.floor(vy)); iz = int(np.floor(vz))
                if ix < -bridge_vox or iy < -bridge_vox or iz < -bridge_vox:
                    continue
                if ix >= nx + bridge_vox or iy >= ny + bridge_vox or iz >= nz + bridge_vox:
                    continue
                x0 = max(0, ix - bridge_vox); x1 = min(nx, ix + bridge_vox + 1)
                y0 = max(0, iy - bridge_vox); y1 = min(ny, iy + bridge_vox + 1)
                z0 = max(0, iz - bridge_vox); z1 = min(nz, iz + bridge_vox + 1)
                backbone_mask[z0:z1, y0:y1, x0:x1] = True

            to_drop = []
            for comp_id in list(candidate_blobs.keys()):
                if np.any((labels_arr == comp_id) & backbone_mask):
                    to_drop.append(comp_id)
            for cid in to_drop:
                candidate_blobs.pop(cid)

        if not candidate_blobs:
            return self._empty_detection(site_key, num_features=num_features)

        residue_diagnostics = {}
        flagged_atom_keys = set()
        misfit_residue_keys = set()
        misfit_frac = float(opts.lr_misfit_frac)

        for res in site.lining_residues:
            side_atoms = [
                a for a in res.atoms
                if a.element > 1 and str(a.name).strip() not in _BACKBONE_NAMES
            ]
            if not side_atoms:
                continue

            n_side = len(side_atoms)
            blob_counts = {}
            in_blob_atoms = []

            for atom in side_atoms:
                ix = int(np.floor((atom.xx - crop_origin[0]) / crop_apix[0]))
                iy = int(np.floor((atom.xy - crop_origin[1]) / crop_apix[1]))
                iz = int(np.floor((atom.xz - crop_origin[2]) / crop_apix[2]))
                if ix < 0 or iy < 0 or iz < 0:
                    continue
                if ix >= nx or iy >= ny or iz >= nz:
                    continue
                comp_id = int(labels_arr[iz, iy, ix])
                if comp_id == 0 or comp_id not in candidate_blobs:
                    continue
                blob_counts[comp_id] = blob_counts.get(comp_id, 0) + 1
                in_blob_atoms.append(atom)

            if not in_blob_atoms:
                continue
            frac = len(in_blob_atoms) / float(n_side)
            if frac < misfit_frac:
                continue

            rkey = _res_key(res)
            misfit_residue_keys.add(rkey)
            for atom in in_blob_atoms:
                flagged_atom_keys.add(_atom_key(atom))

            residue_diagnostics[rkey] = {
                "n_side_atoms": n_side,
                "n_in_blob": len(in_blob_atoms),
                "blob_counts": blob_counts,
                "blob_centroid_depths": [
                    candidate_blobs[c]["centroid_depth"] for c in blob_counts
                ],
                "blob_volumes_A3": [
                    candidate_blobs[c]["volume_A3"] for c in blob_counts
                ],
            }

        return {
            "site_key": site_key,
            "num_components": num_features,
            "candidate_blobs": candidate_blobs,
            "misfit_residue_keys": misfit_residue_keys,
            "flagged_atom_keys": flagged_atom_keys,
            "residue_diagnostics": residue_diagnostics,
        }

    def _empty_detection(self, site_key, num_features=0):
        return {
            "site_key": site_key,
            "num_components": num_features,
            "candidate_blobs": {},
            "misfit_residue_keys": set(),
            "flagged_atom_keys": set(),
            "residue_diagnostics": {},
        }

    # ------------------------------------------------------------------
    # Stage 2 — combined pocket-depth grid
    # ------------------------------------------------------------------

    def build_pocket_depth_grid(self):
        if self.system.density_map is None:
            return None

        full_map = self.system.density_map
        full_shape = full_map.density_map.shape  # (nz, ny, nx)
        combined = np.zeros(full_shape, dtype=np.float32)

        map_origin_xyz = np.asarray(full_map.origin, dtype=float)
        map_apix_xyz = np.asarray(full_map.apix, dtype=float)

        any_stamped = False
        for site_key, detection in self.per_site_detections.items():
            if not detection["misfit_residue_keys"]:
                continue
            site = self.system.binding_sites[site_key]
            dm = site.distance_map
            if dm is None:
                continue
            dm = np.asarray(dm, dtype=np.float32)
            if dm.ndim != 3:
                continue
            nz_s, ny_s, nx_s = dm.shape

            site_origin_xyz = np.asarray(site.origin, dtype=float)
            offset_xyz = np.round(
                (site_origin_xyz - map_origin_xyz) / map_apix_xyz
            ).astype(int)
            ox, oy, oz = int(offset_xyz[0]), int(offset_xyz[1]), int(offset_xyz[2])

            z0 = max(0, oz); z1 = min(full_shape[0], oz + nz_s)
            y0 = max(0, oy); y1 = min(full_shape[1], oy + ny_s)
            x0 = max(0, ox); x1 = min(full_shape[2], ox + nx_s)
            if z0 >= z1 or y0 >= y1 or x0 >= x1:
                continue

            sz0 = z0 - oz; sz1 = sz0 + (z1 - z0)
            sy0 = y0 - oy; sy1 = sy0 + (y1 - y0)
            sx0 = x0 - ox; sx1 = sx0 + (x1 - x0)

            block_dst = combined[z0:z1, y0:y1, x0:x1]
            block_src = dm[sz0:sz1, sy0:sy1, sx0:sx1]
            np.maximum(block_dst, block_src, out=block_dst)
            any_stamped = True

        if not any_stamped:
            return None

        pocket_map = EMMap(
            tuple(float(v) for v in map_origin_xyz),
            tuple(float(v) for v in map_apix_xyz),
            combined,
            full_map.resolution,
        )
        try:
            pocket_map.write_mrc(os.path.join(self.output, "pocket_depth_map.mrc"))
        except Exception:
            pass
        return pocket_map

    # ------------------------------------------------------------------
    # Stage 3 — localised refinement
    # ------------------------------------------------------------------

    def refine(self, pocket_depth_map):
        if not self.flagged_atom_keys:
            return

        opts = self.system.options
        full_struct = self.system.protein.complex_structure

        full_atom_by_key = {}
        for atom in full_struct.atoms:
            if atom.element <= 1:
                continue
            full_atom_by_key[_atom_key(atom)] = atom

        flagged_points = []
        for akey in self.flagged_atom_keys:
            atom = full_atom_by_key.get(akey)
            if atom is not None:
                flagged_points.append([atom.xx, atom.xy, atom.xz])
        if not flagged_points:
            return

        flagged_points = np.asarray(flagged_points, dtype=float)

        neighborhood = float(opts.lr_neighborhood)
        subset_residues = _residues_near_points(
            flagged_points, full_struct, distance_cutoff=neighborhood
        )
        if not subset_residues:
            self.system.log(Messages.chemem_warning(
                "LiningRefine", "refine",
                "No residues in neighborhood subset; skipping refinement.",
            ))
            return

        platform_name = getattr(opts, "platform", None) or "CPU"
        env = ChemEMSimulationSetup(
            protein_structure=full_struct,
            ligand_structure=[],
            residues=subset_residues,
            density_map=self.system.density_map,
            platform_name=platform_name,
            restrain_side_chains=False,
            protein_restraint="none",
            global_k=float(opts.lr_global_k),
            resource_owner=self.system,
        )

        flagged_subset_indices = []
        subset_heavy_indices = []
        for atom in env.complex_structure.atoms:
            if atom.element <= 1:
                continue
            subset_heavy_indices.append(int(atom.idx))
            if _atom_key(atom) in self.flagged_atom_keys:
                flagged_subset_indices.append(int(atom.idx))

        if not flagged_subset_indices:
            self.system.log(Messages.chemem_warning(
                "LiningRefine", "refine",
                "No flagged atoms found in subset after remapping.",
            ))
            return

        flagged_set = set(flagged_subset_indices)
        pin_indices = [i for i in subset_heavy_indices if i not in flagged_set]

        state = env.simulation.context.getState(getPositions=True)
        ref_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        if pin_indices:
            pin_force = ForceBuilder.create_positional_pin(
                pin_indices, ref_nm, k_name="k_lr_pin",
            )
            env.complex_system.addForce(pin_force)

        if pocket_depth_map is not None:
            repel_force = ForceBuilder.create_pocket_repulsive_force(
                pocket_depth_map,
                k_repel=float(opts.lr_repel_k),
                k_name="k_lr_repel",
                normalise=True,
            )
            for idx in flagged_subset_indices:
                repel_force.addBond([int(idx)])
            env.complex_system.addForce(repel_force)

        env.simulation.context.reinitialize(preserveState=True)
        if pin_indices:
            env.simulation.context.setParameter(
                "k_lr_pin", float(opts.lr_backbone_k)
            )

        pre_state = env.simulation.context.getState(getPositions=True)
        pre_nm = pre_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        pre_flagged = np.asarray(pre_nm[flagged_subset_indices], dtype=float)

        print(f"[lining_refine] minimizing {len(subset_heavy_indices)} heavy atoms, "
              f"{len(flagged_subset_indices)} flagged, {len(pin_indices)} pinned")
        env.simulation.minimizeEnergy()

        final_state = env.simulation.context.getState(getPositions=True, getEnergy=True)
        final_pos_A = final_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        env.complex_structure.positions = final_pos_A * unit.angstrom

        post_nm = final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        post_flagged = np.asarray(post_nm[flagged_subset_indices], dtype=float)
        diffs = (post_flagged - pre_flagged) * 10.0  # nm -> Å
        rmsd_A = float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1)))) if diffs.size else 0.0
        max_A = float(np.max(np.linalg.norm(diffs, axis=1))) if diffs.size else 0.0
        energy = float(final_state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

        self._refine_stats = {
            "n_subset_atoms": len(subset_heavy_indices),
            "n_flagged_atoms": len(flagged_subset_indices),
            "n_pinned_atoms": len(pin_indices),
            "rmsd_A": rmsd_A,
            "max_disp_A": max_A,
            "final_energy_kcal_mol": energy,
        }

        # Copy refined subset positions back into the full protein structure.
        subset_key_to_xyz = {}
        for atom in env.complex_structure.atoms:
            subset_key_to_xyz[_atom_key(atom)] = (atom.xx, atom.xy, atom.xz)

        updated = 0
        for atom in full_struct.atoms:
            key = _atom_key(atom)
            if key in subset_key_to_xyz:
                x, y, z = subset_key_to_xyz[key]
                atom.xx = float(x)
                atom.xy = float(y)
                atom.xz = float(z)
                updated += 1
        self._refine_stats["n_full_atoms_updated"] = updated

        # Write refined subset PDB for inspection.
        try:
            env.complex_structure.save(
                os.path.join(self.output, "subset_refined.pdb"),
                overwrite=True,
            )
        except Exception as exc:
            self.system.log(Messages.chemem_warning(
                "LiningRefine", "refine",
                f"Could not write subset PDB: {exc}",
            ))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self):
        opts = self.system.options
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("LINING-RESIDUE REFINEMENT SUMMARY")
        lines.append("=" * 60)
        lines.append("Detection parameters:")
        lines.append(f"  Sigma thr (map):       {opts.lr_sigma_thr}")
        lines.append(f"  Crop pad:              {opts.lr_crop_pad} Å")
        lines.append(f"  Blob volume min:       {opts.lr_blob_vol_min} Å³")
        lines.append(f"  Centroid depth thr:    {opts.lr_centroid_depth_thr} Å")
        lines.append(f"  Misfit atom fraction:  {opts.lr_misfit_frac}")
        lines.append(
            f"  Backbone bridge:       "
            f"{'allowed' if opts.lr_allow_backbone_bridge else 'rejected'}"
        )
        lines.append("-" * 60)
        lines.append(f"Total misfit residues: {len(self.flagged_residue_keys)}")

        if self.per_site_detections:
            for site_key, det in self.per_site_detections.items():
                n_res = len(det["misfit_residue_keys"])
                n_blobs = len(det["candidate_blobs"])
                lines.append(
                    f"\nSite {site_key}: candidate blobs={n_blobs}, "
                    f"misfit residues={n_res}"
                )
                for rkey, diag in det["residue_diagnostics"].items():
                    chain, resnum, resname = rkey
                    depths = ", ".join(f"{d:.2f}" for d in diag["blob_centroid_depths"])
                    vols = ", ".join(f"{v:.1f}" for v in diag["blob_volumes_A3"])
                    lines.append(
                        f"  {chain}:{resname}{resnum}  "
                        f"side={diag['n_in_blob']}/{diag['n_side_atoms']}  "
                        f"depths=[{depths}] Å  vols=[{vols}] Å³"
                    )
        else:
            lines.append("No sites evaluated.")

        if self._refine_stats is not None:
            lines.append("-" * 60)
            lines.append("Refinement:")
            stats = self._refine_stats
            lines.append(f"  Subset heavy atoms:   {stats['n_subset_atoms']}")
            lines.append(f"  Flagged atoms:        {stats['n_flagged_atoms']}")
            lines.append(f"  Pinned atoms:         {stats['n_pinned_atoms']}")
            lines.append(f"  Flagged RMSD:         {stats['rmsd_A']:.3f} Å")
            lines.append(f"  Flagged max disp:     {stats['max_disp_A']:.3f} Å")
            lines.append(f"  Final energy:         {stats['final_energy_kcal_mol']:.2f} kcal/mol")
            lines.append(f"  Global k (density):   {opts.lr_global_k}")
            lines.append(f"  Backbone pin k:       {opts.lr_backbone_k}")
            lines.append(f"  Pocket repulsion k:   {opts.lr_repel_k}")

        lines.append("=" * 60 + "\n")
        self.system.log("\n".join(lines))

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self):
        self.system.log(Messages.create_centered_box("Lining-Residue Refinement"))
        t0 = time.perf_counter()

        self.detect()

        if not self.flagged_residue_keys:
            self.system.log("[lining_refine] No misfit lining residues detected.")
            self.log()
            return

        pocket_depth_map = self.build_pocket_depth_grid()
        self.refine(pocket_depth_map)
        self.log()
        self.system.log(
            f"[lining_refine] Finished in {time.perf_counter() - t0:.2f}s"
        )
