# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

from __future__ import annotations

import os
import time
from typing import Any, List, Optional

import numpy as np
from openmm import unit
from scipy.spatial import cKDTree

from ChemEM.messages import Messages
from ChemEM.parsers.writers import save_structure_parmed

from ChemEM.protocols.core.density import submap_from_structure
from ChemEM.protocols.core.simulation import (
    check_pose_forces,
    update_global_positions,
    update_ligand_positions,
)
from ChemEM.protocols.refine.pose_minimiser import ChemEMSimulationSetup
from ChemEM.protocols.refine.refine_utils import (
    get_residue_positions,
    get_residue_subset_from_points,
)

from .acceptance import AcceptOutcome, get_acceptance
from .diagnostic import (
    atom_fit_quality,
    best_rigid_transform_for_cluster,
    classify_atoms,
    cluster_bad_atoms,
    directed_kick_proposal,
    enumerate_rotatable_heavy_bonds,
    format_diagnostic,
    rank_dihedrals_for_atom,
)
from .direction import (
    build_targets_from_dihedral,
    build_targets_from_gradient,
    build_targets_from_subregion,
)
from .io import write_ligand_outputs, write_summary
from .scorers import get_scorer
from .types import RefinedPose


_DIHEDRAL_ANGLE_STEPS_DEG = (30.0, 60.0, 90.0)
_DIHEDRAL_BONDS_PER_BAD_ATOM = 2


class SearchRefine:
    """
    Metric-modular, gradient-driven local refinement.

    Seed policy:
      - Uses input ligand conformer 0 only. Ignores ligand.docked entries.

    Core loop:
      - Per-iteration metric gradient (FD) on ligand heavy atoms.
      - Capped per-atom targets via CustomExternalForce tether.
      - Pull-relax (optional MD) + minimization.
      - Acceptance strategy (greedy/metropolis/basin_hopping) drives the state.
      - Final ranking by hybrid score (metric - wE*z(E) - wC*z(clash)).
    """

    def __init__(self, system):
        self.system = system
        self.output = None

    # ---------- option helpers ----------

    def _opt(self, name: str, default: Any) -> Any:
        return getattr(self.system.options, name, default)

    def _sr_verbose(self) -> bool:
        return bool(self._opt("sr_verbose", False))

    def _sr_log_every(self) -> int:
        try:
            every = int(self._opt("sr_log_every", 1))
        except Exception:
            every = 1
        return max(1, every)

    def _sr_debug_relax(self) -> bool:
        return bool(self._opt("sr_debug_relax", False))

    def _sr_log(self, message, outer_iter=None, force=False) -> None:
        if not self._sr_verbose():
            return
        if (not force) and (outer_iter is not None):
            if (int(outer_iter) % self._sr_log_every()) != 0:
                return
        self.system.log(message)

    @staticmethod
    def _fmt_record(rec) -> str:
        return (
            f"final={float(rec.get('final_score', np.nan)):.6f} "
            f"score={float(rec.get('score', np.nan)):.6f} "
            f"E={float(rec.get('energy_kcal', np.nan)):.2f} "
            f"clash={float(rec.get('clash_penalty', np.nan)):.4f}"
        )

    # ---------- proposal enumeration ----------

    @staticmethod
    def _enumerate_dihedral_triples(
        bad_atoms_sorted,
        rotatable_bonds,
        mol,
        heavy_coords_A,
        grad,
        max_count: int,
        bonds_per_atom: int = _DIHEDRAL_BONDS_PER_BAD_ATOM,
        angle_steps_deg=_DIHEDRAL_ANGLE_STEPS_DEG,
    ):
        """Build an ordered list of (atom, bond, angle) dihedral proposal descriptors.

        For each bad atom (in badness order), up to ``bonds_per_atom`` best-aligned
        rotatable bonds are chosen; each bond spawns one proposal per angle step,
        with the sign of the rotation chosen so that the atom moves toward its
        per-atom gradient target.
        """
        out = []
        if not rotatable_bonds or not bad_atoms_sorted:
            return out

        for atom_idx in bad_atoms_sorted:
            g = grad[atom_idx]
            g_norm = float(np.linalg.norm(g))
            if g_norm < 1e-12:
                continue
            target_dir = g / g_norm
            try:
                ranked = rank_dihedrals_for_atom(
                    atom_heavy_idx=int(atom_idx),
                    target_dir=target_dir,
                    heavy_coords_A=heavy_coords_A,
                    rotatable_bonds=rotatable_bonds,
                    mol=mol,
                )
            except Exception:
                continue

            for entry in ranked[:bonds_per_atom]:
                sign = 1.0 if entry.alignment > 0 else -1.0
                for step_deg in angle_steps_deg:
                    dtheta_deg = float(sign * step_deg)
                    out.append(
                        {
                            "bad_atom": int(atom_idx),
                            "bond": entry.bond,
                            "side": entry.side_atoms,
                            "delta_theta_rad": float(np.radians(dtheta_deg)),
                            "delta_theta_deg": dtheta_deg,
                            "alignment": float(entry.alignment),
                        }
                    )
                    if len(out) >= max_count:
                        return out
        return out

    @staticmethod
    def _proposal_scale(prop_idx: int, total: int) -> float:
        total = max(1, int(total))
        if total == 1:
            return 1.0
        return float(prop_idx) / float(total)

    @staticmethod
    def _proposal_mode(prop_idx: int, total: int) -> str:
        total = max(1, int(total))
        if total == 1:
            return "plus"
        if total == 2:
            modes = ["plus", "minus"]
        elif total == 3:
            modes = ["plus", "minus", "plus_jitter"]
        else:
            modes = ["plus", "minus", "plus_jitter", "minus_jitter",
                     "random_plus", "random_minus"]
        return modes[(int(prop_idx) - 1) % len(modes)]

    # ---------- map selection ----------

    @staticmethod
    def _first_map_like(obj):
        if obj is None:
            return None
        if hasattr(obj, "density_map"):
            return obj
        if isinstance(obj, (list, tuple)):
            for item in obj:
                if hasattr(item, "density_map"):
                    return item
        return None

    def _select_map(self):
        if bool(self._opt("no_map", False)):
            return None

        mode = str(self._opt("sr_map_source", "confidence")).strip().lower()

        density_map = self._first_map_like(getattr(self.system, "density_map", None))
        confidence_map = self._first_map_like(getattr(self.system, "confidence_map", None))
        difference_map = self._first_map_like(getattr(self.system, "difference_map", None))

        if mode == "raw":
            return density_map or confidence_map or difference_map
        if mode == "difference":
            return difference_map or confidence_map or density_map
        return confidence_map or density_map or difference_map

    # ---------- scoring helpers ----------

    @staticmethod
    def _positions_are_finite(env) -> bool:
        try:
            st = env.simulation.context.getState(getPositions=True)
            pos_nm = np.asarray(
                st.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=np.float64,
            )
        except Exception:
            return False
        return bool(np.isfinite(pos_nm).all())

    @staticmethod
    def _disp_stats_A(current_nm, reference_nm, ligand_indices) -> tuple:
        curr = np.asarray(current_nm, dtype=np.float64)
        ref = np.asarray(reference_nm, dtype=np.float64)
        lig = np.asarray(ligand_indices, dtype=int)
        if curr.shape != ref.shape or lig.size == 0:
            return np.nan, np.nan
        dA = np.linalg.norm(curr[lig] - ref[lig], axis=1) * 10.0
        if dA.size == 0:
            return np.nan, np.nan
        return float(np.sqrt(np.mean(dA * dA))), float(np.max(dA))

    def _log_relax_state(self, env, log_tag, stage, accepted_pos_nm=None, target_pos_nm=None):
        if (not self._sr_verbose()) or (not self._sr_debug_relax()):
            return

        try:
            st = env.simulation.context.getState(
                getEnergy=True, getPositions=True, getForces=True
            )
            pos_nm = np.asarray(
                st.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=np.float64,
            )
            frc = np.asarray(
                st.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer
                ),
                dtype=np.float64,
            )
            lig_idx = np.asarray(env.all_ligand_indices, dtype=int)
            lig_force = frc[lig_idx] if lig_idx.size else np.zeros((0, 3), dtype=np.float64)
            lig_force_norm = np.linalg.norm(lig_force, axis=1) if lig_force.size else np.array([])
            f_max = float(np.max(lig_force_norm)) if lig_force_norm.size else 0.0
            f_rms = (
                float(np.sqrt(np.mean(lig_force_norm * lig_force_norm)))
                if lig_force_norm.size
                else 0.0
            )
            energy_kcal = float(
                st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            )
            finite_pos = bool(np.isfinite(pos_nm).all())

            msg = (
                f"[search_refine][dbg] {log_tag} {stage}: "
                f"E={energy_kcal:.2f} kcal/mol "
                f"F(max={f_max:.3g},rms={f_rms:.3g}) finite={finite_pos}"
            )
            if not finite_pos:
                n_bad = int(pos_nm.size - np.count_nonzero(np.isfinite(pos_nm)))
                msg += f" bad_coords={n_bad}"
            if accepted_pos_nm is not None:
                d_rms_A, d_max_A = self._disp_stats_A(pos_nm, accepted_pos_nm, lig_idx)
                msg += f" d_acc(rms={d_rms_A:.3f}A,max={d_max_A:.3f}A)"
            if target_pos_nm is not None:
                t_rms_A, t_max_A = self._disp_stats_A(pos_nm, target_pos_nm, lig_idx)
                msg += f" d_tgt(rms={t_rms_A:.3f}A,max={t_max_A:.3f}A)"
            self.system.log(msg)
        except Exception as exc:
            self.system.log(
                f"[search_refine][dbg] {log_tag} {stage}: snapshot failed ({exc})"
            )

    @staticmethod
    def _rmsd(coords_a, coords_b) -> float:
        a = np.asarray(coords_a, dtype=np.float64)
        b = np.asarray(coords_b, dtype=np.float64)
        if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
            return float("inf")
        d2 = np.sum((a - b) * (a - b), axis=1)
        return float(np.sqrt(np.mean(d2)))

    @staticmethod
    def _zscore(vals, eps: float = 1e-12) -> np.ndarray:
        x = np.asarray(vals, dtype=np.float64)
        mu = float(np.mean(x))
        sd = float(np.std(x))
        if sd < eps:
            return np.zeros_like(x)
        return (x - mu) / sd

    def _apply_hybrid_scores(self, records: list) -> list:
        if not records:
            return records

        wE = float(self._opt("sr_w_energy", 0.15))
        wC = float(self._opt("sr_w_clash", 0.10))

        energies = [float(r["energy_kcal"]) for r in records]
        clashes = [float(r["clash_penalty"]) for r in records]
        zE = self._zscore(energies)
        zC = self._zscore(clashes)

        for idx, rec in enumerate(records):
            rec["final_score"] = float(rec["score"] - (wE * zE[idx]) - (wC * zC[idx]))
        return records

    def _clash_penalty(self, positions_nm, env, protein_heavy_indices) -> float:
        if not protein_heavy_indices:
            return 0.0

        lig_xyz_A = np.asarray(positions_nm[env.ligand_heavy_indices], dtype=np.float64) * 10.0
        prot_xyz_A = np.asarray(positions_nm[protein_heavy_indices], dtype=np.float64) * 10.0

        if lig_xyz_A.size == 0 or prot_xyz_A.size == 0:
            return 0.0

        tree = cKDTree(prot_xyz_A)
        dists, _ = tree.query(lig_xyz_A, k=1)

        clash_dist = float(self._opt("sr_clash_distance_a", 1.6))
        overlap = np.maximum(0.0, clash_dist - np.asarray(dists, dtype=np.float64))
        return float(np.mean(overlap * overlap))

    def _capture_record(
        self,
        env,
        scorer,
        protein_heavy_indices,
        outer_iter: int,
        proposal_idx: int,
    ) -> Optional[dict]:
        bad, reason = check_pose_forces(env.simulation, env.all_ligand_indices)
        if bad:
            self.system.log(f"[search_refine] skip unstable state: {reason}")
            return None

        state = env.simulation.context.getState(getPositions=True, getEnergy=True)
        positions_nm = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )

        ligand_coords_A = np.asarray(positions_nm[env.all_ligand_indices], dtype=np.float64) * 10.0
        ligand_heavy_coords_A = np.asarray(positions_nm[env.ligand_heavy_indices], dtype=np.float64) * 10.0

        score_terms: dict = {}
        score_val = float(scorer.score(ligand_heavy_coords_A, terms_out=score_terms))

        clash_penalty = self._clash_penalty(positions_nm, env, protein_heavy_indices)
        energy_kcal = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

        return {
            "outer_iter": int(outer_iter),
            "proposal_idx": int(proposal_idx),
            "positions_nm": positions_nm,
            "ligand_coords_A": ligand_coords_A,
            "ligand_heavy_coords_A": ligand_heavy_coords_A,
            "score": float(score_val),
            "score_terms": score_terms,
            "energy_kcal": energy_kcal,
            "clash_penalty": float(clash_penalty),
            # Placeholder until global hybrid reranking; keep informative for logs.
            "final_score": float(score_val),
        }

    # ---------- env / setup ----------

    def _prepare_output(self) -> None:
        self.output = os.path.join(self.system.output, "search_refine")
        os.makedirs(self.output, exist_ok=True)

    def _build_local_env(self, ligand, seed_coords_A):
        ligand.set_positions(np.asarray(seed_coords_A, dtype=np.float64))

        ligand_points = get_residue_positions(ligand.complex_structure.residues[0])
        protein_structure = self.system.protein.complex_structure

        local_structure = get_residue_subset_from_points(
            ligand_points,
            protein_structure,
            distance_cutoff=float(self._opt("sr_pocket_radius", 12.0)),
        )

        source_map = self._select_map()
        local_map = None
        if source_map is not None:
            local_map = submap_from_structure(
                local_structure,
                source_map,
                pad_A=float(self._opt("sr_map_pad_a", 3.0)),
            )

        env = ChemEMSimulationSetup(
            protein_structure=local_structure,
            ligand_structure=[ligand],
            density_map=local_map,
            platform_name=getattr(self.system, "platform", "CPU"),
            protein_restraint="protein",
            pin_k=float(self._opt("sr_pin_k", 5000.0)),
            localise=False,
            global_k=float(self._opt("sr_global_k", 100.0)),
            pin_specs=getattr(self.system.options, "pin_specs", []),
            distance_specs=getattr(self.system.options, "distance_specs", []),
        )

        st = env.simulation.context.getState(getPositions=True)
        pos_nm = np.asarray(
            st.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )
        pos_nm[env.all_ligand_indices] = np.asarray(seed_coords_A, dtype=np.float64) / 10.0
        env.simulation.context.setPositions(pos_nm * unit.nanometer)
        return env, local_map

    # ---------- ranking ----------

    def _dedupe_top_records(self, sorted_records, return_n, rmsd_thr_A):
        selected = []
        for rec in sorted_records:
            too_close = False
            for keep in selected:
                if self._rmsd(rec["ligand_coords_A"], keep["ligand_coords_A"]) < rmsd_thr_A:
                    too_close = True
                    break
            if too_close:
                continue
            selected.append(rec)
            if len(selected) >= return_n:
                break
        return selected

    def _select_final_poses(
        self,
        sorted_records,
        return_n: int,
        rmsd_thr_A: float,
        score_margin: float,
    ):
        """Single-pose by default. When ``return_n > 1``, include additional
        poses only if they are (a) within ``score_margin`` of the best AND
        (b) separated by > ``rmsd_thr_A`` RMSD from everything already selected.
        """
        if not sorted_records:
            return []
        best = sorted_records[0]
        selected = [best]
        if return_n <= 1:
            return selected
        best_score = float(best["final_score"])
        margin = max(0.0, float(score_margin))
        for rec in sorted_records[1:]:
            if len(selected) >= return_n:
                break
            # Records are sorted by final_score descending, so once we drop
            # below the margin the rest can't qualify either.
            if best_score - float(rec["final_score"]) > margin:
                break
            too_close = any(
                self._rmsd(rec["ligand_coords_A"], keep["ligand_coords_A"]) < rmsd_thr_A
                for keep in selected
            )
            if too_close:
                continue
            selected.append(rec)
        return selected

    # ---------- random kick (basin-hopping) ----------

    def _apply_random_kick(self, env, accepted_record, sigma_A: float, seed: int) -> np.ndarray:
        pos_nm = np.asarray(accepted_record["positions_nm"], dtype=np.float64).copy()
        lig_idx = np.asarray(env.ligand_heavy_indices, dtype=int)
        if lig_idx.size == 0 or sigma_A <= 0.0:
            return pos_nm

        rng = np.random.default_rng(int(seed))
        kick_nm = rng.normal(scale=float(sigma_A) * 0.1, size=(lig_idx.size, 3))
        pos_nm[lig_idx] = pos_nm[lig_idx] + kick_nm
        env.simulation.context.setPositions(pos_nm * unit.nanometer)
        return pos_nm

    def _apply_directed_kick(
        self,
        env,
        accepted_record,
        scorer,
        local_map,
        rotatable_bonds,
        mol,
        protein_heavy_indices,
        kick_angle_deg: float,
    ) -> Optional[dict]:
        """Apply a dihedral-delta kick on the worst-Q atom toward its gradient.

        Returns a description dict on success, or None if no viable dihedral
        can be built (the caller should fall back to the Gaussian kick).
        """
        if not rotatable_bonds or local_map is None:
            return None

        heavy_coords_A = np.asarray(
            accepted_record["ligand_heavy_coords_A"], dtype=np.float64
        )

        try:
            grad = scorer.atom_gradient(heavy_coords_A)
        except Exception:
            return None

        accepted_pos_nm = np.asarray(accepted_record["positions_nm"], dtype=np.float64)
        prot_xyz_A = (
            accepted_pos_nm[protein_heavy_indices] * 10.0
            if protein_heavy_indices else None
        )

        try:
            fit_q = atom_fit_quality(
                heavy_coords_A=heavy_coords_A,
                atom_gradient=grad,
                local_map=local_map,
                sigma_ref=float(self._opt("sr_qscore_sigma_ref", 0.6)),
                protein_xyz_A=prot_xyz_A,
            )
        except Exception:
            return None

        kick = directed_kick_proposal(
            heavy_coords_A=heavy_coords_A,
            atom_gradient=grad,
            q_score=fit_q.q_score,
            rotatable_bonds=rotatable_bonds,
            mol=mol,
            kick_angle_deg=float(kick_angle_deg),
        )
        if kick is None:
            return None

        lig_idx = np.asarray(env.ligand_heavy_indices, dtype=int)
        new_heavy_A = kick["new_heavy_coords_A"]
        side = np.asarray(kick["side_atoms"], dtype=int)
        pos_nm = accepted_pos_nm.copy()
        pos_nm[lig_idx[side]] = new_heavy_A[side] * 0.1
        env.simulation.context.setPositions(pos_nm * unit.nanometer)
        return {
            "bond": kick["bond"],
            "bad_atom": int(kick["bad_atom"]),
            "delta_theta_deg": float(kick["delta_theta_deg"]),
            "alignment": float(kick["alignment"]),
            "worst_q": float(fit_q.q_score[int(kick["bad_atom"])]),
        }

    # ---------- proposal relaxation ----------

    def _relax_proposal(
        self,
        env,
        accepted_pos_nm,
        target_pos_nm,
        pull_k: float,
        pre_min_iters: int,
        md_steps: int,
        md_temp_k: float,
        min_max_iters: int,
        prop_seed: int,
        log_tag: str,
    ) -> bool:
        env.set_ligand_tether(target_pos_nm, k_kcal_per_mol_A2=pull_k)
        try:
            self._log_relax_state(
                env,
                log_tag,
                "start",
                accepted_pos_nm=accepted_pos_nm,
                target_pos_nm=target_pos_nm,
            )
            bad, reason = check_pose_forces(env.simulation, env.all_ligand_indices)
            if bad:
                self.system.log(f"[search_refine] {log_tag} pre-check failed: {reason}")
                self._log_relax_state(
                    env,
                    log_tag,
                    "pre-check-failed",
                    accepted_pos_nm=accepted_pos_nm,
                    target_pos_nm=target_pos_nm,
                )
                return False

            if pre_min_iters > 0:
                try:
                    env.simulation.minimizeEnergy(maxIterations=pre_min_iters)
                    self._log_relax_state(
                        env,
                        log_tag,
                        "after-pre-min",
                        accepted_pos_nm=accepted_pos_nm,
                        target_pos_nm=target_pos_nm,
                    )
                except Exception as exc:
                    self.system.log(f"[search_refine] {log_tag} pre-min failed: {exc}")
                    self._log_relax_state(
                        env,
                        log_tag,
                        "pre-min-failed",
                        accepted_pos_nm=accepted_pos_nm,
                        target_pos_nm=target_pos_nm,
                    )
                    return False

            md_failed = False
            if md_steps > 0:
                try:
                    try:
                        env.simulation.context.setVelocitiesToTemperature(
                            md_temp_k * unit.kelvin, prop_seed
                        )
                    except TypeError:
                        env.simulation.context.setVelocitiesToTemperature(
                            md_temp_k * unit.kelvin
                        )
                    env.simulation.step(md_steps)
                    self._log_relax_state(
                        env,
                        log_tag,
                        "after-md",
                        accepted_pos_nm=accepted_pos_nm,
                        target_pos_nm=target_pos_nm,
                    )
                except Exception as exc:
                    md_failed = True
                    self.system.log(
                        f"[search_refine] {log_tag} md failed ({exc}); retrying minimization-only"
                    )
                    self._log_relax_state(
                        env,
                        log_tag,
                        "md-failed",
                        accepted_pos_nm=accepted_pos_nm,
                        target_pos_nm=target_pos_nm,
                    )

            if (not md_failed) and (not self._positions_are_finite(env)):
                md_failed = True
                self.system.log(
                    f"[search_refine] {log_tag} md produced non-finite coordinates; "
                    "retrying minimization-only"
                )
                self._log_relax_state(
                    env,
                    log_tag,
                    "after-md-non-finite",
                    accepted_pos_nm=accepted_pos_nm,
                    target_pos_nm=target_pos_nm,
                )

            def _minimize_from_accepted(failure_label: str) -> bool:
                try:
                    env.simulation.context.setPositions(accepted_pos_nm * unit.nanometer)
                    env.simulation.minimizeEnergy(maxIterations=min_max_iters)
                    self._log_relax_state(
                        env,
                        log_tag,
                        "after-min-fallback",
                        accepted_pos_nm=accepted_pos_nm,
                        target_pos_nm=target_pos_nm,
                    )
                    return True
                except Exception as exc:
                    self.system.log(f"[search_refine] {log_tag} {failure_label}: {exc}")
                    self._log_relax_state(
                        env,
                        log_tag,
                        "min-fallback-failed",
                        accepted_pos_nm=accepted_pos_nm,
                        target_pos_nm=target_pos_nm,
                    )
                    return False

            if md_failed:
                if not _minimize_from_accepted("minimization-only fallback failed"):
                    return False
            else:
                try:
                    env.simulation.minimizeEnergy(maxIterations=min_max_iters)
                    self._log_relax_state(
                        env,
                        log_tag,
                        "after-post-min",
                        accepted_pos_nm=accepted_pos_nm,
                        target_pos_nm=target_pos_nm,
                    )
                except Exception as exc:
                    self.system.log(
                        f"[search_refine] {log_tag} post-min failed: {exc}; "
                        "retrying minimization-only from accepted"
                    )
                    self._log_relax_state(
                        env,
                        log_tag,
                        "post-min-failed",
                        accepted_pos_nm=accepted_pos_nm,
                        target_pos_nm=target_pos_nm,
                    )
                    if not _minimize_from_accepted("post-min fallback failed"):
                        return False

            if not self._positions_are_finite(env):
                self.system.log(
                    f"[search_refine] {log_tag} non-finite coordinates after relax; discarding"
                )
                self._log_relax_state(
                    env,
                    log_tag,
                    "after-relax-non-finite",
                    accepted_pos_nm=accepted_pos_nm,
                    target_pos_nm=target_pos_nm,
                )
                try:
                    env.simulation.context.setPositions(accepted_pos_nm * unit.nanometer)
                except Exception:
                    pass
                return False
        finally:
            env.clear_ligand_tether()
        return True

    # ---------- per-ligand driver ----------

    def _run_ligand_refinement(self, lig_idx: int, ligand) -> Optional[dict]:
        if ligand.mol.GetNumConformers() == 0:
            self.system.log(f"[search_refine] ligand {lig_idx}: no conformers found, skipping")
            return None

        t0 = time.perf_counter()

        seed_coords_A = np.asarray(ligand.mol.GetConformer(0).GetPositions(), dtype=np.float64)
        env, local_map = self._build_local_env(ligand, seed_coords_A)

        if local_map is None:
            self.system.log(
                Messages.chemem_warning(
                    self.__class__.__name__,
                    "_run_ligand_refinement",
                    f"ligand {lig_idx}: no map available after map-source selection",
                )
            )
            return None

        protein_heavy_indices = [
            atom.idx
            for atom in env.complex_structure.atoms
            if (not str(atom.residue.name).startswith("LIG")) and (atom.element != 1)
        ]
        heavy_masses = np.asarray(
            [
                env.complex_system.getParticleMass(int(idx)).value_in_unit(unit.dalton)
                for idx in env.ligand_heavy_indices
            ],
            dtype=np.float64,
        )

        scorer_name = str(self._opt("sr_scorer", "sci")).lower()
        scorer = get_scorer(scorer_name, self.system.options)
        scorer.prepare(env, local_map, heavy_masses, protein_heavy_indices)

        acceptor_name = str(self._opt("sr_accept_strategy", "greedy")).lower()
        acceptor = get_acceptance(acceptor_name, self.system.options)

        initial_record = self._capture_record(
            env, scorer, protein_heavy_indices,
            outer_iter=0, proposal_idx=0,
        )
        if initial_record is None:
            self.system.log(f"[search_refine] ligand {lig_idx}: initial state unstable, skipping")
            return None

        all_records = [initial_record]
        accepted_record = initial_record

        max_outer_iter = int(self._opt("sr_max_outer_iter", 50))
        patience = int(self._opt("sr_patience", 8))
        proposals_per_iter = int(self._opt("sr_proposals_per_iter", 4))
        md_steps = int(self._opt("sr_md_steps_per_iter", 250))
        min_max_iters = int(self._opt("sr_minimise_max_iters", 200))
        pre_min_iters = int(max(0, min(min_max_iters, max(20, min_max_iters // 4))))
        md_temp_k = float(self._opt("sr_md_temp_k", 150.0))
        base_seed = int(self._opt("sr_seed", 1))

        max_atom_delta_A = float(self._opt("sr_max_atom_delta_a", 0.5))
        pull_k = float(self._opt("sr_trust_k", 5.0))
        basin_sigma_A = float(self._opt("sr_basin_hop_sigma_a", 0.3))

        stage = str(self._opt("sr_stage", "v2")).lower()
        is_legacy = stage == "legacy"

        dihedral_per_iter = (
            0 if is_legacy else int(self._opt("sr_dihedral_proposals_per_iter", 0))
        )
        directed_kick_enabled = (
            False if is_legacy else bool(self._opt("sr_directed_kick", False))
        )
        rotatable_bonds: List[tuple] = []
        if dihedral_per_iter > 0 or directed_kick_enabled:
            try:
                rotatable_bonds = enumerate_rotatable_heavy_bonds(ligand.mol)
            except Exception as exc:
                self.system.log(
                    f"[search_refine] ligand {lig_idx}: rotatable-bond enumeration failed "
                    f"({exc}); dihedral proposals and directed kicks disabled"
                )
                rotatable_bonds = []

        subregion_per_iter = (
            0 if is_legacy else int(self._opt("sr_subregion_proposals_per_iter", 0))
        )
        subregion_min_size = int(self._opt("sr_subregion_min_size", 3))
        subregion_max_size = int(self._opt("sr_subregion_max_size", 8))

        self._apply_hybrid_scores([accepted_record])
        self._sr_log(
            f"[search_refine][v] ligand {lig_idx}: scorer={scorer_name} accept={acceptor_name} "
            f"max_outer_iter={max_outer_iter} patience={patience} "
            f"proposals_per_iter={proposals_per_iter} md_steps={md_steps} "
            f"pre_min_iters={pre_min_iters} "
            f"max_atom_delta_A={max_atom_delta_A:.3f} pull_k={pull_k:.3f} "
            f"log_every={self._sr_log_every()}",
            force=True,
        )
        self._sr_log(
            f"[search_refine][v] ligand {lig_idx}: seed {self._fmt_record(accepted_record)}",
            force=True,
        )

        stale_iters = 0
        outer_completed = 0
        n_accept_major = 0
        n_accept_micro = 0
        n_perturb = 0

        for outer in range(1, max_outer_iter + 1):
            if stale_iters >= patience:
                self._sr_log(
                    f"[search_refine][v] ligand {lig_idx}: early stop at iter {outer - 1} "
                    f"(stale={stale_iters}/{patience})",
                    force=True,
                )
                break

            outer_completed = outer
            accepted_pos_nm = np.asarray(accepted_record["positions_nm"], dtype=np.float64)
            env.simulation.context.setPositions(accepted_pos_nm * unit.nanometer)

            heavy_coords_A = np.asarray(
                accepted_record["ligand_heavy_coords_A"], dtype=np.float64
            )
            try:
                grad = scorer.atom_gradient(heavy_coords_A)
            except Exception as exc:
                self.system.log(
                    f"[search_refine] ligand {lig_idx}: iter {outer} gradient eval failed ({exc})"
                )
                stale_iters += 1
                continue

            norms = np.linalg.norm(grad, axis=1) if grad.size else np.array([0.0])
            g_max = float(np.max(norms)) if norms.size else 0.0
            g_mean = float(np.mean(norms)) if norms.size else 0.0
            if (not np.isfinite(g_max)) or g_max <= 1e-12:
                stale_iters += 1
                self._sr_log(
                    f"[search_refine][v] ligand {lig_idx}: iter {outer} "
                    f"metric gradient near-zero; stale={stale_iters}/{patience}",
                    outer_iter=outer,
                )
                continue

            self._sr_log(
                f"[search_refine][v] ligand {lig_idx}: iter {outer}/{max_outer_iter} "
                f"start stale={stale_iters}/{patience} "
                f"grad(mean={g_mean:.3g}, max={g_max:.3g}) "
                f"accepted {self._fmt_record(accepted_record)}",
                outer_iter=outer,
            )

            want_diag = (not is_legacy) and bool(self._opt("sr_diagnostic", False))
            want_target_bad = (not is_legacy) and bool(self._opt("sr_target_bad_only", False))
            want_dihedral = dihedral_per_iter > 0 and bool(rotatable_bonds)
            want_subregion = subregion_per_iter > 0
            bad_mask_for_iter: Optional[np.ndarray] = None

            fit_q = None
            classification = None
            need_diag = want_diag or want_target_bad or want_dihedral or want_subregion
            if need_diag and local_map is not None:
                try:
                    prot_xyz_A = (
                        accepted_pos_nm[protein_heavy_indices] * 10.0
                        if protein_heavy_indices else None
                    )
                    fit_q = atom_fit_quality(
                        heavy_coords_A=heavy_coords_A,
                        atom_gradient=grad,
                        local_map=local_map,
                        sigma_ref=float(self._opt("sr_qscore_sigma_ref", 0.6)),
                        protein_xyz_A=prot_xyz_A,
                    )
                    classification = classify_atoms(
                        fit_q,
                        q_good_thresh=float(self._opt("sr_q_good_thresh", 0.7)),
                        q_bad_thresh=float(self._opt("sr_q_bad_thresh", 0.3)),
                    )
                    if want_diag and self._sr_verbose():
                        self._sr_log(
                            f"[search_refine][v] ligand {lig_idx}: iter {outer} "
                            + format_diagnostic(fit_q, classification),
                            outer_iter=outer,
                            force=True,
                        )
                    if want_target_bad and classification.bad_idx.size > 0:
                        bad_mask_for_iter = classification.bad_idx.astype(int)
                except Exception as exc:
                    if self._sr_verbose():
                        self._sr_log(
                            f"[search_refine][v] ligand {lig_idx}: iter {outer} "
                            f"diagnostic failed: {exc}",
                            outer_iter=outer,
                        )
                    fit_q = None
                    classification = None

            dihedral_triples: List[dict] = []
            if want_dihedral and classification is not None and classification.bad_idx.size > 0:
                bad_sorted = sorted(
                    classification.bad_idx.tolist(),
                    key=lambda i: -float(fit_q.badness[i]),
                )
                dihedral_triples = self._enumerate_dihedral_triples(
                    bad_sorted, rotatable_bonds, ligand.mol,
                    heavy_coords_A, grad, max_count=dihedral_per_iter,
                )

            subregion_specs: List[dict] = []
            if want_subregion and classification is not None and classification.bad_idx.size > 0:
                try:
                    clusters = cluster_bad_atoms(classification.bad_idx, ligand.mol)
                except Exception as exc:
                    clusters = []
                    if self._sr_verbose():
                        self._sr_log(
                            f"[search_refine][v] ligand {lig_idx}: iter {outer} "
                            f"cluster_bad_atoms failed: {exc}",
                            outer_iter=outer,
                        )
                valid_clusters = [
                    c for c in clusters
                    if subregion_min_size <= c.size <= subregion_max_size
                ]
                # Rank clusters by total badness so we target the worst fits first.
                valid_clusters.sort(
                    key=lambda c: -float(np.sum(fit_q.badness[c])),
                )
                for cluster in valid_clusters[:subregion_per_iter]:
                    grad_rows = grad[cluster]
                    norms = np.linalg.norm(grad_rows, axis=1)
                    dirs = np.zeros_like(grad_rows)
                    nz = norms > 1e-12
                    dirs[nz] = grad_rows[nz] / norms[nz, None]
                    if not np.any(nz):
                        continue
                    try:
                        R_mat, t_vec = best_rigid_transform_for_cluster(
                            cluster_coords_A=heavy_coords_A[cluster],
                            target_dirs=dirs,
                            epsilon_A=max_atom_delta_A,
                        )
                    except Exception:
                        continue
                    subregion_specs.append(
                        {
                            "cluster": cluster,
                            "R": R_mat,
                            "t": t_vec,
                        }
                    )

            proposals: List[dict] = []
            n_dihedral = len(dihedral_triples)
            n_subregion = len(subregion_specs)
            for prop_idx in range(1, proposals_per_iter + 1):
                env.simulation.context.setPositions(accepted_pos_nm * unit.nanometer)
                prop_seed = base_seed + (outer * 1000) + prop_idx

                slot = prop_idx - 1
                if slot < n_dihedral:
                    desc = dihedral_triples[slot]
                    prop_mode = (
                        f"dihedral(bond={desc['bond']},atom={desc['bad_atom']},"
                        f"dtheta={desc['delta_theta_deg']:+.0f}deg)"
                    )
                    prop_scale = 1.0
                    target_pos_nm, move_stats = build_targets_from_dihedral(
                        accepted_pos_nm=accepted_pos_nm,
                        ligand_heavy_idx=np.asarray(env.ligand_heavy_indices, dtype=int),
                        heavy_coords_A=heavy_coords_A,
                        bond=desc["bond"],
                        delta_theta=float(desc["delta_theta_rad"]),
                        side_atoms=desc["side"],
                    )
                    move_stats["cap_A"] = float(move_stats.get("max_target_disp_A", 0.0))
                elif slot - n_dihedral < n_subregion:
                    spec = subregion_specs[slot - n_dihedral]
                    prop_mode = f"subregion(n={spec['cluster'].size})"
                    prop_scale = 1.0
                    target_pos_nm, move_stats = build_targets_from_subregion(
                        accepted_pos_nm=accepted_pos_nm,
                        ligand_heavy_idx=np.asarray(env.ligand_heavy_indices, dtype=int),
                        heavy_coords_A=heavy_coords_A,
                        cluster_atoms=spec["cluster"],
                        R=spec["R"],
                        t=spec["t"],
                    )
                    move_stats["cap_A"] = float(move_stats.get("max_target_disp_A", 0.0))
                else:
                    prop_scale = self._proposal_scale(prop_idx, proposals_per_iter)
                    prop_mode = self._proposal_mode(prop_idx, proposals_per_iter)
                    target_pos_nm, move_stats = build_targets_from_gradient(
                        accepted_pos_nm=accepted_pos_nm,
                        ligand_heavy_idx=np.asarray(env.ligand_heavy_indices, dtype=int),
                        grad_heavy=grad,
                        cap_A=max_atom_delta_A,
                        proposal_scale=prop_scale,
                        proposal_mode=prop_mode,
                        rng=np.random.default_rng(prop_seed),
                        bad_mask=bad_mask_for_iter,
                    )

                self._sr_log(
                    f"[search_refine][v] ligand {lig_idx}: iter {outer} "
                    f"proposal {prop_idx}/{proposals_per_iter} begin "
                    f"mode={prop_mode} scale={prop_scale:.3f} cap={move_stats['cap_A']:.3f}A "
                    f"moved={move_stats['moved_atoms']}",
                    outer_iter=outer,
                )

                if move_stats["moved_atoms"] <= 0:
                    self._sr_log(
                        f"[search_refine][v] ligand {lig_idx}: iter {outer} proposal {prop_idx} "
                        f"has no movable atoms under current cap",
                        outer_iter=outer,
                    )
                    continue

                log_tag = f"ligand {lig_idx}: iter {outer} proposal {prop_idx}"
                ok = self._relax_proposal(
                    env,
                    accepted_pos_nm,
                    target_pos_nm,
                    pull_k=pull_k,
                    pre_min_iters=pre_min_iters,
                    md_steps=md_steps,
                    md_temp_k=md_temp_k,
                    min_max_iters=min_max_iters,
                    prop_seed=prop_seed,
                    log_tag=log_tag,
                )
                if not ok:
                    continue

                rec = self._capture_record(
                    env, scorer, protein_heavy_indices,
                    outer_iter=outer, proposal_idx=prop_idx,
                )
                if rec is None:
                    continue

                proposals.append(rec)
                all_records.append(rec)
                self._sr_log(
                    f"[search_refine][v] ligand {lig_idx}: iter {outer} proposal {prop_idx} "
                    f"target_disp(mean={move_stats['mean_target_disp_A']:.3f}A, "
                    f"max={move_stats['max_target_disp_A']:.3f}A) "
                    f"{self._fmt_record(rec)}",
                    outer_iter=outer,
                )

            if not proposals:
                stale_iters += 1
                self._sr_log(
                    f"[search_refine][v] ligand {lig_idx}: iter {outer} "
                    f"no valid proposals (stale={stale_iters}/{patience})",
                    outer_iter=outer,
                )
                continue

            picked, outcome = acceptor.decide(accepted_record, proposals, outer)

            if outcome == AcceptOutcome.MAJOR:
                accepted_record = picked
                n_accept_major += 1
                stale_iters = 0
                self._sr_log(
                    f"[search_refine][v] ligand {lig_idx}: iter {outer} accepted (major) "
                    f"proposal {accepted_record['proposal_idx']} "
                    f"{self._fmt_record(accepted_record)}",
                    outer_iter=outer,
                )
            elif outcome == AcceptOutcome.MICRO:
                accepted_record = picked
                n_accept_micro += 1
                # A micro-accept is still forward progress (small positive delta),
                # so do not advance stale-count toward early stopping.
                stale_iters = max(0, stale_iters - 1)
                self._sr_log(
                    f"[search_refine][v] ligand {lig_idx}: iter {outer} micro-accept "
                    f"proposal {accepted_record['proposal_idx']} "
                    f"stale={stale_iters}/{patience} "
                    f"{self._fmt_record(accepted_record)}",
                    outer_iter=outer,
                )
            elif outcome == AcceptOutcome.PERTURB:
                if picked is not None and picked is not accepted_record:
                    accepted_record = picked
                n_perturb += 1
                kick_seed = base_seed + (outer * 7919)

                kick_info = None
                if directed_kick_enabled and rotatable_bonds:
                    kick_info = self._apply_directed_kick(
                        env=env,
                        accepted_record=accepted_record,
                        scorer=scorer,
                        local_map=local_map,
                        rotatable_bonds=rotatable_bonds,
                        mol=ligand.mol,
                        protein_heavy_indices=protein_heavy_indices,
                        kick_angle_deg=float(self._opt("sr_directed_kick_angle_deg", 90.0)),
                    )

                if kick_info is None:
                    self._apply_random_kick(env, accepted_record, basin_sigma_A, kick_seed)
                    kick_desc = f"gaussian sigma={basin_sigma_A:.3f}A"
                else:
                    kick_desc = (
                        f"directed bond={kick_info['bond']} atom={kick_info['bad_atom']} "
                        f"q={kick_info['worst_q']:.3f} "
                        f"dtheta={kick_info['delta_theta_deg']:+.0f}deg"
                    )

                kicked = self._capture_record(
                    env, scorer, protein_heavy_indices,
                    outer_iter=outer, proposal_idx=-1,
                )
                if kicked is not None:
                    accepted_record = kicked
                    all_records.append(kicked)
                stale_iters = 0
                self._sr_log(
                    f"[search_refine][v] ligand {lig_idx}: iter {outer} perturb kick "
                    f"{kick_desc} {self._fmt_record(accepted_record)}",
                    outer_iter=outer,
                )
            else:
                stale_iters += 1
                self._sr_log(
                    f"[search_refine][v] ligand {lig_idx}: iter {outer} no accept "
                    f"stale={stale_iters}/{patience}",
                    outer_iter=outer,
                )

        # Global reranking over all explored states.
        self._apply_hybrid_scores(all_records)
        ranked = sorted(all_records, key=lambda r: r["final_score"], reverse=True)

        return_n = int(self._opt("sr_return_n", 1))
        rmsd_thr_A = float(self._opt("sr_rmsd_dedupe", 0.5))
        score_margin = float(self._opt("sr_return_score_margin", 0.01))
        selected = self._select_final_poses(
            ranked,
            return_n=return_n,
            rmsd_thr_A=rmsd_thr_A,
            score_margin=score_margin,
        )
        if not selected:
            selected = [ranked[0]]

        best = selected[0]

        env.simulation.context.setPositions(
            np.asarray(best["positions_nm"], dtype=np.float64) * unit.nanometer
        )
        env.complex_structure.positions = (
            np.asarray(best["positions_nm"], dtype=np.float64) * unit.nanometer
        )

        update_global_positions(
            full_structure=self.system.protein.complex_structure,
            local_structure=env.complex_structure,
        )
        update_ligand_positions(
            local_structure=env.complex_structure,
            ligand_objects=[ligand],
        )
        ligand.set_positions(np.asarray(best["ligand_coords_A"], dtype=np.float64))

        ligand.docked = [
            RefinedPose(
                score=-float(rec["final_score"]),
                position=np.asarray(rec["ligand_coords_A"], dtype=np.float64),
                final_score=float(rec["final_score"]),
                sci=float(rec["score"]),
                energy_kcal=float(rec["energy_kcal"]),
                clash_penalty=float(rec["clash_penalty"]),
            )
            for rec in selected
        ]

        write_ligand_outputs(
            self.output, lig_idx, ligand, scorer_name, selected, all_records,
        )

        dt = time.perf_counter() - t0
        summary = {
            "ligand_index": int(lig_idx),
            "scorer": scorer_name,
            "acceptance": acceptor_name,
            "n_explored": int(len(all_records)),
            "n_selected": int(len(selected)),
            "best_final_score": float(best["final_score"]),
            "best_score": float(best["score"]),
            "outer_completed": int(outer_completed),
            "n_accept_major": int(n_accept_major),
            "n_accept_micro": int(n_accept_micro),
            "n_perturb": int(n_perturb),
            "runtime_s": float(dt),
        }
        self._sr_log(
            f"[search_refine][v] ligand {lig_idx}: done outer_completed={outer_completed} "
            f"explored={len(all_records)} accept_major={n_accept_major} "
            f"accept_micro={n_accept_micro} perturb={n_perturb} "
            f"best {self._fmt_record(best)} runtime={dt:.2f}s",
            force=True,
        )
        return summary

    # ---------- entry points ----------

    def write_output(self) -> None:
        pdb_out = os.path.join(self.output, "search_refine_receptor.pdb")
        save_structure_parmed(self.system.protein.complex_structure, pdb_out)

        for lig_idx, ligand in enumerate(self.system.ligand):
            sdf_out = os.path.join(self.output, f"Ligand_{lig_idx}_best.sdf")
            ligand.write_sdf(sdf_out)

    def run(self) -> None:
        self.system.log(Messages.create_centered_box("Search Refine"))
        self._prepare_output()

        summaries: List[dict] = []
        for lig_idx, ligand in enumerate(self.system.ligand):
            self.system.log(f"[search_refine] ligand {lig_idx}: start")
            try:
                result = self._run_ligand_refinement(lig_idx, ligand)
            except Exception as exc:
                self.system.log(
                    Messages.chemem_warning(
                        self.__class__.__name__,
                        "_run_ligand_refinement",
                        f"ligand {lig_idx}: {exc}",
                    )
                )
                continue

            if result is None:
                self.system.log(f"[search_refine] ligand {lig_idx}: skipped")
                continue

            summaries.append(result)
            self.system.log(
                f"[search_refine] ligand {lig_idx}: best final={result['best_final_score']:.6f} "
                f"score={result['best_score']:.6f} explored={result['n_explored']}"
            )

        write_summary(self.output, summaries)
        self.write_output()
