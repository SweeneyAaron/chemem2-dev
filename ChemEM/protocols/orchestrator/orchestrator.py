# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""SmartOrchestrator — 3-gate triage funnel.

Stages, in order:
  DOCK (N x M)  ->  Gate 1 (Q-score)  ->  REFINE  ->  Gate 2 (Q + MMGBSA)
                ->  SEARCH_REFINE  ->  Gate 3 (final pick)  ->  ASSEMBLE
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional

import numpy as np
from openmm import unit

from ChemEM.messages import Messages
from ChemEM.protocols._docking.docking import Docking
from ChemEM.protocols.refine.minimize import Refine
from ChemEM.protocols.refine.search_refine import SearchRefine

from . import io as orch_io
from . import scoring
from . import triage
from .state import PoseCandidate


class SmartOrchestrator:
    """Multi-site, multi-ligand assembly via a 3-gate score-driven funnel."""

    def __init__(self, system):
        self.system = system
        self.output: Optional[str] = None
        self._docking_instance: Optional[Docking] = None

    # -------- option helpers --------
    def _opt(self, name: str, default):
        return getattr(self.system.options, name, default)

    def _log(self, msg: str) -> None:
        self.system.log(msg)

    # -------- entry point --------
    def run(self) -> None:
        self._log(Messages.create_centered_box("Smart Orchestrator"))
        self._setup_output()

        if not self._segmentation_ready():
            self._log(
                "[orchestrator] No binding sites or alpha-mask segments available; "
                "writing receptor-only assembly and exiting."
            )
            self._write_receptor_only()
            return

        if self.system.density_map is None:
            self._log(
                "[orchestrator] No density map is loaded. Q-score is the spine of "
                "the funnel; aborting orchestrator run. Use a different protocol "
                "selection if you intended to run map-free."
            )
            return

        # 1. Dock everything (N x M).
        candidates_by_site = self._dock_and_collect()
        if not any(candidates_by_site.values()):
            self._log("[orchestrator] No docked poses produced; nothing to assemble.")
            self._write_receptor_only()
            return

        # 2. Gate 1 -- Q-score triage.
        self._score_qscore(candidates_by_site)
        gate1 = triage.gate1_select(candidates_by_site, self._opt("orch_gate1_topk", 5))
        orch_io.write_gate_json(gate1, os.path.join(self.output, "gate1.json"))
        self._log(self._gate_summary("Gate 1 (Q-score)", gate1))

        # 3. Cheap refine on Gate 1 survivors.
        self._refine_candidates(gate1)

        # 4. Gate 2 -- Q-score + single-frame MMGBSA.
        self._score_qscore(gate1)
        if not bool(self._opt("orch_skip_mmgbsa", False)):
            self._score_mmgbsa(gate1)
        gate2 = triage.gate2_select(
            gate1,
            self._opt("orch_gate2_topk", 2),
            self._opt("orch_w_qscore", 1.0),
            self._opt("orch_w_mmgbsa", 0.5),
        )
        orch_io.write_gate_json(gate2, os.path.join(self.output, "gate2.json"))
        self._log(self._gate_summary("Gate 2 (Q + MMGBSA)", gate2))

        # 5. SearchRefine on Gate 2 survivors.
        if not bool(self._opt("orch_skip_search_refine", False)):
            self._search_refine_candidates(gate2)

        # 6. Gate 3 -- final pick.
        self._score_qscore(gate2)
        if not bool(self._opt("orch_skip_mmgbsa", False)):
            self._score_mmgbsa(gate2)
        assignments = triage.gate3_select(
            gate2,
            self._opt("orch_w_qscore", 1.0),
            self._opt("orch_w_mmgbsa", 0.5),
        )
        orch_io.write_gate_json(gate2, os.path.join(self.output, "gate3.json"))

        # 7. Assemble.
        orch_io.write_assignments_json(
            assignments, os.path.join(self.output, "assignments.json")
        )
        orch_io.write_summary_json(
            assignments,
            gate_counts={
                site: {
                    "gate1": len(gate1.get(site, [])),
                    "gate2": len(gate2.get(site, [])),
                }
                for site in candidates_by_site.keys()
            },
            path=os.path.join(self.output, "summary.json"),
        )
        orch_io.write_assignment_sdfs(assignments, self.system.ligand, self.output)
        orch_io.write_final_complex_pdb(
            assignments,
            self.system,
            os.path.join(self.output, "final_complex.pdb"),
        )
        self._log(
            f"[orchestrator] Wrote {len(assignments)} assignment(s) to "
            f"{os.path.join(self.output, 'final_complex.pdb')}"
        )

    # -------- internals --------
    def _setup_output(self) -> None:
        base = getattr(self.system, "output", None) or "."
        self.output = os.path.join(base, "orchestrator")
        os.makedirs(self.output, exist_ok=True)

    def _segmentation_ready(self) -> bool:
        sites = getattr(self.system, "binding_sites", None) or {}
        maps = getattr(self.system, "binding_site_maps", None) or {}
        return bool(sites) and bool(maps)

    def _write_receptor_only(self) -> None:
        from ChemEM.parsers.writers import save_structure_parmed
        save_structure_parmed(
            self.system.protein.complex_structure,
            os.path.join(self.output, "final_complex.pdb"),
        )

    def _dock_and_collect(self) -> Dict[str, List[PoseCandidate]]:
        """Run Docking and convert its per-(site, ligand) SiteResult list into
        a per-site PoseCandidate dict (preserves which site each pose came from)."""
        self._log("[orchestrator] Running Docking stage (N x M).")
        dock = Docking(self.system)
        # Force --rescore off; MMGBSA happens at Gate 2/3, not the dock stage.
        prev_rescore = self._opt("rescore", False)
        try:
            self.system.options.rescore = False
            dock.run()
        finally:
            self.system.options.rescore = prev_rescore
        self._docking_instance = dock

        candidates_by_site: Dict[str, List[PoseCandidate]] = {
            str(site_id): [] for site_id in self.system.binding_sites.keys()
        }
        for sr in dock._site_results:
            site_key = str(sr.site_id)
            candidates_by_site.setdefault(site_key, [])
            for pose_idx, (score, coords) in enumerate(sr.poses):
                arr = np.asarray(coords, dtype=float)
                if arr.size == 0 or not np.all(np.isfinite(arr)):
                    continue
                candidates_by_site[site_key].append(
                    PoseCandidate(
                        site_id=site_key,
                        ligand_idx=int(sr.ligand_idx),
                        pose_idx=int(pose_idx),
                        coords=arr,
                        dock_score=float(score),
                        stage="docked",
                    )
                )
        n_total = sum(len(v) for v in candidates_by_site.values())
        self._log(
            f"[orchestrator] Built {n_total} pose candidate(s) across "
            f"{len(candidates_by_site)} site(s)."
        )
        return candidates_by_site

    # ---- scoring ----
    def _score_qscore(self, candidates_by_site: Dict[str, List[PoseCandidate]]) -> None:
        sigma_ref = float(self._opt("sigma_ref", 0.6))
        for site_id, candidates in candidates_by_site.items():
            for c in candidates:
                ligand = self.system.ligand[c.ligand_idx]
                c.qscore = scoring.qscore_pose(
                    c.coords, ligand.mol, self.system.density_map, sigma_ref=sigma_ref
                )

    def _score_mmgbsa(self, candidates_by_site: Dict[str, List[PoseCandidate]]) -> None:
        for site_id, candidates in candidates_by_site.items():
            for c in candidates:
                ligand = self.system.ligand[c.ligand_idx]
                ps = scoring.mmgbsa_single_frame(
                    c.coords, ligand, self.system.protein, pose_idx=c.pose_idx
                )
                if ps is None:
                    c.mmgbsa = None
                    c.notes.append("mmgbsa_failed")
                else:
                    c.mmgbsa = float(ps.deltaG)

    # ---- refine wrappers ----
    def _refine_candidates(self, candidates_by_site: Dict[str, List[PoseCandidate]]) -> None:
        self._log("[orchestrator] Refining Gate 1 survivors (cheap minimisation).")
        for site_id, candidates in candidates_by_site.items():
            for c in candidates:
                self._refine_one(c, refiner_cls=Refine)

    def _search_refine_candidates(
        self, candidates_by_site: Dict[str, List[PoseCandidate]]
    ) -> None:
        self._log("[orchestrator] SearchRefine on Gate 2 survivors.")
        for site_id, candidates in candidates_by_site.items():
            for c in candidates:
                self._refine_one(c, refiner_cls=SearchRefine)

    def _refine_one(self, candidate: PoseCandidate, refiner_cls) -> None:
        """Run one refiner (Refine or SearchRefine) on a single candidate.

        Snapshots & restores the live system state that the refiner mutates:
        - system.ligand (replaced with [the_one_ligand] so Refine processes only it)
        - system.options.local_refine (forced True so per-ligand env is built)
        - protein and ligand parmed positions (so subsequent candidates start clean)
        """
        ligand = self.system.ligand[candidate.ligand_idx]

        snap_ligands = self.system.ligand
        snap_local_refine = getattr(self.system.options, "local_refine", False)
        snap_protein_pos = copy.deepcopy(self.system.protein.complex_structure.positions)
        snap_lig_pos = copy.deepcopy(ligand.complex_structure.positions)
        snap_mol_coords = ligand.mol.GetConformer(0).GetPositions().copy()

        try:
            ligand.set_positions(np.asarray(candidate.coords, dtype=float))
            self.system.ligand = [ligand]
            self.system.options.local_refine = True

            refiner_cls(self.system).run()

            new_coords = ligand.mol.GetConformer(0).GetPositions()
            arr = np.asarray(new_coords, dtype=float)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                candidate.notes.append(f"{refiner_cls.__name__}_diverged")
            else:
                candidate.coords = arr
                candidate.stage = (
                    "search_refined" if refiner_cls is SearchRefine else "refined"
                )
        except Exception as exc:
            candidate.notes.append(f"{refiner_cls.__name__}_error:{type(exc).__name__}")
        finally:
            self.system.ligand = snap_ligands
            self.system.options.local_refine = snap_local_refine
            self.system.protein.complex_structure.positions = snap_protein_pos
            ligand.complex_structure.positions = snap_lig_pos
            # Reset rdkit conformer so cross-candidate state is clean.
            from rdkit.Geometry import Point3D
            conf = ligand.mol.GetConformer(0)
            for i, (x, y, z) in enumerate(snap_mol_coords):
                conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

    # ---- logging helpers ----
    def _gate_summary(self, name: str, by_site: Dict[str, List[PoseCandidate]]) -> str:
        lines = [f"[orchestrator] {name} survivors per site:"]
        for site_id in sorted(by_site.keys(), key=str):
            cs = by_site[site_id]
            best_q = max((c.qscore for c in cs if c.qscore is not None), default=None)
            best_m = min((c.mmgbsa for c in cs if c.mmgbsa is not None), default=None)
            q_str = "n/a" if best_q is None else f"{best_q:.3f}"
            m_str = "n/a" if best_m is None else f"{best_m:.2f}"
            lines.append(
                f"  - {site_id}: n={len(cs)}, best_qscore={q_str}, best_mmgbsa={m_str}"
            )
        return "\n".join(lines)
