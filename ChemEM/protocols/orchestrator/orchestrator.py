# This file is part of the ChemEM software.
#
# Copyright (c) 2026 - Topf Group & Leibniz Institute for Virology (LIV),
# Hamburg, Germany.
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""SmartOrchestrator — 3-gate triage funnel.

Stages, in order:
  DOCK (N x M)  ->  Gate 1 (map fit)  ->  REFINE  ->  Gate 2 (map fit + MMGBSA)
                ->  SMART_REFINE_2  ->  Gate 3 (final pick)  ->  ASSEMBLE
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
#from ChemEM.protocols.refine.search_refine import SearchRefine

from . import io as orch_io
from . import scoring
from . import triage
from .state import PoseCandidate


SmartRefine2 = None
SearchRefine = None


def _smart_refine2_class():
    global SmartRefine2
    if SmartRefine2 is None:
        from ChemEM.protocols.smart_refine_2.smart_refine import (
            SmartRefine2 as _SmartRefine2,
        )

        SmartRefine2 = _SmartRefine2
    return SmartRefine2


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
        self._audit_maps()

        # 1. Dock everything (N x M).
        candidates_by_site = self._dock_and_collect()
        if not any(candidates_by_site.values()):
            self._log("[orchestrator] No docked poses produced; nothing to assemble.")
            self._write_receptor_only()
            return
        self._audit_candidates("00_docked_raw", candidates_by_site)

        # 2. Gate 1 -- map-fit triage.
        self._score_qscore(candidates_by_site)
        self._score_density_fit(candidates_by_site, stage_name="gate1")
        self._assign_rank_scores(candidates_by_site, include_mmgbsa=False)
        self._audit_candidates("01_gate1_scored", candidates_by_site)
        gate1 = triage.gate1_select(
            candidates_by_site,
            self._opt("orch_gate1_topk", 5),
            **self._rank_kwargs(include_mmgbsa=False),
        )
        self._audit_candidates("02_gate1_survivors", gate1, selected_stage=True)
        orch_io.write_gate_json(gate1, os.path.join(self.output, "gate1.json"))
        self._log(self._gate_summary(self._gate_label(1, include_mmgbsa=False), gate1))

        # 3. Cheap refine on Gate 1 survivors.
        self._refine_candidates(gate1)
        self._clear_scores(gate1)
        self._audit_candidates("03_after_cheap_refine", gate1)

        # 4. Gate 2 -- map fit + optional single-frame MMGBSA.
        self._score_qscore(gate1)
        self._score_density_fit(gate1, stage_name="gate2")
        if not bool(self._opt("orch_skip_mmgbsa", False)):
            self._score_mmgbsa(gate1)
        self._assign_rank_scores(gate1, include_mmgbsa=True)
        self._audit_candidates("04_gate2_scored", gate1)
        gate2 = triage.gate2_select(
            gate1,
            self._opt("orch_gate2_topk", 2),
            **self._rank_kwargs(include_mmgbsa=True),
            same_ligand_mmgbsa_window=float(
                self._opt("orch_mmgbsa_pose_window", 0.15)
            ),
        )
        self._audit_candidates("05_gate2_survivors", gate2, selected_stage=True)
        orch_io.write_gate_json(gate2, os.path.join(self.output, "gate2.json"))
        self._log(self._gate_summary(self._gate_label(2, include_mmgbsa=True), gate2))

        # 5. Final refinement on Gate 2 survivors.
        self._final_refine_candidates(gate2)
        self._clear_scores(gate2)
        self._audit_candidates("06_after_final_refine", gate2)

        # 6. Gate 3 -- final pick.
        self._score_qscore(gate2)
        self._score_density_fit(gate2, stage_name="gate3")
        if not bool(self._opt("orch_skip_mmgbsa", False)):
            self._score_mmgbsa(gate2)
        self._assign_rank_scores(gate2, include_mmgbsa=True)
        self._audit_candidates("07_gate3_scored", gate2)
        assignments, rejections = triage.gate3_select(
            gate2,
            **self._rank_kwargs(include_mmgbsa=True),
            min_assignment_score=float(self._opt("orch_min_assignment_score", 3.25)),
            min_density_coverage=float(self._opt("orch_min_density_coverage", 0.30)),
            min_assignment_margin=float(self._opt("orch_min_assignment_margin", 0.15)),
            same_ligand_mmgbsa_window=float(
                self._opt("orch_mmgbsa_pose_window", 0.15)
            ),
            return_rejections=True,
        )
        orch_io.write_gate_json(gate2, os.path.join(self.output, "gate3.json"))
        self._audit_assignments(assignments, selected_stage=True)
        eval_rows, eval_summary = triage.assignment_evaluation_rows(
            assignments,
            rejections,
            self._expected_assignments(),
        )
        self._audit_assignment_eval(eval_rows, eval_summary)
        self._audit_shape_rescore(gate2, selected_stage=True)

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
            rejections=rejections,
            assignment_eval_summary=eval_summary,
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

    def _score_mode(self) -> str:
        mode = str(self._opt("orch_score_mode", "absolute") or "absolute").lower()
        if mode not in {"absolute", "coverage", "qscore"}:
            self._log(
                f"[orchestrator] Unknown score mode {mode!r}; using absolute."
            )
            return "absolute"
        return mode

    def _audit_mode(self) -> str:
        mode = str(self._opt("orch_audit_mode", "full") or "full").lower()
        if mode not in {"full", "scores", "selected", "off"}:
            self._log(
                f"[orchestrator] Unknown audit mode {mode!r}; using full."
            )
            return "full"
        return mode

    def _shape_metrics_mode(self) -> str:
        mode = str(self._opt("orch_shape_metrics", "gate3") or "gate3").lower()
        if mode not in {"off", "gate3", "all"}:
            self._log(
                f"[orchestrator] Unknown shape metrics mode {mode!r}; using gate3."
            )
            return "gate3"
        return mode

    def _density_sci_enabled(self) -> bool:
        if bool(self._opt("orch_compute_density_sci", False)):
            return True
        mode = str(self._opt("orch_density_sci_mode", "auto") or "auto").lower()
        if mode == "on":
            return True
        if mode == "off":
            return False
        if mode != "auto":
            self._log(
                f"[orchestrator] Unknown density SCI mode {mode!r}; using auto."
            )
        return self._score_mode() in {"absolute", "coverage"} and self._audit_mode() != "off"

    def _shape_metrics_enabled(self, stage_name: str | None = None) -> bool:
        if self._audit_mode() == "off":
            return False
        mode = self._shape_metrics_mode()
        if mode == "off":
            return False
        if mode == "all":
            return True
        return str(stage_name or "").lower() == "gate3"

    def _rank_kwargs(self, *, include_mmgbsa: bool) -> dict:
        return {
            "score_mode": self._score_mode(),
            "w_qscore": float(self._opt("orch_w_qscore", 1.0)),
            "w_mmgbsa": (
                float(self._opt("orch_w_mmgbsa", 0.5)) if include_mmgbsa else 0.0
            ),
            "w_qtail": float(self._opt("orch_w_qtail", 0.25)),
            "w_density_coverage": float(self._opt("orch_w_density_coverage", 5.0)),
            "w_density_precision": float(self._opt("orch_w_density_precision", 0.5)),
            "w_density_ccc": float(self._opt("orch_w_density_ccc", 1.0)),
            "w_density_overlap": float(self._opt("orch_w_density_overlap", 1.0)),
        }

    def _gate_label(self, gate_idx: int, *, include_mmgbsa: bool) -> str:
        mode = self._score_mode()
        if mode == "absolute":
            base = "absolute map-fit"
        elif mode == "coverage":
            base = "coverage map-fit"
        else:
            base = "Q-score"
        if include_mmgbsa and not bool(self._opt("orch_skip_mmgbsa", False)):
            base = f"{base} + MMGBSA"
        return f"Gate {gate_idx} ({base})"

    def _assign_rank_scores(
        self,
        candidates_by_site: Dict[str, List[PoseCandidate]],
        *,
        include_mmgbsa: bool,
    ) -> None:
        triage.assign_rank_scores(
            candidates_by_site,
            **self._rank_kwargs(include_mmgbsa=include_mmgbsa),
        )

    def _audit_candidates(
        self,
        stage_name: str,
        candidates_by_site: Dict[str, List[PoseCandidate]],
        *,
        selected_stage: bool = False,
    ) -> None:
        mode = self._audit_mode()
        if mode == "off":
            return
        if mode == "selected" and not selected_stage:
            return
        include_sdfs = mode in {"full", "selected"}
        orch_io.write_audit_candidates(
            candidates_by_site,
            self.system.ligand,
            os.path.join(self.output, "audit"),
            stage_name,
            include_sdfs=include_sdfs,
        )

    def _audit_assignments(
        self,
        assignments,
        *,
        selected_stage: bool = False,
    ) -> None:
        mode = self._audit_mode()
        if mode == "off":
            return
        if mode == "selected" and not selected_stage:
            return
        include_sdfs = mode in {"full", "selected"}
        orch_io.write_audit_assignments(
            assignments,
            self.system.ligand,
            os.path.join(self.output, "audit"),
            include_sdfs=include_sdfs,
        )

    def _audit_assignment_eval(self, rows, summary) -> None:
        mode = self._audit_mode()
        if mode == "off":
            return
        orch_io.write_assignment_eval(
            rows,
            summary,
            os.path.join(self.output, "audit"),
        )

    def _audit_maps(self) -> None:
        if self._audit_mode() == "off":
            return
        orch_io.write_audit_maps(
            getattr(self.system, "binding_site_maps", None) or {},
            os.path.join(self.output, "audit"),
        )

    def _audit_shape_rescore(
        self,
        candidates_by_site: Dict[str, List[PoseCandidate]],
        *,
        selected_stage: bool = False,
    ) -> None:
        if not self._shape_metrics_enabled("gate3"):
            return
        self._audit_candidates(
            "10_shape_rescore",
            candidates_by_site,
            selected_stage=selected_stage,
        )

    def _expected_assignments(self) -> Dict[str, int]:
        spec = str(self._opt("orch_expected_assignments", "") or "").strip()
        if not spec:
            return {}
        expected: Dict[str, int] = {}
        for item in spec.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                self._log(
                    f"[orchestrator] Ignoring malformed expected assignment {item!r}; "
                    "expected site:ligand."
                )
                continue
            site_id, ligand_idx = item.split(":", 1)
            try:
                expected[str(site_id).strip()] = int(str(ligand_idx).strip())
            except Exception:
                self._log(
                    f"[orchestrator] Ignoring malformed expected assignment {item!r}; "
                    "ligand index must be an integer."
                )
        return expected

    @staticmethod
    def _clear_scores(candidates_by_site: Dict[str, List[PoseCandidate]]) -> None:
        for candidates in candidates_by_site.values():
            for c in candidates:
                c.qscore = None
                c.mmgbsa = None
                c.rank_score = None
                c.metrics = {}

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
        n_total = self._count_candidates(candidates_by_site)
        self._log(f"[orchestrator] Q-score scoring {n_total} candidate(s).")
        sigma_ref = float(self._opt("sigma_ref", 0.6))
        for site_id, candidates in candidates_by_site.items():
            for c in candidates:
                ligand = self.system.ligand[c.ligand_idx]
                metrics = scoring.qscore_pose_metrics(
                    c.coords,
                    ligand.mol,
                    self.system.density_map,
                    sigma_ref=sigma_ref,
                )
                if metrics is None:
                    c.qscore = None
                    self._note_once(c, "qscore_failed")
                    continue
                c.qscore = float(metrics["qscore"])
                c.metrics.update(metrics)
        self._log("[orchestrator] Q-score scoring complete.")

    def _site_maps_for(self, site_id):
        maps = getattr(self.system, "binding_site_maps", None) or {}
        if site_id in maps:
            return maps[site_id]
        try:
            int_id = int(site_id)
        except Exception:
            int_id = None
        if int_id is not None and int_id in maps:
            return maps[int_id]
        for key, value in maps.items():
            if str(key) == str(site_id):
                return value
        return None

    def _score_density_fit(
        self,
        candidates_by_site: Dict[str, List[PoseCandidate]],
        *,
        stage_name: str | None = None,
    ) -> None:
        if self._score_mode() not in {"absolute", "coverage"}:
            return

        n_total = self._count_candidates(candidates_by_site)
        compute_sci = self._density_sci_enabled()
        compute_shape = self._shape_metrics_enabled(stage_name)
        extras = []
        if compute_sci:
            extras.append("SCI")
        if compute_shape:
            extras.append("shape")
        suffix = f" with {'/'.join(extras)} diagnostics" if extras else ""
        self._log(
            f"[orchestrator] Density coverage scoring {n_total} candidate(s){suffix}."
        )
        absolute_mode = self._score_mode() == "absolute"
        feature_weights = {
            "density_coverage": float(
                self._opt("orch_w_density_coverage", 5.0 if absolute_mode else 2.0)
            ),
            "density_precision": (
                0.0
                if absolute_mode
                else float(self._opt("orch_w_density_precision", 0.5))
            ),
            "density_ccc": float(
                self._opt("orch_w_density_ccc", 1.0 if absolute_mode else 0.5)
            ),
        }
        for site_id, candidates in candidates_by_site.items():
            site_maps = self._site_maps_for(site_id)
            for c in candidates:
                if site_maps is None:
                    self._note_once(c, "coverage_metrics_failed:no_site_map")
                    continue
                ligand = self.system.ligand[c.ligand_idx]
                try:
                    metrics = scoring.density_fit_metrics(
                        c.coords,
                        ligand.mol,
                        site_maps,
                        threshold_frac=float(
                            self._opt("orch_density_threshold_frac", 0.05)
                        ),
                        feature_weights=feature_weights,
                        compute_sci=compute_sci,
                        compute_shape=compute_shape,
                    )
                except Exception as exc:
                    metrics = None
                    self._note_once(
                        c, f"coverage_metrics_failed:{type(exc).__name__}"
                    )
                if metrics is None:
                    self._note_once(c, "coverage_metrics_failed")
                    continue
                if "shape_metrics_failed" in metrics:
                    self._note_once(
                        c, f"shape_metrics_failed:{metrics['shape_metrics_failed']}"
                    )
                if "skeleton_metrics_failed" in metrics:
                    self._note_once(
                        c,
                        f"skeleton_metrics_failed:{metrics['skeleton_metrics_failed']}",
                    )
                if "density_sci_failed" in metrics:
                    self._note_once(c, f"density_sci_failed:{metrics['density_sci_failed']}")
                if "density_mi_failed" in metrics:
                    self._note_once(
                        c, f"density_mi_failed:{metrics['density_mi_failed']}"
                    )
                c.metrics.update(metrics)
        self._log("[orchestrator] Density coverage scoring complete.")

    def _score_mmgbsa(self, candidates_by_site: Dict[str, List[PoseCandidate]]) -> None:
        n_total = self._count_candidates(candidates_by_site)
        self._log(
            f"[orchestrator] MMGBSA scoring {n_total} candidate(s) "
            "(single-frame OpenMM energy evaluation)."
        )
        done = 0
        for site_id, candidates in candidates_by_site.items():
            for c in candidates:
                done += 1
                label = self._candidate_label(c)
                self._log(f"[orchestrator] MMGBSA {done}/{n_total}: {label}")
                ligand = self.system.ligand[c.ligand_idx]
                ps = scoring.mmgbsa_single_frame(
                    c.coords,
                    ligand,
                    self.system.protein,
                    pose_idx=c.pose_idx,
                    resource_owner=self.system,
                )
                if ps is None:
                    c.mmgbsa = None
                    c.notes.append("mmgbsa_failed")
                    self._log(f"[orchestrator] MMGBSA failed: {label}")
                else:
                    c.mmgbsa = float(ps.deltaG)
                    self._log(
                        f"[orchestrator] MMGBSA complete: {label}, "
                        f"deltaG={c.mmgbsa:.3f}"
                    )
        self._log("[orchestrator] MMGBSA scoring complete.")

    # ---- refine wrappers ----
    def _refine_candidates(self, candidates_by_site: Dict[str, List[PoseCandidate]]) -> None:
        n_total = self._count_candidates(candidates_by_site)
        self._log(
            f"[orchestrator] Refining Gate 1 survivors "
            f"(cheap minimisation): {n_total} candidate(s)."
        )
        done = 0
        for site_id, candidates in candidates_by_site.items():
            for c in candidates:
                done += 1
                label = self._candidate_label(c)
                self._log(f"[orchestrator] Cheap refine {done}/{n_total}: {label}")
                note_start = len(c.notes)
                self._refine_one(c, refiner_cls=Refine)
                new_notes = c.notes[note_start:]
                if new_notes:
                    self._log(
                        f"[orchestrator] Cheap refine note: {label}, "
                        f"notes={';'.join(map(str, new_notes))}"
                    )
                else:
                    self._log(f"[orchestrator] Cheap refine complete: {label}")

    def _final_refiner_name(self) -> str:
        name = getattr(self.system.options, "orch_final_refiner", "smart_refine_2")
        if bool(getattr(self.system.options, "orch_skip_final_refine", False)):
            return "none"
        if bool(getattr(self.system.options, "orch_skip_search_refine", False)):
            return "none"
        return str(name or "smart_refine_2").strip().lower()

    def _final_refiner_class(self, name: str):
        if name == "smart_refine_2":
            return _smart_refine2_class()
        if name == "search_refine":
            if SearchRefine is not None:
                return SearchRefine
            raise ValueError(
                "SearchRefine is not available in this build; use "
                "--orch-final-refiner smart_refine_2 or none"
            )
        if name == "none":
            return None
        raise ValueError(
            f"Unknown orchestrator final refiner {name!r}; "
            "expected smart_refine_2, search_refine, or none"
        )

    def _final_refine_candidates(
        self, candidates_by_site: Dict[str, List[PoseCandidate]]
    ) -> None:
        refiner_name = self._final_refiner_name()
        refiner_cls = self._final_refiner_class(refiner_name)
        if refiner_cls is None:
            self._log("[orchestrator] Final refinement skipped.")
            return

        self._log(
            f"[orchestrator] {refiner_cls.__name__} on Gate 2 survivors: "
            f"{self._count_candidates(candidates_by_site)} candidate(s)."
        )
        done = 0
        n_total = self._count_candidates(candidates_by_site)
        for site_id, candidates in candidates_by_site.items():
            for c in candidates:
                done += 1
                label = self._candidate_label(c)
                self._log(
                    f"[orchestrator] Final refine {done}/{n_total} "
                    f"({refiner_cls.__name__}): {label}"
                )
                note_start = len(c.notes)
                self._refine_one(c, refiner_cls=refiner_cls)
                new_notes = c.notes[note_start:]
                if new_notes:
                    self._log(
                        f"[orchestrator] Final refine note: {label}, "
                        f"notes={';'.join(map(str, new_notes))}"
                    )
                else:
                    self._log(f"[orchestrator] Final refine complete: {label}")

    def _stage_for_refiner(self, refiner_cls) -> str:
        if self._is_smart_refine2_refiner(refiner_cls):
            return "smart_refine_2"
        if SearchRefine is not None and refiner_cls is SearchRefine:
            return "search_refined"
        if refiner_cls is Refine:
            return "refined"
        return getattr(refiner_cls, "__name__", "refined").lower()

    def _is_smart_refine2_refiner(self, refiner_cls) -> bool:
        if SmartRefine2 is not None and refiner_cls is SmartRefine2:
            return True
        return (
            getattr(refiner_cls, "__name__", "") == "SmartRefine2"
            and "smart_refine_2" in getattr(refiner_cls, "__module__", "")
        )

    def _sr2_candidate_output_dir(self, candidate: PoseCandidate) -> str:
        return os.path.join(
            self.output,
            "smart_refine_2",
            f"site_{candidate.site_id}",
            f"ligand_{candidate.ligand_idx}_pose_{candidate.pose_idx}",
        )

    @staticmethod
    def _finite_float(value):
        try:
            out = float(value)
        except Exception:
            return None
        return out if np.isfinite(out) else None

    @staticmethod
    def _safe_int(value):
        try:
            return int(value)
        except Exception:
            return None

    def _sr2_metrics(self, runner, run_result) -> dict:
        result = None
        fit_results = getattr(runner, "fit_results", None)
        if fit_results:
            result = fit_results[0]
        elif isinstance(run_result, (list, tuple)) and run_result:
            result = run_result[0]

        metrics = {}
        if result is not None:
            float_fields = {
                "best_raw_score": "best_raw_score",
                "delta_raw_score": "delta_raw_score",
                "best_objective": "best_objective",
                "delta_objective": "delta_objective",
            }
            for out_key, attr in float_fields.items():
                value = self._finite_float(getattr(result, attr, None))
                if value is not None:
                    metrics[out_key] = value

            int_fields = {
                "best_clash_count": "best_clash_count",
                "steps": "steps",
                "evaluations": "evaluations",
            }
            for out_key, attr in int_fields.items():
                value = self._safe_int(getattr(result, attr, None))
                if value is not None:
                    metrics[out_key] = value

            score_terms = getattr(result, "score_terms", {}) or {}
            final_energy = self._finite_float(
                score_terms.get("final_minimise_energy_kcal")
                if isinstance(score_terms, dict)
                else None
            )
            if final_energy is not None:
                metrics["final_minimise_energy_kcal"] = final_energy

        ligands = getattr(runner, "ligands", None) or []
        refine_ligand = ligands[0] if ligands else None
        if refine_ligand is not None:
            stop_reason = getattr(refine_ligand, "_sr2_stop_reason", None)
            if stop_reason is not None:
                metrics["stop_reason"] = str(stop_reason)
            iterations = self._safe_int(
                getattr(refine_ligand, "_sr2_iterations_completed", None)
            )
            if iterations is not None:
                metrics["iterations_completed"] = iterations
            no_improve = self._safe_int(
                getattr(refine_ligand, "_sr2_no_improve_iters", None)
            )
            if no_improve is not None:
                metrics["no_improve_iters"] = no_improve

        return metrics

    def _refine_one(self, candidate: PoseCandidate, refiner_cls) -> None:
        """Run one refiner on a single candidate.

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

            runner = refiner_cls(self.system)
            is_smart_refine2 = self._is_smart_refine2_refiner(refiner_cls)
            if is_smart_refine2:
                runner.output = self._sr2_candidate_output_dir(candidate)
                os.makedirs(runner.output, exist_ok=True)
            run_result = runner.run()
            if is_smart_refine2:
                candidate.refine_metrics = self._sr2_metrics(runner, run_result)

            new_coords = ligand.mol.GetConformer(0).GetPositions()
            arr = np.asarray(new_coords, dtype=float)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                candidate.notes.append(f"{refiner_cls.__name__}_diverged")
            else:
                candidate.coords = arr
                candidate.stage = self._stage_for_refiner(refiner_cls)
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
    @staticmethod
    def _count_candidates(by_site: Dict[str, List[PoseCandidate]]) -> int:
        return sum(len(candidates) for candidates in by_site.values())

    @staticmethod
    def _candidate_label(candidate: PoseCandidate) -> str:
        return (
            f"site {candidate.site_id} / ligand {candidate.ligand_idx} / "
            f"pose {candidate.pose_idx}"
        )

    @staticmethod
    def _note_once(candidate: PoseCandidate, note: str) -> None:
        if note not in candidate.notes:
            candidate.notes.append(note)

    def _gate_summary(self, name: str, by_site: Dict[str, List[PoseCandidate]]) -> str:
        lines = [f"[orchestrator] {name} survivors per site:"]
        for site_id in sorted(by_site.keys(), key=str):
            cs = by_site[site_id]
            best_q = max((c.qscore for c in cs if c.qscore is not None), default=None)
            best_m = min((c.mmgbsa for c in cs if c.mmgbsa is not None), default=None)
            best_rank = max(
                (c.rank_score for c in cs if c.rank_score is not None),
                default=None,
            )
            best_cov = max(
                (
                    c.metrics.get("density_coverage")
                    for c in cs
                    if c.metrics.get("density_coverage") is not None
                ),
                default=None,
            )
            q_str = "n/a" if best_q is None else f"{best_q:.3f}"
            m_str = "n/a" if best_m is None else f"{best_m:.2f}"
            r_str = "n/a" if best_rank is None else f"{best_rank:.3f}"
            cov_str = "n/a" if best_cov is None else f"{best_cov:.3f}"
            lines.append(
                f"  - {site_id}: n={len(cs)}, best_rank={r_str}, "
                f"best_qscore={q_str}, best_coverage={cov_str}, best_mmgbsa={m_str}"
            )
        return "\n".join(lines)
