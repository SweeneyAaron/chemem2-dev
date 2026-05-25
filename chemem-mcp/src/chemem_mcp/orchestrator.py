"""Single-controller agent loop.

The Agent drives a case through Stage 1 → 5 by repeatedly asking an
:class:`LLMClient` for the next :class:`ControllerDecision` and dispatching
the chosen tool. Pure dependency-injection design: tests pass a
:class:`ScriptedLLMClient`; production code uses :class:`OllamaClient`.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .cleanup import cleanup_run_dir
from .llm import InvalidLLMResponse, LLMClient
from .paths import safe_run_dir
from .prompts import (
    CONTROLLER_SYSTEM_PROMPT,
    build_controller_prompt,
    render_decision_for_log,
)
from .rate_limiter import RateLimiter
from .runtime import get_rate_limiter, get_run_root, runtime_override
from .schemas import ControllerDecision
from .state import CaseState, DECISION_LOG_FILENAME
from .tools import monolithic, primitives, rendering, reports, tweaks

STAGE_ORDER: tuple[str, ...] = (
    "preparation",
    "density_assignment",
    "ion_analysis",
    "smart_refine_2",
    "final_report",
)

# Truncate tool-result payloads at this byte size when logging them so the
# LLM sees meaningful content without exploding the prompt.
TOOL_RESULT_SUMMARY_MAX_CHARS = 3000

# How many decision_log.jsonl entries to feed back to the model as
# `recent_events`. Wide enough that a CCC value computed several turns ago
# is still in view, narrow enough not to dominate the prompt.
RECENT_EVENTS_WINDOW = 16


def _tool_signature(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}::{json.dumps(arguments or {}, sort_keys=True, default=str)}"


def _summarise_tool_result(
    result: Optional[dict[str, Any]],
    *,
    max_chars: int = TOOL_RESULT_SUMMARY_MAX_CHARS,
) -> dict[str, Any]:
    """Compact, agent-visible view of a tool's output.

    Preserves status / tool_tier and a truncated JSON preview of everything
    else, so the controller can see what a call actually produced (binding
    sites, pose ids, scores) on the next step.
    """
    if not result:
        return {"status": None}
    s = json.dumps(result, sort_keys=True, default=str)
    if len(s) <= max_chars:
        return result
    return {
        "status": result.get("status"),
        "tool_tier": result.get("tool_tier"),
        "_truncated": True,
        "_size_chars": len(s),
        "_preview": s[:max_chars],
    }


def _next_stage(current: str) -> Optional[str]:
    if current not in STAGE_ORDER:
        return None
    idx = STAGE_ORDER.index(current)
    if idx + 1 >= len(STAGE_ORDER):
        return None
    return STAGE_ORDER[idx + 1]


def _tool_signature_spec(name: str, fn: Callable[..., Any]) -> str:
    """Return a one-line ``name(arg1, arg2=default)`` spec for a tool.

    ``run_id`` is omitted because the agent injects it automatically — telling
    the model to set it would just invite collisions.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return f"{name}(...)"
    pieces: list[str] = []
    for p in sig.parameters.values():
        if p.name == "run_id":
            continue
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if p.default is inspect.Parameter.empty:
            pieces.append(p.name)
        else:
            pieces.append(f"{p.name}={p.default!r}")
    return f"{name}({', '.join(pieces)})"


def _stage_exit_ready(state: "CaseState") -> bool:
    """Deterministic per-stage EXIT predicate.

    Surfaced to the controller as a boolean so a small model doesn't have to
    re-derive it from counts under pressure.
    """
    stage = state.stage
    if stage == "preparation":
        return (
            len(state.binding_sites) > 0
            and len(state.ligand_library) > 0
            and len(state.segments) > 0
        )
    if stage == "density_assignment":
        # At least one pose docked AND at least one segment has been summarised
        # (segment.candidate_assignments populated by summarise_segment_candidates).
        if len(state.poses) == 0:
            return False
        return any(len(s.candidate_assignments) > 0 for s in state.segments.values())
    if stage == "ion_analysis":
        # Ion search has run (candidates may be empty if nothing ion-like).
        return state.tool_call_count >= 1  # any analysis at this stage counts
    if stage == "smart_refine_2":
        return len(state.refinements) > 0
    if stage == "final_report":
        # Ready to accept once final report has been written (presence on disk).
        return (Path(state.run_dir) / "final" / "report.json").exists()
    return False


# ---------------------------------------------------------------------------
# Tool registry — name → callable. All tools take run_id as first kw arg.
# ---------------------------------------------------------------------------


def _default_registry() -> dict[str, Callable[..., dict[str, Any]]]:
    return {
        # Tier 1 — monolithic
        "prepare_case": monolithic.prepare_case,
        "generate_binding_sites": monolithic.generate_binding_sites,
        "generate_alpha_masks": monolithic.generate_alpha_masks,
        "segment_density_regions": monolithic.segment_density_regions,
        "prepare_ligand_library": monolithic.prepare_ligand_library,
        "prepare_ligand_segments": monolithic.prepare_ligand_segments,
        "dock_ligands_to_segment": monolithic.dock_ligands_to_segment,
        "dock_ligand_segments_to_segment": monolithic.dock_ligand_segments_to_segment,
        "minimise_candidate_pose": monolithic.minimise_candidate_pose,
        "run_ion_search": monolithic.run_ion_search,
        "build_ion_site": monolithic.build_ion_site,
        "run_smart_refine_2": monolithic.run_smart_refine_2,
        "run_search_refine": monolithic.run_search_refine,
        # Tier 2 — atomic primitives
        "compute_ccc": primitives.compute_ccc,
        "compute_mi": primitives.compute_mi,
        "compute_mapq": primitives.compute_mapq,
        "compute_sci": primitives.compute_sci,
        "compute_qscore_per_atom": primitives.compute_qscore_per_atom,
        "crop_map_around_segment": primitives.crop_map_around_segment,
        "mask_map_by_segment": primitives.mask_map_by_segment,
        "subtract_ligand_density": primitives.subtract_ligand_density,
        "compute_ligand_volume": primitives.compute_ligand_volume,
        "compute_segment_volume": primitives.compute_segment_volume,
        "analyse_clashes": primitives.analyse_clashes,
        "analyse_ligand_strain": primitives.analyse_ligand_strain,
        "analyse_interactions": primitives.analyse_interactions,
        "detect_ion_like_density": primitives.detect_ion_like_density,
        "analyse_coordination_geometry": primitives.analyse_coordination_geometry,
        "propose_ion_identity": primitives.propose_ion_identity,
        "score_ion_assignment": primitives.score_ion_assignment,
        "monitor_refinement_stability": primitives.monitor_refinement_stability,
        "score_refined_model": primitives.score_refined_model,
        "analyse_refined_ligands": primitives.analyse_refined_ligands,
        "analyse_refined_ions": primitives.analyse_refined_ions,
        "analyse_residual_density": primitives.analyse_residual_density,
        "compare_refined_to_initial": primitives.compare_refined_to_initial,
        "summarise_refinement_outcome": primitives.summarise_refinement_outcome,
        "summarise_segment_candidates": primitives.summarise_segment_candidates,
        # Tier 3 — parameter-tweak / rerun
        "increase_docking_diversity": tweaks.increase_docking_diversity,
        "narrow_docking_search": tweaks.narrow_docking_search,
        "enable_flexible_rings": tweaks.enable_flexible_rings,
        "loosen_alpha_mask": tweaks.loosen_alpha_mask,
        "tighten_alpha_mask": tweaks.tighten_alpha_mask,
        "rerun_smart_refine_stronger_density": tweaks.rerun_smart_refine_stronger_density,
        "rerun_smart_refine_softer_restraints": tweaks.rerun_smart_refine_softer_restraints,
        "rerun_smart_refine_alt_score": tweaks.rerun_smart_refine_alt_score,
        "rerun_ion_search_lower_threshold": tweaks.rerun_ion_search_lower_threshold,
        # Reports + rendering
        "generate_final_report": reports.generate_final_report,
        "summarise_case_state": reports.summarise_case_state,
        "get_case_disk_usage": reports.get_case_disk_usage,
        "render_snapshot": rendering.render_snapshot,
        "render_ion_site_snapshot": rendering.render_ion_site_snapshot,
        "render_refined_snapshots": rendering.render_refined_snapshots,
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


TERMINAL_ACTIONS: frozenset[str] = frozenset(
    {"stop_ambiguous", "fail_case", "accept_assignments"}
)


@dataclass
class StepRecord:
    decision: ControllerDecision
    tool_result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class Agent:
    llm: LLMClient
    model: str
    tool_registry: dict[str, Callable[..., dict[str, Any]]] = field(
        default_factory=_default_registry
    )
    max_steps: int = 100
    system_prompt: str = CONTROLLER_SYSTEM_PROMPT
    log_fn: Callable[[str], None] = field(default=lambda msg: None)

    # ------------------------------------------------------------------

    def _evidence(self, state: CaseState) -> dict[str, Any]:
        poses_with_metrics: list[dict[str, Any]] = []
        segments_summarised: list[str] = []
        for p in list(state.poses.values())[:40]:
            measured = {
                k: v for k, v in p.scores.model_dump().items() if v is not None
            }
            poses_with_metrics.append(
                {
                    "pose_id": p.pose_id,
                    "segment_id": p.segment_id,
                    "ligand_id": p.ligand_id,
                    "scores": measured,
                }
            )
        for sid, seg in state.segments.items():
            if seg.candidate_assignments:
                segments_summarised.append(sid)
        return {
            "stage": state.stage,
            "stage_exit_ready": _stage_exit_ready(state),
            "controller_step_count": state.controller_step_count,
            "tool_call_count": state.tool_call_count,
            "counts": {
                "binding_sites": len(state.binding_sites),
                "ligand_library": len(state.ligand_library),
                "segments": len(state.segments),
                "poses": len(state.poses),
                "poses_with_any_metric": sum(
                    1 for row in poses_with_metrics if row["scores"]
                ),
                "segments_summarised": len(segments_summarised),
                "ion_candidates": len(state.ion_candidates),
                "refinements": len(state.refinements),
                "accepted_ligands": len(state.accepted_ligands),
            },
            "binding_site_ids": list(state.binding_sites.keys())[:20],
            "ligand_ids": list(state.ligand_library.keys())[:20],
            "segment_ids": list(state.segments.keys())[:20],
            "segments_summarised_ids": segments_summarised[:20],
            "pose_ids": list(state.poses.keys())[:20],
            "poses_with_metrics": poses_with_metrics,
            "ion_candidate_ids": list(state.ion_candidates.keys())[:20],
            "refinement_ids": list(state.refinements.keys())[:20],
            "final_status": state.final_status,
        }

    def _recent_events(
        self, state: CaseState, *, n: int = RECENT_EVENTS_WINDOW
    ) -> list[dict[str, Any]]:
        path = Path(state.run_dir) / DECISION_LOG_FILENAME
        if not path.exists():
            return []
        lines = path.read_text().splitlines()[-n:]
        out: list[dict[str, Any]] = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _rate_limit_summary(self, state: CaseState) -> dict[str, Any]:
        return get_rate_limiter().summary(run_id=state.run_id)

    # ------------------------------------------------------------------

    def _fail_soft(
        self, state: CaseState, reason: str, *, stage: Optional[str] = None
    ) -> ControllerDecision:
        return ControllerDecision(
            controller_step_id=f"failsoft_{uuid.uuid4().hex[:8]}",
            stage=stage or state.stage,
            next_action="stop_ambiguous",
            tool_name=None,
            tool_tier=None,
            tool_arguments={},
            reason=reason,
            evidence_used=[],
            uncertainties=[reason],
            self_critique="Synthetic fail-soft decision; LLM did not return a valid ControllerDecision.",
            red_flags=["llm_response_invalid"],
            confidence=0.0,
        )

    # ------------------------------------------------------------------

    def _tool_signatures(self) -> dict[str, str]:
        return {
            name: _tool_signature_spec(name, fn)
            for name, fn in self.tool_registry.items()
        }

    def step(self, state: CaseState, *, extra_evidence: Optional[dict] = None) -> StepRecord:
        case_summary = self._evidence(state)
        rate_summary = self._rate_limit_summary(state)
        user_msg = build_controller_prompt(
            run_id=state.run_id,
            stage=state.stage,
            step_index=state.controller_step_count,
            case_summary=case_summary,
            rate_limit_summary=rate_summary,
            available_tools=sorted(self.tool_registry.keys()),
            tool_signatures=self._tool_signatures(),
            recent_events=self._recent_events(state),
            extra_evidence=extra_evidence,
        )
        try:
            decision, _meta = self.llm.chat_json(
                model=self.model,
                system=self.system_prompt,
                user=user_msg,
                response_model=ControllerDecision,
            )
        except InvalidLLMResponse as exc:
            decision = self._fail_soft(state, f"LLM error: {exc}")

        state.log_decision(decision)
        self.log_fn(render_decision_for_log(decision))

        record = StepRecord(decision=decision)
        record.tool_result, record.error = self._dispatch(state, decision)
        return record

    # ------------------------------------------------------------------

    def _dispatch(
        self, state: CaseState, decision: ControllerDecision
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        action = decision.next_action
        if action == "run_tool":
            return self._run_tool(state, decision)
        if action == "cleanup_intermediate_files":
            return self._cleanup(state)
        if action == "advance_stage":
            return self._advance_stage(state), None
        if action == "accept_assignments":
            state.final_status = "accepted"
            state.save()
            return None, None
        if action == "stop_ambiguous":
            state.final_status = "ambiguous"
            state.save()
            return None, None
        if action == "fail_case":
            state.final_status = "failed"
            state.save()
            return None, None
        return None, f"unknown next_action: {action}"

    def _run_tool(
        self, state: CaseState, decision: ControllerDecision
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        name = decision.tool_name
        if not name:
            return None, "run_tool with no tool_name"
        tool = self.tool_registry.get(name)
        if tool is None:
            return None, f"unknown tool: {name}"

        public_args = {k: v for k, v in (decision.tool_arguments or {}).items() if k != "run_id"}
        signature = _tool_signature(name, public_args)

        # Hard rule #4 — refuse identical consecutive tool calls. Without this
        # a small model can loop on the same call when it doesn't realise the
        # previous one already succeeded.
        if signature == state.last_tool_signature:
            duplicate = {
                "status": "duplicate_rejected",
                "tool_name": name,
                "tool_arguments": public_args,
                "previous_result_summary": state.last_tool_result_summary,
                "warning": (
                    "you already called this tool with identical arguments and got "
                    "previous_result_summary above; DO NOT repeat. Pick a different "
                    "tool, different arguments, advance_stage, or stop_ambiguous."
                ),
            }
            state.log_tool_call(name, public_args, duplicate)
            return duplicate, None

        kwargs = dict(public_args)
        kwargs["run_id"] = state.run_id  # we own this
        try:
            result = tool(**kwargs)
        except TypeError as exc:
            # Surface bad-arguments failures to the decision log so the model
            # can see what went wrong on its next step. Previously these were
            # silently swallowed and small models would keep guessing.
            expected_sig = _tool_signature_spec(name, tool)
            bad_args_envelope = {
                "status": "bad_arguments",
                "tool_name": name,
                "error": str(exc),
                "expected_signature": expected_sig,
                "rejected_arguments": sorted(public_args.keys()),
                "warning": (
                    f"call rejected: {exc}. Retry with arguments matching the "
                    f"expected signature: {expected_sig}"
                ),
            }
            fresh_state = CaseState.load_or_create(state.run_id, Path(state.run_dir))
            fresh_state.log_tool_call(name, public_args, bad_args_envelope)
            return bad_args_envelope, f"bad_arguments: {exc}"

        # The tool may have mutated state on disk (add_segment, add_ion_candidate,
        # add_refinement). Reload before writing agent bookkeeping so the agent's
        # save does not clobber the tool's mutations.
        fresh_state = CaseState.load_or_create(state.run_id, Path(state.run_dir))
        fresh_state.last_tool_signature = signature
        result_summary = _summarise_tool_result(result)
        fresh_state.last_tool_result_summary = result_summary
        fresh_state.log_tool_call(name, public_args, result_summary)
        return result, None

    def _cleanup(
        self, state: CaseState
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        rl = get_rate_limiter().check_and_increment(
            "cleanup_intermediate_files", state.run_id
        )
        if not rl.allowed:
            return ({"status": "rate_limited", "rate_limit": rl.to_dict()}, None)
        report = cleanup_run_dir(Path(state.run_dir), dry_run=True)
        return report.to_dict(), None

    def _advance_stage(self, state: CaseState) -> dict[str, Any]:
        target = _next_stage(state.stage)
        if target is None:
            return {"status": "noop", "reason": "no later stage"}
        state.set_stage(target)
        return {"status": "ok", "stage": target}

    # ------------------------------------------------------------------

    def run(
        self, run_id: str, *, run_root: Optional[Path] = None
    ) -> CaseState:
        run_dir = safe_run_dir(run_id, run_root or get_run_root())
        state = CaseState.load_or_create(run_id, run_dir)
        for _ in range(self.max_steps):
            record = self.step(state)
            state = CaseState.load_or_create(run_id, run_dir)  # refresh after tool persistence
            if record.decision.next_action in TERMINAL_ACTIONS:
                break
        return state


# ---------------------------------------------------------------------------
# CLI entry point — for the Linux workstation
# ---------------------------------------------------------------------------


def _make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chemem-mcp-orchestrator",
        description="Run the chemem-mcp agent loop against Ollama.",
    )
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-root", default=None, help="Override CHEMEM_MCP_RUN_ROOT.")
    p.add_argument("--controller-model", default="qwen3:14b")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--prepare", action="store_true", help="Stage inputs before running.")
    p.add_argument("--protein-pdb", default=None)
    p.add_argument("--map-mrc", default=None)
    p.add_argument("--ligand-library", default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _make_arg_parser().parse_args(argv)
    from .ollama_client import OllamaClient

    rl = RateLimiter.with_defaults()
    run_root = Path(args.run_root).resolve() if args.run_root else get_run_root()
    with runtime_override(run_root=run_root, rate_limiter=rl):
        if args.prepare:
            if not all([args.protein_pdb, args.map_mrc, args.ligand_library]):
                print("--prepare requires --protein-pdb, --map-mrc, --ligand-library", file=sys.stderr)
                return 2
            prep_kwargs = {
                "protein_pdb": args.protein_pdb,
                "map_mrc": args.map_mrc,
                "ligand_library": args.ligand_library,
            }
            prep_result = monolithic.prepare_case(run_id=args.run_id, **prep_kwargs)
            # Keep state consistent: log the tool call the same way Agent._run_tool
            # would. Otherwise rate_limit_status (which prepare_case bumps) and
            # case_summary.tool_call_count diverge, and a careful controller will
            # correctly flag it as contradictory evidence and stop.
            prep_state = CaseState.load_or_create(
                args.run_id, safe_run_dir(args.run_id, run_root)
            )
            prep_state.log_tool_call(
                "prepare_case",
                prep_kwargs,
                {"status": (prep_result or {}).get("status"), "tool_tier": "monolithic"},
            )

        llm = OllamaClient(base_url=args.ollama_url)
        agent = Agent(
            llm=llm,
            model=args.controller_model,
            max_steps=args.max_steps,
            log_fn=lambda msg: print(msg, file=sys.stderr),
        )
        state = agent.run(args.run_id)
        print(
            json.dumps(
                {
                    "run_id": state.run_id,
                    "stage": state.stage,
                    "final_status": state.final_status,
                    "controller_steps": state.controller_step_count,
                    "tool_calls": state.tool_call_count,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
