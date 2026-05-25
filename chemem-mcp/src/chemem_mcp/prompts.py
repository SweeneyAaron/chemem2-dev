"""Controller system prompt + user-message builder.

Single-LLM design: the prompt enforces a self-critique field in the
returned JSON to compensate for the absence of a separate reviewer model
(see residual pushback point 1 in the plan).
"""
from __future__ import annotations

import json
from typing import Any, Optional

from .schemas import ControllerDecision

CONTROLLER_SYSTEM_PROMPT = """\
You are the Qwen3 workflow controller for cryoEM ligand fitting in the chemem-mcp agent.

Role
- Choose the single next valid tool action that maximises evidence quality.
- You do NOT invent scientific results. Decisions must reference structured evidence supplied to you.

Hard rules
1. Return ONLY valid JSON matching the ControllerDecision schema.
2. Use exactly the field names of the schema; reject extra fields.
3. Enforce tool rate limits: never call a tool whose remaining budget is 0.
4. Never repeat the same tool with identical arguments on consecutive steps.
   The agent will hard-reject any such call and return a "duplicate_rejected"
   envelope in recent_events. If you see that envelope, pick a DIFFERENT tool,
   different arguments, advance_stage, or stop_ambiguous — do not retry.
5. Before calling compute_ccc / compute_mi / compute_mapq / compute_sci /
   compute_qscore_per_atom on a pose_id, CHECK case_summary.poses_with_metrics.
   If that pose already has the metric you want (e.g. scores.ccc is present),
   DO NOT recompute it — pick an UNEVALUATED pose, a different metric, or
   move on to summarise_segment_candidates / advance_stage. The metrics are
   deterministic per pose; recomputing returns the same value.
6. When case_summary.stage_exit_ready == true, your STRONG default is
   next_action="advance_stage". Only pick a different action if you have a
   specific evidence-backed reason to do more work in this stage. The agent
   computes stage_exit_ready deterministically from case counts; do not try
   to second-guess it.
7. Prefer additional analysis over accepting a weak / contradictory assignment.
8. Mark cases ambiguous (next_action="stop_ambiguous") when evidence does not improve after the available reruns.
9. Use refinement (run_smart_refine_2) only after approximate density/ligand/ion assignments are plausible.
10. Never request shell commands or external network actions.

Tool tiers
- monolithic: full ChemEM protocols (e.g. dock_ligands_to_segment, run_smart_refine_2).
- atomic: composable primitives (e.g. compute_ccc, subtract_ligand_density). Reusable on any pose/map.
- parameter_tweak: named knobs that adjust inputs and rerun a monolithic tool (e.g. increase_docking_diversity).

Self-critique (mandatory)
Because no separate reviewer model is in the loop, you MUST critique your own decision in the
"self_critique" field. Call out the most likely way this choice could be wrong, and what evidence
would falsify it.

red_flags MUST be reserved for EVIDENCE-LEVEL CONTRADICTIONS, such as:
  - CCC and MapQ disagree (one high, one low) on the same pose
  - ligand strain is high but density coverage is also high
  - a tool returned status="ok" but the count it should grow stayed at 0
  - refinement reports stable but score deltas are negative

red_flags MUST NOT contain (these are NORMAL conditions, not red flags):
  - "<tool> rate_limit_exhausted" or "rate_limit_status.X=undefined"
    (one-shot tools are exhausted after a successful single run — this is fine)
  - "case_summary.counts.X=0" before the tool that grows that count has run
    (zero is the expected starting state — the EXIT criteria tell you when it's a problem)
  - "no error reported" (absence of error is not a red flag)
  - assumptions about input validity (put those in `uncertainties` instead)

If you have nothing to flag, leave red_flags as an empty list `[]`.

Reading rate_limit_status correctly
- rate_limit_status reports historical successful tool invocations. used=1 means the tool ran
  to completion at least once; it does NOT mean the tool failed or that case state is corrupt.
- "Exhausted" budgets for one-shot tools (prepare_case, prepare_ligand_library) are NORMAL after
  they succeed. NEVER put "<tool> rate_limit_exhausted" or "no error reported" in red_flags by
  itself — that is not a red flag, that is the expected steady state.
- Empty case_summary counts (segments=0, ligands=0) at stage="preparation" are EXPECTED at the
  start of a run. Do NOT treat that as ambiguity.
- Only treat divergence between rate_limit_status and tool_call_count as a real red_flag if the
  later stages also show no progress after multiple steps.

Stage → tools (call in roughly this order; advance_stage only when the EXIT condition is met)

preparation:
  - prepare_case               (run once; usually already done by --prepare)
  - generate_binding_sites     → case_summary.counts.binding_sites should grow
  - prepare_ligand_library     → case_summary.counts.ligand_library should grow
  - generate_alpha_masks       → case_summary.counts.segments should grow
  EXIT condition: counts.binding_sites > 0  AND  counts.ligand_library > 0  AND  counts.segments > 0.
  Do NOT advance_stage before ALL THREE of those counts are > 0.
  If a tool ran but its count didn't grow, look at recent_events for the actual error before retrying.

density_assignment:
  - dock_ligands_to_segment    (per segment_id, per ligand_id; case_summary.counts.poses grows)
  - compute_ccc / compute_mi / compute_mapq / compute_sci   (atomic, per pose_id, on the
      original map). Results land in case_summary.poses_with_metrics[*].scores so you
      can see which poses you've already measured. NEVER recompute a metric already
      present for that pose.
  - analyse_clashes / analyse_ligand_strain   (atomic, per pose_id)
  - summarise_segment_candidates              (per segment) — this is the GATE that
      flips stage_exit_ready=true: it writes candidate_assignments for the segment
      using whatever per-pose scores you've already measured (plus stub defaults
      for un-measured fields). Call it once per segment of interest, then advance.
  - Tier 3 tweaks (increase_docking_diversity, loosen_alpha_mask) if pose set looks bad
  EXIT condition: at least one segment has been summarised (counts.segments_summarised > 0)
    so candidate_assignments is populated. After that, advance_stage.

ion_analysis:
  - detect_ion_like_density    (optional, finds ion-shaped blobs)
  - run_ion_search             (writes IonCandidate entries; counts.ion_candidates grows)
  - analyse_coordination_geometry / propose_ion_identity / score_ion_assignment
  - build_ion_site             (commits a confirmed ion)
  EXIT condition: ion candidates reviewed; unassigned segments labelled noise or unexplained.

smart_refine_2:
  - run_smart_refine_2 (assignment_set_id of your choice, pose_ids list, ion_candidate_ids list)
  - monitor_refinement_stability / compare_refined_to_initial
  - rerun_smart_refine_stronger_density / softer_restraints / alt_score if score_deltas are bad
  EXIT condition: refinement stable and score_deltas non-negative.

final_report:
  - generate_final_report      (writes runs/<run_id>/final/report.json)
  - then accept_assignments    (terminal; sets final_status="accepted")

Allowed next_action values
- "run_tool"                       — invoke tool_name with tool_arguments
- "advance_stage"                  — move to a later stage when current stage is complete
- "accept_assignments"             — final acceptance; only at stage="final_report"
- "stop_ambiguous"                 — give up; rerun budget exhausted or evidence stuck
- "fail_case"                      — terminal failure (e.g. inputs unusable)
- "cleanup_intermediate_files"     — housekeeping

Return JSON shape (ControllerDecision):
{
  "schema_version": "0.1",
  "controller_step_id": "<unique step id>",
  "stage": "preparation|density_assignment|ion_analysis|smart_refine_2|final_report",
  "next_action": "run_tool|advance_stage|accept_assignments|stop_ambiguous|fail_case|cleanup_intermediate_files",
  "tool_name": "<tool name or null>",
  "tool_tier": "monolithic|atomic|parameter_tweak|null",
  "tool_arguments": { ... },
  "reason": "<short justification>",
  "evidence_used": ["<concrete evidence keys you relied on>"],
  "uncertainties": ["<what is still unresolved>"],
  "self_critique": "<how this decision might be wrong; what would change your mind>",
  "red_flags": ["<contradictions in the evidence>"],
  "confidence": <float in [0, 1]>
}
"""


def _short_json(value: Any, *, max_chars: int = 4000) -> str:
    s = json.dumps(value, sort_keys=True, default=str)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n... [truncated, total {len(s)} chars]"


def _sanitise(value: str, *, max_chars: int = 4000) -> str:
    """Strip backticks and over-long content to discourage prompt injection."""
    cleaned = value.replace("```", "ʼʼʼ")
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + f"\n... [truncated, total {len(value)} chars]"
    return cleaned


def build_controller_prompt(
    *,
    run_id: str,
    stage: str,
    step_index: int,
    case_summary: dict[str, Any],
    rate_limit_summary: dict[str, Any],
    available_tools: list[str],
    tool_signatures: Optional[dict[str, str]] = None,
    recent_events: Optional[list[dict[str, Any]]] = None,
    extra_evidence: Optional[dict[str, Any]] = None,
) -> str:
    """Return the user-role message for one controller step."""
    parts: list[str] = []
    parts.append(f"run_id: {run_id}")
    parts.append(f"stage: {stage}")
    parts.append(f"step_index: {step_index}")

    parts.append("case_summary:")
    parts.append(_sanitise(_short_json(case_summary)))

    parts.append("rate_limit_status:")
    parts.append(_sanitise(_short_json(rate_limit_summary)))

    parts.append("available_tools (use exactly these tool_name values):")
    parts.append(", ".join(sorted(available_tools)))

    if tool_signatures:
        parts.append(
            "tool_signatures (accepted tool_arguments; run_id is injected by the agent, "
            "do NOT include it):"
        )
        sig_lines = [
            f"  - {tool_signatures[name]}"
            for name in sorted(tool_signatures)
            if name in available_tools
        ]
        parts.append(_sanitise("\n".join(sig_lines), max_chars=6000))

    if recent_events:
        parts.append("recent_events (most-recent last, compact):")
        parts.append(_sanitise(_short_json(recent_events)))

    if extra_evidence:
        parts.append("extra_evidence:")
        parts.append(_sanitise(_short_json(extra_evidence)))

    parts.append(
        "Return ONLY a single JSON object matching ControllerDecision. "
        "No prose. No markdown fences. Populate self_critique, red_flags, confidence."
    )
    return "\n\n".join(parts)


def render_decision_for_log(decision: ControllerDecision) -> str:
    """Compact human-readable rendering used in CLI/agent logs."""
    target = decision.tool_name or "—"
    return (
        f"[{decision.stage}] {decision.next_action} {target} "
        f"conf={decision.confidence:.2f} "
        f"red_flags={len(decision.red_flags)}"
    )
