from __future__ import annotations

from pathlib import Path

from chemem_mcp.llm import InvalidLLMResponse, ScriptedLLMClient
from chemem_mcp.orchestrator import (
    Agent,
    _next_stage,
    _stage_exit_ready,
    _summarise_tool_result,
    _tool_signature,
    _tool_signature_spec,
)
from chemem_mcp.schemas import ControllerDecision
from chemem_mcp.state import CaseState
from chemem_mcp.paths import safe_run_dir


def _decision(
    *,
    stage: str = "preparation",
    next_action: str = "advance_stage",
    tool_name: str | None = None,
    tool_arguments: dict | None = None,
    step_id: str = "x",
    confidence: float = 0.5,
) -> ControllerDecision:
    return ControllerDecision(
        controller_step_id=step_id,
        stage=stage,
        next_action=next_action,
        tool_name=tool_name,
        tool_arguments=tool_arguments or {},
        reason="test",
        self_critique="synthetic",
        red_flags=[],
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Stage progression
# ---------------------------------------------------------------------------


def test_next_stage_order() -> None:
    assert _next_stage("preparation") == "density_assignment"
    assert _next_stage("density_assignment") == "ion_analysis"
    assert _next_stage("ion_analysis") == "smart_refine_2"
    assert _next_stage("smart_refine_2") == "final_report"
    assert _next_stage("final_report") is None
    assert _next_stage("nonsense") is None


# ---------------------------------------------------------------------------
# Single-step dispatch
# ---------------------------------------------------------------------------


def test_advance_stage_step(runtime) -> None:
    run_root, _ = runtime
    state = CaseState.load_or_create("case001", safe_run_dir("case001", run_root))
    llm = ScriptedLLMClient(script=[_decision(stage="preparation", next_action="advance_stage")])
    agent = Agent(llm=llm, model="test")
    record = agent.step(state)
    assert record.decision.next_action == "advance_stage"
    reloaded = CaseState.load_or_create("case001", Path(state.run_dir))
    assert reloaded.stage == "density_assignment"
    assert reloaded.controller_step_count == 1


def test_run_tool_step_dispatches_to_registry(runtime) -> None:
    run_root, _ = runtime
    state = CaseState.load_or_create("case002", safe_run_dir("case002", run_root))
    llm = ScriptedLLMClient(
        script=[
            _decision(
                stage="preparation",
                next_action="run_tool",
                tool_name="prepare_case",
                tool_arguments={
                    "protein_pdb": "p.pdb",
                    "map_mrc": "m.mrc",
                    "ligand_library": "l.sdf",
                },
            )
        ]
    )
    agent = Agent(llm=llm, model="test")
    record = agent.step(state)
    assert record.tool_result and record.tool_result["status"] == "ok"
    reloaded = CaseState.load_or_create("case002", Path(state.run_dir))
    assert reloaded.tool_call_count == 1


def test_unknown_tool_is_recorded_as_error(runtime) -> None:
    run_root, _ = runtime
    state = CaseState.load_or_create("case003", safe_run_dir("case003", run_root))
    llm = ScriptedLLMClient(
        script=[
            _decision(
                stage="preparation",
                next_action="run_tool",
                tool_name="not_a_real_tool",
            )
        ]
    )
    agent = Agent(llm=llm, model="test")
    record = agent.step(state)
    assert record.error and "unknown tool" in record.error


def test_invalid_llm_response_fails_soft_to_stop_ambiguous(runtime) -> None:
    run_root, _ = runtime
    state = CaseState.load_or_create("case004", safe_run_dir("case004", run_root))
    llm = ScriptedLLMClient(script=[InvalidLLMResponse("schema bust")])
    agent = Agent(llm=llm, model="test")
    record = agent.step(state)
    assert record.decision.next_action == "stop_ambiguous"
    assert "llm_response_invalid" in record.decision.red_flags
    reloaded = CaseState.load_or_create("case004", Path(state.run_dir))
    assert reloaded.final_status == "ambiguous"


def test_accept_assignments_sets_status_accepted(runtime) -> None:
    run_root, _ = runtime
    state = CaseState.load_or_create("case005", safe_run_dir("case005", run_root))
    llm = ScriptedLLMClient(
        script=[_decision(stage="final_report", next_action="accept_assignments")]
    )
    agent = Agent(llm=llm, model="test")
    agent.step(state)
    reloaded = CaseState.load_or_create("case005", Path(state.run_dir))
    assert reloaded.final_status == "accepted"


def test_fail_case_sets_status_failed(runtime) -> None:
    run_root, _ = runtime
    state = CaseState.load_or_create("case006", safe_run_dir("case006", run_root))
    llm = ScriptedLLMClient(script=[_decision(stage="preparation", next_action="fail_case")])
    agent = Agent(llm=llm, model="test")
    agent.step(state)
    reloaded = CaseState.load_or_create("case006", Path(state.run_dir))
    assert reloaded.final_status == "failed"


# ---------------------------------------------------------------------------
# Full loop
# ---------------------------------------------------------------------------


def test_full_loop_advances_then_terminates(runtime) -> None:
    run_root, _ = runtime
    llm = ScriptedLLMClient(
        script=[
            _decision(stage="preparation", next_action="advance_stage", step_id="s1"),
            _decision(stage="density_assignment", next_action="advance_stage", step_id="s2"),
            _decision(stage="ion_analysis", next_action="advance_stage", step_id="s3"),
            _decision(stage="smart_refine_2", next_action="advance_stage", step_id="s4"),
            _decision(
                stage="final_report",
                next_action="accept_assignments",
                step_id="s5",
                confidence=0.9,
            ),
        ]
    )
    agent = Agent(llm=llm, model="test", max_steps=10)
    state = agent.run("case_full")
    assert state.stage == "final_report"
    assert state.final_status == "accepted"
    assert state.controller_step_count == 5


def test_max_steps_cap_respected(runtime) -> None:
    run_root, _ = runtime
    # Infinite no-op script — would never terminate without the cap.
    llm = ScriptedLLMClient(
        script=[_decision(stage="final_report", next_action="run_tool", tool_name="summarise_case_state") for _ in range(20)]
    )
    agent = Agent(llm=llm, model="test", max_steps=3)
    state = agent.run("case_cap")
    assert state.controller_step_count == 3


# ---------------------------------------------------------------------------
# Fix #1 — richer tool result summary
# ---------------------------------------------------------------------------


def test_summarise_tool_result_passthrough_when_small() -> None:
    small = {"status": "ok", "binding_sites": [{"site_id": "site_001"}]}
    assert _summarise_tool_result(small) == small


def test_summarise_tool_result_truncates_huge_payload() -> None:
    huge = {
        "status": "ok",
        "tool_tier": "monolithic",
        "blob": "x" * 10_000,
    }
    summary = _summarise_tool_result(huge, max_chars=500)
    assert summary["status"] == "ok"
    assert summary["tool_tier"] == "monolithic"
    assert summary["_truncated"] is True
    assert summary["_size_chars"] > 500
    assert len(summary["_preview"]) <= 500


def test_run_tool_logs_rich_result_in_decision_log(runtime) -> None:
    """The tool's full output (truncated to TOOL_RESULT_SUMMARY_MAX_CHARS)
    must reach the decision log so the next prompt can include it."""
    run_root, _ = runtime
    state = CaseState.load_or_create("rich_log", safe_run_dir("rich_log", run_root))
    llm = ScriptedLLMClient(
        script=[
            _decision(
                stage="preparation",
                next_action="run_tool",
                tool_name="generate_binding_sites",
                tool_arguments={"max_sites": 2},
            )
        ]
    )
    agent = Agent(llm=llm, model="test")
    agent.step(state)

    import json as _json

    log_path = Path(state.run_dir) / "decision_log.jsonl"
    lines = log_path.read_text().splitlines()
    tool_calls = [_json.loads(line) for line in lines if _json.loads(line)["kind"] == "tool_call"]
    assert tool_calls, "no tool_call entries logged"
    last = tool_calls[-1]["payload"]["result_summary"]
    # Real summary now carries the tool's actual payload, not just status.
    assert last["status"] == "ok"
    assert "binding_sites" in last
    assert last["binding_sites"][0]["site_id"] == "site_001"


# ---------------------------------------------------------------------------
# Fix #3 — stubs persist to CaseState so agent _evidence shows progress
# ---------------------------------------------------------------------------


def test_generate_binding_sites_grows_binding_sites_count(runtime) -> None:
    run_root, _ = runtime
    state = CaseState.load_or_create("ev1", safe_run_dir("ev1", run_root))
    llm = ScriptedLLMClient(
        script=[
            _decision(
                stage="preparation",
                next_action="run_tool",
                tool_name="generate_binding_sites",
                tool_arguments={"max_sites": 3},
            )
        ]
    )
    agent = Agent(llm=llm, model="test")
    agent.step(state)
    reloaded = CaseState.load_or_create("ev1", Path(state.run_dir))
    assert len(reloaded.binding_sites) == 3
    evidence = agent._evidence(reloaded)
    assert evidence["counts"]["binding_sites"] == 3
    assert evidence["binding_site_ids"][0].startswith("site_")


def test_prepare_ligand_library_grows_ligand_count(runtime) -> None:
    run_root, _ = runtime
    state = CaseState.load_or_create("ev2", safe_run_dir("ev2", run_root))
    llm = ScriptedLLMClient(
        script=[
            _decision(
                stage="preparation",
                next_action="run_tool",
                tool_name="prepare_ligand_library",
                tool_arguments={"ligand_input": "lib.sdf", "ligand_ids": ["ATP", "GTP"]},
            )
        ]
    )
    agent = Agent(llm=llm, model="test")
    agent.step(state)
    reloaded = CaseState.load_or_create("ev2", Path(state.run_dir))
    assert set(reloaded.ligand_library.keys()) == {"ATP", "GTP"}
    assert agent._evidence(reloaded)["counts"]["ligand_library"] == 2


def test_dock_grows_poses_count(runtime) -> None:
    run_root, _ = runtime
    state = CaseState.load_or_create("ev3", safe_run_dir("ev3", run_root))
    llm = ScriptedLLMClient(
        script=[
            _decision(
                stage="density_assignment",
                next_action="run_tool",
                tool_name="dock_ligands_to_segment",
                tool_arguments={
                    "segment_id": "site_001_blob_001",
                    "ligand_ids": ["ATP"],
                    "max_poses": 4,
                },
            )
        ]
    )
    agent = Agent(llm=llm, model="test")
    agent.step(state)
    reloaded = CaseState.load_or_create("ev3", Path(state.run_dir))
    assert len(reloaded.poses) == 4
    assert agent._evidence(reloaded)["counts"]["poses"] == 4
    # Pose metadata round-trips
    sample = next(iter(reloaded.poses.values()))
    assert sample.segment_id == "site_001_blob_001"
    assert sample.ligand_id == "ATP"


# ---------------------------------------------------------------------------
# Fix #2 — duplicate-call guard
# ---------------------------------------------------------------------------


def test_tool_signature_canonicalises_argument_order() -> None:
    a = _tool_signature("dock", {"x": 1, "y": 2})
    b = _tool_signature("dock", {"y": 2, "x": 1})
    assert a == b


def test_duplicate_consecutive_call_is_rejected(runtime) -> None:
    run_root, _ = runtime
    state = CaseState.load_or_create("dup_case", safe_run_dir("dup_case", run_root))

    args = {"max_sites": 2}
    dup = _decision(
        stage="preparation",
        next_action="run_tool",
        tool_name="generate_binding_sites",
        tool_arguments=args,
        step_id="dup1",
    )
    dup2 = _decision(
        stage="preparation",
        next_action="run_tool",
        tool_name="generate_binding_sites",
        tool_arguments=args,
        step_id="dup2",
    )
    llm = ScriptedLLMClient(script=[dup, dup2])
    agent = Agent(llm=llm, model="test")

    # First call runs normally.
    rec1 = agent.step(state)
    assert rec1.tool_result and rec1.tool_result["status"] == "ok"

    # Second identical call is rejected by the guard.
    state2 = CaseState.load_or_create("dup_case", Path(state.run_dir))
    rec2 = agent.step(state2)
    assert rec2.tool_result and rec2.tool_result["status"] == "duplicate_rejected"
    assert "DO NOT repeat" in rec2.tool_result["warning"]
    # The model needs to see what the previous call returned so it can act on it
    # rather than just being told "rejected".
    assert rec2.tool_result["previous_result_summary"] is not None
    assert rec2.tool_result["previous_result_summary"].get("status") == "ok"


def test_non_consecutive_duplicate_is_allowed(runtime) -> None:
    """An identical call is fine if a different tool ran in between."""
    run_root, _ = runtime
    state = CaseState.load_or_create("nondup", safe_run_dir("nondup", run_root))

    args = {"max_sites": 2}
    same = lambda step_id: _decision(
        stage="preparation",
        next_action="run_tool",
        tool_name="generate_binding_sites",
        tool_arguments=args,
        step_id=step_id,
    )
    other = _decision(
        stage="preparation",
        next_action="run_tool",
        tool_name="prepare_ligand_library",
        tool_arguments={"ligand_input": "ligs.sdf"},
        step_id="other",
    )
    llm = ScriptedLLMClient(script=[same("a"), other, same("b")])
    agent = Agent(llm=llm, model="test")

    r1 = agent.step(CaseState.load_or_create("nondup", Path(state.run_dir)))
    assert r1.tool_result["status"] == "ok"
    r2 = agent.step(CaseState.load_or_create("nondup", Path(state.run_dir)))
    assert r2.tool_result["status"] == "ok"
    r3 = agent.step(CaseState.load_or_create("nondup", Path(state.run_dir)))
    assert r3.tool_result["status"] == "ok"


def test_tool_state_mutations_survive_agent_save(runtime) -> None:
    """Regression: agent.log_tool_call must not clobber on-disk state that the
    tool just wrote. generate_alpha_masks adds SegmentEvidence entries via
    state.add_segment(); after the agent step those segments must still be in
    state.json when the next iteration reloads.
    """
    run_root, _ = runtime
    state = CaseState.load_or_create(
        "race_case", safe_run_dir("race_case", run_root)
    )
    llm = ScriptedLLMClient(
        script=[
            _decision(
                stage="preparation",
                next_action="run_tool",
                tool_name="generate_alpha_masks",
                tool_arguments={
                    "threshold_sigma": 2.5,
                    "smoothing": 1.0,
                    "min_blob_volume": 50.0,
                    "site_ids": ["site_001"],
                    "blobs_per_site": 2,
                },
            )
        ]
    )
    agent = Agent(llm=llm, model="test")
    rec = agent.step(state)
    assert rec.tool_result and rec.tool_result["status"] == "ok"

    # Next-iteration view of state must see the tool's segments.
    reloaded = CaseState.load_or_create("race_case", Path(state.run_dir))
    assert len(reloaded.segments) == 2, (
        "agent.log_tool_call clobbered tool-side state mutations; "
        f"got segments={list(reloaded.segments.keys())}"
    )
    # And bookkeeping fields the agent owns must also be persisted.
    assert reloaded.last_tool_signature is not None
    assert reloaded.tool_call_count == 1


def test_last_tool_signature_persists_across_reload(runtime) -> None:
    run_root, _ = runtime
    state = CaseState.load_or_create("persist_sig", safe_run_dir("persist_sig", run_root))
    llm = ScriptedLLMClient(
        script=[
            _decision(
                stage="preparation",
                next_action="run_tool",
                tool_name="generate_binding_sites",
                tool_arguments={"max_sites": 1},
            )
        ]
    )
    agent = Agent(llm=llm, model="test")
    agent.step(state)

    reloaded = CaseState.load_or_create("persist_sig", Path(state.run_dir))
    assert reloaded.last_tool_signature is not None
    assert reloaded.last_tool_signature.startswith("generate_binding_sites::")


# ---------------------------------------------------------------------------
# Fix #3 — stage_exit_ready + state-visible stub persistence
# ---------------------------------------------------------------------------


def test_stage_exit_ready_preparation_requires_all_three(runtime) -> None:
    from chemem_mcp.schemas import (
        AlphaMaskParameters,
        BindingSiteRecord,
        LigandLibraryEntry,
        SegmentEvidence,
    )

    run_root, _ = runtime
    state = CaseState.load_or_create("exit_prep", safe_run_dir("exit_prep", run_root))
    assert _stage_exit_ready(state) is False

    state.add_binding_site(
        BindingSiteRecord(site_id="s1", center_xyz=(0, 0, 0), radius_angstrom=5.0)
    )
    assert _stage_exit_ready(state) is False  # still missing ligand + segment

    state.add_ligand(LigandLibraryEntry(ligand_id="ATP", source="l.sdf"))
    assert _stage_exit_ready(state) is False  # still missing segment

    state.add_segment(
        SegmentEvidence(
            segment_id="s1_blob_001",
            center_xyz=(0, 0, 0),
            volume_angstrom3=100,
            peak_density=2.0,
            mean_density=0.5,
            alpha_mask_parameters=AlphaMaskParameters(
                threshold_sigma=2.0, smoothing=1.0, min_blob_volume=50.0
            ),
        )
    )
    assert _stage_exit_ready(state) is True


def test_stage_exit_ready_smart_refine_2_needs_refinement(runtime) -> None:
    run_root, _ = runtime
    state = CaseState.load_or_create("exit_sr2", safe_run_dir("exit_sr2", run_root))
    state.set_stage("smart_refine_2")
    assert _stage_exit_ready(state) is False


def test_evidence_surfaces_stage_exit_ready_and_new_counts(runtime) -> None:
    """The agent must expose stage_exit_ready + the new counts to the prompt."""
    run_root, _ = runtime
    state = CaseState.load_or_create("ev_case", safe_run_dir("ev_case", run_root))

    agent = Agent(llm=ScriptedLLMClient(), model="test")
    ev = agent._evidence(state)
    assert "stage_exit_ready" in ev
    assert ev["stage_exit_ready"] is False
    for key in ("binding_sites", "ligand_library", "segments", "poses"):
        assert key in ev["counts"], key


def test_generate_binding_sites_populates_state(runtime) -> None:
    """Stubbed Tier-1 generate_binding_sites must persist binding sites so the
    next prompt cycle sees counts.binding_sites > 0."""
    from chemem_mcp.tools import monolithic

    run_root, _ = runtime
    res = monolithic.generate_binding_sites(run_id="prep_state_a", max_sites=3)
    assert res["status"] == "ok"
    reloaded = CaseState.load_or_create("prep_state_a", run_root / "prep_state_a")
    assert len(reloaded.binding_sites) == 3


def test_prepare_ligand_library_populates_state(runtime) -> None:
    from chemem_mcp.tools import monolithic

    run_root, _ = runtime
    monolithic.prepare_ligand_library(
        run_id="prep_state_b", ligand_input="ligs.sdf", ligand_ids=["ATP", "GTP"]
    )
    reloaded = CaseState.load_or_create("prep_state_b", run_root / "prep_state_b")
    assert set(reloaded.ligand_library.keys()) == {"ATP", "GTP"}


def test_dock_ligands_to_segment_populates_poses(runtime) -> None:
    from chemem_mcp.tools import monolithic

    run_root, _ = runtime
    monolithic.dock_ligands_to_segment(
        run_id="prep_state_c",
        segment_id="site_001_blob_001",
        ligand_ids=["ATP"],
        max_poses=4,
    )
    reloaded = CaseState.load_or_create("prep_state_c", run_root / "prep_state_c")
    assert len(reloaded.poses) == 4
    sample = next(iter(reloaded.poses.values()))
    assert sample.segment_id == "site_001_blob_001"
    assert sample.ligand_id == "ATP"


# ---------------------------------------------------------------------------
# Fix #4 — bad_arguments surfaced + tool signatures in prompt
# ---------------------------------------------------------------------------


def test_tool_signature_spec_omits_run_id() -> None:
    from chemem_mcp.tools import monolithic

    spec = _tool_signature_spec("generate_binding_sites", monolithic.generate_binding_sites)
    assert spec.startswith("generate_binding_sites(")
    assert "run_id" not in spec  # injected by the agent
    assert "max_sites" in spec


def test_bad_arguments_is_logged_to_decision_log_with_expected_signature(runtime) -> None:
    """A TypeError from the tool must surface as a status='bad_arguments' entry
    in decision_log.jsonl so the next prompt cycle sees the failure."""
    import json as _json

    run_root, _ = runtime
    state = CaseState.load_or_create(
        "bad_args_case", safe_run_dir("bad_args_case", run_root)
    )
    llm = ScriptedLLMClient(
        script=[
            _decision(
                stage="preparation",
                next_action="run_tool",
                tool_name="generate_binding_sites",
                # Wrong args — model hallucinated map_mrc/protein_pdb here.
                tool_arguments={"map_mrc": "/tmp/m.mrc", "protein_pdb": "/tmp/p.pdb"},
            )
        ]
    )
    agent = Agent(llm=llm, model="test")
    record = agent.step(state)

    assert record.error and record.error.startswith("bad_arguments")
    assert record.tool_result and record.tool_result["status"] == "bad_arguments"
    assert "expected_signature" in record.tool_result
    assert "max_sites" in record.tool_result["expected_signature"]

    log = (Path(state.run_dir) / "decision_log.jsonl").read_text().splitlines()
    tool_calls = [_json.loads(line) for line in log if _json.loads(line)["kind"] == "tool_call"]
    assert tool_calls, "bad_arguments call must still produce a tool_call log entry"
    last = tool_calls[-1]["payload"]["result_summary"]
    assert last["status"] == "bad_arguments"
    assert last["tool_name"] == "generate_binding_sites"
    assert "max_sites" in last["expected_signature"]


def test_step_includes_tool_signatures_in_prompt(runtime) -> None:
    """The agent must thread tool_signatures into the user message so the LLM
    can see each tool's accepted arguments."""
    captured: dict[str, Any] = {}

    class CapturingLLM:
        def chat_json(self, **kwargs):
            captured["user"] = kwargs["user"]
            raise InvalidLLMResponse("test stops here", attempts=1)

    run_root, _ = runtime
    state = CaseState.load_or_create("cap", safe_run_dir("cap", run_root))
    agent = Agent(llm=CapturingLLM(), model="test")
    agent.step(state)

    assert "tool_signatures" in captured["user"]
    # spot-check a couple of tools with distinctive arg names
    assert "max_sites" in captured["user"]  # generate_binding_sites signature
    assert "search_radius_angstrom" in captured["user"]  # dock_ligands_to_segment


def test_run_tool_respects_rate_limit(runtime) -> None:
    """When the rate limiter blocks the tool, the tool returns a rate_limited
    envelope (rather than raising), and the agent records it as a tool call."""
    run_root, rl = runtime
    # Exhaust the budget for prepare_case (default limit 1).
    rl.check_and_increment("prepare_case", "rl_case")
    state = CaseState.load_or_create("rl_case", safe_run_dir("rl_case", run_root))
    llm = ScriptedLLMClient(
        script=[
            _decision(
                stage="preparation",
                next_action="run_tool",
                tool_name="prepare_case",
                tool_arguments={
                    "protein_pdb": "p",
                    "map_mrc": "m",
                    "ligand_library": "l",
                },
            )
        ]
    )
    agent = Agent(llm=llm, model="test")
    record = agent.step(state)
    assert record.tool_result and record.tool_result["status"] == "rate_limited"
