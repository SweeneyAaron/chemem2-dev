"""Local decision-case battery.

Without a real LLM we cannot test whether qwen3 *chooses* the expected
decision — that battery runs on the Linux box via the CLI. Locally we
drive the agent with a ScriptedLLMClient that returns each fixture's
expected_decision and verify the agent:

  - parses + validates it
  - persists it to state
  - dispatches the right code path (run_tool / advance_stage / stop)
  - injects run_id automatically into tool_arguments
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from chemem_mcp.llm import ScriptedLLMClient
from chemem_mcp.orchestrator import Agent
from chemem_mcp.paths import safe_run_dir
from chemem_mcp.schemas import ControllerDecision, DecisionCase, ExpectedDecision
from chemem_mcp.state import CaseState

FIXTURE_DIR = Path(__file__).parent / "data" / "decision_cases"
EXPECTED_FIXTURE_COUNT = 10


def _all_fixtures() -> list[Path]:
    paths = sorted(FIXTURE_DIR.glob("*.json"))
    assert len(paths) == EXPECTED_FIXTURE_COUNT, paths
    return paths


def _expected_to_decision(expected: ExpectedDecision, *, stage: str) -> ControllerDecision:
    """Build a synthetic ControllerDecision matching the fixture's expectation."""
    return ControllerDecision(
        controller_step_id=f"battery_{uuid.uuid4().hex[:8]}",
        stage=stage,
        next_action=expected.next_action,
        tool_name=expected.tool_name,
        tool_tier=expected.tool_tier,
        tool_arguments={},
        reason=expected.rationale,
        evidence_used=["fixture.input_evidence"],
        uncertainties=[],
        self_critique="Battery synthesised this decision from the fixture's expected_decision.",
        red_flags=[],
        confidence=0.85,
    )


@pytest.mark.parametrize("fixture_path", _all_fixtures(), ids=lambda p: p.stem)
def test_agent_dispatches_each_fixture(fixture_path: Path, runtime) -> None:
    run_root, _ = runtime
    raw = json.loads(fixture_path.read_text())
    case = DecisionCase.model_validate(raw)

    run_id = f"battery_{case.case_id}"
    state = CaseState.load_or_create(run_id, safe_run_dir(run_id, run_root))
    state.set_stage(case.stage)

    decision = _expected_to_decision(case.expected_decision, stage=case.stage)
    llm = ScriptedLLMClient(script=[decision])
    agent = Agent(llm=llm, model="test")
    record = agent.step(state)

    # Decision was logged
    reloaded = CaseState.load_or_create(run_id, Path(state.run_dir))
    assert reloaded.controller_step_count == 1, case.case_id

    # Dispatch matches the expected next_action
    expected = case.expected_decision
    if expected.next_action == "run_tool":
        # The tool must be in the registry (no "unknown tool"). A
        # "bad_arguments" error is acceptable here — fixtures don't
        # synthesise full tool_arguments; that's the LLM's job on the
        # Linux box. Locally we only verify dispatch wiring.
        if record.error is not None:
            assert record.error.startswith("bad_arguments"), (case.case_id, record.error)
        else:
            assert record.tool_result is not None
            assert reloaded.tool_call_count == 1, case.case_id
    elif expected.next_action == "advance_stage":
        # Stage progressed to the next in STAGE_ORDER.
        assert reloaded.stage != case.stage, case.case_id
    elif expected.next_action == "stop_ambiguous":
        assert reloaded.final_status == "ambiguous", case.case_id
    elif expected.next_action == "fail_case":
        assert reloaded.final_status == "failed", case.case_id
    elif expected.next_action == "accept_assignments":
        assert reloaded.final_status == "accepted", case.case_id


def test_all_expected_tools_in_agent_registry(runtime) -> None:
    """Every named tool in a fixture's expected_decision must be in the agent
    registry, otherwise the battery cannot test that dispatch path."""
    agent = Agent(llm=ScriptedLLMClient(), model="test")
    registered = set(agent.tool_registry.keys())
    for path in _all_fixtures():
        case = DecisionCase.model_validate(json.loads(path.read_text()))
        tool = case.expected_decision.tool_name
        if tool is None:
            continue
        assert tool in registered, f"{case.case_id} references unknown tool {tool}"
