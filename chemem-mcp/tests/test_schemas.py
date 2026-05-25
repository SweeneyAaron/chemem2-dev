from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from chemem_mcp.schemas import (
    SCHEMA_VERSION,
    ControllerDecision,
    FinalReport,
    IonCandidate,
    RefinementOutcome,
    SegmentEvidence,
    ToolTier,
)

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"

SCHEMA_MAP: dict[str, type[BaseModel]] = {
    "segment_evidence.json": SegmentEvidence,
    "ion_candidate.json": IonCandidate,
    "controller_decision.json": ControllerDecision,
    "refinement_outcome.json": RefinementOutcome,
    "final_report.json": FinalReport,
}


@pytest.mark.parametrize("filename,model_cls", list(SCHEMA_MAP.items()))
def test_golden_round_trip(filename: str, model_cls: type[BaseModel]) -> None:
    path = GOLDEN_DIR / filename
    raw = json.loads(path.read_text())
    parsed = model_cls.model_validate(raw)
    dumped = parsed.model_dump(mode="json")
    # round-trip: dumped must validate again
    reparsed = model_cls.model_validate(dumped)
    assert reparsed == parsed


@pytest.mark.parametrize("filename,model_cls", list(SCHEMA_MAP.items()))
def test_golden_has_schema_version(filename: str, model_cls: type[BaseModel]) -> None:
    raw = json.loads((GOLDEN_DIR / filename).read_text())
    assert raw.get("schema_version") == SCHEMA_VERSION


def test_extra_fields_rejected_on_segment() -> None:
    raw = json.loads((GOLDEN_DIR / "segment_evidence.json").read_text())
    raw["totally_made_up_field"] = "boom"
    with pytest.raises(ValidationError):
        SegmentEvidence.model_validate(raw)


def test_controller_decision_requires_self_critique() -> None:
    raw = json.loads((GOLDEN_DIR / "controller_decision.json").read_text())
    raw.pop("self_critique")
    with pytest.raises(ValidationError):
        ControllerDecision.model_validate(raw)


def test_controller_confidence_clamped() -> None:
    raw = json.loads((GOLDEN_DIR / "controller_decision.json").read_text())
    raw["confidence"] = 1.5
    with pytest.raises(ValidationError):
        ControllerDecision.model_validate(raw)
    raw["confidence"] = -0.1
    with pytest.raises(ValidationError):
        ControllerDecision.model_validate(raw)


def test_tool_tier_enum_values() -> None:
    assert ToolTier.MONOLITHIC.value == "monolithic"
    assert ToolTier.ATOMIC.value == "atomic"
    assert ToolTier.PARAMETER_TWEAK.value == "parameter_tweak"


def test_ion_confidence_clamped() -> None:
    raw = json.loads((GOLDEN_DIR / "ion_candidate.json").read_text())
    raw["confidence"] = 2.0
    with pytest.raises(ValidationError):
        IonCandidate.model_validate(raw)


def test_non_ligand_hypothesis_type_restricted() -> None:
    raw = json.loads((GOLDEN_DIR / "segment_evidence.json").read_text())
    raw["non_ligand_hypotheses"][0]["type"] = "alien_artifact"
    with pytest.raises(ValidationError):
        SegmentEvidence.model_validate(raw)
