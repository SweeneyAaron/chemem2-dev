from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from chemem_mcp.llm import InvalidLLMResponse
from chemem_mcp.ollama_client import OllamaClient
from chemem_mcp.schemas import ControllerDecision

GOLDEN_DECISION_JSON = json.dumps(
    {
        "schema_version": "0.1",
        "controller_step_id": "x_001",
        "stage": "preparation",
        "next_action": "advance_stage",
        "tool_name": None,
        "tool_tier": None,
        "tool_arguments": {},
        "reason": "inputs validated",
        "evidence_used": [],
        "uncertainties": [],
        "self_critique": "could miss a malformed input file",
        "red_flags": [],
        "confidence": 0.8,
    }
)


@dataclass
class FakeResponse:
    payload: dict[str, Any]
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self.payload


@dataclass
class FakeHTTP:
    """Captures POST calls and returns scripted responses."""

    responses: list[FakeResponse]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def post(self, url: str, *, json: dict[str, Any], timeout: float) -> FakeResponse:  # noqa: A002
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if not self.responses:
            raise AssertionError("FakeHTTP: no more scripted responses")
        return self.responses.pop(0)


def _ollama_response(content: str) -> FakeResponse:
    return FakeResponse({"message": {"role": "assistant", "content": content}})


def test_chat_json_happy_path() -> None:
    http = FakeHTTP(responses=[_ollama_response(GOLDEN_DECISION_JSON)])
    client = OllamaClient(_http=http)
    parsed, meta = client.chat_json(
        model="qwen3:14b",
        system="system",
        user="user",
        response_model=ControllerDecision,
    )
    assert isinstance(parsed, ControllerDecision)
    assert parsed.next_action == "advance_stage"
    assert meta["attempts"] == 1
    assert len(http.calls) == 1
    sent = http.calls[0]["json"]
    assert sent["model"] == "qwen3:14b"
    assert sent["format"] == "json"
    assert sent["stream"] is False
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][1]["role"] == "user"


def test_chat_json_retries_once_on_invalid_then_succeeds() -> None:
    http = FakeHTTP(
        responses=[
            _ollama_response("not even json"),
            _ollama_response(GOLDEN_DECISION_JSON),
        ]
    )
    client = OllamaClient(_http=http, max_retries=1)
    parsed, meta = client.chat_json(
        model="qwen3:14b",
        system="sys",
        user="usr",
        response_model=ControllerDecision,
    )
    assert isinstance(parsed, ControllerDecision)
    assert meta["attempts"] == 2
    # Second call should have included the assistant's bad response + correction request
    assert len(http.calls) == 2
    second_messages = http.calls[1]["json"]["messages"]
    assert second_messages[-1]["role"] == "user"
    assert "failed validation" in second_messages[-1]["content"]
    assert any(m["role"] == "assistant" and m["content"] == "not even json" for m in second_messages)


def test_chat_json_raises_after_retries_exhausted() -> None:
    http = FakeHTTP(
        responses=[
            _ollama_response("garbage"),
            _ollama_response("still garbage"),
        ]
    )
    client = OllamaClient(_http=http, max_retries=1)
    with pytest.raises(InvalidLLMResponse) as ei:
        client.chat_json(
            model="qwen3:14b",
            system="s",
            user="u",
            response_model=ControllerDecision,
        )
    assert ei.value.attempts == 2
    assert "still garbage" in (ei.value.content or "")


def test_chat_json_retries_on_schema_violation() -> None:
    """A valid JSON object that violates the schema should also trigger retry."""
    bad = json.dumps({"not_what_was_asked": True})
    http = FakeHTTP(
        responses=[
            _ollama_response(bad),
            _ollama_response(GOLDEN_DECISION_JSON),
        ]
    )
    client = OllamaClient(_http=http, max_retries=1)
    parsed, meta = client.chat_json(
        model="qwen3:14b",
        system="s",
        user="u",
        response_model=ControllerDecision,
    )
    assert isinstance(parsed, ControllerDecision)
    assert meta["attempts"] == 2


def test_http_error_propagates() -> None:
    http = FakeHTTP(responses=[FakeResponse({}, status_code=500)])
    client = OllamaClient(_http=http, max_retries=1)
    with pytest.raises(RuntimeError, match="http 500"):
        client.chat_json(
            model="qwen3:14b",
            system="s",
            user="u",
            response_model=ControllerDecision,
        )


def test_temperature_and_format_in_payload() -> None:
    http = FakeHTTP(responses=[_ollama_response(GOLDEN_DECISION_JSON)])
    client = OllamaClient(_http=http, temperature=0.25)
    client.chat_json(
        model="qwen3:14b", system="s", user="u", response_model=ControllerDecision
    )
    sent = http.calls[0]["json"]
    assert sent["options"]["temperature"] == 0.25
    assert sent["format"] == "json"


# ---------------------------------------------------------------------------
# Fix B — tolerant pre-parser
# ---------------------------------------------------------------------------


def test_tolerant_parser_strips_unknown_top_level_keys() -> None:
    """toy_008 regression: a partial rate_limit_check dict (now an unknown
    key) must be stripped, not crash the decision."""
    payload = json.loads(GOLDEN_DECISION_JSON)
    payload["rate_limit_check"] = {"key": "case_x", "limit": 3}  # missing tool_name
    payload["mystery_field"] = "garbage"
    http = FakeHTTP(responses=[_ollama_response(json.dumps(payload))])
    client = OllamaClient(_http=http)
    parsed, meta = client.chat_json(
        model="qwen3:14b", system="s", user="u", response_model=ControllerDecision
    )
    assert isinstance(parsed, ControllerDecision)
    assert meta["attempts"] == 1  # no retry needed
    repairs = meta["tolerant_repairs"]
    assert any("dropped_extra_keys" in r for r in repairs)
    # Both extras must be reported as dropped.
    dropped = [r for r in repairs if r.startswith("dropped_extra_keys=")][0]
    assert "rate_limit_check" in dropped
    assert "mystery_field" in dropped


def test_tolerant_parser_coerces_str_confidence() -> None:
    payload = json.loads(GOLDEN_DECISION_JSON)
    payload["confidence"] = "0.65"  # LLM returned a stringified float
    http = FakeHTTP(responses=[_ollama_response(json.dumps(payload))])
    client = OllamaClient(_http=http)
    parsed, meta = client.chat_json(
        model="qwen3:14b", system="s", user="u", response_model=ControllerDecision
    )
    assert parsed.confidence == 0.65
    assert "coerced_confidence_str_to_float" in meta["tolerant_repairs"]


def test_tolerant_parser_supplies_default_for_missing_red_flags() -> None:
    payload = json.loads(GOLDEN_DECISION_JSON)
    del payload["red_flags"]
    http = FakeHTTP(responses=[_ollama_response(json.dumps(payload))])
    client = OllamaClient(_http=http)
    parsed, meta = client.chat_json(
        model="qwen3:14b", system="s", user="u", response_model=ControllerDecision
    )
    assert parsed.red_flags == []
    assert "defaulted_red_flags" in meta["tolerant_repairs"]


def test_tolerant_parser_no_repairs_when_clean() -> None:
    http = FakeHTTP(responses=[_ollama_response(GOLDEN_DECISION_JSON)])
    client = OllamaClient(_http=http)
    parsed, meta = client.chat_json(
        model="qwen3:14b", system="s", user="u", response_model=ControllerDecision
    )
    assert isinstance(parsed, ControllerDecision)
    # red_flags default + tool_arguments etc. may be filled if absent in payload;
    # the canonical GOLDEN_DECISION_JSON already includes them, so nothing fires.
    assert meta["tolerant_repairs"] == []


# ---------------------------------------------------------------------------
# Fix C — structured retry feedback
# ---------------------------------------------------------------------------


def test_retry_message_is_structured_field_list() -> None:
    """On schema violation, the retry message must enumerate the failed fields
    as bullets (not a raw pydantic dump)."""
    bad = json.dumps(
        {
            "controller_step_id": "x",
            "stage": "preparation",
            "next_action": "advance_stage",
            "reason": "ok",
            # missing self_critique + confidence
        }
    )
    http = FakeHTTP(
        responses=[
            _ollama_response(bad),
            _ollama_response(GOLDEN_DECISION_JSON),
        ]
    )
    client = OllamaClient(_http=http, max_retries=1)
    client.chat_json(
        model="qwen3:14b", system="s", user="u", response_model=ControllerDecision
    )
    retry_msg = http.calls[1]["json"]["messages"][-1]["content"]
    assert "Validation failed on these fields:" in retry_msg
    assert "self_critique" in retry_msg
    assert "- " in retry_msg  # bullet format
