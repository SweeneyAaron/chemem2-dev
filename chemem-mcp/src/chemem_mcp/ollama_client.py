"""Real Ollama-backed LLM client.

Talks to a local Ollama server (``http://localhost:11434`` by default) using
the ``/api/chat`` endpoint with ``format=json`` strict JSON mode.

Two layers of resilience against small-model JSON sloppiness:

1. **Tolerant pre-parser** — before strict validation, strip unknown
   top-level keys, coerce common type mismatches, and supply default
   containers for missing optional list/dict fields. Repairs are recorded
   in ``meta["tolerant_repairs"]`` for observability.
2. **Structured retry** — if strict validation still fails, format
   ``ValidationError.errors()`` as field-specific bullets so the next
   prompt tells the model exactly what to fix.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from .llm import InvalidLLMResponse

T = TypeVar("T", bound=BaseModel)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 1

# Fields whose missing-or-null defaults the tolerant parser will fill in.
_DEFAULT_CONTAINERS: dict[str, Any] = {
    "tool_arguments": {},
    "evidence_used": [],
    "uncertainties": [],
    "red_flags": [],
}

# Fields where stringified numerics should be coerced to float.
_FLOAT_COERCE_FIELDS: tuple[str, ...] = ("confidence",)


def _tolerant_repair(
    raw: Any, response_model: type[BaseModel]
) -> tuple[dict[str, Any], list[str]]:
    """Apply conservative repairs to a parsed JSON dict.

    Returns ``(cleaned_dict, repairs_log)``. If ``raw`` is not a dict, returns
    it unchanged and an empty log so the caller can fall through to strict
    validation (which will raise).
    """
    if not isinstance(raw, dict):
        return raw, []
    repairs: list[str] = []
    allowed = set(response_model.model_fields.keys())

    # 1. Drop unknown top-level keys.
    extras = [k for k in raw if k not in allowed]
    if extras:
        raw = {k: v for k, v in raw.items() if k in allowed}
        repairs.append(f"dropped_extra_keys={sorted(extras)}")

    # 2. Coerce stringified numerics for known float fields.
    for fname in _FLOAT_COERCE_FIELDS:
        if fname in raw and isinstance(raw[fname], str):
            try:
                raw[fname] = float(raw[fname])
                repairs.append(f"coerced_{fname}_str_to_float")
            except ValueError:
                pass  # let strict validation surface the real error

    # 3. Supply defaults for missing/null optional containers.
    for fname, default in _DEFAULT_CONTAINERS.items():
        if fname in allowed and (fname not in raw or raw[fname] is None):
            raw[fname] = default if not isinstance(default, (list, dict)) else type(default)(default)
            repairs.append(f"defaulted_{fname}")

    return raw, repairs


def _format_validation_errors(exc: ValidationError) -> str:
    """Render ValidationError.errors() as bullet points the LLM can act on."""
    lines = ["Validation failed on these fields:"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = err.get("msg", "")
        got = err.get("input", "<unset>")
        # Trim noisy long inputs
        got_repr = repr(got)
        if len(got_repr) > 120:
            got_repr = got_repr[:120] + "...]"
        lines.append(f"- {loc or '<root>'}: {msg} (got {got_repr})")
    lines.append(
        "Return the SAME JSON object with ONLY these fields corrected. "
        "Do not change other fields. No prose. No markdown fences."
    )
    return "\n".join(lines)


@dataclass
class OllamaClient:
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT_S
    temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = DEFAULT_MAX_RETRIES
    # Indirection so tests can pin the HTTP layer.
    _http: Any = None

    def __post_init__(self) -> None:
        if self._http is None:
            self._http = requests

    # ------------------------------------------------------------------

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        response_model: type[T],
    ) -> tuple[T, dict[str, Any]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error: str = ""
        last_content: str = ""
        all_repairs: list[str] = []
        for attempt in range(self.max_retries + 1):
            content = self._post(model, messages)
            last_content = content

            # Try tolerant repair first, then strict validation.
            repairs_this_round: list[str] = []
            try:
                raw = json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = str(exc)[:800]
                raw = None

            if isinstance(raw, dict):
                cleaned, repairs_this_round = _tolerant_repair(raw, response_model)
                all_repairs.extend(repairs_this_round)
                try:
                    parsed = response_model.model_validate(cleaned)
                    return parsed, {
                        "attempts": attempt + 1,
                        "raw_content": content,
                        "model": model,
                        "tolerant_repairs": all_repairs,
                    }
                except ValidationError as exc:
                    last_error = _format_validation_errors(exc)
            elif raw is not None:
                last_error = (
                    "Top-level JSON value must be an object, got "
                    f"{type(raw).__name__}."
                )

            if attempt >= self.max_retries:
                break

            # Feed structured feedback for the retry.
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response failed validation.\n\n"
                        f"{last_error}"
                    ),
                }
            )
        raise InvalidLLMResponse(
            f"Ollama response failed validation after {self.max_retries + 1} attempts: {last_error}",
            content=last_content,
            attempts=self.max_retries + 1,
        )

    # ------------------------------------------------------------------

    def _post(self, model: str, messages: list[dict[str, str]]) -> str:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        resp = self._http.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message") or {}
        content = message.get("content", "")
        if not isinstance(content, str):
            raise InvalidLLMResponse(
                f"unexpected content type from Ollama: {type(content).__name__}",
                content=str(content),
                attempts=1,
            )
        return content
