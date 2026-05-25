"""LLM client abstraction.

Production code uses :class:`OllamaClient` (see ``ollama_client.py``).
Tests inject :class:`ScriptedLLMClient` so the agent loop can be exercised
without a running Ollama process.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class InvalidLLMResponse(RuntimeError):
    """Raised when an LLM response fails JSON / schema validation after retries."""

    def __init__(self, message: str, *, content: Optional[str] = None, attempts: int = 0):
        super().__init__(message)
        self.content = content
        self.attempts = attempts


class LLMClient(Protocol):
    """Minimal contract every LLM backend must satisfy."""

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        response_model: type[T],
    ) -> tuple[T, dict[str, Any]]:
        """Send a chat completion request and return a validated pydantic model.

        Returns ``(parsed_model, meta_dict)`` where ``meta_dict`` includes
        diagnostic keys such as ``attempts`` and ``raw_content``.
        Raises :class:`InvalidLLMResponse` on persistent parsing failure.
        """
        ...


# ---------------------------------------------------------------------------
# Scripted client — for tests
# ---------------------------------------------------------------------------


ScriptItem = Union[BaseModel, dict, Exception]


@dataclass
class ScriptedLLMClient:
    """In-memory client that returns canned responses in order.

    Each item in ``script`` may be:
      * a pydantic ``BaseModel`` instance — returned as-is
      * a ``dict`` — validated against the requested ``response_model``
      * an ``Exception`` — raised when consumed
    """

    script: list[ScriptItem] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        response_model: type[T],
    ) -> tuple[T, dict[str, Any]]:
        self.calls.append(
            {
                "model": model,
                "system": system,
                "user": user,
                "response_model": response_model.__name__,
            }
        )
        if not self.script:
            raise InvalidLLMResponse(
                "scripted client exhausted; no more responses queued",
                attempts=0,
            )
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, BaseModel):
            if not isinstance(item, response_model):
                raise TypeError(
                    f"scripted item is {type(item).__name__}, expected {response_model.__name__}"
                )
            return item, {"scripted": True, "attempts": 1}
        if isinstance(item, dict):
            parsed = response_model.model_validate(item)
            return parsed, {"scripted": True, "attempts": 1}
        raise TypeError(f"unsupported scripted item type: {type(item).__name__}")

    def queue(self, item: ScriptItem) -> None:
        self.script.append(item)
