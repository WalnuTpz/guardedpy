"""Injectable single-completion clients used by the self-owned harness loop."""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Return exactly one untrusted action payload for one context."""

    def complete(self, context: str) -> str:
        """Produce one JSON action payload."""


class ScriptedLLM:
    """Offline deterministic LLM double that records its received contexts."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.contexts: list[str] = []

    def complete(self, context: str) -> str:
        self.contexts.append(context)
        try:
            return next(self._responses)
        except StopIteration as error:
            raise RuntimeError("scripted LLM has no remaining response") from error
