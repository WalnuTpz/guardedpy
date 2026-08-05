"""Injectable single-completion clients used by the self-owned harness loop."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from openai import APIConnectionError, APITimeoutError

from guardedpy.context import LlmContext


class TemporaryProviderFailure(RuntimeError):
    """The provider remained temporarily unavailable after the bounded retry."""


class LLMClient(Protocol):
    """Return exactly one untrusted action payload for one context."""

    def complete(self, context: LlmContext) -> str:
        """Produce one JSON action payload."""


class ScriptedLLM:
    """Offline deterministic LLM double that records its received contexts."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.contexts: list[str] = []

    def complete(self, context: LlmContext) -> str:
        self.contexts.append(context.render())
        try:
            return next(self._responses)
        except StopIteration as error:
            raise RuntimeError("scripted LLM has no remaining response") from error


class DeepSeekClient:
    """One OpenAI-compatible JSON completion using a key fetched at call time."""

    def __init__(
        self,
        key_provider: Callable[[], str],
        model: str,
        transport_factory: Callable[[str], Any],
    ) -> None:
        self._key_provider = key_provider
        self._model = model
        self._transport_factory = transport_factory

    def complete(self, context: LlmContext) -> str:
        """Return the provider payload unchanged, retrying one temporary transport failure."""
        transport = self._transport_factory(self._key_provider())
        for attempt in range(2):
            try:
                response = transport.chat.completions.create(
                    model=self._model,
                    messages=context.messages(),
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content
            except (ConnectionError, TimeoutError, APIConnectionError, APITimeoutError):
                if attempt == 1:
                    raise TemporaryProviderFailure from None
        raise AssertionError("unreachable")
