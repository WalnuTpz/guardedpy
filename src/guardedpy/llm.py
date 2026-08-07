"""Injectable single-completion clients used by the self-owned harness loop."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Callable, Protocol

from openai import APIConnectionError, APITimeoutError

from guardedpy.config import HarnessConfig
from guardedpy.conversation import (
    ModelChunk,
    ProviderMessage,
    ReasoningDelta,
    ResponseFinished,
    TemporaryProviderFailure,
    TextDelta,
    ToolCallDelta,
    ToolDefinition,
)
from guardedpy.context import LlmContext


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
        config: HarnessConfig,
        transport_factory: Callable[[str], Any],
    ) -> None:
        self._key_provider = key_provider
        self._config = config.model_copy(deep=True)
        self._transport_factory = transport_factory

    def complete(self, context: LlmContext) -> str:
        """Return the provider payload unchanged, retrying one temporary transport failure."""
        transport = self._transport_factory(self._key_provider())
        for attempt in range(2):
            try:
                response = transport.chat.completions.create(
                    model=self._config.model,
                    messages=context.messages(),
                    response_format={"type": "json_object"},
                    reasoning_effort=self._config.reasoning_effort,
                    extra_body={"thinking": {"type": "enabled"}},
                )
                return response.choices[0].message.content
            except (ConnectionError, TimeoutError, APIConnectionError, APITimeoutError):
                if attempt == 1:
                    raise TemporaryProviderFailure from None
        raise AssertionError("unreachable")


class DeepSeekConversationModel:
    """Translate one DeepSeek streaming response into conversation chunks."""

    def __init__(
        self,
        key_provider: Callable[[], str],
        config: HarnessConfig,
        transport_factory: Callable[..., Any],
    ) -> None:
        self._key_provider = key_provider
        self._config = config.model_copy(deep=True)
        self._transport_factory = transport_factory

    def stream(
        self,
        messages: tuple[ProviderMessage, ...],
        tools: tuple[ToolDefinition, ...],
    ) -> Iterator[ModelChunk]:
        transport = self._transport_factory(self._key_provider(), max_retries=0)
        request = {
            "model": self._config.model,
            "messages": [_provider_message(message) for message in messages],
            "tools": [_tool_definition(tool) for tool in tools],
            "stream": True,
            "reasoning_effort": self._config.reasoning_effort,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
        for attempt in range(2):
            yielded = False
            try:
                response = transport.chat.completions.create(**request)
                for provider_chunk in response:
                    for chunk in _model_chunks(provider_chunk):
                        yielded = True
                        yield chunk
                return
            except (ConnectionError, TimeoutError, APIConnectionError, APITimeoutError):
                if yielded or attempt == 1:
                    raise TemporaryProviderFailure from None
        raise AssertionError("unreachable")


def _provider_message(message: ProviderMessage) -> dict[str, object]:
    if message.role in ("system", "user"):
        return {"role": message.role, "content": message.content}
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    payload: dict[str, object] = {
        "role": "assistant",
        "content": message.content,
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments_json,
                },
            }
            for call in message.tool_calls
        ]
    if message.reasoning_content:
        payload["reasoning_content"] = message.reasoning_content
    return payload


def _tool_definition(tool: ToolDefinition) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.json_schema,
        },
    }


def _model_chunks(provider_chunk: Any) -> Iterator[ModelChunk]:
    choice = provider_chunk.choices[0]
    delta = choice.delta
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning is not None:
        yield ReasoningDelta(reasoning)
    content = getattr(delta, "content", None)
    if content is not None:
        yield TextDelta(content)
    for call in getattr(delta, "tool_calls", None) or ():
        function = call.function
        yield ToolCallDelta(
            index=call.index,
            id=call.id,
            name=function.name,
            arguments_fragment=function.arguments or "",
        )
    if choice.finish_reason is not None:
        yield ResponseFinished(choice.finish_reason)
