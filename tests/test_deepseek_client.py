"""Offline contract tests for the OpenAI-compatible DeepSeek client."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from guardedpy.context import LlmContext
from guardedpy.llm import DeepSeekClient


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]


class _Completions:
    def __init__(self, responses: list[object]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, _Response)
        return response


class _Transport:
    def __init__(self, responses: list[object]) -> None:
        self.chat = type("Chat", (), {"completions": _Completions(responses)})()


def test_deepseek_client_gets_key_at_call_time_and_uses_json_mode() -> None:
    """Catches eager key access or a completion request that permits non-JSON output."""
    keys: list[str] = []
    transport = _Transport([_Response([_Choice(_Message('{"kind":"finish","summary":"done","status":"blocked"}'))])])
    client = DeepSeekClient(lambda: keys.append("called") or "secret", "deepseek-v4-flash", lambda key: transport)

    assert keys == []
    assert client.complete(LlmContext.minimal()) == '{"kind":"finish","summary":"done","status":"blocked"}'
    assert keys == ["called"]
    assert transport.chat.completions.calls == [
        {
            "model": "deepseek-v4-flash",
            "messages": LlmContext.minimal().messages(),
            "response_format": {"type": "json_object"},
        }
    ]


def test_deepseek_client_returns_invalid_json_unchanged() -> None:
    """Catches adapter-side repair that would hide malformed model output from the action parser."""
    transport = _Transport([_Response([_Choice(_Message("not json"))])])

    assert DeepSeekClient(lambda: "secret", "deepseek-chat", lambda key: transport).complete(LlmContext.minimal()) == "not json"


def test_deepseek_client_retries_only_temporary_transport_failures() -> None:
    """Catches unbounded provider retry or swallowing a non-temporary provider exception."""
    temporary = _Transport(
        [
            ConnectionError("offline"),
            _Response([_Choice(_Message('{"kind":"finish","summary":"done","status":"blocked"}'))]),
        ]
    )
    client = DeepSeekClient(lambda: "secret", "deepseek-chat", lambda key: temporary)

    assert '"kind":"finish"' in client.complete(LlmContext.minimal())
    assert len(temporary.chat.completions.calls) == 2

    permanent = _Transport([ValueError("bad request")])
    with pytest.raises(ValueError, match="bad request"):
        DeepSeekClient(lambda: "secret", "deepseek-chat", lambda key: permanent).complete(LlmContext.minimal())
