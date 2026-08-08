"""Streaming DeepSeek adapter tests using a local fake transport."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import sys
from typing import Iterator

from guardedpy.config import HarnessConfig
from guardedpy.conversation import (
    ConversationAgent,
    ProviderMessage,
    ReasoningDelta,
    ResponseFinished,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
)
from guardedpy.discovery import ProjectProfile
from guardedpy.llm import DeepSeekConversationModel


def _config() -> HarnessConfig:
    return HarnessConfig(
        profile=ProjectProfile(
            root=Path.cwd().resolve(),
            discovery_source="tests_dir",
            source_dirs=(PurePosixPath("src"),),
            test_dirs=(PurePosixPath("tests"),),
            pytest_command=(sys.executable, "-m", "pytest"),
        ),
        model="deepseek-v4-pro",
        reasoning_effort="max",
        timeout_seconds=17,
    )


def _chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[object] | None = None,
    finish_reason: str | None = None,
) -> object:
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    )


class _Completions:
    def __init__(self, responses: list[object]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


class _Transport:
    def __init__(self, responses: list[object]) -> None:
        self.chat = SimpleNamespace(completions=_Completions(responses))


def test_deepseek_retries_one_temporary_transport_failure_before_first_chunk_only() -> None:
    successful_stream = iter(
        [
            _chunk(reasoning="private"),
            _chunk(content="answer"),
            _chunk(finish_reason="stop"),
        ]
    )
    transport = _Transport([ConnectionError("offline"), successful_stream])
    keys: list[str] = []
    model = DeepSeekConversationModel(
        lambda: keys.append("called") or "secret",
        _config(),
        lambda key, **kwargs: transport,
    )
    messages = (
        ProviderMessage(role="system", content="system"),
        ProviderMessage(role="user", content="question"),
        ProviderMessage(
            role="assistant",
            content="calling",
            tool_calls=(ToolCall("call-1", "read_file", '{"path":"README.md"}'),),
            reasoning_content="reason",
        ),
        ProviderMessage(
            role="tool", content="contents", tool_call_id="call-1"
        ),
    )
    tools = (
        ToolDefinition(
            name="read_file",
            description="Read one file",
            json_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
    )

    chunks = list(model.stream(messages, tools))

    assert chunks == [
        ReasoningDelta("private"),
        TextDelta("answer"),
        ResponseFinished("stop"),
    ]
    assert keys == ["called"]
    expected_call = {
        "model": "deepseek-v4-pro",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
                "reasoning_content": "reason",
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "contents"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read one file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            }
        ],
        "stream": True,
        "reasoning_effort": "max",
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    assert transport.chat.completions.calls == [expected_call, expected_call]


def test_deepseek_transport_factory_receives_zero_sdk_retries() -> None:
    transport = _Transport([iter([_chunk(finish_reason="stop")])])

    class RecordingFactory:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def __call__(self, api_key: str, **kwargs: object) -> _Transport:
            self.calls.append((api_key, kwargs))
            return transport

    factory = RecordingFactory()
    model = DeepSeekConversationModel(lambda: "secret", _config(), factory)

    assert list(model.stream((), ())) == [ResponseFinished("stop")]
    assert factory.calls == [("secret", {"max_retries": 0})]


def test_deepseek_maps_fragmented_streamed_tool_calls_to_deltas() -> None:
    first_fragment = _chunk(
        tool_calls=[
            SimpleNamespace(
                index=1,
                id="call-list",
                function=SimpleNamespace(name="list_files", arguments='{"path":"s'),
            ),
            SimpleNamespace(
                index=0,
                id="call-read",
                function=SimpleNamespace(name="read_file", arguments='{"path":"R'),
            ),
        ]
    )
    second_fragment = _chunk(
        tool_calls=[
            SimpleNamespace(
                index=1,
                id=None,
                function=SimpleNamespace(name=None, arguments='rc"}'),
            ),
            SimpleNamespace(
                index=0,
                id=None,
                function=SimpleNamespace(name=None, arguments='EADME.md"}'),
            ),
        ]
    )
    transport = _Transport(
        [iter([first_fragment, second_fragment, _chunk(finish_reason="tool_calls")])]
    )
    model = DeepSeekConversationModel(
        lambda: "secret", _config(), lambda key, **kwargs: transport
    )

    chunks = list(model.stream((), ()))

    assert chunks == [
        ToolCallDelta(1, "call-list", "list_files", '{"path":"s'),
        ToolCallDelta(0, "call-read", "read_file", '{"path":"R'),
        ToolCallDelta(1, None, None, 'rc"}'),
        ToolCallDelta(0, None, None, 'EADME.md"}'),
        ResponseFinished("tool_calls"),
    ]


class _DeltaThenFailure:
    def __iter__(self) -> Iterator[object]:
        yield _chunk(content="already visible")
        raise TimeoutError("late timeout")


def test_stream_failure_after_visible_delta_is_not_retried() -> None:
    transport = _Transport([_DeltaThenFailure()])
    model = DeepSeekConversationModel(
        lambda: "secret", _config(), lambda key, **kwargs: transport
    )
    agent = ConversationAgent(model)
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "hello")

    events = list(agent.run_turn(session_id, turn_id))

    assert len(transport.chat.completions.calls) == 1
    assert [event.kind for event in events] == [
        "turn_started",
        "assistant_item_started",
        "assistant_text_delta",
        "assistant_item_completed",
        "turn_failed",
    ]
    assert events[2].text == "already visible"
    assert events[-1].data == {"code": "provider_temporary_failure"}
    assert "late timeout" not in repr(events)
