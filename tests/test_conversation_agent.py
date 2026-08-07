"""Contract tests for the continuous in-memory conversation protocol."""

from __future__ import annotations

from uuid import uuid4

import pytest

from guardedpy.conversation import (
    ConversationAgent,
    ProviderMessage,
    ReasoningDelta,
    ResponseFinished,
    ScriptedConversationModel,
    TemporaryProviderFailure,
    TextDelta,
    ToolCallDelta,
    TurnNotActiveError,
)


def test_normal_chat_returns_immediate_user_event_then_text_deltas_and_terminal() -> None:
    model = ScriptedConversationModel(
        [[TextDelta("Hello, "), TextDelta("world."), ResponseFinished("stop")]]
    )
    agent = ConversationAgent(model)
    session_id = agent.create_session()

    turn_id, user_event = agent.begin_turn(session_id, "  hello  ")
    later_events = list(agent.run_turn(session_id, turn_id))

    assert user_event.session_id == session_id
    assert user_event.turn_id == turn_id
    assert user_event.sequence == 1
    assert user_event.kind == "user_message"
    assert user_event.item_id is not None
    assert user_event.text == "hello"
    assert [event.kind for event in later_events] == [
        "turn_started",
        "assistant_item_started",
        "assistant_text_delta",
        "assistant_text_delta",
        "assistant_item_completed",
        "turn_completed",
    ]
    assert [event.sequence for event in later_events] == [2, 3, 4, 5, 6, 7]
    assert [event.text for event in later_events if event.kind == "assistant_text_delta"] == [
        "Hello, ",
        "world.",
    ]
    assert later_events[0].item_id is None
    assert later_events[-1].item_id is None
    assert model.received_messages == [
        (ProviderMessage(role="user", content="hello"),)
    ]


def test_assistant_item_has_started_delta_completed_lifecycle() -> None:
    model = ScriptedConversationModel([[ResponseFinished("stop")]])
    agent = ConversationAgent(model)
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "Say nothing")

    events = list(agent.run_turn(session_id, turn_id))

    assert [event.kind for event in events] == [
        "turn_started",
        "assistant_item_started",
        "assistant_item_completed",
        "turn_completed",
    ]
    assistant_events = events[1:3]
    assert assistant_events[0].item_id is not None
    assert {event.item_id for event in assistant_events} == {
        assistant_events[0].item_id
    }


def test_reasoning_is_retained_only_in_provider_message() -> None:
    model = ScriptedConversationModel(
        [
            [
                ReasoningDelta("private "),
                ReasoningDelta("thought"),
                TextDelta("Visible"),
                ResponseFinished("stop"),
            ],
            [TextDelta("Again"), ResponseFinished("stop")],
        ]
    )
    agent = ConversationAgent(model)
    session_id = agent.create_session()
    first_turn_id, _ = agent.begin_turn(session_id, "First")

    first_events = list(agent.run_turn(session_id, first_turn_id))
    second_turn_id, _ = agent.begin_turn(session_id, "Second")
    second_events = list(agent.run_turn(session_id, second_turn_id))

    assert [
        event.text
        for event in first_events + second_events
        if event.kind == "assistant_text_delta"
    ] == ["Visible", "Again"]
    assert "private" not in repr(first_events + second_events)
    prior_assistant = model.received_messages[1][1]
    assert prior_assistant == ProviderMessage(
        role="assistant",
        content="Visible",
        reasoning_content="private thought",
    )


def test_tool_call_fragments_are_joined_but_unavailable_runner_has_zero_side_effects() -> None:
    model = ScriptedConversationModel(
        [
            [
                ToolCallDelta(0, "call-1", "read_file", '{"pa'),
                ToolCallDelta(0, None, None, 'th":"README.md"}'),
                ResponseFinished("tool_calls"),
            ],
            [TextDelta("queued answer"), ResponseFinished("stop")],
        ]
    )

    class Executor:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def execute(self, *args: object) -> None:
            self.calls.append(args)

    executor = Executor()
    agent = ConversationAgent(model, governor=None, executor=executor)
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "Read it")
    queued_turn_id, _ = agent.queue(session_id, "Continue")

    events = list(agent.run_turn(session_id, turn_id))

    first_terminal = next(event for event in events if event.turn_id == turn_id and event.kind == "turn_failed")
    assert first_terminal.data == {"code": "tool_execution_unavailable"}
    assert executor.calls == []
    assert any(
        event.turn_id == queued_turn_id and event.kind == "turn_completed"
        for event in events
    )
    joined_assistant = model.received_messages[1][1]
    assert joined_assistant.tool_calls[0].id == "call-1"
    assert joined_assistant.tool_calls[0].name == "read_file"
    assert joined_assistant.tool_calls[0].arguments_json == '{"path":"README.md"}'


def test_tool_call_budget_fails_before_accepting_a_fifty_first_call() -> None:
    chunks = [
        ToolCallDelta(index, f"call-{index}", "read_file", "{}")
        for index in range(51)
    ]
    model = ScriptedConversationModel(
        [[*chunks, ResponseFinished("tool_calls")]]
    )
    agent = ConversationAgent(model)
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "Too many calls")

    events = list(agent.run_turn(session_id, turn_id))

    assert events[-1].kind == "turn_failed"
    assert events[-1].data == {"code": "round_limit"}


def test_scripted_model_failure_is_a_provider_protocol_error() -> None:
    model = ScriptedConversationModel(
        [TemporaryProviderFailure("scripted transport-looking failure")]
    )
    agent = ConversationAgent(model)
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "Run script")

    events = list(agent.run_turn(session_id, turn_id))

    assert events[-1].kind == "turn_failed"
    assert events[-1].data == {"code": "provider_protocol_error"}
    assert "scripted transport-looking failure" not in repr(events)


def test_interrupt_closes_a_mid_stream_iterator_before_terminal_event() -> None:
    class CloseAwareModel:
        def __init__(self) -> None:
            self.closed = False

        def stream(self, messages: object, tools: object) -> object:
            del messages, tools
            try:
                yield TextDelta("visible")
                yield TextDelta("unreached")
            finally:
                self.closed = True

    model = CloseAwareModel()
    agent = ConversationAgent(model)
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "Interrupt the stream")
    events = iter(agent.run_turn(session_id, turn_id))

    assert next(events).kind == "turn_started"
    assert next(events).kind == "assistant_item_started"
    assert next(events).text == "visible"
    assert agent.interrupt(session_id, turn_id) is None
    assert next(events).kind == "assistant_item_completed"
    assert next(events).kind == "turn_interrupted"
    assert model.closed is True


def test_steer_queue_and_interrupt_preserve_single_active_turn_ownership() -> None:
    model = ScriptedConversationModel(
        [
            [TextDelta("first"), ResponseFinished("stop")],
            [TextDelta("queued"), ResponseFinished("stop")],
        ]
    )
    agent = ConversationAgent(model)
    session_id = agent.create_session()
    first_turn_id, _ = agent.begin_turn(session_id, "First")

    steer_event = agent.steer(session_id, first_turn_id, "More context")
    queued_turn_id, queued_event = agent.queue(session_id, "Next", mode="review")
    with pytest.raises(TurnNotActiveError):
        agent.begin_turn(session_id, "Cannot overlap")

    events = list(agent.run_turn(session_id, first_turn_id))

    assert steer_event.kind == "user_message"
    assert steer_event.sequence == 2
    assert queued_event.kind == "user_message"
    assert queued_event.sequence == 1
    assert queued_event.data == {"queued": "true"}
    assert model.received_messages[0] == (
        ProviderMessage(role="user", content="First"),
        ProviderMessage(role="user", content="More context"),
    )
    assert model.received_messages[1][-1] == ProviderMessage(
        role="user", content="Next"
    )
    assert [
        event.turn_id
        for event in events
        if event.kind == "turn_started"
    ] == [first_turn_id, queued_turn_id]

    interrupted_turn_id, _ = agent.begin_turn(session_id, "Stop before the model")
    assert agent.interrupt(session_id, interrupted_turn_id) is None
    interrupted_events = list(agent.run_turn(session_id, interrupted_turn_id))

    assert [event.kind for event in interrupted_events] == [
        "turn_started",
        "turn_interrupted",
    ]
    assert len(model.received_messages) == 2
    with pytest.raises(TurnNotActiveError):
        agent.interrupt(session_id, uuid4())
    with pytest.raises(TurnNotActiveError):
        agent.queue(session_id, "No active turn")
