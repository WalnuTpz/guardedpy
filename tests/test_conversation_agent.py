"""Contract tests for the continuous in-memory conversation protocol."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from conftest import safe_config
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
from guardedpy.executor import ToolExecutor
from guardedpy.governor import ToolGovernor, governed_tool_definitions


def _tool_response(*calls: tuple[str, str, dict[str, object]]) -> list[object]:
    return [
        *[
            ToolCallDelta(index, call_id, name, json.dumps(arguments))
            for index, (call_id, name, arguments) in enumerate(calls)
        ],
        ResponseFinished("tool_calls"),
    ]


def _governed_agent(tmp_path: Path, responses: list[list[object]]) -> tuple[ConversationAgent, ScriptedConversationModel]:
    config = safe_config(tmp_path)
    model = ScriptedConversationModel(responses)
    return (
        ConversationAgent(
            model,
            governed_tool_definitions(config),
            ToolGovernor(config),
            ToolExecutor(tmp_path, config),
        ),
        model,
    )


def _patch(old: str, new: str) -> str:
    return f"--- a/src/calc.py\n+++ b/src/calc.py\n@@ -1 +1 @@\n-{old}\n+{new}\n"


def _prepare_project(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()


@pytest.mark.parametrize("mode", ("plan", "review"))
def test_read_only_mode_denies_mutation_with_zero_side_effect(
    tmp_path: Path, mode: str
) -> None:
    _prepare_project(tmp_path)
    target = tmp_path / "src" / "calc.py"
    target.write_text("VALUE = 1\n")
    agent, model = _governed_agent(
        tmp_path,
        [
            _tool_response(
                (
                    "forbidden-patch",
                    "apply_patch",
                    {"unified_diff": _patch("VALUE = 1", "VALUE = 2")},
                )
            ),
            [TextDelta("No mutation was performed."), ResponseFinished("stop")],
        ],
    )
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "Inspect only", mode=mode)

    events = list(agent.run_turn(session_id, turn_id))

    tool_result = json.loads(model.received_messages[1][-1].content)
    assert tool_result["code"] == "mode_read_only"
    assert target.read_text() == "VALUE = 1\n"
    assert events[-1].kind == "turn_completed"


def test_repair_turn_receives_assertion_feedback_then_patches_and_full_retests(tmp_path: Path) -> None:
    _prepare_project(tmp_path)
    (tmp_path / "src" / "calc.py").write_text("VALUE = 1\n")
    (tmp_path / "tests" / "test_calc.py").write_text("from src.calc import VALUE\n\ndef test_value():\n    assert VALUE == 20\n")
    agent, model = _governed_agent(tmp_path, [
        _tool_response(("test-red", "run_pytest", {})),
        _tool_response(("read", "read_file", {"path": "src/calc.py"})),
        _tool_response(("fix", "apply_patch", {"unified_diff": _patch("VALUE = 1", "VALUE = 20")})),
        _tool_response(("test-green", "run_pytest", {})),
        [TextDelta("Fixed and verified."), ResponseFinished("stop")],
    ])
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "Repair the failing test")

    events = list(agent.run_turn(session_id, turn_id))

    first_result = json.loads(model.received_messages[1][-1].content)
    assert first_result["feedback"]["kind"] == "assertion_failure"
    assert first_result["feedback"]["node_ids"] == ["tests/test_calc.py::test_value"]
    patch_event = next(
        event for event in events
        if event.kind == "tool_item_completed" and event.data.get("changed_paths")
    )
    assert patch_event.data["tool"] == "apply_patch"
    read_event = next(
        event for event in events
        if event.kind == "tool_item_started" and event.data.get("tool") == "read_file"
    )
    assert read_event.data["path"] == "src/calc.py"
    assert (tmp_path / "src" / "calc.py").read_text() == "VALUE = 20\n"
    assert (events[-1].kind, events[-1].data) == ("turn_completed", {})


def test_invalid_patch_returns_tool_result_then_same_turn_corrects_it(tmp_path: Path) -> None:
    _prepare_project(tmp_path)
    (tmp_path / "src" / "calc.py").write_text("VALUE = 1\n")
    (tmp_path / "tests" / "test_calc.py").write_text("def test_ok(): assert True\n")
    agent, model = _governed_agent(tmp_path, [
        _tool_response(("read", "read_file", {"path": "src/calc.py"})),
        _tool_response(("bad", "apply_patch", {"unified_diff": "not a diff"})),
        _tool_response(("fix", "apply_patch", {"unified_diff": _patch("VALUE = 1", "VALUE = 2")})),
        _tool_response(("full", "run_pytest", {})),
        [ResponseFinished("stop")],
    ])
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "Fix")

    events = list(agent.run_turn(session_id, turn_id))

    invalid_result = json.loads(model.received_messages[2][-1].content)
    assert invalid_result["code"] == "patch_invalid"
    assert (tmp_path / "src" / "calc.py").read_text() == "VALUE = 2\n"
    assert events[-1].kind == "turn_completed"


@pytest.mark.parametrize("stale", (False, True), ids=("unread", "stale"))
def test_unread_or_stale_file_patch_is_denied_without_write(tmp_path: Path, stale: bool) -> None:
    _prepare_project(tmp_path)
    target = tmp_path / "src" / "calc.py"
    target.write_text("VALUE = 1\n")
    responses = []
    if stale:
        responses.append(_tool_response(("read", "read_file", {"path": "src/calc.py"})))
    responses.extend([
        _tool_response(("patch", "apply_patch", {"unified_diff": _patch("VALUE = 1", "VALUE = 2")})),
        [ResponseFinished("stop")],
    ])
    agent, model = _governed_agent(tmp_path, responses)
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "Patch")
    iterator = iter(agent.run_turn(session_id, turn_id))
    if stale:
        for event in iterator:
            if event.kind == "tool_item_completed":
                break
        target.write_text("VALUE = 9\n")

    events = list(iterator)

    result = json.loads(model.received_messages[-1][-1].content)
    assert result["code"] == ("stale_read" if stale else "read_required")
    if not stale:
        assert result["missing_paths"] == ["src/calc.py"]
        patch_event = next(
            event for event in events
            if event.kind == "tool_item_completed" and event.data.get("tool") == "apply_patch"
        )
        assert json.loads(patch_event.data["missing_paths"]) == ["src/calc.py"]
    assert target.read_text() == ("VALUE = 9\n" if stale else "VALUE = 1\n")
    assert events[-1].kind == "turn_completed"


def test_delete_rejection_preserves_target_and_approval_acceptance_continues_same_turn(tmp_path: Path) -> None:
    _prepare_project(tmp_path)
    target = tmp_path / "src" / "obsolete.py"
    target.write_text("obsolete\n")
    agent, model = _governed_agent(tmp_path, [
        _tool_response(("delete-1", "delete_path", {"path": "src/obsolete.py"})),
        [TextDelta("Kept it."), ResponseFinished("stop")],
        _tool_response(("delete-2", "delete_path", {"path": "src/obsolete.py"})),
        _tool_response(("full", "run_pytest", {})),
        [TextDelta("Deleted it."), ResponseFinished("stop")],
    ])
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok(): assert True\n")
    session_id = agent.create_session()
    rejected_turn, _ = agent.begin_turn(session_id, "Maybe delete")
    paused = list(agent.run_turn(session_id, rejected_turn))
    approval_id = next(UUID(event.data["approval_id"]) for event in paused if event.kind == "approval_requested")
    rejected = list(agent.resolve_approval(session_id, rejected_turn, approval_id, False))
    assert target.exists()
    assert rejected[-1].kind == "turn_completed"

    accepted_turn, _ = agent.begin_turn(session_id, "Delete")
    paused = list(agent.run_turn(session_id, accepted_turn))
    approval_id = next(UUID(event.data["approval_id"]) for event in paused if event.kind == "approval_requested")
    accepted = list(agent.resolve_approval(session_id, accepted_turn, approval_id, True))
    assert not target.exists()
    assert accepted[-1].kind == "turn_completed"
    assert json.loads(model.received_messages[-1][-1].content)["feedback"]["kind"] == "passed"


def test_running_a_project_python_program_requires_exact_approval(tmp_path: Path) -> None:
    _prepare_project(tmp_path)
    marker = tmp_path / "executed.txt"
    (tmp_path / "src" / "hello.py").write_text(
        "from pathlib import Path\nPath('executed.txt').write_text('yes')\n"
    )
    agent, _ = _governed_agent(tmp_path, [
        _tool_response(("run", "run_python", {"path": "src/hello.py"})),
        [TextDelta("程序已运行。"), ResponseFinished("stop")],
    ])
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "运行 src/hello.py")

    paused = list(agent.run_turn(session_id, turn_id))
    approval_event = next(event for event in paused if event.kind == "approval_requested")
    assert approval_event.data["path"] == "src/hello.py"
    assert approval_event.data["argv"] == "[]"
    approval_id = UUID(approval_event.data["approval_id"])
    assert not marker.exists()

    completed = list(agent.resolve_approval(session_id, turn_id, approval_id, True))
    assert marker.read_text() == "yes"
    assert completed[-1].kind == "turn_completed"


def test_stale_or_forged_approval_id_cannot_execute_delete(tmp_path: Path) -> None:
    _prepare_project(tmp_path)
    target = tmp_path / "src" / "obsolete.py"
    target.write_text("obsolete\n")
    agent, _ = _governed_agent(tmp_path, [_tool_response(("delete", "delete_path", {"path": "src/obsolete.py"}))])
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "Delete")
    paused = list(agent.run_turn(session_id, turn_id))
    real_id = UUID(next(event.data["approval_id"] for event in paused if event.kind == "approval_requested"))
    with pytest.raises(TurnNotActiveError):
        list(agent.resolve_approval(session_id, turn_id, uuid4(), True))
    assert target.exists()
    agent.interrupt(session_id, turn_id)
    with pytest.raises(TurnNotActiveError):
        list(agent.resolve_approval(session_id, turn_id, real_id, True))
    assert target.exists()


def test_multiple_tool_calls_pause_at_first_approval_and_pair_later_calls_without_execution(tmp_path: Path) -> None:
    _prepare_project(tmp_path)
    target = tmp_path / "src" / "obsolete.py"
    target.write_text("obsolete\n")
    agent, model = _governed_agent(tmp_path, [
        _tool_response(
            ("delete", "delete_path", {"path": "src/obsolete.py"}),
            ("read-later", "read_file", {"path": "src/obsolete.py"}),
        ),
        [ResponseFinished("stop")],
    ])
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "Delete then read")
    paused = list(agent.run_turn(session_id, turn_id))
    approval_id = UUID(next(event.data["approval_id"] for event in paused if event.kind == "approval_requested"))

    events = list(agent.resolve_approval(session_id, turn_id, approval_id, False))

    results = [json.loads(message.content) for message in model.received_messages[1] if message.role == "tool"]
    assert [result["code"] for result in results] == ["approval_rejected", "not_executed_after_approval"]
    assert target.exists()
    assert events[-1].kind == "turn_completed"


def test_duplicate_tool_call_id_fails_before_tool_io(tmp_path: Path) -> None:
    _prepare_project(tmp_path)
    target = tmp_path / "src" / "calc.py"
    target.write_text("VALUE = 1\n")
    agent, model = _governed_agent(tmp_path, [
        _tool_response(
            ("same", "read_file", {"path": "src/calc.py"}),
            ("same", "delete_path", {"path": "src/calc.py"}),
        )
    ])
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "Bad provider response")

    events = list(agent.run_turn(session_id, turn_id))

    assert events[-1].kind == "turn_failed"
    assert events[-1].data == {"code": "provider_protocol_error"}
    assert target.exists()
    assert len(model.received_messages) == 1


def test_mutation_cannot_complete_before_full_pytest_passes(tmp_path: Path) -> None:
    _prepare_project(tmp_path)
    target = tmp_path / "src" / "calc.py"
    target.write_text("VALUE = 1\n")
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok(): assert True\n")
    agent, _ = _governed_agent(tmp_path, [
        _tool_response(("read", "read_file", {"path": "src/calc.py"})),
        _tool_response(("patch", "apply_patch", {"unified_diff": _patch("VALUE = 1", "VALUE = 2")})),
        _tool_response(("targeted", "run_pytest", {"nodes": ["tests/test_ok.py"]})),
        [ResponseFinished("stop")],
    ])
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "Patch")

    events = list(agent.run_turn(session_id, turn_id))

    assert target.read_text() == "VALUE = 2\n"
    assert events[-1].kind == "turn_failed"
    assert events[-1].data == {"code": "verification_required"}


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
    assert model.received_messages[0][-1] == ProviderMessage(role="user", content="hello")


def test_session_context_tells_the_model_to_converse_and_use_governed_tools() -> None:
    model = ScriptedConversationModel([[ResponseFinished("stop")]])
    agent = ConversationAgent(model)
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "hello")

    list(agent.run_turn(session_id, turn_id))

    context = model.received_messages[0]
    assert context[0].role == "system"
    assert "interactive coding agent" in context[0].content
    assert "ordinary conversation" in context[0].content
    assert context[-1] == ProviderMessage(role="user", content="hello")


def test_one_session_can_chat_repair_and_answer_a_grounded_follow_up(tmp_path: Path) -> None:
    _prepare_project(tmp_path)
    (tmp_path / "src" / "calc.py").write_text("VALUE = 1\n")
    (tmp_path / "tests" / "test_calc.py").write_text(
        "from src.calc import VALUE\n\ndef test_value():\n    assert VALUE == 20\n"
    )
    agent, model = _governed_agent(tmp_path, [
        [TextDelta("你好，我可以帮你检查并修改这个项目。"), ResponseFinished("stop")],
        _tool_response(("baseline", "run_pytest", {})),
        _tool_response(("read", "read_file", {"path": "src/calc.py"})),
        _tool_response(("patch", "apply_patch", {"unified_diff": _patch("VALUE = 1", "VALUE = 20")})),
        _tool_response(("verify", "run_pytest", {})),
        [TextDelta("已修复并完成完整 pytest 验证。"), ResponseFinished("stop")],
        [TextDelta("错误是 VALUE 的值错误；我已将它改为 20，pytest 已通过。"), ResponseFinished("stop")],
    ])
    session_id = agent.create_session()

    greeting_turn, _ = agent.begin_turn(session_id, "你好")
    assert [event.kind for event in agent.run_turn(session_id, greeting_turn)][-1] == "turn_completed"

    repair_turn, _ = agent.begin_turn(session_id, "现在仓库里存在若干错误，请帮我全部找出并修复。")
    repair_events = list(agent.run_turn(session_id, repair_turn))
    assert repair_events[-1].kind == "turn_completed"
    assert (tmp_path / "src" / "calc.py").read_text() == "VALUE = 20\n"

    follow_up_turn, _ = agent.begin_turn(session_id, "解释一下你找到了哪些错误")
    follow_up_events = list(agent.run_turn(session_id, follow_up_turn))

    assert follow_up_events[-1].kind == "turn_completed"
    assert "错误是 VALUE" in "".join(
        event.text for event in follow_up_events if event.kind == "assistant_text_delta"
    )
    follow_up_context = model.received_messages[-1]
    assert ProviderMessage(role="user", content="现在仓库里存在若干错误，请帮我全部找出并修复。") in follow_up_context
    assert any(message.role == "tool" and "assertion_failure" in message.content for message in follow_up_context)
    assert any(message.role == "tool" and '"passed"' in message.content for message in follow_up_context)


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
    prior_assistant = next(
        message for message in model.received_messages[1]
        if message.role == "assistant"
    )
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
    joined_assistant = next(
        message for message in model.received_messages[1]
        if message.role == "assistant"
    )
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


def test_steer_queue_and_interrupt_have_single_active_turn_semantics() -> None:
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
    assert model.received_messages[0][-2:] == (
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


def test_steer_arriving_during_a_final_stream_gets_its_own_model_response() -> None:
    class Model:
        def __init__(self) -> None:
            self.messages: list[tuple[ProviderMessage, ...]] = []
            self.agent: ConversationAgent | None = None
            self.session_id = uuid4()
            self.turn_id = uuid4()

        def stream(self, messages: tuple[ProviderMessage, ...], tools: object) -> object:
            del tools
            self.messages.append(messages)
            if len(self.messages) == 1:
                yield TextDelta("先回答这一部分。")
                assert self.agent is not None
                self.agent.steer(self.session_id, self.turn_id, "再补充测试结果")
                yield ResponseFinished("stop")
                return
            yield TextDelta("补充：测试尚未运行。")
            yield ResponseFinished("stop")

    model = Model()
    agent = ConversationAgent(model)
    model.agent = agent
    model.session_id = agent.create_session()
    model.turn_id, _ = agent.begin_turn(model.session_id, "说明当前状态")

    events = list(agent.run_turn(model.session_id, model.turn_id))

    assert [event.text for event in events if event.kind == "assistant_text_delta"] == [
        "先回答这一部分。", "补充：测试尚未运行。"
    ]
    assert model.messages[1][-1] == ProviderMessage(role="user", content="再补充测试结果")
    assert events[-1].kind == "turn_completed"


def test_interrupt_clears_queued_turns_instead_of_promoting_them() -> None:
    agent = ConversationAgent(ScriptedConversationModel([]))
    session_id = agent.create_session()
    active_turn, _ = agent.begin_turn(session_id, "先处理这个")
    queued_turn, _ = agent.queue(session_id, "之后不要运行这个")

    assert agent.interrupt(session_id, active_turn) is None
    events = list(agent.run_turn(session_id, active_turn))

    assert [event.turn_id for event in events if event.kind == "turn_started"] == [active_turn]
    assert all(event.turn_id != queued_turn for event in events)
    fresh_turn, _ = agent.begin_turn(session_id, "现在可以开始新的回合")
    assert fresh_turn != queued_turn


def test_next_turn_goal_is_visible_to_the_model_once_without_becoming_history() -> None:
    model = ScriptedConversationModel(
        [[TextDelta("first"), ResponseFinished("stop")], [TextDelta("second"), ResponseFinished("stop")]]
    )
    agent = ConversationAgent(model)
    session_id = agent.create_session()

    first_turn, _ = agent.begin_turn(session_id, "inspect", goal="Keep the repair minimal")
    list(agent.run_turn(session_id, first_turn))
    second_turn, _ = agent.begin_turn(session_id, "continue")
    list(agent.run_turn(session_id, second_turn))

    assert ProviderMessage(
        role="system", content="Current turn goal: Keep the repair minimal"
    ) in model.received_messages[0]
    assert all(message.content != "Current turn goal: Keep the repair minimal" for message in model.received_messages[1])


def test_new_source_file_needs_no_read_and_its_contract_names_discovered_directories(tmp_path: Path) -> None:
    _prepare_project(tmp_path)
    (tmp_path / "tests" / "test_hello.py").write_text(
        "from src.hello import message\n\ndef test_message(): assert message() == 'hello world'\n"
    )
    patch = (
        "--- /dev/null\n+++ b/src/hello.py\n@@ -0,0 +1,2 @@\n"
        "+def message() -> str:\n+    return 'hello world'\n"
    )
    agent, model = _governed_agent(tmp_path, [
        _tool_response(("create", "apply_patch", {"unified_diff": patch})),
        _tool_response(("verify", "run_pytest", {})),
        [TextDelta("Created and verified."), ResponseFinished("stop")],
    ])
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "Create a hello world program")

    events = list(agent.run_turn(session_id, turn_id))

    assert (tmp_path / "src" / "hello.py").read_text() == "def message() -> str:\n    return 'hello world'\n"
    assert not any(event.data.get("tool") == "read_file" for event in events)
    patch_tool = next(tool for tool in model.received_tools[0] if tool.name == "apply_patch")
    assert "src, tests" in patch_tool.description
    assert "--- /dev/null" in patch_tool.description
    assert "do not read" in patch_tool.description
    assert "ask a concise clarification" in model.received_messages[0][0].content
    assert events[-1].kind == "turn_completed"


def test_new_file_outside_discovered_directories_returns_actionable_feedback(tmp_path: Path) -> None:
    _prepare_project(tmp_path)
    patch = "--- /dev/null\n+++ b/hello.py\n@@ -0,0 +1 @@\n+print('hello world')\n"
    agent, model = _governed_agent(tmp_path, [
        _tool_response(("create", "apply_patch", {"unified_diff": patch})),
        [TextDelta("I will choose an allowed source path."), ResponseFinished("stop")],
    ])
    session_id = agent.create_session()
    turn_id, _ = agent.begin_turn(session_id, "Create a program")

    events = list(agent.run_turn(session_id, turn_id))

    result = json.loads(model.received_messages[1][-1].content)
    assert result["code"] == "new_file_path_not_allowed"
    assert result["allowed_directories"] == ["src", "tests"]
    assert not (tmp_path / "hello.py").exists()
    assert next(event for event in events if event.kind == "tool_item_completed").data["code"] == "new_file_path_not_allowed"
