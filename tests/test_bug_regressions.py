"""Small reproductions for the existing CLI and tool contracts."""

from io import StringIO
import json
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

import pytest

from conftest import safe_config
from guardedpy.conversation import (
    ConversationAgent, ResponseFinished, ScriptedConversationModel, TextDelta,
    ToolCall, ToolCallDelta, Turn,
)
from guardedpy.discovery import discover_project
from guardedpy.executor import ToolExecutor
from guardedpy.governor import ToolGovernor, governed_tool_definitions
from guardedpy.workspace import Workspace


def _call(name, **arguments):
    return ToolCall(str(uuid4()), name, json.dumps(arguments))


def _response(*calls):
    return [
        *(ToolCallDelta(index, call.id, call.name, call.arguments_json)
          for index, call in enumerate(calls)),
        ResponseFinished("tool_calls"),
    ]


def _agent(root, responses):
    config = safe_config(root)
    model = ScriptedConversationModel(responses)
    agent = ConversationAgent(model, governed_tool_definitions(config),
                              ToolGovernor(config), ToolExecutor(root, config))
    return agent, model


def _assert_paired(messages):
    pending = []
    for message in messages:
        if message.role == "tool":
            assert message.tool_call_id == pending.pop(0)
        else:
            assert not pending, "a new message precedes outstanding tool results"
            pending = [call.id for call in message.tool_calls]
    assert not pending


def test_paged_read_authorizes_patch_only_after_all_lines_are_read(tmp_path):
    executor = ToolExecutor(tmp_path, safe_config(tmp_path))
    target = tmp_path / "src/large.py"
    original = "value = 1\n" * 250
    target.write_text(original)
    turn = Turn(uuid4(), uuid4(), "repair", "normal")
    patch = _call("apply_patch", unified_diff=(
        "--- a/src/large.py\n+++ b/src/large.py\n@@ -250 +250 @@\n-value = 1\n+value = 2\n"
    ))
    executor.execute(turn, uuid4(), _call("read_file", path="src/large.py"))
    assert executor.execute(turn, uuid4(), patch).code == "read_required"
    executor.execute(turn, uuid4(), _call("read_file", path="src/large.py", offset=200))
    assert executor.execute(turn, uuid4(), patch).code == "ok"
    assert target.read_text() == "value = 1\n" * 249 + "value = 2\n"


def test_changed_file_cannot_combine_old_and_new_read_pages(tmp_path):
    executor = ToolExecutor(tmp_path, safe_config(tmp_path))
    target = tmp_path / "src/large.py"
    target.write_text("value = 1\n" * 250)
    turn = Turn(uuid4(), uuid4(), "repair", "normal")
    executor.execute(turn, uuid4(), _call("read_file", path="src/large.py"))
    target.write_text("value = 2\n" * 250)
    executor.execute(turn, uuid4(), _call("read_file", path="src/large.py", offset=200))
    result = executor.execute(turn, uuid4(), _call("apply_patch", unified_diff=(
        "--- a/src/large.py\n+++ b/src/large.py\n@@ -1 +1 @@\n-value = 2\n+value = 3\n"
    )))
    assert result.code == "read_required"


@pytest.mark.parametrize(("old_start", "expected"), [(1, "one\nadded\ntwo\n"), (2, "one\ntwo\nadded\n")])
def test_zero_context_insertion_uses_unified_diff_line_numbers(tmp_path, old_start, expected):
    workspace = Workspace(tmp_path, safe_config(tmp_path))
    target = tmp_path / "src/value.py"
    target.write_text("one\ntwo\n")
    result = workspace.apply_patch(
        f"--- a/src/value.py\n+++ b/src/value.py\n@@ -{old_start},0 +{old_start + 1} @@\n+added\n"
    )
    assert result.ok
    assert target.read_text() == expected


def test_patch_handles_standard_missing_final_newline_marker(tmp_path):
    workspace = Workspace(tmp_path, safe_config(tmp_path))
    target = tmp_path / "src/value.py"
    target.write_text("old")
    result = workspace.apply_patch(
        "--- a/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n"
        "-old\n\\ No newline at end of file\n+new\n\\ No newline at end of file\n"
    )
    assert result.ok
    assert target.read_bytes() == b"new"


@pytest.mark.parametrize("config", ["pytest.ini", "setup.cfg"])
def test_discovery_uses_pytest_configuration_instead_of_unrelated_pyproject(tmp_path, config):
    safe_config(tmp_path)
    (tmp_path / "checks").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "example"\n')
    section = "tool:pytest" if config == "setup.cfg" else "pytest"
    (tmp_path / config).write_text(f"[{section}]\ntestpaths = checks\n")
    assert discover_project(tmp_path).test_dirs == (PurePosixPath("checks"),)


def test_interrupting_approval_keeps_next_provider_request_valid(tmp_path):
    agent, model = _agent(tmp_path, [
        _response(_call("delete_path", path="src/value.py"), _call("git_status")),
        [TextDelta("next"), ResponseFinished("stop")],
    ])
    session = agent.create_session()
    turn, _ = agent.begin_turn(session, "delete")
    events = list(agent.run_turn(session, turn))
    assert events[-1].kind == "approval_requested"
    assert agent.interrupt(session, turn).kind == "turn_interrupted"
    next_turn, _ = agent.begin_turn(session, "continue")
    list(agent.run_turn(session, next_turn))
    _assert_paired(model.received_messages[-1])


def test_interrupt_before_tool_execution_prevents_remaining_writes(tmp_path):
    patch = "--- /dev/null\n+++ b/src/created.py\n@@ -0,0 +1 @@\n+created\n"
    agent, model = _agent(tmp_path, [
        _response(_call("apply_patch", unified_diff=patch)),
        [TextDelta("next"), ResponseFinished("stop")],
    ])
    session = agent.create_session()
    turn, _ = agent.begin_turn(session, "create")
    stream = agent.run_turn(session, turn)
    for event in stream:
        if event.kind == "tool_item_started":
            agent.interrupt(session, turn)
            break
    assert list(stream)[-1].kind == "turn_interrupted"
    assert not (tmp_path / "src/created.py").exists()
    next_turn, _ = agent.begin_turn(session, "continue")
    list(agent.run_turn(session, next_turn))
    _assert_paired(model.received_messages[-1])


def test_tool_failure_ends_turn_and_allows_follow_up(tmp_path, monkeypatch):
    agent, model = _agent(tmp_path, [
        _response(_call("read_file", path="src/value.py")),
        [TextDelta("next"), ResponseFinished("stop")],
    ])
    def denied(*args):
        raise PermissionError("unreadable")
    monkeypatch.setattr(agent._executor._workspace, "read_file", denied)
    session = agent.create_session()
    turn, _ = agent.begin_turn(session, "read")
    events = list(agent.run_turn(session, turn))
    assert events[-1].kind == "turn_failed"
    next_turn, _ = agent.begin_turn(session, "continue")
    assert list(agent.run_turn(session, next_turn))[-1].kind == "turn_completed"
    _assert_paired(model.received_messages[-1])


def test_plain_mode_reports_failed_turn_with_nonzero_status(tmp_path, monkeypatch):
    from guardedpy.conversations import ConversationStore
    from guardedpy.runtime import ConversationRuntime
    from guardedpy.terminal import run_plain_conversation
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    agent = ConversationAgent(ScriptedConversationModel([RuntimeError("provider failed")]))
    runtime = ConversationRuntime(agent, ConversationStore(tmp_path))
    output = StringIO()
    assert run_plain_conversation(runtime, "project", StringIO("hello\n"), output) == 1
    assert "本轮未完成" in output.getvalue()


def test_active_session_cannot_be_deleted_from_persistent_store(tmp_path, monkeypatch):
    from guardedpy.conversation import TurnNotActiveError
    from guardedpy.conversations import ConversationStore
    from guardedpy.runtime import ConversationRuntime
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    runtime = ConversationRuntime(ConversationAgent(ScriptedConversationModel([])), ConversationStore(tmp_path))
    first = runtime.create_session("project")
    active = runtime.create_session("project")
    runtime.begin_turn(active, "work")
    with pytest.raises(TurnNotActiveError):
        runtime.delete_session(active)
    assert {summary.id for summary in runtime.store.summaries()} == {first, active}


def test_model_settings_apply_to_next_turn_but_not_active_turn(tmp_path, monkeypatch):
    from guardedpy import cli
    from guardedpy.credentials import CredentialStatus
    from guardedpy.runtime import LocalRuntime, RuntimeServices
    from types import SimpleNamespace

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    profile = safe_config(tmp_path).profile
    local = LocalRuntime(RuntimeServices(SimpleNamespace(status=lambda: CredentialStatus(True))))
    local.setup(profile)
    requests = []

    class Model:
        def __init__(self, key, config, transport):
            self.config = config
            self.calls = 0

        def stream(self, messages, tools):
            requests.append((self.config.model, self.config.reasoning_effort))
            self.calls += 1
            if self.calls == 1:
                yield from _response(_call("list_files"))
            else:
                yield TextDelta("done")
                yield ResponseFinished("stop")

    monkeypatch.setattr(cli, "DeepSeekConversationModel", Model)
    monkeypatch.setattr(cli, "_system_keyring", lambda: None)
    runtime = cli.continuous_runtime(local, profile)
    session = runtime.create_session("project")
    turn, _ = runtime.begin_turn(session, "inspect")
    stream = runtime.run_turn(session, turn)
    for event in stream:
        if event.kind == "tool_item_completed":
            break
    local.update_future_defaults(model="deepseek-v4-pro", reasoning_effort="max")
    runtime.queue(session, "inspect again")
    list(stream)
    assert requests == [("deepseek-v4-flash", "high")] * 2 + [("deepseek-v4-pro", "max")] * 2


@pytest.mark.parametrize("cancel", [False, True])
def test_provider_stream_and_client_close_after_completion_or_cancellation(tmp_path, cancel):
    from types import SimpleNamespace
    from guardedpy.llm import DeepSeekConversationModel
    closed = []

    class Response:
        def __iter__(self):
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="hello"), finish_reason=None,
            )])
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(), finish_reason="stop",
            )])

        def close(self):
            closed.append("response")

    response = Response()
    transport = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)),
        close=lambda: closed.append("transport"),
    )
    model = DeepSeekConversationModel(lambda: "key", safe_config(tmp_path), lambda *args, **kwargs: transport)
    stream = model.stream((), ())
    assert next(stream) == TextDelta("hello")
    if cancel:
        stream.close()
    else:
        list(stream)
    assert closed == ["response", "transport"]


def test_initial_plain_request_runs_before_reading_more_stdin(tmp_path, monkeypatch):
    from guardedpy.conversations import ConversationStore
    from guardedpy.runtime import ConversationRuntime
    from guardedpy.terminal import run_plain_conversation
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    model = ScriptedConversationModel([[TextDelta("done"), ResponseFinished("stop")]])
    runtime = ConversationRuntime(ConversationAgent(model), ConversationStore(tmp_path))

    class Input:
        def __iter__(self):
            assert model.received_messages, "initial request was blocked waiting for stdin EOF"
            yield "/exit"

    assert run_plain_conversation(runtime, "project", Input(), StringIO(), "initial") == 0


def test_tui_waits_for_interrupted_turn_before_creating_session(tmp_path, monkeypatch):
    import asyncio
    from threading import Event
    from guardedpy.conversations import ConversationStore
    from guardedpy.runtime import ConversationRuntime
    from guardedpy.tui import GuardedPyApp, ConversationScreen
    from test_tui import _Runtime, _profile

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    profile = _profile(tmp_path)
    release = Event()

    class Model:
        def stream(self, messages, tools):
            yield TextDelta("working")
            assert release.wait(5)
            yield ResponseFinished("stop")

    runtime = ConversationRuntime(ConversationAgent(Model()), ConversationStore(tmp_path))
    app = GuardedPyApp(_Runtime(profile), profile, conversation=runtime)

    async def check():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.submit("work")
            await pilot.pause()
            original = app._continuous_session_id
            app.submit("/conversations")
            await pilot.pause()
            assert not isinstance(app.screen, ConversationScreen)
            app._new_conversation_resolved(True)
            assert app._continuous_session_id == original
            release.set()
            await pilot.pause()
            assert app._continuous_session_id != original
            assert app._continuous_turn_id is None

    try:
        asyncio.run(check())
    finally:
        release.set()


def test_tui_ignores_events_from_another_session(tmp_path):
    import asyncio
    from guardedpy.conversation import SessionEvent
    from guardedpy.tui import GuardedPyApp, SessionEventReceived
    from test_tui import _Runtime, _profile, _transcript_lines
    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check():
        async with app.run_test():
            app._continuous_session_id = uuid4()
            app.on_session_event_received(SessionEventReceived(SessionEvent(
                uuid4(), uuid4(), 1, "assistant_text_delta", uuid4(), "old session text",
            )))
            assert not _transcript_lines(app)

    asyncio.run(check())


def test_real_collection_error_retains_file_and_exception(tmp_path):
    from guardedpy.feedback import FeedbackCollector, FeedbackKind
    workspace = Workspace(tmp_path, safe_config(tmp_path))
    (tmp_path / "tests/test_invalid.py").write_text("def test_invalid(:\n    pass\n")
    result = FeedbackCollector().collect(workspace.run_pytest(()))
    assert result.kind == FeedbackKind.COLLECTION_ERROR
    assert result.node_ids == ("tests/test_invalid.py",)
    assert "SyntaxError" in result.excerpt


def test_failed_parameter_id_with_spaces_is_preserved():
    from guardedpy.feedback import FeedbackCollector, PytestRun
    result = FeedbackCollector().collect(PytestRun(
        1, "FAILED tests/test_value.py::test_value[hello world] - AssertionError\n", "", False,
    ))
    assert result.node_ids == ("tests/test_value.py::test_value[hello world]",)


def test_patch_header_normalization_cannot_bypass_required_read(tmp_path):
    executor = ToolExecutor(tmp_path, safe_config(tmp_path))
    target = tmp_path / "src/value.py"
    target.write_text("old\n")
    turn = Turn(uuid4(), uuid4(), "repair", "normal")
    call = _call("apply_patch", unified_diff=(
        "--- b/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n-old\n+new\n"
    ))
    assert executor.execute(turn, uuid4(), call).code == "read_required"
    assert target.read_text() == "old\n"
    executor.execute(turn, uuid4(), _call("read_file", path="src/value.py"))
    assert executor.execute(turn, uuid4(), call).code == "ok"


def test_empty_patch_does_not_claim_a_mutation(tmp_path):
    executor = ToolExecutor(tmp_path, safe_config(tmp_path))
    turn = Turn(uuid4(), uuid4(), "repair", "normal")
    result = executor.execute(turn, uuid4(), _call("apply_patch", unified_diff=""))
    assert result.code == "patch_invalid"
    assert not turn.needs_full_verification


def test_plain_stream_joins_text_deltas(tmp_path, monkeypatch):
    from guardedpy.conversations import ConversationStore
    from guardedpy.runtime import ConversationRuntime
    from guardedpy.terminal import run_plain_conversation
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    model = ScriptedConversationModel([[TextDelta("hel"), TextDelta("lo"), ResponseFinished("stop")]])
    runtime = ConversationRuntime(ConversationAgent(model), ConversationStore(tmp_path))
    output = StringIO()
    assert run_plain_conversation(runtime, "project", StringIO("hi\n"), output) == 0
    assert output.getvalue() == "› hi\nhello\n"
