"""Task 22.3 safe-summary and continuous-runtime contracts."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from guardedpy.conversation import (
    ConversationAgent,
    ConversationSummary,
    ProviderMessage,
    ResponseFinished,
    SafeTurnSummary,
    ScriptedConversationModel,
    TextDelta,
    ToolCallDelta,
    VisibleTranscriptEntry,
)
from guardedpy.conversations import ConversationStore
from guardedpy.runtime import ConversationRuntime
from guardedpy.executor import ToolExecutor
from guardedpy.governor import ToolGovernor, governed_tool_definitions
from conftest import safe_config
import json


def test_summary_store_round_trips_only_the_safe_summary(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))  # type: ignore[attr-defined]
    now = datetime.now(timezone.utc)
    summary = ConversationSummary(
        id=uuid4(), project_title="calculator", created_at=now, updated_at=now, turns=()
    )

    store = ConversationStore(tmp_path / "project")
    store.save_summary(summary)

    assert ConversationStore(tmp_path / "project").load_summary(summary.id) == summary


def test_runtime_persists_visible_transcript_and_resumes_the_selected_session(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))  # type: ignore[attr-defined]
    model = ScriptedConversationModel([[TextDelta("可见答复"), ResponseFinished("stop")]])
    runtime = ConversationRuntime(ConversationAgent(model), ConversationStore(tmp_path))
    session_id = runtime.create_session("demo")
    turn_id, immediate = runtime.begin_turn(session_id, "private user text API_KEY=secret")

    events = tuple(runtime.run_turn(session_id, turn_id))
    summary = runtime.summary(session_id)
    restored = ConversationRuntime(ConversationAgent(ScriptedConversationModel([])), ConversationStore(tmp_path))

    assert immediate.kind == "user_message"
    assert events[-1].kind == "turn_completed"
    assert summary.turns[0].terminal_status == "completed"
    assert summary.turns[0].final_text == "本轮已完成。"
    assert summary.transcript == (
        VisibleTranscriptEntry(role="user", text="private user text API_KEY=[已隐藏]"),
        VisibleTranscriptEntry(role="assistant", text="可见答复"),
    )
    assert restored.create_session("demo", session_id) == session_id
    assert restored.summary(session_id).transcript == summary.transcript
    assert any(
        message.role == "user" and message.content == "private user text API_KEY=[已隐藏]"
        for message in restored._agent._sessions[session_id].provider_messages
    )
    assert "private user text" in runtime.store.database_path.read_text(errors="ignore")
    assert "API_KEY=secret" not in runtime.store.database_path.read_text(errors="ignore")


def test_runtime_redacts_authorization_bearer_values_before_persisting(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))  # type: ignore[attr-defined]
    runtime = ConversationRuntime(ConversationAgent(ScriptedConversationModel([])), ConversationStore(tmp_path))
    session_id = runtime.create_session("demo")

    runtime._append_visible_message(session_id, "user", "Authorization: Bearer very-secret-token")

    stored = runtime.store.database_path.read_text(errors="ignore")
    assert "very-secret-token" not in stored
    assert "Authorization: [已隐藏]" in stored


def test_runtime_redacts_complete_basic_authorization_value_before_persisting(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))  # type: ignore[attr-defined]
    runtime = ConversationRuntime(ConversationAgent(ScriptedConversationModel([])), ConversationStore(tmp_path))
    session_id = runtime.create_session("demo")

    runtime._append_visible_message(session_id, "user", "Authorization: Basic dXNlcjpwYXNz")

    stored = runtime.store.database_path.read_text(errors="ignore")
    assert "dXNlcjpwYXNz" not in stored
    assert "Authorization: [已隐藏]" in stored


def test_runtime_records_a_promoted_queued_turn_under_its_own_history(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))  # type: ignore[attr-defined]
    model = ScriptedConversationModel([
        [TextDelta("first"), ResponseFinished("stop")],
        [TextDelta("second"), ResponseFinished("stop")],
    ])
    runtime = ConversationRuntime(ConversationAgent(model), ConversationStore(tmp_path))
    session_id = runtime.create_session("demo")
    first_turn, _ = runtime.begin_turn(session_id, "first request")
    queued_turn, _ = runtime.queue(session_id, "second request")

    events = tuple(runtime.run_turn(session_id, first_turn))
    summary = runtime.summary(session_id)

    assert any(event.turn_id == queued_turn and event.kind == "turn_completed" for event in events)
    assert [entry.text for entry in summary.transcript] == ["first request", "second request", "first", "second"]
    assert [turn.final_text for turn in summary.turns] == ["本轮已完成。", "本轮已完成。"]


def test_restored_provider_context_uses_recent_dialogue_without_embedding_full_transcript_twice() -> None:
    now = datetime.now(timezone.utc)
    entries = tuple(
        VisibleTranscriptEntry(role="user" if index % 2 == 0 else "assistant", text=f"message-{index}")
        for index in range(40)
    )
    summary = ConversationSummary(
        id=uuid4(), project_title="demo", created_at=now, updated_at=now, turns=(), transcript=entries
    )
    agent = ConversationAgent(ScriptedConversationModel([]))
    session_id = agent.create_session(summary)
    messages = agent._sessions[session_id].provider_messages

    assert "message-0" not in messages[1].content
    assert "message-39" in tuple(message.content for message in messages)
    assert all("message-0" not in message.content for message in messages)


def test_restored_provider_context_keeps_only_recent_safe_turn_facts() -> None:
    now = datetime.now(timezone.utc)
    turns = tuple(
        SafeTurnSummary(
            terminal_status="completed", changed_paths=(f"src/value-{index}.py",),
            pytest_outcome="passed", approval_outcome="none", final_text=f"turn-{index}",
        )
        for index in range(40)
    )
    summary = ConversationSummary(
        id=uuid4(), project_title="demo", created_at=now, updated_at=now, turns=turns
    )

    agent = ConversationAgent(ScriptedConversationModel([]))
    session_id = agent.create_session(summary)
    context = agent._sessions[session_id].provider_messages[1].content

    assert "turn-0" not in context
    assert "turn-39" in context


def test_runtime_deletes_current_conversation_only_when_an_older_one_remains(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))  # type: ignore[attr-defined]
    runtime = ConversationRuntime(ConversationAgent(ScriptedConversationModel([])), ConversationStore(tmp_path))
    first = runtime.create_session("demo")
    second = runtime.create_session("demo")

    replacement = runtime.delete_session(second)

    assert replacement is not None and replacement.id == first
    assert tuple(summary.id for summary in runtime.store.summaries()) == (first,)
    assert runtime.delete_session(first) is None


def test_runtime_reuses_a_previously_loaded_history_when_returning_after_delete(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))  # type: ignore[attr-defined]
    runtime = ConversationRuntime(ConversationAgent(ScriptedConversationModel([])), ConversationStore(tmp_path))
    first = runtime.create_session("demo")
    second = runtime.create_session("demo")

    replacement = runtime.delete_session(second)

    assert replacement is not None and replacement.id == first
    assert runtime.create_session("demo", replacement.id) == first


def test_runtime_forwards_a_one_turn_goal_to_the_agent(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))  # type: ignore[attr-defined]
    model = ScriptedConversationModel([[TextDelta("done"), ResponseFinished("stop")]])
    runtime = ConversationRuntime(ConversationAgent(model), ConversationStore(tmp_path))
    session_id = runtime.create_session("demo")

    turn_id, _ = runtime.begin_turn(session_id, "repair", goal="Keep it small")
    list(runtime.run_turn(session_id, turn_id))

    assert ProviderMessage(role="system", content="Current turn goal: Keep it small") in model.received_messages[0]


def test_runtime_summary_maps_safe_tool_facts_from_actual_events(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "value.py").write_text("VALUE = 1\n")
    (tmp_path / "tests" / "test_value.py").write_text("from src.value import VALUE\n\ndef test_value(): assert VALUE == 2\n")
    config = safe_config(tmp_path)
    patch = "--- a/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
    model = ScriptedConversationModel([
        [ToolCallDelta(0, "read", "read_file", json.dumps({"path": "src/value.py"})), ResponseFinished("tool_calls")],
        [ToolCallDelta(0, "patch", "apply_patch", json.dumps({"unified_diff": patch})), ResponseFinished("tool_calls")],
        [ToolCallDelta(0, "test", "run_pytest", "{}"), ResponseFinished("tool_calls")],
        [TextDelta("done"), ResponseFinished("stop")],
    ])
    runtime = ConversationRuntime(
        ConversationAgent(model, governed_tool_definitions(config), ToolGovernor(config), ToolExecutor(tmp_path, config)),
        ConversationStore(tmp_path),
    )
    session_id = runtime.create_session("demo")
    turn_id, _ = runtime.begin_turn(session_id, "repair")

    list(runtime.run_turn(session_id, turn_id))
    summary = runtime.summary(session_id).turns[0]

    assert summary.changed_paths == ("src/value.py",)
    assert summary.pytest_outcome == "passed"
    assert summary.approval_outcome == "none"
    assert summary.final_text == "本轮已完成。已修改 src/value.py。pytest：通过。"
