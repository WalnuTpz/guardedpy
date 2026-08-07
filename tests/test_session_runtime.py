"""Task 22.3 safe-summary and continuous-runtime contracts."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from guardedpy.conversation import (
    ConversationAgent,
    ConversationSummary,
    ResponseFinished,
    ScriptedConversationModel,
    TextDelta,
    ToolCallDelta,
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


def test_runtime_persists_terminal_turn_without_provider_thread(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))  # type: ignore[attr-defined]
    model = ScriptedConversationModel([[TextDelta("hello"), ResponseFinished("stop")]])
    runtime = ConversationRuntime(ConversationAgent(model), ConversationStore(tmp_path))
    session_id = runtime.create_session("demo")
    turn_id, immediate = runtime.begin_turn(session_id, "private user text")

    events = tuple(runtime.run_turn(session_id, turn_id))
    summary = runtime.summary(session_id)

    assert immediate.kind == "user_message"
    assert events[-1].kind == "turn_completed"
    assert summary.turns[0].terminal_status == "completed"
    assert summary.turns[0].final_text == "hello"
    assert "private user text" not in runtime.store.database_path.read_text(errors="ignore")


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
        ConversationAgent(model, governed_tool_definitions(), ToolGovernor(config), ToolExecutor(tmp_path, config)),
        ConversationStore(tmp_path),
    )
    session_id = runtime.create_session("demo")
    turn_id, _ = runtime.begin_turn(session_id, "repair")

    list(runtime.run_turn(session_id, turn_id))
    summary = runtime.summary(session_id).turns[0]

    assert summary.changed_paths == ("src/value.py",)
    assert summary.pytest_outcome == "passed"
    assert summary.approval_outcome == "none"
