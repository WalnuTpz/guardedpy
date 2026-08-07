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
)
from guardedpy.conversations import ConversationStore
from guardedpy.runtime import ConversationRuntime


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
