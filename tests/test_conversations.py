"""Safe project-scoped session-summary storage contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from uuid import uuid4

from guardedpy.config import app_state_dir
from guardedpy.conversation import ConversationSummary
from guardedpy.conversations import ConversationStore


def test_store_keeps_safe_session_summaries_without_retired_task_timeline_api(
    tmp_path: Path, monkeypatch: object
) -> None:
    """The store persists safe summaries, not the retired TaskState timeline."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))  # type: ignore[attr-defined]
    now = datetime.now(timezone.utc)
    summary = ConversationSummary(
        id=uuid4(), project_title="calculator", created_at=now, updated_at=now, turns=()
    )

    project = tmp_path / "project"
    legacy_database = app_state_dir(project) / "conversations.sqlite3"
    legacy_database.parent.mkdir(parents=True)
    with sqlite3.connect(legacy_database) as connection:
        connection.execute("CREATE TABLE conversations (id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE conversation_tasks (id TEXT PRIMARY KEY)")

    store = ConversationStore(project)
    store.save_summary(summary)

    recovered = ConversationStore(tmp_path / "project")
    assert recovered.load_summary(summary.id) == summary
    assert recovered.summaries() == (summary,)
    for retired_api in ("create", "attach_task", "list", "tasks"):
        assert not hasattr(recovered, retired_api)
    with sqlite3.connect(recovered.database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"conversation_summaries", "conversations", "conversation_tasks"} <= tables
