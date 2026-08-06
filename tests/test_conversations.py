"""Safe project-scoped conversation index contracts."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4


def test_conversation_store_survives_restart_and_isolates_project_roots(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Catches conversation references leaking between projects or storing mutable transcript text."""
    from guardedpy.conversations import ConversationStore

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))  # type: ignore[attr-defined]
    first_root = tmp_path / "first"
    other_root = tmp_path / "other"
    first_root.mkdir()
    other_root.mkdir()
    task_id = uuid4()

    store = ConversationStore(first_root)
    conversation = store.create()
    store.attach_task(conversation.id, task_id)

    recovered = ConversationStore(first_root)
    summaries = recovered.list()
    assert len(summaries) == 1
    assert summaries[0].id == conversation.id
    assert summaries[0].task_ids == (task_id,)
    assert recovered.tasks(conversation.id) == (task_id,)
    assert ConversationStore(other_root).list() == ()
    assert set(summaries[0].model_dump()) == {"id", "created_at", "updated_at", "task_ids"}
