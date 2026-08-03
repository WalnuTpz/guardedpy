"""Behavioral coverage for the minimal persistent audit trail."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from guardedpy.domain import FeedbackKind, PolicyVerdict, TaskStatus
from guardedpy.events import EventStore, RunEvent


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EventStore:
    """Use a real, disposable app-state location separate from the project."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return EventStore(project_root)


def test_append_persists_only_the_allowed_audit_fields(store: EventStore) -> None:
    """Catches storing a full action body instead of its safe summary and hash."""
    task_id = uuid4()

    store.append(
        RunEvent(
            task_id=task_id,
            task_status=TaskStatus.RUNNING,
            action_summary="patch source",
            action_hash="abc123",
            policy_verdict=PolicyVerdict.ALLOW,
            approval_granted=True,
            feedback_kind=FeedbackKind.ASSERTION_FAILURE,
            feedback_excerpt="assertion failed",
            retry_count=2,
            stop_reason="round_limit",
        )
    )

    event = store.events_for(task_id)[0]
    assert event.task_status is TaskStatus.RUNNING
    assert event.action_summary == "patch source"
    assert event.action_hash == "abc123"
    assert event.policy_verdict is PolicyVerdict.ALLOW
    assert event.approval_granted is True
    assert event.feedback_kind is FeedbackKind.ASSERTION_FAILURE
    assert event.feedback_excerpt == "assertion failed"
    assert event.retry_count == 2
    assert event.stop_reason == "round_limit"

    with sqlite3.connect(store.database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
    assert columns == {
        "id",
        "task_id",
        "task_status",
        "action_summary",
        "action_hash",
        "policy_verdict",
        "approval_granted",
        "feedback_kind",
        "feedback_excerpt",
        "retry_count",
        "stop_reason",
        "created_at",
    }


def test_store_database_is_created_in_external_app_state(
    store: EventStore, tmp_path: Path
) -> None:
    """Catches placing agent-auditable state inside the selected project root."""
    store.append(RunEvent(task_id=uuid4(), task_status=TaskStatus.PENDING))

    assert store.database_path.is_file()
    assert store.database_path.is_relative_to(tmp_path / "state")
    assert not store.database_path.is_relative_to(tmp_path / "project")


def test_mark_unfinished_interrupted_adds_terminal_event_only_for_active_tasks(
    store: EventStore,
) -> None:
    """Catches service restart recovery that resumes or mutates completed tasks."""
    active_task_id = uuid4()
    completed_task_id = uuid4()
    store.append(RunEvent(task_id=active_task_id, task_status=TaskStatus.RUNNING))
    store.append(RunEvent(task_id=completed_task_id, task_status=TaskStatus.COMPLETED))

    interrupted = store.mark_unfinished_interrupted()

    assert interrupted == (active_task_id,)
    active_events = store.events_for(active_task_id)
    assert active_events[-1].task_status is TaskStatus.INTERRUPTED
    assert active_events[-1].stop_reason == "service_restarted"
    assert [event.task_status for event in store.events_for(completed_task_id)] == [
        TaskStatus.COMPLETED
    ]
