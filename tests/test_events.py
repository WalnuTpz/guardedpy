"""Behavioral coverage for the minimal persistent audit trail."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from guardedpy.actions import parse_action
from guardedpy.domain import FeedbackKind, PolicyVerdict, TaskStatus
from guardedpy.events import EventStore, FeedbackAudit, RunEvent, StopReason


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EventStore:
    """Use a real, disposable app-state location separate from the project."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return EventStore(project_root)


def test_append_projects_an_action_and_feedback_without_persisting_their_text(
    store: EventStore,
) -> None:
    """Catches audit storage using untrusted action or pytest text directly."""
    task_id = uuid4()
    action = parse_action(
        '{"kind":"apply_patch","summary":"replace every secret","diff":"--- a/key.py +++ b/key.py +sk-short-key"}'
    )

    stored = store.append(
        RunEvent(
            task_id=task_id,
            task_status=TaskStatus.RUNNING,
            action=action,
            policy_verdict=PolicyVerdict.ALLOW,
            approval_granted=True,
            feedback=FeedbackAudit(
                kind=FeedbackKind.ASSERTION_FAILURE,
                node_id="tests/test_events.py::test_projection",
            ),
            retry_count=2,
            stop_reason=StopReason.ROUND_LIMIT,
        )
    )

    assert stored.task_status is TaskStatus.RUNNING
    assert stored.action_summary == "apply source patch"
    assert stored.action_hash == action.stable_hash()
    assert stored.policy_verdict is PolicyVerdict.ALLOW
    assert stored.approval_granted is True
    assert stored.feedback_kind is FeedbackKind.ASSERTION_FAILURE
    assert stored.feedback_excerpt == "pytest assertion failure at tests/test_events.py::test_projection"
    assert stored.retry_count == 2
    assert stored.stop_reason is StopReason.ROUND_LIMIT
    assert "sk-short-key" not in repr(stored)
    assert "replace every secret" not in repr(stored)

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


@pytest.mark.parametrize(
    ("field_name", "unsafe_content"),
    [
        ("action_summary", "sk-short-key"),
        ("action_summary", "--- a/source.py +++ b/source.py @@ -1 +1 @@ -old +new"),
        ("feedback_excerpt", "context: prior model output and source tree"),
        ("action_summary", "delete_path src/guardedpy/events.py"),
        ("action_hash", "sk-short-key"),
        ("stop_reason", "untrusted arbitrary reason"),
    ],
)
def test_run_event_rejects_all_direct_audit_text_before_any_event_is_written(
    store: EventStore, field_name: str, unsafe_content: str
) -> None:
    """Catches short secrets, minified payloads, and context entering audit fields."""
    task_id = uuid4()

    with pytest.raises(ValidationError):
        RunEvent(
            task_id=task_id,
            task_status=TaskStatus.RUNNING,
            **{field_name: unsafe_content},
        )

    assert store.events_for(task_id) == []


def test_feedback_audit_rejects_a_node_that_could_carry_arbitrary_text() -> None:
    """Catches a feedback template interpolating arbitrary pytest output into SQLite."""

    with pytest.raises(ValidationError, match="feedback node"):
        FeedbackAudit(kind=FeedbackKind.ASSERTION_FAILURE, node_id="sk-short-key")


def test_mark_unfinished_interrupted_adds_terminal_event_only_for_active_tasks(
    store: EventStore,
) -> None:
    """Catches service restart recovery that resumes or mutates completed tasks."""
    active_task_ids = (uuid4(), uuid4(), uuid4())
    completed_task_id = uuid4()
    for task_id, status in zip(
        active_task_ids,
        (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.WAITING_APPROVAL),
        strict=True,
    ):
        store.append(RunEvent(task_id=task_id, task_status=status))
    store.append(RunEvent(task_id=completed_task_id, task_status=TaskStatus.COMPLETED))

    interrupted = store.mark_unfinished_interrupted()

    assert set(interrupted) == set(active_task_ids)
    for task_id in active_task_ids:
        active_events = store.events_for(task_id)
        assert active_events[-1].task_status is TaskStatus.INTERRUPTED
        assert active_events[-1].stop_reason == "service_restarted"
    assert [event.task_status for event in store.events_for(completed_task_id)] == [
        TaskStatus.COMPLETED
    ]
