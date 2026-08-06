"""Behavioral coverage for the minimal persistent audit trail."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from guardedpy.actions import parse_action
from conftest import safe_config
from guardedpy.domain import FeedbackKind, PolicyVerdict, TaskIntent, TaskState, TaskStatus
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
            permanent_eligible=True,
            feedback=FeedbackAudit(
                kind=FeedbackKind.ASSERTION_FAILURE,
                node_id="tests/test_projection.py::test_projection",
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
    assert stored.permanent_eligible is True
    assert stored.feedback_kind is FeedbackKind.ASSERTION_FAILURE
    assert stored.feedback_excerpt == "pytest assertion failure"
    assert stored.feedback_node_id == "tests/test_projection.py::test_projection"
    assert stored.retry_count == 2
    assert stored.stop_reason is StopReason.ROUND_LIMIT
    assert "sk-short-key" not in repr(stored)
    assert "replace every secret" not in repr(stored)

    with sqlite3.connect(store.database_path) as connection:
        stored_feedback = connection.execute(
            "SELECT feedback_kind, feedback_excerpt, feedback_node_id FROM events WHERE task_id = ?",
            (str(task_id),),
        ).fetchone()
        columns = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
    assert stored_feedback == (
        "assertion_failure",
        "pytest assertion failure",
        "tests/test_projection.py::test_projection",
    )
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
        "feedback_node_id",
        "retry_count",
        "stop_reason",
        "created_at",
    }


def test_feedback_audit_rejects_node_identifier_over_fixed_bound() -> None:
    """Catches untrusted pytest node identifiers bypassing the audit size limit."""
    with pytest.raises(ValidationError):
        FeedbackAudit(kind=FeedbackKind.ASSERTION_FAILURE, node_id="n" * 501)


def test_legacy_sqlite_schema_migrates_permanent_eligibility_and_feedback_node(
    store: EventStore,
) -> None:
    """Catches additive audit fields making an existing local event database unusable."""
    with sqlite3.connect(store.database_path) as connection:
        connection.executescript(
            """
            DROP TABLE event_policies;
            DROP TABLE events;
            CREATE TABLE events (
                id INTEGER PRIMARY KEY,
                task_id TEXT NOT NULL,
                task_status TEXT NOT NULL,
                action_summary TEXT,
                action_hash TEXT,
                policy_verdict TEXT,
                approval_granted INTEGER,
                feedback_kind TEXT,
                feedback_excerpt TEXT,
                retry_count INTEGER,
                stop_reason TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE event_policies (
                event_id INTEGER PRIMARY KEY,
                rule_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                action_projection TEXT,
                FOREIGN KEY(event_id) REFERENCES events(id)
            );
            """
        )

    migrated = EventStore(store.project_root)
    task_id = uuid4()
    migrated.append(
        RunEvent(
            task_id=task_id,
            task_status=TaskStatus.WAITING_APPROVAL,
            policy_verdict=PolicyVerdict.APPROVAL_REQUIRED,
            policy_rule_id="command.approval_required",
            policy_reason="the constrained command requires approval",
            permanent_eligible=True,
            feedback=FeedbackAudit(
                kind=FeedbackKind.ASSERTION_FAILURE,
                node_id="tests/test_projection.py::test_projection",
            ),
        )
    )

    stored = migrated.events_for(task_id)[0]
    assert stored.permanent_eligible is True
    assert stored.feedback_node_id == "tests/test_projection.py::test_projection"
    with sqlite3.connect(store.database_path) as connection:
        event_columns = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
        policy_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(event_policies)")
        }
    assert "feedback_node_id" in event_columns
    assert "permanent_eligible" in policy_columns


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


def test_register_task_persists_static_metadata_and_forces_pending_state(
    store: EventStore,
) -> None:
    """Catches restart history losing task identity/config or inheriting a forged status."""
    config = safe_config(store.project_root).model_copy(
        update={"model": "deepseek-v4-pro", "timeout_seconds": 45}
    )
    task = TaskState(
        description="Keep this task after restart",
        intent=TaskIntent.CODING,
        config=config,
        status=TaskStatus.COMPLETED,
    )

    store.register_task(task)

    restored = EventStore(store.project_root).tasks()
    assert len(restored) == 1
    assert restored[0].id == task.id
    assert restored[0].description == "Keep this task after restart"
    assert restored[0].intent is TaskIntent.CODING
    assert restored[0].config == config
    assert restored[0].status is TaskStatus.PENDING


def test_existing_event_database_adds_task_metadata_without_losing_events(
    store: EventStore,
) -> None:
    """Catches the additive task-history migration replacing an existing audit database."""
    existing_task_id = uuid4()
    store.append(RunEvent(task_id=existing_task_id, task_status=TaskStatus.COMPLETED))
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DROP TABLE IF EXISTS task_metadata")
        connection.commit()

    migrated = EventStore(store.project_root)
    new_task = TaskState(
        description="Registered after migration",
        intent=TaskIntent.CODING,
        config=safe_config(store.project_root),
    )
    migrated.register_task(new_task)

    assert migrated.events_for(existing_task_id)[0].task_status is TaskStatus.COMPLETED
    assert [task.id for task in migrated.tasks()] == [new_task.id]


def test_manual_mode_metadata_schema_migrates_without_blocking_new_intent_tasks(
    store: EventStore,
) -> None:
    """Catches legacy NOT NULL mode columns preventing registration after the intent migration."""
    existing = TaskState(
        description="Existing task",
        intent=TaskIntent.CODING,
        config=safe_config(store.project_root),
    )
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DROP TABLE task_metadata")
        connection.execute(
            """
            CREATE TABLE task_metadata (
                task_id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                mode TEXT NOT NULL,
                config_json TEXT NOT NULL,
                bugfix_target TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO task_metadata VALUES (?, ?, ?, ?, ?)",
            (
                str(existing.id),
                existing.description,
                "feature",
                existing.config.model_dump_json(),
                None,
            ),
        )
        connection.execute(
            "INSERT INTO task_states (task_id, task_status) VALUES (?, ?)",
            (str(existing.id), TaskStatus.COMPLETED.value),
        )
        connection.commit()

    migrated = EventStore(store.project_root)
    new_task = TaskState(
        description="New intent task",
        intent=TaskIntent.REVIEW,
        config=safe_config(store.project_root),
    )
    migrated.register_task(new_task)

    restored = migrated.tasks()
    assert [(task.id, task.intent) for task in restored] == [
        (existing.id, TaskIntent.CODING),
        (new_task.id, TaskIntent.REVIEW),
    ]
