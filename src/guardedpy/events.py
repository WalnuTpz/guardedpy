"""Minimal SQLite audit events kept outside the governed project root."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import UUID

from pydantic import BaseModel

from guardedpy.config import app_state_dir
from guardedpy.domain import FeedbackKind, PolicyVerdict, TaskStatus


MAX_AUDIT_TEXT_LENGTH = 500


class RunEvent(BaseModel):
    """The deliberately small, durable representation of one task event."""

    task_id: UUID
    task_status: TaskStatus
    action_summary: str | None = None
    action_hash: str | None = None
    policy_verdict: PolicyVerdict | None = None
    approval_granted: bool | None = None
    feedback_kind: FeedbackKind | None = None
    feedback_excerpt: str | None = None
    retry_count: int | None = None
    stop_reason: str | None = None
    id: int | None = None
    created_at: datetime | None = None


def _validate_audit_fragment(value: str | None) -> str | None:
    """Accept only a short, single-line human-readable audit fragment."""
    if value is None:
        return None
    if len(value) > MAX_AUDIT_TEXT_LENGTH:
        raise ValueError("audit text exceeds the persistent fragment limit")
    if "\n" in value or "\r" in value:
        raise ValueError("audit text must be a single-line fragment")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(decoded, (dict, list)):
        raise ValueError("audit text must not contain a structured action payload")
    return value


class EventStore:
    """Append-only task events with a separate current-state index."""

    def __init__(self, project_root: Path) -> None:
        self.database_path = app_state_dir(project_root) / "events.sqlite3"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def append(self, event: RunEvent) -> RunEvent:
        """Persist one minimal audit event and update the task's current state."""
        action_summary = _validate_audit_fragment(event.action_summary)
        feedback_excerpt = _validate_audit_fragment(event.feedback_excerpt)
        created_at = datetime.now(timezone.utc)
        with closing(sqlite3.connect(self.database_path)) as connection:
            cursor = connection.execute(
                """
                INSERT INTO events (
                    task_id, task_status, action_summary, action_hash,
                    policy_verdict, approval_granted, feedback_kind,
                    feedback_excerpt, retry_count, stop_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event.task_id),
                    event.task_status.value,
                    action_summary,
                    event.action_hash,
                    event.policy_verdict.value if event.policy_verdict else None,
                    event.approval_granted,
                    event.feedback_kind.value if event.feedback_kind else None,
                    feedback_excerpt,
                    event.retry_count,
                    event.stop_reason,
                    created_at.isoformat(),
                ),
            )
            connection.execute(
                """
                INSERT INTO task_states (task_id, task_status) VALUES (?, ?)
                ON CONFLICT(task_id) DO UPDATE SET task_status = excluded.task_status
                """,
                (str(event.task_id), event.task_status.value),
            )
            connection.commit()
            return event.model_copy(update={"id": cursor.lastrowid, "created_at": created_at})

    def events_for(self, task_id: UUID) -> list[RunEvent]:
        """Return a task's audit trail in insertion order."""
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE task_id = ? ORDER BY id", (str(task_id),)
            ).fetchall()
        return [
            RunEvent(
                id=row[0],
                task_id=UUID(row[1]),
                task_status=TaskStatus(row[2]),
                action_summary=row[3],
                action_hash=row[4],
                policy_verdict=PolicyVerdict(row[5]) if row[5] else None,
                approval_granted=bool(row[6]) if row[6] is not None else None,
                feedback_kind=FeedbackKind(row[7]) if row[7] else None,
                feedback_excerpt=row[8],
                retry_count=row[9],
                stop_reason=row[10],
                created_at=datetime.fromisoformat(row[11]),
            )
            for row in rows
        ]

    def mark_unfinished_interrupted(self) -> tuple[UUID, ...]:
        """Record interrupted terminal events for tasks left active at service restart."""
        active_statuses = (
            TaskStatus.PENDING.value,
            TaskStatus.RUNNING.value,
            TaskStatus.WAITING_APPROVAL.value,
        )
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT task_id FROM task_states
                WHERE task_status IN (?, ?, ?)
                ORDER BY task_id
                """,
                active_statuses,
            ).fetchall()

        interrupted = tuple(UUID(row[0]) for row in rows)
        for task_id in interrupted:
            self.append(
                RunEvent(
                    task_id=task_id,
                    task_status=TaskStatus.INTERRUPTED,
                    stop_reason="service_restarted",
                )
            )
        return interrupted

    def _initialize(self) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
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
                CREATE TABLE IF NOT EXISTS task_states (
                    task_id TEXT PRIMARY KEY,
                    task_status TEXT NOT NULL
                );
                """
            )
            connection.commit()
