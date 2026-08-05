"""Minimal SQLite audit events kept outside the governed project root."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
import sqlite3
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from guardedpy.actions import Action, stable_hash
from guardedpy.config import app_state_dir
from guardedpy.domain import FeedbackKind, PolicyVerdict, TaskStatus


class StopReason(StrEnum):
    SERVICE_RESTARTED = "service_restarted"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    ROUND_LIMIT = "round_limit"
    REPEATED_ACTION = "repeated_action"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    UNRECOVERABLE_ERROR = "unrecoverable_error"


_ACTION_SUMMARIES = {
    "list_files": "list workspace files",
    "read_file": "read workspace file",
    "apply_patch": "apply source patch",
    "delete_path": "delete workspace path",
    "run_pytest": "run configured tests",
    "run_command": "run approved command",
    "request_approval": "request action approval",
    "finish": "finish task",
}
_FEEDBACK_TEMPLATES = {
    FeedbackKind.PASSED: "pytest passed",
    FeedbackKind.ASSERTION_FAILURE: "pytest assertion failure",
    FeedbackKind.COLLECTION_ERROR: "pytest collection error",
    FeedbackKind.EXECUTION_ERROR: "pytest execution error",
    FeedbackKind.TIMEOUT: "pytest timed out",
}

class FeedbackAudit(BaseModel):
    """The narrow, structured feedback input allowed to reach audit storage."""

    model_config = ConfigDict(extra="forbid")

    kind: FeedbackKind
    node_id: str | None = None


class RunEvent(BaseModel):
    """Structured event input from the orchestrator, before audit projection."""

    model_config = ConfigDict(extra="forbid")

    task_id: UUID
    task_status: TaskStatus
    action: Action | None = None
    policy_verdict: PolicyVerdict | None = None
    approval_granted: bool | None = None
    feedback: FeedbackAudit | None = None
    retry_count: int | None = None
    stop_reason: StopReason | None = None


class StoredRunEvent(BaseModel):
    """The fixed-format representation returned from persistent audit storage."""

    model_config = ConfigDict(extra="forbid")

    task_id: UUID
    task_status: TaskStatus
    action_summary: str | None = None
    action_hash: str | None = None
    policy_verdict: PolicyVerdict | None = None
    approval_granted: bool | None = None
    feedback_kind: FeedbackKind | None = None
    feedback_excerpt: str | None = None
    retry_count: int | None = None
    stop_reason: StopReason | None = None
    id: int | None = None
    created_at: datetime | None = None


def _project_action(action: Action | None) -> tuple[str | None, str | None]:
    if action is None:
        return None, None
    return _ACTION_SUMMARIES[action.kind], stable_hash(action)


def _project_feedback(feedback: FeedbackAudit | None) -> tuple[FeedbackKind | None, str | None]:
    if feedback is None:
        return None, None
    return feedback.kind, _FEEDBACK_TEMPLATES[feedback.kind]


class EventStore:
    """Append-only task events with a separate current-state index."""

    def __init__(self, project_root: Path) -> None:
        self.database_path = app_state_dir(project_root) / "events.sqlite3"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def append(self, event: RunEvent) -> StoredRunEvent:
        """Persist one fixed-format audit projection and update the task's state."""
        action_summary, action_hash = _project_action(event.action)
        feedback_kind, feedback_excerpt = _project_feedback(event.feedback)
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
                    action_hash,
                    event.policy_verdict.value if event.policy_verdict else None,
                    event.approval_granted,
                    feedback_kind.value if feedback_kind else None,
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
            return StoredRunEvent(
                task_id=event.task_id,
                task_status=event.task_status,
                action_summary=action_summary,
                action_hash=action_hash,
                policy_verdict=event.policy_verdict,
                approval_granted=event.approval_granted,
                feedback_kind=feedback_kind,
                feedback_excerpt=feedback_excerpt,
                retry_count=event.retry_count,
                stop_reason=event.stop_reason,
                id=cursor.lastrowid,
                created_at=created_at,
            )

    def events_for(self, task_id: UUID) -> list[StoredRunEvent]:
        """Return a task's audit trail in insertion order."""
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE task_id = ? ORDER BY id", (str(task_id),)
            ).fetchall()
        return [
            StoredRunEvent(
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
                stop_reason=StopReason(row[10]) if row[10] else None,
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
                    stop_reason=StopReason.SERVICE_RESTARTED,
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
