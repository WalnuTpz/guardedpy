"""Project-scoped, text-free conversation references."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict

from guardedpy.config import app_state_dir
from guardedpy.conversation import ConversationSummary as SafeConversationSummary


class ConversationSummary(BaseModel):
    """A durable conversation identity with only safe task references."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    task_ids: tuple[UUID, ...]


class ConversationStore:
    """Persist the minimal conversation index beside existing hashed application state."""

    def __init__(self, project_root: Path) -> None:
        self.database_path = app_state_dir(project_root) / "conversations.sqlite3"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_tasks (
                    conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    task_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (conversation_id, task_id)
                );
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id TEXT PRIMARY KEY,
                    summary_json TEXT NOT NULL
                );
                """
            )

    def save_summary(self, summary: SafeConversationSummary) -> None:
        """Durably replace one whitelist-validated session summary."""
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO conversation_summaries (id, summary_json) VALUES (?, ?)",
                (str(summary.id), summary.model_dump_json()),
            )
            connection.commit()

    def load_summary(self, conversation_id: UUID) -> SafeConversationSummary:
        """Load a safe summary without reconstructing provider history."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT summary_json FROM conversation_summaries WHERE id = ?",
                (str(conversation_id),),
            ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        return SafeConversationSummary.model_validate_json(row[0])

    def summaries(self) -> tuple[SafeConversationSummary, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT summary_json FROM conversation_summaries"
            ).fetchall()
        return tuple(SafeConversationSummary.model_validate_json(row[0]) for row in rows)

    def create(self) -> ConversationSummary:
        conversation_id = uuid4()
        now = datetime.now(timezone.utc)
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
                (str(conversation_id), now.isoformat(), now.isoformat()),
            )
            connection.commit()
        return ConversationSummary(id=conversation_id, created_at=now, updated_at=now, task_ids=())

    def attach_task(self, conversation_id: UUID, task_id: UUID) -> None:
        now = datetime.now(timezone.utc)
        with closing(self._connect()) as connection:
            if connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (str(conversation_id),)
            ).fetchone() is None:
                raise ValueError("conversation does not exist")
            cursor = connection.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM conversation_tasks WHERE conversation_id = ?",
                (str(conversation_id),),
            )
            position = cursor.fetchone()[0]
            result = connection.execute(
                "INSERT OR IGNORE INTO conversation_tasks (conversation_id, task_id, position) VALUES (?, ?, ?)",
                (str(conversation_id), str(task_id), position),
            )
            if result.rowcount == 0:
                return
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now.isoformat(), str(conversation_id))
            )
            connection.commit()

    def list(self) -> tuple[ConversationSummary, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return tuple(self._summary(UUID(row[0]), row[1], row[2]) for row in rows)

    def tasks(self, conversation_id: UUID) -> tuple[UUID, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT task_id FROM conversation_tasks WHERE conversation_id = ? ORDER BY position",
                (str(conversation_id),),
            ).fetchall()
        return tuple(UUID(row[0]) for row in rows)

    def _summary(self, conversation_id: UUID, created_at: str, updated_at: str) -> ConversationSummary:
        return ConversationSummary(
            id=conversation_id,
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(updated_at),
            task_ids=self.tasks(conversation_id),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
