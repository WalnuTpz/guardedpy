"""Project-scoped, text-free conversation references."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3
from uuid import UUID

from guardedpy.config import app_state_dir
from guardedpy.conversation import ConversationSummary as SafeConversationSummary


class ConversationStore:
    """Persist the minimal conversation index beside existing hashed application state."""

    def __init__(self, project_root: Path) -> None:
        self.database_path = app_state_dir(project_root) / "conversations.sqlite3"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(
                """
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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
