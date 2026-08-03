"""User-approved, project-scoped memories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from time import time_ns
from uuid import UUID, uuid4

from guardedpy.config import app_state_dir


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A memory candidate or an approved memory."""

    id: UUID
    task_id: UUID
    text: str
    approved_at: int | None = None


class MemoryStore:
    """Persist only memories a user has explicitly approved."""

    def __init__(self, project_root: Path) -> None:
        self._path = app_state_dir(project_root) / "memories.json"
        self._proposals: dict[UUID, MemoryEntry] = {}
        self._approved = self._load()

    def propose(self, task_id: UUID, text: str) -> MemoryEntry:
        entry = MemoryEntry(id=uuid4(), task_id=task_id, text=text)
        self._proposals[entry.id] = entry
        return entry

    def proposals(self) -> list[MemoryEntry]:
        """Return only the current process's unapproved memory candidates."""
        return list(self._proposals.values())

    def approved(self) -> list[MemoryEntry]:
        """Return approved memories for the local memory-management view."""
        return list(self._approved.values())

    def approve(self, memory_id: UUID) -> MemoryEntry:
        proposal = self._proposals.pop(memory_id)
        entry = MemoryEntry(
            id=proposal.id,
            task_id=proposal.task_id,
            text=proposal.text,
            approved_at=time_ns(),
        )
        self._approved[entry.id] = entry
        self._save()
        return entry

    def search(self, query: str) -> list[MemoryEntry]:
        query_words = set(_keywords(query))
        if not query_words:
            return []
        matches = [
            entry
            for entry in self._approved.values()
            if query_words.intersection(_keywords(entry.text))
        ]
        return sorted(
            matches,
            key=lambda entry: (
                -len(query_words.intersection(_keywords(entry.text))),
                -(entry.approved_at or 0),
            ),
        )[:5]

    def delete(self, memory_id: UUID) -> None:
        if memory_id in self._proposals:
            del self._proposals[memory_id]
            return
        if memory_id not in self._approved:
            raise KeyError(memory_id)
        del self._approved[memory_id]
        self._save()

    def _load(self) -> dict[UUID, MemoryEntry]:
        if not self._path.exists():
            return {}
        records = json.loads(self._path.read_text())
        return {
            UUID(record["id"]): MemoryEntry(
                id=UUID(record["id"]),
                task_id=UUID(record["task_id"]),
                text=record["text"],
                approved_at=record["approved_at"],
            )
            for record in records
        }

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "id": str(entry.id),
                "task_id": str(entry.task_id),
                "text": entry.text,
                "approved_at": entry.approved_at,
            }
            for entry in self._approved.values()
        ]
        self._path.write_text(json.dumps(records))


def _keywords(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))
