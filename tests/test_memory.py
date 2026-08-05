from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from guardedpy.memory import MemoryStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryStore:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return MemoryStore(tmp_path / "project")


def test_proposed_memory_is_not_persisted_until_approved(
    store: MemoryStore, tmp_path: Path
) -> None:
    project_root = tmp_path / "project"
    store.propose(uuid4(), "Use the src and tests layout")

    assert MemoryStore(project_root).search("layout") == []

    approved = store.propose(uuid4(), "Keep parser errors concise")
    store.approve(approved.id)

    assert [item.text for item in MemoryStore(project_root).search("parser")] == [
        "Keep parser errors concise"
    ]


def test_search_orders_keyword_overlap_then_recency_and_limits_to_five(
    store: MemoryStore,
) -> None:
    entries = [
        store.propose(uuid4(), text)
        for text in (
            "parser output",
            "parser schema",
            "parser schema validation",
            "parser logging",
            "parser actions",
            "parser tooling",
        )
    ]
    for entry in entries:
        store.approve(entry.id)

    assert [item.text for item in store.search("parser schema")] == [
        "parser schema validation",
        "parser schema",
        "parser tooling",
        "parser actions",
        "parser logging",
    ]


def test_delete_removes_an_approved_memory_from_future_search(
    store: MemoryStore,
) -> None:
    entry = store.propose(uuid4(), "Do not use shell strings")
    store.approve(entry.id)

    store.delete(entry.id)

    assert store.search("shell") == []
