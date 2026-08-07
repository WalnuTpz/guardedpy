"""Focused factual executor contracts for Task 22.2."""

from pathlib import Path
from uuid import uuid4

from conftest import safe_config
from guardedpy.conversation import ToolCall, Turn
from guardedpy.executor import ToolExecutor


def test_executor_reads_then_applies_a_source_patch_and_marks_verification(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    target = tmp_path / "src" / "value.py"
    target.write_text("VALUE = 'broken'\n")
    turn = Turn(id=uuid4(), session_id=uuid4(), initial_text="repair", mode="normal")
    executor = ToolExecutor(tmp_path, safe_config(tmp_path))

    read = executor.execute(turn, uuid4(), ToolCall("read", "read_file", '{"path":"src/value.py"}'))
    patch = executor.execute(
        turn,
        uuid4(),
        ToolCall(
            "patch", "apply_patch",
            '{"unified_diff":"--- a/src/value.py\\n+++ b/src/value.py\\n@@ -1 +1 @@\\n-VALUE = \'broken\'\\n+VALUE = \'fixed\'\\n"}',
        ),
    )

    assert read.code == "ok"
    assert patch.changed_paths == ("src/value.py",)
    assert target.read_text() == "VALUE = 'fixed'\n"
    assert turn.needs_full_verification is True


def test_executor_rejects_patch_outside_discovered_source_and_test_roots(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    target = tmp_path / "README.md"
    target.write_text("old\n")
    turn = Turn(id=uuid4(), session_id=uuid4(), initial_text="repair", mode="normal")
    executor = ToolExecutor(tmp_path, safe_config(tmp_path))
    executor.execute(turn, uuid4(), ToolCall("read", "read_file", '{"path":"README.md"}'))

    result = executor.execute(turn, uuid4(), ToolCall(
        "patch", "apply_patch", '{"unified_diff":"--- a/README.md\\n+++ b/README.md\\n@@ -1 +1 @@\\n-old\\n+new\\n"}'
    ))

    assert result.code == "patch_invalid"
    assert target.read_text() == "old\n"


def test_tail_read_does_not_authorize_patch_of_an_unread_prefix(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    target = tmp_path / "src" / "value.py"
    target.write_text("first\nsecond\n")
    turn = Turn(id=uuid4(), session_id=uuid4(), initial_text="repair", mode="normal")
    executor = ToolExecutor(tmp_path, safe_config(tmp_path))
    executor.execute(turn, uuid4(), ToolCall("read", "read_file", '{"path":"src/value.py","offset":1,"limit":1}'))

    result = executor.execute(turn, uuid4(), ToolCall(
        "patch", "apply_patch", '{"unified_diff":"--- a/src/value.py\\n+++ b/src/value.py\\n@@ -1 +1 @@\\n-first\\n+fixed\\n"}'
    ))

    assert result.code == "read_required"
    assert target.read_text() == "first\nsecond\n"
