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
