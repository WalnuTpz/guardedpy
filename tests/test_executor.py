"""Focused factual executor contracts for Task 22.2."""

from pathlib import Path
from uuid import uuid4

from conftest import safe_config
from guardedpy.conversation import ToolCall, Turn
from guardedpy.executor import ToolExecutor
from guardedpy.feedback import PytestRun


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
    assert result.provider_result["missing_paths"] == ["src/value.py"]
    assert target.read_text() == "first\nsecond\n"


def test_executor_reports_every_unread_existing_patch_target(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "value.py").write_text("VALUE = 1\n")
    (tmp_path / "tests" / "test_value.py").write_text("def test_value(): assert True\n")
    turn = Turn(id=uuid4(), session_id=uuid4(), initial_text="repair", mode="normal")
    executor = ToolExecutor(tmp_path, safe_config(tmp_path))

    result = executor.execute(turn, uuid4(), ToolCall(
        "patch", "apply_patch",
        "{\"unified_diff\":\"--- a/src/value.py\\n+++ b/src/value.py\\n@@ -1 +1 @@\\n-VALUE = 1\\n+VALUE = 2\\n--- a/tests/test_value.py\\n+++ b/tests/test_value.py\\n@@ -1 +1 @@\\n-def test_value(): assert True\\n+def test_value(): assert False\\n\"}",
    ))

    assert result.code == "read_required"
    assert result.provider_result["missing_paths"] == ["src/value.py", "tests/test_value.py"]


def test_executor_runs_a_project_python_file_without_a_prior_read(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "hello.py").write_text("print('hello world')\n")
    turn = Turn(
        id=uuid4(), session_id=uuid4(), initial_text="run it", mode="normal", needs_full_verification=True
    )
    executor = ToolExecutor(tmp_path, safe_config(tmp_path))

    result = executor.execute(
        turn, uuid4(), ToolCall("run", "run_python", '{"path":"src/hello.py"}')
    )

    assert result.code == "ok"
    assert result.provider_result["output"] == "hello world\n"
    assert turn.needs_full_verification is True


def test_executor_rejects_non_python_program_targets(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "notes.txt").write_text("not executable\n")
    turn = Turn(id=uuid4(), session_id=uuid4(), initial_text="run it", mode="normal")
    executor = ToolExecutor(tmp_path, safe_config(tmp_path))

    result = executor.execute(
        turn, uuid4(), ToolCall("run", "run_python", '{"path":"src/notes.txt"}')
    )

    assert result.code == "not_python_file"


def test_executor_keeps_verification_for_zero_collection(tmp_path: Path, monkeypatch: object) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_real.py").write_text("def test_real(): assert False\n")
    turn = Turn(id=uuid4(), session_id=uuid4(), initial_text="repair", mode="normal", needs_full_verification=True)
    executor = ToolExecutor(tmp_path, safe_config(tmp_path))
    monkeypatch.setattr(executor._workspace, "run_pytest", lambda nodes: PytestRun(0, "collected 0 items\n", "", False))  # type: ignore[attr-defined]

    result = executor.execute(turn, uuid4(), ToolCall("test", "run_pytest", "{}"))

    assert result.provider_result["kind"] == "execution_error"
    assert turn.needs_full_verification is True


def test_executor_normalizes_pytest_nodes_before_provider_output(tmp_path: Path, monkeypatch: object) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_real.py").write_text("def test_real(): assert False\n")
    executor = ToolExecutor(tmp_path, safe_config(tmp_path))
    turn = Turn(id=uuid4(), session_id=uuid4(), initial_text="repair", mode="normal")
    monkeypatch.setattr(
        executor._workspace, "run_pytest",
        lambda nodes: PytestRun(1, "FAILED tests/test_real.py::test_real - AssertionError\nFAILED ../outside.py::test_x - AssertionError\n", "", False),
    )  # type: ignore[attr-defined]

    result = executor.execute(turn, uuid4(), ToolCall("test", "run_pytest", "{}"))

    assert result.provider_result["nodes"] == ("tests/test_real.py::test_real",)


def test_existing_patch_with_header_like_removed_content_uses_only_real_headers(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    target = tmp_path / "src" / "value.py"
    target.write_text("-- marker\n")
    turn = Turn(id=uuid4(), session_id=uuid4(), initial_text="repair", mode="normal")
    executor = ToolExecutor(tmp_path, safe_config(tmp_path))
    executor.execute(turn, uuid4(), ToolCall("read", "read_file", '{"path":"src/value.py"}'))

    result = executor.execute(turn, uuid4(), ToolCall(
        "patch", "apply_patch", '{"unified_diff":"--- a/src/value.py\\n+++ b/src/value.py\\n@@ -1 +1 @@\\n--- marker\\n++ marker\\n"}'
    ))

    assert result.code == "ok"
    assert target.read_text() == "+ marker\n"
