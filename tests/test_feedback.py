from __future__ import annotations

from pathlib import Path, PurePosixPath
import subprocess
import sys
from unittest.mock import patch

import pytest

from conftest import safe_config
from guardedpy.config import HarnessConfig
from guardedpy.discovery import ProjectProfile
from guardedpy.domain import FeedbackKind
from guardedpy.feedback import FeedbackCollector, PytestFeedback, PytestRun
from guardedpy.workspace import Workspace


def test_collector_extracts_assertion_failure_node_and_bounded_excerpt() -> None:
    """Catches feedback that loses the failed test identity or returns all pytest output."""
    run = PytestRun(
        1,
        "FAILED tests/test_a.py::test_x - assertion failed\nE       assert 1 == 2\n" + "x" * 2_000,
        "",
        False,
    )

    result = FeedbackCollector().collect(run)

    assert result.kind is FeedbackKind.ASSERTION_FAILURE
    assert result.node_ids == ("tests/test_a.py::test_x",)
    assert "assert 1 == 2" in result.excerpt
    assert len(result.excerpt) <= 800
    assert "x" * 2_000 not in result.excerpt


def test_collector_treats_failed_runtime_with_node_as_execution_error() -> None:
    """Catches a FAILED node line turning an exception in the test into red TDD evidence."""
    run = PytestRun(
        1,
        "FAILED tests/test_parser.py::test_bad_input - TypeError: parser() missing 1 required argument\n",
        "",
        False,
    )

    result = FeedbackCollector().collect(run)

    assert result.kind is FeedbackKind.EXECUTION_ERROR
    assert result.node_ids == ("tests/test_parser.py::test_bad_input",)


def test_collector_treats_noncollection_error_as_execution_error_before_assertion() -> None:
    """Catches an ERROR-phase fixture failure being mistaken for an assertion-only suite."""
    run = PytestRun(
        1,
        "FAILED tests/test_example.py::test_assertion - AssertionError: assert 0\n"
        "ERROR tests/test_example.py::test_fixture\n",
        "",
        False,
    )

    result = FeedbackCollector().collect(run)

    assert result.kind is FeedbackKind.EXECUTION_ERROR
    assert result.node_ids == ()


def test_collector_does_not_treat_source_assert_in_a_runtime_traceback_as_assertion_evidence() -> None:
    """Catches a source assertion line making a TypeError look like an assertion failure."""
    run = PytestRun(
        1,
        "FAILED tests/test_parser.py::test_bad_input - TypeError: unsupported input\n"
        "src/parser.py:8: in parse\n"
        "    assert payload is not None\n"
        "E   TypeError: unsupported input\n",
        "",
        False,
    )

    result = FeedbackCollector().collect(run)

    assert result.kind is FeedbackKind.EXECUTION_ERROR
    assert result.node_ids == ("tests/test_parser.py::test_bad_input",)


def test_collector_prioritizes_timeout_without_inferring_a_node() -> None:
    """Catches a timed-out run being misreported from its incomplete output."""
    result = FeedbackCollector().collect(
        PytestRun(-1, "FAILED tests/test_a.py::test_x", "", True)
    )

    assert result.kind is FeedbackKind.TIMEOUT
    assert result.node_ids == ()


@pytest.mark.parametrize(
    ("run", "expected_kind", "expected_nodes"),
    [
        (PytestRun(0, "1 passed in 0.01s\n", "", False), FeedbackKind.PASSED, ()),
        (
            PytestRun(2, "ERROR collecting tests/test_b.py\n", "", False),
            FeedbackKind.COLLECTION_ERROR,
            ("tests/test_b.py",),
        ),
        (
            PytestRun(3, "INTERNALERROR unexpected failure\n", "", False),
            FeedbackKind.EXECUTION_ERROR,
            (),
        ),
    ],
)
def test_collector_classifies_the_remaining_pytest_outcomes(
    run: PytestRun, expected_kind: FeedbackKind, expected_nodes: tuple[str, ...]
) -> None:
    """Catches a classifier branch that maps a pytest result to the wrong feedback kind."""
    result = FeedbackCollector().collect(run)

    assert result.kind is expected_kind
    assert result.node_ids == expected_nodes


def test_normalize_nodes_keeps_ordered_existing_test_nodes_and_drops_unsafe_tokens(
    tmp_path: Path,
) -> None:
    """Catches automatic repair targets retaining duplicates, missing files, or root escapes."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_a.py").write_text("def test_a() -> None:\n    assert False\n")
    feedback = PytestFeedback(
        FeedbackKind.ASSERTION_FAILURE,
        (
            "tests/test_a.py::test_a",
            "tests/test_a.py::test_a",
            "tests/missing.py::test_missing",
            "../outside.py::test_outside",
        ),
        "assert False",
    )

    normalized = FeedbackCollector.normalize_nodes(
        feedback, tmp_path, (PurePosixPath("tests"),)
    )

    assert normalized == PytestFeedback(
        FeedbackKind.ASSERTION_FAILURE,
        ("tests/test_a.py::test_a",),
        "assert False",
    )


def test_collector_keeps_every_observed_assertion_node_beyond_twenty() -> None:
    """Catches automatic repair discovery silently truncating a large repair set."""
    nodes = tuple(f"tests/test_many.py::test_{index}" for index in range(25))
    run = PytestRun(
        1,
        "\n".join(f"FAILED {node} - AssertionError" for node in nodes),
        "",
        False,
    )

    feedback = FeedbackCollector().collect(run)

    assert feedback.kind is FeedbackKind.ASSERTION_FAILURE
    assert feedback.node_ids == nodes


def test_run_pytest_uses_the_selected_root_as_its_working_directory(tmp_path: Path) -> None:
    """Catches a runner that executes project tests from the harness process directory."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_cwd.py").write_text(
        "from pathlib import Path\n\n"
        "def test_selected_root_is_cwd() -> None:\n"
        f"    assert Path.cwd() == Path({str(tmp_path)!r})\n"
    )
    config = safe_config(tmp_path)

    result = Workspace(tmp_path, config).run_pytest(("tests/test_cwd.py",))

    assert result.timed_out is False
    assert result.exit_code == 0
    assert "1 passed" in result.stdout


def test_run_pytest_returns_a_structured_timeout_result(tmp_path: Path) -> None:
    """Catches a test timeout that escapes as a subprocess exception instead of feedback."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_wait.py").write_text(
        "import time\n\n"
        "def test_waits() -> None:\n"
        "    time.sleep(10)\n"
    )
    config = safe_config(tmp_path).model_copy(update={"timeout_seconds": 5})

    result = Workspace(tmp_path, config).run_pytest(("tests/test_wait.py",))

    assert result.exit_code == -1
    assert result.timed_out is True


def test_run_pytest_explicitly_disables_shell_execution(tmp_path: Path) -> None:
    """Catches a runner that relies on subprocess's default shell setting."""
    config = safe_config(tmp_path)
    completed = subprocess.CompletedProcess(("pytest",), 0, "1 passed\n", "")

    with patch("guardedpy.workspace.subprocess.run", return_value=completed) as run:
        result = Workspace(tmp_path, config).run_pytest(("tests/test_a.py",))

    assert result.exit_code == 0
    assert run.call_args.kwargs["shell"] is False


def test_run_pytest_rejects_a_target_outside_configured_test_directories(
    tmp_path: Path,
) -> None:
    """Catches a runner that lets model-supplied targets select files outside test roots."""
    config = safe_config(tmp_path)

    with pytest.raises(ValueError, match="configured test directories"):
        Workspace(tmp_path, config).run_pytest(("src/module.py",))


@pytest.mark.parametrize("target", ("", "-k"))
def test_run_pytest_rejects_empty_or_option_targets_even_with_project_test_root(
    tmp_path: Path, target: str
) -> None:
    """Catches a target value that turns pytest selection into a command option or full run."""
    config = HarnessConfig(
        profile=ProjectProfile(
            root=tmp_path.resolve(),
            discovery_source="root_tests",
            source_dirs=(PurePosixPath("."),),
            test_dirs=(PurePosixPath("."),),
            pytest_command=(sys.executable, "-m", "pytest"),
        )
    )

    with pytest.raises(ValueError, match="must name files"):
        Workspace(tmp_path, config).run_pytest((target,))
