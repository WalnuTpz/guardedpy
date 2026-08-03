"""Offline behavioral coverage for the governed agent loop."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from uuid import uuid4

import pytest

from guardedpy.config import HarnessConfig
from guardedpy.domain import TaskMode, TaskState, TaskStatus, TddPhase
from guardedpy.events import EventStore, RunEvent, StopReason
from guardedpy.llm import ScriptedLLM
from guardedpy.orchestrator import TaskOrchestrator


def _config() -> HarnessConfig:
    return HarnessConfig(
        source_dirs=(Path("src"),),
        test_dirs=(Path("tests"),),
        pytest_command=(sys.executable, "-m", "pytest", "-q"),
    )


def _action(**payload: object) -> str:
    return json.dumps(payload)


def _bugfix_task() -> TaskState:
    return TaskState(description="Repair the selected failure", mode=TaskMode.BUGFIX, config=_config())


def test_scripted_loop_returns_failure_feedback_then_corrects_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a loop that drops pytest feedback or finishes before an actual full green run."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "value.py").write_text("VALUE = 'broken'\n")
    (tmp_path / "tests" / "test_value.py").write_text(
        "from pathlib import Path\n\n"
        "def test_value_is_fixed() -> None:\n"
        "    assert Path('src/value.py').read_text() == \"VALUE = 'fixed'\\n\"\n"
    )
    llm = ScriptedLLM(
        [
            _action(kind="read_file", summary="inspect value", path="src/value.py"),
            _action(kind="run_pytest", summary="observe failure", targets=["tests/test_value.py"]),
            _action(
                kind="apply_patch",
                summary="repair value",
                diff="--- a/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n-VALUE = 'broken'\n+VALUE = 'fixed'\n",
            ),
            _action(kind="run_pytest", summary="run full suite", targets=[]),
            _action(kind="finish", summary="work complete", status="completed"),
        ]
    )
    task = _bugfix_task()

    finished = TaskOrchestrator(tmp_path, llm).run(task)

    assert finished.status is TaskStatus.COMPLETED
    assert (tmp_path / "src" / "value.py").read_text() == "VALUE = 'fixed'\n"
    assert any("assertion_failure" in context for context in llm.contexts[2:])
    events = EventStore(tmp_path).events_for(task.id)
    assert any(event.feedback_kind and event.feedback_kind.value == "assertion_failure" for event in events)
    assert events[-1].stop_reason is StopReason.COMPLETED


def test_invalid_model_json_stops_without_executing_a_workspace_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an invalid model response reaching a tool dispatcher."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    target = tmp_path / "untouched.txt"
    target.write_text("safe\n")
    task = _bugfix_task()

    stopped = TaskOrchestrator(tmp_path, ScriptedLLM(["not-json"])).run(task)

    assert stopped.status is TaskStatus.BLOCKED
    assert target.read_text() == "safe\n"
    events = EventStore(tmp_path).events_for(task.id)
    assert len(events) == 2
    assert events[-1].stop_reason is StopReason.INVALID_MODEL_OUTPUT


def test_repeated_action_stops_before_a_second_tool_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches repeated model actions consuming tool budget indefinitely."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "notes.txt").write_text("one\n")
    action = _action(kind="read_file", summary="inspect note", path="notes.txt")
    task = _bugfix_task()

    stopped = TaskOrchestrator(tmp_path, ScriptedLLM([action, action])).run(task)

    assert stopped.status is TaskStatus.BLOCKED
    assert EventStore(tmp_path).events_for(task.id)[-1].stop_reason is StopReason.REPEATED_ACTION


def test_repeated_operation_with_only_a_changed_summary_still_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches model-controlled summaries bypassing the no-progress repeat detector."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "notes.txt").write_text("one\n")
    task = _bugfix_task()
    llm = ScriptedLLM(
        [
            _action(kind="read_file", summary="first wording", path="notes.txt"),
            _action(kind="read_file", summary="different wording", path="notes.txt"),
        ]
    )

    stopped = TaskOrchestrator(tmp_path, llm).run(task)

    assert stopped.status is TaskStatus.BLOCKED
    assert EventStore(tmp_path).events_for(task.id)[-1].stop_reason is StopReason.REPEATED_ACTION


def test_cancellation_of_waiting_task_is_terminal_and_exact_approval_does_not_run_wrong_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches cancellation being ignored and approval hashes authorizing a different action."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    target = tmp_path / "obsolete.txt"
    target.write_text("remove only when approved\n")
    task = _bugfix_task()
    action = _action(kind="delete_path", summary="delete obsolete file", path="obsolete.txt")
    orchestrator = TaskOrchestrator(tmp_path, ScriptedLLM([action]))

    waiting = orchestrator.run(task)
    assert waiting.status is TaskStatus.WAITING_APPROVAL
    wrong = orchestrator.resolve_approval(task.id, "not-the-pending-hash", approved=True)
    cancelled = orchestrator.cancel(task.id)

    assert wrong is False
    assert target.exists()
    assert cancelled.status is TaskStatus.CANCELLED
    assert EventStore(tmp_path).events_for(task.id)[-1].stop_reason is StopReason.CANCELLED


def test_exact_approved_action_executes_once_and_keeps_full_action_out_of_event_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a matching approval that fails to execute, leaks arguments, or remains reusable."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    target = tmp_path / "obsolete.txt"
    target.write_text("remove only when approved\n")
    task = _bugfix_task()
    action = _action(kind="delete_path", summary="delete obsolete file", path="obsolete.txt")
    orchestrator = TaskOrchestrator(tmp_path, ScriptedLLM([action]))

    waiting = orchestrator.run(task)
    assert waiting.status is TaskStatus.WAITING_APPROVAL
    action_hash = EventStore(tmp_path).events_for(task.id)[-1].action_hash
    approved = orchestrator.resolve_approval(task.id, action_hash or "", approved=True)
    again = orchestrator.resolve_approval(task.id, action_hash or "", approved=True)

    assert approved is True
    assert target.exists() is False
    assert again is False
    assert task.status is TaskStatus.RUNNING
    assert "obsolete.txt" not in repr(EventStore(tmp_path).events_for(task.id))


def test_orchestrator_marks_previously_active_persisted_tasks_interrupted_on_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a new harness instance silently resuming a prior active task."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    store = EventStore(tmp_path)
    prior_id = uuid4()
    store.append(RunEvent(task_id=prior_id, task_status=TaskStatus.RUNNING))

    TaskOrchestrator(tmp_path, ScriptedLLM([]))

    interrupted = EventStore(tmp_path).events_for(prior_id)[-1]
    assert interrupted.task_status is TaskStatus.INTERRUPTED
    assert interrupted.stop_reason is StopReason.SERVICE_RESTARTED


def test_failed_patch_does_not_advance_tdd_phase_before_the_tool_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches recording a source patch before workspace atomic patching has succeeded."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "value.py").write_text("VALUE = 'broken'\n")
    (tmp_path / "tests" / "test_value.py").write_text("def test_stays_red() -> None:\n    assert False\n")
    task = _bugfix_task()
    llm = ScriptedLLM(
        [
            _action(kind="run_pytest", summary="observe red", targets=["tests/test_value.py"]),
            _action(kind="read_file", summary="inspect value", path="src/value.py"),
            _action(
                kind="apply_patch",
                summary="attempt repair",
                diff="--- a/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n-VALUE = 'not current'\n+VALUE = 'fixed'\n",
            ),
        ]
    )

    stopped = TaskOrchestrator(tmp_path, llm).run(task)

    assert stopped.status is TaskStatus.BLOCKED
    assert stopped.tdd_phase is TddPhase.RED_OBSERVED
    assert (tmp_path / "src" / "value.py").read_text() == "VALUE = 'broken'\n"


def test_pytest_execution_error_does_not_count_as_the_required_red_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a collector or timeout error advancing TDD as if a test assertion had failed."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    task = TaskState(
        description="Repair a test",
        mode=TaskMode.BUGFIX,
        config=HarnessConfig(
            source_dirs=(Path("src"),),
            test_dirs=(Path("tests"),),
            pytest_command=(sys.executable, "-c", "raise SystemExit(3)"),
        ),
    )
    llm = ScriptedLLM(
        [
            _action(kind="run_pytest", summary="try configured suite", targets=[]),
            _action(kind="finish", summary="cannot proceed", status="blocked"),
        ]
    )

    TaskOrchestrator(tmp_path, llm).run(task)

    assert '"tdd_phase": "test_design"' in llm.contexts[1]
