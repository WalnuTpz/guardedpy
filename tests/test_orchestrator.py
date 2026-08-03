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
from guardedpy.memory import MemoryStore
from guardedpy.orchestrator import TaskOrchestrator
from guardedpy.workspace import Workspace


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


def test_orchestrator_injects_relevant_approved_memories_into_the_llm_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches approved memories being accepted by a builder but never reaching a real loop call."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    task = _bugfix_task()
    memories = MemoryStore(tmp_path)
    for number in range(6):
        memories.approve(memories.propose(task.id, f"repair selected failure memory {number}").id)
    llm = ScriptedLLM([_action(kind="finish", summary="stop", status="blocked")])

    TaskOrchestrator(tmp_path, llm, memory_store=memories).run(task)

    context = json.loads(llm.contexts[0])
    approved_memories = context["context"]["approved_memories"]
    assert len(approved_memories) == 5
    assert "repair selected failure memory 0" not in approved_memories
    assert all(memory.startswith("repair selected failure memory") for memory in approved_memories)


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


def test_approved_action_consumes_its_original_round_budget_before_resuming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a resumed task receiving a fresh round budget after approval."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    target = tmp_path / "obsolete.txt"
    target.write_text("remove only once\n")
    delete = _action(kind="delete_path", summary="delete obsolete file", path="obsolete.txt")
    llm = ScriptedLLM([delete, _action(kind="finish", summary="should not run", status="blocked")])
    task = _bugfix_task()
    orchestrator = TaskOrchestrator(tmp_path, llm, max_rounds=1)

    waiting = orchestrator.run(task)
    action_hash = EventStore(tmp_path).events_for(task.id)[-1].action_hash
    assert waiting.status is TaskStatus.WAITING_APPROVAL
    assert orchestrator.resolve_approval(task.id, action_hash or "", approved=True) is True

    resumed = orchestrator.run(task)

    assert resumed.status is TaskStatus.BLOCKED
    assert EventStore(tmp_path).events_for(task.id)[-1].stop_reason is StopReason.ROUND_LIMIT
    assert len(llm.contexts) == 1
    assert target.exists() is False


def test_resumed_task_keeps_repeat_history_across_waiting_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches approval resume resetting the repeated-operation detector."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    target = tmp_path / "obsolete.txt"
    target.write_text("remove only once\n")
    first = _action(kind="delete_path", summary="delete obsolete file", path="obsolete.txt")
    second = _action(kind="delete_path", summary="wording changed", path="obsolete.txt")
    task = _bugfix_task()
    orchestrator = TaskOrchestrator(tmp_path, ScriptedLLM([first, second]), max_rounds=2)

    waiting = orchestrator.run(task)
    action_hash = EventStore(tmp_path).events_for(task.id)[-1].action_hash
    assert waiting.status is TaskStatus.WAITING_APPROVAL
    assert orchestrator.resolve_approval(task.id, action_hash or "", approved=True) is True

    stopped = orchestrator.run(task)

    assert stopped.status is TaskStatus.BLOCKED
    assert EventStore(tmp_path).events_for(task.id)[-1].stop_reason is StopReason.REPEATED_ACTION
    assert target.exists() is False


def test_successful_approved_delete_invalidates_full_suite_green_before_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a delete succeeding after green while completed finish remains permitted."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "tests").mkdir()
    target = tmp_path / "tests" / "test_obsolete.py"
    target.write_text("def test_still_green() -> None:\n    assert True\n")
    task = _bugfix_task()
    task.tdd_phase = TddPhase.IMPLEMENTATION
    delete = _action(kind="delete_path", summary="delete obsolete test", path="tests/test_obsolete.py")
    llm = ScriptedLLM(
        [
            _action(kind="run_pytest", summary="run full suite", targets=[]),
            delete,
            _action(kind="finish", summary="finish after delete", status="completed"),
            _action(kind="finish", summary="report blocked", status="blocked"),
        ]
    )
    orchestrator = TaskOrchestrator(tmp_path, llm)

    waiting = orchestrator.run(task)
    action_hash = EventStore(tmp_path).events_for(task.id)[-1].action_hash
    assert waiting.status is TaskStatus.WAITING_APPROVAL
    assert orchestrator.resolve_approval(task.id, action_hash or "", approved=True) is True

    stopped = orchestrator.run(task)

    assert stopped.status is TaskStatus.BLOCKED
    assert EventStore(tmp_path).events_for(task.id)[-1].stop_reason is StopReason.BLOCKED
    assert target.exists() is False


@pytest.mark.parametrize(
    ("method", "response"),
    [
        ("list_files", _action(kind="list_files", summary="list project", path=".")),
        ("read_file", _action(kind="read_file", summary="read note", path="note.txt")),
        ("run_pytest", _action(kind="run_pytest", summary="run tests", targets=[])),
    ],
)
def test_allowed_workspace_tool_exception_stops_and_audits_the_failed_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, response: str
) -> None:
    """Catches allowed tool errors escaping the loop or losing their audit action."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "note.txt").write_text("note\n")

    def raise_tool(*_args: object, **_kwargs: object) -> object:
        raise OSError("workspace failed")

    monkeypatch.setattr(Workspace, method, raise_tool)
    task = _bugfix_task()

    stopped = TaskOrchestrator(tmp_path, ScriptedLLM([response])).run(task)

    event = EventStore(tmp_path).events_for(task.id)[-1]
    assert stopped.status is TaskStatus.BLOCKED
    assert event.stop_reason is StopReason.UNRECOVERABLE_ERROR
    assert event.action_hash == EventStore(tmp_path).events_for(task.id)[1].action_hash


def test_allowed_patch_exception_stops_and_audits_the_failed_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches source-patch failures escaping after a permitted read and red test."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "value.py").write_text("VALUE = 'broken'\n")

    def raise_patch(*_args: object, **_kwargs: object) -> object:
        raise OSError("patch failed")

    monkeypatch.setattr(Workspace, "apply_patch", raise_patch)
    patch = _action(
        kind="apply_patch",
        summary="repair value",
        diff="--- a/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n-VALUE = 'broken'\n+VALUE = 'fixed'\n",
    )
    task = _bugfix_task()
    task.tdd_phase = TddPhase.RED_OBSERVED

    stopped = TaskOrchestrator(
        tmp_path,
        ScriptedLLM([_action(kind="read_file", summary="inspect value", path="src/value.py"), patch]),
    ).run(task)

    event = EventStore(tmp_path).events_for(task.id)[-1]
    assert stopped.status is TaskStatus.BLOCKED
    assert event.stop_reason is StopReason.UNRECOVERABLE_ERROR
    assert event.action_summary == "apply source patch"


def test_approved_delete_exception_stops_and_audits_the_failed_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an approval-path delete failure leaving a task running without a terminal audit."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    target = tmp_path / "obsolete.txt"
    target.write_text("remove only when approved\n")

    def raise_delete(*_args: object, **_kwargs: object) -> object:
        raise OSError("delete failed")

    monkeypatch.setattr(Workspace, "delete_path", raise_delete)
    task = _bugfix_task()
    orchestrator = TaskOrchestrator(
        tmp_path,
        ScriptedLLM([_action(kind="delete_path", summary="delete obsolete file", path="obsolete.txt")]),
    )

    waiting = orchestrator.run(task)
    action_hash = EventStore(tmp_path).events_for(task.id)[-1].action_hash
    assert waiting.status is TaskStatus.WAITING_APPROVAL
    assert orchestrator.resolve_approval(task.id, action_hash or "", approved=True) is True

    event = EventStore(tmp_path).events_for(task.id)[-1]
    assert task.status is TaskStatus.BLOCKED
    assert event.stop_reason is StopReason.UNRECOVERABLE_ERROR
    assert event.action_hash == action_hash
    assert target.exists()


def test_orchestrator_rejects_a_second_task_while_first_waits_for_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a second task entering the loop while a dangerous action awaits approval."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "obsolete.txt").write_text("remove only when approved\n")
    first = _bugfix_task()
    second = _bugfix_task()
    llm = ScriptedLLM(
        [
            _action(kind="delete_path", summary="delete obsolete file", path="obsolete.txt"),
            _action(kind="list_files", summary="second task action", path="."),
        ]
    )
    orchestrator = TaskOrchestrator(tmp_path, llm)

    waiting = orchestrator.run(first)

    assert waiting.status is TaskStatus.WAITING_APPROVAL
    with pytest.raises(ValueError, match="another task is already active"):
        orchestrator.run(second)
    assert len(llm.contexts) == 1
    assert EventStore(tmp_path).events_for(second.id) == []
