"""Offline behavioral coverage for the governed agent loop."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import sys
from uuid import uuid4

import pytest

import guardedpy.orchestrator as orchestrator_module
from guardedpy.actions import RunCommandAction
from guardedpy.command_rules import CommandRuleStore
from guardedpy.config import HarnessConfig
from guardedpy.domain import ApprovalDecision, TaskMode, TaskState, TaskStatus, TddPhase
from guardedpy.events import EventStore, RunEvent, StopReason
from guardedpy.feedback import PytestRun
from guardedpy.llm import ScriptedLLM
from guardedpy.memory import MemoryStore
from guardedpy.orchestrator import TaskOrchestrator
from guardedpy.workspace import ToolResult, Workspace


def _config() -> HarnessConfig:
    return HarnessConfig(
        source_dirs=(Path("src"),),
        test_dirs=(Path("tests"),),
        pytest_command=(sys.executable, "-m", "pytest", "-q"),
    )


def _action(**payload: object) -> str:
    return json.dumps(payload)


def _bugfix_task() -> TaskState:
    return TaskState(
        description="Repair the selected failure",
        mode=TaskMode.BUGFIX,
        bugfix_target="tests/test_value.py::test_value_is_fixed",
        config=_config(),
    )


def _feature_task() -> TaskState:
    return TaskState(description="Add a created module", mode=TaskMode.FEATURE, config=_config())


def test_submit_registers_a_pending_task_before_its_background_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches WebUI admission bypassing the core's one-active-task registry."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    task = _bugfix_task()
    orchestrator = TaskOrchestrator(tmp_path, ScriptedLLM([]))

    submitted = orchestrator.submit(task)

    assert submitted is task
    with pytest.raises(ValueError, match="another task is already active"):
        orchestrator.submit(_bugfix_task())


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
            _action(kind="run_pytest", summary="observe failure", targets=[]),
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


def test_terminal_memory_proposal_enters_queue_but_not_persistent_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a model proposal being persisted or ignored before human approval."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    task = _bugfix_task()
    memory_store = MemoryStore(tmp_path)
    sentinel = "SENTINEL-PRIVATE-PROPOSAL"
    proposal = _action(kind="propose_memory", summary="save parser note", text=sentinel)
    finish = _action(kind="finish", summary="stop", status="blocked")
    llm = ScriptedLLM([proposal, finish])

    result = TaskOrchestrator(tmp_path, llm, memory_store=memory_store).run(task)

    assert result.status is TaskStatus.BLOCKED
    assert [entry.text for entry in memory_store.proposals()] == [sentinel]
    assert MemoryStore(tmp_path).search(sentinel) == []
    assert sentinel not in llm.contexts[1]
    assert sentinel not in repr(EventStore(tmp_path).events_for(task.id))


def test_orchestrator_records_a_normalized_created_test_before_source_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a raw `src/../tests` creation being lost after the workspace write."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_existing.py").write_text(
        "def test_existing() -> None:\n    assert True\n"
    )
    create_test = """--- /dev/null
+++ b/src/../tests/test_created.py
@@ -0,0 +1,2 @@
+def test_created() -> None:
+    assert False
"""
    create_source = """--- /dev/null
+++ b/src/created.py
@@ -0,0 +1 @@
+VALUE = 'created'
"""
    llm = ScriptedLLM(
        [
            _action(kind="run_pytest", summary="establish baseline", targets=[]),
            _action(kind="apply_patch", summary="create test", diff=create_test),
            _action(kind="run_pytest", summary="observe red", targets=["tests/test_created.py"]),
            _action(kind="apply_patch", summary="create source", diff=create_source),
            _action(kind="finish", summary="stop", status="blocked"),
        ]
    )

    stopped = TaskOrchestrator(tmp_path, llm).run(_feature_task())

    assert stopped.status is TaskStatus.BLOCKED
    assert (tmp_path / "src" / "created.py").read_text() == "VALUE = 'created'\n"


def test_source_spelled_through_tests_parent_does_not_register_as_a_test_or_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches raw `tests/../src` being registered as a test after it is written."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_existing.py").write_text(
        "def test_existing() -> None:\n    assert True\n"
    )
    create_test = """--- /dev/null
+++ b/tests/test_created.py
@@ -0,0 +1,2 @@
+def test_created() -> None:
+    assert False
"""
    create_source = """--- /dev/null
+++ b/tests/../src/created.py
@@ -0,0 +1 @@
+VALUE = 'created'
"""
    task = _feature_task()
    llm = ScriptedLLM(
        [
            _action(kind="run_pytest", summary="establish baseline", targets=[]),
            _action(kind="apply_patch", summary="create test", diff=create_test),
            _action(kind="run_pytest", summary="observe red", targets=["tests/test_created.py"]),
            _action(kind="apply_patch", summary="create source", diff=create_source),
            _action(kind="finish", summary="stop", status="blocked"),
        ]
    )

    stopped = TaskOrchestrator(tmp_path, llm).run(task)

    assert stopped.status is TaskStatus.BLOCKED
    assert (tmp_path / "src" / "created.py").read_text() == "VALUE = 'created'\n"
    assert EventStore(tmp_path).events_for(task.id)[-1].stop_reason is StopReason.BLOCKED


def test_failed_workspace_test_creation_does_not_register_a_feature_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a failed atomic create supplying feature red evidence anyway."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_existing.py").write_text(
        "def test_existing() -> None:\n    assert True\n"
    )
    create_test = """--- /dev/null
+++ b/tests/test_created.py
@@ -0,0 +1,2 @@
+def test_created() -> None:
+    assert False
"""

    monkeypatch.setattr(
        Workspace,
        "apply_patch",
        lambda *_args: ToolResult(False, "Patch rejected", {"reason": "invalid_patch"}),
    )
    llm = ScriptedLLM(
        [
            _action(kind="run_pytest", summary="establish baseline", targets=[]),
            _action(kind="apply_patch", summary="create test", diff=create_test),
            _action(kind="finish", summary="stop", status="blocked"),
        ]
    )
    task = _feature_task()
    orchestrator = TaskOrchestrator(tmp_path, llm)

    orchestrator.run(task)

    assert not (tmp_path / "tests" / "test_created.py").exists()
    assert "tests/test_created.py" not in orchestrator._policy._new_test_paths.get(task.id, set())


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


def test_cancelled_task_does_not_dispatch_action_returned_after_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an action returned by the LLM after cancellation reaching the workspace."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    task = _bugfix_task()
    workspace_calls: list[PurePosixPath] = []
    action = _action(kind="list_files", summary="list project", path=".")

    class BlockingLLM:
        def complete(self, _context: object) -> str:
            orchestrator.cancel(task.id)
            return action

    def list_files(_self: Workspace, path: PurePosixPath) -> object:
        workspace_calls.append(path)
        raise AssertionError("cancelled action reached workspace")

    monkeypatch.setattr(Workspace, "list_files", list_files)
    orchestrator = TaskOrchestrator(tmp_path, BlockingLLM())
    orchestrator.submit(task)

    stopped = orchestrator.run(task)

    assert stopped.status is TaskStatus.CANCELLED
    assert workspace_calls == []
    assert EventStore(tmp_path).events_for(task.id)[-1].stop_reason is StopReason.CANCELLED


def test_cancelled_task_does_not_dispatch_an_approved_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches approval consuming cancellation and dispatching its previously allowed action."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    target = tmp_path / "obsolete.txt"
    target.write_text("preserve when cancellation wins\n")
    task = _bugfix_task()
    delete = _action(kind="delete_path", summary="delete obsolete file", path="obsolete.txt")
    orchestrator = TaskOrchestrator(tmp_path, ScriptedLLM([delete]))

    waiting = orchestrator.run(task)
    action_hash = EventStore(tmp_path).events_for(task.id)[-1].action_hash
    original_apply_approval = orchestrator._policy.apply_approval

    def cancel_after_approval(*args: object, **kwargs: object) -> object:
        result = original_apply_approval(*args, **kwargs)
        orchestrator.cancel(task.id)
        return result

    monkeypatch.setattr(orchestrator._policy, "apply_approval", cancel_after_approval)

    approved = orchestrator.resolve_approval(task.id, action_hash or "", approved=True)

    assert approved is False
    assert task.status is TaskStatus.CANCELLED
    assert target.exists()
    assert EventStore(tmp_path).events_for(task.id)[-1].stop_reason is StopReason.CANCELLED


def test_cancellation_during_action_parsing_remains_terminal_before_approval_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches parsing-time cancellation being overwritten by waiting-for-approval state."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    task = _bugfix_task()
    action = _action(kind="delete_path", summary="delete obsolete file", path="obsolete.txt")
    orchestrator = TaskOrchestrator(tmp_path, ScriptedLLM([action]))
    original_parse = orchestrator_module.parse_action

    def cancel_then_parse(payload: str) -> object:
        orchestrator.cancel(task.id)
        return original_parse(payload)

    monkeypatch.setattr(orchestrator_module, "parse_action", cancel_then_parse)
    orchestrator.submit(task)

    stopped = orchestrator.run(task)

    assert stopped.status is TaskStatus.CANCELLED
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
    audit = repr(EventStore(tmp_path).events_for(task.id))
    assert "Path: obsolete.txt" in audit
    assert "delete obsolete file" not in audit
    assert '"kind":"delete_path"' not in audit


def test_explicit_approval_request_pauses_with_safe_nonpermanent_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches explicit HITL being treated as ordinary feedback and exposing model reason."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    sentinel = "SENTINEL-MODEL-APPROVAL-REASON"
    request = _action(
        kind="request_approval",
        summary="ask for judgment",
        reason=sentinel,
    )
    llm = ScriptedLLM([request, _action(kind="finish", summary="stop", status="blocked")])
    task = _bugfix_task()

    waiting = TaskOrchestrator(tmp_path, llm).run(task)
    pending = EventStore(tmp_path).events_for(task.id)[-1]

    assert waiting.status is TaskStatus.WAITING_APPROVAL
    assert len(llm.contexts) == 1
    assert pending.action_projection == "Approval request"
    assert pending.permanent_eligible is False
    assert sentinel not in repr(pending)


def test_rejecting_explicit_approval_request_blocks_without_resuming_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches rejection consuming the pause without producing a blocked terminal state."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    llm = ScriptedLLM(
        [
            _action(kind="request_approval", summary="ask", reason="human judgment needed"),
            _action(kind="finish", summary="must not resume", status="blocked"),
        ]
    )
    task = _bugfix_task()
    orchestrator = TaskOrchestrator(tmp_path, llm)

    waiting = orchestrator.run(task)
    assert waiting.status is TaskStatus.WAITING_APPROVAL
    action_hash = EventStore(tmp_path).events_for(task.id)[-1].action_hash

    accepted = orchestrator.resolve_approval(task.id, action_hash or "", decision="reject")

    assert accepted is False
    assert task.status is TaskStatus.BLOCKED
    assert len(llm.contexts) == 1
    assert EventStore(tmp_path).events_for(task.id)[-1].approval_granted is False


def test_allow_once_on_explicit_approval_request_resumes_without_tool_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches consent executing a phantom tool or failing to resume the next model round."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    workspace_calls: list[str] = []
    monkeypatch.setattr(
        Workspace,
        "list_files",
        lambda *_args: workspace_calls.append("list_files"),
    )
    llm = ScriptedLLM(
        [
            _action(kind="request_approval", summary="ask", reason="human judgment needed"),
            _action(kind="finish", summary="stop after consent", status="blocked"),
        ]
    )
    task = _bugfix_task()
    orchestrator = TaskOrchestrator(tmp_path, llm)

    waiting = orchestrator.run(task)
    assert waiting.status is TaskStatus.WAITING_APPROVAL
    action_hash = EventStore(tmp_path).events_for(task.id)[-1].action_hash

    accepted = orchestrator.resolve_approval(task.id, action_hash or "", decision="once")
    resumed = orchestrator.run(task)

    assert accepted is True
    assert resumed.status is TaskStatus.BLOCKED
    assert len(llm.contexts) == 2
    assert workspace_calls == []


def test_forged_always_for_explicit_approval_request_keeps_exact_pending_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches invalid permanent consent consuming a non-command approval request."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    task = _bugfix_task()
    orchestrator = TaskOrchestrator(
        tmp_path,
        ScriptedLLM(
            [_action(kind="request_approval", summary="ask", reason="human judgment needed")]
        ),
    )

    waiting = orchestrator.run(task)
    assert waiting.status is TaskStatus.WAITING_APPROVAL
    action_hash = EventStore(tmp_path).events_for(task.id)[-1].action_hash

    forged = orchestrator.resolve_approval(task.id, action_hash or "", decision="always")

    assert forged is False
    assert task.status is TaskStatus.WAITING_APPROVAL
    assert orchestrator.resolve_approval(task.id, action_hash or "", decision="once") is True
    assert task.status is TaskStatus.RUNNING


def test_once_approval_is_consumed_without_creating_a_persistent_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a one-time command decision silently becoming reusable permission."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    action = _action(
        kind="run_command",
        summary="install approved package once",
        args=["python", "-m", "pip", "install", "ruff"],
    )
    monkeypatch.setattr(
        TaskOrchestrator,
        "_run_command",
        lambda _self, _action: ToolResult(True, "simulated command", {}),
    )
    task = _bugfix_task()
    orchestrator = TaskOrchestrator(tmp_path, ScriptedLLM([action]))

    waiting = orchestrator.run(task)
    assert waiting.status is TaskStatus.WAITING_APPROVAL
    action_hash = EventStore(tmp_path).events_for(task.id)[-1].action_hash
    accepted = orchestrator.resolve_approval(task.id, action_hash or "", decision="once")
    replayed = orchestrator.resolve_approval(task.id, action_hash or "", decision="once")

    assert accepted is True
    assert replayed is False
    assert task.status is TaskStatus.RUNNING
    assert CommandRuleStore(tmp_path).list_rules() == []


@pytest.mark.parametrize("approval_decision", ("once", "always"))
def test_branch_change_before_decision_keeps_push_pending_and_unexecuted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approval_decision: ApprovalDecision,
) -> None:
    """Catches a stale branch-bound approval being consumed or dispatched."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    branch = "main"
    action = _action(
        kind="run_command",
        summary="push current branch",
        args=["git", "push", "origin", "main"],
    )
    executions: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        TaskOrchestrator,
        "_run_command",
        lambda _self, command: executions.append(command.args)
        or ToolResult(True, "simulated command", {}),
    )
    task = _bugfix_task()
    orchestrator = TaskOrchestrator(
        tmp_path,
        ScriptedLLM([action]),
        current_branch_provider=lambda: branch,
    )

    waiting = orchestrator.run(task)
    action_hash = EventStore(tmp_path).events_for(task.id)[-1].action_hash
    branch = "feature"
    accepted = orchestrator.resolve_approval(
        task.id,
        action_hash or "",
        decision=approval_decision,
    )

    assert accepted is False
    assert waiting.status is TaskStatus.WAITING_APPROVAL
    assert executions == []


def test_approval_registration_uses_the_original_single_fresh_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches policy and orchestrator registering different live-branch decisions."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    branch = "main"
    reads: list[str] = []

    def current_branch() -> str:
        nonlocal branch
        value = branch
        reads.append(value)
        if len(reads) == 1:
            branch = "feature"
        return value

    action = _action(
        kind="run_command",
        summary="push current branch",
        args=["git", "push", "origin", "main"],
    )
    executions: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        TaskOrchestrator,
        "_run_command",
        lambda _self, command: executions.append(command.args)
        or ToolResult(True, "simulated command", {}),
    )
    task = _bugfix_task()
    orchestrator = TaskOrchestrator(
        tmp_path,
        ScriptedLLM([action]),
        current_branch_provider=current_branch,
    )

    waiting = orchestrator.run(task)
    action_hash = EventStore(tmp_path).events_for(task.id)[-1].action_hash
    stale = orchestrator.resolve_approval(
        task.id,
        action_hash or "",
        decision="once",
    )
    branch = "main"
    retried = orchestrator.resolve_approval(
        task.id,
        action_hash or "",
        decision="once",
    )

    assert waiting.status is TaskStatus.RUNNING
    assert stale is False
    assert retried is True
    assert executions == [("git", "push", "origin", "main")]


@pytest.mark.parametrize("approval_decision", ("once", "always"))
def test_branch_is_revalidated_immediately_before_approved_push_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approval_decision: ApprovalDecision,
) -> None:
    """Catches a branch change between approval validation and tool dispatch."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    branch_reads = 0

    def current_branch() -> str:
        nonlocal branch_reads
        branch_reads += 1
        return "feature" if branch_reads >= 3 else "main"

    action = _action(
        kind="run_command",
        summary="push current branch",
        args=["git", "push", "origin", "main"],
    )
    executions: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        TaskOrchestrator,
        "_run_command",
        lambda _self, command: executions.append(command.args)
        or ToolResult(True, "simulated command", {}),
    )
    task = _bugfix_task()
    orchestrator = TaskOrchestrator(
        tmp_path,
        ScriptedLLM([action]),
        current_branch_provider=current_branch,
    )

    waiting = orchestrator.run(task)
    action_hash = EventStore(tmp_path).events_for(task.id)[-1].action_hash
    accepted = orchestrator.resolve_approval(
        task.id,
        action_hash or "",
        decision=approval_decision,
    )

    assert accepted is False
    assert task.status is TaskStatus.BLOCKED
    assert executions == []
    assert CommandRuleStore(tmp_path).list_rules() == []
    events = EventStore(tmp_path).events_for(task.id)
    assert not any(event.approval_granted is True for event in events)
    assert events[-1].approval_granted is False


@pytest.mark.parametrize(
    "approval_kwargs",
    [
        {"decision": "invalid"},
        {"decision": True},
        {"approved": "yes"},
    ],
)
def test_invalid_approval_input_does_not_execute_and_pending_remains_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approval_kwargs: dict[str, object],
) -> None:
    """Catches orchestration popping or executing pending work before validating input."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    target = tmp_path / "obsolete.txt"
    target.write_text("preserve until a valid decision\n")
    task = _bugfix_task()
    action = _action(kind="delete_path", summary="delete obsolete file", path="obsolete.txt")
    orchestrator = TaskOrchestrator(tmp_path, ScriptedLLM([action]))

    waiting = orchestrator.run(task)
    action_hash = EventStore(tmp_path).events_for(task.id)[-1].action_hash
    invalid = orchestrator.resolve_approval(
        task.id,
        action_hash or "",
        **approval_kwargs,  # type: ignore[arg-type]
    )

    assert invalid is False
    assert waiting.status is TaskStatus.WAITING_APPROVAL
    assert target.exists()
    assert orchestrator.resolve_approval(task.id, action_hash or "", decision="once") is True
    assert target.exists() is False


def test_always_approval_survives_store_restart_without_raw_pending_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches permanent approval failing to derive durable constrained permission."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    pending = RunCommandAction(
        kind="run_command",
        summary="do not persist this pending action",
        args=("python", "-m", "pip", "install", "ruff"),
    )
    monkeypatch.setattr(
        TaskOrchestrator,
        "_run_command",
        lambda _self, _action: ToolResult(True, "simulated command", {}),
    )
    task = _bugfix_task()
    orchestrator = TaskOrchestrator(tmp_path, ScriptedLLM([pending.model_dump_json()]))

    orchestrator.run(task)
    action_hash = EventStore(tmp_path).events_for(task.id)[-1].action_hash

    assert orchestrator.resolve_approval(task.id, action_hash or "", decision="always") is True
    assert CommandRuleStore(tmp_path).matches(pending, None) is True


def test_external_rule_revocation_is_seen_by_the_same_long_lived_orchestrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an orchestrator retaining a stale in-memory rule after WebUI revocation."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    command = RunCommandAction(
        kind="run_command",
        summary="safe whitespace check",
        args=("git", "diff", "--no-ext-diff", "--check"),
    )
    store = CommandRuleStore(tmp_path)
    rule = store.add_from(command, None)
    finish = _action(kind="finish", summary="stop first task", status="blocked")
    llm = ScriptedLLM([command.model_dump_json(), finish, command.model_dump_json()])
    monkeypatch.setattr(
        TaskOrchestrator,
        "_run_command",
        lambda _self, _action: ToolResult(True, "simulated command", {}),
    )
    orchestrator = TaskOrchestrator(tmp_path, llm, command_rules=store)

    first = orchestrator.run(_bugfix_task())
    assert first.status is TaskStatus.BLOCKED
    assert CommandRuleStore(tmp_path).delete(rule.id) is True

    second = orchestrator.run(_bugfix_task())

    assert second.status is TaskStatus.WAITING_APPROVAL
    pending = EventStore(tmp_path).events_for(second.id)[-1]
    assert pending.policy_rule_id == "command.read_only_approval_required"


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
    task.bugfix_target = "tests/test_value.py::test_stays_red"
    llm = ScriptedLLM(
        [
            _action(kind="run_pytest", summary="observe red", targets=[]),
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
        bugfix_target="tests/test_value.py::test_value_is_fixed",
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


def test_pytest_audit_discards_failed_token_outside_configured_test_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches arbitrary bounded stdout tokens being persisted as pytest node IDs."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    sentinel = "sk-fake-feedback-token"
    monkeypatch.setattr(
        Workspace,
        "run_pytest",
        lambda *_args: PytestRun(
            1,
            f"FAILED {sentinel} - AssertionError\nE   AssertionError\n",
            "",
            False,
        ),
    )
    task = _bugfix_task()
    llm = ScriptedLLM(
        [
            _action(kind="run_pytest", summary="run configured suite", targets=[]),
            _action(kind="finish", summary="stop", status="blocked"),
        ]
    )

    TaskOrchestrator(tmp_path, llm).run(task)

    feedback_event = next(
        event
        for event in EventStore(tmp_path).events_for(task.id)
        if event.feedback_kind is not None
    )
    assert feedback_event.feedback_node_id is None
    assert sentinel not in repr(feedback_event)


def test_pytest_audit_discards_untrusted_suffix_from_rooted_test_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches parametrized or forged node suffix text entering persistent audit data."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_value.py").write_text(
        "def test_value() -> None:\n    assert True\n"
    )
    sentinel = "sk-fake-node-suffix"
    monkeypatch.setattr(
        Workspace,
        "run_pytest",
        lambda *_args: PytestRun(
            1,
            f"FAILED tests/test_value.py::{sentinel} - AssertionError\n"
            "E   AssertionError\n",
            "",
            False,
        ),
    )
    task = _bugfix_task()
    llm = ScriptedLLM(
        [
            _action(kind="run_pytest", summary="run configured suite", targets=[]),
            _action(kind="finish", summary="stop", status="blocked"),
        ]
    )

    TaskOrchestrator(tmp_path, llm).run(task)

    feedback_event = next(
        event
        for event in EventStore(tmp_path).events_for(task.id)
        if event.feedback_kind is not None
    )
    assert feedback_event.feedback_node_id == "tests/test_value.py"
    assert sentinel not in repr(feedback_event)


def test_pytest_audit_discards_nonexistent_path_inside_configured_test_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a forged root-looking FAILED path being treated as a real test file."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    sentinel = "tests/sk-fake-test-path.py"
    monkeypatch.setattr(
        Workspace,
        "run_pytest",
        lambda *_args: PytestRun(
            1,
            f"FAILED {sentinel} - AssertionError\nE   AssertionError\n",
            "",
            False,
        ),
    )
    task = _bugfix_task()

    TaskOrchestrator(
        tmp_path,
        ScriptedLLM(
            [
                _action(kind="run_pytest", summary="run configured suite", targets=[]),
                _action(kind="finish", summary="stop", status="blocked"),
            ]
        ),
    ).run(task)

    feedback_event = next(
        event
        for event in EventStore(tmp_path).events_for(task.id)
        if event.feedback_kind is not None
    )
    assert feedback_event.feedback_node_id is None
    assert sentinel not in repr(feedback_event)


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
    (tmp_path / "tests" / "test_still_green.py").write_text(
        "def test_still_green() -> None:\n    assert True\n"
    )
    target = tmp_path / "obsolete.txt"
    target.write_text("remove after approval\n")
    task = _bugfix_task()
    task.tdd_phase = TddPhase.IMPLEMENTATION
    delete = _action(kind="delete_path", summary="delete obsolete file", path="obsolete.txt")
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
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "value.py").write_text("VALUE = 'broken'\n")
    (tmp_path / "tests" / "test_value.py").write_text(
        "from pathlib import Path\n\n"
        "def test_value_is_fixed() -> None:\n"
        "    assert Path('src/value.py').read_text() == \"VALUE = 'fixed'\\n\"\n"
    )

    def raise_patch(*_args: object, **_kwargs: object) -> object:
        raise OSError("patch failed")

    monkeypatch.setattr(Workspace, "apply_patch", raise_patch)
    patch = _action(
        kind="apply_patch",
        summary="repair value",
        diff="--- a/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n-VALUE = 'broken'\n+VALUE = 'fixed'\n",
    )
    task = _bugfix_task()

    stopped = TaskOrchestrator(
        tmp_path,
        ScriptedLLM(
            [
                _action(kind="run_pytest", summary="establish broken baseline", targets=[]),
                _action(kind="read_file", summary="inspect value", path="src/value.py"),
                patch,
            ]
        ),
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
