"""Automatic coding-path and deterministic read-only intent assurance."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

from guardedpy.actions import (
    ApplyPatchAction,
    DeletePathAction,
    FinishAction,
    ListFilesAction,
    ProposeMemoryAction,
    ReadFileAction,
    RequestApprovalAction,
    RunCommandAction,
    RunPytestAction,
)
from guardedpy.config import HarnessConfig
from guardedpy.context import LlmContext
from guardedpy.discovery import ProjectProfile
from guardedpy.domain import (
    PolicyVerdict,
    TaskIntent,
    TaskPath,
    TaskState,
    TaskStatus,
    TddPhase,
)
from guardedpy.orchestrator import TaskOrchestrator
from guardedpy.policy import PolicyEngine


class RecordingLLM:
    """Record the first core-built context while returning a fixed safe stop."""

    def __init__(self) -> None:
        self.contexts: list[LlmContext] = []

    def complete(self, context: LlmContext) -> str:
        self.contexts.append(context)
        return json.dumps({"kind": "finish", "summary": "stop", "status": "blocked"})


def _config(root: Path, *, timeout_seconds: int = 120) -> HarnessConfig:
    (root / "src").mkdir(exist_ok=True)
    (root / "tests").mkdir(exist_ok=True)
    return HarnessConfig(
        profile=ProjectProfile(
            root=root.resolve(),
            discovery_source="tests_dir",
            source_dirs=(Path("src"),),
            test_dirs=(Path("tests"),),
            pytest_command=(sys.executable, "-m", "pytest"),
        ),
        timeout_seconds=timeout_seconds,
    )


def _task(root: Path, *, intent: TaskIntent = TaskIntent.CODING) -> TaskState:
    return TaskState(description="Handle the project", intent=intent, config=_config(root))


def test_task_intent_is_immutable_after_creation(tmp_path: Path) -> None:
    """Catches a running task switching between coding and read-only governance."""
    task = _task(tmp_path)

    with pytest.raises(ValidationError, match="frozen"):
        task.intent = TaskIntent.PLAN


def test_passing_complete_baseline_selects_feature_before_first_llm_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches feature classification happening after the model can propose a mutation."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_green.py").write_text(
        "from pathlib import Path\n\n"
        "def test_green() -> None:\n"
        "    Path('baseline-ran').write_text('yes')\n"
        "    assert True\n"
    )
    llm = RecordingLLM()
    task = _task(tmp_path)

    result = TaskOrchestrator(tmp_path, llm).run(task)

    assert (tmp_path / "baseline-ran").read_text() == "yes"
    assert len(llm.contexts) == 1
    assert llm.contexts[0].trusted_data["task"]["path"] == "feature"
    assert result.path is TaskPath.FEATURE
    assert result.repair_targets == ()
    assert result.tdd_phase is TddPhase.FINISHED


def test_two_baseline_assertion_failures_create_repair_set_before_first_llm_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches repair classification retaining only one selected failure node."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_a.py").write_text("def test_a() -> None:\n    assert 1 == 2\n")
    (tests / "test_b.py").write_text("def test_b() -> None:\n    assert 'a' == 'b'\n")
    llm = RecordingLLM()
    task = _task(tmp_path)

    result = TaskOrchestrator(tmp_path, llm).run(task)

    expected = ("tests/test_a.py::test_a", "tests/test_b.py::test_b")
    assert len(llm.contexts) == 1
    assert llm.contexts[0].trusted_data["task"] == {
        "description": "Handle the project",
        "intent": "coding",
        "path": "repair",
        "repair_targets": expected,
        "review_path": None,
    }
    assert result.path is TaskPath.REPAIR
    assert result.repair_targets == expected


@pytest.mark.parametrize(
    ("test_source", "timeout_seconds"),
    [
        ("from missing_dependency import value\n", 120),
        ("import time\n\ndef test_wait() -> None:\n    time.sleep(10)\n", 5),
    ],
    ids=("collection-error", "timeout"),
)
def test_invalid_complete_baseline_blocks_without_calling_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    test_source: str,
    timeout_seconds: int,
) -> None:
    """Catches an invalid baseline being exposed to the model as mutation context."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_invalid.py").write_text(test_source)
    llm = RecordingLLM()
    task = TaskState(
        description="Handle invalid baseline",
        intent=TaskIntent.CODING,
        config=_config(tmp_path, timeout_seconds=timeout_seconds),
    )

    result = TaskOrchestrator(tmp_path, llm).run(task)

    assert result.status is TaskStatus.BLOCKED
    assert result.path is TaskPath.BASELINE_PENDING
    assert llm.contexts == []


def test_caller_supplied_coding_path_cannot_bypass_automatic_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches mutable/caller-supplied path state skipping the mandatory preflight."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_invalid.py").write_text("from missing_dependency import value\n")
    llm = RecordingLLM()
    task = TaskState(
        description="Cannot bypass baseline",
        intent=TaskIntent.CODING,
        config=_config(tmp_path),
        path=TaskPath.FEATURE,
    )
    task.path = TaskPath.FEATURE

    result = TaskOrchestrator(tmp_path, llm).run(task)

    assert result.status is TaskStatus.BLOCKED
    assert llm.contexts == []


def test_mixed_assertion_and_execution_failure_baseline_blocks_without_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches one assertion line turning a mixed invalid baseline into a repair set."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_mixed.py").write_text(
        "def test_runtime_error() -> None:\n    raise TypeError('broken runtime')\n\n"
        "def test_assertion() -> None:\n    assert 1 == 2\n"
    )
    llm = RecordingLLM()
    task = _task(tmp_path)

    result = TaskOrchestrator(tmp_path, llm).run(task)

    assert result.status is TaskStatus.BLOCKED
    assert result.repair_targets == ()
    assert llm.contexts == []


def test_review_intent_validates_and_retains_only_existing_project_path(
    tmp_path: Path,
) -> None:
    """Catches review tasks accepting missing/root-escaping paths or losing their scope."""
    config = _config(tmp_path)
    (tmp_path / "src" / "value.py").write_text("VALUE = 1\n")

    task = TaskState(
        description="Review value",
        intent=TaskIntent.REVIEW,
        review_path="src/value.py",
        config=config,
    )

    assert task.review_path == "src/value.py"
    with pytest.raises(ValidationError, match="review path"):
        TaskState(
            description="Escape",
            intent=TaskIntent.REVIEW,
            review_path="../outside.py",
            config=config,
        )
    with pytest.raises(ValidationError, match="review path"):
        TaskState(
            description="Missing",
            intent=TaskIntent.REVIEW,
            review_path="src/missing.py",
            config=config,
        )
    with pytest.raises(ValidationError, match="review path"):
        TaskState(
            description="Plan",
            intent=TaskIntent.PLAN,
            review_path="src/value.py",
            config=config,
        )


@pytest.mark.parametrize("intent", (TaskIntent.PLAN, TaskIntent.REVIEW))
def test_read_only_intents_allow_only_list_read_and_finish(
    tmp_path: Path, intent: TaskIntent
) -> None:
    """Catches plan or review tasks reaching any state-changing or command action."""
    task = _task(tmp_path, intent=intent)
    policy = PolicyEngine(tmp_path)
    source_diff = "--- a/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n-old\n+new\n"
    denied_actions = (
        ApplyPatchAction(kind="apply_patch", summary="patch", diff=source_diff),
        DeletePathAction(kind="delete_path", summary="delete", path="src/value.py"),
        RunPytestAction(kind="run_pytest", summary="test", targets=()),
        RunCommandAction(kind="run_command", summary="command", args=("git", "diff", "--check")),
        RequestApprovalAction(kind="request_approval", summary="approve", reason="write"),
        ProposeMemoryAction(kind="propose_memory", summary="remember", text="note"),
    )

    decisions = [policy.decide(task, action) for action in denied_actions]

    assert [(decision.verdict, decision.rule_id) for decision in decisions] == [
        (PolicyVerdict.DENY, "read_only.action_denied")
    ] * len(denied_actions)
    assert policy.decide(
        task, ListFilesAction(kind="list_files", summary="list", path=".")
    ).verdict is PolicyVerdict.ALLOW
    assert policy.decide(
        task, ReadFileAction(kind="read_file", summary="read", path="src/value.py")
    ).verdict is PolicyVerdict.ALLOW
    assert policy.decide(
        task, FinishAction(kind="finish", summary="done", status="completed")
    ).verdict is PolicyVerdict.ALLOW
