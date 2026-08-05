"""Contracts for the framework-independent local runtime and execution lease."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from guardedpy.actions import RunCommandAction
from guardedpy.command_rules import CommandRuleStore
from guardedpy.config import HarnessConfig
from guardedpy.credentials import CredentialStatus
from guardedpy.domain import TaskMode, TaskStatus
from guardedpy.lease import ExecutionLease
from guardedpy.llm import ScriptedLLM
from guardedpy.orchestrator import TaskOrchestrator
from guardedpy.runtime import LocalRuntime, RuntimeBusyError, RuntimeServices


@dataclass
class FakeCredentials:
    configured: bool = False
    keys: list[str] = field(default_factory=list)

    def status(self) -> CredentialStatus:
        return CredentialStatus(configured=self.configured)

    def set_key(self, key: str) -> None:
        self.keys.append(key)
        self.configured = True

    def clear_key(self) -> None:
        self.configured = False


def _config() -> HarnessConfig:
    return HarnessConfig(
        source_dirs=(Path("src"),),
        test_dirs=(Path("tests"),),
        pytest_command=("pytest",),
    )


def _runtime(project_root: Path, responses: list[list[str]] | None = None) -> LocalRuntime:
    scripts = iter(responses or [[]])

    def factory(root: Path, config: HarnessConfig, memory: object) -> TaskOrchestrator:
        del config
        return TaskOrchestrator(root, ScriptedLLM(next(scripts)), memory_store=memory)

    return LocalRuntime(
        RuntimeServices(credentials=FakeCredentials(), orchestrator_factory=factory)
    )


@pytest.fixture(autouse=True)
def _isolated_state_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()


def test_execution_lease_releases_the_nonblocking_file_lock(tmp_path: Path) -> None:
    first = ExecutionLease(tmp_path)
    second = ExecutionLease(tmp_path)

    assert first.try_acquire() is True
    assert second.try_acquire() is False

    first.release()

    assert second.try_acquire() is True
    second.release()


def test_second_runtime_cannot_create_a_task_while_first_runtime_holds_active_lease(
    tmp_path: Path,
) -> None:
    first = _runtime(tmp_path)
    second = _runtime(tmp_path)
    first.setup(tmp_path, _config(), api_key=None)
    second.setup(tmp_path, _config(), api_key=None)

    task = first.create_task("repair value", TaskMode.BUGFIX, "tests/test_value.py::test_value")

    with pytest.raises(RuntimeBusyError, match="已有任务正在运行"):
        second.create_task("another task", TaskMode.FEATURE, None)

    assert first.task(task.id) is task
    assert [stored.id for stored in second.tasks()] == [task.id]


def test_terminal_run_releases_task_lease_for_another_runtime(tmp_path: Path) -> None:
    finished = '{"kind":"finish","summary":"PRIVATE","status":"completed"}'
    first = _runtime(tmp_path, [[finished]])
    second = _runtime(tmp_path)
    first.setup(tmp_path, _config(), api_key=None)
    second.setup(tmp_path, _config(), api_key=None)
    task = first.create_task("finish", TaskMode.FEATURE, None)

    assert first.run(task.id).status is TaskStatus.BLOCKED
    assert second.create_task("next task", TaskMode.FEATURE, None).status is TaskStatus.PENDING


def test_runtime_returns_event_store_safe_projection_without_pending_action_body(
    tmp_path: Path,
) -> None:
    runtime = _runtime(
        tmp_path,
        [['{"kind":"finish","summary":"PRIVATE","status":"blocked"}']],
    )
    runtime.setup(tmp_path, _config(), api_key=None)
    task = runtime.create_task("inspect", TaskMode.FEATURE, None)

    runtime.run(task.id)

    assert all("PRIVATE" not in event.model_dump_json() for event in runtime.events(task.id))


def test_runtime_exposes_memory_rule_and_nonsecret_credential_operations(tmp_path: Path) -> None:
    proposal = '{"kind":"propose_memory","summary":"remember","text":"Use focused tests"}'
    finish = '{"kind":"finish","summary":"stop","status":"completed"}'
    runtime = _runtime(tmp_path, [[proposal, finish]])
    runtime.setup(tmp_path, _config(), api_key=None)
    task = runtime.create_task("remember a convention", TaskMode.FEATURE, None)

    runtime.run(task.id)
    memory = runtime.memory_proposals()[0]

    assert runtime.approve_memory(memory.id).id == memory.id
    assert [entry.id for entry in runtime.memories()] == [memory.id]
    runtime.delete_memory(memory.id)
    assert runtime.memories() == []

    rule = CommandRuleStore(tmp_path).add_from(
        RunCommandAction(
            kind="run_command",
            summary="safe check",
            args=("git", "diff", "--no-ext-diff", "--check"),
        ),
        current_branch=None,
    )
    assert [item.id for item in runtime.command_rules()] == [rule.id]
    assert runtime.delete_command_rule(rule.id) is True
    assert runtime.command_rules() == []

    assert runtime.credential_status().configured is False
    runtime.update_credential("fake-test-key")
    assert runtime.credential_status().configured is True
    runtime.clear_credential()
    assert runtime.credential_status().configured is False
