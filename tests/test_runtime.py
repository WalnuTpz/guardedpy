"""Contracts for the framework-independent local runtime and execution lease."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread

import pytest
from pydantic import ValidationError

import guardedpy.runtime as runtime_module
from guardedpy.actions import RunCommandAction
from guardedpy.command_rules import CommandRuleStore
from guardedpy.config import HarnessConfig, project_config_path
from guardedpy.credentials import CredentialStatus
from guardedpy.domain import TaskMode, TaskState, TaskStatus
from guardedpy.events import EventStore
from guardedpy.lease import ExecutionLease, GlobalStateLease
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


def _project_root(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    return root


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


def test_runtime_recovery_interrupts_restored_tasks_and_releases_the_active_gate(
    tmp_path: Path,
) -> None:
    """Catches restart recovery leaving persisted work active in the runtime cache."""
    first = _runtime(tmp_path)
    first.setup(tmp_path, _config(), api_key=None)
    task = first.create_task("interrupted", TaskMode.FEATURE, None)
    first._release_lease()

    recovered = _runtime(tmp_path)
    interrupted = recovered.recover_interrupted_tasks()

    assert interrupted == (task.id,)
    assert recovered.task(task.id).status is TaskStatus.INTERRUPTED
    assert recovered.create_task("next", TaskMode.FEATURE, None).status is TaskStatus.PENDING


def test_runtime_rejects_a_blank_task_description(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.setup(tmp_path, _config(), api_key=None)

    with pytest.raises(ValueError, match="description"):
        runtime.create_task("  ", TaskMode.FEATURE, None)


def test_runtime_registers_a_task_after_orchestrator_startup_recovery(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.setup(tmp_path, _config(), api_key=None)

    task = runtime.create_task("keep pending", TaskMode.FEATURE, None)

    assert EventStore(tmp_path).tasks()[0].id == task.id
    assert EventStore(tmp_path).tasks()[0].status is TaskStatus.PENDING
    assert EventStore(tmp_path).events_for(task.id) == []


def test_invalid_bugfix_target_releases_the_lease_before_another_runtime_mutates(
    tmp_path: Path,
) -> None:
    first = _runtime(tmp_path)
    second = _runtime(tmp_path)
    first.setup(tmp_path, _config(), api_key=None)
    second.setup(tmp_path, _config(), api_key=None)

    with pytest.raises(ValidationError):
        first.create_task("repair", TaskMode.BUGFIX, "  ")

    assert second.create_task("another task", TaskMode.FEATURE, None).status is TaskStatus.PENDING


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


def test_global_execution_lease_blocks_a_second_project_task_but_preserves_index_history(
    tmp_path: Path,
) -> None:
    """Catches project-scoped task admission allowing two live tasks at once."""
    first_root = _project_root(tmp_path, "first")
    second_root = _project_root(tmp_path, "second")
    first = _runtime(first_root)
    second = _runtime(second_root)
    first.setup(first_root, _config(), api_key=None)
    second.setup(second_root, _config(), api_key=None)

    first_task = first.create_task("first task", TaskMode.FEATURE, None)

    with pytest.raises(RuntimeBusyError):
        second.create_task("second task", TaskMode.FEATURE, None)

    reader = _runtime(second_root)
    assert [task.id for task in reader.tasks()] == [first_task.id]


def test_setup_rejects_a_second_project_while_a_global_task_is_active(tmp_path: Path) -> None:
    """Catches setup changing project state while another entrypoint owns live work."""
    first_root = _project_root(tmp_path, "first")
    second_root = _project_root(tmp_path, "second")
    first = _runtime(first_root)
    second = _runtime(second_root)
    first.setup(first_root, _config(), api_key=None)
    first.create_task("first task", TaskMode.FEATURE, None)

    with pytest.raises(RuntimeBusyError):
        second.setup(second_root, _config(), api_key=None)

    assert not project_config_path(second_root).exists()


@dataclass
class BlockingCredentials(FakeCredentials):
    first_entered: Event = field(default_factory=Event)
    release_first: Event = field(default_factory=Event)
    second_entered: Event = field(default_factory=Event)

    def set_key(self, key: str) -> None:
        if key == "first":
            self.first_entered.set()
            assert self.release_first.wait(1)
        if key == "second":
            self.second_entered.set()
        super().set_key(key)


def test_cross_project_credential_updates_serialize_the_shared_keyring(tmp_path: Path) -> None:
    first_root = _project_root(tmp_path, "first")
    second_root = _project_root(tmp_path, "second")
    credentials = BlockingCredentials()

    def factory(root: Path, config: HarnessConfig, memory: object) -> TaskOrchestrator:
        del root, config, memory
        raise AssertionError("credential update must not create an orchestrator")

    first = LocalRuntime(RuntimeServices(credentials=credentials, orchestrator_factory=factory))
    second = LocalRuntime(RuntimeServices(credentials=credentials, orchestrator_factory=factory))
    first.setup(first_root, _config(), api_key=None)
    second.setup(second_root, _config(), api_key=None)

    failures: list[RuntimeBusyError] = []

    def update_second() -> None:
        try:
            second.update_credential("second")
        except RuntimeBusyError as error:
            failures.append(error)

    first_thread = Thread(target=first.update_credential, args=("first",))
    second_thread = Thread(target=update_second)
    first_thread.start()
    assert credentials.first_entered.wait(1)
    second_thread.start()
    second_entered_while_first_was_running = credentials.second_entered.wait(0.2)
    credentials.release_first.set()
    first_thread.join()
    second_thread.join()

    assert second_entered_while_first_was_running is False
    assert credentials.keys == ["first"]
    assert len(failures) == 1


def test_setup_releases_its_project_lease_when_global_state_is_busy(tmp_path: Path) -> None:
    global_lease = GlobalStateLease()
    assert global_lease.try_acquire() is True
    first = _runtime(tmp_path)
    second = _runtime(tmp_path)

    with pytest.raises(RuntimeBusyError):
        first.setup(tmp_path, _config(), api_key=None)

    global_lease.release()
    second.setup(tmp_path, _config(), api_key=None)


@dataclass
class FailingSubmitOrchestrator:
    def submit(self, task: TaskState) -> TaskState:
        del task
        raise RuntimeError("submit failed")

    def run(self, task: TaskState) -> TaskState:
        return task

    def cancel(self, task_id: object) -> TaskState:
        raise AssertionError(task_id)

    def resolve_approval(self, task_id: object, action_hash: str, *, decision: str) -> bool:
        del task_id, action_hash, decision
        return False


def test_failing_submit_keeps_global_index_locked_until_rollback_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_root = _project_root(tmp_path, "first")
    second_root = _project_root(tmp_path, "second")
    first = LocalRuntime(
        RuntimeServices(
            credentials=FakeCredentials(),
            orchestrator_factory=lambda root, config, memory: FailingSubmitOrchestrator(),
        )
    )
    second = _runtime(second_root, [[], []])
    first.setup(first_root, _config(), api_key=None)
    second.setup(second_root, _config(), api_key=None)
    restore_started = Event()
    allow_restore = Event()
    original_restore = runtime_module._restore_file
    errors: list[RuntimeError] = []

    def blocked_restore(path: Path, content: bytes | None) -> None:
        if path == runtime_module.local_state_path():
            restore_started.set()
            assert allow_restore.wait(1)
        original_restore(path, content)

    def create_failing_task() -> None:
        try:
            first.create_task("fail", TaskMode.FEATURE, None)
        except RuntimeError as error:
            errors.append(error)

    monkeypatch.setattr(runtime_module, "_restore_file", blocked_restore)
    failing_thread = Thread(target=create_failing_task)
    failing_thread.start()
    assert restore_started.wait(1)
    try:
        with pytest.raises(RuntimeBusyError):
            second.create_task("interleaved", TaskMode.FEATURE, None)
    finally:
        allow_restore.set()
        failing_thread.join()

    assert errors and str(errors[0]) == "submit failed"
    second_task = second.create_task("after rollback", TaskMode.FEATURE, None)
    reader = _runtime(second_root)
    assert [task.id for task in reader.tasks()] == [second_task.id]


def test_same_runtime_task_rollback_cannot_release_a_serialized_winner_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a failed concurrent admission releasing the succeeding task's shared lease."""
    first_root = _project_root(tmp_path, "first")
    second_root = _project_root(tmp_path, "second")
    restore_started = Event()
    allow_restore = Event()
    original_restore = runtime_module._restore_file
    failed: list[RuntimeError] = []
    winners: list[TaskState] = []

    class SuccessfulSubmitOrchestrator(FailingSubmitOrchestrator):
        def submit(self, task: TaskState) -> TaskState:
            return task

    orchestrators = iter((FailingSubmitOrchestrator(), SuccessfulSubmitOrchestrator()))
    first = LocalRuntime(
        RuntimeServices(
            credentials=FakeCredentials(),
            orchestrator_factory=lambda root, config, memory: next(orchestrators),
        )
    )
    second = _runtime(second_root)
    first.setup(first_root, _config(), api_key=None)
    second.setup(second_root, _config(), api_key=None)

    def blocked_restore(path: Path, content: bytes | None) -> None:
        if path == runtime_module.local_state_path():
            restore_started.set()
            assert allow_restore.wait(1)
        original_restore(path, content)

    def create_failed_task() -> None:
        try:
            first.create_task("fail", TaskMode.FEATURE, None)
        except RuntimeError as error:
            failed.append(error)

    def create_winning_task() -> None:
        winners.append(first.create_task("winner", TaskMode.FEATURE, None))

    monkeypatch.setattr(runtime_module, "_restore_file", blocked_restore)
    failing_thread = Thread(target=create_failed_task)
    winner_thread = Thread(target=create_winning_task)
    failing_thread.start()
    assert restore_started.wait(1)
    winner_thread.start()
    allow_restore.set()
    failing_thread.join()
    winner_thread.join()

    assert [str(error) for error in failed] == ["submit failed"]
    assert [task.description for task in winners] == ["winner"]
    with pytest.raises(RuntimeBusyError):
        second.create_task("must remain blocked", TaskMode.FEATURE, None)


@dataclass
class ReplacingOrchestrator:
    task: TaskState | None = None

    def submit(self, task: TaskState) -> TaskState:
        self.task = task
        return task

    def run(self, task: TaskState) -> TaskState:
        return task.model_copy(update={"status": TaskStatus.COMPLETED})

    def cancel(self, task_id: object) -> TaskState:
        assert self.task is not None
        assert task_id == self.task.id
        return self.task.model_copy(update={"status": TaskStatus.CANCELLED})

    def resolve_approval(self, task_id: object, action_hash: str, *, decision: str) -> bool:
        del task_id, action_hash, decision
        return False


def _replacing_runtime(credentials: FakeCredentials) -> LocalRuntime:
    return LocalRuntime(
        RuntimeServices(
            credentials=credentials,
            orchestrator_factory=lambda root, config, memory: ReplacingOrchestrator(),
        )
    )


def test_run_caches_a_replacement_terminal_task_and_releases_its_lease(tmp_path: Path) -> None:
    credentials = FakeCredentials()
    first = _replacing_runtime(credentials)
    second = _runtime(tmp_path)
    first.setup(tmp_path, _config(), api_key=None)
    second.setup(tmp_path, _config(), api_key=None)
    task = first.create_task("finish", TaskMode.FEATURE, None)

    returned = first.run(task.id)

    assert first.task(task.id) is returned
    assert returned.status is TaskStatus.COMPLETED
    assert second.create_task("next", TaskMode.FEATURE, None).status is TaskStatus.PENDING


def test_cancel_caches_a_replacement_terminal_task_and_releases_its_lease(tmp_path: Path) -> None:
    credentials = FakeCredentials()
    first = _replacing_runtime(credentials)
    second = _runtime(tmp_path)
    first.setup(tmp_path, _config(), api_key=None)
    second.setup(tmp_path, _config(), api_key=None)
    task = first.create_task("cancel", TaskMode.FEATURE, None)

    returned = first.cancel(task.id)

    assert first.task(task.id) is returned
    assert returned.status is TaskStatus.CANCELLED
    assert second.create_task("next", TaskMode.FEATURE, None).status is TaskStatus.PENDING
