"""Framework-independent local runtime for governed GuardedPy tasks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from threading import RLock
from typing import Callable, Protocol
from uuid import UUID

import yaml

from guardedpy.command_rules import CommandRuleStore
from guardedpy.config import (
    HarnessConfig,
    app_state_dir,
    load_config,
    load_or_create_discovered_config,
    local_state_path,
    project_config_path,
    save_discovered_config,
    update_future_defaults as updated_future_defaults,
)
from guardedpy.credentials import CredentialStatus
from guardedpy.discovery import ProjectProfile
from guardedpy.domain import ApprovalDecision, CommandApprovalRule, TaskIntent, TaskState, TaskStatus
from guardedpy.events import EventStore, StoredRunEvent
from guardedpy.lease import ExecutionLease, GlobalExecutionLease, GlobalStateLease
from guardedpy.memory import MemoryEntry, MemoryStore


class CredentialPort(Protocol):
    """The non-secret credential operations required by local adapters."""

    def status(self) -> CredentialStatus: ...

    def set_key(self, key: str) -> None: ...

    def clear_key(self) -> None: ...


class OrchestratorPort(Protocol):
    """The task lifecycle operations owned by the harness core."""

    def submit(self, task: TaskState) -> TaskState: ...

    def run(self, task: TaskState) -> TaskState: ...

    def cancel(self, task_id: UUID) -> TaskState: ...

    def resolve_approval(
        self, task_id: UUID, action_hash: str, *, decision: ApprovalDecision
    ) -> bool: ...


OrchestratorFactory = Callable[[Path, HarnessConfig, MemoryStore], OrchestratorPort]


@dataclass(frozen=True, slots=True)
class RuntimeServices:
    """Injectable non-secret services used by the local runtime."""

    credentials: CredentialPort
    orchestrator_factory: OrchestratorFactory


class RuntimeBusyError(RuntimeError):
    """Raised when another local runtime owns the mutation lease."""

    def __init__(self) -> None:
        super().__init__("已有任务正在运行。")


class RuntimeNotConfiguredError(RuntimeError):
    """Raised when an operation requires a selected project configuration."""

    def __init__(self) -> None:
        super().__init__("尚未完成设置。")


class RuntimeTaskNotFoundError(KeyError):
    """Raised when a task is absent from the safe local task index."""

    def __init__(self) -> None:
        super().__init__("未找到任务。")


class RuntimeMemoryNotFoundError(KeyError):
    """Raised when a memory is absent from the selected project's store."""

    def __init__(self) -> None:
        super().__init__("未找到记忆。")


class RuntimeCommandRuleNotFoundError(KeyError):
    """Raised when a command rule is absent from the selected project's store."""

    def __init__(self) -> None:
        super().__init__("未找到命令规则。")


_ACTIVE_STATUSES = {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.WAITING_APPROVAL}
_TERMINAL_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.BLOCKED,
    TaskStatus.CANCELLED,
    TaskStatus.INTERRUPTED,
}


class LocalRuntime:
    """Own local setup, task lifecycle, lease and safe persisted projections."""

    def __init__(self, services: RuntimeServices) -> None:
        self._services = services
        self._project_root: Path | None = None
        self._config: HarnessConfig | None = None
        self._memory_store: MemoryStore | None = None
        self._lease: ExecutionLease | None = None
        self._execution_lease: GlobalExecutionLease | None = None
        self._lifecycle_lock = RLock()
        self._tasks: dict[UUID, TaskState] = {}
        self._orchestrators: dict[UUID, OrchestratorPort] = {}
        self._task_roots: dict[UUID, Path] = {}
        self._running_task_ids: set[UUID] = set()
        self._restore_local_state()

    @property
    def project_root(self) -> Path | None:
        """Return the selected project root, if setup state is available."""
        return self._project_root

    @property
    def config(self) -> HarnessConfig | None:
        """Return the selected non-secret configuration snapshot."""
        return self._config

    def setup(self, profile: ProjectProfile, api_key: str | None) -> None:
        """Atomically save discovered non-secret state and optionally write the keyring key."""
        with self._lifecycle_lock:
            root = profile.root
            if self._has_active_task():
                raise RuntimeBusyError()
            execution_lease = GlobalExecutionLease()
            if not execution_lease.try_acquire():
                raise RuntimeBusyError()
            lease = ExecutionLease(root)
            if not lease.try_acquire():
                execution_lease.release()
                raise RuntimeBusyError()
            global_lease: GlobalStateLease | None = None
            try:
                global_lease = self._acquire_global_state_lease()
                task_roots = _read_local_state()[1] if local_state_path().exists() else {}
                config_path = project_config_path(root)
                index_path = local_state_path()
                previous_config = _snapshot_file(config_path)
                previous_index = _snapshot_file(index_path)
                try:
                    config = load_or_create_discovered_config(profile, config_path.parent.parent)
                    _write_local_state(root, task_roots)
                    if api_key is not None and api_key.strip():
                        self._services.credentials.set_key(api_key)
                except Exception:
                    _restore_file(config_path, previous_config)
                    _restore_file(index_path, previous_index)
                    raise
                self._project_root = root
                self._config = config
                self._memory_store = MemoryStore(root)
                self._lease = lease
                self._execution_lease = execution_lease
                self._task_roots = task_roots
            finally:
                if global_lease is not None:
                    global_lease.release()
                lease.release()
                execution_lease.release()

    def update_future_defaults(
        self, *, model: str | None = None, reasoning_effort: str | None = None
    ) -> HarnessConfig:
        """Persist validated selections for later tasks while no task is active."""
        with self._lifecycle_lock:
            if self._has_active_task():
                raise RuntimeBusyError()
            root, _, _ = self._configured()
            release_after = self._require_mutation_lease()
            global_lease: GlobalStateLease | None = None
            try:
                global_lease = self._acquire_global_state_lease()
                current = _load_config(root)
                changed = updated_future_defaults(
                    current, model=model, reasoning_effort=reasoning_effort
                )
                _write_snapshot(root, changed)
                self._config = changed
                return changed
            finally:
                if global_lease is not None:
                    global_lease.release()
                if release_after:
                    self._release_lease()

    def create_task(
        self,
        description: str,
        intent: TaskIntent = TaskIntent.CODING,
        review_path: str | None = None,
        session_goal: str | None = None,
    ) -> TaskState:
        """Register one pending task and retain the execution lease for its lifecycle."""
        with self._lifecycle_lock:
            return self._create_task(description, intent, review_path, session_goal)

    def _create_task(
        self, description: str, intent: TaskIntent, review_path: str | None,
        session_goal: str | None,
    ) -> TaskState:
        root, config, memory_store = self._configured()
        description = description.strip()
        if not description:
            raise ValueError("task description must be nonblank")
        snapshot = config.model_copy(deep=True)
        task = TaskState(
            description=description,
            intent=intent,
            config=snapshot,
            review_path=review_path,
            session_goal=session_goal,
        )
        if self._has_active_task() or not self._acquire_lease():
            raise RuntimeBusyError()
        registered = False
        index_written = False
        global_lease: GlobalStateLease | None = None
        try:
            global_lease = self._acquire_global_state_lease()
            current = _load_config(root)
            self._config = current
            snapshot = current.model_copy(deep=True)
            task = task.model_copy(update={"config": snapshot}, deep=True)
            event_store = EventStore(root)
            orchestrator = self._services.orchestrator_factory(root, snapshot, memory_store)
            index_path = local_state_path()
            previous_index = _snapshot_file(index_path)
            task_roots = _read_local_state()[1] if index_path.exists() else {}
            event_store.register_task(task)
            registered = True
            task_roots[task.id] = root
            _write_local_state(root, task_roots)
            index_written = True
            orchestrator.submit(task)
        except Exception:
            if index_written:
                _restore_file(index_path, previous_index)
            if registered:
                event_store.discard_task_registration(task.id)
            self._release_lease()
            raise
        finally:
            if global_lease is not None:
                global_lease.release()
        self._tasks[task.id] = task
        self._orchestrators[task.id] = orchestrator
        self._task_roots = task_roots
        return task

    def task(self, task_id: UUID) -> TaskState:
        """Return one registered task without requiring the mutation lease."""
        task = self._tasks.get(task_id)
        if task is not None:
            return task
        root = self._task_root(task_id)
        for stored in EventStore(root).tasks():
            if stored.id == task_id:
                return stored
        raise RuntimeTaskNotFoundError()

    def run(self, task_id: UUID) -> TaskState:
        """Advance one locally-owned task, releasing its lease when it becomes terminal."""
        with self._lifecycle_lock:
            if task_id in self._running_task_ids:
                raise RuntimeBusyError()
            task, orchestrator = self._owned_task(task_id)
            if not self._acquire_lease():
                raise RuntimeBusyError()
            self._running_task_ids.add(task_id)
        return self._run(task_id, task, orchestrator)

    def _run(
        self, task_id: UUID, task: TaskState, orchestrator: OrchestratorPort
    ) -> TaskState:
        result = task
        try:
            result = orchestrator.run(task)
        finally:
            with self._lifecycle_lock:
                self._running_task_ids.discard(task_id)
                current = self._tasks.get(task_id)
                if current is not None and current.status in _TERMINAL_STATUSES:
                    result = current
                else:
                    self._tasks[task_id] = result
                if result.status in _TERMINAL_STATUSES:
                    self._release_lease()
        return result

    def resolve_approval(
        self, task_id: UUID, action_hash: str, decision: ApprovalDecision
    ) -> bool:
        """Resolve one exact pending approval through the owning orchestrator."""
        with self._lifecycle_lock:
            task, orchestrator = self._owned_task(task_id)
            if not self._acquire_lease():
                raise RuntimeBusyError()
        return self._resolve_approval(task_id, task, orchestrator, action_hash, decision)

    def _resolve_approval(
        self,
        task_id: UUID,
        task: TaskState,
        orchestrator: OrchestratorPort,
        action_hash: str,
        decision: ApprovalDecision,
    ) -> bool:
        try:
            return orchestrator.resolve_approval(task_id, action_hash, decision=decision)
        finally:
            with self._lifecycle_lock:
                if task.status in _TERMINAL_STATUSES:
                    self._release_lease()

    def cancel(self, task_id: UUID) -> TaskState:
        """Cancel one locally-owned task and release its terminal execution lease."""
        with self._lifecycle_lock:
            return self._cancel(task_id)

    def _cancel(self, task_id: UUID) -> TaskState:
        task, orchestrator = self._owned_task(task_id)
        if not self._acquire_lease():
            raise RuntimeBusyError()
        cancelled = task
        try:
            cancelled = orchestrator.cancel(task_id)
            self._tasks[task_id] = cancelled
            return cancelled
        finally:
            if cancelled.status in _TERMINAL_STATUSES:
                self._release_lease()

    def tasks(self) -> list[TaskState]:
        """Read persisted task history even when another runtime owns the lease."""
        self._sync_task_roots()
        stored_tasks: list[TaskState] = []
        for root in dict.fromkeys(self._task_roots.values()):
            stored_tasks.extend(EventStore(root).tasks())
        return [self._tasks.get(task.id, task) for task in stored_tasks]

    def recover_interrupted_tasks(self) -> tuple[UUID, ...]:
        """Persist restart interruptions and refresh every visible recovered task."""
        with self._lifecycle_lock:
            return self._recover_interrupted_tasks()

    def _recover_interrupted_tasks(self) -> tuple[UUID, ...]:
        global_lease = self._acquire_global_state_lease()
        try:
            interrupted: list[UUID] = []
            for root in dict.fromkeys(self._task_roots.values()):
                lease = ExecutionLease(root)
                if not lease.try_acquire():
                    continue
                try:
                    store = EventStore(root)
                    recovered_ids = store.mark_unfinished_interrupted()
                    if not recovered_ids:
                        continue
                    stored_tasks = {task.id: task for task in store.tasks()}
                    for task_id in recovered_ids:
                        self._tasks[task_id] = stored_tasks[task_id]
                        self._orchestrators.pop(task_id, None)
                    interrupted.extend(recovered_ids)
                finally:
                    lease.release()
            return tuple(interrupted)
        finally:
            global_lease.release()

    def events(self, task_id: UUID) -> list[StoredRunEvent]:
        """Return only EventStore's fixed safe event projection."""
        return EventStore(self._task_root(task_id)).events_for(task_id)

    def memory_proposals(self) -> list[MemoryEntry]:
        """Return current-process pending memory proposals."""
        return self._configured()[2].proposals()

    def memories(self) -> list[MemoryEntry]:
        """Return approved memories for the selected project."""
        return self._configured()[2].approved()

    def approve_memory(self, memory_id: UUID) -> MemoryEntry:
        """Approve one pending memory while holding the project lease."""
        release_after = self._require_mutation_lease()
        try:
            return self._configured()[2].approve(memory_id)
        except KeyError as error:
            raise RuntimeMemoryNotFoundError() from error
        finally:
            if release_after:
                self._release_lease()

    def delete_memory(self, memory_id: UUID) -> None:
        """Delete one pending or approved memory while holding the project lease."""
        release_after = self._require_mutation_lease()
        try:
            self._configured()[2].delete(memory_id)
        except KeyError as error:
            raise RuntimeMemoryNotFoundError() from error
        finally:
            if release_after:
                self._release_lease()

    def command_rules(self) -> list[CommandApprovalRule]:
        """Return the selected project's structured command approval rules."""
        root, _, _ = self._configured()
        return CommandRuleStore(root).list_rules()

    def delete_command_rule(self, rule_id: str) -> bool:
        """Revoke one structured command rule while holding the project lease."""
        release_after = self._require_mutation_lease()
        try:
            deleted = CommandRuleStore(self._configured()[0]).delete(rule_id)
            if not deleted:
                raise RuntimeCommandRuleNotFoundError()
            return True
        finally:
            if release_after:
                self._release_lease()

    def credential_status(self) -> CredentialStatus:
        """Return only whether a credential is configured."""
        return self._services.credentials.status()

    def update_credential(self, api_key: str) -> None:
        """Store a credential through the injected keyring boundary under the lease."""
        release_after = self._require_mutation_lease()
        global_lease: GlobalStateLease | None = None
        try:
            global_lease = self._acquire_global_state_lease()
            self._services.credentials.set_key(api_key)
        finally:
            if global_lease is not None:
                global_lease.release()
            if release_after:
                self._release_lease()

    def clear_credential(self) -> None:
        """Clear the credential through the injected keyring boundary under the lease."""
        release_after = self._require_mutation_lease()
        global_lease: GlobalStateLease | None = None
        try:
            global_lease = self._acquire_global_state_lease()
            self._services.credentials.clear_key()
        finally:
            if global_lease is not None:
                global_lease.release()
            if release_after:
                self._release_lease()

    def _configured(self) -> tuple[Path, HarnessConfig, MemoryStore]:
        if self._project_root is None or self._config is None or self._memory_store is None:
            raise RuntimeNotConfiguredError()
        return self._project_root, self._config, self._memory_store

    def _owned_task(self, task_id: UUID) -> tuple[TaskState, OrchestratorPort]:
        task = self._tasks.get(task_id)
        orchestrator = self._orchestrators.get(task_id)
        if task is None or orchestrator is None:
            raise RuntimeTaskNotFoundError()
        return task, orchestrator

    def _task_root(self, task_id: UUID) -> Path:
        self._sync_task_roots()
        root = self._task_roots.get(task_id)
        if root is None:
            raise RuntimeTaskNotFoundError()
        return root

    def _sync_task_roots(self) -> None:
        if local_state_path().exists():
            self._task_roots.update(_read_local_state()[1])

    def _restore_local_state(self) -> None:
        if not local_state_path().exists():
            return
        try:
            root, task_roots = _read_local_state()
            config = _load_config(root)
            if any(not (root / path).is_dir() for path in config.source_dirs + config.test_dirs):
                return
        except Exception:
            return
        self._project_root = root
        self._config = config
        self._memory_store = MemoryStore(root)
        self._lease = ExecutionLease(root)
        self._task_roots = task_roots

    def _has_active_task(self) -> bool:
        return any(task.status in _ACTIVE_STATUSES for task in self._tasks.values())

    def _acquire_lease(self) -> bool:
        root, _, _ = self._configured()
        if self._lease is None:
            self._lease = ExecutionLease(root)
        if self._execution_lease is None:
            self._execution_lease = GlobalExecutionLease()
        acquired_execution_lease = False
        if not self._execution_lease.held:
            if not self._execution_lease.try_acquire():
                return False
            acquired_execution_lease = True
        if self._lease.held or self._lease.try_acquire():
            return True
        if acquired_execution_lease:
            self._execution_lease.release()
        return False

    def _require_mutation_lease(self) -> bool:
        if self._lease is not None and self._lease.held:
            return False
        if not self._acquire_lease():
            raise RuntimeBusyError()
        return True

    def _release_lease(self) -> None:
        if self._lease is not None:
            self._lease.release()
        if self._execution_lease is not None:
            self._execution_lease.release()

    @staticmethod
    def _acquire_global_state_lease() -> GlobalStateLease:
        lease = GlobalStateLease()
        if not lease.try_acquire():
            raise RuntimeBusyError()
        return lease


def _write_snapshot(project_root: Path, config: HarnessConfig) -> None:
    save_discovered_config(config, project_config_path(project_root).parent.parent)


def _write_local_state(project_root: Path, task_roots: dict[UUID, Path]) -> None:
    payload = {
        "selected_project": str(project_root.resolve()),
        "task_roots": {
            str(task_id): str(task_root.resolve()) for task_id, task_root in task_roots.items()
        },
    }
    _atomic_write(local_state_path(), yaml.safe_dump(payload, sort_keys=False).encode())


def _read_local_state() -> tuple[Path, dict[UUID, Path]]:
    payload = yaml.safe_load(local_state_path().read_text())
    if not isinstance(payload, dict) or set(payload) != {"selected_project", "task_roots"}:
        raise ValueError("local state has invalid fields")
    selected_project = payload["selected_project"]
    task_roots = payload["task_roots"]
    if not isinstance(selected_project, str) or not isinstance(task_roots, dict):
        raise ValueError("local state has invalid values")
    root = _restored_project_root(selected_project)
    return root, {
        UUID(task_id): _restored_project_root(task_root)
        for task_id, task_root in task_roots.items()
        if isinstance(task_id, str) and isinstance(task_root, str)
    }


def _restored_project_root(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("stored project root is invalid")
    return path.resolve()


def _load_config(project_root: Path) -> HarnessConfig:
    return load_config(project_config_path(project_root), project_root)


def _snapshot_file(path: Path) -> bytes | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise OSError("state path is not a file")
    return path.read_bytes()


def _restore_file(path: Path, content: bytes | None) -> None:
    if content is None:
        path.unlink(missing_ok=True)
        return
    _atomic_write(path, content)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
