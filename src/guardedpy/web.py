"""Local FastAPI controls around the governed GuardedPy core."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from threading import Thread
from typing import Any, Callable
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import keyring
from openai import OpenAI
from pydantic import ValidationError
import uvicorn
import yaml

from guardedpy.config import (
    HarnessConfig,
    load_config,
    local_state_path,
    project_config_path,
)
from guardedpy.command_rules import CommandRuleStore
from guardedpy.credentials import (
    CredentialBackendUnavailableError,
    CredentialService,
    CredentialStatus,
    KeyringBackend,
)
from guardedpy.demo import create_demo_app
from guardedpy.domain import (
    ApprovalDecision,
    CommandApprovalRule,
    CommandRuleKind,
    PolicyVerdict,
    TaskMode,
    TaskState,
    TaskStatus,
    is_approval_decision,
)
from guardedpy.events import EventStore, StoredRunEvent
from guardedpy.llm import DeepSeekClient
from guardedpy.memory import MemoryStore
from guardedpy.orchestrator import TaskOrchestrator
from guardedpy.runtime import CredentialPort, OrchestratorFactory, OrchestratorPort, RuntimeServices


WebServices = RuntimeServices


@dataclass(slots=True)
class _LocalState:
    project_root: Path | None = None
    config: HarnessConfig | None = None
    memory_store: MemoryStore | None = None
    task: TaskState | None = None
    orchestrator: OrchestratorPort | None = None
    thread: Thread | None = None
    tasks: dict[UUID, TaskState] = field(default_factory=dict)
    orchestrators: dict[UUID, OrchestratorPort] = field(default_factory=dict)
    task_roots: dict[UUID, Path] = field(default_factory=dict)
    mutation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_ACTIVE_STATUSES = {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.WAITING_APPROVAL}
_TERMINAL_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.BLOCKED,
    TaskStatus.CANCELLED,
    TaskStatus.INTERRUPTED,
}
_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_CREDENTIAL_ERROR = "凭据存储不可用。"
_SETUP_ERROR = "无法保存设置。"
_ACTIVE_TASK_ERROR = "已有任务正在运行。"
_TASK_START_ERROR = "无法启动任务。"
_TASK_NOT_FOUND_ERROR = "未找到任务。"
_CREDENTIAL_UPDATE_ERROR = "无法更新凭据。"
_APPROVAL_STALE_ERROR = "审批请求已失效。"
_MEMORY_STORE_NOT_FOUND_ERROR = "未找到记忆存储。"
_MEMORY_NOT_FOUND_ERROR = "未找到记忆。"
_PROJECT_NOT_FOUND_ERROR = "未找到项目。"
_COMMAND_RULE_NOT_FOUND_ERROR = "未找到命令规则。"


def create_app(mode: str, services: WebServices) -> FastAPI:
    """Create the local-only application with injected core services."""
    if mode != "local":
        raise ValueError("create_app only supports local mode")

    app = FastAPI()

    @app.exception_handler(RequestValidationError)
    async def local_validation_error(_request: Request, _error: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "请求参数无效。"})

    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
    app.state.local = _load_startup_state()

    def credential_status() -> tuple[CredentialStatus, str | None]:
        try:
            return services.credentials.status(), None
        except CredentialBackendUnavailableError:
            return CredentialStatus(configured=False), _CREDENTIAL_ERROR

    def render_setup(request: Request, *, error: str | None = None, status_code: int = 200) -> HTMLResponse:
        state: _LocalState = app.state.local
        status, credential_error = credential_status()
        if credential_error is not None:
            error = credential_error
            status_code = 503
        return _TEMPLATES.TemplateResponse(
            request,
            "setup.html",
            {
                "request": request,
                "page": "setup",
                "error": error,
                "configured": status.configured,
                "project_root": str(state.project_root) if state.project_root else "",
                "source_dirs": _display_paths(state.config.source_dirs) if state.config else "src",
                "test_dirs": _display_paths(state.config.test_dirs) if state.config else "tests",
                "pytest_command": shlex.join(state.config.pytest_command) if state.config else "pytest",
                "model": state.config.model if state.config else "deepseek-chat",
                "timeout_seconds": state.config.timeout_seconds if state.config else 30,
            },
            status_code=status_code,
        )

    def render_credentials(
        request: Request, *, error: str | None = None, status_code: int = 200
    ) -> HTMLResponse:
        status, credential_error = credential_status()
        if credential_error is not None:
            error = credential_error
            status_code = 503
        return _TEMPLATES.TemplateResponse(
            request,
            "base.html",
            {
                "configured": status.configured,
                "error": error,
                "page": "credentials",
            },
            status_code=status_code,
        )

    def render_task(request: Request, *, error: str | None = None, status_code: int = 200) -> HTMLResponse:
        state: _LocalState = app.state.local
        return _TEMPLATES.TemplateResponse(
            request,
            "task.html",
            {
                "error": error,
                "page": "tasks",
                "context_task": state.task,
                "configured": state.config is not None,
                "task": state.task,
                "tasks": list(state.tasks.values()),
            },
            status_code=status_code,
        )

    def task_for(task_id: UUID) -> TaskState:
        task = app.state.local.tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=_TASK_NOT_FOUND_ERROR)
        return task

    def task_events(task_id: UUID) -> list[StoredRunEvent]:
        state: _LocalState = app.state.local
        project_root = state.task_roots.get(task_id)
        if project_root is None:
            raise HTTPException(status_code=404, detail=_TASK_NOT_FOUND_ERROR)
        return EventStore(project_root).events_for(task_id)

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        state: _LocalState = app.state.local
        if state.config is None:
            return render_setup(request)
        return render_task(request)

    @app.get("/setup", response_class=HTMLResponse)
    async def setup_page(request: Request) -> HTMLResponse:
        return render_setup(request)

    @app.post("/setup", response_class=HTMLResponse)
    async def setup(request: Request) -> Response:
        state: _LocalState = app.state.local
        if _has_active_task(state):
            return render_task(request, error=_ACTIVE_TASK_ERROR, status_code=409)
        form = await request.form()
        status, credential_error = credential_status()
        if credential_error is not None:
            return render_setup(request, error=_CREDENTIAL_ERROR, status_code=503)
        try:
            project_root, config, api_key = _validated_setup(
                form, require_api_key=not status.configured
            )
        except (TypeError, ValueError, ValidationError):
            return render_setup(request, error=_SETUP_ERROR, status_code=422)

        async with state.mutation_lock:
            if _has_active_task(state):
                return render_task(request, error=_ACTIVE_TASK_ERROR, status_code=409)
            try:
                memory_store = MemoryStore(project_root)
                config_path = project_config_path(project_root)
                index_path = local_state_path()
                previous_config = _snapshot_file(config_path)
                previous_index = _snapshot_file(index_path)
            except Exception:
                return render_setup(request, error=_SETUP_ERROR, status_code=422)
            try:
                _write_snapshot(project_root, config)
                _write_local_state(project_root, state.task_roots)
                if api_key.strip():
                    services.credentials.set_key(api_key)
            except CredentialBackendUnavailableError:
                _restore_file(config_path, previous_config)
                _restore_file(index_path, previous_index)
                return render_setup(request, error=_CREDENTIAL_ERROR, status_code=503)
            except Exception:
                _restore_file(config_path, previous_config)
                _restore_file(index_path, previous_index)
                return render_setup(request, error=_SETUP_ERROR, status_code=422)

            state.project_root = project_root
            state.config = config
            state.memory_store = memory_store
        return RedirectResponse("/tasks/new", status_code=303)

    @app.get("/tasks/new", response_class=HTMLResponse)
    async def new_task(request: Request) -> Response:
        if app.state.local.config is None:
            return RedirectResponse("/", status_code=303)
        return render_task(request)

    @app.post("/tasks", response_class=HTMLResponse)
    async def create_task(request: Request) -> Response:
        state: _LocalState = app.state.local
        form = await request.form()
        description = str(form.get("description", "")).strip()
        bugfix_target = str(form.get("bugfix_target", "")).strip()
        try:
            task_mode = TaskMode(str(form.get("mode", "")))
        except ValueError:
            return render_task(request, error=_TASK_START_ERROR, status_code=422)
        if not description:
            return render_task(request, error=_TASK_START_ERROR, status_code=422)
        if task_mode is TaskMode.BUGFIX and not bugfix_target:
            return render_task(request, error=_TASK_START_ERROR, status_code=422)
        async with state.mutation_lock:
            if state.config is None or state.project_root is None or state.memory_store is None:
                return render_setup(request, error=_SETUP_ERROR, status_code=422)
            if _has_active_task(state):
                return render_task(request, error=_ACTIVE_TASK_ERROR, status_code=409)

            task = TaskState(
                description=description,
                mode=task_mode,
                bugfix_target=bugfix_target if task_mode is TaskMode.BUGFIX else None,
                config=state.config,
            )
            orchestrator = services.orchestrator_factory(
                state.project_root, state.config, state.memory_store
            )
            event_store = EventStore(state.project_root)
            index_path = local_state_path()
            try:
                previous_index = _snapshot_file(index_path)
            except Exception:
                return render_task(request, error=_TASK_START_ERROR, status_code=422)
            registered = False
            index_written = False
            try:
                event_store.register_task(task)
                registered = True
                task_roots = {**state.task_roots, task.id: state.project_root}
                _write_local_state(state.project_root, task_roots)
                index_written = True
                orchestrator.submit(task)
            except ValueError:
                if index_written:
                    _restore_file(index_path, previous_index)
                if registered:
                    event_store.discard_task_registration(task.id)
                return render_task(request, error=_ACTIVE_TASK_ERROR, status_code=409)
            except Exception:
                if index_written:
                    _restore_file(index_path, previous_index)
                if registered:
                    event_store.discard_task_registration(task.id)
                return render_task(request, error=_TASK_START_ERROR, status_code=422)
            state.task = task
            state.orchestrator = orchestrator
            state.tasks[task.id] = task
            state.orchestrators[task.id] = orchestrator
            state.task_roots = task_roots
            thread = Thread(target=orchestrator.run, args=(task,), daemon=True)
            state.thread = thread
        thread.start()
        return RedirectResponse(f"/tasks/{task.id}", status_code=303)

    @app.get("/tasks/{task_id}", response_class=HTMLResponse)
    async def task_detail(task_id: UUID, request: Request) -> HTMLResponse:
        task = task_for(task_id)
        events = task_events(task_id)
        approval_event = next(
            (
                event
                for event in reversed(events)
                if event.task_status is TaskStatus.WAITING_APPROVAL
                and event.policy_verdict is PolicyVerdict.APPROVAL_REQUIRED
                and event.action_hash is not None
            ),
            None,
        )
        return _TEMPLATES.TemplateResponse(
            request,
            "task_detail.html",
            {
                "task": task,
                "events": events,
                "approval_event": approval_event,
                "terminal": task.status in _TERMINAL_STATUSES,
                "page": "task_detail",
                "context_task": task,
            },
        )

    @app.get("/tasks/{task_id}/events")
    async def task_event_feed(task_id: UUID) -> list[dict[str, object]]:
        task_for(task_id)
        return [event.model_dump(mode="json") for event in task_events(task_id)]

    @app.post("/tasks/{task_id}/approval", response_class=HTMLResponse)
    async def resolve_task_approval(task_id: UUID, request: Request) -> Response:
        state: _LocalState = app.state.local
        task = task_for(task_id)
        orchestrator = state.orchestrators.get(task_id)
        if orchestrator is None:
            raise HTTPException(status_code=404, detail=_TASK_NOT_FOUND_ERROR)
        form = await request.form()
        action_hash = str(form.get("action_hash", ""))
        decision = str(form.get("decision", ""))
        if not is_approval_decision(decision):
            raise HTTPException(status_code=409, detail=_APPROVAL_STALE_ERROR)

        was_waiting = task.status is TaskStatus.WAITING_APPROVAL
        accepted = orchestrator.resolve_approval(task_id, action_hash, decision=decision)
        if accepted:
            thread = Thread(target=orchestrator.run, args=(task,), daemon=True)
            state.thread = thread
            thread.start()
            return RedirectResponse(f"/tasks/{task_id}", status_code=303)
        if decision == "reject" and was_waiting and task.status is TaskStatus.BLOCKED:
            return RedirectResponse(f"/tasks/{task_id}", status_code=303)
        raise HTTPException(status_code=409, detail=_APPROVAL_STALE_ERROR)

    @app.post("/tasks/{task_id}/cancel", response_class=HTMLResponse)
    async def cancel_task(task_id: UUID, request: Request) -> Response:
        state: _LocalState = app.state.local
        task = state.tasks.get(task_id)
        orchestrator = state.orchestrators.get(task_id)
        if task is None or orchestrator is None:
            return render_task(request, error=_TASK_NOT_FOUND_ERROR, status_code=404)
        cancelled = orchestrator.cancel(task_id)
        state.tasks[task_id] = cancelled
        if state.task is not None and state.task.id == task_id:
            state.task = cancelled
        return RedirectResponse("/tasks/new", status_code=303)

    @app.get("/memories", response_class=HTMLResponse)
    async def memories(request: Request) -> HTMLResponse:
        memory_store = app.state.local.memory_store
        if memory_store is None:
            raise HTTPException(status_code=404, detail=_MEMORY_STORE_NOT_FOUND_ERROR)
        return _TEMPLATES.TemplateResponse(
            request,
            "memory.html",
            {
                "proposals": memory_store.proposals(),
                "approved": memory_store.approved(),
                "page": "memories",
            },
        )

    @app.post("/memories/{memory_id}/approve", response_class=HTMLResponse)
    async def approve_memory(memory_id: UUID) -> Response:
        memory_store = app.state.local.memory_store
        if memory_store is None:
            raise HTTPException(status_code=404, detail=_MEMORY_NOT_FOUND_ERROR)
        try:
            memory_store.approve(memory_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=_MEMORY_NOT_FOUND_ERROR) from None
        return RedirectResponse("/memories", status_code=303)

    @app.post("/memories/{memory_id}/delete", response_class=HTMLResponse)
    async def delete_memory(memory_id: UUID) -> Response:
        memory_store = app.state.local.memory_store
        if memory_store is None:
            raise HTTPException(status_code=404, detail=_MEMORY_NOT_FOUND_ERROR)
        try:
            memory_store.delete(memory_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=_MEMORY_NOT_FOUND_ERROR) from None
        return RedirectResponse("/memories", status_code=303)

    @app.get("/settings/credentials", response_class=HTMLResponse)
    async def credentials_page(request: Request) -> HTMLResponse:
        return render_credentials(request)

    @app.post("/settings/credentials", response_class=HTMLResponse)
    async def update_credentials(request: Request) -> Response:
        form = await request.form()
        api_key = str(form.get("api_key", ""))
        if not api_key.strip():
            return render_credentials(
                request, error=_CREDENTIAL_UPDATE_ERROR, status_code=422
            )
        try:
            services.credentials.set_key(api_key)
        except CredentialBackendUnavailableError:
            return render_credentials(request, error=_CREDENTIAL_ERROR, status_code=503)
        return RedirectResponse("/settings/credentials", status_code=303)

    @app.post("/settings/credentials/clear", response_class=HTMLResponse)
    async def clear_credentials(request: Request) -> Response:
        try:
            services.credentials.clear_key()
        except CredentialBackendUnavailableError:
            return render_credentials(request, error=_CREDENTIAL_ERROR, status_code=503)
        return RedirectResponse("/settings/credentials", status_code=303)

    @app.get("/settings/command-rules", response_class=HTMLResponse)
    async def command_rules(request: Request) -> HTMLResponse:
        project_root = app.state.local.project_root
        if project_root is None:
            raise HTTPException(status_code=404, detail=_PROJECT_NOT_FOUND_ERROR)
        rules = [
            (rule, _command_rule_projection(rule))
            for rule in CommandRuleStore(project_root).list_rules()
        ]
        return _TEMPLATES.TemplateResponse(
            request,
            "command_rules.html",
            {"rules": rules, "page": "command_rules"},
        )

    @app.post("/settings/command-rules/{rule_id}/delete", response_class=HTMLResponse)
    async def delete_command_rule(rule_id: str) -> Response:
        project_root = app.state.local.project_root
        if project_root is None:
            raise HTTPException(status_code=404, detail=_PROJECT_NOT_FOUND_ERROR)
        if not CommandRuleStore(project_root).delete(rule_id):
            raise HTTPException(status_code=404, detail=_COMMAND_RULE_NOT_FOUND_ERROR)
        return RedirectResponse("/settings/command-rules", status_code=303)

    return app


def _validated_setup(
    form: Any, *, require_api_key: bool = True
) -> tuple[Path, HarnessConfig, str]:
    project_root = Path(str(form.get("project_root", ""))).expanduser()
    if not project_root.is_dir():
        raise ValueError("project root is invalid")
    project_root = project_root.resolve()
    source_dirs = _configured_directories(str(form.get("source_dirs", "")), project_root)
    test_dirs = _configured_directories(str(form.get("test_dirs", "")), project_root)
    command = tuple(shlex.split(str(form.get("pytest_command", ""))))
    model = str(form.get("model", "")).strip()
    api_key = str(form.get("api_key", ""))
    if not command or not model or (require_api_key and not api_key.strip()):
        raise ValueError("required setup value is empty")
    config = HarnessConfig(
        source_dirs=source_dirs,
        test_dirs=test_dirs,
        pytest_command=command,
        model=model,
        timeout_seconds=int(str(form.get("timeout_seconds", ""))),
    )
    return project_root, config, api_key


def _command_rule_projection(rule: CommandApprovalRule) -> str:
    """Format only the structured fields admitted by command-family validation."""
    if rule.kind is CommandRuleKind.GIT_DIFF_CHECK:
        return "Git whitespace check"
    if rule.kind is CommandRuleKind.GIT_PUSH:
        return f"Git push to origin/{rule.branch}"
    return f"Python package install: {', '.join(rule.package_specs)}"


def _configured_directories(value: str, project_root: Path) -> tuple[Path, ...]:
    entries = tuple(Path(entry.strip()) for entry in value.replace("\n", ",").split(",") if entry.strip())
    if not entries:
        raise ValueError("at least one directory is required")
    for entry in entries:
        if entry.is_absolute() or ".." in entry.parts or not (project_root / entry).is_dir():
            raise ValueError("configured directory is invalid")
    return entries


def _write_snapshot(project_root: Path, config: HarnessConfig) -> None:
    """Persist only the schema-validated, non-secret project configuration."""
    snapshot = {
        "source_dirs": [str(path) for path in config.source_dirs],
        "test_dirs": [str(path) for path in config.test_dirs],
        "pytest_command": list(config.pytest_command),
        "model": config.model,
        "timeout_seconds": config.timeout_seconds,
    }
    _atomic_write(
        project_config_path(project_root),
        yaml.safe_dump(snapshot, sort_keys=False).encode(),
    )


def _display_paths(paths: tuple[Path, ...]) -> str:
    return ", ".join(str(path) for path in paths)


def _has_active_task(state: _LocalState) -> bool:
    return any(task.status in _ACTIVE_STATUSES for task in state.tasks.values())


def _write_local_state(project_root: Path, task_roots: dict[UUID, Path]) -> None:
    payload = {
        "selected_project": str(project_root.resolve()),
        "task_roots": {
            str(task_id): str(task_root.resolve())
            for task_id, task_root in task_roots.items()
        },
    }
    _atomic_write(
        local_state_path(),
        yaml.safe_dump(payload, sort_keys=False).encode(),
    )


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


def _load_startup_state() -> _LocalState:
    state = _LocalState()
    try:
        project_root, task_roots = _read_local_state()
        config = load_config(project_config_path(project_root), project_root)
        if any(not (project_root / path).is_dir() for path in config.source_dirs + config.test_dirs):
            raise ValueError("configured directory is unavailable")

        tasks_by_root: dict[Path, dict[UUID, TaskState]] = {}
        for task_root in dict.fromkeys(task_roots.values()):
            store = EventStore(task_root)
            store.mark_unfinished_interrupted()
            tasks_by_root[task_root] = {task.id: task for task in store.tasks()}

        tasks: dict[UUID, TaskState] = {}
        for task_id, task_root in task_roots.items():
            task = tasks_by_root[task_root].get(task_id)
            if task is None:
                raise ValueError("task index has no matching metadata")
            tasks[task_id] = task

        state.project_root = project_root
        state.config = config
        state.memory_store = MemoryStore(project_root)
        state.tasks = tasks
        state.task_roots = task_roots
        state.task = next(reversed(tasks.values()), None)
    except Exception:
        return _LocalState()
    return state


def _read_local_state() -> tuple[Path, dict[UUID, Path]]:
    payload = yaml.safe_load(local_state_path().read_text())
    if not isinstance(payload, dict) or set(payload) != {"selected_project", "task_roots"}:
        raise ValueError("local state has invalid fields")
    selected_value = payload["selected_project"]
    root_values = payload["task_roots"]
    if not isinstance(selected_value, str) or not isinstance(root_values, dict):
        raise ValueError("local state has invalid values")
    project_root = _restored_project_root(selected_value)
    task_roots: dict[UUID, Path] = {}
    for task_id, task_root in root_values.items():
        if not isinstance(task_id, str) or not isinstance(task_root, str):
            raise ValueError("task index has invalid values")
        task_roots[UUID(task_id)] = _restored_project_root(task_root)
    return project_root, task_roots


def _restored_project_root(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("stored project root is invalid")
    return path.resolve()


def _system_keyring() -> KeyringBackend:
    return keyring.get_keyring()


def _deepseek_transport(
    api_key: str,
    *,
    timeout_seconds: int,
    openai_factory: Callable[..., Any] | None = None,
) -> Any:
    factory = OpenAI if openai_factory is None else openai_factory
    return factory(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=timeout_seconds,
        max_retries=0,
    )


def _current_git_branch(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                "symbolic-ref",
                "--quiet",
                "--short",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=5,
        )
    except Exception:
        return None
    branch = completed.stdout.strip()
    if completed.returncode != 0 or not branch:
        return None
    return branch


def local_services(
    *,
    transport_factory: Callable[[str], Any] | None = None,
    current_branch_provider: Callable[[Path], str | None] | None = None,
) -> WebServices:
    """Compose real local dependencies without asking keyring for a key at startup."""
    credentials = CredentialService(_system_keyring())
    branch_provider = current_branch_provider or _current_git_branch

    def orchestrator_factory(
        project_root: Path, config: HarnessConfig, memory_store: MemoryStore
    ) -> TaskOrchestrator:
        configured_transport_factory = transport_factory
        if configured_transport_factory is None:
            configured_transport_factory = lambda api_key: _deepseek_transport(
                api_key,
                timeout_seconds=config.timeout_seconds,
            )
        llm = DeepSeekClient(credentials.get_key, config.model, configured_transport_factory)
        return TaskOrchestrator(
            project_root,
            llm,
            memory_store=memory_store,
            current_branch_provider=lambda: branch_provider(project_root),
        )

    return WebServices(credentials=credentials, orchestrator_factory=orchestrator_factory)


def serve() -> None:
    """Run local controls by default or the separately composed public demo."""
    parser = argparse.ArgumentParser(prog="guardedpy")
    parser.add_argument("mode", choices=("serve", "demo"), default="serve", nargs="?")
    arguments = parser.parse_args()
    if arguments.mode == "demo":
        uvicorn.run(create_demo_app(), host="127.0.0.1")
        return
    uvicorn.run(create_app("local", local_services()), host="127.0.0.1")
