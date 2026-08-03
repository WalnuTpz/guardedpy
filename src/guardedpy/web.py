"""Local FastAPI controls around the governed GuardedPy core."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
from threading import Thread
from typing import Any, Callable, Protocol
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import keyring
from openai import OpenAI
from pydantic import ValidationError
import uvicorn
import yaml

from guardedpy.config import HarnessConfig
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


class CredentialPort(Protocol):
    """The non-secret credential operations needed by the local WebUI."""

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
class WebServices:
    """Injectable local dependencies; ASGI tests never touch OS keyring or network."""

    credentials: CredentialPort
    orchestrator_factory: OrchestratorFactory


@dataclass(slots=True)
class _LocalState:
    project_root: Path | None = None
    config: HarnessConfig | None = None
    memory_store: MemoryStore | None = None
    task: TaskState | None = None
    orchestrator: OrchestratorPort | None = None
    thread: Thread | None = None


_ACTIVE_STATUSES = {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.WAITING_APPROVAL}
_TERMINAL_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.BLOCKED,
    TaskStatus.CANCELLED,
    TaskStatus.INTERRUPTED,
}
_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_CREDENTIAL_ERROR = "Credential store is unavailable."


def create_app(mode: str, services: WebServices) -> FastAPI:
    """Create the local-only application with injected core services."""
    if mode != "local":
        raise ValueError("create_app only supports local mode")

    app = FastAPI()
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
    app.state.local = _LocalState()

    def credential_status() -> tuple[CredentialStatus, str | None]:
        try:
            return services.credentials.status(), None
        except CredentialBackendUnavailableError:
            return CredentialStatus(configured=False), _CREDENTIAL_ERROR

    def render_setup(request: Request, *, error: str | None = None, status_code: int = 200) -> HTMLResponse:
        status, credential_error = credential_status()
        if credential_error is not None:
            error = credential_error
            status_code = 503
        return _TEMPLATES.TemplateResponse(
            request,
            "setup.html",
            {
                "request": request,
                "error": error,
                "configured": status.configured,
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
            {"error": error, "configured": state.config is not None, "task": state.task},
            status_code=status_code,
        )

    def task_for(task_id: UUID) -> TaskState:
        task = app.state.local.task
        if task is None or task.id != task_id:
            raise HTTPException(status_code=404, detail="Task was not found.")
        return task

    def task_events(task_id: UUID) -> list[StoredRunEvent]:
        state: _LocalState = app.state.local
        if state.project_root is None:
            raise HTTPException(status_code=404, detail="Task was not found.")
        return EventStore(state.project_root).events_for(task_id)

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        state: _LocalState = app.state.local
        if state.config is None:
            return render_setup(request)
        return render_task(request)

    @app.post("/setup", response_class=HTMLResponse)
    async def setup(request: Request) -> Response:
        state: _LocalState = app.state.local
        if state.task is not None and state.task.status in _ACTIVE_STATUSES:
            return render_task(request, error="Another task is active.", status_code=409)
        form = await request.form()
        try:
            project_root, config, api_key = _validated_setup(form)
        except (TypeError, ValueError, ValidationError):
            return render_setup(request, error="Setup could not be saved.", status_code=422)

        try:
            memory_store = MemoryStore(project_root)
            _write_snapshot(project_root, config)
            services.credentials.set_key(api_key)
        except Exception:
            return render_setup(request, error="Setup could not be saved.", status_code=422)

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
        if state.config is None or state.project_root is None or state.memory_store is None:
            return render_setup(request, error="Setup could not be saved.", status_code=422)
        form = await request.form()
        description = str(form.get("description", "")).strip()
        bugfix_target = str(form.get("bugfix_target", "")).strip()
        try:
            task_mode = TaskMode(str(form.get("mode", "")))
        except ValueError:
            return render_task(request, error="Task could not be started.", status_code=422)
        if not description:
            return render_task(request, error="Task could not be started.", status_code=422)
        if task_mode is TaskMode.BUGFIX and not bugfix_target:
            return render_task(request, error="Task could not be started.", status_code=422)
        if state.task is not None and state.task.status in _ACTIVE_STATUSES:
            return render_task(request, error="Another task is active.", status_code=409)

        task = TaskState(
            description=description,
            mode=task_mode,
            bugfix_target=bugfix_target if task_mode is TaskMode.BUGFIX else None,
            config=state.config,
        )
        orchestrator = services.orchestrator_factory(state.project_root, state.config, state.memory_store)
        try:
            orchestrator.submit(task)
        except ValueError:
            return render_task(request, error="Another task is active.", status_code=409)
        state.task = task
        state.orchestrator = orchestrator
        thread = Thread(target=orchestrator.run, args=(task,), daemon=True)
        state.thread = thread
        thread.start()
        return RedirectResponse("/tasks/new", status_code=303)

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
        if state.orchestrator is None:
            raise HTTPException(status_code=404, detail="Task was not found.")
        form = await request.form()
        action_hash = str(form.get("action_hash", ""))
        decision = str(form.get("decision", ""))
        if not is_approval_decision(decision):
            raise HTTPException(status_code=409, detail="Approval is stale.")

        was_waiting = task.status is TaskStatus.WAITING_APPROVAL
        accepted = state.orchestrator.resolve_approval(task_id, action_hash, decision=decision)
        if accepted:
            thread = Thread(target=state.orchestrator.run, args=(task,), daemon=True)
            state.thread = thread
            thread.start()
            return RedirectResponse(f"/tasks/{task_id}", status_code=303)
        if decision == "reject" and was_waiting and task.status is TaskStatus.BLOCKED:
            return RedirectResponse(f"/tasks/{task_id}", status_code=303)
        raise HTTPException(status_code=409, detail="Approval is stale.")

    @app.post("/tasks/{task_id}/cancel", response_class=HTMLResponse)
    async def cancel_task(task_id: UUID, request: Request) -> Response:
        state: _LocalState = app.state.local
        if state.task is None or state.orchestrator is None or state.task.id != task_id:
            return render_task(request, error="Task was not found.", status_code=404)
        state.task = state.orchestrator.cancel(task_id)
        return RedirectResponse("/tasks/new", status_code=303)

    @app.get("/memories", response_class=HTMLResponse)
    async def memories(request: Request) -> HTMLResponse:
        memory_store = app.state.local.memory_store
        if memory_store is None:
            raise HTTPException(status_code=404, detail="Memory store was not found.")
        return _TEMPLATES.TemplateResponse(
            request,
            "memory.html",
            {"proposals": memory_store.proposals(), "approved": memory_store.approved()},
        )

    @app.post("/memories/{memory_id}/approve", response_class=HTMLResponse)
    async def approve_memory(memory_id: UUID) -> Response:
        memory_store = app.state.local.memory_store
        if memory_store is None:
            raise HTTPException(status_code=404, detail="Memory was not found.")
        try:
            memory_store.approve(memory_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Memory was not found.") from None
        return RedirectResponse("/memories", status_code=303)

    @app.post("/memories/{memory_id}/delete", response_class=HTMLResponse)
    async def delete_memory(memory_id: UUID) -> Response:
        memory_store = app.state.local.memory_store
        if memory_store is None:
            raise HTTPException(status_code=404, detail="Memory was not found.")
        try:
            memory_store.delete(memory_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Memory was not found.") from None
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
                request, error="Credential could not be updated.", status_code=422
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
            raise HTTPException(status_code=404, detail="Project was not found.")
        rules = [
            (rule, _command_rule_projection(rule))
            for rule in CommandRuleStore(project_root).list_rules()
        ]
        return _TEMPLATES.TemplateResponse(
            request,
            "command_rules.html",
            {"rules": rules},
        )

    @app.post("/settings/command-rules/{rule_id}/delete", response_class=HTMLResponse)
    async def delete_command_rule(rule_id: str) -> Response:
        project_root = app.state.local.project_root
        if project_root is None:
            raise HTTPException(status_code=404, detail="Project was not found.")
        if not CommandRuleStore(project_root).delete(rule_id):
            raise HTTPException(status_code=404, detail="Command rule was not found.")
        return RedirectResponse("/settings/command-rules", status_code=303)

    return app


def _validated_setup(form: Any) -> tuple[Path, HarnessConfig, str]:
    project_root = Path(str(form.get("project_root", ""))).expanduser()
    if not project_root.is_dir():
        raise ValueError("project root is invalid")
    project_root = project_root.resolve()
    source_dirs = _configured_directories(str(form.get("source_dirs", "")), project_root)
    test_dirs = _configured_directories(str(form.get("test_dirs", "")), project_root)
    command = tuple(shlex.split(str(form.get("pytest_command", ""))))
    model = str(form.get("model", "")).strip()
    api_key = str(form.get("api_key", ""))
    if not command or not model or not api_key.strip():
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
    (project_root / "harness.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False))


def _system_keyring() -> KeyringBackend:
    return keyring.get_keyring()


def _deepseek_transport(api_key: str) -> Any:
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def local_services(
    *, transport_factory: Callable[[str], Any] = _deepseek_transport
) -> WebServices:
    """Compose real local dependencies without asking keyring for a key at startup."""
    credentials = CredentialService(_system_keyring())

    def orchestrator_factory(
        project_root: Path, config: HarnessConfig, memory_store: MemoryStore
    ) -> TaskOrchestrator:
        llm = DeepSeekClient(credentials.get_key, config.model, transport_factory)
        return TaskOrchestrator(project_root, llm, memory_store=memory_store)

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
