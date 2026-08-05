"""Local FastAPI controls around the governed GuardedPy core."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from threading import Thread
from typing import Any, Callable, Protocol
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import keyring
from openai import OpenAI
from pydantic import ValidationError
import uvicorn
import yaml

from guardedpy.config import HarnessConfig
from guardedpy.credentials import CredentialService, CredentialStatus, KeyringBackend
from guardedpy.domain import TaskMode, TaskState, TaskStatus
from guardedpy.llm import DeepSeekClient
from guardedpy.memory import MemoryStore
from guardedpy.orchestrator import TaskOrchestrator


class CredentialPort(Protocol):
    """The non-secret credential operations needed by the local WebUI."""

    def status(self) -> CredentialStatus: ...

    def set_key(self, key: str) -> None: ...

    def get_key(self) -> str: ...

    def clear_key(self) -> None: ...


class OrchestratorPort(Protocol):
    """The task lifecycle operations owned by the harness core."""

    def submit(self, task: TaskState) -> TaskState: ...

    def run(self, task: TaskState) -> TaskState: ...

    def cancel(self, task_id: UUID) -> TaskState: ...


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
_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def create_app(mode: str, services: WebServices) -> FastAPI:
    """Create the local-only application with injected core services."""
    if mode != "local":
        raise ValueError("create_app only supports local mode")

    app = FastAPI()
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
    app.state.local = _LocalState()

    def render_setup(request: Request, *, error: str | None = None, status_code: int = 200) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request,
            "setup.html",
            {
                "request": request,
                "error": error,
                "configured": services.credentials.status().configured,
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

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        state: _LocalState = app.state.local
        if state.config is None:
            return render_setup(request)
        return render_task(request)

    @app.post("/setup", response_class=HTMLResponse)
    async def setup(request: Request) -> Response:
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

        state: _LocalState = app.state.local
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
        try:
            task_mode = TaskMode(str(form.get("mode", "")))
        except ValueError:
            return render_task(request, error="Task could not be started.", status_code=422)
        if not description:
            return render_task(request, error="Task could not be started.", status_code=422)
        if state.task is not None and state.task.status in _ACTIVE_STATUSES:
            return render_task(request, error="Another task is active.", status_code=409)

        task = TaskState(description=description, mode=task_mode, config=state.config)
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

    @app.post("/tasks/{task_id}/cancel", response_class=HTMLResponse)
    async def cancel_task(task_id: UUID, request: Request) -> Response:
        state: _LocalState = app.state.local
        if state.task is None or state.orchestrator is None or state.task.id != task_id:
            return render_task(request, error="Task was not found.", status_code=404)
        state.task = state.orchestrator.cancel(task_id)
        return RedirectResponse("/tasks/new", status_code=303)

    @app.get("/settings/credentials", response_class=HTMLResponse)
    async def credentials_page(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request,
            "base.html",
            {"configured": services.credentials.status().configured, "page": "credentials"},
        )

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
    """Run the local-only control surface; public demo uses a separate factory."""
    uvicorn.run(create_app("local", local_services()), host="127.0.0.1")
