"""Local FastAPI HTML controls adapted to the framework-independent runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
from threading import Thread
from typing import Any, Callable
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import keyring
from openai import OpenAI
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

from guardedpy.api import create_api_router
from guardedpy.command_rules import CommandRuleStore
from guardedpy.config import HarnessConfig
from guardedpy.credentials import CredentialBackendUnavailableError, CredentialService, CredentialStatus, KeyringBackend
from guardedpy.demo import create_demo_app
from guardedpy.domain import ApprovalDecision, CommandApprovalRule, CommandRuleKind, PolicyVerdict, TaskMode, TaskStatus, is_approval_decision
from guardedpy.events import EventStore, StoredRunEvent
from guardedpy.llm import DeepSeekClient
from guardedpy.memory import MemoryEntry, MemoryStore
from guardedpy.orchestrator import TaskOrchestrator
from guardedpy.runtime import (
    CredentialPort,
    LocalRuntime,
    OrchestratorFactory,
    OrchestratorPort,
    RuntimeBusyError,
    RuntimeCommandRuleNotFoundError,
    RuntimeMemoryNotFoundError,
    RuntimeNotConfiguredError,
    RuntimeServices,
    RuntimeTaskNotFoundError,
)


WebServices = RuntimeServices

_ACTIVE_STATUSES = {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.WAITING_APPROVAL}
_TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.BLOCKED, TaskStatus.CANCELLED, TaskStatus.INTERRUPTED}
_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_CREDENTIAL_ERROR = "凭据存储不可用。"
_SETUP_ERROR = "无法保存设置。"
_ACTIVE_TASK_ERROR = "已有任务正在运行。"
_TASK_START_ERROR = "无法启动任务。"
_TASK_NOT_FOUND_ERROR = "未找到任务。"
_CREDENTIAL_UPDATE_ERROR = "无法更新凭据。"
_APPROVAL_STALE_ERROR = "审批请求已失效。"
_MEMORY_NOT_FOUND_ERROR = "未找到记忆。"
_MEMORY_STORE_NOT_FOUND_ERROR = "未找到记忆存储。"
_PROJECT_NOT_FOUND_ERROR = "未找到项目。"
_COMMAND_RULE_NOT_FOUND_ERROR = "未找到命令规则。"


def create_app(mode: str, services: WebServices) -> FastAPI:
    """Create local HTML and JSON adapters over one shared LocalRuntime."""
    if mode != "local":
        raise ValueError("create_app only supports local mode")

    app = FastAPI()
    runtime = LocalRuntime(services)
    _mark_restored_tasks_interrupted(runtime)
    app.state.runtime = runtime
    app.state.local = _RuntimeStateView(runtime)

    @app.exception_handler(RequestValidationError)
    async def local_validation_error(_request: Request, _error: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "请求参数无效。"})

    @app.exception_handler(StarletteHTTPException)
    async def local_http_error(request: Request, error: StarletteHTTPException) -> Response:
        if (
            type(error) is StarletteHTTPException
            and error.status_code == 404
            and request.url.path.startswith("/api/v1/")
        ):
            return JSONResponse(status_code=404, content={"detail": "未找到资源。"})
        return await http_exception_handler(request, error)

    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
    app.include_router(create_api_router(runtime))

    def credential_status() -> tuple[CredentialStatus, str | None]:
        try:
            return runtime.credential_status(), None
        except CredentialBackendUnavailableError:
            return CredentialStatus(configured=False), _CREDENTIAL_ERROR

    def render_setup(request: Request, *, error: str | None = None, status_code: int = 200) -> HTMLResponse:
        status, credential_error = credential_status()
        if credential_error is not None:
            error = credential_error
            status_code = 503
        config = runtime.config
        return _TEMPLATES.TemplateResponse(
            request,
            "setup.html",
            {
                "request": request,
                "page": "setup",
                "error": error,
                "configured": status.configured,
                "project_root": str(runtime.project_root) if runtime.project_root else "",
                "source_dirs": _display_paths(config.source_dirs) if config else "src",
                "test_dirs": _display_paths(config.test_dirs) if config else "tests",
                "pytest_command": shlex.join(config.pytest_command) if config else "pytest",
                "model": config.model if config else "deepseek-chat",
                "timeout_seconds": config.timeout_seconds if config else 30,
            },
            status_code=status_code,
        )

    def render_credentials(request: Request, *, error: str | None = None, status_code: int = 200) -> HTMLResponse:
        status, credential_error = credential_status()
        if credential_error is not None:
            error = credential_error
            status_code = 503
        return _TEMPLATES.TemplateResponse(request, "base.html", {"configured": status.configured, "error": error, "page": "credentials"}, status_code=status_code)

    def render_task(request: Request, *, error: str | None = None, status_code: int = 200) -> HTMLResponse:
        tasks = runtime.tasks()
        current = tasks[-1] if tasks else None
        return _TEMPLATES.TemplateResponse(request, "task.html", {"error": error, "page": "tasks", "context_task": current, "configured": runtime.config is not None, "task": current, "tasks": tasks}, status_code=status_code)

    def task_for(task_id: UUID):
        try:
            return runtime.task(task_id)
        except RuntimeTaskNotFoundError as error:
            raise HTTPException(status_code=404, detail=_TASK_NOT_FOUND_ERROR) from error

    def task_events(task_id: UUID) -> list[StoredRunEvent]:
        try:
            return runtime.events(task_id)
        except RuntimeTaskNotFoundError as error:
            raise HTTPException(status_code=404, detail=_TASK_NOT_FOUND_ERROR) from error

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        return render_setup(request) if runtime.config is None else render_task(request)

    @app.get("/setup", response_class=HTMLResponse)
    async def setup_page(request: Request) -> HTMLResponse:
        return render_setup(request)

    @app.post("/setup", response_class=HTMLResponse)
    async def setup(request: Request) -> Response:
        if _has_active_task(runtime):
            return render_task(request, error=_ACTIVE_TASK_ERROR, status_code=409)
        form = await request.form()
        status, credential_error = credential_status()
        if credential_error is not None:
            return render_setup(request, error=_CREDENTIAL_ERROR, status_code=503)
        try:
            project_root, config, api_key = _validated_setup(form, require_api_key=not status.configured)
            runtime.setup(project_root, config, api_key if api_key.strip() else None)
        except RuntimeBusyError:
            return render_task(request, error=_ACTIVE_TASK_ERROR, status_code=409)
        except CredentialBackendUnavailableError:
            return render_setup(request, error=_CREDENTIAL_ERROR, status_code=503)
        except (TypeError, ValueError, ValidationError, OSError, RuntimeError):
            return render_setup(request, error=_SETUP_ERROR, status_code=422)
        return RedirectResponse("/tasks/new", status_code=303)

    @app.get("/tasks/new", response_class=HTMLResponse)
    async def new_task(request: Request) -> Response:
        if runtime.config is None:
            return RedirectResponse("/", status_code=303)
        return render_task(request)

    @app.post("/tasks", response_class=HTMLResponse)
    async def create_task(request: Request) -> Response:
        form = await request.form()
        description = str(form.get("description", "")).strip()
        bugfix_target = str(form.get("bugfix_target", "")).strip()
        try:
            task_mode = TaskMode(str(form.get("mode", "")))
            if not description or (task_mode is TaskMode.BUGFIX and not bugfix_target):
                raise ValueError("invalid task form")
            task = runtime.create_task(description, task_mode, bugfix_target or None)
        except RuntimeNotConfiguredError:
            return render_setup(request, error=_SETUP_ERROR, status_code=422)
        except RuntimeBusyError:
            return render_task(request, error=_ACTIVE_TASK_ERROR, status_code=409)
        except (ValueError, ValidationError, OSError):
            return render_task(request, error=_TASK_START_ERROR, status_code=422)
        Thread(target=runtime.run, args=(task.id,), daemon=True).start()
        return RedirectResponse(f"/tasks/{task.id}", status_code=303)

    @app.get("/tasks/{task_id}", response_class=HTMLResponse)
    async def task_detail(task_id: UUID, request: Request) -> HTMLResponse:
        task = task_for(task_id)
        events = task_events(task_id)
        approval_event = next((event for event in reversed(events) if event.task_status is TaskStatus.WAITING_APPROVAL and event.policy_verdict is PolicyVerdict.APPROVAL_REQUIRED and event.action_hash is not None), None)
        return _TEMPLATES.TemplateResponse(request, "task_detail.html", {"task": task, "events": events, "approval_event": approval_event, "terminal": task.status in _TERMINAL_STATUSES, "page": "task_detail", "context_task": task})

    @app.get("/tasks/{task_id}/events")
    async def task_event_feed(task_id: UUID) -> list[dict[str, object]]:
        task_for(task_id)
        return [event.model_dump(mode="json") for event in task_events(task_id)]

    @app.post("/tasks/{task_id}/approval", response_class=HTMLResponse)
    async def resolve_task_approval(task_id: UUID, request: Request) -> Response:
        task = task_for(task_id)
        form = await request.form()
        action_hash = str(form.get("action_hash", ""))
        decision = str(form.get("decision", ""))
        if not is_approval_decision(decision):
            raise HTTPException(status_code=409, detail=_APPROVAL_STALE_ERROR)
        was_waiting = task.status is TaskStatus.WAITING_APPROVAL
        try:
            accepted = runtime.resolve_approval(task_id, action_hash, decision)
        except RuntimeBusyError:
            raise HTTPException(status_code=409, detail=_ACTIVE_TASK_ERROR) from None
        except RuntimeTaskNotFoundError:
            raise HTTPException(status_code=404, detail=_TASK_NOT_FOUND_ERROR) from None
        if accepted:
            Thread(target=runtime.run, args=(task_id,), daemon=True).start()
            return RedirectResponse(f"/tasks/{task_id}", status_code=303)
        if decision == "reject" and was_waiting and runtime.task(task_id).status is TaskStatus.BLOCKED:
            return RedirectResponse(f"/tasks/{task_id}", status_code=303)
        raise HTTPException(status_code=409, detail=_APPROVAL_STALE_ERROR)

    @app.post("/tasks/{task_id}/cancel", response_class=HTMLResponse)
    async def cancel_task(task_id: UUID, request: Request) -> Response:
        try:
            runtime.cancel(task_id)
        except RuntimeTaskNotFoundError:
            return render_task(request, error=_TASK_NOT_FOUND_ERROR, status_code=404)
        except RuntimeBusyError:
            return render_task(request, error=_ACTIVE_TASK_ERROR, status_code=409)
        return RedirectResponse("/tasks/new", status_code=303)

    @app.get("/memories", response_class=HTMLResponse)
    async def memories(request: Request) -> HTMLResponse:
        try:
            proposals = runtime.memory_proposals()
            approved = runtime.memories()
        except RuntimeNotConfiguredError:
            raise HTTPException(status_code=404, detail=_MEMORY_STORE_NOT_FOUND_ERROR) from None
        return _TEMPLATES.TemplateResponse(request, "memory.html", {"proposals": proposals, "approved": approved, "page": "memories"})

    @app.post("/memories/{memory_id}/approve", response_class=HTMLResponse)
    async def approve_memory(memory_id: UUID) -> Response:
        try:
            runtime.approve_memory(memory_id)
        except (RuntimeNotConfiguredError, RuntimeMemoryNotFoundError):
            raise HTTPException(status_code=404, detail=_MEMORY_NOT_FOUND_ERROR) from None
        except RuntimeBusyError:
            raise HTTPException(status_code=409, detail=_ACTIVE_TASK_ERROR) from None
        return RedirectResponse("/memories", status_code=303)

    @app.post("/memories/{memory_id}/delete", response_class=HTMLResponse)
    async def delete_memory(memory_id: UUID) -> Response:
        try:
            runtime.delete_memory(memory_id)
        except (RuntimeNotConfiguredError, RuntimeMemoryNotFoundError):
            raise HTTPException(status_code=404, detail=_MEMORY_NOT_FOUND_ERROR) from None
        except RuntimeBusyError:
            raise HTTPException(status_code=409, detail=_ACTIVE_TASK_ERROR) from None
        return RedirectResponse("/memories", status_code=303)

    @app.get("/settings/credentials", response_class=HTMLResponse)
    async def credentials_page(request: Request) -> HTMLResponse:
        return render_credentials(request)

    @app.post("/settings/credentials", response_class=HTMLResponse)
    async def update_credentials(request: Request) -> Response:
        api_key = str((await request.form()).get("api_key", ""))
        if not api_key.strip():
            return render_credentials(request, error=_CREDENTIAL_UPDATE_ERROR, status_code=422)
        try:
            runtime.update_credential(api_key)
        except CredentialBackendUnavailableError:
            return render_credentials(request, error=_CREDENTIAL_ERROR, status_code=503)
        except RuntimeBusyError:
            return render_credentials(request, error=_ACTIVE_TASK_ERROR, status_code=409)
        return RedirectResponse("/settings/credentials", status_code=303)

    @app.post("/settings/credentials/clear", response_class=HTMLResponse)
    async def clear_credentials(request: Request) -> Response:
        try:
            runtime.clear_credential()
        except CredentialBackendUnavailableError:
            return render_credentials(request, error=_CREDENTIAL_ERROR, status_code=503)
        except RuntimeBusyError:
            return render_credentials(request, error=_ACTIVE_TASK_ERROR, status_code=409)
        return RedirectResponse("/settings/credentials", status_code=303)

    @app.get("/settings/command-rules", response_class=HTMLResponse)
    async def command_rules(request: Request) -> HTMLResponse:
        try:
            rules = [(rule, _command_rule_projection(rule)) for rule in runtime.command_rules()]
        except RuntimeNotConfiguredError:
            raise HTTPException(status_code=404, detail=_PROJECT_NOT_FOUND_ERROR) from None
        return _TEMPLATES.TemplateResponse(request, "command_rules.html", {"rules": rules, "page": "command_rules"})

    @app.post("/settings/command-rules/{rule_id}/delete", response_class=HTMLResponse)
    async def delete_command_rule(rule_id: str) -> Response:
        try:
            runtime.delete_command_rule(rule_id)
        except RuntimeNotConfiguredError:
            raise HTTPException(status_code=404, detail=_PROJECT_NOT_FOUND_ERROR) from None
        except RuntimeCommandRuleNotFoundError:
            raise HTTPException(status_code=404, detail=_COMMAND_RULE_NOT_FOUND_ERROR) from None
        except RuntimeBusyError:
            raise HTTPException(status_code=409, detail=_ACTIVE_TASK_ERROR) from None
        return RedirectResponse("/settings/command-rules", status_code=303)

    return app


class _RuntimeStateView:
    """Read-only compatibility view for templates while the runtime owns all state."""

    def __init__(self, runtime: LocalRuntime) -> None:
        self._runtime = runtime

    @property
    def project_root(self) -> Path | None:
        return self._runtime.project_root

    @property
    def config(self) -> HarnessConfig | None:
        return self._runtime.config

    @property
    def memory_store(self) -> Any:
        return self._runtime._memory_store

    @property
    def tasks(self) -> dict[UUID, Any]:
        return {task.id: task for task in self._runtime.tasks()}

    @property
    def task(self) -> Any:
        tasks = self._runtime.tasks()
        return tasks[-1] if tasks else None


def _validated_setup(form: Any, *, require_api_key: bool = True) -> tuple[Path, HarnessConfig, str]:
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
    return project_root, HarnessConfig(source_dirs=source_dirs, test_dirs=test_dirs, pytest_command=command, model=model, timeout_seconds=int(str(form.get("timeout_seconds", "")))), api_key


def _command_rule_projection(rule: CommandApprovalRule) -> str:
    if rule.kind is CommandRuleKind.GIT_DIFF_CHECK:
        return "Git whitespace check"
    if rule.kind is CommandRuleKind.GIT_PUSH:
        return f"Git push to origin/{rule.branch}"
    return f"Python package install: {', '.join(rule.package_specs)}"


def _configured_directories(value: str, project_root: Path) -> tuple[Path, ...]:
    entries = tuple(Path(entry.strip()) for entry in value.replace("\n", ",").split(",") if entry.strip())
    if not entries or any(entry.is_absolute() or ".." in entry.parts or not (project_root / entry).is_dir() for entry in entries):
        raise ValueError("configured directory is invalid")
    return entries


def _display_paths(paths: tuple[Path, ...]) -> str:
    return ", ".join(str(path) for path in paths)


def _has_active_task(runtime: LocalRuntime) -> bool:
    return any(task.status in _ACTIVE_STATUSES for task in runtime.tasks())


def _mark_restored_tasks_interrupted(runtime: LocalRuntime) -> None:
    """Retain the old WebUI restart contract without creating another task store."""
    for project_root in dict.fromkeys(runtime._task_roots.values()):
        EventStore(project_root).mark_unfinished_interrupted()


def _system_keyring() -> KeyringBackend:
    return keyring.get_keyring()


def _deepseek_transport(api_key: str, *, timeout_seconds: int, openai_factory: Callable[..., Any] | None = None) -> Any:
    factory = OpenAI if openai_factory is None else openai_factory
    return factory(api_key=api_key, base_url="https://api.deepseek.com", timeout=timeout_seconds, max_retries=0)


def _current_git_branch(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(["git", "-C", str(project_root), "symbolic-ref", "--quiet", "--short", "HEAD"], capture_output=True, text=True, check=False, shell=False, timeout=5)
    except Exception:
        return None
    branch = completed.stdout.strip()
    return branch if completed.returncode == 0 and branch else None


def local_services(*, transport_factory: Callable[[str], Any] | None = None, current_branch_provider: Callable[[Path], str | None] | None = None) -> WebServices:
    """Compose real local dependencies without looking up a key at startup."""
    credentials = CredentialService(_system_keyring())
    branch_provider = current_branch_provider or _current_git_branch

    def orchestrator_factory(project_root: Path, config: HarnessConfig, memory_store: Any) -> TaskOrchestrator:
        configured_transport_factory = transport_factory or (lambda api_key: _deepseek_transport(api_key, timeout_seconds=config.timeout_seconds))
        llm = DeepSeekClient(credentials.get_key, config.model, configured_transport_factory)
        return TaskOrchestrator(project_root, llm, memory_store=memory_store, current_branch_provider=lambda: branch_provider(project_root))

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
