"""ASGI coverage for the local setup and task-control surface."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
import yaml

from guardedpy.config import HarnessConfig
from guardedpy.credentials import CredentialStatus
from guardedpy.domain import TaskMode, TaskState, TaskStatus


def _web_module() -> Any:
    """Turn the expected missing WebUI module into a useful red failure."""
    try:
        return importlib.import_module("guardedpy.web")
    except ModuleNotFoundError as error:
        pytest.fail(f"local WebUI factory is missing: {error.name}")


async def _request(app: Any, method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


@dataclass
class FakeCredentials:
    configured: bool = False
    set_calls: list[str] = field(default_factory=list)

    def status(self) -> CredentialStatus:
        return CredentialStatus(configured=self.configured)

    def set_key(self, key: str) -> None:
        self.set_calls.append(key)
        self.configured = True

    def get_key(self) -> str:
        if not self.configured:
            raise RuntimeError("key is absent")
        return self.set_calls[-1]

    def clear_key(self) -> None:
        self.configured = False


@dataclass
class FakeOrchestrator:
    timeline: list[str]
    submitted: list[TaskState] = field(default_factory=list)
    cancelled: list[UUID] = field(default_factory=list)

    def submit(self, task: TaskState) -> TaskState:
        self.timeline.append("submit")
        self.submitted.append(task)
        return task

    def run(self, task: TaskState) -> TaskState:
        self.timeline.append("run")
        task.status = TaskStatus.RUNNING
        return task

    def cancel(self, task_id: UUID) -> TaskState:
        self.timeline.append("cancel")
        self.cancelled.append(task_id)
        task = self.submitted[0]
        task.status = TaskStatus.CANCELLED
        return task


def _project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    return root


def _setup_data(root: Path, **overrides: str) -> dict[str, str]:
    data = {
        "project_root": str(root),
        "source_dirs": "src",
        "test_dirs": "tests",
        "pytest_command": "pytest",
        "model": "deepseek-chat",
        "timeout_seconds": "30",
        "api_key": "secret-key",
    }
    data.update(overrides)
    return data


def _app(credentials: FakeCredentials, orchestrator: FakeOrchestrator) -> Any:
    web = _web_module()
    return web.create_app(
        "local",
        web.WebServices(credentials=credentials, orchestrator_factory=lambda root, config, memory: orchestrator),
    )


def test_setup_saves_a_nonsecret_validated_snapshot_without_echoing_the_key(tmp_path: Path) -> None:
    """Catches setup persisting or rendering an API key instead of sending it straight to keyring."""
    root = _project_root(tmp_path)
    credentials = FakeCredentials()
    orchestrator = FakeOrchestrator([])
    app = _app(credentials, orchestrator)

    response = asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root)))
    credentials_page = asyncio.run(_request(app, "GET", "/settings/credentials"))

    assert response.status_code == 303
    assert response.headers["location"] == "/tasks/new"
    assert credentials.set_calls == ["secret-key"]
    assert "secret-key" not in response.text
    assert "secret-key" not in credentials_page.text
    snapshot = yaml.safe_load((root / "harness.yaml").read_text())
    assert snapshot == {
        "source_dirs": ["src"],
        "test_dirs": ["tests"],
        "pytest_command": ["pytest"],
        "model": "deepseek-chat",
        "timeout_seconds": 30,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"project_root": "missing-root"},
        {"source_dirs": "missing"},
        {"pytest_command": ""},
        {"timeout_seconds": "not-a-number"},
        {"api_key": ""},
    ],
)
def test_invalid_setup_has_one_generic_error_and_no_side_effect(tmp_path: Path, overrides: dict[str, str]) -> None:
    """Catches malformed setup writing state or exposing validation details."""
    root = _project_root(tmp_path)
    credentials = FakeCredentials()
    app = _app(credentials, FakeOrchestrator([]))

    response = asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root, **overrides)))

    assert response.status_code == 422
    assert response.text.count("Setup could not be saved.") == 1
    assert credentials.set_calls == []
    assert not (root / "harness.yaml").exists()
    assert "secret-key" not in response.text


def test_snapshot_write_failure_does_not_store_the_submitted_key(tmp_path: Path) -> None:
    """Catches setup storing a key when the validated configuration snapshot cannot be written."""
    root = _project_root(tmp_path)
    (root / "harness.yaml").mkdir()
    credentials = FakeCredentials()
    app = _app(credentials, FakeOrchestrator([]))

    response = asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root)))

    assert response.status_code == 422
    assert response.text.count("Setup could not be saved.") == 1
    assert credentials.set_calls == []


def test_task_creation_submits_before_starting_one_daemon_thread_and_rejects_a_second_active_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a run thread started before registration or parallel active task admission."""
    root = _project_root(tmp_path)
    credentials = FakeCredentials()
    timeline: list[str] = []
    orchestrator = FakeOrchestrator(timeline)
    app = _app(credentials, orchestrator)
    web = _web_module()

    class CapturingThread:
        instances: list["CapturingThread"] = []

        def __init__(self, *, target: Any, args: tuple[TaskState, ...], daemon: bool) -> None:
            timeline.append("thread")
            self.target = target
            self.args = args
            self.daemon = daemon
            self.started = False
            self.instances.append(self)

        def start(self) -> None:
            self.started = True

    monkeypatch.setattr(web, "Thread", CapturingThread)
    assert asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root))).status_code == 303

    created = asyncio.run(
        _request(app, "POST", "/tasks", data={"mode": "bugfix", "description": "Repair a test"})
    )
    duplicate = asyncio.run(
        _request(app, "POST", "/tasks", data={"mode": "feature", "description": "Add another"})
    )

    assert created.status_code == 303
    assert created.headers["location"] == "/tasks/new"
    assert [task.mode for task in orchestrator.submitted] == [TaskMode.BUGFIX]
    assert timeline == ["submit", "thread"]
    assert len(CapturingThread.instances) == 1
    assert CapturingThread.instances[0].daemon is True
    assert CapturingThread.instances[0].started is True
    assert duplicate.status_code == 409


def test_blank_task_description_is_rejected_and_cancellation_delegates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches blank task admission and a cancel control that bypasses the orchestrator."""
    root = _project_root(tmp_path)
    orchestrator = FakeOrchestrator([])
    app = _app(FakeCredentials(), orchestrator)
    web = _web_module()

    class DormantThread:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def start(self) -> None:
            return None

    monkeypatch.setattr(web, "Thread", DormantThread)
    asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root)))

    blank = asyncio.run(_request(app, "POST", "/tasks", data={"mode": "feature", "description": "  "}))
    created = asyncio.run(
        _request(app, "POST", "/tasks", data={"mode": "feature", "description": "Add a feature"})
    )
    task_id = orchestrator.submitted[0].id
    cancelled = asyncio.run(_request(app, "POST", f"/tasks/{task_id}/cancel"))

    assert blank.status_code == 422
    assert created.status_code == 303
    assert cancelled.status_code == 303
    assert cancelled.headers["location"] == "/tasks/new"
    assert orchestrator.cancelled == [task_id]


def test_local_factory_rejects_demo_mode() -> None:
    """Catches an accidental demo-mode entry point sharing local project capabilities."""
    web = _web_module()

    with pytest.raises(ValueError, match="local"):
        web.create_app("demo", web.WebServices(FakeCredentials(), lambda root, config, memory: FakeOrchestrator([])))


def test_local_services_defers_keyring_lookup_until_a_completion_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches application construction reading the keyring before a provider completion is requested."""
    web = _web_module()
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    calls: list[str] = []

    class Keyring:
        def get_password(self, service_name: str, username: str) -> str:
            calls.append("get")
            return "secret-key"

        def set_password(self, service_name: str, username: str, password: str) -> None:
            del service_name, username, password

        def delete_password(self, service_name: str, username: str) -> None:
            del service_name, username

    class Completions:
        def create(self, **kwargs: Any) -> Any:
            del kwargs
            return type("Response", (), {"choices": [type("Choice", (), {"message": type("Message", (), {"content": '{"kind":"finish","summary":"stop","status":"blocked"}'})()})()]})()

    transport = type("Transport", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    monkeypatch.setattr(web, "_system_keyring", lambda: Keyring())
    services = web.local_services(transport_factory=lambda key: transport)
    config = HarnessConfig(source_dirs=(Path("src"),), test_dirs=(Path("tests"),), pytest_command=("pytest",))
    orchestrator = services.orchestrator_factory(tmp_path, config, web.MemoryStore(tmp_path))

    assert calls == []
    orchestrator.run(TaskState(description="Stop", mode=TaskMode.BUGFIX, config=config))
    assert calls == ["get"]
