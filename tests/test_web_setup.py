"""ASGI coverage for the local setup and task-control surface."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import importlib
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import UUID

import httpx
import pytest
import yaml

from guardedpy.config import HarnessConfig, local_state_path, project_config_path
from guardedpy.credentials import CredentialBackendUnavailableError, CredentialStatus
from guardedpy.domain import TaskMode, TaskState, TaskStatus
from guardedpy.events import EventStore


def _web_module() -> Any:
    """Turn the expected missing WebUI module into a useful red failure."""
    try:
        return importlib.import_module("guardedpy.web")
    except ModuleNotFoundError as error:
        pytest.fail(f"local WebUI factory is missing: {error.name}")


@pytest.fixture(autouse=True)
def _isolated_application_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep every WebUI test's product state outside the developer's real profile."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


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

    def clear_key(self) -> None:
        self.configured = False


@dataclass
class UnavailableCredentials:
    def status(self) -> CredentialStatus:
        raise CredentialBackendUnavailableError("backend leaked-secret is unavailable")

    def set_key(self, key: str) -> None:
        raise CredentialBackendUnavailableError(f"backend rejected {key}")

    def clear_key(self) -> None:
        raise CredentialBackendUnavailableError("backend leaked-secret is unavailable")


@dataclass
class SetUnavailableCredentials(FakeCredentials):
    def set_key(self, key: str) -> None:
        raise CredentialBackendUnavailableError(f"backend rejected {key}")


@dataclass
class UnexpectedSetFailureCredentials(FakeCredentials):
    def set_key(self, key: str) -> None:
        raise RuntimeError(f"unexpected failure for {key}")


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


def test_ui_credential_protocol_exposes_only_nonsecret_operations() -> None:
    """Catches the local WebUI dependency contract regaining a secret-read operation."""
    web = _web_module()

    public_operations = {
        name
        for name, value in vars(web.CredentialPort).items()
        if not name.startswith("_") and callable(value)
    }

    assert public_operations == {"status", "set_key", "clear_key"}


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
    snapshot_path = project_config_path(root)
    snapshot = yaml.safe_load(snapshot_path.read_text())
    assert snapshot == {
        "source_dirs": ["src"],
        "test_dirs": ["tests"],
        "pytest_command": ["pytest"],
        "model": "deepseek-chat",
        "timeout_seconds": 30,
    }
    assert not (root / "harness.yaml").exists()
    assert "secret-key" not in local_state_path().read_text()


def test_setup_get_shows_current_nonsecret_values_and_blank_key_keeps_configured_credential(
    tmp_path: Path,
) -> None:
    """Catches reconfiguration requiring or replacing a credential that is already stored."""
    root = _project_root(tmp_path)
    credentials = FakeCredentials()
    app = _app(credentials, FakeOrchestrator([]))
    assert asyncio.run(
        _request(app, "POST", "/setup", data=_setup_data(root))
    ).status_code == 303

    page = asyncio.run(_request(app, "GET", "/setup"))
    reconfigured = asyncio.run(
        _request(
            app,
            "POST",
            "/setup",
            data=_setup_data(
                root,
                api_key="",
                model="deepseek-reasoner",
                timeout_seconds="45",
            ),
        )
    )

    assert page.status_code == 200
    assert str(root.resolve()) in page.text
    assert 'value="src"' in page.text
    assert 'value="tests"' in page.text
    assert 'value="pytest"' in page.text
    assert 'value="deepseek-chat"' in page.text
    assert 'value="30"' in page.text
    assert "secret-key" not in page.text
    assert reconfigured.status_code == 303
    assert credentials.set_calls == ["secret-key"]
    assert yaml.safe_load(project_config_path(root).read_text())["model"] == "deepseek-reasoner"


def test_setup_key_failure_restores_external_config_and_local_index_exactly(
    tmp_path: Path,
) -> None:
    """Catches failed credential replacement partially committing either state file."""
    root = _project_root(tmp_path)
    initial_app = _app(FakeCredentials(), FakeOrchestrator([]))
    assert asyncio.run(
        _request(initial_app, "POST", "/setup", data=_setup_data(root))
    ).status_code == 303
    config_before = project_config_path(root).read_bytes()
    index_before = local_state_path().read_bytes()

    failing_app = _app(
        SetUnavailableCredentials(configured=True),
        FakeOrchestrator([]),
    )
    failed = asyncio.run(
        _request(
            failing_app,
            "POST",
            "/setup",
            data=_setup_data(
                root,
                api_key="never-persist-this",
                model="deepseek-reasoner",
            ),
        )
    )

    assert failed.status_code == 503
    assert project_config_path(root).read_bytes() == config_before
    assert local_state_path().read_bytes() == index_before
    assert "never-persist-this" not in failed.text + local_state_path().read_text()


@pytest.mark.parametrize(
    "local_state",
    [
        None,
        "selected_project: [broken\n",
        "selected_project: relative/project\ntask_roots: {}\n",
        "selected_project: /missing/project\ntask_roots: {}\nsecret: leaked\n",
    ],
)
def test_missing_or_malformed_startup_state_fails_closed_to_setup(
    tmp_path: Path, local_state: str | None
) -> None:
    """Catches invalid external pointers being trusted or exposing startup diagnostics."""
    if local_state is not None:
        path = local_state_path()
        path.parent.mkdir(parents=True)
        path.write_text(local_state)
    factory_calls: list[Path] = []
    web = _web_module()
    app = web.create_app(
        "local",
        web.WebServices(
            credentials=FakeCredentials(configured=True),
            orchestrator_factory=lambda root, config, memory: factory_calls.append(root),
        ),
    )

    response = asyncio.run(_request(app, "GET", "/"))

    assert response.status_code == 200
    assert "Connect one project" in response.text
    assert "broken" not in response.text
    assert "leaked" not in response.text
    assert app.state.local.config is None
    assert factory_calls == []


def test_all_task_active_gate_and_old_task_controls_use_uuid_indexed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches lifecycle routes consulting only the most recently created task/orchestrator."""
    root = _project_root(tmp_path / "first")
    replacement_root = _project_root(tmp_path / "replacement")
    web = _web_module()
    orchestrators: list[CompletingOrchestrator] = []

    class InlineThread:
        def __init__(self, *, target: Any, args: tuple[TaskState, ...], daemon: bool) -> None:
            del daemon
            self.target = target
            self.args = args

        def start(self) -> None:
            self.target(*self.args)

    @dataclass
    class CompletingOrchestrator:
        tasks: dict[UUID, TaskState] = field(default_factory=dict)
        approvals: list[UUID] = field(default_factory=list)
        cancellations: list[UUID] = field(default_factory=list)

        def submit(self, task: TaskState) -> TaskState:
            self.tasks[task.id] = task
            return task

        def run(self, task: TaskState) -> TaskState:
            task.status = TaskStatus.BLOCKED
            return task

        def cancel(self, task_id: UUID) -> TaskState:
            self.cancellations.append(task_id)
            self.tasks[task_id].status = TaskStatus.CANCELLED
            return self.tasks[task_id]

        def resolve_approval(
            self, task_id: UUID, action_hash: str, *, decision: str
        ) -> bool:
            del action_hash, decision
            self.approvals.append(task_id)
            return False

    def factory(root: Path, config: Any, memory: Any) -> CompletingOrchestrator:
        del root, config, memory
        orchestrator = CompletingOrchestrator()
        orchestrators.append(orchestrator)
        return orchestrator

    monkeypatch.setattr(web, "Thread", InlineThread)
    app = web.create_app(
        "local", web.WebServices(credentials=FakeCredentials(), orchestrator_factory=factory)
    )
    assert asyncio.run(
        _request(app, "POST", "/setup", data=_setup_data(root))
    ).status_code == 303
    for description in ("First", "Second"):
        assert asyncio.run(
            _request(
                app,
                "POST",
                "/tasks",
                data={"mode": "feature", "description": description},
            )
        ).status_code == 303

    first_id = next(iter(orchestrators[0].tasks))
    first_task = orchestrators[0].tasks[first_id]
    first_task.status = TaskStatus.PENDING
    blocked_setup = asyncio.run(
        _request(app, "POST", "/setup", data=_setup_data(replacement_root))
    )
    assert blocked_setup.status_code == 409
    assert not project_config_path(replacement_root).exists()

    first_task.status = TaskStatus.WAITING_APPROVAL
    stale_approval = asyncio.run(
        _request(
            app,
            "POST",
            f"/tasks/{first_id}/approval",
            data={"action_hash": "stale", "decision": "once"},
        )
    )
    first_task.status = TaskStatus.BLOCKED
    cancelled = asyncio.run(_request(app, "POST", f"/tasks/{first_id}/cancel"))

    assert stale_approval.status_code == 409
    assert orchestrators[0].approvals == [first_id]
    assert orchestrators[1].approvals == []
    assert cancelled.status_code == 303
    assert orchestrators[0].cancellations == [first_id]
    assert orchestrators[1].cancellations == []


def test_credential_update_clear_and_unavailable_backend_do_not_echo_key(tmp_path: Path) -> None:
    """Catches credential mutations leaking a submitted key or backend diagnostics."""
    root = _project_root(tmp_path)
    credentials = FakeCredentials()
    app = _app(credentials, FakeOrchestrator([]))
    assert asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root))).status_code == 303

    updated = asyncio.run(
        _request(app, "POST", "/settings/credentials", data={"api_key": "new-secret"})
    )
    cleared = asyncio.run(_request(app, "POST", "/settings/credentials/clear"))
    page = asyncio.run(_request(app, "GET", "/settings/credentials"))

    assert updated.status_code == 303
    assert cleared.status_code == 303
    assert credentials.set_calls == ["secret-key", "new-secret"]
    assert "new-secret" not in updated.text + cleared.text + page.text
    assert 'type="password"' in page.text
    assert 'action="/settings/credentials/clear"' in page.text

    unavailable = _app(UnavailableCredentials(), FakeOrchestrator([]))
    unavailable_page = asyncio.run(_request(unavailable, "GET", "/settings/credentials"))
    unavailable_update = asyncio.run(
        _request(
            unavailable,
            "POST",
            "/settings/credentials",
            data={"api_key": "never-echo-this"},
        )
    )
    unavailable_clear = asyncio.run(
        _request(unavailable, "POST", "/settings/credentials/clear")
    )

    assert unavailable_page.status_code == 503
    assert unavailable_update.status_code == 503
    assert unavailable_clear.status_code == 503
    body = unavailable_page.text + unavailable_update.text + unavailable_clear.text
    assert body.count("Credential store is unavailable.") == 3
    assert "never-echo-this" not in body
    assert "leaked-secret" not in body


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
    assert not project_config_path(root).exists()
    assert not local_state_path().exists()
    assert "secret-key" not in response.text


def test_snapshot_write_failure_does_not_store_the_submitted_key(tmp_path: Path) -> None:
    """Catches setup storing a key when the validated configuration snapshot cannot be written."""
    root = _project_root(tmp_path)
    snapshot = project_config_path(root)
    snapshot.parent.mkdir(parents=True)
    snapshot.mkdir()
    credentials = FakeCredentials()
    app = _app(credentials, FakeOrchestrator([]))

    response = asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root)))

    assert response.status_code == 422
    assert response.text.count("Setup could not be saved.") == 1
    assert credentials.set_calls == []


@pytest.mark.parametrize("existing_snapshot", [None, b"original snapshot bytes\n"])
def test_setup_keyring_failure_restores_the_previous_snapshot_and_uses_safe_error(
    tmp_path: Path, existing_snapshot: bytes | None
) -> None:
    """Catches partial setup committing a new snapshot after keyring mutation fails."""
    root = _project_root(tmp_path)
    snapshot = project_config_path(root)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if existing_snapshot is not None:
        snapshot.write_bytes(existing_snapshot)
    app = _app(SetUnavailableCredentials(), FakeOrchestrator([]))

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/setup",
            data=_setup_data(root, api_key="never-render-this-key"),
        )
    )

    assert response.status_code == 503
    assert response.text.count("Credential store is unavailable.") == 1
    assert "never-render-this-key" not in response.text
    assert "backend rejected" not in response.text
    if existing_snapshot is None:
        assert snapshot.exists() is False
    else:
        assert snapshot.read_bytes() == existing_snapshot


def test_unexpected_setup_key_failure_is_not_misreported_as_keyring_unavailable(
    tmp_path: Path,
) -> None:
    """Catches unrelated setup exceptions being relabelled as a credential backend outage."""
    root = _project_root(tmp_path)
    snapshot = project_config_path(root)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(b"keep this exact snapshot\n")
    app = _app(UnexpectedSetFailureCredentials(), FakeOrchestrator([]))

    response = asyncio.run(
        _request(app, "POST", "/setup", data=_setup_data(root, api_key="hidden-key"))
    )

    assert response.status_code == 422
    assert response.text.count("Setup could not be saved.") == 1
    assert "Credential store is unavailable." not in response.text
    assert "hidden-key" not in response.text
    assert snapshot.read_bytes() == b"keep this exact snapshot\n"


def test_active_task_rejects_setup_replacement_before_any_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches setup replacement moving an active task to another root or key."""
    first_root = _project_root(tmp_path / "first")
    second_root = _project_root(tmp_path / "second")
    credentials = FakeCredentials()
    orchestrator = FakeOrchestrator([])
    app = _app(credentials, orchestrator)
    web = _web_module()

    class DormantThread:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def start(self) -> None:
            return None

    monkeypatch.setattr(web, "Thread", DormantThread)
    assert asyncio.run(_request(app, "POST", "/setup", data=_setup_data(first_root))).status_code == 303
    assert asyncio.run(
        _request(app, "POST", "/tasks", data={"mode": "feature", "description": "Keep root"})
    ).status_code == 303

    replaced = asyncio.run(
        _request(
            app,
            "POST",
            "/setup",
            data=_setup_data(second_root, api_key="second-secret"),
        )
    )

    assert replaced.status_code == 409
    assert app.state.local.project_root == first_root.resolve()
    assert credentials.set_calls == ["secret-key"]
    assert not project_config_path(second_root).exists()
    assert "second-secret" not in replaced.text


def test_active_task_rejects_setup_before_form_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches malformed setup or parser work taking precedence over the active-task gate."""
    root = _project_root(tmp_path)
    app = _app(FakeCredentials(), FakeOrchestrator([]))
    web = _web_module()

    class DormantThread:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def start(self) -> None:
            return None

    monkeypatch.setattr(web, "Thread", DormantThread)
    assert asyncio.run(
        _request(app, "POST", "/setup", data=_setup_data(root))
    ).status_code == 303
    assert asyncio.run(
        _request(
            app,
            "POST",
            "/tasks",
            data={"mode": "feature", "description": "Keep setup locked"},
        )
    ).status_code == 303
    form_calls: list[str] = []

    async def controlled_form(request: Any) -> dict[str, str]:
        form_calls.append(request.url.path)
        return {}

    monkeypatch.setattr(web.Request, "form", controlled_form)

    rejected = asyncio.run(_request(app, "POST", "/setup", data={}))

    assert rejected.status_code == 409
    assert form_calls == []


def test_setup_rechecks_active_task_after_form_parse_races_with_task_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches setup passing its active check before a concurrent task is registered."""
    first_root = _project_root(tmp_path / "first")
    second_root = _project_root(tmp_path / "second")
    credentials = FakeCredentials()
    orchestrator = FakeOrchestrator([])
    app = _app(credentials, orchestrator)
    web = _web_module()

    class DormantThread:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def start(self) -> None:
            return None

    monkeypatch.setattr(web, "Thread", DormantThread)
    assert asyncio.run(_request(app, "POST", "/setup", data=_setup_data(first_root))).status_code == 303

    async def exercise_race() -> tuple[httpx.Response, httpx.Response]:
        setup_is_parsing = asyncio.Event()
        resume_setup = asyncio.Event()
        original_form = web.Request.form

        async def controlled_form(request: Any) -> Any:
            if request.url.path == "/setup":
                setup_is_parsing.set()
                await resume_setup.wait()
            return await original_form(request)

        monkeypatch.setattr(web.Request, "form", controlled_form)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            setup_request = asyncio.create_task(
                client.post(
                    "/setup",
                    data=_setup_data(second_root, api_key="racing-secret"),
                )
            )
            await setup_is_parsing.wait()
            task_response = await client.post(
                "/tasks", data={"mode": "feature", "description": "Win setup race"}
            )
            resume_setup.set()
            return await setup_request, task_response

    setup_response, task_response = asyncio.run(exercise_race())

    assert task_response.status_code == 303
    assert setup_response.status_code == 409
    assert app.state.local.project_root == first_root.resolve()
    assert credentials.set_calls == ["secret-key"]
    assert not project_config_path(second_root).exists()
    assert "racing-secret" not in setup_response.text


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
        _request(
            app,
            "POST",
            "/tasks",
            data={
                "mode": "bugfix",
                "description": "Repair a test",
                "bugfix_target": "tests/test_value.py::test_value_is_fixed",
            },
        )
    )
    duplicate = asyncio.run(
        _request(app, "POST", "/tasks", data={"mode": "feature", "description": "Add another"})
    )

    assert created.status_code == 303
    assert created.headers["location"] == f"/tasks/{orchestrator.submitted[0].id}"
    assert [task.mode for task in orchestrator.submitted] == [TaskMode.BUGFIX]
    assert orchestrator.submitted[0].bugfix_target == "tests/test_value.py::test_value_is_fixed"
    assert timeline == ["submit", "thread"]
    assert len(CapturingThread.instances) == 1
    assert CapturingThread.instances[0].daemon is True
    assert CapturingThread.instances[0].started is True
    assert duplicate.status_code == 409


def test_task_index_failure_rolls_back_sqlite_registration_before_submit_or_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a failed task-index write leaving an unreachable durable pending task."""
    root = _project_root(tmp_path)
    orchestrator = FakeOrchestrator([])
    app = _app(FakeCredentials(), orchestrator)
    web = _web_module()
    assert asyncio.run(
        _request(app, "POST", "/setup", data=_setup_data(root))
    ).status_code == 303
    index_before = local_state_path().read_bytes()
    threads: list[object] = []

    class CapturingThread:
        def __init__(self, **kwargs: Any) -> None:
            threads.append(kwargs)

        def start(self) -> None:
            raise AssertionError("a failed task registration must not start a thread")

    def fail_index_write(project_root: Path, task_roots: dict[UUID, Path]) -> None:
        del project_root, task_roots
        raise OSError("simulated local index failure")

    monkeypatch.setattr(web, "Thread", CapturingThread)
    monkeypatch.setattr(web, "_write_local_state", fail_index_write)

    failed = asyncio.run(
        _request(
            app,
            "POST",
            "/tasks",
            data={"mode": "feature", "description": "Must stay reachable"},
        )
    )

    assert failed.status_code == 422
    assert EventStore(root).tasks() == []
    assert local_state_path().read_bytes() == index_before
    assert orchestrator.submitted == []
    assert app.state.local.tasks == {}
    assert threads == []


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


def test_bugfix_form_requires_and_submits_a_selected_target_while_feature_remains_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a UI that admits targetless bugfixes or forces a target on feature tasks."""
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

    page = asyncio.run(_request(app, "GET", "/tasks/new"))
    missing_target = asyncio.run(
        _request(app, "POST", "/tasks", data={"mode": "bugfix", "description": "Repair parser"})
    )
    feature = asyncio.run(
        _request(app, "POST", "/tasks", data={"mode": "feature", "description": "Add parser docs"})
    )

    assert 'name="bugfix_target"' in page.text
    assert missing_target.status_code == 422
    assert feature.status_code == 303
    assert orchestrator.submitted[0].bugfix_target is None


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
    orchestrator.run(
        TaskState(
            description="Stop",
            mode=TaskMode.BUGFIX,
            bugfix_target="tests/test_value.py::test_value_is_fixed",
            config=config,
        )
    )
    assert calls == ["get"]


def test_local_git_branch_probe_uses_a_fixed_read_only_bounded_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches branch discovery invoking a shell, network command, or unbounded child."""
    web = _web_module()
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, "main\n", "")

    monkeypatch.setattr(web.subprocess, "run", run)

    branch = web._current_git_branch(tmp_path)

    assert branch == "main"
    assert calls == [
        (
            [
                "git",
                "-C",
                str(tmp_path),
                "symbolic-ref",
                "--quiet",
                "--short",
                "HEAD",
            ],
            {
                "capture_output": True,
                "text": True,
                "check": False,
                "shell": False,
                "timeout": 5,
            },
        )
    ]


@pytest.mark.parametrize(
    "outcome",
    [
        subprocess.CompletedProcess(["git"], 1, "main\n", ""),
        subprocess.CompletedProcess(["git"], 0, "\n", ""),
        OSError("git unavailable"),
        subprocess.TimeoutExpired(["git"], 5),
    ],
)
def test_local_git_branch_probe_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: subprocess.CompletedProcess[str] | Exception,
) -> None:
    """Catches probe failures or detached/blank output becoming a trusted branch."""
    web = _web_module()

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(web.subprocess, "run", run)

    assert web._current_git_branch(tmp_path) is None


def test_local_services_uses_live_project_branch_to_pause_exact_push_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches local composition omitting branch governance or executing before approval."""
    web = _web_module()
    import guardedpy.orchestrator as orchestrator_module

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    class Keyring:
        def get_password(self, service_name: str, username: str) -> str:
            del service_name, username
            return "test-key"

        def set_password(self, service_name: str, username: str, password: str) -> None:
            del service_name, username, password

        def delete_password(self, service_name: str, username: str) -> None:
            del service_name, username

    class Completions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            message = type(
                "Message",
                (),
                {
                    "content": (
                        '{"kind":"run_command","summary":"push current branch",'
                        '"args":["git","push","origin","main"]}'
                    )
                },
            )()
            return type(
                "Response",
                (),
                {"choices": [type("Choice", (), {"message": message})()]},
            )()

    transport = type(
        "Transport",
        (),
        {"chat": type("Chat", (), {"completions": Completions()})()},
    )()
    branch_roots: list[Path] = []
    command_calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(web, "_system_keyring", lambda: Keyring())
    monkeypatch.setattr(
        orchestrator_module.subprocess,
        "run",
        lambda arguments, **kwargs: command_calls.append(tuple(arguments))
        or subprocess.CompletedProcess(arguments, 0, "", ""),
    )
    services = web.local_services(
        transport_factory=lambda key: transport,
        current_branch_provider=lambda root: branch_roots.append(root) or "main",
    )
    config = HarnessConfig(
        source_dirs=(Path("src"),),
        test_dirs=(Path("tests"),),
        pytest_command=("pytest",),
    )
    orchestrator = services.orchestrator_factory(
        tmp_path,
        config,
        web.MemoryStore(tmp_path),
    )

    waiting = orchestrator.run(
        TaskState(
            description="Push current branch",
            mode=TaskMode.BUGFIX,
            bugfix_target="tests/test_value.py::test_value_is_fixed",
            config=config,
        )
    )

    assert waiting.status is TaskStatus.WAITING_APPROVAL
    assert branch_roots and set(branch_roots) == {tmp_path.resolve()}
    assert command_calls == []


def test_serve_binds_uvicorn_to_loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a local-control server accidentally binding a network-reachable interface."""
    web = _web_module()
    services = object()
    application = object()
    received: dict[str, object] = {}

    monkeypatch.setattr(web, "local_services", lambda: services)
    monkeypatch.setattr(web, "create_app", lambda mode, injected: application)
    monkeypatch.setattr(
        web.uvicorn,
        "run",
        lambda app, *, host: received.update({"app": app, "host": host}),
    )
    monkeypatch.setattr(sys, "argv", ["guardedpy", "serve"])

    web.serve()

    assert received == {"app": application, "host": "127.0.0.1"}
